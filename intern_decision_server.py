"""FastAPI host for vLLM's /v1/systemone decision API on the transformers
(Hugging Face) Intern-Decision backend.

The HTTP surface follows vLLM's
examples/features/structured_diffusion/structured_server.py, so the same clients
work unchanged:

    GET  /health              -> {"status": "ok"}
    GET  /v1/models          -> OpenAI-shaped model list
    POST /v1/systemone        -> the Jev decision API
    POST /v1/chat/completions -> the same decision from an OpenAI-shaped call

POST /v1/systemone takes Jev's request body: {"model", "state", "questions"}.
"questions" maps an id to {"type", "instructions", "criteria"}, where "type" is
"noul", "choice" or "score" and the criteria shape follows the type:
  noul:   optional {"true": ..., "false": ...} descriptions
  choice: option name -> description or null
  score:  ordered list of levels
Answers take Jev's shapes:
  noul:   {"noul": p}
  choice: {"choice", "probabilities", "confidence"}
  score:  {"score", "legend", "probabilities", "confidence"}
Images go ahead of the state, either as multipart/form-data with the JSON body in
a part named "request" and each image as a file part, or as an "images" array of
data: URLs or {"content_type", "base64"} objects in the JSON body.

Differences from the vLLM server, because this backend is a causal language model
doing one forward pass and not a diffusion model denoising a canvas:
  * the read is deterministic, so "seed" is accepted and ignored
  * "instructions", "samples", "auto_max", "auto_threshold", "steps", "think",
    "ask", "chunk_rows", "chunk_prompt", "sequential" and the per-question
    "depends_on"/"ask_if"/"alone" keys are not supported here and are rejected
    with 422 rather than silently ignored
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

MODEL_DIR = Path(os.environ.get("MODEL_DIR", r"D:\models\Intern-Decision-4B")).resolve()
MODEL_NAME = os.environ.get("MODEL_NAME", "intern-decision-4b")
DEVICE = os.environ.get("DEVICE", "auto")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ATTN = os.environ.get("ATTN_IMPLEMENTATION", "sdpa")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "8192"))
API_KEY = os.environ.get("API_KEY", "")

# The checkpoint's own limits, enforced by the HF backend below.
MAX_QUESTIONS = 16
MAX_IMAGES = 8
ANSWER_SYMBOLS = 62  # A-Z, a-z, 0-9

# vLLM's canvas server supports these as body extensions. This backend cannot,
# so they are refused instead of being dropped.
UNSUPPORTED_BODY_KEYS = (
    "instructions",
    "samples",
    "auto_max",
    "auto_threshold",
    "steps",
    "think",
    "ask",
    "chunk_rows",
    "chunk_prompt",
    "sequential",
)
UNSUPPORTED_QUESTION_KEYS = ("depends_on", "ask_if", "alone")

if not MODEL_DIR.is_dir():
    raise SystemExit(f"model directory not found: {MODEL_DIR}")
sys.path.insert(0, str(MODEL_DIR))

from inference import DecisionEngine  # noqa: E402


def gpu_report() -> str:
    """Free/total memory of every visible GPU, for the startup banner."""
    cards = []
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        cards.append(f"cuda:{index} {free_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB free")
    return ", ".join(cards)


def resolve_device(requested: str) -> str:
    """A concrete device string for a possibly-"auto" DEVICE.

    "auto" takes the visible CUDA device with the most free memory. A 9 GB
    checkpoint does not fit beside anything else on a 10 GB card, and the requests
    that fail there fail with CUDA out of memory, so the emptiest card wins.
    """
    if requested.lower() != "auto":
        return requested
    if not torch.cuda.is_available():
        print("device auto: no CUDA device, falling back to cpu", flush=True)
        return "cpu"
    best = max(
        range(torch.cuda.device_count()),
        key=lambda index: torch.cuda.mem_get_info(index)[0],
    )
    print(f"device auto: {gpu_report()} -> cuda:{best}", flush=True)
    return f"cuda:{best}"


class SchemaError(ValueError):
    """A request that does not fit the decision schema."""


class BadRequest(ValueError):
    """A request that is not readable as a Jev request body."""


class CudaMemory(RuntimeError):
    """The forward pass did not fit in the GPU's memory."""


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


def split_data_url(value: str) -> tuple[str, str]:
    """-> (content type, base64 payload) of a data:image/... URL."""
    header, _, payload = value.partition(",")
    if not header.startswith("data:image/") or not payload:
        raise SchemaError("a data:image/... URL with a base64 payload")
    if ";base64" not in header:
        raise SchemaError("a data:image/... URL that is base64 encoded")
    return header[len("data:") : -len(";base64")], payload


def jev_questions(body: dict) -> dict[str, dict]:
    """The HF backend's questions from a Jev request body.

    The Jev question shape and the checkpoint's question shape are the same:
    "type", "instructions" and "criteria", with the criteria shape following
    the type. Only the vLLM-only per-question keys are refused.
    """
    for key in UNSUPPORTED_BODY_KEYS:
        if key in body:
            raise SchemaError(
                f"{key}: not supported by this transformers backend "
                "(vLLM canvas extension)"
            )
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise SchemaError("questions: needs a non-empty map of id -> question")
    if len(questions) > MAX_QUESTIONS:
        raise SchemaError(f"questions: at most {MAX_QUESTIONS} per request")
    out: dict[str, dict] = {}
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid:
            raise SchemaError("question ids must be nonempty strings")
        if not isinstance(question, dict):
            raise SchemaError(f"question {qid!r}: must be an object")
        for key in UNSUPPORTED_QUESTION_KEYS:
            if key in question:
                raise SchemaError(
                    f"question {qid!r}: {key} is not supported by this "
                    "transformers backend (vLLM canvas extension)"
                )
        kind = question.get("type")
        if kind not in ("noul", "choice", "score"):
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        instructions = question.get("instructions", "")
        criteria = question.get("criteria")
        if kind == "noul":
            if criteria is not None and not isinstance(criteria, dict):
                raise SchemaError(
                    f"question {qid!r}: noul criteria must be an object with "
                    "true and false"
                )
        elif kind == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise SchemaError(
                    f"question {qid!r}: choice criteria must map option names "
                    "to descriptions"
                )
        else:
            if not isinstance(criteria, list) or not criteria:
                raise SchemaError(
                    f"question {qid!r}: score criteria must be an ordered "
                    "list of levels"
                )
        if kind != "noul" and len(criteria) > ANSWER_SYMBOLS:
            raise SchemaError(
                f"question {qid!r}: at most {ANSWER_SYMBOLS} options, one "
                "single-token answer symbol each"
            )
        item: dict[str, Any] = {"type": kind, "instructions": instructions}
        if criteria is not None:
            item["criteria"] = criteria
        out[qid] = item
    return out


def jev_images(value: Any) -> list[tuple[str, str]]:
    """Image parts from the body's "images": data URLs, or objects with
    content_type and base64."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise SchemaError("images: must be an array")
    if len(value) > MAX_IMAGES:
        raise SchemaError(f"images: at most {MAX_IMAGES} per request")
    out: list[tuple[str, str]] = []
    for i, im in enumerate(value):
        if isinstance(im, str) and im.startswith("data:image/"):
            out.append(split_data_url(im))
        elif (
            isinstance(im, dict)
            and str(im.get("content_type", "")).startswith("image/")
            and isinstance(im.get("base64"), str)
        ):
            out.append((im["content_type"], im["base64"]))
        else:
            raise SchemaError(
                f"images[{i}]: a data:image/... URL or an object with "
                "content_type and base64"
            )
    return out


def chat_schema(value: Any) -> dict[str, dict]:
    """The HF backend's questions from a chat schema message.

    vLLM's chat shape lists the questions as {"questions": [{"id", "type",
    "instructions", "options" | "levels"}]}, so it is folded back into the Jev
    request shape and read the same way.
    """
    if not isinstance(value, dict):
        raise SchemaError("the system message must be a JSON object")
    items = value.get("questions")
    if not isinstance(items, list) or not items:
        raise SchemaError("questions: needs a non-empty list of questions")
    questions: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            raise SchemaError("each question must be an object")
        qid = item.get("id")
        if not isinstance(qid, str) or not qid:
            raise SchemaError("each question needs a nonempty string id")
        if qid in questions:
            raise SchemaError(f"question {qid!r}: duplicate id")
        kind = item.get("type")
        question: dict[str, Any] = {
            "type": kind,
            "instructions": item.get("instructions", ""),
        }
        if kind == "choice":
            options = item.get("options")
            if not isinstance(options, list) or not options:
                raise SchemaError(f"question {qid!r}: options must be a list")
            question["criteria"] = {
                str(o["name"]): o.get("description") for o in options
            }
        elif kind == "score":
            question["criteria"] = item.get("levels")
        elif kind == "noul":
            if item.get("criteria") is not None:
                question["criteria"] = item["criteria"]
        else:
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        questions[qid] = question
    body = {k: value[k] for k in UNSUPPORTED_BODY_KEYS if k in value}
    body["questions"] = questions
    return jev_questions(body)


async def read_body(request: Request) -> tuple[dict, list[tuple[str, str]]]:
    """-> (body, images): a JSON body, or multipart/form-data with the JSON
    in a part named request and each image as a file part, in order."""
    ctype = request.headers.get("content-type", "")
    if not ctype.lower().startswith("multipart/form-data"):
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError as e:
            raise BadRequest(f"body is not valid JSON: {e}") from e
        if not isinstance(body, dict):
            raise BadRequest("body must be a JSON object")
        return body, []
    form = await request.form()
    raw = form.get("request")
    if raw is None:
        raise BadRequest("multipart needs a part named request holding the JSON body")
    if hasattr(raw, "read"):
        raw = await raw.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as e:
        raise BadRequest(f"request part is not valid JSON: {e}") from e
    if not isinstance(body, dict):
        raise BadRequest("request part must be a JSON object")
    images: list[tuple[str, str]] = []
    for key, value in form.multi_items():
        if key == "request":
            continue
        part_type = getattr(value, "content_type", None) or ""
        if not part_type.startswith("image/"):
            raise BadRequest(f"part {key!r}: neither the request JSON nor an image")
        data = await value.read() if hasattr(value, "read") else value
        images.append((part_type, base64.b64encode(data).decode()))
    if len(images) > MAX_IMAGES:
        raise BadRequest(f"images: at most {MAX_IMAGES} per request")
    return body, images


def materialize(images: list[tuple[str, str]]) -> tuple[list[str], Any]:
    """-> (local paths, cleanup). The HF backend reads images from disk."""
    if not images:
        return [], lambda: None
    paths: list[str] = []
    tmpdir = tempfile.mkdtemp(prefix="intern-decision-")
    try:
        for i, (ctype, payload) in enumerate(images):
            suffix = "." + (ctype.split("/")[-1] or "png").replace("jpeg", "jpg")
            path = Path(tmpdir) / f"image-{i}{suffix}"
            try:
                path.write_bytes(base64.b64decode(payload, validate=True))
            except (binascii.Error, ValueError) as e:
                raise SchemaError(f"images[{i}]: not valid base64: {e}") from e
            paths.append(str(path))
    except BaseException:
        _rmtree(tmpdir)
        raise

    def cleanup() -> None:
        _rmtree(tmpdir)

    return paths, cleanup


def _rmtree(path: str) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# The decision read
# ---------------------------------------------------------------------------


def infer(state: Any, questions: dict[str, dict], image_paths: list[str]) -> dict:
    """One deterministic forward pass through the checkpoint's own engine."""
    request: dict[str, Any] = {"state": state, "questions": questions}
    if image_paths:
        request["images"] = list(image_paths)
    engine = app.state.engine
    with app.state.lock:
        try:
            return engine.predict(request)
        except torch.cuda.OutOfMemoryError as exc:
            # The read did not fit. Hand the allocator's cached blocks back to
            # the driver, then give the same read one more try before failing.
            torch.cuda.empty_cache()
            print(f"systemone: CUDA out of memory, retrying: {exc}", flush=True)
            try:
                return engine.predict(request)
            except torch.cuda.OutOfMemoryError as retry:
                raise CudaMemory(str(retry)) from retry
        finally:
            # Keep the card at the checkpoint's own footprint between reads
            # rather than letting unused blocks accumulate.
            torch.cuda.empty_cache()


def jev_answer(question: dict, answer: dict) -> dict:
    """One answer in Jev's shapes, as the vLLM server returns it."""
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": answer["noul"]}
    if kind == "choice":
        return {
            "type": "choice",
            "choice": answer["choice"],
            "probabilities": answer["probabilities"],
            "confidence": answer["confidence"],
        }
    return {
        "type": "score",
        "score": answer["score"],
        "legend": answer["legend"],
        "probabilities": answer["probabilities"],
        "confidence": answer["confidence"],
    }


def build_result(questions: dict[str, dict], result: dict) -> dict:
    """Jev's answer set, with this backend's diagnostics alongside."""
    answers = {
        qid: jev_answer(question, result["answers"][qid])
        for qid, question in questions.items()
    }
    labels = " ".join(
        f"{qid}={answer.get('choice', answer.get('noul', answer.get('score')))}"
        for qid, answer in answers.items()
    )
    timing = result["timing"]["inference_ms"]
    print(f"systemone: {labels} {timing:.0f}ms", flush=True)
    return {
        "model": MODEL_NAME,
        "answers": answers,
        "usage": {
            "input_tokens": result["usage"]["input_tokens"],
            "output_tokens": result["usage"]["output_tokens"],
        },
        "diagnostics": {
            "backend": result["backend"],
            "timing": {"reads": result["usage"]["decision_count"], "total_ms": timing},
            "prompt_tokens": result["usage"]["input_tokens"],
            "calibration": result.get("calibration"),
        },
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def check_auth(request: Request) -> JSONResponse | None:
    if not API_KEY:
        return None
    if request.headers.get("authorization", "") == f"Bearer {API_KEY}":
        return None
    return error(401, "missing or wrong bearer token", "authentication_error")


def error(code: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse(
        status_code=code, content={"error": {"message": message, "type": kind}}
    )


def server_error(exc: Exception) -> JSONResponse:
    """A failed read as a 500, with CUDA memory failures named and actionable.

    A card that cannot hold the read reports either a clean out-of-memory error or,
    when a segment cannot grow, a caching-allocator assertion. Both mean the same
    thing to the caller: the server needs a restart on an emptier card.
    """
    detail = str(exc) if isinstance(exc, CudaMemory) else repr(exc)
    if "out of memory" in detail or "INTERNAL ASSERT FAILED" in detail:
        detail = (
            f"CUDA out of memory: {exc}. Restart start_intern_decision.bat, "
            "which stops the old server and picks the emptiest GPU."
        )
    return error(500, detail, "server_error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    device = resolve_device(DEVICE)
    print(f"loading {MODEL_NAME} from {MODEL_DIR} on {device}", flush=True)
    started = time.perf_counter()
    app.state.engine = DecisionEngine(
        checkpoint=MODEL_DIR,
        device=device,
        dtype=DTYPE,
        attn_implementation=ATTN,
        max_length=MAX_LENGTH,
    )
    # One forward pass per GPU, so requests take the engine in turn.
    app.state.lock = threading.Lock()
    print(f"loaded in {time.perf_counter() - started:.1f}s", flush=True)
    if torch.cuda.is_available():
        print(f"gpu: {gpu_report()}", flush=True)
    yield
    del app.state.engine


app = FastAPI(title="Intern-Decision systemone", lifespan=lifespan)


@app.exception_handler(StarletteHTTPException)
async def http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Unknown routes take vLLM's error shape rather than FastAPI's."""
    if exc.status_code == 404:
        return error(404, "unknown route", "invalid_request_error")
    return error(exc.status_code, str(exc.detail), "invalid_request_error")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/systemone")
async def systemone(request: Request):
    unauthorized = check_auth(request)
    if unauthorized is not None:
        return unauthorized
    try:
        body, image_parts = await read_body(request)
        questions = jev_questions(body)
    except BadRequest as e:
        return error(400, f"invalid body: {e}", "invalid_request_error")
    except SchemaError as e:
        return error(422, str(e), "validation_error")
    if "state" not in body:
        return error(422, "state: required", "validation_error")
    try:
        image_parts += jev_images(body.get("images"))
    except SchemaError as e:
        return error(422, str(e), "validation_error")
    try:
        paths, cleanup = materialize(image_parts)
    except SchemaError as e:
        return error(422, str(e), "validation_error")
    try:
        result = await run_in_threadpool(infer, body["state"], questions, paths)
    except Exception as e:
        return server_error(e)
    finally:
        cleanup()
    return JSONResponse(build_result(questions, result))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    unauthorized = check_auth(request)
    if unauthorized is not None:
        return unauthorized
    try:
        body, image_parts = await read_body(request)
    except BadRequest as e:
        return error(400, f"invalid body: {e}", "invalid_request_error")
    messages = body.get("messages") or []
    if (
        len(messages) != 2
        or messages[0].get("role") not in ("system", "developer")
        or messages[1].get("role") != "user"
    ):
        return error(
            400,
            "a structured request is exactly two messages: the schema "
            "(system) and the state JSON (user)",
            "invalid_request_error",
        )
    try:
        schema = json.loads(message_text(messages[0]))
        questions = chat_schema(schema)
        content = messages[1].get("content")
        if isinstance(content, list):
            state = content
            image_parts += jev_images(
                [
                    part["image_url"]["url"]
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") == "image_url"
                    and isinstance(part.get("image_url"), dict)
                    and isinstance(part["image_url"].get("url"), str)
                ]
            )
        else:
            state = message_text(messages[1]).strip()
            json.loads(state)
    except SchemaError as e:
        return error(400, str(e), "invalid_request_error")
    except (json.JSONDecodeError, AttributeError, KeyError, TypeError) as e:
        return error(
            400,
            "system must be a JSON question schema and user must be JSON "
            f"state or image parts: {e}",
            "invalid_request_error",
        )
    try:
        paths, cleanup = materialize(image_parts)
    except SchemaError as e:
        return error(400, str(e), "invalid_request_error")
    try:
        result = await run_in_threadpool(infer, state, questions, paths)
    except Exception as e:
        return server_error(e)
    finally:
        cleanup()
    completed = build_result(questions, result)
    return JSONResponse(
        {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", MODEL_NAME),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(completed["answers"], indent=2),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": completed["usage"]["input_tokens"],
                "completion_tokens": completed["usage"]["output_tokens"],
                "total_tokens": completed["usage"]["input_tokens"]
                + completed["usage"]["output_tokens"],
            },
        }
    )


def message_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content if isinstance(content, str) else ""
