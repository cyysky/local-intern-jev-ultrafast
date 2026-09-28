"""Exercise the /v1/systemone error paths against a running server."""
import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8011"
CHOICE = {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}}
Q = {"team": CHOICE}
BASE_BODY = {"state": "s", "questions": Q}


def call(name, body, path="/v1/systemone"):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            print(f"{name:36s} -> {r.status}")
    except urllib.error.HTTPError as e:
        err = json.loads(e.read())["error"]
        print(f"{name:36s} -> {e.code} {err['type']}: {err['message']}")


call("ok baseline", BASE_BODY)
call("body key: think", {**BASE_BODY, "think": 4})
call("body key: sequential", {**BASE_BODY, "sequential": True})
call("body key: instructions", {**BASE_BODY, "instructions": "ctx"})
call("q key: depends_on", {"state": "s", "questions": {"team": {**CHOICE, "depends_on": ["x"]}}})
call("q key: ask_if", {"state": "s", "questions": {"team": {**CHOICE, "ask_if": {"x": ["a"]}}}})
call("q key: alone", {"state": "s", "questions": {"team": {**CHOICE, "alone": True}}})
call("bad type", {"state": "s", "questions": {"team": {"type": "bogus", "criteria": {"a": "A"}}}})
call("choice criteria not a map", {"state": "s", "questions": {"team": {"type": "choice", "criteria": ["a", "b"]}}})
call("score criteria not a list", {"state": "s", "questions": {"team": {"type": "score", "criteria": {"a": "A"}}}})
call("no questions", {"state": "s"})
call("empty questions", {"state": "s", "questions": {}})
call("no state", {"questions": Q})
call("too many options", {"state": "s", "questions": {"t": {"type": "choice", "criteria": {str(i): "d" for i in range(70)}}}})
call("too many questions", {"state": "s", "questions": {str(i): {"type": "noul"} for i in range(20)}})
call("too many images", {"state": "s", "images": ["data:image/png;base64,AAAA"] * 9, "questions": Q})
call("bad image entry", {"state": "s", "images": ["http://x/y.png"], "questions": Q})
call("unknown route", BASE_BODY, "/v1/nope")
