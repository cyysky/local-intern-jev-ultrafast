"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import json
import time
from copy import deepcopy
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import model
from jev_ultrafast.browser import StalePage, browser_operation, fingerprint


def page():
    state = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


def decision(action="e1"):
    return {
        "choice": action,
        "operation": "TYPE_TEXT",
        "target": "1",
        "confidence": 1.0,
        "probabilities": {action: 1.0},
        "latency_ms": 10,
        "usage": {},
    }


@pytest.mark.parametrize("mutation", ["unknown", "nan", "missing", "negative", "non_max", "confidence"])
def test_invalid_choice_is_rejected(mutation):
    a = choice(["a", "b"], "a")
    if mutation == "unknown":
        a["choice"] = "invented"
    elif mutation == "nan":
        a["probabilities"]["a"] = float("nan")
    elif mutation == "missing":
        del a["probabilities"]["b"]
    elif mutation == "negative":
        a["probabilities"]["b"] = -1
    elif mutation == "non_max":
        a["choice"] = "b"
    else:
        a["confidence"] = 5
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.validate_choice(a, {"a", "b"})


def test_one_index_per_node_with_operation_specific_targets():
    elements, targets, controls = model.action_space(page()["actions"])
    assert len(elements) == 2
    assert elements[0]["operations"] == ["TYPE_TEXT", "CLICK"]
    assert targets["TYPE_TEXT"]["1"]["id"] == "e1"
    assert targets["CLICK"]["1"]["id"] == "e2"
    assert targets["CLICK"]["2"]["id"] == "e3"
    assert "WAIT" in controls


def test_all_heads_are_one_request_and_only_matching_head_executes(monkeypatch):
    calls = []

    def post(_url, _key, body):
        calls.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(["1"], "1"),
                "click_target": {"choice": "invented"},
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 1
    assert d["operation"] == "TYPE_TEXT" and d["target"] == "1" and d["choice"] == "e1"
    assert set(calls[0]["questions"]) == {"operation", "click_target", "type_text_target"}


def test_click_cannot_consume_a_text_target(monkeypatch):
    def post(_url, _key, body):
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "type_text_target": choice(["1"], "1"),
                "click_target": choice(["1", "2", "999"], "999"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(page(), "Find a book", [])


def test_target_head_receives_control_state_and_full_next_step_rules(monkeypatch):
    p = page()
    p["actions"].insert(0, {
        "id": "toggle", "kind": "click", "label": "Free cancellation", "node": 30,
        "role": "checkbox", "checked": "true", "selected": False,
    })

    def post(_url, _key, body):
        questions = body["questions"]
        target = questions["click_target"]
        assert target["criteria"]["1"]["checked"] == "true"
        assert target["criteria"]["1"]["selected"] is False
        assert questions["operation"]["instructions"]["rules"] in target["instructions"]["rules"]
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(target["criteria"], "3"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(p, "Search with free cancellation", [])
    assert d["choice"] == "e3"


def test_local_systemone_needs_no_credential_and_reports_provider_errors(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("SYSTEMONE_BASE_URL", "http://127.0.0.1:8011/v1/")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "300")
    sent = {}

    def post(url, json, headers, timeout):
        sent.update(url=url, headers=headers, timeout=timeout)
        return Mock(is_error=True, status_code=422, json=lambda: {"error": {"message": "at most 62 options"}})

    monkeypatch.setattr(model, "CLIENT", Mock(post=post))
    assert model.systemone_url() == "http://127.0.0.1:8011/v1/systemone"
    with pytest.raises(RuntimeError, match="at most 62 options"):
        model.post_json(model.systemone_url(), "", {})
    assert sent["headers"] == {} and sent["timeout"] == 300.0


def test_choose_posts_to_the_configured_systemone_endpoint(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("SYSTEMONE_BASE_URL", "http://127.0.0.1:8011/v1")
    calls = []

    def post(url, key, body):
        calls.append((url, key))
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "click_target": choice(body["questions"]["click_target"]["criteria"], "2"),
            },
        }

    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert calls == [("http://127.0.0.1:8011/v1/systemone", "")]
    assert d["choice"] == "e3"


def stopping_answers(operations, confidence, others):
    """A DONE head alongside weaker, distinct heads that sum to one."""
    assert set(operations) == {"DONE", *others}
    probabilities = {"DONE": confidence, **others}
    assert abs(sum(probabilities.values()) - 1) < 1e-9
    return {
        "model": "test",
        "answers": {
            "operation": {"choice": "DONE", "confidence": confidence, "probabilities": probabilities},
            "click_target": choice(["1", "2"], "2"),
        },
    }


WEAKER = {"CLICK": 0.3, "TYPE_TEXT": 0.16, "WAIT": 0.08, "BLOCKED": 0.04}


def test_a_shaky_operation_is_handed_to_the_text_helper(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setenv("THINKING_CONFIDENCE_THRESHOLD", "0.7")
    helper_calls, sent = [], {}

    def post(url, key, body):
        if url.endswith("/systemone"):
            sent.update(body)
            operations = body["questions"]["operation"]["criteria"]
            return stopping_answers(operations, 0.42, WEAKER)
        helper_calls.append((url, key, body))
        return {"choices": [{"message": {"content": json.dumps({"choice": "CLICK"})}}], "usage": {}}

    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert d["operation"] == "CLICK" and d["choice"] == "e3"
    assert [record["field"] for record in d["thinking"]] == ["operation"]
    url, key, body = helper_calls[0]
    assert url == "https://api.deepseek.com/v1/chat/completions" and key == "test"
    payload = json.loads(body["messages"][1]["content"])
    assert payload["field"] == "operation" and payload["state"] == sent["state"]
    assert set(payload["allowed_options"]) == set(sent["questions"]["operation"]["criteria"])


def test_a_confident_operation_is_not_handed_to_the_text_helper(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setenv("THINKING_CONFIDENCE_THRESHOLD", "0.7")

    def post(url, _key, _body):
        assert url.endswith("/systemone"), "the helper must not be called"
        return stopping_answers(
            _body["questions"]["operation"]["criteria"],
            0.95,
            {"CLICK": 0.03, "TYPE_TEXT": 0.01, "WAIT": 0.007, "BLOCKED": 0.003},
        )

    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert d["operation"] == "DONE" and d["choice"] == "DONE" and d["thinking"] == []


def test_a_shaky_target_is_handed_to_the_text_helper(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setenv("THINKING_CONFIDENCE_THRESHOLD", "0.7")
    fields = []

    def post(url, _key, body):
        if url.endswith("/systemone"):
            return {
                "model": "test",
                "answers": {
                    "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                    "click_target": {"choice": "1", "confidence": 0.65, "probabilities": {"1": 0.65, "2": 0.35}},
                },
            }
        fields.append(json.loads(body["messages"][1]["content"])["field"])
        return {"choices": [{"message": {"content": json.dumps({"choice": "2"})}}], "usage": {}}

    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert fields == ["click_target"] and d["target"] == "2" and d["choice"] == "e3"
    assert [record["field"] for record in d["thinking"]] == ["click_target"]


def test_a_failing_helper_keeps_the_local_answer(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setenv("THINKING_CONFIDENCE_THRESHOLD", "0.7")

    def post(url, _key, body):
        if url.endswith("/systemone"):
            return stopping_answers(body["questions"]["operation"]["criteria"], 0.42, WEAKER)
        raise RuntimeError("helper is down")

    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert d["operation"] == "DONE" and d["choice"] == "DONE" and d["thinking"] == []


def test_quoted_task_text_still_uses_the_llm(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    context = model.field_context('Fly from "Zurich" to London', page()["actions"][0], page(), [])
    assert model.field_text(context)[0] == "Zurich"
    assert post.call_count == 1
    sent = json.loads(post.call_args.args[2]["messages"][1]["content"])
    assert sent["goal"] == 'Fly from "Zurich" to London'


def test_missing_text_credential_stops_before_guessing(monkeypatch):
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TEXT_MODEL_API_KEY"):
        model.field_text({"goal": 'Enter "Zurich"'})


@pytest.fixture
def runner():
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.pending_text = None
    p = page()
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p)),
        "page": p,
        "decision": decision(),
        "goal": "Find a book",
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
    }
    return a


def test_stale_decision_is_consumed_before_any_mutation(runner):
    runner.state["browser"].fresh.return_value = False
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["decision"] is None


def test_generated_text_reused_only_for_identical_retry_context(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 1
    assert runner.state["browser"].act.call_count == 2  # The first call rejects before any browser input.
    assert runner.pending_text is None


def test_changed_field_context_does_not_reuse_generated_text(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["page"]["text"] = "Different page context"
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 2


def test_loading_waits_do_not_trigger_no_progress_stop(runner):
    for _ in range(5):
        runner.state["decision"] = decision("wait")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert len(runner.state["history"]) == 5 and runner.state["status"] == "ready"


def test_stale_observation_preserves_executed_action(runner):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].observe.side_effect = StalePage("changed")
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["history"][-1]["action"] == "Go"
    runner.state["browser"].act.assert_called_once()


def test_observation_is_one_atomic_browser_read(monkeypatch):
    import jev_ultrafast.browser as browser

    p = page()
    cdp = Mock(return_value={"result": {"value": p}})
    monkeypatch.setattr(browser, "cdp", cdp)
    actual = browser_operation({"operation": "observe", "session": "test", "screenshot": False})
    assert actual["actions"] == p["actions"]
    assert cdp.call_count == 1
    assert cdp.call_args.args[0] == "Runtime.evaluate"


def test_executor_rejects_a_stale_page_before_browser_input(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.fresh = Mock(return_value=False)
    operation = Mock()
    monkeypatch.setattr(browser, "browser_operation", operation)
    with pytest.raises(StalePage):
        b.act(page()["actions"][0], page(), "book")
    operation.assert_not_called()


@pytest.mark.parametrize("response", [{"exceptionDetails": {}}, {"result": {}}])
def test_interrupted_dropdown_mutation_cannot_be_retried_as_stale(monkeypatch, response):
    import jev_ultrafast.browser as browser

    # A navigation can destroy the evaluation result after the change event already fired.
    if "exceptionDetails" in response:
        response["exceptionDetails"] = {"text": "Execution context destroyed"}
    cdp = Mock(return_value=response)
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(RuntimeError, match="Dropdown execution"):
        browser_operation({"operation": "act", "session": "test", "action": {
            "id": "e1", "kind": "select", "node": 1, "value": "Design",
        }})
    assert cdp.call_count == 1


def test_fingerprint_tracks_values_and_identity_not_screenshots():
    p = page()
    other = deepcopy(p)
    other["screenshot"] = "changed"
    assert fingerprint(p) == fingerprint(other)
    other["actions"][0]["node"] = 99
    assert fingerprint(p) != fingerprint(other)


@pytest.mark.parametrize("changed", ["Departure", "Where from?", "Where to?", "year"])
def test_flight_verification_rejects_wrong_trip(changed):
    from examples.flights import verify

    actual = {
        "url": "https://www.google.com/travel/flights/search?tfs=example",
        "text": "Track prices from Zürich to London departing 2026-09-20",
        "actions": [
            {"label": k, "value": v}
            for k, v in [
                ("Change ticket type. One way", "One way"),
                ("Where from?", "Zürich"),
                ("Where to?", "London"),
                ("Departure", "Sun, Sep 20"),
                ("Nonstop flight on Sunday, September 20. Select flight", ""),
            ]
        ],
    }
    assert verify(actual)["passed"]
    if changed == "year":
        actual["text"] = actual["text"].replace("2026", "2027")
    else:
        next(a for a in actual["actions"] if a["label"] == changed)["value"] = "wrong"
    assert not verify(actual)["passed"]


@pytest.mark.parametrize(
    "content", ["Thinking: Zurich", '{"text":null}', '{"text":"Zurich","extra":true}', '{"text":123}']
)
def test_text_helper_rejects_invalid_values(monkeypatch, content):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find a flight"})


def test_navigation_during_prediction_reobserves_without_action(runner):
    runner.state["browser"].fresh.side_effect = StalePage("Document navigating")
    runner.command("tick")
    assert runner.state["status"] == "ready"
    assert runner.state["decision"] is None
    runner.state["browser"].act.assert_not_called()


@pytest.mark.parametrize(
    "message",
    [
        "{'code': -32602, 'message': 'No target with given id found'}",
        "{'code': -32001, 'message': 'Session with given id not found.'}",
    ],
)
def test_a_tab_that_is_gone_is_named_as_such(message):
    import jev_ultrafast.browser as browser

    assert browser.is_gone(RuntimeError(message))
    assert issubclass(browser.Gone, StalePage)


def test_an_ordinary_cdp_error_is_not_gone():
    import jev_ultrafast.browser as browser

    assert not browser.is_gone(RuntimeError("Page.captureScreenshot timed out after 30s"))


def test_closing_a_tab_chrome_already_closed_is_not_an_error(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.target, b.session = "gone-target", "gone-session"
    cdp = Mock(side_effect=RuntimeError("{'code': -32602, 'message': 'No target with given id found'}"))
    monkeypatch.setattr(browser, "cdp", cdp)
    b.close()
    assert b.target is None and b.session is None
    cdp.assert_called_once_with("Target.closeTarget", targetId="gone-target")


def test_closing_a_live_tab_still_reports_a_real_failure(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.target, b.session = "live-target", "live-session"
    cdp = Mock(side_effect=RuntimeError("the daemon is not running"))
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(RuntimeError, match="daemon is not running"):
        b.close()
    assert b.target is None


def test_observation_after_the_tab_is_gone_reopens_the_page(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.session = "gone-session"
    b.url = "https://example.test/"
    reopened, observed = [], []

    def reopen():
        reopened.append(True)
        b.session = "fresh-session"

    def operation(request):
        if request["session"] == "gone-session":
            raise browser.Gone("Session with given id not found.")
        observed.append(request["session"])
        return {"url": "https://example.test/step", "actions": [], "screenshot": "x"}

    monkeypatch.setattr(b, "reopen", reopen)
    monkeypatch.setattr(browser, "browser_operation", operation)
    monkeypatch.setattr(browser, "warm_frame", Mock())
    info = b.observe(screenshot=True)
    assert reopened == [True]
    assert observed == ["fresh-session"]
    assert b.url == "https://example.test/step"
    assert info["url"] == "https://example.test/step"


def test_a_capture_that_arrives_never_asks_for_the_front(monkeypatch):
    import jev_ultrafast.browser as browser

    calls = []
    monkeypatch.setattr(browser, "cdp", lambda method, **params: calls.append(method) or {"data": "frame"})
    assert browser.capture_frame("session") == "frame"
    assert calls == ["Page.captureScreenshot"]


def test_a_capture_asks_for_the_front_only_when_no_frame_arrives(monkeypatch):
    import jev_ultrafast.browser as browser

    calls = []

    def cdp(method, **params):
        if method == "Page.captureScreenshot" and method not in calls:
            calls.append(method)
            raise TimeoutError("Page.captureScreenshot timed out after 30s waiting for the daemon")
        calls.append(method)
        return {"data": "frame"}

    monkeypatch.setattr(browser, "cdp", cdp)
    assert browser.capture_frame("session") == "frame"
    assert calls == ["Page.captureScreenshot", "Page.bringToFront", "Page.captureScreenshot"]
