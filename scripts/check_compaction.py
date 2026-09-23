"""Deterministic compaction acceptance using real checkpoints and SDK requests.

Summary responses are generated from the material actually received by the fake
SDK. Storage/log failures are injected below the production guards. Filesystem
cases read real files. No network requests are made by this script.
"""

import asyncio
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from check_react import (
    LocalClient, SETTINGS as BASE_SETTINGS, TOOL, call, message,
    outputs, read_events, recording_failure, response,
)
from check_sessions import checkpoint_fault, saved_state
from check_session_review_fixes import resources, closed_resources
from check_text import snapshot
from check_tracing import capture
from lineage_agent.agent import compaction, graph, runtime
from lineage_agent.agent.sessions import SessionError, SessionManager
from lineage_agent.config.settings import ConfigurationError
from lineage_agent.entrypoints import cli
from lineage_agent.models.summarization import SUMMARY_INSTRUCTIONS
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL


CHECK_DIR = Path(__file__).resolve().parents[1] / "logs" / "checks" / str(uuid4())
SETTINGS = replace(
    BASE_SETTINGS, log_dir=CHECK_DIR, context_budget_tokens=24000,
    max_output_tokens=256, context_safety_margin_tokens=64,
    summary_max_output_tokens=512, summary_max_bytes=1200,
)
RUNS = []
PRIVATE = "COMPACTION_EXCEPTION_PRIVATE_01e3"
KEYS = ("requirements", "facts", "proposals", "corrections", "open_items")


def is_summary(request):
    return request.get("instructions") == SUMMARY_INSTRUCTIONS


def material_for(request):
    assert is_summary(request)
    assert "tools" not in request and request["tool_choice"] == "none"
    assert request["stream"] is False and request["store"] is False
    assert request["truncation"] == "disabled"
    assert len(request["input"]) == 1 and request["input"][0]["role"] == "user"
    return json.loads(request["input"][0]["content"])


def source_ids(material):
    result = []

    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get("source_id"), str):
                result.append(value["source_id"])
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(material)
    return list(dict.fromkeys(result))


def summary_document(request, text="Earlier confirmed facts remain available."):
    ids = source_ids(material_for(request))
    assert ids, "summary must carry explicit stable source identifiers"
    result = {key: [] for key in KEYS}
    result["facts"] = [{"text": text, "source_ids": ids[:1]}]
    return result


def summary_response(request, *, text="Earlier confirmed facts remain available.", usage=(20, 5, 25)):
    return response(message(json.dumps(summary_document(request, text), ensure_ascii=False)), usage=usage)


@contextmanager
def captured_measurements():
    observations = []

    def wrap(original, purpose):
        def measure(request, risks, settings, **kwargs):
            before, annotations = deepcopy(request), deepcopy(risks)
            result = original(request, risks, settings, **kwargs)
            assert before == request and annotations == risks
            observations.append({"purpose": purpose, "request": before, "budget": deepcopy(result)})
            return result
        return measure

    with patch.object(graph, "check_context_budget", side_effect=wrap(graph.check_context_budget, "business")), \
            patch.object(compaction, "check_context_budget", side_effect=wrap(compaction.check_context_budget, "compaction")):
        yield observations


class Client(LocalClient):
    """Capture actual kwargs, then choose summary/normal responses independently."""

    def __init__(self, replies=(), *, summary=None):
        super().__init__(replies)
        self.summary = summary or summary_response

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if is_summary(kwargs):
            result = self.summary(kwargs)
        else:
            try:
                result = next(self.replies)
            except StopIteration:
                raise AssertionError("unexpected extra business model request") from None
        if isinstance(result, Exception):
            raise result
        return result


def audit(turn, client, scenario, observations=None):
    state = turn.state
    events = read_events(state)
    business = [item for item in client.requests if not is_summary(item)]
    summaries = [item for item in client.requests if is_summary(item)]
    assert state["model_calls"] == len(business)
    assert state["summary_calls"] == len(summaries)
    assert len(state["input"]) == len(state["context_risks"])
    assert 0 <= state["history_end"] <= len(state["input"])
    assert all(item["run_id"] == state["run_id"] and item["session_id"] == turn.session_id for item in events)
    starts = [item for item in events if item["event"] == "model.started"]
    summary_starts = [item for item in events if item["event"] == "summary.started"]
    if state["log_complete"]:
        assert len(starts) == len(business) and len(summary_starts) == len(summaries)
        assert events[-1]["event"] == "run.ended"
    for request, event in zip(business, starts):
        results = [item for item in request["input"] if item.get("type") == "function_call_output"]
        assert event["input_tool_result_call_ids"] == [item["call_id"] for item in results]
        assert len(results) == len({item["call_id"] for item in results})
        assert request["truncation"] == "disabled"
    for request in summaries:
        material_for(request)
    assert PRIVATE not in json.dumps(events, ensure_ascii=False)
    assert PRIVATE not in (state.get("error") or "")
    compaction = state.get("compaction") or {}
    active = state["active_context"]
    print(json.dumps({
        "scenario": scenario, "session_id": turn.session_id, "run_id": state["run_id"],
        "checkpoint_id": turn.checkpoint_id, "checkpoint_complete": turn.checkpoint_complete,
        "checkpoint_error": turn.checkpoint_error, "summary_id": compaction.get("summary_id"),
        "version_id": active["version_id"], "compaction_status": compaction.get("status"),
        "compaction_reason": compaction.get("reason"),
        "steps": list(dict.fromkeys(item["step_id"] for item in events if "step_id" in item)),
        "call_ids": [item["call_id"] for item in outputs(state)],
        "model_calls": state["model_calls"], "summary_calls": state["summary_calls"],
        "usage_totals": state["usage_totals"], "summary_usage_totals": state["summary_usage_totals"],
        "log_path": state["log_path"], "fault_injection_observations": observations or [],
    }, ensure_ascii=False))
    RUNS.append(state["run_id"])
    return events


async def run(manager, sid, prompt, replies=(), *, summary=None, scenario, observations=None):
    client = Client(replies, summary=summary)
    with resources() as observed, captured_measurements() as measured, patch.object(runtime, "create_client", return_value=client) as factory:
        turn = await manager.run_turn(sid, prompt)
    assert not factory.called or client.closed
    closed_resources(manager, sid, observed)
    if turn.checkpoint_complete:
        assert await saved_state(manager, turn) == turn.state
    audit(turn, client, scenario, observations)
    cursor = 0
    for request in client.requests:
        matching = [(index, item) for index, item in enumerate(measured[cursor:], cursor)
                    if item["request"] == request
                    and item["purpose"] == ("compaction" if is_summary(request) else "business")]
        assert matching and matching[0][1]["budget"]["decision"] == "allow", (
            "actual SDK kwargs were not the same request that passed budget measurement"
        )
        cursor, counted = matching[0]
        cursor += 1
        assert request["max_output_tokens"] == counted["budget"]["output_reserve_tokens"]
    return turn, client


async def seed(*, settings=SETTINGS, manager=None, sid=None, count=4, tools=None, executor=None,
               prefix="seed", prompt_size=2000, answer_size=2500):
    manager = manager or SessionManager()
    sid = sid or manager.create_session(settings, tools=tools, execute_tool=executor)
    turns = []
    for index in range(count):
        prompt = f"OLD_SOURCE_{index}: " + "a" * prompt_size
        turn, client = await run(manager, sid, prompt, [response(message("b" * answer_size))], scenario=f"{prefix}_{index + 1}")
        assert turn.state["stop_reason"] == "completed" and turn.state["summary_calls"] == 0
        assert turn.checkpoint_complete
        if turns:
            assert client.requests[0]["input"] == turns[-1].state["input"] + [{"role": "user", "content": prompt}]
        turns.append(turn)
    return manager, sid, turns


TRIGGER = "CURRENT_QUESTION_MUST_REMAIN: " + "c" * 5000


async def check_configuration():
    defaults = {
        "context_keep_recent_turns": 2, "context_target_percent": 70,
        "summary_max_output_tokens": 2048, "summary_max_bytes": 8192,
    }
    for field, expected in defaults.items():
        assert getattr(BASE_SETTINGS, field) == expected
        for value in (True, False, 0, -1, "1", 1.5):
            try:
                replace(BASE_SETTINGS, **{field: value})
            except ConfigurationError:
                pass
            else:
                raise AssertionError((field, value))
    for target in (90, 100, 101):
        try:
            replace(BASE_SETTINGS, context_target_percent=target)
        except ConfigurationError:
            pass
        else:
            raise AssertionError("target must stay below trigger")
    print(json.dumps({"scenario": "configuration", "defaults": defaults}))


async def check_stable_history_and_records():
    manager, sid, turns = await seed(prefix="stable")
    version = turns[0].state["active_context"]["version_id"]
    assert all(item.state["active_context"]["version_id"] == version for item in turns)
    record = await manager.get_record(sid, turns[0].state["run_id"])
    before = deepcopy(record)
    assert record["input"] == turns[0].state["input"]
    assert record["admitted_input"] == record["input"]
    assert len(record["model_results"]) == 1 and not record["summary_results"]
    record["input"][0]["content"] = "MUTATE_RETURNED_RECORD"
    turns[0].state["input"][0]["content"] = "MUTATE_RETURNED_STATE"
    assert await manager.get_record(sid, turns[0].state["run_id"]) == before


async def check_activation_and_stable_reuse():
    manager, sid, old = await seed(prefix="activation")
    originals = {item.state["run_id"]: await manager.get_record(sid, item.state["run_id"]) for item in old}
    prior_version = old[-1].state["active_context"]["version_id"]
    turn, client = await run(manager, sid, TRIGGER, [response(message("current answer"))], scenario="activation")
    state = turn.state
    assert state["stop_reason"] == "completed", state.get("error")
    assert state["summary_calls"] == state["model_calls"] == 1
    assert state["summary_usage_totals"]["total_tokens"] == 25
    assert state["usage_totals"]["total_tokens"] == 37
    assert state["result"]["usage"]["total_tokens"] == 12
    assert state["summary_result"]["usage"]["total_tokens"] == 25
    assert state["active_context"]["version_id"] != prior_version
    assert state["compaction"]["status"] == "used"
    assert state["compaction"]["confirmed_checkpoint_id"]
    summary_request, business = client.requests
    assert is_summary(summary_request) and not is_summary(business)
    material = material_for(summary_request)
    assert TRIGGER not in json.dumps(material)
    assert business["input"][-1] == {"role": "user", "content": TRIGGER}
    recent = old[-1].state["input"][-4:]
    assert business["input"][-5:-1] == recent
    assert business["input"][0]["role"] == "assistant"
    assert "derived_context_summary" in business["input"][0]["content"]
    assert "OLD_SOURCE_0" not in json.dumps(business["input"])
    events = read_events(state)
    activated = next(item for item in events if item["event"] == "compaction.activated")
    started = next(item for item in events if item["event"] == "model.started")
    assert events.index(activated) < events.index(started)
    checkpoint = state["compaction"]["confirmed_checkpoint_id"]
    assert any(item["event"] == "checkpoint.saved" and item["checkpoint_id"] == checkpoint for item in events[:events.index(activated)])
    for run_id, original in originals.items():
        assert await manager.get_record(sid, run_id) == original
    current_record = await manager.get_record(sid, state["run_id"])
    assert current_record["input"][0] == {"role": "user", "content": TRIGGER}
    assert len(current_record["summary_results"]) == len(current_record["model_results"]) == 1
    follow, actual = await run(manager, sid, "Short follow-up", [response(message("follow-up answer"))], scenario="stable_reuse")
    assert follow.state["summary_calls"] == 0
    assert follow.state["active_context"]["version_id"] == state["active_context"]["version_id"]
    assert actual.requests[0]["input"] == state["input"] + [{"role": "user", "content": "Short follow-up"}]


async def check_refusal_boundaries():
    manager = SessionManager()
    sid = manager.create_session(SETTINGS)
    oversized = "NEW_UNSENT_" + "x" * 26000
    turn, actual = await run(manager, sid, oversized, scenario="no_old_history")
    assert turn.state["stop_reason"] == "context_budget_exceeded"
    assert turn.state["history_end"] == 0 and not actual.requests
    short, actual = await run(manager, sid, "short", [response(message())], scenario="short_after_refusal")
    assert actual.requests[0]["input"] == [{"role": "user", "content": "short"}]
    assert short.state["summary_calls"] == 0
    manager, sid, old = await seed(prefix="protected")
    turn, actual = await run(manager, sid, "CURRENT_TOO_BIG_" + "x" * 20000, scenario="protected_current_too_large")
    assert turn.state["stop_reason"] == "context_budget_exceeded" and not actual.requests
    assert turn.state["active_context"]["version_id"] == old[-1].state["active_context"]["version_id"]
    assert turn.state["history_end"] == len(old[-1].state["input"])


async def check_summary_failure_cases():
    def document_case(transform):
        def build(request):
            document = summary_document(request)
            return response(message(transform(document)))
        return build

    cases = (
        ("service", lambda _: RuntimeError(PRIVATE), "error"),
        ("invalid_json", lambda _: response(message("not JSON")), "error"),
        ("duplicate_keys", lambda _: response(message('{"facts":[],"facts":[]}')), "error"),
        ("unknown_source", document_case(lambda doc: json.dumps({**doc, "facts": [{"text": "bad", "source_ids": ["forged:0"]}]})), "error"),
        ("empty", document_case(lambda _: json.dumps({key: [] for key in KEYS})), "error"),
        ("tool_request", lambda _: response(call("summary_must_not_execute")), "error"),
        ("refusal", lambda _: response({**message(), "content": [{"type": "refusal", "refusal": "no"}]}), "error"),
        ("incomplete", lambda req: response(message(json.dumps(summary_document(req))), status="incomplete"), "error"),
        ("oversized", lambda req: summary_response(req, text="d" * 5000), "context_budget_exceeded"),
    )
    for name, make_summary, stop in cases:
        manager, sid, old = await seed(prefix=f"failure_{name}")
        turn, actual = await run(manager, sid, TRIGGER, summary=make_summary, scenario=f"summary_{name}")
        assert turn.state["stop_reason"] == stop, (name, turn.state["error"])
        assert turn.state["summary_calls"] == 1 and turn.state["model_calls"] == 0
        assert len(actual.requests) == 1 and is_summary(actual.requests[0])
        assert turn.state["active_context"]["version_id"] == old[-1].state["active_context"]["version_id"]
        assert turn.state["history_end"] == len(old[-1].state["input"])
        assert turn.state["tool_calls"] == 0
        if name == "service":
            assert turn.state["summary_usage_totals"]["missing_usage_calls"] == 1
            assert turn.state["usage_totals"]["missing_usage_calls"] == 1
        else:
            assert turn.state["summary_result"] is not None
            assert turn.state["usage_totals"]["total_tokens"] == (25 if name == "oversized" else 12)


async def check_usage_missing():
    manager, sid, _ = await seed(prefix="usage")
    turn, _ = await run(
        manager, sid, TRIGGER, [response(message(), usage=(7, None, None))],
        summary=lambda req: summary_response(req, usage=None), scenario="summary_usage_missing",
    )
    assert turn.state["stop_reason"] == "completed"
    assert turn.state["summary_usage_totals"] == {
        "input_tokens": None, "output_tokens": None, "total_tokens": None, "missing_usage_calls": 1,
    }
    assert turn.state["usage_totals"] == {
        "input_tokens": 7, "output_tokens": None, "total_tokens": None, "missing_usage_calls": 2,
    }
    next_turn, _ = await run(manager, sid, "next", [response(message())], scenario="usage_reset")
    assert next_turn.state["summary_calls"] == 0
    assert next_turn.state["usage_totals"]["total_tokens"] == 12
    assert next_turn.state["summary_usage_totals"]["missing_usage_calls"] == 0


async def check_summary_and_candidate_budget_boundaries():
    oversized_reserve = replace(SETTINGS, summary_max_output_tokens=19000)
    manager, sid, old = await seed(settings=oversized_reserve, prefix="summary_request_budget")
    turn, actual = await run(manager, sid, TRIGGER, scenario="summary_input_budget_exceeded")
    assert turn.state["stop_reason"] == "context_budget_exceeded" and not actual.requests
    assert turn.state["compaction"]["reason"] == "summary_budget_exceeded"
    assert turn.state["compaction"]["summary_budget"]["decision"] == "exceeded"
    assert turn.state["summary_calls"] == turn.state["model_calls"] == 0
    assert turn.state["usage_totals"]["missing_usage_calls"] == 0
    assert turn.state["active_context"]["version_id"] == old[-1].state["active_context"]["version_id"]

    manager, sid, old = await seed(prefix="summary_measurement")
    with patch.object(compaction, "prepare_summary_request", side_effect=ValueError(PRIVATE)):
        turn, actual = await run(manager, sid, TRIGGER, scenario="summary_preparation_error")
    assert turn.state["stop_reason"] == "error" and not actual.requests
    assert turn.state["error_code"] == "context_measurement_error"
    assert turn.state["summary_calls"] == 0 and turn.state["usage_totals"]["missing_usage_calls"] == 0
    assert turn.state["active_context"]["version_id"] == old[-1].state["active_context"]["version_id"]

    # Explicit measurement fault injection tests the final candidate guard. The
    # correct upper-bound selection makes this rejection unreachable for static
    # valid inputs; it must nevertheless never dispatch an unapproved candidate.
    manager, sid, old = await seed(prefix="candidate_growth")
    original = compaction.check_context_budget

    def candidate_growth(request, risks, settings, **kwargs):
        result = original(request, risks, settings, **kwargs)
        if not is_summary(request):
            result["required_tokens"] = settings.context_budget_tokens
            result["decision"] = "exceeded"
        return result

    with patch.object(compaction, "check_context_budget", side_effect=candidate_growth):
        turn, actual = await run(manager, sid, TRIGGER, scenario="candidate_budget_fault_injection")
    assert turn.state["stop_reason"] == "context_budget_exceeded"
    assert turn.state["compaction"]["reason"] == "summary_not_effective"
    assert turn.state["summary_calls"] == 1 and turn.state["model_calls"] == 0
    assert len(actual.requests) == 1 and is_summary(actual.requests[0])
    assert turn.state["active_context"]["version_id"] == old[-1].state["active_context"]["version_id"]
    assert turn.state["usage_totals"]["total_tokens"] == 25


async def check_opaque_barriers_and_omissions():
    for kind, first_input in (
        ("unknown", [{"role": "user", "content": "a" * 2000}, {"type": "vendor_private", "payload": "OPAQUE_PRIVATE"}]),
        ("system", [{"role": "system", "content": "Never summarize this authority."}, {"role": "user", "content": "a" * 2000}]),
        ("nontext", [{"role": "user", "content": [{"type": "input_text", "text": "a" * 2000}, {"type": "input_image", "image_url": "https://invalid.test/image"}]}]),
    ):
        manager = SessionManager()
        sid = manager.create_session(replace(SETTINGS, opaque_fallback_tokens=100))
        first, _ = await run(manager, sid, first_input, [response(message("b" * 2500))], scenario=f"opaque_{kind}_seed")
        manager, sid, _ = await seed(manager=manager, sid=sid, count=3, prefix=f"opaque_{kind}")
        turn, actual = await run(manager, sid, TRIGGER, scenario=f"opaque_{kind}_barrier")
        assert turn.state["stop_reason"] == "context_budget_exceeded" and not actual.requests
        assert turn.state["summary_calls"] == 0
        assert turn.state["compaction"]["reason"] == "protected_history"
        assert first.state["input"] == turn.state["input"][:len(first.state["input"])]

    manager = SessionManager()
    sid = manager.create_session(SETTINGS)
    encrypted = "ENCRYPTED_OPAQUE_MUST_NOT_ENTER_SUMMARY"
    first, _ = await run(manager, sid, "a" * 2000, [response(
        {"type": "reasoning", "id": "rs_omitted", "summary": [], "encrypted_content": encrypted},
        message("b" * 2500), usage=(10, 17, 27),
    )], scenario="encrypted_seed")
    manager, sid, _ = await seed(manager=manager, sid=sid, count=3, prefix="encrypted_recent")
    turn, actual = await run(manager, sid, TRIGGER, [response(message())], scenario="encrypted_omitted")
    assert turn.state["stop_reason"] == "completed" and turn.state["summary_calls"] == 1
    material = material_for(actual.requests[0])
    assert material["opaque_omissions"]
    assert encrypted not in json.dumps(material)
    assert encrypted not in json.dumps(actual.requests[1])
    record = await manager.get_record(sid, first.state["run_id"])
    assert encrypted in json.dumps(record)


async def check_real_file_and_failure_source():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        body = "FACT_FROM_FILE_ONLY_PURPLE_17\n" + "background " * 630
        filename = "中文 note.md"
        (root / filename).write_bytes(body.encode("utf-8"))
        before, cwd = snapshot(root), Path.cwd()
        manager = SessionManager()
        sid = manager.create_session(
            SETTINGS, tools=[TEXT_TOOL],
            execute_tool=create_filesystem_executor(root),
        )
        first, first_client = await run(manager, sid, "Read the source and attempt missing file.", [
            response(call("real_read", name="read_text_file", arguments=json.dumps({"path": filename}, ensure_ascii=False)),
                     call("missing_read", name="read_text_file", arguments='{"path":"missing.txt"}')),
            response(message("Source inspected; missing file failed.")),
        ], scenario="real_file_source")
        assert first.state["tool_outcomes"]["succeeded"] == first.state["tool_outcomes"]["failed"] == 1
        actual_outputs = outputs(first.state)
        assert actual_outputs == first_client.requests[-1]["input"][-2:]
        assert json.loads(json.loads(actual_outputs[0]["output"])["result"])["content"] == body
        manager, sid, _ = await seed(manager=manager, sid=sid, count=2, prefix="file_recent", prompt_size=1000, answer_size=1000)
        prompt = "CURRENT_UNCHANGED_" + "x" * 10000
        turn, captured = await run(manager, sid, prompt, [response(message("purple fact from note"))], scenario="real_file_compacted")
        assert turn.state["stop_reason"] == "completed", turn.state["compaction"]
        material = material_for(captured.requests[0])
        text = json.dumps(material, ensure_ascii=False)
        assert body in [json.loads(json.loads(item["payload"]["output"])["result"])["content"]
                        for item in material["sources"] if item["payload"].get("type") == "function_call_output"
                        and json.loads(item["payload"]["output"])["ok"]]
        assert "execution_error" in text and '"failed": 1' in text
        assert "missing_read" in text and "real_read" in text
        assert turn.state["tool_calls"] == 0, "compaction must not rerun old tools"
        assert not outputs(turn.state)
        assert await manager.get_record(sid, first.state["run_id"])
        assert snapshot(root) == before and Path.cwd() == cwd


async def check_repeated_compaction():
    settings = replace(SETTINGS, context_budget_tokens=30000, context_target_percent=80, context_keep_recent_turns=1)
    # Use a real executor invocation producing enough new material for a second
    # recovery in the same user request; each business request gets at most one.
    async def large_result(name, arguments):
        return "REAL_TOOL_RESULT_" + "z" * 6500

    # Sessions bind the executor at creation; construct this session correctly
    # through its public API instead of replacing a live permission binding.
    other = SessionManager()
    other_id = other.create_session(settings, tools=[TOOL], execute_tool=large_result)
    manager, sid, seeds = await seed(
        settings=settings, manager=other, sid=other_id, count=8,
        prefix="multi_bound", prompt_size=1400, answer_size=1400,
    )
    previous_version = seeds[-1].state["active_context"]["version_id"]
    turn, client = await run(manager, sid, "CURRENT_MULTI_" + "c" * 4200, [
        response(call("fresh_big_result")), response(message("completed after two recoveries")),
    ], scenario="same_run_two_recoveries")
    assert turn.state["stop_reason"] == "completed", turn.state["compaction"]
    assert turn.state["summary_calls"] == 2 and turn.state["model_calls"] == 2
    assert turn.state["tool_calls"] == turn.state["tool_outcomes"]["succeeded"] == 1
    assert [is_summary(item) for item in client.requests] == [True, False, True, False]
    first_material, second_material = map(material_for, (client.requests[0], client.requests[2]))
    assert second_material["previous_summary"] is not None
    first_ids = set(source_ids({"sources": first_material["sources"]}))
    second_ids = set(source_ids({"sources": second_material["sources"]}))
    assert not first_ids & second_ids
    assert turn.state["active_context"]["version_id"] != previous_version
    generated = outputs(turn.state)
    assert len(generated) == 1 and generated[0]["call_id"] == "fresh_big_result"
    assert generated[0] in client.requests[-1]["input"]
    assert turn.state["usage_totals"]["total_tokens"] == 74
    record = await manager.get_record(sid, turn.state["run_id"])
    assert len(record["model_results"]) == len(record["summary_results"]) == 2
    assert record["input"][0]["content"].startswith("CURRENT_MULTI_")


async def check_unattempted_recovery_record():
    settings = replace(SETTINGS, context_budget_tokens=30000, context_target_percent=80, context_keep_recent_turns=5)

    async def large_result(name, arguments):
        return "z" * 6500

    manager, sid, old = await seed(
        settings=settings, count=8, tools=[TOOL], executor=large_result,
        prefix="unattempted_recovery", prompt_size=1400, answer_size=1400,
    )
    turn, client = await run(manager, sid, "CURRENT_MULTI_" + "c" * 4200, [response(call("preserved_big"))], scenario="second_recovery_without_attempt")
    assert turn.state["stop_reason"] == "context_budget_exceeded"
    assert turn.state["summary_calls"] == turn.state["model_calls"] == turn.state["tool_calls"] == 1
    assert [is_summary(item) for item in client.requests] == [True, False]
    assert turn.state["compaction"]["status"] == "stopped"
    assert turn.state["compaction"]["reason"] == "no_eligible_history"
    actual_summary_id = json.loads(client.requests[1]["input"][0]["content"])["summary_id"]
    assert actual_summary_id != turn.state["compaction"]["summary_id"]
    record = await manager.get_record(sid, turn.state["run_id"])
    assert len(record["summary_results"]) == len(record["model_results"]) == 1
    summary = record["summary_results"][0]
    assert summary["compaction"]["summary_id"] == actual_summary_id
    assert summary["result"] == turn.state["summary_result"]
    assert summary["compaction"]["request_no"] == 1
    assert turn.state["history_end"] == len(turn.state["input"])
    output = outputs(turn.state)[0]
    assert output["call_id"] == "preserved_big"
    assert not any(output in item["input"] for item in client.requests), "generated result was not sent"
    assert turn.state["usage_totals"]["total_tokens"] == 37


async def check_runtime_tracing():
    marker = "DERIVED_SUMMARY_BODY_NEVER_UPLOAD_83e0"
    baseline = None
    for name, enabled, transport_mode in (("disabled", False, "ok"), ("enabled", True, "ok"), ("upload_failure", True, "connection")):
        with capture(LANGSMITH_TRACING="false"):
            manager, sid, _ = await seed(prefix=f"trace_{name}")
        with capture(transport_mode, LANGSMITH_TRACING="true" if enabled else "false") as (transport, stderr):
            turn, client = await run(manager, sid, TRIGGER, [response(message("final answer"))],
                                     summary=lambda req: summary_response(req, text=marker), scenario=f"runtime_trace_{name}")
        assert turn.state["stop_reason"] == "completed"
        assert marker in json.dumps(client.requests[-1])
        assert marker in turn.state["summary_result"]["text"]
        comparable = {key: turn.state[key] for key in ("model_calls", "summary_calls", "tool_calls", "usage_totals", "summary_usage_totals", "stop_reason")}
        if baseline is None:
            baseline = comparable
        assert comparable == baseline
        if enabled:
            wire = json.dumps(transport.sent, ensure_ascii=False)
            assert marker not in wire and "OLD_SOURCE_0" not in wire
            summaries = [span for span in transport.sent if span["extra"]["metadata"].get("purpose") == "summary"]
            assert len(summaries) == 1
            assert summaries[0]["extra"]["metadata"]["session_id"] == sid
            business = [span for span in transport.sent if span["run_type"] == "llm" and span not in summaries]
            assert len(business) == 1 and business[0]["inputs"]["summary_content_omitted"] is True
            assert "final answer" in wire
            assert bool(stderr.getvalue()) is (transport_mode != "ok")
        else:
            assert not transport.sent
    # Existing business-reply tracing still permits quoted derived content.
    # Its presence here must not be confused with summary payload publication.
    with capture(LANGSMITH_TRACING="false"):
        manager, sid, _ = await seed(prefix="trace_quoted")
    with capture() as (transport, _):
        turn, _ = await run(manager, sid, TRIGGER, [response(message(marker))],
                            summary=lambda req: summary_response(req, text=marker), scenario="runtime_trace_answer_quote")
    summaries = [span for span in transport.sent if span["extra"]["metadata"].get("purpose") == "summary"]
    assert marker not in json.dumps(summaries)
    assert marker in json.dumps([span["outputs"] for span in transport.sent if span["run_type"] == "llm" and span not in summaries])


async def check_history_admission_and_cli():
    settings = replace(SETTINGS, max_model_calls=1)
    manager = SessionManager()
    sid = manager.create_session(settings)
    rejected, _ = await run(manager, sid, [{"role": "user", "content": 123}], scenario="invalid_new_input_before_compaction")
    assert rejected.state["history_end"] == 0
    failed, _ = await run(manager, sid, "EARLIER_FAILED_GOAL " + "a" * 4000, [
        response(call("unexecuted_private", arguments='{"key":"UNEXECUTED_PRIVATE_ARGUMENT"}')),
    ], scenario="failed_old_turn")
    assert failed.state["stop_reason"] == "max_model_calls"
    assert failed.state["tool_outcomes"]["skipped"] == 1 and failed.state["history_end"] == 1
    manager, sid, _ = await seed(settings=settings, manager=manager, sid=sid, count=3, prefix="failed_old_recent")
    turn, actual = await run(manager, sid, TRIGGER, [response(message("business answer only"))], scenario="failed_old_turn_compacted")
    assert turn.state["stop_reason"] == "completed"
    material = material_for(actual.requests[0])
    assert "UNEXECUTED_PRIVATE_ARGUMENT" not in json.dumps(material)
    assert not any(item["payload"].get("content") == 123 for item in material["sources"])
    outcome = next(item for item in material["turn_outcomes"] if item["run_id"] == failed.state["run_id"])
    assert outcome["stop_reason"] == "max_model_calls" and outcome["tool_outcomes"]["skipped"] == 1
    original = await manager.get_record(sid, failed.state["run_id"])
    assert original["input"][-1]["call_id"] == "unexecuted_private"
    assert original["admitted_input"] == original["input"][:1]
    invalid = await manager.get_record(sid, rejected.state["run_id"])
    assert invalid["input"] == [{"role": "user", "content": 123}] and invalid["admitted_input"] == []
    stdout, stderr = StringIO(), StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        assert cli.display_run(turn.state, settings) == 0
    assert stdout.getvalue().strip() == "business answer only"
    assert turn.state["summary_result"]["text"] not in stdout.getvalue() + stderr.getvalue()
    assert turn.state["compaction"]["summary_id"] in stderr.getvalue()
    assert turn.state["compaction"]["confirmed_checkpoint_id"] in stderr.getvalue()
    assert "业务 1，摘要 1" in stderr.getvalue()

    # Drive the real interactive loop through two requests on the already seeded
    # manager; both answers remain distinct from the internal summary response.
    manager, sid, _ = await seed(prefix="cli_interactive")
    original_run = manager.run_turn
    delivered = []

    async def observe_turn(session_id, prompt):
        result = await original_run(session_id, prompt)
        delivered.append(result)
        return result

    clients = [Client([response(message("compressed business"))]), Client([response(message("followup business"))])]
    stdout, stderr = StringIO(), StringIO()
    with patch.object(cli, "SessionManager", return_value=manager), \
            patch.object(manager, "create_session", return_value=sid), \
            patch.object(manager, "run_turn", side_effect=observe_turn), \
            patch.object(runtime, "create_client", side_effect=clients), \
            patch("builtins.input", side_effect=[TRIGGER, "next", "/exit"]), \
            redirect_stdout(stdout), redirect_stderr(stderr):
        assert await cli.interactive(SETTINGS, {}) == 0
    assert stdout.getvalue().splitlines() == ["compressed business", "followup business"]
    assert len(delivered) == 2
    for index, (result, client) in enumerate(zip(delivered, clients, strict=True)):
        assert client.closed
        audit(result, client, f"interactive_compaction_{index}")
    assert delivered[0].state["summary_calls"] == 1 and delivered[1].state["summary_calls"] == 0
    assert delivered[0].state["active_context"]["version_id"] == delivered[1].state["active_context"]["version_id"]


async def check_session_isolation():
    manager = SessionManager()
    sessions = []
    for index in range(2):
        sid = manager.create_session(SETTINGS)
        _, _, old = await seed(manager=manager, sid=sid, prefix=f"isolation_{index}")
        sessions.append((sid, old))
    clients = [Client([response(message(f"answer_{index}"))], summary=lambda req, i=index: summary_response(req, text=f"ONLY_SESSION_{i}")) for index in range(2)]
    with patch.object(runtime, "create_client", side_effect=clients):
        turns = await asyncio.gather(*(manager.run_turn(sid, TRIGGER) for sid, _ in sessions))
    for index, (turn, client, (sid, old)) in enumerate(zip(turns, clients, sessions, strict=True)):
        audit(turn, client, f"session_isolation_{index}")
        assert client.closed and turn.checkpoint_complete
        assert turn.state["stop_reason"] == "completed" and turn.state["summary_calls"] == 1
        assert f"ONLY_SESSION_{index}" in turn.state["input"][0]["content"]
        assert f"ONLY_SESSION_{1-index}" not in json.dumps(turn.state)
        ids = source_ids(material_for(client.requests[0]))
        own_runs = {item.state["run_id"] for item in old}
        assert all(identifier.split(":")[0] in own_runs for identifier in ids)
        saved = deepcopy(turn.state)
        turn.state["input"][0]["content"] = "MUTATED_SUMMARY"
        turn.state["compaction"]["sources"]["added_turns"].clear()
        assert await saved_state(manager, turn) == saved
        other_sid = sessions[1-index][0]
        try:
            await manager.get_record(other_sid, turn.state["run_id"])
        except SessionError:
            pass
        else:
            raise AssertionError("cross-session record lookup succeeded")
    assert turns[0].state["active_context"]["version_id"] != turns[1].state["active_context"]["version_id"]


async def check_checkpoint_boundaries():
    cases = (
        ("candidate", "put", lambda cp, meta, *_: (cp["channel_values"].get("compaction") or {}).get("status") == "candidate"),
        ("candidate_pending", "put_writes", lambda writes, *_: (dict(writes).get("compaction") or {}).get("status") == "candidate"),
        ("activation", "put", lambda cp, meta, *_: (cp["channel_values"].get("compaction") or {}).get("status") == "pending_save"),
        ("final", "put", lambda cp, meta, *_: meta.get("source") == "update"),
    )
    for name, method, predicate in cases:
        manager, sid, _ = await seed(prefix=f"storage_{name}")
        with checkpoint_fault(manager, method, predicate) as observations:
            turn, actual = await run(manager, sid, TRIGGER, [response(message())], scenario=f"checkpoint_{name}", observations=observations)
        assert observations, f"fault injection missed {name}"
        assert not turn.checkpoint_complete and turn.checkpoint_id is None
        assert turn.state["summary_calls"] == 1
        expected_business = 1 if name == "final" else 0
        assert turn.state["model_calls"] == expected_business
        assert turn.state["summary_result"] is not None
        assert turn.state["usage_totals"]["total_tokens"] == 25 + 12 * expected_business
        assert len(actual.requests) == 1 + expected_business
        try:
            await manager.run_turn(sid, "must stay blocked")
        except SessionError as exc:
            assert exc.code == "session_blocked"
        else:
            raise AssertionError("save-fault session was not blocked")

    # The model node can stop at the budget gate without a new model result.
    # Its retained result may describe a tool batch that has already executed.
    executed = []

    async def large_result(name, arguments):
        executed.append((name, deepcopy(arguments)))
        return "ALREADY_EXECUTED_RESULT_" + "z" * 6500

    manager, sid, _ = await seed(
        tools=[TOOL], executor=large_result, prefix="storage_preparing_after_tools",
    )
    predicate = lambda cp, meta, *_: (
        (cp["channel_values"].get("compaction") or {}).get("status") == "preparing"
        and cp["channel_values"].get("model_calls") == 1
    )
    with checkpoint_fault(manager, "put", predicate) as observations:
        turn, actual = await run(
            manager, sid, "Inspect the current tool result.",
            [response(call("already_executed"))],
            scenario="checkpoint_preparing_after_executed_tools", observations=observations,
        )
    state = turn.state
    assert observations, "fault injection missed the post-tool budget gate"
    assert turn.checkpoint_error == "checkpoint_write_error"
    assert not turn.checkpoint_complete and turn.checkpoint_id is None
    assert state["stop_reason"] == "error" and state["compaction"]["status"] == "preparing"
    assert state["model_calls"] == state["tool_calls"] == len(executed) == 1
    assert state["summary_calls"] == 0 and len(actual.requests) == 1
    assert state["tool_outcomes"] == {"succeeded": 1, "rejected": 0, "failed": 0, "skipped": 0}
    result = outputs(state)
    assert len(result) == 1 and result[0]["call_id"] == "already_executed"
    assert json.loads(result[0]["output"]) == {
        "ok": True, "result": "ALREADY_EXECUTED_RESULT_" + "z" * 6500,
    }
    assert result[0] not in actual.requests[0]["input"], "result was generated but not sent"
    assert state["history_end"] == len(state["input"])
    assert state["usage_totals"]["total_tokens"] == 12
    events = read_events(state)
    assert not any(item["event"] == "tool.skipped" for item in events)
    assert any(item["event"] == "tool.result_created" and item["call_id"] == "already_executed" for item in events)
    try:
        await manager.run_turn(sid, "must stay blocked")
    except SessionError as exc:
        assert exc.code == "session_blocked"
    else:
        raise AssertionError("post-tool save-fault session was not blocked")


async def check_log_boundaries():
    for event, summary_calls in (("summary.started", 0), ("summary.returned", 1), ("compaction.activated", 1)):
        manager, sid, _ = await seed(prefix=f"log_{event}")
        with recording_failure(event) as observations:
            turn, client = await run(manager, sid, TRIGGER, [response(message())], scenario=f"log_failure_{event}", observations=observations)
        assert observations
        assert not turn.state["log_complete"] and turn.state["stop_reason"] == "error"
        assert turn.state["summary_calls"] == summary_calls and turn.state["model_calls"] == 0
        assert len(client.requests) == summary_calls
        if summary_calls:
            assert turn.state["summary_result"] is not None
            assert turn.state["usage_totals"]["total_tokens"] == 25
    manager, sid, _ = await seed(prefix="log_summary_failed")
    with recording_failure("summary.failed") as observations:
        turn, client = await run(manager, sid, TRIGGER, summary=lambda _: RuntimeError(PRIVATE),
                                 scenario="summary_failure_logging_failure", observations=observations)
    assert observations and turn.state["error_code"] == "logging_error"
    assert turn.state["summary_calls"] == 1 and turn.state["model_calls"] == 0
    assert turn.state["summary_result"] is None
    assert turn.state["usage_totals"]["missing_usage_calls"] == 1
    assert len(client.requests) == 1


async def main():
    checks = (
        check_configuration, check_stable_history_and_records,
        check_activation_and_stable_reuse, check_refusal_boundaries,
        check_summary_failure_cases, check_usage_missing,
        check_summary_and_candidate_budget_boundaries,
        check_opaque_barriers_and_omissions, check_real_file_and_failure_source,
        check_repeated_compaction, check_unattempted_recovery_record,
        check_session_isolation, check_runtime_tracing,
        check_history_admission_and_cli,
        check_checkpoint_boundaries, check_log_boundaries,
    )
    for check in checks:
        await check()
        print(f"PASS {check.__name__}")
    print(f"PASS {len(checks)} compaction groups; {len(RUNS)} captured runs; logs: {CHECK_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
