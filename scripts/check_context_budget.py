"""Offline context-budget acceptance with real graph/checkpoints and SDK capture.

The counted request is captured before dispatch and compared with the actual SDK
arguments. Filesystem scenarios read real temporary files; storage and transport
faults are explicitly injected. No model or tracing network service is used.
"""

import asyncio
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from io import StringIO
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from check_support import check_group, observe_state, observe_turn
from check_react import (
    FaultyFile, LocalClient, SETTINGS as BASE_SETTINGS, TOOL, audit_recording, call,
    fixed_tool, message, outputs, read_events, recording_failure, response,
)
from check_sessions import checkpoint_fault, saved_state
from check_session_review_fixes import resources, closed_resources
from check_text import snapshot
from check_tracing import capture
from lineage_agent.agent import graph, runtime
from lineage_agent.agent.sessions import SessionError, SessionManager
from lineage_agent.config.settings import ConfigurationError, load_settings
from lineage_agent.entrypoints import cli
from lineage_agent.models.budget import annotate_context_risks, check_context_budget
from lineage_agent.models.requests import prepare_request
from lineage_agent.recording.recorder import RunRecorder
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL


CHECK_DIR = Path(__file__).resolve().parents[1] / "logs" / "checks" / str(uuid4())
SETTINGS = replace(BASE_SETTINGS, log_dir=CHECK_DIR)
RUNS = []
PRIVATE = "BUDGET_PRIVATE_EXCEPTION_NEVER_RECORD_95dc"


def compact_bytes(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def same_saved_facts(left, right):
    """NaN can be checkpointed although it is not valid JSON or equal to itself."""
    if type(left) is float and type(right) is float and math.isnan(left) and math.isnan(right):
        return True
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_saved_facts(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same_saved_facts(a, b) for a, b in zip(left, right, strict=True))
    return left == right


@contextmanager
def capture_measurements():
    observations = []
    original = graph.check_context_budget

    def measure(request, risks, settings, **kwargs):
        before, risk_before = deepcopy(request), deepcopy(risks)
        diagnostic = original(request, risks, settings, **kwargs)
        if diagnostic["decision"] != "error":
            assert request == before and risks == risk_before, "measurement mutated its source"
        observations.append({"request": before, "risks": risk_before, "diagnostic": deepcopy(diagnostic)})
        return diagnostic

    with patch.object(graph, "check_context_budget", side_effect=measure):
        yield observations


def audit(state, client, measured, *, scenario, turn=None, faults=None):
    if turn is not None:
        observe_turn(turn)
    else:
        observe_state(state)
    events = audit_recording(state, client, scenario=scenario, failed_writes=faults)
    assert state["model_calls"] == len(client.requests)
    assert len(state["context_risks"]) == len(state["input"])
    checks = [event for event in events if event["event"] == "context_budget.checked"]
    starts = [event for event in events if event["event"] == "model.started"]
    for request, started in zip(client.requests, starts):
        matching = [value for value in measured if value["diagnostic"]["step_id"] == started["step_id"]]
        assert len(matching) == 1
        value = matching[0]
        assert value["diagnostic"]["decision"] == "allow"
        assert value["request"] == request, "budget measured a different request from the one dispatched"
        assert request["max_output_tokens"] == value["diagnostic"]["output_reserve_tokens"]
        assert request["truncation"] == "disabled" and request["stream"] is False and request["store"] is False
        if started["final_answer_call"]:
            assert request["instructions"] == graph.FINAL_ANSWER_INSTRUCTIONS and request["tool_choice"] == "none"
        else:
            assert "instructions" not in request and "tool_choice" not in request
        assert not {"remaining_model_calls", "read_budget", "read_budget_error"} & request.keys()
        recorded = next(event for event in checks if event["step_id"] == started["step_id"])
        assert events.index(recorded) < events.index(started)
        assert recorded["input_tool_result_call_ids"] == [
            item["call_id"] for item in request["input"] if item.get("type") == "function_call_output"
        ]
        assert recorded["request_sent"] is False
    if state["log_complete"]:
        assert len(checks) == len(measured) or (
            not measured and len(checks) == 1 and checks[0]["decision"] == "error"
        ), "only a failed preparation can stop before the measurement entrypoint"
        assert events[-1]["context_budget"] == state["context_budget"]
    for event, value in zip(checks, measured):
        for key, expected in value["diagnostic"].items():
            assert event.get(key) == expected, (key, event.get(key), expected)
    assert PRIVATE not in json.dumps(events, ensure_ascii=False)
    assert PRIVATE not in (state.get("error") or "")
    print(json.dumps({
        "scenario": scenario, "session_id": turn.session_id if turn else None,
        "run_id": state["run_id"], "checkpoint_id": turn.checkpoint_id if turn else None,
        "checkpoint_complete": turn.checkpoint_complete if turn else None,
        "budget": state["context_budget"],
        "steps": [event["step_id"] for event in checks],
        "call_ids": [item["call_id"] for item in outputs(state)],
        "actual_model_requests": len(client.requests), "log_path": state["log_path"],
        "fault_injection_observations": faults or [],
    }, ensure_ascii=False))
    RUNS.append(state["run_id"])
    return events


async def run(replies, *, prompt="question", settings=SETTINGS, tools=None, executor=None,
              scenario, manager=None, sid=None, faults=None):
    client = LocalClient(replies)
    with resources() as observed, capture_measurements() as measured, patch.object(runtime, "create_client", return_value=client) as factory:
        if manager is None:
            state = await runtime.run_agent(prompt, settings, tools=tools, execute_tool=executor)
            observe_state(state)
            result = None
        else:
            result = await manager.run_turn(sid, prompt)
            observe_turn(result)
            state = result.state
    assert not factory.called or client.closed
    if manager is not None:
        closed_resources(manager, sid, observed)
        if result.checkpoint_complete:
            assert same_saved_facts(await saved_state(manager, result), state)
    else:
        for recorder in observed[0]:
            assert recorder._closed
        for tracer in observed[1]:
            assert tracer._closed and (tracer._worker is None or not tracer._worker.is_alive())
    audit(state, client, measured, scenario=scenario, turn=result, faults=faults)
    return result if result is not None else state, client, measured


async def check_configuration():
    defaults = {
        "context_budget_tokens": 32768, "max_output_tokens": 4096,
        "context_safety_margin_tokens": 1024, "opaque_fallback_tokens": 4096,
        "context_trigger_percent": 90,
    }
    assert all(getattr(SETTINGS, key) == value for key, value in defaults.items())
    for field in defaults:
        invalid = [True, False, -1, 1.5, "2"]
        if field != "context_safety_margin_tokens":
            invalid.append(0)
        if field == "context_trigger_percent":
            invalid.append(101)
        for value in invalid:
            try:
                replace(SETTINGS, **{field: value})
            except ConfigurationError:
                pass
            else:
                raise AssertionError((field, value))
    replace(SETTINGS, context_safety_margin_tokens=0)
    try:
        replace(SETTINGS, context_budget_tokens=5120, context_trigger_percent=100)
    except ConfigurationError:
        pass
    else:
        raise AssertionError("threshold must leave room beyond fixed reserves")
    with TemporaryDirectory() as temporary:
        env = {
            "BASE_URL": SETTINGS.base_url, "MODEL_NAME": SETTINGS.model_name,
            "CONTEXT_BUDGET_TOKENS": "123456", "MAX_OUTPUT_TOKENS": "5000",
            "CONTEXT_SAFETY_MARGIN_TOKENS": "0", "OPAQUE_FALLBACK_TOKENS": "9000",
            "CONTEXT_TRIGGER_PERCENT": "80",
        }
        with patch.dict(os.environ, env, clear=True):
            loaded = load_settings(Path(temporary) / "missing.env")
        assert loaded.context_budget_tokens == 123456 and loaded.max_output_tokens == 5000
        assert loaded.context_safety_margin_tokens == 0 and loaded.opaque_fallback_tokens == 9000
        assert loaded.context_trigger_percent == 80
    print(json.dumps({"scenario": "configuration_validation", "defaults": defaults}))


async def check_transparent_request_and_boundary():
    prompt = [{"role": "user", "content": "中文 \"quoted\" \\ slash\nline"}]
    original = deepcopy(prompt)
    baseline, client, measured = await run([response(message())], prompt=prompt, scenario="transparent_reference")
    request = client.requests[0]
    diagnostic = measured[0]["diagnostic"]
    assert diagnostic["transparent_tokens"] == compact_bytes(request)
    assert diagnostic["opaque_items"] == diagnostic["opaque_risk_tokens"] == 0
    required = diagnostic["required_tokens"]
    assert required == compact_bytes(request) + SETTINGS.max_output_tokens + SETTINGS.context_safety_margin_tokens
    assert prompt == original
    for delta, expected in ((1, "completed"), (0, "context_budget_exceeded"), (-1, "context_budget_exceeded")):
        # This fixture isolates the exact business-request boundary. Compression
        # is unavailable, but its target still needs room for the business reserve.
        configured = replace(
            SETTINGS, context_budget_tokens=required + delta,
            context_trigger_percent=100, context_target_percent=99,
        )
        state, actual, _ = await run([response(message())], prompt=prompt, settings=configured, scenario=f"boundary_{delta:+d}")
        assert state["stop_reason"] == expected
        assert state["model_calls"] == (1 if delta == 1 else 0)
        assert state["context_budget"]["request_sent"] is (delta == 1)
        if delta != 1:
            assert not actual.requests and state["history_end"] == 0
            assert state["usage_totals"] == {"input_tokens": None, "output_tokens": None, "total_tokens": None, "missing_usage_calls": 0}
    state, actual, checked = await run(
        [response(message())], tools=[TOOL], executor=fixed_tool,
        settings=replace(SETTINGS, max_model_calls=1), scenario="schema_and_final_instructions_in_budget",
    )
    assert actual.requests[0]["instructions"] == graph.FINAL_ANSWER_INSTRUCTIONS
    assert checked[0]["request"] == actual.requests[0]
    prepared = prepare_request(SETTINGS.model_name, original, tools=[TOOL], instructions="预算指令", tool_choice="none", max_output_tokens=4096)
    diag = check_context_budget(prepared, [None], SETTINGS, step_id="unit")
    assert diag["transparent_tokens"] == compact_bytes(prepared)
    stripped = {key: value for key, value in prepared.items() if key not in {"tools", "instructions", "tool_choice"}}
    assert compact_bytes(prepared) > compact_bytes(stripped)
    assert state["model_calls"] == 1


async def check_stable_history_and_rejected_input():
    settings = replace(SETTINGS, context_budget_tokens=11000, context_trigger_percent=100)
    manager = SessionManager()
    sid = manager.create_session(settings)
    source = [{"role": "user", "content": [{"type": "input_text", "text": "fact BLUE_19"}]}]
    first, _, _ = await run([response(message("BLUE_19 saved"))], manager=manager, sid=sid, prompt=source, scenario="history_first")
    saved = deepcopy(first.state)
    source[0]["content"][0]["text"] = "CALLER_MUTATION"
    first.state["input"][0]["content"][0]["text"] = "RETURN_MUTATION"
    large = "UNSENT_HUGE_" + "x" * 12000
    refused, client, _ = await run([], manager=manager, sid=sid, prompt=large, scenario="history_first_call_rejected")
    assert not client.requests and refused.checkpoint_complete
    assert refused.state["input"][-1]["content"] == large
    assert refused.state["history_end"] == len(saved["input"])
    assert refused.state["input"][:len(saved["input"])] == saved["input"]
    third, actual, _ = await run([response(message("BLUE_19"))], manager=manager, sid=sid, prompt="Recall?", scenario="history_after_rejection")
    assert actual.requests[0]["input"] == saved["input"] + [{"role": "user", "content": "Recall?"}]
    assert large not in json.dumps(actual.requests)
    assert third.state["model_calls"] == 1 and third.state["usage_totals"]["total_tokens"] == 12
    assert third.state["history_end"] == len(third.state["input"])


async def check_tool_result_limit_and_real_file():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        body = "REAL_BUDGET_BODY_" + "资料" * 2500
        (root / "budget source.txt").write_text(body, encoding="utf-8")
        original, cwd = snapshot(root), Path.cwd()
        settings = replace(SETTINGS, context_budget_tokens=11000, context_trigger_percent=100)
        executor = create_filesystem_executor(root)
        manager = SessionManager()
        sid = manager.create_session(settings, tools=[TEXT_TOOL], execute_tool=executor)
        request = call("real_read", name="read_text_file", arguments='{"path":"budget source.txt"}')
        first, client, measured = await run([response(request, usage=(25, 4, 29))], manager=manager, sid=sid, scenario="file_result_exceeds_next_request")
        state = first.state
        assert state["stop_reason"] == "context_budget_exceeded" and state["model_calls"] == state["tool_calls"] == 1
        assert state["tool_outcomes"]["succeeded"] == 1 and len(measured) == 2
        result = outputs(state)[0]
        assert json.loads(json.loads(result["output"])["result"])["content"] == body
        assert result not in client.requests[0]["input"] and measured[1]["request"]["input"][-1] == result
        assert state["history_end"] == len(state["input"]) and first.checkpoint_complete
        assert state["usage_totals"] == {"input_tokens": 25, "output_tokens": 4, "total_tokens": 29, "missing_usage_calls": 0}
        again, second_client, _ = await run([], manager=manager, sid=sid, prompt="Recall the source", scenario="file_result_stays_in_history")
        assert not second_client.requests and again.state["tool_calls"] == again.state["model_calls"] == 0
        assert again.state["input"][:len(state["input"])] == state["input"]
        assert outputs(again.state) == [result]
        assert again.state["history_end"] == state["history_end"]
        assert snapshot(root) == original and Path.cwd() == cwd
        # A generous budget proves the same genuine output reaches the SDK unchanged.
        roomy = replace(SETTINGS, context_budget_tokens=100000)
        ok, actual, checked = await run([response(request), response(message("used body"))], settings=roomy, tools=[TEXT_TOOL], executor=executor, scenario="real_file_result_dispatched")
        assert actual.requests[1]["input"][-1] == outputs(ok)[0] and len(checked) == 2
        assert body in json.loads(outputs(ok)[0]["output"])["result"]
        assert body not in Path(ok["log_path"]).read_text(encoding="utf-8")


def opaque_item(secret="cipher", **extra):
    return {"type": "reasoning", "id": "r_budget", "summary": [{"type": "summary_text", "text": "visible reasoning"}], "encrypted_content": secret, **extra}


async def check_opaque_counting():
    items = [
        opaque_item(),
        {"type": "provider_unknown", "nested": {"encrypted_content": "do not double count"}},
        {"role": "user", "content": [{"type": "input_text", "text": "visible sibling"}, {"type": "input_image", "image_url": "PRIVATE_IMAGE"}]},
    ]
    original = deepcopy(items)
    risks = annotate_context_risks(items, fallback_tokens=4096, run_id="caller")
    assert [item["opaque_count"] for item in risks] == [1, 1, 1]
    assert all(item["risk_tokens"] == 4096 and item["source_model_call"] is None for item in risks)
    request = prepare_request(SETTINGS.model_name, items)
    diag = check_context_budget(request, risks, SETTINGS, step_id="opaque-unit")
    assert diag["opaque_items"] == 3 and diag["opaque_risk_tokens"] == 3 * 4096
    larger_cipher = deepcopy(items)
    larger_cipher[0]["encrypted_content"] *= 5000
    grown = check_context_budget(prepare_request(SETTINGS.model_name, larger_cipher), risks, SETTINGS)
    assert grown["transparent_tokens"] == diag["transparent_tokens"], "encrypted bytes were charged as transparent text"
    visible_growth = deepcopy(items)
    visible_growth[0]["summary"][0]["text"] += "visible growth" * 10
    assert check_context_budget(prepare_request(SETTINGS.model_name, visible_growth), risks, SETTINGS)["transparent_tokens"] > diag["transparent_tokens"]
    assert items == original
    for output_tokens in (None, 0, -1, True, "99", 1.5):
        usage = {"output_tokens": output_tokens}
        annotated = annotate_context_risks(items, fallback_tokens=777, run_id="model-source", model_call=2, usage=usage)
        assert all(item["risk_tokens"] == 777 for item in annotated)
    genuine = annotate_context_risks(items, fallback_tokens=777, run_id="model-source", model_call=2, usage={"output_tokens": 91})
    assert all(item["risk_tokens"] == 91 and item["source_model_call"] == 2 for item in genuine)
    assert check_context_budget(request, genuine, SETTINGS)["opaque_risk_tokens"] == 273
    for invalid_risks in ([], [None, None, None], [{**genuine[0], "opaque_count": 2}, *genuine[1:]]):
        assert check_context_budget(request, invalid_risks, SETTINGS)["decision"] == "error"
    # Caller-supplied usage in an opaque protocol item is not source evidence.
    forged = [{"type": "provider_unknown", "usage": {"output_tokens": 1}}]
    assert annotate_context_risks(forged, fallback_tokens=777, run_id="caller")[0]["risk_tokens"] == 777
    string_result = [call("opaque_string"), {"type": "function_call_output", "call_id": "opaque_string", "output": '{"encrypted_content":"ordinary tool text"}'}]
    assert annotate_context_risks(string_result, fallback_tokens=777, run_id="caller") == [None, None]
    print(json.dumps({"scenario": "opaque_counting_units", "diagnostic": diag, "model_risk_tokens": 273}))


async def check_opaque_provenance_and_isolation():
    manager = SessionManager()
    settings = replace(SETTINGS, context_budget_tokens=100000)
    a, b = manager.create_session(settings), manager.create_session(settings)
    second_opaque = {"type": "provider_extension", "hidden": "PRIVATE_SECOND_OPAQUE"}
    raw = {"status": "completed", "output": [opaque_item("PRIVATE_CIPHER_A"), second_opaque, message("A")],
           "usage": {"input_tokens": 12, "output_tokens": 73, "total_tokens": 85}}
    first, _, _ = await run([raw], manager=manager, sid=a, scenario="opaque_source_a")
    source = deepcopy(first.state)
    opaque = next(value for value in source["context_risks"] if value is not None)
    assert opaque["risk_tokens"] == 73 and opaque["source_run_id"] == source["run_id"] and opaque["source_model_call"] == 1
    first.state["context_risks"][1]["risk_tokens"] = 999999
    other, _, _ = await run([response(opaque_item("PRIVATE_CIPHER_B"), message("B"), usage=None)], manager=manager, sid=b, scenario="opaque_source_missing_usage_b")
    assert next(value for value in other.state["context_risks"] if value)["risk_tokens"] == settings.opaque_fallback_tokens
    again, client, checks = await run([response(message("follow up"), usage=(10, 999, 1009))], manager=manager, sid=a, prompt="A follow-up", scenario="opaque_source_inherited_unchanged")
    assert client.requests[0]["input"][:len(source["input"])] == source["input"]
    assert checks[0]["diagnostic"]["opaque_risk_tokens"] == 146, "same-response opaque units must not share the output allowance"
    assert again.state["context_risks"][1] == source["context_risks"][1]
    assert "PRIVATE_CIPHER_B" not in json.dumps(client.requests)
    assert again.state["usage_totals"]["output_tokens"] == 999 and again.state["model_calls"] == 1
    assert await saved_state(manager, first) == source


async def check_file_sessions_isolation():
    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        manager, cases = SessionManager(), []
        for name, limit in (("A", 20000), ("B", 30000)):
            root = base / name
            root.mkdir()
            (root / "source.txt").write_text(f"FACT_{name}_BODY", encoding="utf-8")
            settings = replace(SETTINGS, context_budget_tokens=limit)
            sid = manager.create_session(settings, tools=[TEXT_TOOL], execute_tool=create_filesystem_executor(root, max_total_bytes=11))
            cases.append((name, sid, root, settings))
        before, cwd = snapshot(base), Path.cwd()
        historical = {}
        for name, sid, _, settings in cases:
            request = call(f"root_{name}", name="read_text_file", arguments='{"path":"source.txt"}')
            turn, actual, _ = await run([response(request), response(message(f"FACT_{name}_BODY"))], manager=manager, sid=sid, scenario=f"root_{name}_first")
            assert turn.state["context_budget"]["limit_tokens"] == settings.context_budget_tokens
            historical[name] = deepcopy(turn.state)
            assert json.loads(json.loads(outputs(turn.state)[0]["output"])["result"])["content"] == f"FACT_{name}_BODY"
            assert actual.requests[1]["input"][-1] == outputs(turn.state)[0]
            starts = [event for event in read_events(turn.state) if event["event"] == "model.started"]
            assert [event["read_budget"]["remaining_bytes"] for event in starts] == [11, 0]
            assert all("instructions" not in request and request["tools"] == [TEXT_TOOL] for request in actual.requests)
        for name, sid, _, settings in reversed(cases):
            turn, actual, _ = await run([response(message(f"FACT_{name}_BODY"))], manager=manager, sid=sid, prompt="Recall the source", scenario=f"root_{name}_followup")
            assert actual.requests[0]["input"] == historical[name]["input"] + [{"role": "user", "content": "Recall the source"}]
            assert ("FACT_B_BODY" if name == "A" else "FACT_A_BODY") not in json.dumps(actual.requests)
            assert turn.state["model_calls"] == 1 and turn.state["tool_calls"] == 0
            assert turn.state["context_budget"]["limit_tokens"] == settings.context_budget_tokens
            started = next(event for event in read_events(turn.state) if event["event"] == "model.started")
            assert started["read_budget"] == {"limit_bytes": 11, "used_bytes": 0, "remaining_bytes": 11}
            assert started["remaining_model_calls"] == settings.max_model_calls
            assert "instructions" not in actual.requests[0] and actual.requests[0]["tools"] == [TEXT_TOOL]
        assert snapshot(base) == before and Path.cwd() == cwd


async def check_every_call_and_usage():
    state, client, checked = await run([
        response(call("one"), usage=(10, None, None)),
        response(call("two"), usage=None),
        response(message("complete"), usage=(30, 5, 35)),
    ], tools=[TOOL], executor=fixed_tool, scenario="every_model_call_checked")
    assert len(client.requests) == len(checked) == 3
    assert state["usage_totals"] == {"input_tokens": 40, "output_tokens": 5, "total_tokens": 35, "missing_usage_calls": 2}
    assert [request["input"][-1].get("call_id") for request in client.requests[1:]] == ["one", "two"]
    assert [item["diagnostic"]["step_id"] for item in checked] == ["step-0001", "step-0003", "step-0005"]


async def check_measurement_failure():
    invalid = [float("nan"), float("inf"), object()]
    cycle = {"type": "provider_extension"}
    cycle["cycle"] = cycle
    inputs = [[{"type": "provider_extension", "value": value}] for value in invalid] + [[cycle]]
    for index, items in enumerate(inputs):
        # No checkpointer for intentionally nonserializable values: this isolates
        # the pre-dispatch measurement boundary from checkpoint serialization.
        state, client, _ = await run([], prompt=items, scenario=f"measurement_unserializable_{index}")
        assert not client.requests and state["model_calls"] == 0
        assert state["error_code"] == "context_measurement_error"
        assert state["context_budget"]["decision"] == "error" and state["context_budget"]["request_sent"] is False
        assert state["usage_totals"]["missing_usage_calls"] == 0
    with patch.object(runtime, "prepare_request", side_effect=ValueError(PRIVATE)):
        state, _, _ = await run([], scenario="request_preparation_failure")
    assert state["error_code"] == "context_measurement_error"


async def check_session_invalid_storage_boundaries():
    """A real saver failure precedes measurement; JSON rejection is separate.

    These cases do not inject storage faults: InMemorySaver's actual serializer
    decides which input facts can be checkpointed. Invalid JSON that it accepts
    still must fail the model-request measurement and remain non-inheritable.
    """
    cycle = {"type": "provider_extension"}
    cycle["cycle"] = cycle
    cases = (
        ("object", {"type": "provider_extension", "value": object()}, False),
        ("cycle", cycle, False),
        ("set", {"type": "provider_extension", "value": {"one", "two"}}, True),
        ("nan", {"type": "provider_extension", "value": float("nan")}, True),
    )
    for label, invalid, serializable in cases:
        manager = SessionManager()
        sid = manager.create_session(SETTINGS)
        first, _, _ = await run([response(message("SAVED_VALID_HISTORY"))], manager=manager, sid=sid, prompt="remember the fact", scenario=f"session_invalid_{label}_prior")
        prior = deepcopy(first.state)
        prior_storage = deepcopy(manager._checkpointer.storage[sid])
        refused, client, measured = await run([], manager=manager, sid=sid, prompt=[invalid], scenario=f"session_invalid_{label}_submission")
        state = refused.state
        assert not client.requests and state["model_calls"] == state["tool_calls"] == 0
        assert state["usage_totals"] == {"input_tokens": None, "output_tokens": None, "total_tokens": None, "missing_usage_calls": 0}
        assert state["history_end"] == len(prior["input"])
        assert state["input"][:len(prior["input"])] == prior["input"]
        assert len(state["input"]) == len(prior["input"]) + 1
        for namespace, checkpoints in prior_storage.items():
            for checkpoint_id, stored in checkpoints.items():
                assert manager._checkpointer.storage[sid][namespace][checkpoint_id] == stored, "failed submission modified a previous checkpoint"
        if serializable:
            assert await saved_state(manager, first) == prior
            assert refused.checkpoint_complete and len(measured) == 1
            assert state["error_code"] == "context_measurement_error"
            assert state["context_budget"]["decision"] == "error" and not state["context_budget"]["request_sent"]
            third, actual, _ = await run([response(message("SAVED_VALID_HISTORY"))], manager=manager, sid=sid, prompt="Recall the fact", scenario=f"session_invalid_{label}_followup")
            assert third.checkpoint_complete and third.state["stop_reason"] == "completed"
            assert actual.requests[0]["input"] == prior["input"] + [{"role": "user", "content": "Recall the fact"}]
        else:
            assert refused.checkpoint_error == "checkpoint_write_error" and not refused.checkpoint_complete
            assert refused.checkpoint_id is None and state["context_budget"] is None and not measured
            assert state["error_code"] == "checkpoint_write_error"
            assert not any(event["event"] == "context_budget.checked" for event in read_events(state))
            if label == "cycle":
                assert state["input"][-1]["cycle"] is state["input"][-1], "returned facts lost the original cyclic relation"
            else:
                assert type(state["input"][-1]["value"]) is object
            with patch.object(runtime, "create_client", side_effect=AssertionError("blocked session tried to start")):
                try:
                    await manager.run_turn(sid, "must remain blocked")
                except SessionError as exc:
                    assert exc.code == "session_blocked"
                else:
                    raise AssertionError("uncertain save must block this session")
            print(json.dumps({"scenario": f"session_invalid_{label}_blocked", "session_id": sid,
                              "run_id": state["run_id"], "prior_checkpoint_id": first.checkpoint_id,
                              "checkpoint_error": refused.checkpoint_error, "log_path": state["log_path"],
                              "failure_source": "actual_checkpoint_serialization"}))


async def check_budget_recording_failures():
    for event in ("context_budget.checked", "model.started"):
        with recording_failure(event) as faults:
            state, client, measured = await run([], scenario=f"logging_failure_{event}", faults=faults)
        assert faults and not client.requests and state["model_calls"] == 0
        assert state["context_budget"]["decision"] == "allow" and state["context_budget"]["request_sent"] is False
        assert state["error_code"] == "logging_error" and not state["log_complete"]
    manager = SessionManager()
    sid = manager.create_session(SETTINGS)
    with recording_failure("context_budget.checked") as faults:
        turn, _, _ = await run([], manager=manager, sid=sid, prompt="x" * 40000, scenario="budget_exceeded_and_log_failure", faults=faults)
    assert turn.checkpoint_complete and turn.state["context_budget"]["decision"] == "exceeded"
    assert turn.state["error_code"] == "logging_error" and turn.state["history_end"] == 0
    with recording_failure("context_budget.checked") as faults:
        state, client, _ = await run([], prompt=[{"type": "provider_extension", "value": float("nan")}], scenario="measurement_error_and_log_failure", faults=faults)
    assert faults and not client.requests and state["model_calls"] == 0
    assert state["context_budget"]["decision"] == "error"
    assert state["context_budget"]["error_code"] == "context_measurement_error"
    assert state["error_code"] == "logging_error" and not state["log_complete"]
    assert state["usage_totals"]["missing_usage_calls"] == 0

    # Fail the actual write only at the second model step, after a real tool
    # completed. Capture safe attempted-event metadata rather than fake a State.
    original_open, faults, streams = RunRecorder._open, [], []
    class FailSecondBudget(FaultyFile):
        def write(self, value):
            event = json.loads(value)
            if event["event"] == "context_budget.checked" and event.get("step_id") == "step-0003":
                self.event_name = "context_budget.checked"
            return super().write(value)
    def open_file(recorder):
        stream = FailSecondBudget(original_open(recorder), faults)
        streams.append(stream)
        return stream
    async def huge_result(name, arguments):
        return "TOOL_FACT_PRESERVED_" + "x" * 40000
    with patch.object(RunRecorder, "_open", open_file):
        state, client, _ = await run([response(call("before_budget_log_failure"), usage=(22, 3, 25))], tools=[TOOL], executor=huge_result, scenario="tool_result_budget_recording_failure", faults=faults)
    assert faults and all(stream.stream.closed for stream in streams)
    assert state["context_budget"]["decision"] == "exceeded" and state["error_code"] == "logging_error"
    assert state["model_calls"] == state["tool_calls"] == 1 and len(client.requests) == 1
    assert state["usage_totals"]["total_tokens"] == 25 and state["usage_totals"]["missing_usage_calls"] == 0
    assert json.loads(outputs(state)[0]["output"])["ok"] is True and state["history_end"] == len(state["input"])


async def check_budget_checkpoint_failures():
    for stage in ("node", "final"):
        manager = SessionManager()
        sid = manager.create_session(SETTINGS)
        def predicate(checkpoint, metadata, *_):
            if stage == "final":
                return metadata["source"] == "update"
            # Recovery now has its own preparation checkpoint. Fail the settled
            # budget-stop node, preserving this check's original terminal boundary.
            values = checkpoint["channel_values"]
            return (metadata["source"] == "loop"
                    and values.get("stop_reason") == "context_budget_exceeded"
                    and (values.get("context_budget") or {}).get("decision") == "exceeded")
        with checkpoint_fault(manager, "put", predicate) as faults:
            turn, client, _ = await run([], manager=manager, sid=sid, prompt="x" * 40000, scenario=f"budget_{stage}_checkpoint_failure", faults=faults)
        assert faults and not client.requests and not turn.checkpoint_complete
        assert turn.checkpoint_error == "checkpoint_write_error" and turn.checkpoint_id is None
        assert turn.state["context_budget"]["decision"] == "exceeded"
        assert turn.state["error_code"] == "context_budget_exceeded"
        assert turn.state["usage_totals"]["missing_usage_calls"] == 0
    manager = SessionManager()
    async def huge_result(name, arguments):
        return "TOOL_FACT_PRESERVED_" + "x" * 40000
    sid = manager.create_session(SETTINGS, tools=[TOOL], execute_tool=huge_result)
    with checkpoint_fault(manager, "put", lambda cp, meta, *_: meta["source"] == "update") as faults:
        turn, client, _ = await run([response(call("before_budget_save_failure"), usage=(22, 3, 25))], manager=manager, sid=sid, scenario="tool_result_budget_final_save_failure", faults=faults)
    assert faults and turn.state["context_budget"]["decision"] == "exceeded"
    assert turn.state["model_calls"] == turn.state["tool_calls"] == 1 and len(client.requests) == 1
    assert turn.state["usage_totals"]["total_tokens"] == 25 and len(outputs(turn.state)) == 1
    assert not turn.checkpoint_complete and turn.checkpoint_error == "checkpoint_write_error"


async def check_tracing_diagnostics():
    results = []
    for mode in ("disabled", "ok", "connection"):
        with capture("connection" if mode == "connection" else "ok", LANGSMITH_TRACING="false" if mode == "disabled" else "true") as (transport, stderr):
            state, client, _ = await run([], prompt="x" * 40000, scenario=f"tracing_budget_{mode}")
        assert not client.requests and state["context_budget"]["decision"] == "exceeded"
        if mode == "disabled":
            assert not transport.sent
        else:
            serialized = json.dumps(transport.sent)
            assert "context_budget_exceeded" in serialized and "trigger_tokens" in serialized
            assert "x" * 40000 not in serialized
            assert not any(span["run_type"] == "llm" for span in transport.sent)
            if mode == "connection":
                assert stderr.getvalue() and "PRIVATE_TRANSPORT_EXCEPTION" not in stderr.getvalue()
        results.append(state)
    fields = ("stop_reason", "error_code", "model_calls", "tool_calls", "usage_totals", "context_budget")
    assert all({key: result[key] for key in fields} == {key: results[0][key] for key in fields} for result in results[1:])
    with capture() as (transport, _):
        state, client, measured = await run([response(message("normal"))], scenario="tracing_allowed_budget")
    assert state["model_calls"] == 1 and len(client.requests) == 1
    assert "trigger_tokens" in json.dumps(transport.sent)


async def check_cli():
    settings = replace(SETTINGS, context_budget_tokens=8000)
    state, _, _ = await run([], prompt="x" * 10000, settings=settings, scenario="cli_budget_state")
    with redirect_stderr(StringIO()) as error, redirect_stdout(StringIO()) as output:
        assert cli.display_run(state, settings) == 4
    assert not output.getvalue() and state["run_id"] in error.getvalue()
    assert "预算" in error.getvalue()
    clients = [LocalClient([]), LocalClient([response(message("accepted follow-up"))])]
    displayed = []
    original_display = cli.display_turn
    def display(turn, configured):
        observe_turn(turn)
        displayed.append(turn)
        return original_display(turn, configured)
    with patch("builtins.input", side_effect=["x" * 10000, "small", "/exit"]), patch.object(runtime, "create_client", side_effect=clients), patch.object(cli, "display_turn", side_effect=display), capture_measurements() as measured, redirect_stderr(StringIO()) as error, redirect_stdout(StringIO()) as output:
        assert await cli.interactive(settings, {}) == 0
    assert "accepted follow-up" in output.getvalue()
    assert not clients[0].requests and len(clients[1].requests) == 1 and all(client.closed for client in clients)
    assert clients[1].requests[0]["input"] == [{"role": "user", "content": "small"}]
    assert len(displayed) == len(measured) == 2
    for index, (turn, client, checked) in enumerate(zip(displayed, clients, measured, strict=True)):
        assert turn.checkpoint_complete
        audit(turn.state, client, [checked], scenario=f"interactive_budget_turn_{index + 1}", turn=turn)
    print(json.dumps({"scenario": "interactive_continues_after_budget_refusal", "model_calls": [len(client.requests) for client in clients]}))


async def main():
    checks = (
        check_configuration, check_transparent_request_and_boundary,
        check_stable_history_and_rejected_input, check_tool_result_limit_and_real_file,
        check_opaque_counting, check_opaque_provenance_and_isolation,
        check_file_sessions_isolation,
        check_every_call_and_usage, check_measurement_failure,
        check_session_invalid_storage_boundaries,
        check_budget_recording_failures, check_budget_checkpoint_failures,
        check_tracing_diagnostics, check_cli,
    )
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
        for check in checks:
            with check_group(f"check_context_budget/{check.__name__}"):
                await check()
            print(f"PASS {check.__name__}")
    print(f"Passed {len(checks)} context budget groups; {len(RUNS)} captured runs. Logs: {CHECK_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
