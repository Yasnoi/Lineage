"""Reusable fixtures and evidence assertions for budget."""


import json
import math
from benchmarks.checks.support.events import observe_state, observe_turn
from benchmarks.checks.support.runtime import LocalClient, SETTINGS as BASE_SETTINGS, audit_recording, outputs
from benchmarks.checks.support.session_faults import resources, closed_resources
from benchmarks.checks.support.sessions import saved_state
from benchmarks.core.files import ROOT
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from lineage_agent.agent import graph, runtime
from unittest.mock import patch
from uuid import uuid4

CHECK_DIR = ROOT / "logs" / "checks" / str(uuid4())

# Boundary fixtures retain their original scale when production defaults grow.
SETTINGS = replace(
    BASE_SETTINGS, log_dir=CHECK_DIR, context_budget_tokens=32768,
    max_output_tokens=4096, context_safety_margin_tokens=1024,
)

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

def opaque_item(secret="cipher", **extra):
    return {"type": "reasoning", "id": "r_budget", "summary": [{"type": "summary_text", "text": "visible reasoning"}], "encrypted_content": secret, **extra}
