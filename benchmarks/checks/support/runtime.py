"""Reusable fixtures and evidence assertions for runtime."""


import json
from benchmarks.checks.support.events import evidence, observe_state
from benchmarks.core.files import ROOT
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from lineage_agent.agent import runtime
from lineage_agent.config.settings import ModelSettings
from lineage_agent.recording.recorder import RunRecorder
from lineage_agent.tools.types import ToolArgumentError
from openai.types.responses import Response, ResponseUsage
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

CHECK_ID = str(uuid4())

CHECK_DIR = ROOT / "logs" / "checks" / CHECK_ID

SETTINGS = ModelSettings(
    base_url="https://unused.invalid/v1", model_name="local-check", log_dir=CHECK_DIR,
)

TOOL = {
    "type": "function",
    "name": "lookup_sample",
    "description": "Return a fixed value for the requested sample.",
    "parameters": {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    },
    "strict": True,
}

def message(text="done"):
    return {
        "type": "message", "id": "msg_local", "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }

def call(call_id="call_1", *, name="lookup_sample", arguments='{"key":"sample"}'):
    return {
        "type": "function_call", "id": f"fc_{call_id}",
        "call_id": call_id, "name": name, "arguments": arguments,
        "status": "completed",
    }

def response(*items, usage=(10, 2, 12), status="completed"):
    value = Response.model_validate({
        "id": "resp_local", "created_at": 0, "object": "response",
        "model": "local-check", "status": status,
        "output": list(items), "parallel_tool_calls": False,
        "tool_choice": "auto", "tools": [], "usage": None,
    })
    if usage is not None:
        # Construct partial usage deliberately: compatible endpoints can omit fields.
        value.usage = ResponseUsage.model_construct(
            input_tokens=usage[0], output_tokens=usage[1], total_tokens=usage[2],
            input_tokens_details={"cached_tokens": 0},
            output_tokens_details={"reasoning_tokens": 0},
        )
    return value

class LocalClient:
    """Records SDK request arguments and yields predetermined SDK responses."""

    def __init__(self, replies, *, trace=None):
        self.replies = iter(replies)
        self.requests = []
        self.closed = False
        self.trace = trace
        self.responses = SimpleNamespace(create=self.create)

    async def create(self, **kwargs):
        if self.trace is not None:
            self.trace.append(("model", len(self.requests) + 1))
        self.requests.append(deepcopy(kwargs))
        try:
            item = next(self.replies)
        except StopIteration as exc:
            raise AssertionError("unexpected extra model request") from exc
        if isinstance(item, Exception):
            raise item
        return item

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True

async def fixed_tool(name, arguments):
    assert name == "lookup_sample"
    if set(arguments) != {"key"} or not isinstance(arguments["key"], str):
        raise ToolArgumentError("key must be a string and the only parameter")
    return f"fixed:{arguments['key']}"

def read_events(state):
    path = Path(state["log_path"])
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def audit_recording(state, client, *, scenario=None, failed_writes=None):
    """Compare records with independently captured actual SDK requests."""
    observe_state(state)
    events = read_events(state)
    evidence(
        state["log_path"], kind="agent_log", run_id=state["run_id"],
        steps=list(dict.fromkeys(event["step_id"] for event in events if "step_id" in event)),
        call_ids=list(dict.fromkeys(event["call_id"] for event in events if "call_id" in event)),
    )
    assert Path(state["log_path"]).is_absolute()
    assert all(event["run_id"] == state["run_id"] for event in events)
    assert [event["event_seq"] for event in events] == list(range(1, len(events) + 1))
    assert all(event["timestamp"] for event in events)
    started = [event for event in events if event["event"] == "model.started"]
    if state["log_complete"]:
        assert events[0]["event"] == "run.started"
        assert events[-1]["event"] == "run.ended"
        assert "step_id" not in events[0] and "call_id" not in events[0]
        assert len(started) == len(client.requests)
        for outcome, event_type in (
            ("succeeded", "tool.succeeded"), ("rejected", "tool.rejected"),
            ("failed", "tool.failed"), ("skipped", "tool.skipped"),
        ):
            assert state["tool_outcomes"][outcome] == sum(
                event["event"] == event_type for event in events
            ), (outcome, state["tool_outcomes"], events)
        assert sum(event["event"] == "tool.started" for event in events) == state["tool_calls"]
        created_ids = [
            event["call_id"] for event in events if event["event"] == "tool.result_created"
        ]
        initial_ids = [
            item["call_id"] for item in (client.requests[0]["input"] if client.requests else [])
            if item.get("type") == "function_call_output"
        ]
        actual_ids = [item["call_id"] for item in outputs(state)]
        if client.requests:
            assert actual_ids == initial_ids + created_ids
    for event, request in zip(started, client.requests):
        expected = [
            item["call_id"] for item in request["input"]
            if item.get("type") == "function_call_output"
        ]
        assert event["input_tool_result_call_ids"] == expected, (event, request)
        assert event["node"] == "model" and "call_id" not in event
    for event in events:
        if event["event"].startswith("tool."):
            assert "step_id" in event and "request_index" in event
    print(json.dumps({
        "scenario": scenario,
        "run_id": state["run_id"], "stop_reason": state["stop_reason"],
        "model_calls": state["model_calls"], "tool_requests": state["tool_requests"],
        "tool_calls": state["tool_calls"],
        "tool_outcomes": state["tool_outcomes"],
        "errors": [
            {key: event[key] for key in ("event", "step_id", "call_id", "error_code", "stage") if key in event}
            for event in events if "error_code" in event
        ],
        "steps": list(dict.fromkeys(event["step_id"] for event in events if "step_id" in event)),
        "call_ids": list(dict.fromkeys(event["call_id"] for event in events if "call_id" in event)),
        "requested_call_ids": [
            item["call_id"] for item in state["input"]
            if item.get("type") == "function_call" and isinstance(item.get("call_id"), str)
        ],
        "log_complete": state["log_complete"], "log_path": state["log_path"],
        "fault_injection_observations": failed_writes or [],
    }, ensure_ascii=False))
    return events

async def run(
    replies, *, input="question", tools=None, execute_tool=None, limit=5,
    scenario=None, trace=None, failed_writes=None, context_budget_tokens=32768,
):
    client = LocalClient(replies, trace=trace)
    with patch.object(runtime, "create_client", return_value=client) as factory:
        state = await runtime.run_agent(
            input, replace(SETTINGS, max_model_calls=limit, context_budget_tokens=context_budget_tokens),
            tools=tools, execute_tool=execute_tool,
        )
        observe_state(state)
    assert not factory.called or client.closed, "every created client must close"
    assert state["model_calls"] == len(client.requests)
    audit_recording(state, client, scenario=scenario, failed_writes=failed_writes)
    return state, client

def counts(state, model, requested, executed, reason="completed"):
    assert (state["model_calls"], state["tool_requests"], state["tool_calls"]) == (
        model, requested, executed
    ), state
    assert state["stop_reason"] == reason, state

def outputs(state):
    return [item for item in state["input"] if item.get("type") == "function_call_output"]

def assert_batch_returned(state, client, requests, outcomes, attempts):
    """Check ordered protocol results against the next real SDK invocation."""
    counts(state, 2, len(requests), attempts)
    assert state["tool_outcomes"] == {
        name: outcomes.count(name) for name in ("succeeded", "rejected", "failed", "skipped")
    }
    results = outputs(state)
    ids = [item["call_id"] for item in requests]
    assert [item["call_id"] for item in results] == ids
    actual_input = client.requests[1]["input"]
    assert actual_input == client.requests[0]["input"] + requests + results
    assert [item for item in actual_input if item.get("type") == "function_call_output"] == results
    events = read_events(state)
    created = [event for event in events if event["event"] == "tool.result_created"]
    assert [event["call_id"] for event in created] == ids
    assert [event["request_index"] for event in created] == list(range(1, len(ids) + 1))
    assert len({event["step_id"] for event in created}) == 1
    for request, outcome, result, event in zip(requests, outcomes, results, created):
        envelope = json.loads(result["output"])
        assert event["outcome"] == outcome and event["ok"] is envelope["ok"]
        recorded = [
            item for item in events
            if item["event"] == f"tool.{outcome}" and item.get("call_id") == request["call_id"]
        ]
        assert len(recorded) == 1
        assert recorded[0]["event_seq"] < event["event_seq"]
        if outcome == "succeeded":
            assert set(envelope) == {"ok", "result"} and envelope["ok"] is True
        else:
            assert set(envelope) == {"ok", "error"} and envelope["ok"] is False
            if "reason_code" in event:
                assert set(envelope["error"]) == {"code", "message", "reason_code", "reason_message"}
                assert envelope["error"]["reason_code"] == event["reason_code"]
                assert envelope["error"]["reason_message"] == event["reason_message"]
            else:
                assert set(envelope["error"]) == {"code", "message"}
            assert envelope["error"]["code"] == recorded[0]["error_code"]
    next_model = [event for event in events if event["event"] == "model.started"][1]
    assert all(event["event_seq"] < next_model["event_seq"] for event in created)
    assert next_model["input_tool_result_call_ids"] == ids
    assert client.closed and state["log_complete"] is True
    return [json.loads(item["output"]) for item in results]

class FaultyFile:
    """Inject I/O errors and retain only safe metadata about failed operations."""

    def __init__(self, stream, observations, event_name=None, *, fail_close=False, fail_flush=False):
        self.stream = stream
        self.observations = observations
        self.event_name = event_name
        self.fail_close = fail_close
        self.fail_flush = fail_flush
        self.last_item = None

    def observe(self, operation, persistence):
        safe_fields = (
            "run_id", "event", "event_seq", "step_id", "node", "call_id",
            "tool_name", "request_index", "stage", "error_code",
        )
        self.observations.append({
            "source": "fault_injection", "operation": operation, "persistence": persistence,
            **{key: self.last_item[key] for key in safe_fields if key in self.last_item},
        })

    def write(self, value):
        self.last_item = json.loads(value)
        if self.last_item["event"] == self.event_name and not self.fail_flush:
            self.observe("write", "not_written")
            raise OSError("synthetic secret IO failure")
        return self.stream.write(value)

    def flush(self):
        if self.fail_flush and self.last_item["event"] == self.event_name:
            self.observe("flush", "not_confirmed")
            raise OSError("synthetic secret flush failure")
        return self.stream.flush()

    def close(self):
        self.stream.close()
        if self.fail_close:
            raise OSError("synthetic secret close failure")

@contextmanager
def recording_failure(event_name=None, *, create_failure=False, fail_close=False, fail_flush=False):
    original_open = RunRecorder._open
    observations, streams = [], []

    def open_file(recorder):
        if create_failure:
            raise OSError("synthetic secret create failure")
        stream = FaultyFile(
            original_open(recorder), observations, event_name,
            fail_close=fail_close, fail_flush=fail_flush,
        )
        streams.append(stream)
        return stream

    with patch.object(RunRecorder, "_open", open_file):
        try:
            yield observations
        finally:
            assert all(stream.stream.closed for stream in streams), "recording files must close"

def assert_failed_event(state, client, observations, event_name, *, stage):
    assert state["stop_reason"] == "error" and state["log_complete"] is False
    assert client.closed and "synthetic secret" not in state["error"]
    assert len(observations) == 1
    attempted = observations[0]
    assert attempted["source"] == "fault_injection" and attempted["persistence"] == "not_written"
    assert attempted["operation"] == "write" and attempted["event"] == event_name
    assert attempted["run_id"] == state["run_id"] and attempted["stage"] == stage
    events = read_events(state)
    assert all(event["event"] != event_name for event in events)
    assert all(event["event"] != "run.ended" for event in events)
    assert len([event for event in events if event["event"] == "model.started"]) == len(client.requests)
    assert "synthetic secret" not in Path(state["log_path"]).read_text(encoding="utf-8")
    return attempted, events
