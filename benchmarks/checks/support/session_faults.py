"""Reusable fixtures and evidence assertions for session faults."""


import json
from benchmarks.checks.support.events import evidence, observe_turn
from benchmarks.checks.support.runtime import LocalClient, SETTINGS as BASE_SETTINGS, fixed_tool, read_events
from benchmarks.checks.support.sessions import audit_turn, saved_state
from benchmarks.core.files import ROOT
from contextlib import contextmanager
from dataclasses import replace
from lineage_agent.agent import runtime, sessions
from lineage_agent.agent.graph import FINAL_ANSWER_INSTRUCTIONS
from unittest.mock import patch
from uuid import uuid4

CHECK_DIR = ROOT / "logs" / "checks" / str(uuid4())

SETTINGS = replace(BASE_SETTINGS, log_dir=CHECK_DIR)

PRIVATE = "REVIEW_PRIVATE_EXCEPTION_NEVER_RECORD_8340"

RUNS = []

class CountingClient(LocalClient):
    def __init__(self, replies):
        super().__init__(replies)
        self.close_calls = 0

    async def __aexit__(self, *args):
        self.close_calls += 1
        await super().__aexit__(*args)

@contextmanager
def resources():
    """Observe resource ownership without replacing cleanup or state results."""
    recorders, tracers = [], []
    recorder_factory, trace_factory = runtime.RunRecorder, runtime.start_trace

    def recorder(*args, **kwargs):
        value = recorder_factory(*args, **kwargs)
        recorders.append(value)
        evidence(value.path, kind="run_log", run_id=value.run_id)
        return value

    def tracer(*args, **kwargs):
        value = trace_factory(*args, **kwargs)
        tracers.append(value)
        return value

    with patch.object(runtime, "RunRecorder", side_effect=recorder), patch.object(
        runtime, "start_trace", side_effect=tracer,
    ):
        yield recorders, tracers

def closed_resources(manager, sid, observed):
    recorders, tracers = observed
    for recorder in recorders:
        evidence(recorder.path, kind="run_log", run_id=recorder.run_id, session_id=sid)
    assert sid not in manager._checkpointer._recorders
    for recorder in recorders:
        assert recorder._closed
        stream = getattr(recorder._stream, "stream", recorder._stream)
        assert stream is None or stream.closed
    for tracer in tracers:
        assert tracer._closed
        assert tracer._worker is None or not tracer._worker.is_alive()

async def run_turn(manager, sid, replies, *, scenario, prompt="question", observations=None, client=None):
    client = client or CountingClient(replies)
    with resources() as observed, patch.object(runtime, "create_client", return_value=client) as factory:
        result = await manager.run_turn(sid, prompt)
        observe_turn(result)
    closed_resources(manager, sid, observed)
    assert client.close_calls == (1 if factory.called else 0)
    audit_turn(result, client, scenario, observations)
    starts = [event for event in read_events(result.state) if event["event"] == "model.started"]
    for request, event in zip(client.requests, starts, strict=True):
        if event["final_answer_call"]:
            assert request["instructions"] == FINAL_ANSWER_INSTRUCTIONS and request["tool_choice"] == "none"
        else:
            assert "instructions" not in request and "tool_choice" not in request
        assert not {"remaining_model_calls", "read_budget", "read_budget_error"} & request.keys()
    assert PRIVATE not in json.dumps(read_events(result.state), ensure_ascii=False)
    assert PRIVATE not in (result.state["error"] or "")
    if result.checkpoint_complete:
        assert await saved_state(manager, result) == result.state
    RUNS.append(result.state["run_id"])
    return result, client

async def blocked(manager, sid):
    with patch.object(runtime, "create_client", side_effect=AssertionError("blocked session started")):
        try:
            await manager.run_turn(sid, "must not execute")
        except sessions.SessionError as exc:
            assert exc.code == "session_blocked"
        else:
            raise AssertionError("session did not remain blocked")

class InvalidMapping(dict):
    def __getitem__(self, key):
        raise RuntimeError(PRIVATE)

class MetadataExecutor:
    def __init__(self, *, fail_at=(), invalid=False, inner=None):
        self.fail_at = set(fail_at)
        self.invalid = invalid
        self.inner = inner
        self.reads = 0

    @property
    def read_budget(self):
        self.reads += 1
        if self.reads in self.fail_at or -1 in self.fail_at:
            if self.invalid:
                return InvalidMapping()
            raise RuntimeError(PRIVATE)
        return {"limit_bytes": 9, "used_bytes": 0, "remaining_bytes": 9}

    async def __call__(self, name, arguments):
        return await (self.inner or fixed_tool)(name, arguments)

def report_aborted(sid, checkpoint_id, observed, scenario):
    if not observed[0]:
        print(json.dumps({"scenario": scenario, "session_id": sid, "run_id": None, "last_confirmed_checkpoint_id": checkpoint_id, "log_path": None, "execution_started": False, "checkpoint_unchanged": True}, ensure_ascii=False))
    for recorder in observed[0]:
        evidence(recorder.path, kind="run_log", run_id=recorder.run_id,
                 session_id=sid, last_confirmed_checkpoint_id=checkpoint_id)
        events = [json.loads(line) for line in recorder.path.read_text(encoding="utf-8").splitlines()]
        assert all(event.get("session_id") == sid for event in events)
        assert not any(event["event"] == "model.started" for event in events)
        assert PRIVATE not in json.dumps(events, ensure_ascii=False)
        print(json.dumps({"scenario": scenario, "session_id": sid, "run_id": recorder.run_id, "last_confirmed_checkpoint_id": checkpoint_id, "log_path": str(recorder.path), "model_calls": 0, "checkpoint_unchanged": True}, ensure_ascii=False))
