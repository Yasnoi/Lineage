"""Reusable fixtures and evidence assertions for sessions."""


import asyncio
import json
from benchmarks.checks.support.events import observe_turn
from benchmarks.checks.support.runtime import LocalClient, SETTINGS, audit_recording, call, outputs
from benchmarks.core.files import ROOT
from contextlib import contextmanager
from dataclasses import replace
from langgraph.checkpoint.memory import InMemorySaver
from lineage_agent.agent import runtime, sessions
from unittest.mock import patch
from uuid import uuid4

CHECK_DIR = ROOT / "logs" / "checks" / str(uuid4())

SETTINGS = replace(SETTINGS, log_dir=CHECK_DIR)

RUNS = []

PRIVATE = "SESSION_PRIVATE_EXCEPTION_NEVER_LOG"

def config(session_id, checkpoint_id=None):
    values = {"thread_id": session_id}
    if checkpoint_id is not None:
        values["checkpoint_id"] = checkpoint_id
    return {"configurable": values}

async def saved_state(manager, turn):
    checkpoint = await manager._checkpointer.aget_tuple(config(turn.session_id, turn.checkpoint_id))
    assert checkpoint is not None
    assert checkpoint.config["configurable"]["checkpoint_id"] == turn.checkpoint_id
    values = checkpoint.checkpoint["channel_values"]
    return {key: values[key] for key in turn.state}

def audit_turn(turn, client, scenario, observations=None):
    observe_turn(turn)
    state = turn.state
    assert state["model_calls"] == len(client.requests)
    assert 0 <= state["history_end"] <= len(state["input"])
    events = audit_recording(state, client, scenario=scenario, failed_writes=observations)
    assert all(event.get("session_id") == turn.session_id for event in events)
    if turn.checkpoint_complete:
        assert turn.checkpoint_id and turn.checkpoint_error is None
    else:
        assert turn.checkpoint_id is None and turn.checkpoint_error
    if state["log_complete"]:
        ended = events[-1]
        assert ended["checkpoint_finalization"] in {"pending", "failed"}
        assert ended.get("last_confirmed_checkpoint_id") != turn.checkpoint_id or turn.checkpoint_id is None
    for index, request in enumerate(client.requests):
        # Compare protocol results with the real requests, not only log records.
        actual = [item for item in request["input"] if item.get("type") == "function_call_output"]
        assert actual == outputs(state)[:len(actual)]
        assert request["stream"] is False and request["store"] is False
        assert "previous_response_id" not in request
    assert PRIVATE not in json.dumps(events, ensure_ascii=False)
    assert PRIVATE not in (state.get("error") or "")
    print(json.dumps({
        "scenario": scenario, "session_id": turn.session_id, "run_id": state["run_id"],
        "checkpoint_id": turn.checkpoint_id, "checkpoint_complete": turn.checkpoint_complete,
        "checkpoint_error": turn.checkpoint_error, "history_end": state["history_end"],
        "steps": list(dict.fromkeys(event["step_id"] for event in events if "step_id" in event)),
        "call_ids": list(dict.fromkeys(event["call_id"] for event in events if "call_id" in event)),
        "log_path": state["log_path"], "fault_injection_observations": observations or [],
    }, ensure_ascii=False))
    RUNS.append(state["run_id"])
    return events

async def turn(manager, session_id, replies, *, prompt="question", scenario, client=None, observations=None):
    client = client or LocalClient(replies)
    with patch.object(runtime, "create_client", return_value=client) as factory:
        result = await manager.run_turn(session_id, prompt)
        observe_turn(result)
    assert not factory.called or client.closed, "every opened client must close"
    audit_turn(result, client, scenario, observations)
    if result.checkpoint_complete:
        assert await saved_state(manager, result) == result.state
    return result, client

async def rejected(manager, session_id, prompt, expected):
    with patch.object(runtime, "create_client", side_effect=AssertionError("rejected request opened a client")):
        try:
            await manager.run_turn(session_id, prompt)
        except sessions.SessionError as exc:
            assert exc.code == expected, exc.code
            assert PRIVATE not in str(exc)
        else:
            raise AssertionError(f"expected session rejection: {expected}")
    print(json.dumps({"scenario": expected, "session_id": session_id, "execution_started": False}))

def file_call(identifier, path, name="read_text_file"):
    args = {"path": path}
    if name == "list_directory":
        args["after"] = None
    return call(identifier, name=name, arguments=json.dumps(args, ensure_ascii=False))

class GatedClient(LocalClient):
    def __init__(self, replies):
        super().__init__(replies)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def create(self, **kwargs):
        self.started.set()
        await self.release.wait()
        return await super().create(**kwargs)

@contextmanager
def checkpoint_fault(manager, method, predicate):
    """Fail the real saver operation beneath its guarded subclass, not final State."""
    original = getattr(InMemorySaver, method)
    observations = []

    def inject(saver, cfg, *args, **kwargs):
        if saver is manager._checkpointer and not observations and predicate(*args, **kwargs):
            observations.append({
                "source": "fault_injection", "operation": method,
                "session_id": cfg["configurable"]["thread_id"],
                "checkpoint_id": cfg["configurable"].get("checkpoint_id"),
                "persistence": "operation_failed",
            })
            raise OSError(PRIVATE)
        return original(saver, cfg, *args, **kwargs)

    with patch.object(InMemorySaver, method, inject):
        yield observations

class CloseFailureClient(LocalClient):
    async def __aexit__(self, *_):
        self.closed = True
        raise OSError(PRIVATE)
