"""Process-local session acceptance using native checkpoints and captured SDK input."""

import asyncio
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from check_react import (
    LocalClient, SETTINGS, TOOL, audit_recording, call, fixed_tool, message,
    outputs, read_events, recording_failure, response,
)
from check_text import snapshot
from check_tracing import capture, stable
from lineage_agent.agent import runtime, sessions
from lineage_agent.agent.checkpointing import CheckpointError
from lineage_agent.agent.sessions import SessionManager
from lineage_agent.entrypoints import cli
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL


CHECK_DIR = Path(__file__).resolve().parents[1] / "logs" / "checks" / str(uuid4())
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


async def check_history_and_copy_isolation():
    manager = SessionManager()
    sid = manager.create_session(SETTINGS)
    source = [{"role": "user", "content": [{"type": "input_text", "text": "LOCAL_FACT_BLUE_17"}]}]
    original = deepcopy(source)
    first, _ = await turn(manager, sid, [response(message("Noted BLUE_17"))], prompt=source, scenario="history_first")
    saved = deepcopy(first.state)
    source[0]["content"][0]["text"] = "MUTATED_CALLER"
    first.state["input"][0]["content"][0]["text"] = "MUTATED_RETURN"
    first.state["result"]["raw_response"]["output"][0]["content"][0]["text"] = "MUTATED_RAW"
    first.state["usage_totals"]["total_tokens"] = 999
    assert await saved_state(manager, first) == saved
    second, client = await turn(manager, sid, [response(message("BLUE_17"))], prompt="What was the fact?", scenario="history_followup")
    expected = saved["input"] + [{"role": "user", "content": "What was the fact?"}]
    assert client.requests[0]["input"] == expected
    assert client.requests[0]["input"][:len(original)] == original
    assert second.state["model_calls"] == 1 and second.state["usage_totals"]["total_tokens"] == 12
    assert second.state["run_id"] != first.state["run_id"]
    assert second.state["history_end"] == len(second.state["input"]) == 4
    assert first.checkpoint_id != second.checkpoint_id
    assert await saved_state(manager, first) == saved, "historical checkpoint was mutated"
    other = SessionManager()
    await rejected(other, sid, "not a disk-backed session", "unknown_session")


def file_call(identifier, path, name="read_text_file"):
    args = {"path": path}
    if name == "list_directory":
        args["after"] = None
    return call(identifier, name=name, arguments=json.dumps(args, ensure_ascii=False))


async def check_file_history_and_budget():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        body = "SESSION_BODY_HISTORY_中文"
        name = "记录 space.md"
        (root / name).write_text(body, encoding="utf-8")
        before, cwd = snapshot(root), Path.cwd()
        size = len(body.encode())
        executor = create_filesystem_executor(root, max_bytes=size, max_total_bytes=size)
        tools = deepcopy([DIRECTORY_TOOL, TEXT_TOOL])
        manager = SessionManager()
        sid = manager.create_session(SETTINGS, tools=tools, execute_tool=executor)
        tools[1]["name"] = "MUTATED_TOOL_NAME"
        first, client = await turn(manager, sid, [
            response(file_call("files_first", ".", "list_directory")),
            response(file_call("read_first", name)), response(message("Found the local source.")),
        ], prompt="Find and inspect the source.", scenario="file_history_first")
        first_outputs = deepcopy(outputs(first.state))
        assert json.loads(json.loads(first_outputs[-1]["output"])["result"])["content"] == body
        assert client.requests[-1]["input"][-1] == first_outputs[-1]
        assert first.state["tool_calls"] == 2 and first.state["model_calls"] == 3
        second, following = await turn(manager, sid, [response(message(body))], prompt="What was in that source?", scenario="file_history_followup")
        assert following.requests[0]["input"] == first.state["input"] + [{"role": "user", "content": "What was in that source?"}]
        assert [item for item in following.requests[0]["input"] if item.get("type") == "function_call_output"] == first_outputs
        assert second.state["tool_calls"] == second.state["tool_requests"] == 0
        third, _ = await turn(manager, sid, [response(file_call("read_again", name)), response(message("Read again."))], scenario="file_budget_reset")
        assert third.state["tool_outcomes"]["succeeded"] == 1
        for result, expected in ((first, size), (second, 0), (third, size)):
            events = read_events(result.state)
            assert events[0]["read_budget"]["used_bytes"] == 0
            assert events[-1]["read_budget"]["used_bytes"] == expected
            assert body not in Path(result.state["log_path"]).read_text(encoding="utf-8")
        assert executor.read_budget["used_bytes"] == 0
        assert snapshot(root) == before and Path.cwd() == cwd


async def check_failure_history_boundaries():
    malformed = response(message()).model_dump(mode="json")
    malformed["output"][0]["role"] = "developer"
    scenarios = (
        ("model_failure", [RuntimeError(PRIVATE)], 5, "error", 0),
        ("model_limit", [response(call("unexecuted"))], 1, "max_model_calls", 1),
        ("invalid_protocol", [malformed], 5, "error", 0),
        ("duplicate_calls", [response(call("duplicate"), call("duplicate"))], 5, "error", 2),
        ("tool_error_then_answer", [response(call("rejected", arguments='{"key":1}')), response(message("Could not read."))], 5, "completed", 1),
    )
    for scenario, replies, limit, reason, requests in scenarios:
        manager = SessionManager()
        sid = manager.create_session(replace(SETTINGS, max_model_calls=limit), tools=[TOOL], execute_tool=fixed_tool)
        first, _ = await turn(manager, sid, replies, prompt="Initial request", scenario=scenario)
        assert first.state["stop_reason"] == reason and first.state["tool_requests"] == requests
        inherited = deepcopy(first.state["input"][:first.state["history_end"]])
        second, client = await turn(manager, sid, [response(message("Next request completed."))], prompt="Continue independently", scenario=scenario + "_next")
        assert client.requests[0]["input"] == inherited + [{"role": "user", "content": "Continue independently"}]
        assert runtime._input_association_error(client.requests[0]["input"]) is None
        assert second.state["stop_reason"] == "completed" and second.state["error"] is None
        assert second.state["model_calls"] == 1 and second.state["tool_calls"] == 0
        assert second.state["tool_outcomes"] == dict.fromkeys(("succeeded", "rejected", "failed", "skipped"), 0)
        if scenario == "tool_error_then_answer":
            assert len(outputs(second.state)) == 1
            assert json.loads(outputs(second.state)[0]["output"])["ok"] is False
        else:
            assert not outputs(second.state)


async def check_partial_batch_and_usage():
    manager = SessionManager()
    sid = manager.create_session(SETTINGS, tools=[TOOL], execute_tool=fixed_tool)
    with recording_failure("tool.result_created") as observations:
        first, client = await turn(manager, sid, [response(call("partial_a"), call("partial_b"))], scenario="partial_batch_log_failure", observations=observations)
    assert first.state["tool_calls"] == 1 and first.state["tool_outcomes"]["skipped"] == 1
    assert len(outputs(first.state)) == 1 and len(client.requests) == 1
    assert first.state["history_end"] == 1
    second, captured = await turn(manager, sid, [response(message("fresh"), usage=None)], scenario="after_partial_batch")
    assert len(captured.requests[0]["input"]) == 2 and not outputs(second.state)
    assert second.state["usage_totals"] == {"input_tokens": None, "output_tokens": None, "total_tokens": None, "missing_usage_calls": 1}
    third, _ = await turn(manager, sid, [response(call("known"), usage=(5, None, 8)), response(message("answer"), usage=(7, 3, 10))], scenario="usage_partial_and_known")
    assert third.state["usage_totals"] == {"input_tokens": 12, "output_tokens": 3, "total_tokens": 18, "missing_usage_calls": 1}
    assert third.state["result"]["usage"] == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}


async def check_invalid_new_input():
    manager = SessionManager()
    sid = manager.create_session(SETTINGS)
    first, _ = await turn(manager, sid, [response(message("valid historical fact"))], scenario="before_bad_association")
    bad_input = [{"role": "user", "content": "uncommitted new request"}, call("no_result")]
    second, client = await turn(manager, sid, [], prompt=bad_input, scenario="bad_new_input_association")
    assert not client.requests and second.state["error_code"] == "missing_tool_result"
    assert second.state["history_end"] == len(first.state["input"])
    third, client = await turn(manager, sid, [response(message("valid historical fact"))], prompt="Recall valid history", scenario="after_bad_association")
    assert client.requests[0]["input"] == first.state["input"] + [{"role": "user", "content": "Recall valid history"}]
    assert third.state["history_end"] == 4


class GatedClient(LocalClient):
    def __init__(self, replies):
        super().__init__(replies)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def create(self, **kwargs):
        self.started.set()
        await self.release.wait()
        return await super().create(**kwargs)


async def check_concurrency():
    with TemporaryDirectory() as temporary:
        roots = [Path(temporary) / name for name in ("A", "B")]
        manager = SessionManager()
        ids = []
        for index, root in enumerate(roots):
            root.mkdir()
            (root / "fact.txt").write_text(f"ISOLATED_FACT_{index}", encoding="utf-8")
            ids.append(manager.create_session(SETTINGS, tools=[TEXT_TOOL], execute_tool=create_filesystem_executor(root)))
        first = GatedClient([response(file_call("a_read", "fact.txt")), response(message("A noted."))])
        second = LocalClient([response(file_call("b_read", "fact.txt")), response(message("B noted."))])
        with patch.object(runtime, "create_client", side_effect=[first, second]) as factory:
            pending = asyncio.create_task(manager.run_turn(ids[0], "Inspect A"))
            await asyncio.wait_for(first.started.wait(), 5)
            try:
                await manager.run_turn(ids[0], "must not overlap")
            except sessions.SessionError as exc:
                assert exc.code == "session_busy"
            else:
                raise AssertionError("overlapping turn was accepted")
            b = await manager.run_turn(ids[1], "Inspect B")
            first.release.set()
            a = await pending
        assert factory.call_count == 2 and first.closed and second.closed
        for result, client, index in ((a, first, 0), (b, second, 1)):
            audit_turn(result, client, f"interleaved_{index}")
            assert f"ISOLATED_FACT_{index}" in json.dumps(outputs(result.state))
            assert f"ISOLATED_FACT_{1-index}" not in json.dumps(result.state)
            later, observed = await turn(manager, ids[index], [response(message("Remembered."))], prompt="Which fact?", scenario=f"interleaved_followup_{index}")
            assert observed.requests[0]["input"] == result.state["input"] + [{"role": "user", "content": "Which fact?"}]
            assert later.state["tool_calls"] == 0
        assert a.state["run_id"] != b.state["run_id"] and a.checkpoint_id != b.checkpoint_id


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


async def check_checkpoint_write_failures():
    scenarios = (
        ("initial_save", "put", lambda cp, meta, *_: meta["source"] == "input", 0, 0),
        ("model_checkpoint", "put", lambda cp, meta, *_: meta["source"] == "loop" and cp["channel_values"].get("model_calls") == 1, 1, 0),
        ("tools_checkpoint", "put", lambda cp, meta, *_: meta["source"] == "loop" and cp["channel_values"].get("tool_calls") == 1, 1, 1),
        ("model_pending", "put_writes", lambda writes, *_: dict(writes).get("model_calls") == 1, 1, 0),
        ("tools_pending", "put_writes", lambda writes, *_: dict(writes).get("tool_calls") == 1, 1, 1),
        ("final_save", "put", lambda cp, meta, *_: meta["source"] == "update", 2, 1),
    )
    for scenario, method, predicate, model_calls, tool_calls in scenarios:
        manager = SessionManager()
        sid = manager.create_session(SETTINGS, tools=[TOOL], execute_tool=fixed_tool)
        with checkpoint_fault(manager, method, predicate) as observations:
            result, client = await turn(manager, sid, [response(call("before_fault")), response(message("retained answer"))], scenario=scenario, observations=observations)
        assert observations, f"fault did not reach actual storage: {scenario}"
        assert not result.checkpoint_complete and result.checkpoint_id is None
        assert result.state["model_calls"] == model_calls and result.state["tool_calls"] == tool_calls
        assert len(client.requests) == model_calls
        assert result.state["usage_totals"]["total_tokens"] == (12 * model_calls if model_calls else None)
        assert len(outputs(result.state)) == tool_calls
        skipped = [event for event in read_events(result.state) if event["event"] == "tool.skipped"]
        if scenario in {"model_checkpoint", "model_pending"}:
            assert result.state["tool_requests"] == result.state["tool_outcomes"]["skipped"] == 1
            assert len(skipped) == 1 and skipped[0]["reason"] == "checkpoint_error"
            assert skipped[0]["step_id"] == "step-0001" and skipped[0]["call_id"] == "before_fault"
            assert skipped[0]["request_index"] == 1 and skipped[0]["node"] == "model"
            assert not outputs(result.state)
        else:
            assert result.state["tool_outcomes"]["skipped"] == 0 and not skipped
        if scenario == "final_save":
            assert result.state["stop_reason"] == "completed" and result.state["result"]["text"] == "retained answer"
        else:
            assert result.state["stop_reason"] == "error"
        await rejected(manager, sid, "must not retry", "session_blocked")


async def check_checkpoint_and_skipped_log_failure():
    manager = SessionManager()
    sid = manager.create_session(SETTINGS, tools=[TOOL], execute_tool=fixed_tool)
    client = LocalClient([response(call("skip_when_checkpoint_failed")), response(message("must not run"))])
    predicate = lambda cp, meta, *_: meta["source"] == "loop" and cp["channel_values"].get("model_calls") == 1
    with checkpoint_fault(manager, "put", predicate) as checkpoint_observations:
        with recording_failure("tool.skipped") as log_observations:
            with patch.object(runtime, "create_client", return_value=client):
                result = await manager.run_turn(sid, "checkpoint failure and log failure")
    assert len(checkpoint_observations) == len(log_observations) == 1
    audit_turn(result, client, "checkpoint_and_skip_log_failure", checkpoint_observations + log_observations)
    assert client.closed and len(client.requests) == 1
    assert not result.checkpoint_complete and result.checkpoint_error == "checkpoint_write_error"
    assert result.state["stop_reason"] == "error" and result.state["error_code"] == "logging_error"
    assert result.state["log_complete"] is False
    assert result.state["tool_calls"] == 0 and result.state["tool_requests"] == 1
    assert result.state["tool_outcomes"] == {"succeeded": 0, "rejected": 0, "failed": 0, "skipped": 1}
    assert result.state["usage_totals"]["total_tokens"] == 12 and not outputs(result.state)
    attempted = log_observations[0]
    assert attempted["event"] == "tool.skipped" and attempted["step_id"] == "step-0001"
    assert attempted["call_id"] == "skip_when_checkpoint_failed"
    assert attempted["persistence"] == "not_written"
    assert not any(event["event"] == "tool.skipped" for event in read_events(result.state))
    await rejected(manager, sid, "must not execute", "session_blocked")


async def check_checkpoint_read_failure():
    manager = SessionManager()
    sid = manager.create_session(SETTINGS)
    first, _ = await turn(manager, sid, [response(message("saved"))], scenario="before_read_failure")
    original = deepcopy(manager._checkpointer.storage[sid])
    with checkpoint_fault(manager, "get_tuple", lambda *_: True) as observations:
        result, client = await turn(manager, sid, [], scenario="checkpoint_read_failure", observations=observations)
    assert observations and not client.requests and not result.checkpoint_complete
    assert result.state["model_calls"] == 0
    await rejected(manager, sid, "must not auto-reload", "session_blocked")
    assert manager._checkpointer.storage[sid] == original


async def check_creation_and_start_log_failure():
    with patch.object(InMemorySaver, "__init__", side_effect=OSError(PRIVATE)):
        try:
            SessionManager()
        except CheckpointError as exc:
            assert exc.code == "checkpoint_create_error" and PRIVATE not in str(exc)
        else:
            raise AssertionError("checkpoint construction fault was hidden")
    print(json.dumps({"scenario": "checkpoint_create_failure", "execution_started": False}))
    manager = SessionManager()
    sid = manager.create_session(SETTINGS)
    first, _ = await turn(manager, sid, [response(message("prior fact"))], scenario="before_start_log_failure")
    with recording_failure("run.started") as observations:
        failed, client = await turn(manager, sid, [], prompt="request with log failure", scenario="start_log_failure_retains_history", observations=observations)
    assert not client.requests and not failed.state["log_complete"]
    assert failed.state["input"][:len(first.state["input"])] == first.state["input"]
    third, captured = await turn(manager, sid, [response(message("prior fact"))], prompt="Recall the earlier fact", scenario="after_start_log_failure")
    assert captured.requests[0]["input"][:len(first.state["input"])] == first.state["input"]
    assert third.state["model_calls"] == 1


class CloseFailureClient(LocalClient):
    async def __aexit__(self, *_):
        self.closed = True
        raise OSError(PRIVATE)


async def check_finalization_combinations():
    for failure in ("client", "log", "both"):
        manager = SessionManager()
        sid = manager.create_session(SETTINGS)
        client = CloseFailureClient([response(message("answer remains"))]) if failure != "log" else LocalClient([response(message("answer remains"))])
        with checkpoint_fault(manager, "put", lambda cp, meta, *_: meta["source"] == "update") as observations:
            if failure == "client":
                result, _ = await turn(manager, sid, [], client=client, scenario="final_checkpoint_and_client", observations=observations)
            else:
                with recording_failure("run.ended") as logs:
                    result, _ = await turn(manager, sid, [], client=client, scenario="final_checkpoint_and_" + failure, observations=observations)
                assert logs and not result.state["log_complete"]
        assert observations and client.closed and not result.checkpoint_complete
        assert result.state["model_calls"] == 1 and result.state["usage_totals"]["total_tokens"] == 12
        assert result.state["result"]["text"] == "answer remains"
        if failure != "log":
            assert result.state["cleanup_error"]["code"] == "client_close_error"
        assert result.state["error_code"] == ("client_close_error" if failure == "client" else "logging_error")


async def check_tracing_sessions():
    results = []
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        body = "SESSION_TOOL_BODY_NOT_IN_TOOL_TRACE"
        (root / "note.txt").write_text(body, encoding="utf-8")
        for enabled, mode in ((False, "ok"), (True, "ok"), (True, "connection")):
            manager = SessionManager()
            sid = manager.create_session(SETTINGS, tools=[TEXT_TOOL], execute_tool=create_filesystem_executor(root))
            with capture(mode, LANGSMITH_TRACING="true" if enabled else "false") as (transport, stderr):
                first, _ = await turn(manager, sid, [response(file_call("trace_read", "note.txt")), response(message("Source ready."))], scenario=f"trace_first_{enabled}_{mode}")
                second, client = await turn(manager, sid, [response(message(body))], prompt="Quote the source.", scenario=f"trace_followup_{enabled}_{mode}")
            results.append((first.state, second.state, client.requests))
            if not enabled:
                assert not transport.sent
            elif mode == "ok":
                assert not stderr.getvalue(), stderr.getvalue()
                assert transport.sent
                for span in transport.sent:
                    assert span["extra"]["metadata"]["session_id"] == sid
                    assert body not in json.dumps(span.get("inputs", {}))
                    if span["run_type"] == "tool":
                        assert body not in json.dumps(span.get("outputs", {}))
                model_outputs = [span.get("outputs") for span in transport.sent if span["run_type"] == "llm"]
                assert body in json.dumps(model_outputs), "model quotes keep their existing text policy"
            else:
                assert stderr.getvalue() and PRIVATE not in stderr.getvalue()
        base_first, base_second, base_requests = results[0]
        for first, second, requests in results[1:]:
            assert stable(first) == stable(base_first)
            assert stable(second) == stable(base_second) and requests == base_requests


def check_cli():
    """Drive the real CLI; substitute input and SDK, not session results."""
    ids = []
    actual_create = SessionManager.create_session

    def remember_session(manager, *args, **kwargs):
        identifier = actual_create(manager, *args, **kwargs)
        ids.append(identifier)
        return identifier

    def input_lines():
        yield "Remember CLI_FACT"
        yield "What fact?"
        yield "/new"
        yield "Fresh conversation"
        yield "/switch " + ids[0]
        yield "Back to the first fact"
        yield "/exit"

    clients = [LocalClient([response(message(text))]) for text in ("CLI_FACT", "CLI_FACT", "fresh", "CLI_FACT")]
    turns = []
    real_display = cli.display_turn

    def capture_turn(result, settings):
        turns.append(result)
        return real_display(result, settings)
    out, err = StringIO(), StringIO()
    with patch("sys.argv", ["lineage-agent"]), patch("builtins.input", side_effect=input_lines()), patch.object(
        cli, "load_settings", return_value=SETTINGS,
    ), patch.object(SessionManager, "create_session", remember_session), patch.object(cli, "display_turn", capture_turn), patch.object(runtime, "create_client", side_effect=clients), redirect_stdout(out), redirect_stderr(err):
        assert cli.main() == 0
    assert all(client.closed for client in clients)
    assert len(clients[1].requests[0]["input"]) == 3
    assert clients[2].requests[0]["input"] == [{"role": "user", "content": "Fresh conversation"}]
    assert len(clients[3].requests[0]["input"]) == 5
    assert "Fresh conversation" not in json.dumps(clients[3].requests[0]["input"])
    assert "CLI_FACT" in out.getvalue()
    assert "SESSION_PRIVATE" not in err.getvalue()
    for index, (result, client) in enumerate(zip(turns, clients, strict=True)):
        audit_turn(result, client, f"cli_interactive_{index}")
    print(json.dumps({"scenario": "cli_interactive_new_switch_exit", "session_ids": ids, "actual_model_requests": 4}))
    for ending, expected in ((EOFError(), 0), (KeyboardInterrupt(), 130)):
        with patch("sys.argv", ["lineage-agent"]), patch("builtins.input", side_effect=ending), patch.object(
            cli, "load_settings", return_value=SETTINGS,
        ), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            assert cli.main() == expected
    client = LocalClient([response(message("single answer"))])
    with patch("sys.argv", ["lineage-agent", "single question"]), patch.object(cli, "load_settings", return_value=SETTINGS), patch.object(
        runtime, "create_client", return_value=client,
    ), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        assert cli.main() == 0
    assert client.closed and len(client.requests) == 1

    manager = SessionManager()
    turns.clear()
    clients = [LocalClient([response(message("answer before failed save"))]), LocalClient([response(message("new session answer"))])]
    with checkpoint_fault(manager, "put", lambda cp, meta, *_: meta["source"] == "update") as observations:
        with patch("sys.argv", ["lineage-agent"]), patch("builtins.input", side_effect=["first", "blocked", "/new", "independent", "/exit"]), patch.object(
            cli, "load_settings", return_value=SETTINGS,
        ), patch.object(cli, "SessionManager", return_value=manager), patch.object(cli, "display_turn", capture_turn), patch.object(
            runtime, "create_client", side_effect=clients,
        ) as factory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            assert cli.main() == 0
    assert factory.call_count == len(turns) == 2
    assert not turns[0].checkpoint_complete and turns[1].checkpoint_complete
    for index, (result, client) in enumerate(zip(turns, clients, strict=True)):
        audit_turn(result, client, f"cli_save_failure_then_new_{index}", observations if index == 0 else None)


async def main():
    checks = (
        check_history_and_copy_isolation, check_file_history_and_budget,
        check_failure_history_boundaries, check_partial_batch_and_usage, check_invalid_new_input,
        check_concurrency, check_checkpoint_write_failures, check_checkpoint_and_skipped_log_failure,
        check_checkpoint_read_failure,
        check_creation_and_start_log_failure, check_finalization_combinations, check_tracing_sessions,
    )
    for check in checks:
        await check()
        print(f"PASS {check.__name__}")
    return len(checks)


if __name__ == "__main__":
    groups = asyncio.run(main())
    check_cli()
    print("PASS check_cli")
    print(json.dumps({"groups": groups + 1, "session_runs": len(RUNS), "single_run_checks": 1, "check_log_dir": str(CHECK_DIR)}, ensure_ascii=False))
