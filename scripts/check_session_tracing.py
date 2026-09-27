"""Offline session association checks using native checkpoints and SDK uploads."""

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict
from unittest.mock import patch
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from check_react import LocalClient, call, message, response
from check_tracing import capture, stable
from check_support import check_group, observe_harness, observe_turn
from lineage_agent.agent.checkpointing import CHECKPOINT_ERRORS, CheckpointError, SessionCheckpointer
from lineage_agent.agent import runtime
from lineage_agent.agent.sessions import SessionManager
from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.responses import generate_response
from lineage_agent.recording.recorder import RunRecorder
from lineage_agent.recording.tracing import start_trace
from lineage_agent.tools.filesystem.text import TEXT_TOOL
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL
from lineage_agent.tools.filesystem.executor import create_filesystem_executor


CHECK_DIR = Path(__file__).resolve().parents[1] / "logs" / "checks" / str(uuid4())
SETTINGS = ModelSettings(base_url="https://unused.invalid/v1", model_name="session-trace-check", log_dir=CHECK_DIR)
BODY = "SESSION_TOOL_BODY_PRIVATE_762c"
PRIVATE_STATE = "CHECKPOINT_ONLY_PRIVATE_459e"


class SavedState(TypedDict):
    input: list[dict]
    private: str


async def run_case(*, enabled=True, mode="ok", checkpoint_error=None):
    session_id = str(uuid4())
    recorder = RunRecorder(CHECK_DIR)
    saver = SessionCheckpointer()
    saver.bind_run(session_id, recorder)
    config = {"configurable": {"thread_id": session_id}}
    request = call("session-read", name="read_text_file", arguments='{"path":"source.txt"}')
    body = {"path": "source.txt", "content": BODY, "encoding": "utf-8", "bytes_read": len(BODY), "complete": True}
    envelope = {"ok": True, "result": json.dumps(body)}
    output = {"type": "function_call_output", "call_id": request["call_id"], "output": json.dumps(envelope)}
    items = [{"role": "user", "content": "Use the saved source."}, request, output]
    original = deepcopy(items)
    source_state = {"input": items, "private": PRIVATE_STATE}
    graph = StateGraph(SavedState)
    graph.add_node("keep", lambda state: {})
    graph.add_edge(START, "keep")
    graph.add_edge("keep", END)
    compiled = graph.compile(checkpointer=saver)
    with capture(mode, LANGSMITH_TRACING="true" if enabled else "false") as (transport, stderr):
        tracer = start_trace(recorder.run_id, SETTINGS)
        tracer.bind_session(session_id)
        recorder.emit("run.started")
        with tracing_context(enabled=False, parent=False):
            await compiled.ainvoke(source_state, config, durability="sync")
            saved = await compiled.aget_state(config)
        checkpoint_id = saver.last_confirmed(session_id)
        observe_harness(
            "checkpoint_trace_harness", session_id=session_id, run_id=recorder.run_id,
            checkpoint_id=checkpoint_id, log_path=str(recorder.path),
        )
        assert saved.values == source_state and checkpoint_id
        step_id = recorder.next_step()
        token = tracer.tool_start(request, step_id=step_id, request_index=1)
        tracer.tool_end(token, output=envelope, outcome="succeeded", stage="execution")
        tracer.set_model_step(recorder.next_step(), 1)
        async with LocalClient([response(message("The saved source was used."))]) as client:
            result = await generate_response(client, SETTINGS.model_name, items, tools=[TEXT_TOOL], tracer=tracer)
        tracer.model_complete()
        state = {
            **source_state, "result": result, "model_calls": 1, "tool_calls": 1,
            "tool_requests": 1, "tool_outcomes": {"succeeded": 1, "rejected": 0, "failed": 0, "skipped": 0},
            "usage_totals": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12, "missing_usage_calls": 0},
            "stop_reason": "error" if checkpoint_error else "completed",
            "error_code": checkpoint_error.code if checkpoint_error else None,
        }
        state_before = deepcopy(state)
        tracer.finish_checkpoint(checkpoint_id, checkpoint_error)
        tracer.finish(state)
        await tracer.close()
        assert state == state_before and items == original and client.closed
        assert client.requests[0]["input"] == original
        assert BODY in json.dumps(client.requests)
        recorder.emit("run.ended", checkpoint_id=checkpoint_id)
        saver.unbind_run(session_id)
        assert recorder.close()
        assert bool(stderr.getvalue()) == (enabled and mode != "ok"), stderr.getvalue()
    events = [json.loads(line) for line in recorder.path.read_text(encoding="utf-8").splitlines()]
    assert all(event["session_id"] == session_id and event["run_id"] == recorder.run_id for event in events)
    assert any(event["event"] == "checkpoint.saved" and event["checkpoint_id"] == checkpoint_id for event in events)
    assert PRIVATE_STATE not in json.dumps(events) and BODY not in json.dumps(events)
    spans = transport.sent
    if enabled:
        assert {span["run_type"] for span in spans} == {"chain", "tool", "llm"}
        assert all(span["extra"]["metadata"]["session_id"] == session_id for span in spans)
        root = next(span for span in spans if span["run_type"] == "chain")
        metadata = root["extra"]["metadata"]
        assert metadata["checkpoint_id"] == checkpoint_id
        assert metadata["checkpoint_saved"] is (checkpoint_error is None)
        assert metadata["checkpoint_error"] == (checkpoint_error.code if checkpoint_error else None)
        if checkpoint_error:
            assert root["error"] == checkpoint_error.code
        wire = json.dumps(spans)
        assert BODY not in wire and PRIVATE_STATE not in wire
        assert "content_omitted" in wire and "source.txt" in wire
        assert "The saved source was used." in wire
    else:
        assert not spans
    print(
        f"PASS session tracing enabled={enabled} upload={mode} error={checkpoint_error.code if checkpoint_error else 'none'} "
        f"session_id={session_id} run_id={recorder.run_id} checkpoint_id={checkpoint_id} "
        f"step_id=step-0001,step-0002 call_id=session-read log={recorder.path}"
    )
    return client.requests, result


async def main():
    with check_group("check_session_tracing/checkpoint_trace_equivalence"):
        baseline = await run_case(enabled=False)
        assert await run_case() == baseline
        assert await run_case(mode="connection") == baseline
    with check_group("check_session_tracing/checkpoint_error_metadata"):
        for code in CHECKPOINT_ERRORS:
            assert await run_case(checkpoint_error=CheckpointError(code)) == baseline
    with TemporaryDirectory(prefix="lineage-session-trace-") as directory:
        root = Path(directory)
        source = BODY + "\nfact=azure-coral-58"
        (root / "source.txt").write_bytes(source.encode("utf-8"))
        with check_group("check_session_tracing/session_history_trace_equivalence"):
            baseline = await run_sessions(root, source, enabled=False)
            assert await run_sessions(root, source) == baseline
            assert await run_sessions(root, source, mode="connection") == baseline
        with check_group("check_session_tracing/session_final_save_failure"):
            assert await run_sessions(root, source, final_save_failure=True) == baseline
        with check_group("check_session_tracing/check_skipped_after_save_failure"):
            await check_skipped_after_save_failure(root)
        with check_group("check_session_tracing/fixture_integrity"):
            assert (root / "source.txt").read_text(encoding="utf-8") == source
    print(f"PASS 6 session tracing groups (6 checkpoint harness runs, 9 Agent turns); logs={CHECK_DIR}")


async def run_sessions(root, source, *, enabled=True, mode="ok", final_save_failure=False):
    """Exercise the actual session/runtime wiring, with full-State tracing blocked."""
    settings = replace(SETTINGS, project_root=root)
    manager = SessionManager()
    session_id = manager.create_session(
        settings, tools=[DIRECTORY_TOOL, TEXT_TOOL], execute_tool=create_filesystem_executor(root),
    )
    first_reply = response(message("fact=azure-coral-58, source.txt"))
    first_reply.metadata = {"checkpoint_private": PRIVATE_STATE}
    clients = [
        LocalClient([
            response(call("directory", name="list_directory", arguments='{"path":".","after":null}')),
            response(call("text", name="read_text_file", arguments='{"path":"source.txt"}')),
            first_reply,
        ]),
        LocalClient([response(message("The earlier fact was azure-coral-58, from source.txt."))]),
    ]
    original_put = manager._checkpointer.put
    finalizations = 0

    def put(config, checkpoint, metadata, versions):
        nonlocal finalizations
        if metadata.get("source") == "update":
            finalizations += 1
            if final_save_failure and finalizations == 2:
                raise OSError("PRIVATE_FINAL_SAVE_FAULT")
        return original_put(config, checkpoint, metadata, versions)

    with capture(mode, LANGSMITH_TRACING="true" if enabled else "false") as (transport, stderr):
        with patch.object(runtime, "create_client", side_effect=clients), patch.object(manager._checkpointer, "put", side_effect=put):
            turns = [await manager.run_turn(session_id, "Find the text source and report its fact.")]
            observe_turn(turns[0])
            with tracing_context(enabled=False, parent=False):
                saved = await manager._checkpointer.aget_tuple({"configurable": {
                    "thread_id": session_id, "checkpoint_id": turns[0].checkpoint_id,
                }})
            assert PRIVATE_STATE in json.dumps(saved.checkpoint["channel_values"])
            assert BODY in json.dumps(saved.checkpoint["channel_values"])
            turns.append(await manager.run_turn(session_id, "What was that earlier fact, and its source?"))
            observe_turn(turns[1])
        assert bool(stderr.getvalue()) == (enabled and mode != "ok"), stderr.getvalue()
    assert all(client.closed for client in clients)
    assert len(clients[0].requests) == 3 and len(clients[1].requests) == 1
    outputs = [item for item in clients[1].requests[0]["input"] if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in outputs] == ["directory", "text"]
    assert json.loads(json.loads(outputs[1]["output"])["result"])["content"] == source
    assert outputs == [item for item in clients[0].requests[-1]["input"] if item.get("type") == "function_call_output"]
    assert turns[0].checkpoint_complete and turns[0].checkpoint_error is None
    assert turns[1].checkpoint_complete is (not final_save_failure)
    assert turns[1].checkpoint_error == ("checkpoint_write_error" if final_save_failure else None)
    spans = transport.sent
    if enabled:
        assert len(spans) == 8
        assert all(span["extra"]["metadata"]["session_id"] == session_id for span in spans)
        assert BODY not in json.dumps(spans) and PRIVATE_STATE not in json.dumps(spans)
        assert "azure-coral-58" in json.dumps(spans)
        for span in spans:
            if span["run_type"] == "llm":
                for item in span["inputs"].get("input", []):
                    if item.get("type") == "function_call_output" and item["call_id"] == "text":
                        filtered = json.loads(item["output"])
                        assert filtered["result"]["content_omitted"] is True
                        assert "azure-coral-58" not in json.dumps(filtered)
            if span["run_type"] == "tool" and span["name"] == "read_text_file":
                assert span["outputs"]["result"]["content_omitted"] is True
        for index, turn in enumerate(turns):
            trace_root = next(span for span in spans if span["id"] == turn.state["run_id"])
            metadata = trace_root["extra"]["metadata"]
            assert metadata["checkpoint_id"] == turn.checkpoint_id
            assert metadata["checkpoint_saved"] is turn.checkpoint_complete
            assert metadata["checkpoint_error"] == turn.checkpoint_error
            assert metadata["outcome"] == ("failed" if turn.checkpoint_error else "succeeded")
            assert trace_root["outputs"]["model_calls"] == (3 if index == 0 else 1)
            assert trace_root["outputs"]["text"] == turn.state["result"]["text"]
            if turn.checkpoint_error:
                assert trace_root["error"] == "checkpoint_write_error"
    else:
        assert not spans
    for index, turn in enumerate(turns):
        state = turn.state
        assert state["stop_reason"] == "completed" and state["log_complete"]
        assert state["model_calls"] == (3 if index == 0 else 1)
        assert state["tool_calls"] == (2 if index == 0 else 0)
        events = [json.loads(line) for line in Path(state["log_path"]).read_text(encoding="utf-8").splitlines()]
        assert all(event["session_id"] == session_id for event in events)
        starts = [event for event in events if event["event"] == "model.started"]
        assert [event["input_tool_result_call_ids"] for event in starts] == [
            [item["call_id"] for item in request["input"] if item.get("type") == "function_call_output"]
            for request in clients[index].requests
        ]
        assert BODY not in json.dumps(events) and PRIVATE_STATE not in json.dumps(events)
        print(
            f"PASS actual session tracing turn={index + 1} enabled={enabled} upload={mode} "
            f"session_id={session_id} run_id={state['run_id']} checkpoint_id={turn.checkpoint_id} "
            f"checkpoint_error={turn.checkpoint_error} steps={[event['step_id'] for event in starts]} "
            f"call_ids=directory,text log={state['log_path']}"
        )
    stable_states = [stable(turn.state) for turn in turns]
    return stable_states, [client.requests for client in clients]


async def check_skipped_after_save_failure(root):
    manager = SessionManager()
    calls = []
    executor = create_filesystem_executor(root)

    async def counted_executor(name, arguments):
        calls.append((name, deepcopy(arguments)))
        return await executor(name, arguments)

    session_id = manager.create_session(
        replace(SETTINGS, project_root=root), tools=[TEXT_TOOL], execute_tool=counted_executor,
    )
    client = LocalClient([response(call("not-executed", name="read_text_file", arguments='{"path":"source.txt"}'))])
    original_put = manager._checkpointer.put

    def put(config, checkpoint, metadata, versions):
        if metadata.get("source") == "loop" and checkpoint["channel_values"].get("model_calls") == 1:
            raise OSError("PRIVATE_AFTER_MODEL_SAVE_FAULT")
        return original_put(config, checkpoint, metadata, versions)

    with capture() as (transport, stderr), patch.object(runtime, "create_client", return_value=client), patch.object(
        manager._checkpointer, "put", side_effect=put,
    ):
        turn = await manager.run_turn(session_id, "Use the source file.")
        observe_turn(turn)
        assert not stderr.getvalue(), stderr.getvalue()
    state = turn.state
    assert client.closed and len(client.requests) == 1 and not calls
    assert not turn.checkpoint_complete and turn.checkpoint_error == "checkpoint_write_error"
    assert state["model_calls"] == 1 and state["tool_requests"] == 1 and state["tool_calls"] == 0
    assert state["tool_outcomes"] == {"succeeded": 0, "rejected": 0, "failed": 0, "skipped": 1}
    assert state["usage_totals"]["total_tokens"] == 12
    spans = transport.sent
    tool = next(span for span in spans if span["run_type"] == "tool")
    metadata = tool["extra"]["metadata"]
    assert metadata["call_id"] == "not-executed" and metadata["session_id"] == session_id
    assert metadata["outcome"] == "skipped" and metadata["skip_reason"] == "checkpoint_error"
    trace_root = next(span for span in spans if span["run_type"] == "chain")
    assert trace_root["extra"]["metadata"]["outcome"] == "failed"
    assert trace_root["error"] == "checkpoint_write_error"
    events = [json.loads(line) for line in Path(state["log_path"]).read_text(encoding="utf-8").splitlines()]
    skipped = next(event for event in events if event["event"] == "tool.skipped")
    assert skipped["call_id"] == "not-executed" and skipped["reason"] == "checkpoint_error"
    assert not any(event["event"] == "tool.result_created" for event in events)
    assert "PRIVATE_AFTER_MODEL_SAVE_FAULT" not in json.dumps([events, spans])
    print(
        f"PASS actual session checkpoint failure skips tool session_id={session_id} run_id={state['run_id']} "
        f"checkpoint_id={turn.checkpoint_id} last_confirmed={manager._checkpointer.last_confirmed(session_id)} "
        f"step_id={metadata['step_id']} call_id=not-executed log={state['log_path']}"
    )


if __name__ == "__main__":
    asyncio.run(main())
