"""recording / sessions behavior contracts."""


import asyncio
import json
from benchmarks.checks.support.events import check_group, observe_turn
from benchmarks.checks.support.runtime import LocalClient, call, response
from benchmarks.checks.support.session_traces import BODY, CHECK_DIR, SETTINGS, run_case, run_sessions
from benchmarks.checks.support.tracing import capture
from copy import deepcopy
from dataclasses import replace
from lineage_agent.agent import runtime
from lineage_agent.agent.checkpointing import CHECKPOINT_ERRORS, CheckpointError
from lineage_agent.agent.sessions import SessionManager
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

async def main():
    with check_group("recording.sessions.checkpoint_trace_equivalence"):
        baseline = await run_case(enabled=False)
        assert await run_case() == baseline
        assert await run_case(mode="connection") == baseline
    with check_group("recording.sessions.checkpoint_error_metadata"):
        for code in CHECKPOINT_ERRORS:
            assert await run_case(checkpoint_error=CheckpointError(code)) == baseline
    with TemporaryDirectory(prefix="lineage-session-trace-") as directory:
        root = Path(directory)
        source = BODY + "\nfact=azure-coral-58"
        (root / "source.txt").write_bytes(source.encode("utf-8"))
        with check_group("recording.sessions.session_history_trace_equivalence"):
            baseline = await run_sessions(root, source, enabled=False)
            assert await run_sessions(root, source) == baseline
            assert await run_sessions(root, source, mode="connection") == baseline
        with check_group("recording.sessions.session_final_save_failure"):
            failed_states, failed_requests = await run_sessions(root, source, final_save_failure=True)
            saved_states, saved_requests = baseline
            assert failed_requests == saved_requests and failed_states[0] == saved_states[0]
            failed, saved = failed_states[1], saved_states[1]
            # Business completion remains public, but ownership changes only
            # after the final save is confirmed. Compare every other State field.
            assert {key: value for key, value in failed.items() if key not in {"active_context", "execution"}} == {
                key: value for key, value in saved.items() if key not in {"active_context", "execution"}}
            assert "final_record_id" not in failed["execution"]
            assert saved["execution"]["final_record_id"] == "<execution-final_record_id>"
            assert failed["execution"] == {key: value for key, value in saved["execution"].items()
                                           if key != "final_record_id"}
            current, historical = failed["active_context"], saved["active_context"]
            transitioned = {"current_run_id", "turn_start", "initial_input_count", "groups", "turns"}
            assert {key: value for key, value in current.items() if key not in transitioned} == {
                key: value for key, value in historical.items() if key not in transitioned}
            assert current["current_run_id"] == "<current-run>" and historical["current_run_id"] is None
            assert current["turns"] == historical["turns"][:-1]
            assert current["turn_start"] == saved_states[0]["history_end"]
            assert current["initial_input_count"] == 1 and historical["initial_input_count"] == 0
            assert historical["turn_start"] == saved["history_end"] and historical["groups"] == []
            assert len(current["groups"]) == 1 and current["groups"][0]["closed"] is True
            assert current["groups"][0]["start"] == current["turn_start"] + 1
            assert current["groups"][0]["end"] == saved["history_end"]
            assert historical["turns"][-1] == {
                "run_id": "<current-run>", "item_count": saved["history_end"] - current["turn_start"],
                "group_count": 1, "record_end": failed["record_sequence"],
            }
        with check_group("recording.sessions.skipped_after_save_failure"):
            await check_skipped_after_save_failure(root)
        with check_group("recording.sessions.fixture_integrity"):
            assert (root / "source.txt").read_text(encoding="utf-8") == source
    print(f"PASS 6 session tracing groups (6 checkpoint harness runs, 9 Agent turns); logs={CHECK_DIR}")

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
    assert state["tool_outcomes"] == {"succeeded": 0, "rejected": 0, "failed": 0, "skipped": 0}
    assert state["usage_totals"]["total_tokens"] == 12
    spans = transport.sent
    assert not any(span["run_type"] == "tool" for span in spans)
    archive = manager._checkpointer.archive(session_id)
    point = archive.read_execution(state["run_id"])
    assert point["phase"] == "response_saved"
    attempts = archive.attempts(point["state"]["execution"]["operation_id"])
    assert len(attempts) == 1 and attempts[0]["result"]["raw_response"]["output"][0]["call_id"] == "not-executed"
    assert not any(operation["kind"] == "tool" for operation in archive.operations(state["run_id"]))
    trace_root = next(span for span in spans if span["run_type"] == "chain")
    assert trace_root["extra"]["metadata"]["outcome"] == "failed"
    assert trace_root["error"] == "checkpoint_write_error"
    events = [json.loads(line) for line in Path(state["log_path"]).read_text(encoding="utf-8").splitlines()]
    assert not any(event["event"] in {"tool.started", "tool.skipped"} for event in events)
    assert not any(event["event"] == "tool.result_created" for event in events)
    assert "PRIVATE_AFTER_MODEL_SAVE_FAULT" not in json.dumps([events, spans])
    print(
        f"PASS session checkpoint failure retains pending tool session_id={session_id} run_id={state['run_id']} "
        f"checkpoint_id={turn.checkpoint_id} last_confirmed={manager._checkpointer.last_confirmed(session_id)} "
        f"execution_point={point['point_id']} call_id=not-executed log={state['log_path']}"
    )

def execute():
    asyncio.run(main())
