"""Synthetic subprocess scenarios for published snapshots and interrupted Runs."""

import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from benchmarks.checks.support.runtime import LocalClient, SETTINGS, TOOL, call, message, response
from benchmarks.tests.test_prefix_compaction import _Client
from lineage_agent.agent import runtime
from lineage_agent.agent.context import ContextRecords
from lineage_agent.agent.record_codec import encode_record
from lineage_agent.agent.records import RecordStore
from lineage_agent.agent.sessions import RecoveryChoice, SessionError, SessionManager
from lineage_agent.artifacts.store import ArtifactError


def emit(value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def digest(value):
    return hashlib.sha256(encode_record({"value": value})).hexdigest()


def settings(root):
    return replace(SETTINGS, application_root=root, log_dir=root / "logs",
                   context_budget_tokens=24000, max_output_tokens=256,
                   context_safety_margin_tokens=64, max_model_calls=25,
                   summary_history_max_output_tokens=512, summary_run_max_output_tokens=256)


async def synthetic_tool(name, arguments):
    assert name == TOOL["name"]
    return "original-detail=37; " + arguments["key"] + "x" * 3200


def replies(prefix):
    return [response(call(f"{prefix}{i}", arguments=json.dumps({"key": f"{prefix}{i}"})))
            for i in range(10)] + [response(message("completed"), usage=None)]


async def run(manager, sid, text, client):
    with patch.object(runtime, "create_client", return_value=client):
        turn = await manager.run_turn(sid, text)
    assert turn.snapshot_status == "published", (turn.checkpoint_error, turn.state.get("error_code"))
    assert turn.checkpoint_complete and turn.state["stop_reason"] == "completed"
    return turn


async def create(root):
    # Deliberately bypass context managers: process exit must preserve publication
    # without relying on a manager's orderly close or a final shutdown flush.
    manager = SessionManager()
    sid = manager.create_session(settings(root), tools=[TOOL], execute_tool=synthetic_tool)
    artifact = manager.artifacts(sid).put("persistent application artifact", tool_name="application")
    turns = []
    for prefix in ("a", "b"):
        turn = await run(manager, sid, "retain original requirement 17 " + prefix, _Client(replies(prefix)))
        assert turn.state["summary_calls"] > 0
        turns.append(turn)
    archive = manager._checkpointer.archive(sid)
    view = archive.view(turn.state["resource_view_id"])
    records = RecordStore(manager._checkpointer, sid)
    cursor_data = None
    for ref in view["history_scopes"]:
        page = await records.search(ref, ["x"])
        if page["next_cursor"]:
            cursor_data = {"history_ref": ref, "cursor": page["next_cursor"],
                           "next_digest": digest(await records.search(ref, ["x"], page["next_cursor"]))}
            break
    assert cursor_data is not None
    source_ref = f"history://record/{turns[0].state['run_id']}/0"
    payload = {
        "status": "created", "conversation_id": sid, "snapshot_id": turn.snapshot_id,
        "run_ids": [item.state["run_id"] for item in turns],
        "record_digests": [digest(await manager.get_record(sid, item.state["run_id"])) for item in turns],
        "state_digest": digest(turn.state), "artifact_ref": artifact["artifact_ref"],
        "source_ref": source_ref, "source_digest": digest(await records.read(source_ref, 0)),
        "summary_digests": {identity: digest(archive.summary(identity, view["view_id"])) for identity in view["summaries"]},
        "cursor": cursor_data,
        "summary_calls": [item.state["summary_calls"] for item in turns],
    }
    emit(payload)
    os._exit(0)


async def reopen(root, payload):
    sid = payload["conversation_id"]
    directory = root / ".lineage" / sid
    database_before = (directory / "conversation.sqlite").read_bytes()
    with SessionManager() as manager:
        with patch.object(runtime, "create_client", side_effect=AssertionError("opening executed a model")):
            listed = manager.list_sessions(root)
            assert len(listed) == 1 and listed[0]["status"] == "ready"
            opened = await manager.open_session(sid, application_root=root, tools=[TOOL], execute_tool=synthetic_tool)
        assert opened.snapshot_id == payload["snapshot_id"]
        saver = manager._checkpointer
        saved = await saver.read_confirmed(sid, saver.last_confirmed(sid))
        assert digest(saved.checkpoint["channel_values"]) == payload["state_digest"]
        assert manager.artifacts(sid).read(payload["artifact_ref"], 0)["text"] == "persistent application artifact"
        archive = saver.archive(sid)
        view_id = saver.confirmed_view(sid)
        for identity, expected in payload["summary_digests"].items():
            assert digest(archive.summary(identity, view_id)) == expected
        context = ContextRecords(saver, sid)
        for span in saved.checkpoint["channel_values"]["active_context"]["summary_spans"]:
            assert await context.summary(span["reference"])
        for run_id, expected in zip(payload["run_ids"], payload["record_digests"], strict=True):
            assert digest(await manager.get_record(sid, run_id)) == expected
        records = RecordStore(saver, sid)
        assert digest(await records.read(payload["source_ref"], 0)) == payload["source_digest"]
        cursor = payload["cursor"]
        assert digest(await records.search(cursor["history_ref"], ["x"], cursor["cursor"])) == cursor["next_digest"]
        assert (directory / "conversation.sqlite").read_bytes() == database_before
        third = await run(manager, sid, "continue with original requirement 17 c", _Client(replies("c")))
        assert third.state["summary_calls"] > 0
        assert third.state["run_id"] not in payload["run_ids"]
        lineage = archive.published_runs(third.snapshot_id)
        assert [item["run_id"] for item in lineage] == [*payload["run_ids"], third.state["run_id"]]
        for run_id, expected in zip(payload["run_ids"], payload["record_digests"], strict=True):
            assert digest(await manager.get_record(sid, run_id)) == expected
        assert digest(await records.search(cursor["history_ref"], ["x"], cursor["cursor"])) == cursor["next_digest"]
        emit({"status": "continued", "conversation_id": sid, "snapshots": len(lineage),
              "summary_calls": third.state["summary_calls"], "old_records": len(payload["run_ids"]),
              "cursor_preserved": True})


async def interrupt(root):
    manager = SessionManager()
    executions = []

    async def tool(name, arguments):
        executions.append(arguments["key"])
        return "interrupted-only-observation " + "x" * 15000

    sid = manager.create_session(settings(root), tools=[TOOL], execute_tool=tool)
    first = await run(manager, sid, "published seed", LocalClient([response(message("published answer"))]))

    class AwaitingSecondResponse(LocalClient):
        async def create(self, **request):
            if self.requests:
                saver = manager._checkpointer
                archive = saver.archive(sid)
                view = archive.view(saver.confirmed_view(sid))
                unfinished = archive.unfinished_runs()
                assert len(unfinished) == 1 and len(executions) == 1
                run_id = unfinished[0]["run_id"]
                assert run_id in view["record_files"] and view["artifacts"]
                events = archive.events(run_id)
                model = [event for event in events if event["kind"] == "model_result"]
                assert len(model) == 1 and model[0]["payload"]["result"]["usage"]["total_tokens"] == 12
                emit({"status": "waiting_for_second_response", "conversation_id": sid,
                      "snapshot_id": first.snapshot_id, "published_run": first.state["run_id"],
                      "unfinished_run": run_id, "artifact_refs": view["artifacts"],
                      "event_digest": digest(events), "known_tokens": 12,
                      "history_sha256": hashlib.sha256((archive.directory / "history" / f"{run_id}.jsonl").read_bytes()).hexdigest()})
                # A parent-controlled handshake marks the exact interruption
                # boundary; no timer or guessed tool/model timing is involved.
                command = await asyncio.to_thread(sys.stdin.readline)
                raise AssertionError("parent must terminate this waiting worker: " + command)
            return await super().create(**request)

    with patch.object(runtime, "create_client", return_value=AwaitingSecondResponse([response(call("interrupted"))])):
        await manager.run_turn(sid, "unfinished new input")
    raise AssertionError("the interrupted worker must not finish its Run")


async def probe(root, payload):
    with SessionManager() as manager:
        try:
            await manager.open_session(payload["conversation_id"], application_root=root,
                                       tools=[TOOL], execute_tool=synthetic_tool)
        except SessionError as error:
            assert error.code == "conversation_busy", error.code
            emit({"status": "busy", "error_code": error.code})
        else:
            raise AssertionError("two processes acquired the same conversation")


async def recover_interrupted(root, payload):
    sid, run_id = payload["conversation_id"], payload["unfinished_run"]
    directory = root / ".lineage" / sid
    history = directory / "history" / f"{run_id}.jsonl"
    assert hashlib.sha256(history.read_bytes()).hexdigest() == payload["history_sha256"]
    executions = []

    async def compatible_tool(name, arguments):
        executions.append(arguments["key"])
        return "new invocation"

    with SessionManager() as manager:
        with patch.object(runtime, "create_client", side_effect=AssertionError("opening executed model")):
            try:
                await manager.open_session(sid, application_root=root, tools=[TOOL], execute_tool=compatible_tool)
            except SessionError as error:
                assert error.code == "recovery_required", error.code
                assert error.details == {"snapshot_id": payload["snapshot_id"], "unfinished_run_ids": [run_id]}
            else:
                raise AssertionError("unfinished Run was silently resumed")
            assert not manager._sessions and not executions
            await manager.open_session(sid, application_root=root, tools=[TOOL], execute_tool=compatible_tool,
                                       recovery=RecoveryChoice(payload["snapshot_id"], (run_id,)))
        saver = manager._checkpointer
        archive = saver.archive(sid)
        view = archive.view(saver.confirmed_view(sid))
        assert run_id not in view["record_files"] and not view["artifacts"]
        assert digest(archive.events(run_id)) == payload["event_digest"]
        assert archive.recovery_choices()[-1]["unfinished_run_ids"] == [run_id]
        try:
            await manager.get_record(sid, run_id)
        except SessionError as error:
            assert error.code == "unknown_record"
        else:
            raise AssertionError("unfinished Run became queryable")
        for ref in payload["artifact_refs"]:
            try:
                manager.artifacts(sid).read(ref, 0)
            except ArtifactError as error:
                assert error.code == "artifact_not_found"
            else:
                raise AssertionError("interrupted Artifact leaked into restored view")
            assert (directory / archive.artifact(ref)["path"]).exists()
        client = LocalClient([response(message("new turn after explicit choice"))])
        next_turn = await run(manager, sid, "a genuinely new request", client)
        assert next_turn.state["run_id"] not in (run_id, payload["published_run"])
        assert not executions and len(client.requests) == 1
        assert "unfinished new input" not in str(client.requests)
        assert "interrupted-only-observation" not in str(client.requests)
        assert run_id not in next_turn.state["record_files"]
        assert hashlib.sha256(history.read_bytes()).hexdigest() == payload["history_sha256"]
        assert digest(archive.events(run_id)) == payload["event_digest"]
        assert not archive.unfinished_runs()
        emit({"status": "recovered_by_choice", "no_reexecution": True,
              "known_tokens": payload["known_tokens"], "interrupted_resources_preserved": True,
              "snapshots": len(archive.published_runs(next_turn.snapshot_id))})


async def main():
    mode, root = sys.argv[1], Path(sys.argv[2])
    if mode == "create":
        await create(root)
    elif mode == "interrupt":
        await interrupt(root)
    else:
        payload = json.loads(sys.stdin.readline())
        await {"reopen": reopen, "probe": probe, "recover-interrupted": recover_interrupted}[mode](root, payload)


if __name__ == "__main__":
    asyncio.run(main())
