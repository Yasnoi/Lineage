"""Published conversation reopening, isolation and failure boundaries."""

import asyncio
from copy import deepcopy
from contextlib import closing
from dataclasses import replace
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from langgraph.errors import NodeCancelledError

from benchmarks.checks.support.runtime import LocalClient, SETTINGS, TOOL, call, message, response
from lineage_agent.agent import runtime
from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.records import RecordStore
from lineage_agent.agent.recovery import validate_snapshot_state
from lineage_agent.agent.sessions import RecoveryChoice, SessionError, SessionManager
from lineage_agent.archive import ArchiveError, ArchiveStore
from lineage_agent.artifacts.store import ArtifactError, ArtifactStore
from lineage_agent.config.settings import ConfigurationError
from lineage_agent.recording.tracing import RunTrace


class SessionRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="session-recovery-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = replace(SETTINGS, application_root=self.root, log_dir=self.root / "logs")
        tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"})
        tracing.start()
        self.addCleanup(tracing.stop)
        asyncio.get_event_loop().slow_callback_duration = 10

    async def turn(self, manager, sid, text="request", replies=None):
        client = LocalClient(replies or [response(message("answer"))])
        with patch.object(runtime, "create_client", return_value=client):
            result = await manager.run_turn(sid, text)
        return result, client

    def body_bytes(self, sid):
        directory = self.root / ".lineage" / sid
        return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*")
                if path.is_file() and path.name != "conversation.lock"}

    async def test_three_immutable_snapshots_open_has_no_execution_and_continue(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings)
            artifact = manager.artifacts(sid).put("original artifact", tool_name="application")
            snapshots, records = [], []
            for n in range(3):
                turn, _ = await self.turn(manager, sid, f"turn {n}")
                self.assertEqual(turn.snapshot_status, "published", turn.checkpoint_error)
                snapshots.append(manager._checkpointer.archive(sid).read_snapshot(turn.snapshot_id))
                records.append(await manager.get_record(sid, turn.state["run_id"]))
            archive = manager._checkpointer.archive(sid)
            self.assertEqual([item["sequence"] for item in archive.published_runs(turn.snapshot_id)], [1, 2, 3])
            self.assertEqual([archive.read_snapshot(s["snapshot_id"]) for s in snapshots], snapshots)
        before = self.body_bytes(sid)
        with SessionManager() as manager, patch.object(runtime, "create_client") as create:
            opened = await manager.open_session(sid, application_root=self.root)
            create.assert_not_called()
            self.assertEqual(opened.snapshot_id, turn.snapshot_id)
            self.assertEqual(self.body_bytes(sid), before)
            self.assertEqual(manager.artifacts(sid).read(artifact["artifact_ref"], 0)["text"], "original artifact")
            self.assertEqual([await manager.get_record(sid, r["run_id"]) for r in records], records)
            restored = await manager._checkpointer.read_confirmed(sid, manager._sessions[sid].last_checkpoint_id)
            self.assertEqual(restored.checkpoint["channel_values"]["input"], turn.state["input"])
            self.assertNotEqual(manager._sessions[sid].last_checkpoint_id, turn.checkpoint_id)
            next_turn, client = await self.turn(manager, sid, "next")
            self.assertEqual(next_turn.snapshot_status, "published", next_turn.checkpoint_error)
            self.assertEqual(client.requests[0]["input"][:-1], turn.state["input"][:turn.state["history_end"]])
            self.assertNotEqual(next_turn.state["run_id"], turn.state["run_id"])
            self.assertEqual(next_turn.state["conversation_id"], sid)
        self.assertEqual(SessionManager.list_sessions(self.root)[0]["status"], "ready")

    async def test_unfinished_run_is_preserved_and_excluded_after_explicit_choice(self):
        async def tool(name, arguments):
            return "abandoned-observation " + "x" * 15000
        with SessionManager() as manager:
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=tool)
            first, _ = await self.turn(manager, sid)
            class Interrupted(LocalClient):
                async def create(self, **kwargs):
                    if self.requests:
                        raise asyncio.CancelledError()
                    return await super().create(**kwargs)
            with patch.object(runtime, "create_client", return_value=Interrupted([response(call())])):
                with self.assertRaises((asyncio.CancelledError, NodeCancelledError)):
                    await manager.run_turn(sid, "interrupted input")
            archive = manager._checkpointer.archive(sid)
            unfinished = archive.unfinished_runs()
            interrupted_id = unfinished[0]["run_id"]
            events = archive.events(interrupted_id)
            self.assertTrue(any(event["kind"] == "execution_facts" for event in events))
            with self.assertRaises(SessionError) as blocked:
                await manager.run_turn(sid, "cannot clear blocked")
            self.assertEqual(blocked.exception.code, "session_blocked")
        interrupted_file = self.root / ".lineage" / sid / "history" / f"{interrupted_id}.jsonl"
        preserved = interrupted_file.read_bytes()
        with SessionManager() as manager:
            with self.assertRaises(SessionError) as required:
                await manager.open_session(sid, application_root=self.root, tools=[TOOL], execute_tool=tool)
            self.assertEqual(required.exception.code, "recovery_required")
            self.assertFalse(manager._sessions)
            with self.assertRaises(SessionError) as stale:
                await manager.open_session(sid, application_root=self.root, tools=[TOOL], execute_tool=tool,
                                           recovery=RecoveryChoice(first.snapshot_id, ("different-run",)))
            self.assertEqual(stale.exception.code, "recovery_changed")
            choice = RecoveryChoice(first.snapshot_id, (interrupted_id,))
            await manager.open_session(sid, application_root=self.root, tools=[TOOL], execute_tool=tool, recovery=choice)
            saver = manager._checkpointer
            view = saver.archive(sid).view(saver.confirmed_view(sid))
            self.assertNotIn(interrupted_id, view["record_files"])
            self.assertFalse(view["artifacts"])
            with self.assertRaises(SessionError):
                await manager.get_record(sid, interrupted_id)
            resumed, client = await self.turn(manager, sid, "new input")
            self.assertEqual(resumed.snapshot_status, "published", resumed.checkpoint_error)
            self.assertNotIn("abandoned-observation", str(client.requests))
            self.assertNotIn(interrupted_id, resumed.state["record_files"])
            self.assertEqual(interrupted_file.read_bytes(), preserved)
            self.assertEqual(saver.archive(sid).events(interrupted_id), events)
            self.assertFalse(saver.archive(sid).unfinished_runs())
        with SessionManager() as manager:
            await manager.open_session(sid, application_root=self.root, tools=[TOOL], execute_tool=tool)

    async def test_failed_publish_keeps_old_head_and_reopen_requires_choice(self):
        for failure in ("begin_run", "publish_snapshot"):
            with self.subTest(failure=failure), SessionManager() as manager:
                sid = manager.create_session(self.settings)
                first, _ = await self.turn(manager, sid)
                with patch.object(ArchiveStore, failure, side_effect=ArchiveError("archive_write_error")):
                    failed, client = await self.turn(manager, sid, "will not publish")
                self.assertFalse(failed.checkpoint_complete)
                self.assertEqual(failed.snapshot_status, "not_published")
                self.assertEqual(manager._checkpointer.archive(sid).conversation()["current_snapshot_id"], first.snapshot_id)
                self.assertEqual(len(client.requests), 0 if failure == "begin_run" else 1)
                with self.assertRaises(SessionError):
                    await manager.run_turn(sid, "blocked")
                manager.close_session(sid)
                if failure == "publish_snapshot":
                    with self.assertRaises(SessionError) as required:
                        await manager.open_session(sid, application_root=self.root)
                    self.assertEqual(required.exception.code, "recovery_required")
                else:
                    await manager.open_session(sid, application_root=self.root)

    async def test_error_and_call_limit_states_reopen_without_admitting_tail(self):
        for replies, settings in (([RuntimeError("synthetic")], self.settings),
                                   ([response(call())], replace(self.settings, max_model_calls=1))):
            with self.subTest(replies=str(replies)), SessionManager() as manager:
                async def tool(*_):
                    return "tool fact"
                sid = manager.create_session(settings, tools=[TOOL], execute_tool=tool)
                ended, _ = await self.turn(manager, sid, replies=replies)
                self.assertEqual(ended.snapshot_status, "published", ended.checkpoint_error)
                expected = ended.state["input"][:ended.state["history_end"]]
                manager.close_session(sid)
                await manager.open_session(sid, application_root=self.root, tools=[TOOL], execute_tool=tool)
                next_turn, client = await self.turn(manager, sid, "next")
                self.assertEqual(next_turn.snapshot_status, "published", next_turn.checkpoint_error)
                self.assertEqual(client.requests[0]["input"][:-1], expected)

    async def test_no_snapshot_unknown_directory_and_preflight_rejection_do_not_fake_restore(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings)
            with patch.object(runtime, "create_client", side_effect=ConfigurationError("missing credential")):
                with self.assertRaises(runtime.PreExecutionConfigurationError):
                    await manager.run_turn(sid, "request")
            self.assertFalse(manager._checkpointer.archive(sid).unfinished_runs())
            self.assertFalse(manager._sessions[sid].blocked)
            self.assertEqual(SessionManager.list_sessions(self.root)[0]["status"], "no_snapshot")
            manager.close_session(sid)
            with self.assertRaises(SessionError) as empty:
                await manager.open_session(sid, application_root=self.root)
            self.assertEqual(empty.exception.code, "no_snapshot")
            with self.assertRaises(SessionError):
                await manager.open_session("missing", application_root=self.root)
            self.assertFalse((self.root / ".lineage" / "missing").exists())

    async def test_missing_or_corrupted_resources_fail_without_partial_registration(self):
        for kind in ("history", "artifact", "version", "old_version_without_lock"):
            with self.subTest(kind=kind), SessionManager() as manager:
                sid = manager.create_session(self.settings)
                artifact = manager.artifacts(sid).put("must stay intact", tool_name="application")
                turn, _ = await self.turn(manager, sid)
                manager.artifacts(sid).read(artifact["artifact_ref"], 0)
                manager.close_session(sid)
                directory = self.root / ".lineage" / sid
                if kind == "history":
                    (directory / "history" / f'{turn.state["run_id"]}.jsonl').write_bytes(b"truncated")
                elif kind == "artifact":
                    next((directory / "artifacts").glob("*.raw")).unlink()
                else:
                    with closing(sqlite3.connect(directory / "conversation.sqlite")) as connection:
                        connection.execute("PRAGMA user_version=" + ("1" if kind == "old_version_without_lock" else "999"))
                        connection.commit()
                    if kind == "old_version_without_lock":
                        (directory / "conversation.lock").unlink()
                for _ in range(2):
                    with self.assertRaises(SessionError) as bad:
                        await manager.open_session(sid, application_root=self.root)
                    self.assertNotEqual(bad.exception.code, "conversation_busy")
                    if "version" in kind:
                        self.assertEqual(bad.exception.code, "archive_version_error")
                    self.assertNotIn(sid, manager._sessions)

    async def test_open_failure_releases_lease_and_borrowed_store_and_stale_writers(self):
        with ArtifactStore() as store, SessionManager() as manager:
            sid = manager.create_session(self.settings, artifact_store=store)
            old = manager.artifacts(sid)
            artifact = old.put("old readable body", tool_name="application")
            await self.turn(manager, sid)
            manager.close_session(sid)
            with self.assertRaises(ArtifactError):
                old.put("must not write", tool_name="application")
            with patch.object(SessionCheckpointer, "restore_state", side_effect=CheckpointError("checkpoint_write_error")):
                with self.assertRaises(SessionError):
                    await manager.open_session(sid, application_root=self.root, artifact_store=store)
            self.assertFalse(store._closed)
            await manager.open_session(sid, application_root=self.root, artifact_store=store)
            self.assertEqual(old.read(artifact["artifact_ref"], 0)["text"], "old readable body")
            with self.assertRaises(ArtifactError):
                old.put("must not regain authority", tool_name="application")

    async def test_turn_ownership_and_record_boundaries_cannot_be_reassigned(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings)
            await self.turn(manager, sid, "first")
            turn, _ = await self.turn(manager, sid, "second")
            archive = manager._checkpointer.archive(sid)
            swapped = deepcopy(turn.state)
            turns = swapped["active_context"]["turns"]
            turns[0]["run_id"], turns[1]["run_id"] = turns[1]["run_id"], turns[0]["run_id"]
            truncated = deepcopy(turn.state)
            truncated["active_context"]["turns"][0]["record_end"] = 1
            for changed in (swapped, truncated):
                with self.assertRaises(ArchiveError):
                    validate_snapshot_state(changed, archive, turn.state["resource_view_id"])

    async def test_postpublication_trace_failure_does_not_undo_published_turn(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings)
            with patch.object(RunTrace, "close", side_effect=OSError("private trace cleanup")), \
                    self.assertLogs("lineage_agent.agent.runtime", level="WARNING") as warnings:
                turn, _ = await self.turn(manager, sid)
            self.assertEqual(turn.snapshot_status, "published")
            self.assertTrue(turn.checkpoint_complete)
            self.assertNotIn("private trace cleanup", str(warnings.output))
            self.assertEqual((await manager.get_record(sid, turn.state["run_id"]))["stop_reason"], "completed")
            manager.close_session(sid)
            opened = await manager.open_session(sid, application_root=self.root)
            self.assertEqual(opened.snapshot_id, turn.snapshot_id)

    async def test_unknown_ack_is_blocked_but_new_process_observes_committed_publication(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings)
            first, _ = await self.turn(manager, sid)
            native = ArchiveStore.publish_snapshot
            def commit_without_ack(archive, *args, **kwargs):
                native(archive, *args, **kwargs)
                raise ArchiveError("archive_publication_unknown")
            with patch.object(ArchiveStore, "publish_snapshot", new=commit_without_ack):
                turn, client = await self.turn(manager, sid, "published but acknowledgement lost")
            self.assertEqual(turn.snapshot_status, "unknown")
            self.assertFalse(turn.checkpoint_complete)
            self.assertEqual(len(client.requests), 1)
            with self.assertRaises(SessionError) as blocked:
                await manager.run_turn(sid, "do not replay")
            self.assertEqual(blocked.exception.code, "session_blocked")
            manager.close_session(sid)
            opened = await manager.open_session(sid, application_root=self.root)
            self.assertNotEqual(opened.snapshot_id, first.snapshot_id)
            self.assertEqual((await manager.get_record(sid, turn.state["run_id"]))["model_calls"], 1)


if __name__ == "__main__":
    unittest.main()
