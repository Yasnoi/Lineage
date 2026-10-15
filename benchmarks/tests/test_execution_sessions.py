"""Explicit execution inspection, idle opening, and same-Run continuation."""

import asyncio
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from langgraph.errors import NodeCancelledError

from benchmarks.checks.support.runtime import LocalClient, SETTINGS, response, message
from lineage_agent.agent import runtime
from lineage_agent.agent.checkpointing import CheckpointError
from lineage_agent.agent.sessions import SessionError, SessionManager, RunInspection, RunRecoveryChoice, RunResumed
from lineage_agent.entrypoints import cli


class ExecutionSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="execution-session-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = replace(SETTINGS, application_root=self.root, log_dir=self.root / "logs")
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING": "false",
                                                 "LANGCHAIN_TRACING_V2": "false"}))
        asyncio.get_event_loop().slow_callback_duration = 10

    async def interrupted(self):
        class Interrupted(LocalClient):
            async def create(self, **kwargs):
                raise asyncio.CancelledError()
        with SessionManager() as manager:
            sid = manager.create_session(self.settings)
            with patch.object(runtime, "create_client", return_value=Interrupted([])):
                with self.assertRaises((asyncio.CancelledError, NodeCancelledError)):
                    await manager.run_turn(sid, "keep this original input")
            archive = manager._checkpointer.archive(sid)
            run_id = archive.unfinished_runs()[0]["run_id"]
            self.assertIsNotNone(archive.read_execution(run_id))
        return sid, run_id

    async def test_first_run_inspection_idle_open_then_same_run_resume_and_repeat(self):
        sid, run_id = await self.interrupted()
        history = self.root / ".lineage" / sid / "history" / f"{run_id}.jsonl"
        before = history.read_bytes()
        with SessionManager() as manager:
            with patch.object(runtime, "create_client", side_effect=AssertionError("opening must not call model")):
                inspection = manager.inspect_run(sid, run_id, application_root=self.root)
                listed = manager.list_sessions(self.root)
                self.assertEqual(listed[0]["unfinished_runs"][0]["execution_phase"], inspection.phase)
                output = StringIO()
                with redirect_stderr(output):
                    cli._show_sessions(manager, self.root)
                self.assertIn(f"{run_id}={inspection.phase}", output.getvalue())
                self.assertIsNone(inspection.choice.snapshot_id)
                self.assertTrue(inspection.uncertain_attempts)
                self.assertEqual(inspection.quota["usage_totals"]["missing_usage_calls"], 1)
                self.assertIsNone(inspection.quota["usage_totals"]["total_tokens"])
                self.assertEqual(inspection.quota["used"].get("model_calls", 0), 0)
                self.assertEqual(inspection.quota["reserved"]["model_calls"], 1)
                self.assertEqual(inspection.quota["remaining_model_calls"], self.settings.max_model_calls - 1)
                opened = await manager.open_run(sid, run_id, choice=inspection.choice, application_root=self.root)
                with self.assertRaises(SessionError) as pending:
                    await manager.run_turn(sid, "not a resume")
                self.assertEqual(pending.exception.code, "pending_run")
                with self.assertRaises(SessionError) as record:
                    await manager.get_record(sid, run_id)
                self.assertEqual(record.exception.code, "unknown_record")
                self.assertEqual(history.read_bytes(), before)
            client = LocalClient([response(message("finished original Run"))])
            with patch.object(runtime, "create_client", return_value=client):
                result = await manager.resume_run(sid, choice=opened.choice)
                duplicate = await manager.resume_run(sid, choice=opened.choice)
            self.assertEqual(result.status, "completed", result)
            self.assertEqual(result.phase, "sealed")
            self.assertEqual(result.quota["charged"]["model_calls"], 2)
            self.assertEqual(result.quota["usage_totals"]["missing_usage_calls"], 1)
            self.assertEqual(result.quota["usage_totals"]["total_tokens"], 12)
            self.assertEqual(len(result.uncertain_attempts), 1)
            self.assertEqual(result, duplicate)
            self.assertEqual(len(client.requests), 1)
            self.assertEqual(result.turn.state["run_id"], run_id)
            self.assertEqual(result.turn.state["conversation_id"], sid)
            self.assertEqual(result.turn.state["model_calls"], 2)
            self.assertEqual((await manager.get_record(sid, run_id))["run_id"], run_id)
            self.assertEqual(history.read_bytes(), before)
            self.assertEqual(len(result.turn.state["record_files"][run_id]["segments"]), 2)
        with SessionManager() as manager:
            inspection = manager.inspect_run(sid, run_id, application_root=self.root)
            self.assertEqual(inspection.status, "completed")
            self.assertIsNone(inspection.choice)

    async def test_stale_choice_failed_open_releases_lease_and_does_not_register(self):
        sid, run_id = await self.interrupted()
        with SessionManager() as manager:
            inspection = manager.inspect_run(sid, run_id, application_root=self.root)
            with self.assertRaises(SessionError) as changed:
                await manager.open_run(sid, run_id, application_root=self.root,
                                       choice=replace(inspection.choice, revision=inspection.choice.revision + 1))
            self.assertEqual(changed.exception.code, "recovery_changed")
            self.assertFalse(manager._sessions)
            again = manager.inspect_run(sid, run_id, application_root=self.root)
            self.assertEqual(again.choice.execution_point_id, inspection.choice.execution_point_id)
            opened = await manager.open_run(sid, run_id, application_root=self.root, choice=again.choice)
            manager.close_session(opened.session_id)

    async def test_lifecycle_without_any_execution_point_cannot_open_empty_task(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings)
            manager._checkpointer.archive(sid).begin_run("no-boundary", None)
        with SessionManager() as manager:
            inspection = manager.inspect_run(sid, "no-boundary", application_root=self.root)
            self.assertEqual(inspection.status, "unrecoverable")
            self.assertIsNone(inspection.choice)
            fake = RunRecoveryChoice("no-boundary", None, "missing", 0, "choice")
            with self.assertRaises(SessionError) as missing:
                await manager.open_run(sid, "no-boundary", choice=fake, application_root=self.root)
            self.assertEqual(missing.exception.code, "no_execution_point")

    async def test_failed_idle_import_releases_handles_and_can_be_opened_elsewhere(self):
        sid, run_id = await self.interrupted()
        with SessionManager() as manager:
            inspection = manager.inspect_run(sid, run_id, application_root=self.root)
            with patch.object(manager._checkpointer, "restore_state", side_effect=CheckpointError("checkpoint_create_error")):
                with self.assertRaises(SessionError):
                    await manager.open_run(sid, run_id, choice=inspection.choice, application_root=self.root)
            self.assertFalse(manager._sessions)
        with SessionManager() as other:
            again = other.inspect_run(sid, run_id, application_root=self.root)
            opened = await other.open_run(sid, run_id, choice=again.choice, application_root=self.root)
            self.assertEqual(opened.run_id, run_id)

    async def test_paused_result_is_cached_without_second_dispatch(self):
        from lineage_agent.agent.execution import ExecutionPaused
        sid, run_id = await self.interrupted()
        with SessionManager() as manager:
            inspection = manager.inspect_run(sid, run_id, application_root=self.root)
            opened = await manager.open_run(sid, run_id, choice=inspection.choice, application_root=self.root)
            with patch.object(runtime, "_resume_agent", side_effect=ExecutionPaused("tool_recovery_unsupported")) as execute:
                first = await manager.resume_run(sid, choice=opened.choice)
                second = await manager.resume_run(sid, choice=opened.choice)
            self.assertEqual(first.status, "paused")
            self.assertEqual(first.phase, "request_ready")
            self.assertEqual(first.quota["reserved"]["model_calls"], 1)
            self.assertEqual(len(first.uncertain_attempts), 1)
            self.assertEqual(first, second)
            execute.assert_awaited_once()
            with self.assertRaises(SessionError) as blocked:
                await manager.run_turn(sid, "not allowed")
            self.assertEqual(blocked.exception.code, "session_blocked")

    async def test_paused_diagnostic_read_failure_preserves_original_reason(self):
        from lineage_agent.agent.execution import ExecutionPaused
        from lineage_agent.archive import ArchiveError
        sid, run_id = await self.interrupted()
        with SessionManager() as manager:
            inspection = manager.inspect_run(sid, run_id, application_root=self.root)
            opened = await manager.open_run(sid, run_id, choice=inspection.choice, application_root=self.root)
            archive = manager._checkpointer.archive(sid)
            with patch.object(runtime, "_resume_agent", side_effect=ExecutionPaused("tool_recovery_unsupported")), \
                    patch.object(archive, "read_execution", side_effect=ArchiveError("archive_integrity_error")), \
                    patch.object(archive, "budget_projection", side_effect=ArchiveError("archive_integrity_error")):
                result = await manager.resume_run(sid, choice=opened.choice)
            self.assertEqual(result.status, "paused")
            self.assertEqual(result.reason, "tool_recovery_unsupported")
            self.assertIsNone(result.phase)
            self.assertEqual(result.quota, {})
            self.assertEqual(len(result.warnings), 2)


class ExecutionCliTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        choice = RunRecoveryChoice("run", None, "point", 3, "choice")
        self.inspection = RunInspection("conversation", "run", "ready", "request_ready", choice,
            uncertain_attempts=("attempt",), quota={"remaining_model_calls": 2, "model_call_limit": 3, "used": {"model_calls": 1}})
        self.paths = {"application_root": Path("."), "project_root": None, "log_dir": Path("logs")}

    async def test_non_terminal_resume_only_inspects(self):
        manager = SimpleNamespace(inspect_run=Mock(return_value=self.inspection), open_run=AsyncMock(), resume_run=AsyncMock())
        reader = SimpleNamespace(confirm=AsyncMock(return_value=True))
        with redirect_stderr(StringIO()), patch.object(cli.sys.stdin, "isatty", return_value=False):
            result = await cli._resume_run(manager, "conversation", "run", self.paths, input_reader=reader)
        self.assertIsNone(result)
        manager.open_run.assert_not_called()
        manager.resume_run.assert_not_called()
        reader.confirm.assert_not_called()

    async def test_terminal_confirmation_and_cancel_do_not_create_new_run(self):
        opened = SimpleNamespace(session_id="conversation", choice=self.inspection.choice, settings=SETTINGS)
        manager = SimpleNamespace(inspect_run=Mock(return_value=self.inspection), open_run=AsyncMock(return_value=opened),
                                  resume_run=AsyncMock(return_value=RunResumed("conversation", "run", "paused", reason="policy")))
        reader = SimpleNamespace(confirm=AsyncMock(side_effect=[False, True]))
        output = StringIO()
        with redirect_stderr(output), patch.object(cli.sys.stdin, "isatty", return_value=True), \
                patch.object(cli.sys.stdout, "isatty", return_value=True), patch.object(output, "isatty", return_value=True), \
                patch.object(cli, "load_current_limits", return_value={}):
            self.assertIsNone(await cli._resume_run(manager, "conversation", "run", self.paths, input_reader=reader))
            manager.open_run.assert_not_called()
            result = await cli._resume_run(manager, "conversation", "run", self.paths, input_reader=reader)
        self.assertIs(result, opened)
        manager.open_run.assert_awaited_once()
        manager.resume_run.assert_awaited_once_with("conversation", choice=self.inspection.choice)
        self.assertIn("暂停", output.getvalue())


if __name__ == "__main__":
    unittest.main()
