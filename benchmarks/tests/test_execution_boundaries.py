"""Failure windows between durable results and adopted business state."""

import asyncio
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmarks.checks.support.runtime import LocalClient, SETTINGS, TOOL, call, response, message
from lineage_agent.agent import runtime
from lineage_agent.agent.sessions import SessionError, SessionManager
from lineage_agent.archive import ArchiveError
from lineage_agent.artifacts.store import ArtifactError, SessionArtifacts


class ExecutionBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="execution-boundaries-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = replace(SETTINGS, application_root=self.root, log_dir=self.root / "logs")
        self.enterContext(patch.dict("os.environ", {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING": "false",
                                                   "LANGCHAIN_TRACING_V2": "false"}))
        asyncio.get_event_loop().slow_callback_duration = 10

    async def fail_confirmation(self, phase, *, final_record=False):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings)
            archive = manager._checkpointer.archive(sid)
            confirm = archive.confirm_execution

            def fail(point_id, checkpoint_id, expected_parent):
                point = archive.execution(point_id, confirmed=False)
                if point["phase"] == phase and bool(point["state"]["execution"].get("final_record_id")) == final_record:
                    raise ArchiveError("archive_write_error")
                return confirm(point_id, checkpoint_id, expected_parent)

            client = LocalClient([response(message("saved answer"), usage=(9, 3, 12))])
            with patch.object(archive, "confirm_execution", side_effect=fail), \
                    patch.object(runtime, "create_client", return_value=client):
                failed = await manager.run_turn(sid, "original question")
            self.assertFalse(failed.checkpoint_complete)
            self.assertEqual(len(client.requests), 1)
            self.assertIsNone(archive.conversation()["current_snapshot_id"])
            with self.assertRaises(SessionError) as blocked:
                await manager.run_turn(sid, "must not unblock")
            self.assertEqual(blocked.exception.code, "session_blocked")
            run_id = failed.state["run_id"]
            before = (self.root / ".lineage" / sid / "history" / f"{run_id}.jsonl").read_bytes()
        return sid, run_id, before

    async def complete(self, sid, run_id):
        with SessionManager() as manager:
            inspected = manager.inspect_run(sid, run_id, application_root=self.root)
            opened = await manager.open_run(sid, run_id, choice=inspected.choice, application_root=self.root)
            client = LocalClient([])
            with patch.object(runtime, "create_client", return_value=client):
                resumed = await manager.resume_run(sid, choice=opened.choice)
            self.assertEqual(resumed.status, "completed", resumed)
            self.assertEqual(client.requests, [])
            self.assertEqual(resumed.turn.state["run_id"], run_id)
            self.assertEqual(resumed.turn.state["model_calls"], 1)
            self.assertEqual(resumed.turn.state["usage_totals"], {
                "input_tokens": 9, "output_tokens": 3, "total_tokens": 12, "missing_usage_calls": 0})
            archive = manager._checkpointer.archive(sid)
            self.assertEqual(len(archive.published_runs(resumed.turn.snapshot_id)), 1)
            self.assertEqual(len([item for item in resumed.turn.state["input"]
                                  if item.get("role") == "assistant"]), 1)
            return resumed.turn.state

    async def test_result_saved_before_response_point_reuses_without_request(self):
        sid, run_id, before = await self.fail_confirmation("response_saved")
        await self.complete(sid, run_id)
        self.assertEqual((self.root / ".lineage" / sid / "history" / f"{run_id}.jsonl").read_bytes(), before)

    async def test_unconfirmed_history_tail_does_not_duplicate_answer_or_usage(self):
        sid, run_id, before = await self.fail_confirmation("finalizing")
        state = await self.complete(sid, run_id)
        self.assertEqual((self.root / ".lineage" / sid / "history" / f"{run_id}.jsonl").read_bytes(), before)
        self.assertEqual(len(state["record_files"][run_id]["segments"]), 2)

    async def test_final_candidate_before_confirm_uses_fixed_payload_and_identity(self):
        sid, run_id, _ = await self.fail_confirmation("finalizing", final_record=True)
        await self.complete(sid, run_id)

    async def test_confirmed_final_candidate_publish_failure_does_not_rerun(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings)
            archive = manager._checkpointer.archive(sid)
            with patch.object(archive, "publish_snapshot", side_effect=ArchiveError("archive_write_error")), \
                    patch.object(runtime, "create_client", return_value=LocalClient([response(message("saved answer"), usage=(9,3,12))])):
                failed = await manager.run_turn(sid, "original question")
            self.assertFalse(failed.checkpoint_complete)
            run_id = failed.state["run_id"]
            self.assertTrue(archive.read_execution(run_id)["state"]["execution"]["final_record_id"])
        await self.complete(sid, run_id)

    async def test_final_event_before_candidate_registration_does_not_conflict_on_resume(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings)
            archive = manager._checkpointer.archive(sid)
            save = archive.save_operation

            def fail_candidate(run_id, identity, kind, payload):
                if kind == "finalization":
                    raise ArchiveError("archive_write_error")
                return save(run_id, identity, kind, payload)

            with patch.object(archive, "save_operation", side_effect=fail_candidate), \
                    patch.object(runtime, "create_client", return_value=LocalClient([
                        response(message("saved answer"), usage=(9, 3, 12))])):
                failed = await manager.run_turn(sid, "original question")
            self.assertFalse(failed.checkpoint_complete)
            run_id = failed.state["run_id"]
            self.assertEqual(archive.read_execution(run_id)["phase"], "finalizing")
        await self.complete(sid, run_id)

    async def test_second_tool_save_failure_retains_completed_prefix_without_adopting_half_group(self):
        entered = []

        async def executor(name, arguments):
            entered.append(arguments["key"])
            return "observed " + arguments["key"]

        save, count = SessionArtifacts.put_candidate, 0

        def fail_second(binding, *args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise ArtifactError("artifact_write_error")
            return save(binding, *args, **kwargs)

        with SessionManager() as manager:
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=executor)
            client = LocalClient([response(*[call(identity, arguments='{"key":"' + identity + '"}')
                                             for identity in ("one", "two", "three")])])
            with patch.object(SessionArtifacts, "put_candidate", new=fail_second), \
                    patch.object(runtime, "create_client", return_value=client):
                result = await manager.run_turn(sid, "input")
            self.assertEqual(entered, ["one", "two"])
            self.assertFalse(result.checkpoint_complete)
            self.assertEqual(result.state["tool_calls"], 2)
            self.assertEqual(result.state["tool_outcomes"], {"succeeded": 2, "rejected": 0, "failed": 0, "skipped": 0})
            self.assertEqual([fact["call_id"] for fact in result.state["execution_facts"]], ["one", "two"])
            self.assertFalse(any(item.get("type") == "function_call_output" for item in result.state["input"]))


if __name__ == "__main__":
    unittest.main()
