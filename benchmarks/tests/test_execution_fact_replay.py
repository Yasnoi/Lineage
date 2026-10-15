"""Stable response receipts across local processing and diagnostic failures."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmarks.checks.support.runtime import LocalClient, SETTINGS, message, response
from lineage_agent.agent import runtime
from lineage_agent.agent.checkpointing import SessionCheckpointer
from lineage_agent.agent.sessions import SessionManager
from lineage_agent.archive import ArchiveError, ArchiveStore
from lineage_agent.recording.recorder import RunRecorder
from lineage_agent.storage import conversation_paths


class ExecutionFactTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="execution-facts-")
        self.addCleanup(temporary.cleanup)
        self.archive = ArchiveStore(conversation_paths(Path(temporary.name), "conversation").directory)
        self.addCleanup(self.archive.close)
        self.archive.register_conversation({"settings": {"model": "test"}}, [])
        self.archive.begin_run("run", None)
        self.saver = SessionCheckpointer()
        self.saver.bind_archive("conversation", self.archive)

    def operation(self, kind, *, complete=True, status="completed"):
        metadata = {"lane": "R", "status": "preparing", "step_id": "fixed-request",
                    "lanes": {"R": {"summary_id": "candidate", "status": "prepared"}}}
        self.archive.save_operation("run", "operation", kind, {"request": {}, "metadata": metadata})
        attempt = self.archive.start_attempt("operation", {kind + "_calls": 1})
        if complete:
            self.archive.finish_attempt(attempt["attempt_id"], {
                "raw_response": response(message("saved response"), status=status).model_dump(mode="json")})
        return {"run_id": "run", "model_calls": 0, "summary_calls": 0, "execution_facts": [],
                "compaction": {"status": "preparing"},
                "execution": {"point_id": "source-point", "operation_id": "operation"}}

    def test_summary_receipt_uses_fixed_request_and_keeps_distinct_processing_facts(self):
        state = self.operation("summary")
        failed = {"summary_calls": 1, "summary_result": None,
                  "compaction": {"status": "failed", "reason": "logging_error"}}
        adopted = {"summary_calls": 1, "summary_result": None,
                   "compaction": {"status": "history_ready", "reason": None}}
        self.saver.archive_facts("conversation", state, failed, "summarize_R")
        self.saver.archive_facts("conversation", state, adopted, "summarize_R")
        self.saver.archive_facts("conversation", state, deepcopy(adopted), "summarize_R")
        events = self.archive.events("run")
        results = [event["payload"] for event in events if event["kind"] == "summary_result"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["result"]["text"], "saved response")
        self.assertEqual(results[0]["compaction"]["provenance_phase"], "request_fixed")
        self.assertEqual(results[0]["compaction"]["summary_lane"], "R")
        self.assertEqual(results[0]["compaction"]["summary_id"], "candidate")
        transitions = [event["payload"] for event in events if event["kind"] == "compaction"]
        self.assertEqual([event["compaction"]["status"] for event in transitions], ["failed", "history_ready"])
        self.assertTrue(all(event["source_execution_id"] == "source-point" for event in transitions))

    def test_unknown_attempt_has_no_invented_complete_response(self):
        state = self.operation("model", complete=False)
        self.saver.archive_facts("conversation", state, {"model_calls": 1, "result": None}, "send_request")
        self.assertFalse(self.archive.events("run"))
        self.assertEqual(len(self.archive.budget_projection("run")["unknown_attempt_ids"]), 1)

    def test_raw_response_error_is_not_replaced_with_logging_error(self):
        state = self.operation("model", status="incomplete")
        self.saver.archive_facts("conversation", state, {
            "model_calls": 1, "result": None, "error_code": "logging_error"}, "model")
        saved = self.archive.events("run")[0]["payload"]
        self.assertEqual(saved["error_code"], "incomplete_response")
        self.assertEqual(saved["result"]["text"], "saved response")


class ResponseReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_log_failure_before_failed_adoption_does_not_block_saved_response_replay(self):
        with TemporaryDirectory(prefix="response-replay-") as temp, patch.dict("os.environ", {
                "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}):
            asyncio.get_running_loop().slow_callback_duration = 10
            root = Path(temp)
            settings = replace(SETTINGS, application_root=root, log_dir=root / "logs")
            with SessionManager() as manager:
                sid = manager.create_session(settings)
                archive = manager._checkpointer.archive(sid)
                confirm, emit = archive.confirm_execution, RunRecorder.emit

                def fail_confirmation(point_id, checkpoint_id, expected_parent):
                    if archive.execution(point_id, confirmed=False)["phase"] == "finalizing":
                        raise ArchiveError("archive_write_error")
                    return confirm(point_id, checkpoint_id, expected_parent)

                def fail_log(recorder, event, **details):
                    if event == "model.returned":
                        recorder._fail("write")
                        return False
                    return emit(recorder, event, **details)

                client = LocalClient([response(message("saved answer"), usage=(9, 3, 12))])
                with patch.object(archive, "confirm_execution", side_effect=fail_confirmation), \
                        patch.object(RunRecorder, "emit", fail_log), \
                        patch.object(runtime, "create_client", return_value=client):
                    failed = await manager.run_turn(sid, "original question")
                self.assertFalse(failed.checkpoint_complete)
                self.assertEqual(len(client.requests), 1)
                run = failed.state["run_id"]
                receipt = next(event["payload"] for event in archive.events(run) if event["kind"] == "model_result")
                self.assertIsNone(receipt["error_code"])
            with SessionManager() as manager:
                inspected = manager.inspect_run(sid, run, application_root=root)
                opened = await manager.open_run(sid, run, choice=inspected.choice, application_root=root)
                client = LocalClient([])
                with patch.object(runtime, "create_client", return_value=client):
                    resumed = await manager.resume_run(sid, choice=opened.choice)
                self.assertEqual(resumed.status, "completed", resumed)
                self.assertEqual(client.requests, [])
                self.assertEqual(resumed.turn.state["usage_totals"]["total_tokens"], 12)
                saved = await manager.get_record(sid, run)
                self.assertEqual(len(saved["model_results"]), 1)
                self.assertEqual(saved["model_results"][0]["result"]["text"], "saved answer")


if __name__ == "__main__":
    unittest.main()
