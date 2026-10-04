"""Record-file failures preserve paid facts and never replay business work."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.base import empty_checkpoint

from benchmarks.checks.support.runtime import LocalClient, SETTINGS, TOOL, call, message, response
from lineage_agent.agent import runtime
from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.record_codec import decode_record
from lineage_agent.agent.record_journal import JournalError, RecordJournal
from lineage_agent.agent.records import make_record_batch
from lineage_agent.agent.sessions import SessionError, SessionManager
from lineage_agent.artifacts.store import ArtifactStore


class RecordRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="record-runtime-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = replace(SETTINGS, log_dir=self.root / "logs", application_root=self.root,
                                context_budget_tokens=250000)
        self.executed = []
        self.tracing = patch.dict("os.environ", {
            "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
        })
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    async def tool(self, name, arguments):
        self.executed.append(arguments)
        return "actual small tool result, key=37"

    async def run_case(self, replies):
        client = LocalClient(replies)
        with patch.object(runtime, "create_client", return_value=client):
            state = await runtime.run_agent("do the original task", self.settings, tools=[TOOL], execute_tool=self.tool)
        return state, client

    def originals(self, state):
        path = self.root / ".lineage" / state["conversation_id"] / "history" / (state["run_id"] + ".jsonl")
        return [decode_record(line) for line in path.read_bytes().splitlines(keepends=True)]

    async def test_initial_file_failure_stops_before_client_or_tool(self):
        with patch.object(runtime, "RecordJournal", side_effect=JournalError("record_create_error")), \
                patch.object(runtime, "create_client") as create:
            state = await runtime.run_agent("original requirement", self.settings, tools=[TOOL], execute_tool=self.tool)
        self.assertEqual(state["error_code"], "record_create_error")
        self.assertEqual(state["input"], [{"role": "user", "content": "original requirement"}])
        self.assertEqual(state["model_calls"], 0)
        self.assertEqual(state["tool_calls"], 0)
        self.assertFalse(self.executed)
        create.assert_not_called()

    async def test_model_append_failure_preserves_response_and_skips_tool_once(self):
        native = RecordJournal.append
        def fail(journal, run_id, batch):
            if any(record["kind"] == "model" for record in batch):
                raise JournalError("record_write_error")
            return native(journal, run_id, batch)
        with patch.object(RecordJournal, "append", new=fail):
            state, client = await self.run_case([response(call("paid-call")), response(message("must not send"))])
        self.assertEqual(state["error_code"], "record_write_error")
        self.assertEqual(state["model_calls"], 1)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(state["tool_outcomes"]["skipped"], 1)
        self.assertEqual(state["result"]["tool_calls"][0]["call_id"], "paid-call")
        self.assertTrue(any(item.get("call_id") == "paid-call" for item in state["input"]))
        self.assertFalse(self.executed)
        self.assertEqual([record["kind"] for record in self.originals(state)], ["input"])
        self.assertEqual(state["active_context"]["current_run_id"], state["run_id"])

    async def test_tool_append_failure_keeps_actual_execution_and_unconfirmed_tail(self):
        native = RecordJournal.append
        def fail(journal, run_id, batch):
            result = native(journal, run_id, batch)
            if any(record["kind"] == "tool" for record in batch):
                raise JournalError("record_write_error")
            return result
        with patch.object(RecordJournal, "append", new=fail):
            state, client = await self.run_case([response(call("executed-call")), response(message("must not send"))])
        self.assertEqual(state["error_code"], "record_write_error")
        self.assertEqual(len(self.executed), 1)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(state["tool_calls"], 1)
        self.assertEqual(state["tool_outcomes"]["succeeded"], 1)
        self.assertEqual(state["tool_outcomes"]["skipped"], 0)
        self.assertTrue(any("key=37" in item.get("output", "") for item in state["input"]))
        self.assertEqual(state["execution_facts"][0]["outcome"], "succeeded")
        originals = self.originals(state)
        self.assertEqual([record["kind"] for record in originals], ["input", "model", "tool"])
        self.assertEqual(originals[-1]["execution"]["outcome"], "succeeded")
        # The physically appended tool line was never admitted by a checkpoint.
        self.assertEqual(state["record_files"][state["run_id"]]["record_count"], 2)
        self.assertIsNone(state["active_context"]["run_summary"])

    async def test_final_outcome_append_failure_keeps_answer_without_history_transition(self):
        native = RecordJournal.append
        def fail(journal, run_id, batch):
            if any(record["kind"] == "outcome" for record in batch):
                raise JournalError("record_write_error")
            return native(journal, run_id, batch)
        with patch.object(RecordJournal, "append", new=fail):
            state, client = await self.run_case([response(message("actual final answer"))])
        self.assertEqual(state["stop_reason"], "error")
        self.assertEqual(state["error_code"], "record_write_error")
        self.assertEqual(state["result"]["text"], "actual final answer")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(state["active_context"]["current_run_id"], state["run_id"])
        self.assertFalse(state["active_context"]["turns"])
        self.assertEqual([record["kind"] for record in self.originals(state)], ["input", "model"])

    async def test_final_checkpoint_write_failure_retains_file_and_old_active_view(self):
        native = SessionCheckpointer.put
        def fail(saver, config, checkpoint, metadata, versions):
            if metadata.get("source") == "update":
                raise OSError("private final checkpoint failure")
            return native(saver, config, checkpoint, metadata, versions)
        with patch.object(SessionCheckpointer, "put", new=fail):
            state, client = await self.run_case([response(message("actual final answer"))])
        self.assertEqual(state["error_code"], "checkpoint_write_error")
        self.assertEqual(state["result"]["text"], "actual final answer")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(state["active_context"]["current_run_id"], state["run_id"])
        self.assertFalse(state["active_context"]["turns"])
        self.assertEqual(self.originals(state)[-1]["outcome"]["stop_reason"], "completed")

    async def test_final_confirmation_failure_does_not_publish_history_or_allow_session_retry(self):
        native = SessionCheckpointer.read_confirmed
        captured = []
        async def fail(saver, scope, checkpoint):
            saved = await native(saver, scope, checkpoint)
            values = saved.checkpoint["channel_values"]
            if values.get("active_context", {}).get("current_run_id", "missing") is None:
                parent = saver.get_tuple(saved.parent_config)
                captured.append(deepcopy(parent.checkpoint["channel_values"]["active_context"]))
                raise CheckpointError("checkpoint_read_error")
            return saved
        with SessionManager() as manager:
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=self.tool)
            client = LocalClient([response(message("actual final answer"))])
            with patch.object(SessionCheckpointer, "read_confirmed", new=fail), \
                    patch.object(runtime, "create_client", return_value=client):
                turn = await manager.run_turn(sid, "original requirement")
            self.assertFalse(turn.checkpoint_complete)
            self.assertEqual(turn.checkpoint_error, "checkpoint_read_error")
            self.assertEqual(len(captured), 1)
            self.assertEqual(turn.state["active_context"], captured[0])
            self.assertEqual(turn.state["result"]["text"], "actual final answer")
            with self.assertRaises(SessionError) as error:
                await manager.run_turn(sid, "must not restart")
            self.assertEqual(error.exception.code, "session_blocked")
            with self.assertRaises(SessionError):
                await manager.get_record(sid, turn.state["run_id"])
            self.assertEqual(len(client.requests), 1)
            self.assertEqual(self.originals(turn.state)[-1]["outcome"]["stop_reason"], "completed")

    async def saved_original(self):
        sid, run = str(uuid4()), str(uuid4())
        saver = SessionCheckpointer()
        self.addCleanup(saver.release, sid)
        journal = RecordJournal(self.root / ".lineage" / sid / "history")
        saver.bind_history(sid, journal)
        receipts, watermark = journal.append(run, make_record_batch(run, 0, [{"role": "user", "content": "confirmed original"}]))
        values = {"run_id": run, "record_batch": receipts, "record_files": {run: watermark},
                  "record_sequence": 1, "history_scopes": {}}
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = deepcopy(values)
        checkpoint["channel_versions"] = dict.fromkeys(values, 1)
        config = await saver.aput({"configurable": {"thread_id": sid, "checkpoint_ns": ""}},
                                  checkpoint, {}, checkpoint["channel_versions"])
        return saver, sid, config, values, journal

    async def test_existing_confirmed_file_manifest_cannot_be_cleared(self):
        saver, sid, config, values, journal = await self.saved_original()
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {**values, "record_files": {}, "record_batch": []}
        checkpoint["channel_versions"] = dict.fromkeys(values, 2)
        with self.assertRaises(CheckpointError) as error:
            await saver.aput(config, checkpoint, {}, checkpoint["channel_versions"])
        self.assertEqual(error.exception.code, "record_integrity_error")
        self.assertEqual(saver.last_confirmed(sid), config["configurable"]["checkpoint_id"])
        saved = await saver.read_confirmed(sid, config["configurable"]["checkpoint_id"])
        self.assertEqual(saved.checkpoint["channel_values"]["record_files"], values["record_files"])
        self.assertEqual(journal.records(values["run_id"], values["record_files"][values["run_id"]])[0]["item"]["content"], "confirmed original")

    async def test_receipt_outside_confirmed_file_manifest_is_rejected(self):
        saver, sid, config, values, _ = await self.saved_original()
        orphan = deepcopy(values["record_batch"][0])
        orphan["run_id"] = str(uuid4())
        orphan["source_ref"] = f"history://record/{orphan['run_id']}/0"
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {**values, "record_batch": [orphan]}
        checkpoint["channel_versions"] = dict.fromkeys(values, 2)
        with self.assertRaises(CheckpointError) as error:
            await saver.aput(config, checkpoint, {}, checkpoint["channel_versions"])
        self.assertEqual(error.exception.code, "record_integrity_error")
        self.assertEqual(saver.last_confirmed(sid), config["configurable"]["checkpoint_id"])

    async def test_manager_close_releases_checkpoints_keeps_files_and_borrowed_store(self):
        with ArtifactStore() as borrowed:
            manager = SessionManager()
            sid = manager.create_session(self.settings, artifact_store=borrowed)
            view = manager.artifacts(sid)
            artifact = view.put("persistent borrowed artifact", tool_name="custom")
            client = LocalClient([response(message("finished"))])
            with patch.object(runtime, "create_client", return_value=client):
                turn = await manager.run_turn(sid, "original requirement")
            self.assertTrue(turn.checkpoint_complete)
            directory = self.root / ".lineage" / sid
            files = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
            self.assertTrue(any(path.suffix == ".jsonl" for path in files))
            self.assertTrue(any(path.suffix == ".raw" for path in files))
            journal = manager._checkpointer.history_journal(sid)
            manager.close()
            self.assertFalse(manager._sessions)
            self.assertNotIn(sid, manager._checkpointer.storage)
            self.assertNotIn(sid, manager._checkpointer._history_journals)
            with self.assertRaises(CheckpointError):
                await manager._checkpointer.read_confirmed(sid, turn.checkpoint_id)
            with self.assertRaises(JournalError):
                journal.records(turn.state["run_id"], turn.state["record_files"][turn.state["run_id"]])
            self.assertEqual({path: path.read_bytes() for path in files}, files)
            self.assertFalse(borrowed._closed)
            self.assertEqual(view.read(artifact["artifact_ref"], 0)["text"], "persistent borrowed artifact")
            view.put("caller retains its handle", tool_name="custom")


if __name__ == "__main__":
    unittest.main()
