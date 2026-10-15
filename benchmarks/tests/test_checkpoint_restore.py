"""Snapshot import creates idle native state without business publication."""

from copy import deepcopy
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from langgraph.graph import END, START, StateGraph

from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.context import new_context
from lineage_agent.agent.record_codec import encode_record
from lineage_agent.agent.record_journal import RecordJournal
from lineage_agent.agent.records import make_record_batch
from lineage_agent.agent.state import AgentState
from lineage_agent.archive import ArchiveStore
from lineage_agent.artifacts.store import ArtifactStore
from lineage_agent.storage import conversation_paths


class CheckpointRestoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="checkpoint-restore-")
        self.addCleanup(self.temp.cleanup)
        self.paths = conversation_paths(Path(self.temp.name), "conversation")
        self.archive = ArchiveStore(self.paths.directory)
        self.addCleanup(self.archive.close)
        self.saver = SessionCheckpointer()
        self.addCleanup(self.saver.release, "scope")
        self.store = ArtifactStore()
        self.addCleanup(self.store.close)
        self.artifacts = self.store.bind_session("scope", directory=self.paths.artifacts)
        self.items = [{"role": "user", "content": "original"}, {"role": "assistant", "content": "answer"}]
        with_journal = RecordJournal(self.paths.history)
        try:
            batch = make_record_batch("run", 0, self.items)
            receipts, watermark = with_journal.append("run", batch)
        finally:
            with_journal.close()
        self.view_id = self.archive.prepare_view(record_files={"run": watermark},
                                                history_scopes={}, summaries=[], artifacts=[])
        self.archive.confirm_view(self.view_id, "old-process-checkpoint")
        active = new_context()
        active.update(turn_start=2, turns=[{"run_id": "run", "item_count": 2,
                    "group_count": 0, "record_end": 2}], item_refs=[r["source_ref"] for r in batch])
        usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None, "missing_usage_calls": 1}
        self.state = {
            "input": deepcopy(self.items), "history_end": 2, "context_risks": [None, None],
            "context_budget": None, "active_context": active, "resource_view_id": self.view_id,
            "compaction": None, "compaction_events": [], "record_batch": receipts,
            "record_files": {"run": watermark}, "record_sequence": 2, "record_revision": 1,
            "record_admitted_through": 2, "history_scopes": {}, "record_metrics": {},
            "execution_facts": [], "run_call_ids": [], "summary_calls": 0, "summary_result": None,
            "summary_usage_totals": {**usage, "missing_usage_calls": 0}, "run_id": "run",
            "conversation_id": "conversation", "log_path": "original-log", "log_complete": True,
            "result": None, "model_calls": 1, "tool_requests": 0, "tool_calls": 0,
            "tool_outcomes": {"succeeded": 0, "rejected": 0, "failed": 0, "skipped": 0},
            "usage_totals": usage, "stop_reason": "completed", "error": None,
            "error_code": None, "cleanup_error": None,
        }

    async def test_restore_imports_exact_idle_state_and_never_republishes_view(self):
        before = self.paths.database.read_bytes()
        with patch.object(self.archive, "confirm_view", side_effect=AssertionError("must not publish")), \
                patch.object(self.archive, "prepare_view", side_effect=AssertionError("must not create view")), \
                patch.object(self.archive, "append_event", side_effect=AssertionError("must not add fact")):
            checkpoint_id = await self.saver.restore_state("scope", self.state,
                                                          archive=self.archive, artifacts=self.artifacts)
        self.assertNotEqual(checkpoint_id, "old-process-checkpoint")
        saved = await self.saver.read_confirmed("scope", checkpoint_id)
        self.assertEqual(encode_record(saved.checkpoint["channel_values"]), encode_record(self.state))
        self.assertEqual(self.saver.confirmed_view("scope"), self.view_id)
        self.assertEqual(self.paths.database.read_bytes(), before)
        self.assertEqual(self.archive.view_for_checkpoint("old-process-checkpoint"), self.view_id)
        # A graph with the production entry-node shape also sees an idle state.
        graph = StateGraph(AgentState)
        graph.add_node("model", lambda _: self.fail("restoration cannot invoke model"))
        graph.add_edge(START, "model")
        graph.add_edge("model", END)
        imported = await graph.compile(checkpointer=self.saver).aget_state({"configurable": {"thread_id": "scope"}})
        self.assertEqual(imported.next, ())
        self.assertEqual(imported.tasks, ())
        self.assertIsNone(imported.values["usage_totals"]["total_tokens"])

    async def test_native_save_failure_removes_temporary_binding_without_closing_borrowed_archive(self):
        with patch.object(self.saver, "put", side_effect=OSError("native memory failure")):
            with self.assertRaises(CheckpointError):
                await self.saver.restore_state("scope", self.state, archive=self.archive, artifacts=self.artifacts)
        self.assertIsNone(self.saver.last_confirmed("scope"))
        self.assertIsNone(self.saver.failure("scope"))
        self.assertIsNone(self.saver.get_tuple({"configurable": {"thread_id": "scope"}}))
        self.assertEqual(self.archive.view(self.view_id)["view_id"], self.view_id)
        checkpoint_id = await self.saver.restore_state("scope", self.state, archive=self.archive, artifacts=self.artifacts)
        self.assertEqual(self.saver.last_confirmed("scope"), checkpoint_id)

    async def test_invalid_snapshot_is_rejected_before_creating_native_state(self):
        broken = deepcopy(self.state)
        broken["input"][0]["content"] = "different original"
        with self.assertRaises(CheckpointError) as error:
            await self.saver.restore_state("scope", broken, archive=self.archive, artifacts=self.artifacts)
        self.assertEqual(error.exception.code, "archive_integrity_error")
        self.assertIsNone(self.saver.last_confirmed("scope"))
        self.assertEqual(self.archive.view(self.view_id)["view_id"], self.view_id)

    async def test_cancelled_import_releases_temporary_scope_and_preserves_archive(self):
        with patch("langgraph.graph.state.CompiledStateGraph.aget_state", side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await self.saver.restore_state("scope", self.state, archive=self.archive, artifacts=self.artifacts)
        self.assertIsNone(self.saver.last_confirmed("scope"))
        self.assertNotIn("scope", self.saver._archives)
        self.assertIsNone(self.saver.get_tuple({"configurable": {"thread_id": "scope"}}))
        self.assertEqual(self.archive.view(self.view_id)["view_id"], self.view_id)


if __name__ == "__main__":
    unittest.main()
