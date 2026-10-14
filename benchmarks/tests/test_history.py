"""Original records, immutable recall ranges, and bounded native history tools."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.base import empty_checkpoint

from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.context import ContextRecords, new_context, read_archived_record
from lineage_agent.agent.records import RecordStore, make_record_batch, record_projection
from lineage_agent.agent.record_journal import JournalError, RecordJournal
from lineage_agent.archive import ArchiveError, ArchiveStore
from lineage_agent.observations.manager import ObservationManager
from lineage_agent.tools.execution import execute_calls
from lineage_agent.tools.history import HISTORY_TOOLS, HISTORY_TOOL_NAMES, HistoryExecutor
from lineage_agent.tools.types import (
    ToolArgumentError, ToolExecutionError, inline_tool_output, serialize_tool_output, tool_output_bytes,
)


class _NoArtifacts:
    def put(self, *args, **kwargs):
        raise AssertionError("history recall must not be archived")


class HistoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.saver = SessionCheckpointer()
        self.sid, self.run = str(uuid4()), str(uuid4())
        self.temporary = TemporaryDirectory(prefix="history-files-test-")
        self.addCleanup(self.temporary.cleanup)
        self.journal = RecordJournal(Path(self.temporary.name) / ".lineage" / self.sid / "history")
        self.saver.bind_history(self.sid, self.journal)
        self.addCleanup(self.saver.release, self.sid)
        self.records = RecordStore(self.saver, self.sid)
        self.config = {"configurable": {"thread_id": self.sid, "checkpoint_ns": ""}}
        self.values = {"run_id": self.run, "record_batch": [], "record_files": {}, "record_sequence": 0, "history_scopes": {}}
        self.version = 0

    async def save(self, **updates):
        self.version += 1
        if "record_batch" in updates:
            receipts, watermark = self.journal.append(self.run, updates["record_batch"])
            updates["record_batch"] = receipts
            updates["record_files"] = {**self.values["record_files"], self.run: watermark}
            updates["record_sequence"] = watermark["record_count"]
        self.values.update(deepcopy(updates))
        self.values["resource_view_id"] = self.saver.prepare_state(self.sid, self.values)
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = deepcopy(self.values)
        checkpoint["channel_versions"] = dict.fromkeys(self.values, self.version)
        self.config = await self.saver.aput(self.config, checkpoint, {"step": self.version}, checkpoint["channel_versions"])
        return self.config["configurable"]["checkpoint_id"]

    async def seed(self, texts):
        batch = make_record_batch(self.run, 0, [{"role": "user", "content": text} for text in texts])
        checkpoint = await self.save(record_batch=batch)
        scope = await self.records.register_scope([item["source_ref"] for item in batch], checkpoint_id=checkpoint)
        await self.save(history_scopes={scope["history_ref"]: scope})
        return batch, scope

    async def test_records_survive_active_input_replacement_and_fixed_snapshot(self):
        first = make_record_batch(self.run, 0, [{"role": "user", "content": "要求保留数字 17；不得修改文件。"}])
        first_id = await self.save(record_batch=first, input=[first[0]["item"]])
        second = make_record_batch(self.run, 1, [{"type": "function_call_output", "call_id": "c1", "output": "earlier exact detail"}], group_id=1, kind="tool")
        await self.save(record_batch=second, input=[{"role": "assistant", "content": "derived summary"}])
        result = await self.records.records(self.run)
        self.assertEqual(result, first + second)
        self.assertEqual(await self.records.records(self.run, first_id), first)
        with self.assertRaises(CheckpointError):
            await self.records.get(second[0]["source_ref"], first_id)
        self.records.clear_index()
        self.assertEqual(await self.records.records(self.run), result)

    async def test_multi_run_history_order_survives_sorted_archive_encoding(self):
        self.run = "z-first-run"
        first = make_record_batch(self.run, 0, [{"role": "user", "content": "first turn"}])
        await self.save(run_id=self.run, record_batch=first)
        self.run = "a-later-run"
        second = make_record_batch(self.run, 0, [{"role": "user", "content": "later turn"}])
        await self.save(run_id=self.run, record_batch=second)
        archive = self.saver.archive(self.sid)
        view_id = self.saver.confirmed_view(self.sid)
        view = archive.view(view_id)
        self.assertEqual(list(view["record_files"]), ["a-later-run", "z-first-run"])
        self.assertEqual(view["record_order"], ["z-first-run", "a-later-run"])
        rebuilt = RecordStore(archive=archive, view_id=view_id)
        self.addCleanup(rebuilt.close)
        self.assertEqual(await rebuilt.records(), first + second)
        scope = await rebuilt.register_scope([second[0]["source_ref"], first[0]["source_ref"]])
        self.assertEqual([bounds["run_id"] for bounds in scope["ranges"]], ["z-first-run", "a-later-run"])

    async def test_original_record_creation_preserves_invalid_input_for_admission(self):
        items = [{"type": "function_call", "name": "tool", "call_id": [], "arguments": "{}"}]
        batch = make_record_batch(self.run, 0, items)
        self.assertEqual(batch[0]["item"], items[0])
        self.assertEqual(batch[0]["call_id"], [])
        calls = [{"type": "function_call", "name": "tool", "call_id": "skipped", "arguments": "{}"}]
        facts = [{"call_id": "skipped", "outcome": "skipped", "protocol_output_created": False}]
        skipped = make_record_batch(self.run, 0, calls, kind="model", call_facts=facts)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["execution"]["outcome"], "skipped")

    async def test_native_saved_non_json_opaque_values_do_not_poison_later_records(self):
        first = make_record_batch(self.run, 0, [{"type": "unknown_extension", "values": {1, 2}, "number": float("nan")}])
        await self.save(record_batch=first)
        second = make_record_batch(self.run, 1, [{"role": "user", "content": "later valid request"}])
        await self.save(record_batch=second)
        records = await self.records.records(self.run)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["item"]["values"], {1, 2})
        self.assertEqual((await self.records.get(second[0]["source_ref"]))["item"]["content"], "later valid request")
        text, omitted = record_projection(records[0])
        self.assertTrue(omitted)
        self.assertNotIn("NaN", text)
        self.records.clear_index()
        self.assertEqual(len(await self.records.records(self.run)), 2)

    async def test_missing_sequences_and_conflicting_immutable_identity_fail(self):
        batch, _ = await self.seed(["original"])
        changed = deepcopy(batch)
        changed[0]["item"]["content"] = "changed"
        with self.assertRaises(JournalError):
            await self.save(record_batch=changed)
        self.assertEqual((await self.records.records())[0]["item"]["content"], "original")
        self.run = str(uuid4())
        with self.assertRaises(JournalError):
            await self.save(record_batch=make_record_batch(self.run, 1, [{"role": "user", "content": "gap"}]))

    async def test_unconfirmed_checkpoints_and_released_cache_are_unreadable(self):
        checkpoint = empty_checkpoint()
        raw = self.saver.put(self.config, checkpoint, {}, {})
        with self.assertRaises(CheckpointError):
            await self.saver.read_confirmed(self.sid, raw["configurable"]["checkpoint_id"])
        batch, _ = await self.seed(["confirmed"])
        await self.records.get(batch[0]["source_ref"])
        self.saver.release(self.sid)
        with self.assertRaises(CheckpointError):
            await self.records.get(batch[0]["source_ref"])

    async def test_scope_is_immutable_and_contains_unreferenced_originals(self):
        batch, scope = await self.seed(["summary retained this", "omitted detail key=37; must not use 73", "other"])
        self.assertEqual(scope["ranges"], [{"run_id": self.run, "start": 0, "end": 3}])
        result = await self.records.search(scope["history_ref"], ["key=37"])
        self.assertEqual(result["hits"][0]["source_ref"], batch[1]["source_ref"])
        self.assertIn("must not use 73", result["hits"][0]["text"])
        late = make_record_batch(self.run, 3, [{"role": "assistant", "content": "key=37 new response"}])
        await self.save(record_batch=late)
        result = await self.records.search(scope["history_ref"], ["key=37"])
        self.assertEqual(len(result["hits"]), 1)
        altered = deepcopy(scope)
        altered["ranges"][0]["end"] = 4
        with self.assertRaises(CheckpointError):
            await self.save(history_scopes={scope["history_ref"]: altered})

    async def test_scope_registration_failure_blocks_saver_without_publishing_scope(self):
        batch, old_scope = await self.seed(["accepted original"])
        old_view = self.saver.confirmed_view(self.sid)
        new_ref = f"history://scope/{uuid4()}"
        with patch.object(self.records.archive, "save_scope", side_effect=ArchiveError("archive_write_error")):
            with self.assertRaises(CheckpointError) as failure:
                await self.records.register_scope([batch[0]["source_ref"]], history_ref=new_ref)
        self.assertEqual(failure.exception.code, "archive_write_error")
        self.assertEqual(self.saver.failure(self.sid).code, "archive_write_error")
        self.assertEqual(self.saver.confirmed_view(self.sid), old_view)
        self.assertEqual(await self.records.records(), batch)
        self.assertEqual((await self.records.search(old_scope["history_ref"], ["accepted"]))["query_status"], "matched")
        with self.assertRaises(ToolExecutionError):
            await self.records.search(new_ref, ["accepted"])
        with self.assertRaises(CheckpointError) as blocked:
            self.saver.prepare_state(self.sid, self.values)
        self.assertEqual(blocked.exception.code, "archive_write_error")

    async def test_read_unicode_escape_pagination_matches_fixed_projection(self):
        batch, _ = await self.seed(['中😀"\\\t\nKEY' * 4000])
        expected, omitted = record_projection(batch[0])
        start, parts = 0, []
        while True:
            result = await self.records.read(batch[0]["source_ref"], start)
            self.assertLessEqual(tool_output_bytes(inline_tool_output(serialize_tool_output(result))), 8192)
            self.assertEqual(result["text"], expected[result["start"]:result["end"]])
            self.assertEqual(result["opaque_content_omitted"], omitted)
            self.assertGreater(result["end"], start)
            parts.append(result["text"])
            if result["next_start"] is None:
                break
            start = result["next_start"]
        self.assertEqual("".join(parts), expected)
        self.assertGreater(len(parts), 1)
        for start, end in ((True, None), (-1, None), (0, len(expected) + 1), (2, 1)):
            with self.assertRaises(ToolArgumentError):
                await self.records.read(batch[0]["source_ref"], start, end)

    async def test_search_pages_have_no_omissions_and_cursors_are_bound(self):
        batch, scope = await self.seed([("尾\\\"" * 80 + "KEY KEY2\n") * 16, "KEY trailing"])
        expected = []
        for record in batch:
            text, _ = record_projection(record)
            position = text.find("KEY")
            while position >= 0:
                expected.append((record["source_ref"], position))
                position = text.find("KEY", position + 1)
        cursor, found, saved_cursor = None, [], None
        while True:
            result = await self.records.search(scope["history_ref"], ["KEY", "KEY2"], cursor)
            self.assertLessEqual(len(result["hits"]), 10)
            self.assertLessEqual(tool_output_bytes(inline_tool_output(serialize_tool_output(result))), 8192)
            found.extend((hit["source_ref"], hit["matches"][0]["start"]) for hit in result["hits"])
            if not result["has_more"]:
                break
            cursor = result["next_cursor"]
            saved_cursor = cursor
        self.assertEqual(found, expected)
        self.assertIsNotNone(saved_cursor)
        for keywords, invalid in ((["changed"], saved_cursor), (["KEY", "KEY2"], saved_cursor[:-4] + "AAAA")):
            with self.assertRaises(ToolArgumentError):
                await self.records.search(scope["history_ref"], keywords, invalid)
        second_scope = await self.records.register_scope([batch[-1]["source_ref"]])
        await self.save(history_scopes={scope["history_ref"]: scope, second_scope["history_ref"]: second_scope})
        with self.assertRaises(ToolArgumentError):
            await self.records.search(second_scope["history_ref"], ["KEY", "KEY2"], saved_cursor)

    async def test_cross_session_reference_is_rejected(self):
        batch, scope = await self.seed(["private scoped record"])
        other = RecordStore(self.saver, str(uuid4()))
        with self.assertRaises(ToolExecutionError):
            await other.read(batch[0]["source_ref"], 0)
        with self.assertRaises(ToolExecutionError):
            await other.search(scope["history_ref"], ["private"])

    async def test_rebuild_preserves_references_ranges_and_cursor_without_checkpoints(self):
        batch, scope = await self.seed(["KEY " * 25, "KEY original tail"])
        first = await self.records.search(scope["history_ref"], ["KEY"])
        cursor = first["next_cursor"]
        self.assertIsNotNone(cursor)
        expected = await self.records.search(scope["history_ref"], ["KEY"], cursor)
        view_id = self.saver.confirmed_view(self.sid)
        checkpoint_id = self.saver.last_confirmed(self.sid)
        directory = self.journal.directory.parent
        self.saver.delete_thread(self.sid)
        self.assertEqual(await self.records.records(checkpoint_id=checkpoint_id), batch)
        self.saver.release(self.sid)
        archive = ArchiveStore(directory, create=False)
        self.addCleanup(archive.close)
        rebuilt = RecordStore(archive=archive, view_id=view_id)
        self.addCleanup(rebuilt.close)
        self.assertEqual(await rebuilt.search(scope["history_ref"], ["KEY"], cursor), expected)
        self.assertEqual(await rebuilt.records(), batch)
        self.assertEqual((await rebuilt.read(batch[-1]["source_ref"], 0))["complete"], True)

    async def test_confirmed_summary_is_readable_before_run_seal_without_old_checkpoint(self):
        batch, scope = await self.seed(["original detail"])
        summary_id = str(uuid4())
        await self.save(compaction={"lanes": {"R": {"summary_id": summary_id, "status": "generated",
            "body": "actual summary body", "scope": scope, "previous": [],
            "source_refs": [batch[0]["source_ref"]]}}})
        reference = {"summary_id": summary_id, "lane": "R", "history_ref": scope["history_ref"],
                     "ranges": deepcopy(scope["ranges"]), "checkpoint_id": "unavailable-old-checkpoint"}
        self.saver.delete_thread(self.sid)
        context = ContextRecords(self.saver, self.sid)
        self.assertEqual(await context.summary(reference), "actual summary body")
        changed = deepcopy(reference)
        changed["ranges"][0]["end"] += 1
        with self.assertRaises(CheckpointError):
            await context.summary(changed)

    async def _reject_changed_summary_reference(self, change):
        batch, scope = await self.seed(["original detail"])
        summary_id = str(uuid4())
        reference = {"summary_id": summary_id, "lane": "R", "history_ref": scope["history_ref"],
                     "ranges": deepcopy(scope["ranges"]), "checkpoint_id": "diagnostic-only"}
        active = new_context()
        active.update(run_summary=reference, current_run_id=self.run,
                      summary_spans=[{"index": 0, "lane": "R", "summary_id": summary_id,
                                      "run_id": self.run, "reference": reference}])
        await self.save(active_context=active, compaction={"lanes": {"R": {
            "summary_id": summary_id, "status": "generated", "body": "archived summary",
            "scope": scope, "previous": [], "source_refs": [batch[0]["source_ref"]]}}})
        old_view = self.saver.confirmed_view(self.sid)
        changed = deepcopy(active)
        if change == "ranges":
            changed["run_summary"]["ranges"][0]["end"] += 1
        elif change == "history_ref":
            changed["run_summary"]["history_ref"] = f"history://scope/{uuid4()}"
        elif change == "span_summary_id":
            changed["summary_spans"][0]["summary_id"] = str(uuid4())
        else:
            changed["summary_spans"][0]["lane"] = "H"
        with self.assertRaises(CheckpointError) as rejected:
            await self.save(active_context=changed)
        self.assertEqual(rejected.exception.code, "archive_integrity_error")
        self.assertEqual(self.saver.confirmed_view(self.sid), old_view)
        self.assertEqual(await ContextRecords(self.saver, self.sid).summary(reference), "archived summary")
        with self.assertRaises(ArchiveError):
            self.saver.archive(self.sid).view(self.values["resource_view_id"])

    async def test_summary_reference_changed_ranges_cannot_be_committed(self):
        await self._reject_changed_summary_reference("ranges")

    async def test_summary_reference_changed_history_ref_cannot_be_committed(self):
        await self._reject_changed_summary_reference("history_ref")

    async def test_summary_span_changed_identity_cannot_be_committed(self):
        await self._reject_changed_summary_reference("span_summary_id")

    async def test_summary_span_changed_lane_cannot_be_committed(self):
        await self._reject_changed_summary_reference("span_lane")

    async def test_rebuilt_scope_excludes_unconfirmed_tail_and_later_confirmed_records(self):
        batch, scope = await self.seed(["KEY accepted"])
        old_view = self.saver.confirmed_view(self.sid)
        later = make_record_batch(self.run, 1, [{"role": "user", "content": "KEY later confirmed"}])
        await self.save(record_batch=later)
        tail = make_record_batch(self.run, 2, [{"role": "user", "content": "KEY unconfirmed tail"}])
        self.journal.append(self.run, tail)
        rebuilt = RecordStore(archive=self.saver.archive(self.sid), view_id=old_view)
        self.addCleanup(rebuilt.close)
        self.assertEqual(await rebuilt.records(), batch)
        self.assertEqual([hit["source_ref"] for hit in (await self.records.search(scope["history_ref"], ["KEY"]))["hits"]],
                         [batch[0]["source_ref"]])
        with self.assertRaises(CheckpointError):
            await rebuilt.get(later[0]["source_ref"])
        with self.assertRaises(CheckpointError):
            await self.records.get(tail[0]["source_ref"])

    async def test_cached_rebuilt_reader_detects_tampering_and_missing_body(self):
        batch, scope = await self.seed(["confirmed original"])
        rebuilt = RecordStore(archive=self.saver.archive(self.sid), view_id=self.saver.confirmed_view(self.sid))
        self.addCleanup(rebuilt.close)
        self.assertEqual(await rebuilt.records(), batch)
        path = self.journal.path(self.run)
        original = path.read_bytes()
        path.write_bytes(original.replace(b"original", b"modified"))
        with self.assertRaises(ToolExecutionError):
            await rebuilt.read(batch[0]["source_ref"], 0)
        path.write_bytes(original[:-1])
        with self.assertRaises(ToolExecutionError):
            await rebuilt.search(scope["history_ref"], ["original"])
        path.unlink()
        with self.assertRaises(CheckpointError):
            await rebuilt.records()

    async def test_sealed_run_details_rebuild_without_graph_or_checkpoint_mapping(self):
        archive = self.saver.archive(self.sid)
        first = {"model_call": 1, "result": {"text": "first", "usage": None},
                 "error_code": None, "checkpoint_id": None}
        second = {"model_call": 2, "result": {"text": "second", "usage": {"total_tokens": 7}},
                  "error_code": "model_error", "checkpoint_id": None}
        summary = {"summary_call": 1, "result": {"text": "unused", "usage": None},
                   "compaction": {"status": "failed", "reason": "invalid_summary"}, "checkpoint_id": None}
        for kind, payload in (("model_result", first), ("summary_result", summary), ("model_result", second)):
            archive.append_event(self.run, kind, payload)
        items = [{"role": "user", "content": "request"}, {"role": "assistant", "content": "actual response"}]
        batch = make_record_batch(self.run, 0, items)
        active = new_context()
        active["turns"] = [{"run_id": self.run, "item_count": len(items), "record_end": len(batch)}]
        checkpoint = await self.save(record_batch=batch, input=items, active_context=active,
                                     history_end=len(items), record_admitted_through=len(batch),
                                     model_calls=2, summary_calls=1, usage_totals={"total_tokens": 7},
                                     stop_reason="error", error_code="model_error")
        view_id = self.saver.confirmed_view(self.sid)
        staged = archive.stage_run(self.run, {"final_state": deepcopy(self.values), "fact_end": 3})
        archive.seal_run(staged, view_id, checkpoint)
        archive.append_event(self.run, "model_result", {**second, "model_call": 3})
        directory = archive.directory
        self.saver.release(self.sid)
        rebuilt = ArchiveStore(directory, create=False)
        self.addCleanup(rebuilt.close)
        # These mappings are only diagnostic after the run has been sealed.
        with rebuilt._connection:
            rebuilt._connection.execute("DELETE FROM checkpoints")
        result = await read_archived_record(rebuilt, self.run, view_id)
        self.assertEqual(result["input"], items)
        self.assertEqual([item["model_call"] for item in result["model_results"]], [1, 2])
        self.assertIsNone(result["model_results"][0]["result"]["usage"])
        self.assertEqual(result["summary_results"][0]["compaction"]["status"], "failed")
        self.assertIsNone(result["summary_results"][0]["result"]["usage"])

    async def test_native_history_recall_has_no_artifact_or_file_budget(self):
        batch, scope = await self.seed(["small original detail " * 400])
        class Delegate:
            read_budget = {"limit_bytes": 20, "used_bytes": 20, "remaining_bytes": 0}
            async def __call__(self, name, arguments):
                raise AssertionError("history must not enter file executor")
        executor = HistoryExecutor(self.records, Delegate())
        calls = [{"type": "function_call", "name": "history_read", "call_id": "r1",
                  "arguments": json.dumps({"source_ref": batch[0]["source_ref"], "start": 0, "end": None})},
                 {"type": "function_call", "name": "history_search", "call_id": "r2",
                  "arguments": json.dumps({"history_ref": scope["history_ref"], "keywords": ["original"], "cursor": None})}]
        result = await execute_calls(calls, HISTORY_TOOLS, executor,
                                     observation_manager=ObservationManager(_NoArtifacts()), native_recall_tools=HISTORY_TOOL_NAMES)
        self.assertEqual(result["tool_outcomes"]["succeeded"], 2)
        for item in result["output_items"]:
            self.assertLessEqual(len(item["output"].encode("utf-8")), 8192)
            self.assertEqual(json.loads(item["output"])["observation"]["mode"], "inline")
        self.assertEqual(executor.read_budget["remaining_bytes"], 0)
        with self.assertRaises(ToolArgumentError):
            await executor("history_read", {"source_ref": batch[0]["source_ref"], "start": 0, "end": None, "session_id": self.sid})

    async def test_opaque_projection_omits_encrypted_and_unknown_content(self):
        batch = make_record_batch(self.run, 0, [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "visible"}], "encrypted_content": "SECRET"},
            {"type": "image_generation_call", "result": "SECRET IMAGE"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": {"encrypted_content": "SECRET"}}]},
        ], kind="model", group_id=1)
        for record in batch:
            text, omitted = record_projection(record)
            self.assertNotIn("SECRET", text)
            self.assertTrue(omitted)
        self.assertIn("visible", record_projection(batch[0])[0])
        self.assertIn("SECRET", str(batch[0]["item"]))

    async def test_tool_facts_distinguish_execution_and_observation_failure(self):
        class FailingObservation:
            def observe(self, *args, **kwargs):
                return {"ok": True, "observation": {"status": "unavailable", "stage": "save", "error": {"code": "observation_save_error"}}}
        async def executor(name, args):
            return "executed successfully"
        calls = [{"type": "function_call", "name": "tool", "call_id": "c1", "arguments": "{}"},
                 {"type": "function_call", "name": "missing", "call_id": "c2", "arguments": "{}"}]
        result = await execute_calls(calls, [{"type": "function", "name": "tool", "parameters": {}}], executor,
                                     observation_manager=FailingObservation())
        self.assertEqual([fact["outcome"] for fact in result["call_facts"]], ["succeeded", "rejected"])
        self.assertEqual([fact["executor_entered"] for fact in result["call_facts"]], [True, False])
        records = make_record_batch(self.run, 0, result["output_items"], kind="tool", group_id=1, call_facts=result["call_facts"])
        self.assertEqual(records[0]["execution"]["outcome"], "succeeded")
        self.assertNotIn("output", records[0]["execution"])

    async def test_interruption_keeps_actual_facts_without_fabricating_results(self):
        async def executor(name, args):
            if args["number"] == 1:
                return "first completed"
            raise asyncio.CancelledError()
        calls = [{"type": "function_call", "name": "tool", "call_id": f"c{index}", "arguments": json.dumps({"number": index})} for index in range(1, 4)]
        try:
            await execute_calls(calls, [{"type": "function", "name": "tool", "parameters": {}}], executor,
                                observation_manager=ObservationManager(_NoArtifacts()))
        except asyncio.CancelledError as error:
            batch = error.tool_batch
        else:
            self.fail("interruption must propagate")
        self.assertEqual(len(batch["output_items"]), 1)
        self.assertEqual([fact["outcome"] for fact in batch["call_facts"]], ["succeeded", "interrupted", "skipped"])
        records = make_record_batch(self.run, 0, batch["output_items"], kind="tool", group_id=1, call_facts=batch["call_facts"])
        self.assertEqual(len(records), 3)
        self.assertIsNone(records[1]["item"])
        self.assertIsNone(records[2]["item"])


if __name__ == "__main__":
    unittest.main()
