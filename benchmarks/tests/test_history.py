"""Original records, immutable recall ranges, and bounded native history tools."""

import asyncio
from copy import deepcopy
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.base import empty_checkpoint

from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.records import RecordStore, make_record_batch, record_projection
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
        self.records = RecordStore(self.saver, self.sid)
        self.config = {"configurable": {"thread_id": self.sid, "checkpoint_ns": ""}}
        self.values = {"run_id": self.run, "record_batch": [], "history_scopes": {}}
        self.version = 0

    async def save(self, **updates):
        self.version += 1
        self.values.update(deepcopy(updates))
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
        await self.save(record_batch=changed)
        with self.assertRaises(CheckpointError):
            await self.records.records()
        self.saver.release(self.sid)
        self.records.clear_index()
        self.config = {"configurable": {"thread_id": self.sid, "checkpoint_ns": ""}}
        await self.save(record_batch=make_record_batch(self.run, 1, [{"role": "user", "content": "gap"}]), history_scopes={})
        with self.assertRaises(CheckpointError):
            await self.records.records()

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
        await self.save(history_scopes={scope["history_ref"]: altered})
        with self.assertRaises(CheckpointError):
            await self.records.scope(scope["history_ref"])

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
