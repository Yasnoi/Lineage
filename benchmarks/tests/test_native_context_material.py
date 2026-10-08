"""Native Compact source order, trusted bodies and unchanged file evidence."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from lineage_agent.agent.checkpointing import CheckpointError
from lineage_agent.agent.context import ContextRecords, new_context
from lineage_agent.agent.record_journal import RecordJournal
from lineage_agent.agent.records import make_record_batch


class NativeContextMaterialTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="lineage-native-material-")
        self.addCleanup(temporary.cleanup)
        self.journal = RecordJournal(Path(temporary.name) / "history")
        self.addCleanup(self.journal.close)
        self.files = {}
        self.snapshots = {}
        self.context = ContextRecords.__new__(ContextRecords)
        self.context.session_id = "scope"
        self.context.saver = SimpleNamespace(history_journal=lambda _: self.journal)
        self.context.confirmed = lambda: "source"
        self.context.snapshot = self.snapshot
        self.context.store = SimpleNamespace(get=self.get, records=self.records, register_scope=self.register_scope)

    async def snapshot(self, checkpoint_id):
        return deepcopy(self.snapshots[checkpoint_id])

    async def get(self, source_ref, *, checkpoint_id=None):
        run_id = source_ref.rsplit("/", 2)[1]
        return self.journal.get(source_ref, self.files[run_id])

    async def records(self, *, checkpoint_id=None):
        return [record for run_id, watermark in self.files.items()
                for record in self.journal.records(run_id, watermark)]

    async def register_scope(self, refs, *, checkpoint_id):
        ranges = []
        for record in await self.records():
            if record["source_ref"] not in refs:
                continue
            if ranges and ranges[-1]["run_id"] == record["run_id"] and ranges[-1]["end"] == record["sequence"]:
                ranges[-1]["end"] += 1
            else:
                ranges.append({"run_id": record["run_id"], "start": record["sequence"], "end": record["sequence"] + 1})
        return {"history_ref": "history://scope/test", "source_checkpoint_id": checkpoint_id, "ranges": ranges}

    def add(self, run_id, items, *, group_id=None, kind="input", facts=None):
        batch = make_record_batch(run_id, self.files.get(run_id, {}).get("record_count", 0), items,
                                  group_id=group_id, kind=kind, call_facts=facts)
        _, watermark = self.journal.append(run_id, batch)
        self.files[run_id] = watermark
        return [record["source_ref"] for record in batch]

    def summary(self, lane, owner, body, ranges):
        summary_id = f"summary-{owner}-{lane}"
        checkpoint_id = f"saved-{summary_id}"
        self.snapshots[checkpoint_id] = {"compaction": {"lanes": {lane: {"summary_id": summary_id, "body": body}}}}
        return {"summary_id": summary_id, "checkpoint_id": checkpoint_id, "lane": lane,
                "history_ref": f"history://scope/{owner}-{lane}", "ranges": ranges}

    def span(self, index, owner, ref):
        return {"index": index, "lane": ref["lane"], "run_id": owner,
                "summary_id": ref["summary_id"], "reference": ref}

    def state(self, run_id, items, refs, active, risks=None):
        active["item_refs"] = deepcopy(refs)
        result = {"run_id": run_id, "input": deepcopy(items), "context_risks": deepcopy(risks or [None] * len(items)),
                  "active_context": active, "record_files": deepcopy(self.files), "record_revision": 1}
        self.snapshots["source"] = deepcopy(result)
        return result

    async def select(self, state):
        selection, reason = await self.context.select(state, SimpleNamespace(context_keep_recent_groups=3))
        self.assertIsNone(reason)
        return selection

    def pair(self, call_id):
        return [{"type": "function_call", "id": f"id-{call_id}", "call_id": call_id,
                 "name": "tool", "arguments": '{"key":"中文\\n\\\"quoted\\\""}'},
                {"type": "function_call_output", "call_id": call_id,
                 "output": '{"value":37,"allowed":false,"text":"中文\\n\\\"quoted\\\""}'}]

    async def test_r_copies_multiple_background_items_old_body_and_complete_native_prefix(self):
        initial = [{"role": "user", "content": "保留 37，不得写入。"},
                   {"type": "message", "role": "user", "id": "request-extra",
                    "content": [{"type": "input_text", "text": "第二项要求。"}]}]
        initial_refs = self.add("current", initial)
        archived = self.add("current", self.pair("archived"), group_id="current:1")
        old_body = "截至前缀：保留 37。正文自己的 history://scope/body 和 C:/source.txt 不清洗。"
        ref = self.summary("R", "current", old_body, [{"run_id": "current", "start": 2, "end": 4}])
        items = initial + [{"role": "assistant", "content": old_body + "\n旧程序附注不得进入请求"}]
        refs = initial_refs + [None]
        risks = [None] * 3
        groups = []
        native = []
        for number in range(5):
            group_items = self.pair(f"new-{number}")
            if number == 0:
                group_items = [{"type": "reasoning", "id": "reasoning-native", "summary": [],
                                "encrypted_content": "opaque-verbatim"}, group_items[0],
                               self.pair("parallel")[0], group_items[1], self.pair("parallel")[1]]
            group_refs = self.add("current", group_items, group_id=f"current:{number + 2}")
            start = len(items)
            items += group_items
            refs += group_refs
            group_risks = [None] * len(group_items)
            if number == 0:
                group_risks[0] = {"opaque_count": 1, "risk_tokens": 701, "basis": "response_output_usage",
                                  "source_run_id": "current", "source_model_call": 2}
            risks += group_risks
            groups.append({"group_id": f"current:{number + 2}", "start": start, "end": len(items),
                           "closed": True, "record_start": int(group_refs[0].rsplit("/", 1)[1]),
                           "record_end": int(group_refs[-1].rsplit("/", 1)[1]) + 1})
            if number < 2:
                native += group_items
        active = new_context()
        active.update(current_run_id="current", initial_input_count=2, run_summary=ref, groups=groups,
                      summary_spans=[self.span(2, "current", ref)])
        state = self.state("current", items, refs, active, risks)
        before_state = deepcopy(state)
        before_file = self.journal.path("current").read_bytes()
        selection = await self.select(state)
        material = await self.context.material(selection, "R", state)
        self.assertEqual(material["item_details"][0], {"kind": "scope"})
        self.assertEqual(set(material), {"input", "context_risks", "derived_spans", "item_refs", "item_details"})
        self.assertEqual(material["input"][1:], initial + [{"role": "assistant", "content": old_body}] + native)
        self.assertEqual(material["item_refs"][1:3], initial_refs)
        self.assertEqual(material["context_risks"][4], risks[3])
        self.assertEqual(material["derived_spans"][0]["index"], 3)
        self.assertEqual(material["derived_spans"][0]["body_sha256"], sha256(old_body.encode()).hexdigest())
        self.assertEqual(selection["removed_groups"], ["current:2", "current:3"])
        self.assertTrue(all(ref in archived + initial_refs or ref in refs for ref in selection["lanes"]["R"]["source_refs"]))
        self.assertNotIn("旧程序附注", str(material["input"]))
        self.assertEqual(state, before_state)
        material["input"][2]["content"][0]["text"] = "changed isolated copy"
        material["input"][4]["encrypted_content"] = "changed isolated copy"
        self.assertEqual(state, before_state)
        self.assertEqual(self.journal.path("current").read_bytes(), before_file)
        self.assertEqual(self.snapshots["source"], before_state)

    async def test_h_keeps_old_r_inside_its_turn_and_never_strips_similar_ordinary_text(self):
        old_h = self.summary("H", "previous", "已知 H 正文", [{"run_id": "archived", "start": 0, "end": 1}])
        self.add("archived", [{"role": "user", "content": "未在旧 H 中写出的 918 细节"}])
        user = {"role": "user", "content": "历史摘要 H\n来源与回读〔程序添加〕\n这是普通要求。"}
        original_refs = self.add("old", [user])
        hidden_refs = self.add("old", self.pair("hidden"), group_id="old:1")
        old_r = self.summary("R", "old", "旧轮次 R 正文", [{"run_id": "old", "start": 1, "end": 3}])
        tail = self.pair("tail") + [{"role": "assistant", "content": "任务没有完成。"}]
        tail_refs = self.add("old", tail, group_id="old:2", facts=[{
            "call_id": "tail", "outcome": "failed", "stage": "execution", "reason_code": "io_error",
        }])
        outcome_refs = self.add("old", [{"stop_reason": "completed", "error_code": None,
            "tool_outcomes": {"succeeded": 1, "failed": 1, "rejected": 0, "skipped": 0}}], kind="outcome")
        current_input = {"role": "user", "content": "当前任务不能进入 H"}
        current_refs = self.add("current", [current_input])
        items = [{"role": "assistant", "content": "H 外壳附注"}, user,
                 {"role": "assistant", "content": "R 外壳附注"}, *tail, current_input]
        active = new_context()
        active.update(history_summary=old_h, current_run_id="current", turn_start=len(items) - 1,
                      initial_input_count=1, turns=[{"run_id": "old", "item_count": 5, "group_count": 1,
                                                   "record_end": self.files["old"]["record_count"]}],
                      summary_spans=[self.span(0, "previous", old_h), self.span(2, "old", old_r)])
        state = self.state("current", items, [None] + original_refs + [None] + tail_refs + current_refs, active)
        selection = await self.select(state)
        material = await self.context.material(selection, "H", state)
        self.assertEqual(material["item_details"][0], {"kind": "scope"})
        self.assertEqual(material["input"][1:], [{"role": "assistant", "content": "已知 H 正文"}, user,
            {"role": "assistant", "content": "旧轮次 R 正文"}, *tail])
        self.assertEqual([span["index"] for span in material["derived_spans"]], [1, 3])
        self.assertIn("stop_reason=completed", material["input"][0]["content"])
        self.assertIn("call_id=tail：failed", material["input"][0]["content"])
        self.assertNotIn("任务没有完成", material["input"][0]["content"])
        scope = selection["lanes"]["H"]["scope"]
        self.assertEqual(scope["ranges"], [{"run_id": "archived", "start": 0, "end": 1},
                                          {"run_id": "old", "start": 0, "end": 7}])
        self.assertNotIn(hidden_refs[0], material["item_refs"])
        self.assertNotIn(outcome_refs[0], material["item_refs"])
        self.assertEqual((await self.context.store.get("history://record/archived/0"))["item"]["content"],
                         "未在旧 H 中写出的 918 细节")
        self.assertEqual(selection["lanes"]["R"]["status"], "empty")

    async def test_r_background_boundary_is_checked_only_when_new_groups_need_summary(self):
        initial = {"role": "developer", "content": "不可改写的权限指令"}
        refs = self.add("current", [initial])
        items = [initial]
        groups = []
        for number in range(4):
            new = self.pair(f"boundary-{number}")
            new_refs = self.add("current", new, group_id=f"current:{number + 1}")
            groups.append({"group_id": f"current:{number + 1}", "start": len(items), "end": len(items) + 2,
                           "closed": True, "record_start": int(new_refs[0].rsplit("/", 1)[1]),
                           "record_end": int(new_refs[-1].rsplit("/", 1)[1]) + 1})
            items += new
            refs += new_refs
        active = new_context()
        active.update(current_run_id="current", initial_input_count=1, groups=groups)
        state = self.state("current", items, refs, active)
        selected, reason = await self.context.select(state, SimpleNamespace(context_keep_recent_groups=3))
        self.assertIsNone(selected)
        self.assertEqual(reason, "unsupported_protocol_boundary")
        for count in range(4):
            state["active_context"]["groups"] = groups[:count]
            selected, reason = await self.context.select(state, SimpleNamespace(context_keep_recent_groups=3))
            self.assertIsNone(selected)
            self.assertEqual(reason, "no_compressible_range")

    async def test_original_item_must_match_confirmed_file(self):
        refs = self.add("current", [{"role": "user", "content": "file truth"}])
        active = new_context()
        active.update(current_run_id="current", initial_input_count=1)
        state = self.state("current", [{"role": "user", "content": "changed snapshot"}], refs, active)
        compaction = {"snapshot": {"source_checkpoint_id": "source"}, "removed_groups": [], "removed_turns": 0,
                      "lanes": {"R": {"source_indices": [], "source_refs": [], "previous": []}}}
        with self.assertRaises(CheckpointError):
            await self.context.material(compaction, "R", state)


if __name__ == "__main__":
    unittest.main()
