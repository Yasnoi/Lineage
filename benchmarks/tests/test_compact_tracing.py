"""App-owned summary positions survive safe binding without leaking bodies."""

from copy import deepcopy
import json
from typing import get_args
import unittest
from uuid import uuid4

from benchmarks.checks.support.runtime import message, response
from benchmarks.checks.support.tracing import SETTINGS, capture
from lineage_agent.agent.state import CompactionStatus
from lineage_agent.entrypoints.cli import format_run_summary
from lineage_agent.models.responses import parse_response
from lineage_agent.recording.trace_data import TraceSanitizer
from lineage_agent.recording.tracing import start_trace


PRIVATE = "PRIVATE_DERIVED_BODY_9720"


def span(index, *, lane="R"):
    run_id, summary_id, checkpoint_id = (str(uuid4()) for _ in range(3))
    return {"index": index, "lane": lane, "summary_id": summary_id, "run_id": run_id,
            "reference": {"summary_id": summary_id, "checkpoint_id": checkpoint_id,
                          "lane": lane, "history_ref": "history://scope/" + str(uuid4()),
                          "ranges": [{"run_id": run_id, "start": 0, "end": 10}], "document": PRIVATE},
            "message": PRIVATE}


class CompactTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_more_than_lineage_cap_is_fully_masked_after_runtrace_binding(self):
        active = {"version_id": str(uuid4()), "summary_spans": [span(index) for index in range(40)]}
        items = [{"role": "assistant", "content": PRIVATE + str(index)} for index in range(40)]
        items.append({"role": "user", "content": "Retain this original user question, including NOT 13."})
        original = deepcopy((active, items))
        with capture() as (transport, stderr):
            tracer = start_trace(str(uuid4()), SETTINGS)
            tracer.bind_active_context(active)
            token = tracer.model_start("offline", items, [])
            tracer.model_end(token, result=parse_response(response(message("answer"))))
            tracer.finish()
            await tracer.close()
        self.assertFalse(stderr.getvalue())
        wire = json.dumps(transport.sent)
        self.assertNotIn(PRIVATE, wire)
        self.assertIn("NOT 13", wire)
        self.assertEqual((active, items), original)
        business = next(item for item in transport.sent if item["name"] == "OpenAI Responses")
        self.assertEqual(business["inputs"]["summary_items_omitted"], 40)
        published = business["inputs"]["active_context"]
        self.assertEqual(len(published["summary_spans"]), 32)
        self.assertEqual(published["summary_span_count"], 40)
        self.assertEqual(published["summary_spans_omitted"], 8)

    def test_ordinary_marker_and_bad_positions_do_not_hide_messages(self):
        sanitizer = TraceSanitizer([], None)
        marker = '{"kind":"derived_context_summary","lane":"R","document":"ordinary"}'
        items = [{"role": "assistant", "content": marker}, {"role": "user", "content": "user stays"}]
        for active in ({}, {"summary": {"summary_id": str(uuid4())}},
                       {"summary_spans": [span(-1), span(True), span(100), {"index": 0, "lane": "R"}]}):
            with self.subTest(active=active):
                projected = sanitizer.model_inputs("offline", items, [], active_context=active)
                self.assertIn("ordinary", json.dumps(projected))
                self.assertIn("user stays", json.dumps(projected))
                self.assertNotIn("summary_content_omitted", projected)
        projected = sanitizer.model_inputs("offline", items, [], active_context={"summary_spans": [span(1)]})
        self.assertIn("user stays", json.dumps(projected))
        self.assertNotIn("summary_content_omitted", projected)

    def test_lane_metrics_keep_safe_failures_without_derived_content(self):
        metadata = {"status": "stopped", "reason": "recovery_margin_insufficient", "attempt_no": 2,
                    "original_required": 1000, "candidate_required": 1100, "net_reclaimed": -100,
                    "recovery_margin": -100, "snapshot": PRIVATE, "document": PRIVATE,
                    "recovery_budget": {"base_required": 500, "hard_capacity": 300, "soft_target": 100,
                                        "margin_required": 100, "target_bytes": 300, "private": PRIVATE},
                    "lanes": {lane: {"summary_id": str(uuid4()), "status": status, "model_called": called,
                                     "target_bytes": 120, "summary_bytes": 150, "summary_call": 2,
                                     "material": PRIVATE, "document": PRIVATE}
                              for lane, status, called in (("H", "reused", False), ("R", "generated", True))}}
        projected = TraceSanitizer.compaction(metadata)
        self.assertEqual(projected["net_reclaimed"], -100)
        self.assertEqual(projected["recovery_margin"], -100)
        self.assertEqual(projected["attempt_no"], 2)
        self.assertEqual(projected["reason"], "recovery_margin_insufficient")
        self.assertFalse(projected["lanes"]["H"]["model_called"])
        self.assertTrue(projected["lanes"]["R"]["model_called"])
        self.assertEqual(projected["recovery_budget"]["hard_capacity"], 300)
        self.assertNotIn(PRIVATE, json.dumps(projected))

    def test_all_compaction_statuses_have_safe_cli_labels(self):
        state = {"run_id": str(uuid4()), "tool_outcomes": {"succeeded": 0, "rejected": 0, "failed": 0, "skipped": 0},
                 "log_complete": True, "log_path": "test.log", "model_calls": 1, "summary_calls": 2,
                 "active_context": {"version_id": str(uuid4())}}
        for status in get_args(CompactionStatus):
            with self.subTest(status=status):
                rendered = format_run_summary({**state, "compaction": {
                    "status": status, "attempt_no": 2, "document": PRIVATE,
                    "lanes": {"H": {"status": "reused"}, "R": {"status": "generated"}},
                    "net_reclaimed": 123, "recovery_margin": 456,
                }})
                self.assertNotIn("上下文压缩：未确认", rendered)
                self.assertNotIn(PRIVATE, rendered)
                self.assertIn("第 2 次尝试", rendered)
                self.assertIn("H：复用", rendered)
                self.assertIn("R：已生成", rendered)


if __name__ == "__main__":
    unittest.main()
