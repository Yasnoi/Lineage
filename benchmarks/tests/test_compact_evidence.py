"""Verify Compact evidence bindings without treating answers as semantic proof."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from benchmarks.core.compact_evidence import covered, history_coverage, turn_compact_evidence
from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.requests import prepare_request
from lineage_agent.models.summarization import prepare_summary_request
from lineage_agent.tools.types import inline_tool_output, serialize_tool_output


def pair(name, call_id, arguments, raw):
    return [{"type": "function_call", "call_id": call_id, "name": name,
             "arguments": json.dumps(arguments)},
            {"type": "function_call_output", "call_id": call_id,
             "output": serialize_tool_output(inline_tool_output(serialize_tool_output(raw)))}]


class CompactEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.settings = ModelSettings(model_name="test", base_url="https://example.invalid")
        self.source_ref = "history://record/run1/2"
        self.path = "evaluation-fixtures/02.txt"
        self.text = "额度37，不得自动放行；修正旧额度137。"
        self.original = pair("read_text_file", "file-2", {"path": self.path}, {"path": self.path, "content": self.text})
        self.summary = prepare_summary_request(self.settings, {"sources": [
            {"source_id": self.source_ref, "payload": json.dumps({"item": self.original[1]}, ensure_ascii=False)}]},
            target_bytes=2048, lane="R")
        projection = json.dumps({"item": self.original[1]}, ensure_ascii=False)
        self.recall = pair("history_search", "search", {"history_ref": "scope", "keywords": ["37"], "cursor": None},
                           {"hits": [{"source_ref": self.source_ref}]})
        self.recall += pair("history_read", "read", {"source_ref": self.source_ref, "start": 0, "end": None},
                            {"source_ref": self.source_ref, "text": projection, "start": 0, "end": len(projection)})
        self.observer = SimpleNamespace(output_dir=self.output, requests={1: [
            prepare_request("test", self.original), self.summary, prepare_request("test", self.recall)]},
            response_outputs=[{"turn": 1, "purpose": "business", "output": [self.original[0], self.recall[0], self.recall[2]]}])
        self.event = {"status": "used", "request_no": 5, "attempt_no": 1, "lanes": {"R": {"summary_id": "s"}},
                      "history_scopes": [{"ranges": [{"run_id": "run1", "start": 0, "end": 3}]}]}
        self.state = {"run_id": "run1", "compaction_events": [self.event]}
        self.mechanism = {"target_path": self.path, "source_text": self.text, "chain_paths": [self.path], "record_key": "37"}

    def test_half_open_scope_and_foreign_reference(self):
        bounds = [{"run_id": "run1", "start": 2, "end": 3}]
        self.assertTrue(covered(self.source_ref, bounds))
        for ref in ("history://record/run1/3", "history://record/foreign/2", "bad", None):
            self.assertFalse(covered(ref, bounds))

    def test_recall_binds_original_coverage_and_transport_without_scoring_answer(self):
        checks = history_coverage(self.observer, [self.state], self.mechanism)
        self.assertTrue(all(check["status"] == "verified" for check in checks), checks)
        saved = json.loads((self.output / "mechanism.json").read_text())
        self.assertEqual(saved["summary_omission_condition"], "requires_semantic_review")
        self.assertNotIn("answer_correct", saved["checks"])
        self.state["compaction_events"][0]["status"] = "candidate"
        checks = history_coverage(self.observer, [self.state], self.mechanism)
        self.assertEqual(next(check["status"] for check in checks if check["name"] == "target_record_covered"), "gap")

    def test_foreign_read_or_missing_terminal_result_cannot_verify(self):
        self.observer.requests[1][-1]["input"][2]["arguments"] = json.dumps({"source_ref": "history://record/other/2"})
        checks = history_coverage(self.observer, [self.state], self.mechanism)
        self.assertEqual(next(check["status"] for check in checks if check["name"] == "history_read_used"), "gap")

    def test_structured_source_payload_matches_same_original_projection(self):
        source_material = json.loads(self.summary["input"][0]["content"])
        source = source_material["sources"][0]
        source["payload"] = json.loads(source["payload"])
        self.summary["input"][0]["content"] = json.dumps(source_material, ensure_ascii=False)
        checks = history_coverage(self.observer, [self.state], self.mechanism)
        self.assertTrue(all(check["status"] == "verified" for check in checks), checks)

    def test_metrics_keep_every_attempt_and_only_current_turn_recalls(self):
        second = deepcopy(self.event)
        second.update(attempt_no=2, status="stopped")
        self.state["compaction_events"].append(second)
        metrics, path = turn_compact_evidence(self.observer, self.state, 1, self.settings)
        self.assertEqual(len(metrics["compact_events"]["value"]), 2)
        self.assertEqual(metrics["summary_lanes"]["value"], {"H": 0, "R": 1})
        self.assertEqual(metrics["recall_calls"]["value"]["history_read"], 1)
        self.assertTrue(path.exists())
        self.observer.requests[2] = [prepare_request("test", self.recall)]
        metrics, _ = turn_compact_evidence(self.observer, {"run_id": "run2"}, 2, self.settings)
        self.assertEqual(metrics["recall_calls"]["value"]["history_read"], 0)

    def test_protocol_shapes_preserve_nulls_and_extensions_without_bodies(self):
        self.observer.response_outputs = [{"turn": 1, "purpose": "business", "request_id": "request",
            "output": [{"type": "message", "phase": "commentary", "status": "completed",
                        "content": [{"type": "output_text", "text": "PRIVATE_MESSAGE"}]},
                       {"type": "reasoning", "content": None, "summary": [],
                        "encrypted_content": "PRIVATE_ENCRYPTED", "extension": "PRIVATE_EXTENSION"}]}]
        _, path = turn_compact_evidence(self.observer, self.state, 1, self.settings)
        value = json.loads(path.read_text(encoding="utf-8"))
        shapes = value["business_protocol_shapes"][0]["output"]
        self.assertEqual(shapes[0]["phase"], "commentary")
        self.assertEqual(shapes[0]["content_parts"], [{"keys": ["text", "type"], "type": "output_text"}])
        self.assertEqual(shapes[1]["field_types"]["content"], "NoneType")
        self.assertIn("extension", shapes[1]["keys"])
        self.assertNotIn("PRIVATE_", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
