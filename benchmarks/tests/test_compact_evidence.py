"""Verify Compact evidence bindings without treating answers as semantic proof."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from benchmarks.core.compact_evidence import covered, history_coverage, native_binding, turn_compact_evidence
from benchmarks.core.evidence import digest
from lineage_agent.agent.record_journal import RecordJournal
from lineage_agent.agent.records import make_record_batch, record_projection
from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.requests import prepare_request
from lineage_agent.models.summarization import prepare_summary_request, summary_budget_settings
from lineage_agent.models.budget import check_context_budget
from lineage_agent.models.summary_input import INPUT_FORMAT, record_details, serialize_conversation
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
        self.journal = RecordJournal(self.output / "history")
        self.addCleanup(self.journal.close)
        self.original_records = make_record_batch("run1", 0, [{"role": "user", "content": "read the chain"}, *self.original])
        _, self.watermark = self.journal.append("run1", self.original_records)
        self.items = [
            {"role": "user", "content": "Current Run prefix; first original input is read-only background."},
            *[record["item"] for record in self.original_records]]
        self.details = [{"kind": "scope"}, *[record_details(record) for record in self.original_records]]
        rendered = serialize_conversation(self.items, item_details=self.details)
        self.summary = prepare_summary_request(self.settings, rendered, lane="R")
        projection = record_projection(self.original_records[2])[0]
        self.recall = pair("history_search", "search", {"history_ref": "scope", "keywords": ["37"], "cursor": None},
                           {"hits": [{"source_ref": self.source_ref}]})
        self.recall += pair("history_read", "read", {"source_ref": self.source_ref, "start": 0, "end": None},
                            {"source_ref": self.source_ref, "text": projection, "start": 0, "end": len(projection)})
        self.observer = SimpleNamespace(output_dir=self.output, requests={1: [
            prepare_request("test", self.original), self.summary, prepare_request("test", self.recall)]},
            response_outputs=[{"turn": 1, "purpose": "business", "output": [self.original[0], self.recall[0], self.recall[2]]}])
        self.metadata = {"lane": "R", "request_no": 5, "attempt_no": 1,
            "snapshot": {"record_files": {"run1": self.watermark}},
            "summary_input_metadata": {"item_refs": [None, *[record["source_ref"] for record in self.original_records]],
                                       "context_risks": [None], "derived_spans": [], "item_details": self.details,
                                       "format": INPUT_FORMAT, "source_spans": rendered.source_spans},
            "lanes": {"R": {"source_refs": [record["source_ref"] for record in self.original_records[1:]],
                "previous": [], "source_files": [{"run_id": "run1", "path": str(self.journal.path("run1")),
                                                  "start": 1, "end": 3}],
                "scope": {"ranges": [{"run_id": "run1", "start": 1, "end": 3}]},
                "summary_request_sha256": digest(self.summary),
                "summary_budget": check_context_budget(self.summary, [None] * len(self.summary["input"]),
                                                       summary_budget_settings(self.settings, "R"))}}}
        self.observer.summary_contexts = {digest(self.summary): self.metadata}
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

    def test_native_sources_match_file_and_sidecar_without_transport_wrappers(self):
        before = self.journal.path("run1").read_bytes()
        binding = native_binding(self.observer, self.summary)
        self.assertTrue(binding["verified"])
        self.assertEqual(binding["input_records"], self.original_records)
        self.assertEqual(len(self.summary["input"]), 1)
        self.assertIn(self.text, self.summary["input"][0]["content"])
        self.assertEqual(self.journal.path("run1").read_bytes(), before)
        _, path = turn_compact_evidence(self.observer, self.state, 1, self.settings)
        evidence = json.loads(path.read_text(encoding="utf-8"))["summary_requests"][0]
        self.assertNotIn("material", evidence)
        self.assertEqual(evidence["input"], self.summary["input"])
        self.assertTrue(evidence["source_binding"]["verified"])
        self.assertTrue(evidence["budget_matches_preflight"])

    def test_missing_or_corrupt_sidecar_cannot_manufacture_source_proof(self):
        for defect in ("missing", "request_hash", "reference", "watermark"):
            with self.subTest(defect=defect):
                observer = deepcopy(self.observer)
                metadata = observer.summary_contexts[digest(self.summary)]
                if defect == "missing":
                    observer.summary_contexts.clear()
                elif defect == "request_hash":
                    metadata["lanes"]["R"]["summary_request_sha256"] = "wrong"
                elif defect == "reference":
                    metadata["summary_input_metadata"]["item_refs"][-1] = "history://record/foreign/2"
                else:
                    metadata["snapshot"]["record_files"]["run1"]["prefix_sha256"] = "0" * 64
                self.assertFalse(native_binding(observer, self.summary)["verified"])

    def test_changed_native_output_or_file_cannot_verify(self):
        request = deepcopy(self.summary)
        request["input"][0]["content"] = request["input"][0]["content"].replace(self.text, "changed result")
        observer = deepcopy(self.observer)
        observer.summary_contexts[digest(request)] = deepcopy(self.metadata)
        observer.summary_contexts[digest(request)]["lanes"]["R"]["summary_request_sha256"] = digest(request)
        self.assertFalse(native_binding(observer, request)["verified"])
        path = self.journal.path("run1")
        body = path.read_bytes()
        path.write_bytes(body.replace(b'read the chain', b'read the other', 1))
        self.assertFalse(native_binding(self.observer, self.summary)["verified"])

    def test_native_order_and_readonly_background_cannot_lose_their_bindings(self):
        for defect in ("reordered", "unbound_readonly"):
            with self.subTest(defect=defect):
                request = deepcopy(self.summary)
                metadata = deepcopy(self.metadata)
                if defect == "reordered":
                    pass  # The source order changed but the actual rendered request did not.
                    metadata["summary_input_metadata"]["item_refs"][2:] = reversed(
                        metadata["summary_input_metadata"]["item_refs"][2:])
                else:
                    metadata["summary_input_metadata"]["item_refs"][1] = None
                metadata["lanes"]["R"]["summary_request_sha256"] = digest(request)
                observer = deepcopy(self.observer)
                observer.summary_contexts[digest(request)] = metadata
                self.assertFalse(native_binding(observer, request)["verified"])

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

    def test_unsent_terminal_recall_result_is_counted_with_honest_provenance(self):
        self.observer.requests[1].pop()
        self.state["input"] = self.recall
        metrics, path = turn_compact_evidence(self.observer, self.state, 1, self.settings)
        self.assertEqual(metrics["recall_calls"]["value"]["history_search"], 1)
        self.assertEqual(metrics["recall_calls"]["value"]["history_read"], 1)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["recalls"]["history_read"][0]["evidence_source"], "final_state_input")
        coverage = history_coverage(self.observer, [self.state], self.mechanism)
        for name in ("history_search_used", "history_read_used", "history_result_matches_original", "target_original_removed"):
            self.assertEqual(next(check["status"] for check in coverage if check["name"] == name), "verified")

    def test_schema_overhead_is_measured_without_inventing_opaque_risks(self):
        from lineage_agent.tools.history import HISTORY_TOOLS
        self.observer.requests[1] = [prepare_request("test", [{"type": "reasoning", "id": "r1",
            "encrypted_content": "opaque", "summary": []}], tools=HISTORY_TOOLS)]
        metrics, _ = turn_compact_evidence(self.observer, self.state, 1, self.settings)
        self.assertGreater(metrics["history_tool_fixed_budget_overhead"]["value"], 0)
        self.assertEqual(metrics["history_tool_fixed_budget_overhead_basis"]["value"],
                         "actual request fields with identical empty input")


if __name__ == "__main__":
    unittest.main()
