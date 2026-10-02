"""Offline packet failures preserve frozen results and existing reviewer work."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.core import runner
from benchmarks.core.contracts import known, summarize
from benchmarks.core.files import atomic_write_json, sha256_file
from benchmarks.evaluators import review


def review_fixture(directory: Path) -> dict:
    answer = directory / "answer.md"
    answer.write_text("Offline fixture answer with source support.\n", encoding="utf-8")
    rubric = {
        "version": "fixture-rubric", "dimensions": [{"id": "facts", "description": "事实"}],
        "cases": {"fixture": {"conditions": [{"id": "source-support", "dimension": "facts",
                    "description": "来源支持回答", "required": True}], "severe_errors": []}},
    }
    evidence = {"kind": "answer", "path": str(answer), "source": "offline fixture", "availability": "available"}
    unit = {
        "suite_id": "fixture", "unit_id": "fixture.r1", "case_id": "fixture", "repeat": 1,
        "category": "research", "definition_version": "fixture-v1", "granularity": "task",
        "execution_mode": "real_model", "execution_status": "completed", "verdict": "pending_review",
        "conditions": {**{key: known({"version": "fixture-only"}) for key in ("model", "configuration", "materials")},
                       "rubric": known(rubric)},
        "agent_outcome": known("completed"), "assessment_basis": "Offline test fixture; no model called.",
        "dimensions": [{"id": "facts", "verdict": "pending_review"}], "issues": [], "evidence": [],
        "metrics": {"duration_seconds": known(0)}, "coverage": [],
        "turns": [{"turn": 1, "execution_status": "completed", "run_id": known("fixture-agent-run"),
                   "session_id": known("fixture-session"), "checkpoint_id": known("fixture-checkpoint"),
                   "agent_outcome": known("completed"), "answer_evidence": [evidence],
                   "answer_sha256": sha256_file(answer), "process_evidence": []}],
    }
    return {"schema_version": "3", "run_id": "fixture-run", "selected_suites": ["fixture"],
            "started_at": "2026-10-02T00:00:00+00:00", "ended_at": "2026-10-02T00:00:01+00:00",
            "baseline": {}, "units": [unit], "summary": summarize([unit]), "framework_errors": []}


def filled_template(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(reviewer="Fixture reviewer", reviewed_at="2026-10-02T01:00:00+00:00")
    for entry in value["assessments"]:
        for condition in entry["conditions"]:
            condition.update(verdict="passed", basis="Offline fixture evidence supports this condition.",
                             evidence_refs=["answer:turn:1:1"], evidence_support="sufficient")
    return value


class ReviewRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.result = review_fixture(self.directory)
        self.result_path = self.directory / "result.json"
        atomic_write_json(self.result_path, self.result)

    def test_runner_packet_failure_keeps_result_binding_and_lists_saved_files(self):
        definition = {"unit_id": "fixture"}
        suite = {"suite_id": "fixture", "version": "fixture-v1", "executor": "agent", "units": [definition]}
        executor = SimpleNamespace(make_unit=lambda *_: deepcopy(self.result["units"][0]),
                                   execute_unit=lambda *_, **__: False)
        original_json = runner.atomic_write_json
        original_packet_json = review.atomic_write_json
        original_packet_text = review.atomic_write_text
        for failing_name in ("review-template.json", "review.md"):
            with self.subTest(failing_name=failing_name):
                captured = {}

                def capture_plan(path, value):
                    if path.name == "plan.json":
                        captured["directory"] = path.parent
                    original_json(path, value)

                def packet_json(path, value):
                    captured["result_bytes"] = (path.parent / "result.json").read_bytes()
                    if path.name == failing_name:
                        raise OSError("fixture packet failure")
                    original_packet_json(path, value)

                def packet_text(path, value):
                    if path.name == failing_name:
                        raise OSError("fixture packet failure")
                    original_packet_text(path, value)

                with (patch.object(runner, "load_suite", return_value=suite),
                      patch.object(runner, "import_module", return_value=executor),
                      patch.object(runner, "collect_baseline", return_value={}),
                      patch.object(runner, "atomic_write_json", side_effect=capture_plan),
                      patch.object(review, "atomic_write_json", side_effect=packet_json),
                      patch.object(review, "atomic_write_text", side_effect=packet_text),
                      self.assertRaises(review.ReviewSaveError) as caught):
                    runner.run_benchmark("fixture", root=self.directory)
                directory = captured["directory"]
                result_path = directory / "result.json"
                self.assertEqual(result_path.read_bytes(), captured["result_bytes"])
                saved = set(caught.exception.saved.values())
                expected = {directory / "result.json", directory / "report.md"}
                if failing_name == "review.md":
                    expected.add(directory / "review-template.json")
                    template = filled_template(directory / "review-template.json")
                    self.assertEqual(template["binding"]["result_sha256"], sha256_file(result_path))
                    derived = review.apply_review(caught.exception.result, template, result_path)
                    self.assertEqual(derived["units"][0]["verdict"], "passed")
                self.assertEqual(saved, expected)
                self.assertTrue(all(path.is_file() for path in saved))
                self.assertEqual(caught.exception.result["framework_errors"], [])

    def test_prepare_creates_fresh_packets_and_preserves_filled_template(self):
        review.make_review_packet(self.result, self.directory)
        template_path = self.directory / "review-template.json"
        atomic_write_json(template_path, filled_template(template_path))
        originals = {path: path.read_bytes() for path in (self.result_path, template_path, self.directory / "review.md")}
        first = review.prepare_review_packet(self.directory)
        first_bytes = {path: path.read_bytes() for path in first["saved"].values()}
        second = review.prepare_review_packet(self.directory)
        self.assertEqual(first["exit_code"], 0)
        self.assertEqual(second["exit_code"], 0)
        self.assertNotEqual(set(first["saved"].values()), set(second["saved"].values()))
        for path, content in {**originals, **first_bytes}.items():
            self.assertEqual(path.read_bytes(), content)
        for prepared in (first, second):
            template = prepared["saved"]["review-template.json"]
            self.assertEqual(template.parent.parent, self.directory / "review-packets")
            value = filled_template(template)
            self.assertEqual(value["binding"]["result_sha256"], sha256_file(self.result_path))
            self.assertEqual(review.apply_review(self.result, value, self.result_path)["units"][0]["verdict"], "passed")

    def test_prepare_recovers_old_stale_template_without_overwriting_it(self):
        review.make_review_packet(self.result, self.directory)
        template_path = self.directory / "review-template.json"
        original = template_path.read_bytes()
        self.result["framework_errors"].append({"stage": "saving", "code": "review_packet_failed",
                                                 "message": "Historical packet failure.", "evidence": []})
        atomic_write_json(self.result_path, self.result)
        prepared = review.prepare_review_packet(self.directory)
        self.assertEqual(template_path.read_bytes(), original)
        value = filled_template(prepared["saved"]["review-template.json"])
        self.assertEqual(value["binding"]["result_sha256"], sha256_file(self.result_path))
        review.apply_review(self.result, value, self.result_path)

    def test_prepare_failure_can_retry_without_changing_partial_packet(self):
        original_result = self.result_path.read_bytes()
        with (patch.object(review, "atomic_write_text", side_effect=OSError("fixture packet failure")),
              self.assertRaises(review.ReviewSaveError) as caught):
            review.prepare_review_packet(self.directory)
        partial = caught.exception.saved["template"]
        original_template = partial.read_bytes()
        prepared = review.prepare_review_packet(self.directory)
        self.assertEqual(partial.read_bytes(), original_template)
        self.assertEqual(self.result_path.read_bytes(), original_result)
        self.assertNotEqual(partial, prepared["saved"]["review-template.json"])

    def test_prepare_rejects_reviewed_result_before_creating_output(self):
        self.result["units"][0]["review"] = {"complete": False, "reviewer": "Already assessed"}
        atomic_write_json(self.result_path, self.result)
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            review.prepare_review_packet(self.directory)
        self.assertFalse((self.directory / "review-packets").exists())


if __name__ == "__main__":
    unittest.main()
