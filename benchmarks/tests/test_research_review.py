"""Offline review workflow checks; no model or semantic scorer is used."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.core.files import atomic_write_json, sha256_file
from benchmarks.core.contracts import known, unknown, summarize
from benchmarks.reporting.tasks import (
    validate_research_result, summarize_research, research_exit_code,
)
from benchmarks.evaluators.review import (
    ReviewSaveError, apply_review, import_review, make_review_packet,
)


RUBRIC = {
    "version": "test-only-v1", "dimensions": [
        {"id": "facts_evidence", "description": "事实与证据"},
        {"id": "constraints_history", "description": "约束与历史"},
    ], "cases": {"fixture": {
        "conditions": [{"id": "factual", "dimension": "facts_evidence", "required": True,
                        "description": "被引用的资料须支持具体结论。", "source_refs": ["code/README.md#facts"]}],
        "severe_errors": [{"id": "invented", "description": "编造执行结果。"}],
        "mechanism_review": [{"id": "no_recent_leak", "required": True,
                              "description": "近期输入和回答没有复述被压缩的目标事实。"}],
    }},
}


class ReviewTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.answer = self.directory / "answer.md"
        self.answer.write_text("明确标记的测试替身回答，不是真实 Research 结果。\n", encoding="utf-8")
        evidence = {"kind": "answer", "path": str(self.answer), "source": "test fixture", "availability": "available"}
        unit = {
            "suite_id": "test-only", "unit_id": "formal-1-fixture", "case_id": "fixture",
            "category": "research", "definition_version": "fixture-v1", "phase": "formal", "repeat": 1,
            "granularity": "research_case", "execution_mode": "real_model",
            "conditions": {name: known({"version": "fixture-only"}) for name in ("model", "configuration", "materials", "rubric")},
            "execution_status": "completed", "verdict": "pending_review", "agent_outcome": known("completed"),
            "assessment_basis": "Unit-test fixture only; no real model used.",
            "dimensions": [{"id": "facts_evidence", "verdict": "pending_review"}],
            "issues": [], "evidence": [], "metrics": {"duration_seconds": known(1)},
            "coverage": [{"name": "summary_used", "status": "verified", "reason": "fixture only", "evidence": []}],
            "turns": [{"turn": 1, "execution_status": "completed", "run_id": known("fixture-run"),
                       "session_id": known("fixture-session"), "checkpoint_id": known("fixture-checkpoint"),
                       "agent_outcome": known("completed"), "answer_evidence": evidence,
                       "answer_sha256": sha256_file(self.answer), "process_evidence": []}],
        }
        self.result = {
            "schema_version": "3", "run_id": "fixture-evaluation", "started_at": "2026-09-27T00:00:00+00:00",
            "ended_at": "2026-09-27T00:00:01+00:00", "selected_suites": ["test-only"],
            "baseline": {}, "units": [unit], "summary": summarize([unit]), "framework_errors": [],
        }
        unit["conditions"]["rubric"] = known({"version": RUBRIC["version"]})
        self.path = self.directory / "result.json"
        atomic_write_json(self.path, self.result)

    def template(self):
        make_review_packet(self.result, self.directory, rubric=RUBRIC)
        value = json.loads((self.directory / "review-template.json").read_text(encoding="utf-8"))
        value.update(reviewer="test human", reviewed_at="2026-09-27T01:00:00+00:00")
        return value

    def completed_review(self):
        value = self.template()
        entry = value["assessments"][0]
        for item in entry["conditions"] + entry["mechanism_review"]:
            item.update(verdict="passed", basis="测试人工记录：核对回答及过程支持此条件。",
                        evidence_refs=["answer:turn:1:1"], evidence_support="sufficient")
        entry["severe_errors"][0].update(present=False, basis="测试人工记录：未发现此严重错误。",
                                        evidence_refs=["answer:turn:1:1"])
        return value

    def test_packet_keeps_pending_and_includes_answer_rubric_and_references(self):
        value = self.template()
        self.assertEqual(value["assessments"][0]["conditions"][0]["verdict"], "pending_review")
        text = (self.directory / "review.md").read_text(encoding="utf-8")
        self.assertIn(self.answer.read_text(encoding="utf-8"), text)
        self.assertIn("引用存在不代表", text)
        self.assertIn("source:code/README.md#facts", text)
        self.assertEqual(self.result["units"][0]["verdict"], "pending_review")

    def test_valid_pass_is_new_result_and_dimensions_exclude_inapplicable(self):
        review = self.completed_review()
        derived = apply_review(self.result, review, self.path, rubric=RUBRIC)
        self.assertEqual(derived["units"][0]["verdict"], "passed")
        self.assertEqual(research_exit_code(derived), 0)
        counts = summarize_research(derived["units"])["formal"]
        self.assertEqual(counts["planned_samples"], 1)
        self.assertEqual(counts["reviewed_samples"], 1)
        self.assertNotIn("constraints_history", counts["dimensions"])
        self.assertEqual(self.result["units"][0]["verdict"], "pending_review")

    def test_insufficient_evidence_cannot_pass_even_with_real_reference(self):
        review = self.completed_review()
        review["assessments"][0]["conditions"][0]["evidence_support"] = "insufficient"
        with self.assertRaisesRegex(ValueError, "insufficient support"):
            apply_review(self.result, review, self.path, rubric=RUBRIC)

    def test_quality_failure_is_completed_failed(self):
        review = self.completed_review()
        review["assessments"][0]["conditions"][0].update(verdict="failed", evidence_support="insufficient")
        derived = apply_review(self.result, review, self.path, rubric=RUBRIC)
        self.assertEqual(derived["units"][0]["execution_status"], "completed")
        self.assertEqual(derived["units"][0]["verdict"], "failed")
        self.assertEqual(research_exit_code(derived), 1)

    def test_severe_error_overrides_dimension_pass(self):
        review = self.completed_review()
        review["assessments"][0]["severe_errors"][0]["present"] = True
        derived = apply_review(self.result, review, self.path, rubric=RUBRIC)
        self.assertEqual(derived["units"][0]["verdict"], "failed")

    def test_partial_review_stays_pending(self):
        review = self.template()
        derived = apply_review(self.result, review, self.path, rubric=RUBRIC)
        self.assertEqual(derived["units"][0]["verdict"], "pending_review")
        self.assertEqual(research_exit_code(derived), 3)

    def test_human_recent_leak_failure_records_gap_not_summary_success(self):
        review = self.completed_review()
        review["assessments"][0]["mechanism_review"][0]["verdict"] = "failed"
        derived = apply_review(self.result, review, self.path, rubric=RUBRIC)
        self.assertEqual(derived["units"][0]["coverage"][-1]["status"], "gap")
        self.assertNotEqual(derived["units"][0]["verdict"], "passed")
        self.assertEqual(research_exit_code(derived), 2)

    def test_missing_mechanism_and_reread_gaps_survive_review(self):
        for name in ("summary_not_triggered", "unexpected_reread"):
            with self.subTest(name=name):
                self.result["units"][0]["coverage"] = [{"name": name, "status": "gap", "reason": "fixture gap", "evidence": []}]
                atomic_write_json(self.path, self.result)
                template_dir = self.directory / name
                template_dir.mkdir()
                atomic_write_json(template_dir / "result.json", self.result)
                make_review_packet(self.result, template_dir, rubric=RUBRIC)
                review = json.loads((template_dir / "review-template.json").read_text(encoding="utf-8"))
                review.update(reviewer="test human", reviewed_at="2026-09-27T01:00:00+00:00")
                for item in review["assessments"][0]["conditions"] + review["assessments"][0]["mechanism_review"]:
                    item.update(verdict="passed", basis="test human judgment", evidence_refs=["answer:turn:1:1"], evidence_support="sufficient")
                review["assessments"][0]["severe_errors"][0].update(present=False, basis="test", evidence_refs=["answer:turn:1:1"])
                derived = apply_review(self.result, review, self.path, rubric=RUBRIC)
                self.assertEqual(derived["units"][0]["coverage"][0]["status"], "gap")
                self.assertEqual(research_exit_code(derived), 2)

    def test_tampered_binding_answer_rule_or_material_is_rejected(self):
        review = self.completed_review()
        for label in ("run_id", "result_sha256", "rubric_sha256"):
            with self.subTest(label=label):
                changed = deepcopy(review)
                changed["binding"][label] = "tampered"
                with self.assertRaisesRegex(ValueError, "binding"):
                    apply_review(self.result, changed, self.path, rubric=RUBRIC)
        changed_rule = deepcopy(RUBRIC)
        changed_rule["cases"]["fixture"]["conditions"][0]["description"] = "relaxed"
        with self.assertRaisesRegex(ValueError, "binding"):
            apply_review(self.result, review, self.path, rubric=changed_rule)
        self.answer.write_text("tampered answer", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "answer changed"):
            apply_review(self.result, review, self.path, rubric=RUBRIC)

    def test_missing_reviewer_basis_unknown_ref_and_rule_omission_rejected(self):
        original = self.completed_review()
        mutations = (
            lambda value: value.update(reviewer=""),
            lambda value: value["assessments"][0]["conditions"][0].update(basis=""),
            lambda value: value["assessments"][0]["conditions"][0].update(evidence_refs=["invented"]),
            lambda value: value["assessments"][0].update(severe_errors=[]),
            lambda value: value.update(reviewed_at="2026-09-27T01:00:00"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                review = deepcopy(original)
                mutate(review)
                with self.assertRaises(ValueError):
                    apply_review(self.result, review, self.path, rubric=RUBRIC)

    def test_revisions_are_independent_and_do_not_modify_raw_execution(self):
        review = self.completed_review()
        review_path = self.directory / "filled.json"
        atomic_write_json(review_path, review)
        original = self.path.read_bytes()
        first = import_review(self.path, review_path, rubric=RUBRIC)
        second = import_review(self.path, review_path, rubric=RUBRIC)
        self.assertNotEqual(first["saved"]["result"], second["saved"]["result"])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertTrue(all(path.is_file() for path in first["saved"].values()))

    def test_write_failure_reports_only_saved_paths_and_preserves_facts(self):
        review = self.completed_review()
        review_path = self.directory / "filled.json"
        atomic_write_json(review_path, review)
        with patch("benchmarks.evaluators.review.atomic_write_text", side_effect=OSError("injected")):
            with self.assertRaises(ReviewSaveError) as caught:
                import_review(self.path, review_path, rubric=RUBRIC)
        self.assertEqual(set(caught.exception.saved), {"review", "result"})
        self.assertEqual(caught.exception.result["units"][0]["verdict"], "passed")
        self.assertTrue(all(path.is_file() for path in caught.exception.saved.values()))
        saved_result = json.loads(caught.exception.saved["result"].read_text(encoding="utf-8"))
        self.assertEqual(research_exit_code(saved_result), 2)
        self.assertEqual(saved_result["framework_errors"][-1]["code"], "review_artifact_write_failed")

    def test_existing_packet_cannot_overwrite_user_work(self):
        self.template()
        existing = self.directory / "review-template.json"
        existing.write_text("user edits", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            make_review_packet(self.result, self.directory, rubric=RUBRIC)
        self.assertEqual(existing.read_text(encoding="utf-8"), "user edits")

    def test_unstarted_and_unknown_data_stay_explicit(self):
        unit = self.result["units"][0]
        unit["execution_status"] = "not_run"
        unit["turns"][0].update(execution_status="not_run", run_id=unknown("not started"),
                                answer_evidence={"kind": "answer", "path": str(self.answer), "source": "fixture", "availability": "missing"})
        self.result["summary"] = summarize([unit])
        atomic_write_json(self.path, self.result)
        validate_research_result(self.result)
        review = self.template()
        derived = apply_review(self.result, review, self.path, rubric=RUBRIC)
        self.assertEqual(derived["units"][0]["verdict"], "pending_review")
        self.assertEqual(research_exit_code(derived), 2)

    def test_missing_answer_cannot_be_semantically_scored(self):
        unit = self.result["units"][0]
        unit["turns"][0]["answer_evidence"]["availability"] = "missing"
        atomic_write_json(self.path, self.result)
        review = self.completed_review()
        with self.assertRaisesRegex(ValueError, "actual saved answer"):
            apply_review(self.result, review, self.path, rubric=RUBRIC)

    def test_no_answer_target_failure_can_use_observed_agent_execution(self):
        rubric = deepcopy(RUBRIC)
        rubric["dimensions"].append({"id": "target_coverage", "description": "目标覆盖"})
        rubric["cases"]["fixture"]["conditions"].append({
            "id": "target", "dimension": "target_coverage", "required": True,
            "description": "实际交付要求的回答。", "source_refs": [],
        })
        turn = self.result["units"][0]["turns"][0]
        turn["answer_evidence"]["availability"] = "missing"
        turn["agent_outcome"] = known("model_call_limit")
        atomic_write_json(self.path, self.result)
        make_review_packet(self.result, self.directory, rubric=rubric)
        review = json.loads((self.directory / "review-template.json").read_text(encoding="utf-8"))
        review.update(reviewer="test human", reviewed_at="2026-09-27T01:00:00+00:00")
        review["assessments"][0]["conditions"][1].update(
            verdict="failed", basis="实际 Agent 运行额度耗尽，未交付要求的回答。",
            evidence_refs=["execution:turn:1"], evidence_support="sufficient")
        derived = apply_review(self.result, review, self.path, rubric=rubric)
        unit = derived["units"][0]
        self.assertEqual(unit["verdict"], "failed")
        self.assertEqual(unit["dimensions"][0]["verdict"], "pending_review")
        self.assertEqual(research_exit_code(derived), 1)
        review["assessments"][0]["conditions"][1]["verdict"] = "passed"
        with self.assertRaisesRegex(ValueError, "actual saved answer"):
            apply_review(self.result, review, self.path, rubric=rubric)

    def test_no_answer_failure_requires_observed_agent_run(self):
        rubric = deepcopy(RUBRIC)
        rubric["dimensions"][0]["id"] = "target_coverage"
        rubric["cases"]["fixture"]["conditions"][0]["dimension"] = "target_coverage"
        turn = self.result["units"][0]["turns"][0]
        turn.update(execution_status="failed", run_id=unknown("no returned Agent run"))
        turn["answer_evidence"]["availability"] = "missing"
        atomic_write_json(self.path, self.result)
        make_review_packet(self.result, self.directory, rubric=rubric)
        review = json.loads((self.directory / "review-template.json").read_text(encoding="utf-8"))
        review.update(reviewer="test human", reviewed_at="2026-09-27T01:00:00+00:00")
        review["assessments"][0]["conditions"][0].update(
            verdict="failed", basis="不能把没有 Agent 运行证据的失败当质量失败。",
            evidence_refs=["execution:turn:1"], evidence_support="sufficient")
        with self.assertRaisesRegex(ValueError, "actual saved answer"):
            apply_review(self.result, review, self.path, rubric=rubric)

    def test_usage_summary_keeps_missing_and_does_not_add_summary_subset(self):
        unit = self.result["units"][0]
        unit["metrics"].update(input_tokens=known(120), input_tokens_missing_requests=known(1),
                               summary_usage_totals=known({"input_tokens": 40}))
        counts = summarize_research([unit])["formal"]["metrics"]
        self.assertEqual(counts["input_tokens"]["observed_sum"]["value"], 120)
        self.assertEqual(counts["input_tokens"]["missing_request_fields"]["value"], 1)
        self.assertEqual(counts["output_tokens"]["observed_sum"]["availability"], "unknown")


if __name__ == "__main__":
    unittest.main()
