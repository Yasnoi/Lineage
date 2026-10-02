"""Evidence-oriented checks for the current result contract and status separation."""

import json
import unittest

from benchmarks.core.contracts import (
    exit_code,
    known,
    not_applicable,
    summarize,
    unknown,
    validate_result,
)
from benchmarks.reporting.render import render_report
from benchmarks.tests.contract_examples import (
    EXAMPLE_NOTICE,
    example_result,
    research_examples,
)


class ResultContractTests(unittest.TestCase):
    def test_unavailable_values_never_fabricate_measurements(self):
        for value in (unknown("未取得"), not_applicable("未调用")):
            self.assertIsNone(value["value"])
            self.assertTrue(value["reason"])
            self.assertNotEqual(value["availability"], "known")
        self.assertEqual(known(0)["value"], 0)
        self.assertEqual(known(False)["value"], False)

    def test_completed_research_is_pending_until_assessed(self):
        result = example_result()
        validate_result(result)
        counts = result["summary"]["research"]
        self.assertEqual(counts["total"], 3)
        self.assertEqual(counts["execution_status"]["completed"], 3)
        self.assertEqual(counts["verdict"]["passed"], 0)
        self.assertEqual(counts["verdict"]["pending_review"], 3)
        self.assertEqual(exit_code(result), 3)

    def test_unsupported_result_versions_are_rejected_by_all_public_readers(self):
        result = example_result()
        for version in ("1", "2", "4", None):
            result["schema_version"] = version
            for reader in (validate_result, exit_code, render_report):
                with self.subTest(version=version, reader=reader.__name__), self.assertRaises(ValueError):
                    reader(result)

    def test_expected_agent_error_does_not_fail_control_assertions(self):
        unit = research_examples()[0]
        unit.update(
            category="control",
            execution_mode="deterministic_stub",
            verdict="passed",
            agent_outcome=known("execution_error"),
            assessment_basis="错误终态是故障注入场景的预期结果，断言通过。",
            dimensions=[],
        )
        unit.pop("turns")
        result = example_result([unit])
        validate_result(result)
        self.assertEqual(exit_code(result), 0)

    def test_mixed_outcomes_keep_each_count_and_exit_precedence(self):
        units = research_examples()
        units[0].update(category="control", verdict="passed")
        for dimension in units[0]["dimensions"]:
            dimension["verdict"] = "passed"
        units[1].update(category="control", verdict="failed")
        result = example_result(units)
        self.assertEqual(exit_code(result), 1)
        self.assertEqual(result["summary"]["control"]["verdict"]["passed"], 1)
        self.assertEqual(result["summary"]["control"]["verdict"]["failed"], 1)
        self.assertEqual(result["summary"]["research"]["verdict"]["pending_review"], 1)
        for state in ("failed", "blocked", "not_run"):
            with self.subTest(state=state):
                units[-1]["execution_status"] = state
                mixed = example_result(units)
                validate_result(mixed)
                self.assertEqual(exit_code(mixed), 2)
                self.assertEqual(
                    mixed["summary"]["research"]["execution_status"][state], 1
                )
                self.assertEqual(mixed["units"][1]["verdict"], "failed")
        result["framework_errors"].append(
            {
                "stage": "report",
                "code": "save_failed",
                "exception_type": "OSError",
                "message": "保存失败",
                "evidence": [],
            }
        )
        self.assertEqual(exit_code(result), 2)
        self.assertEqual(exit_code(example_result([])), 2)

    def test_invalid_states_and_false_passes_are_rejected(self):
        invalid_pairs = [
            ("unknown", "pending_review"),
            ("completed", "unscored"),
            ("failed", "passed"),
            ("blocked", "failed"),
            ("not_run", "passed"),
        ]
        for state, verdict in invalid_pairs:
            with self.subTest(state=state, verdict=verdict):
                result = example_result()
                result["units"][0].update(execution_status=state, verdict=verdict)
                with self.assertRaises(ValueError):
                    validate_result(result)

    def test_research_examples_preserve_session_and_summary_provenance(self):
        result = json.loads(json.dumps(example_result(), ensure_ascii=False))
        validate_result(result)
        single, follow_up, summary = result["units"]
        self.assertEqual(len(single["turns"]), 1)
        self.assertEqual(
            single["turns"][0]["session_id"]["availability"], "not_applicable"
        )
        for unit in (follow_up, summary):
            sessions = {turn["session_id"]["value"] for turn in unit["turns"]}
            self.assertEqual(len(sessions), 1)
            self.assertTrue(all(turn["checkpoint_id"]["value"] for turn in unit["turns"]))
        self.assertNotEqual(
            follow_up["turns"][0]["session_id"], summary["turns"][0]["session_id"]
        )
        before, _, after = summary["turns"]
        self.assertNotEqual(before["active_version"], after["active_version"])
        self.assertEqual(after["summary"]["source_turns"], [1, 2])
        self.assertEqual(
            after["summary"]["activated_version"], after["active_version"]["value"]
        )
        self.assertIn("只读研究", before["task"])
        self.assertNotIn("只读研究", after["task"])
        self.assertEqual(
            after["summary"]["mechanism_evidence"]["kind"], "summary_activation"
        )
        for unit in result["units"]:
            self.assertEqual(unit["execution_mode"], "contract_example")
            self.assertIn(EXAMPLE_NOTICE, unit["assessment_basis"])
            self.assertFalse(unit["conditions"]["scoring"]["value"]["agent_accessible"])
            self.assertGreater(len(unit["dimensions"]), 1)
            self.assertTrue(all(d["verdict"] == "pending_review" for d in unit["dimensions"]))
            for turn in unit["turns"]:
                self.assertNotEqual(
                    turn["answer_evidence"][0]["path"],
                    turn["process_evidence"][0]["path"],
                )
            self.assertTrue(all(e["availability"] == "missing" for e in unit["evidence"]))

    def test_report_retains_example_limits_and_separate_evidence(self):
        result = example_result()
        report = render_report(result)
        self.assertIn(result["run_id"], report)
        self.assertIn("契约样例", report)
        self.assertIn("不是实际 Research 验收", report)
        self.assertIn("pending_review", report)
        self.assertIn("completed", report)
        for unit in result["units"]:
            self.assertIn(unit["unit_id"], report)
            self.assertIn(unit["evidence"][0]["path"], report)
            self.assertIn(unit["evidence"][1]["path"], report)
        self.assertEqual(result["summary"], summarize(result["units"]))


if __name__ == "__main__":
    unittest.main()
