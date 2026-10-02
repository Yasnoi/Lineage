"""Reports expose decisions and evidence without repeating source manifests."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.core.contracts import known, unknown, summarize
from benchmarks.core.executors.control import make_unit
from benchmarks.core.files import ROOT
from benchmarks.core.registry import load_suite
from benchmarks.reporting.render import render_report
from benchmarks.reporting.tasks import render_research_report, research_exit_code, summarize_research
from benchmarks.tests.test_review_recovery import review_fixture


class ReportReadabilityTests(unittest.TestCase):
    def test_research_readers_reject_old_version_even_when_coverage_has_gap(self):
        with TemporaryDirectory() as temporary:
            result = review_fixture(Path(temporary))
            result["units"][0]["coverage"] = [{"name": "fixture-gap", "status": "gap",
                                               "reason": "Fixture mechanism gap.", "evidence": []}]
            for version in ("1", "2"):
                result["schema_version"] = version
                for reader in (research_exit_code, render_research_report):
                    with self.subTest(version=version, reader=reader.__name__):
                        with self.assertRaisesRegex(ValueError, "schema_version"):
                            reader(result)

    def test_control_summary_precedes_details_and_source_hashes_stay_in_json(self):
        definitions = load_suite("agent-contracts")["units"][:2]
        units = [make_unit(definition, ROOT) for definition in definitions]
        fingerprint = "a" * 64
        baseline = {"agent_commit": known("fixture-commit"), "working_tree_dirty": known(False),
                    "python_version": known("fixture-python"),
                    "framework_sources": {"benchmarks/checks/execution/react.py": known(fingerprint)}}
        units[0]["execution_status"] = "blocked"
        units[0]["issues"] = [{"stage": "preflight", "code": "fixture_missing",
                                "message": "Fixture entrypoint is missing.", "evidence": []}]
        units[0]["groups"][0]["coverage_gaps"] = [{"scenario": "fixture-filesystem", "applicable": True,
                                                   "reason": "Filesystem capability unavailable."}]
        units[0]["groups"][0]["issues"] = [{"stage": "group", "code": "fixture_group_error",
                                              "message": "Fixture group could not finish.", "evidence": []}]
        result = {"schema_version": "3", "run_id": "offline-report-fixture", "started_at": "start", "ended_at": "end",
                  "selected_suites": ["agent-contracts"], "units": units, "baseline": baseline,
                  "summary": summarize(units), "framework_errors": []}
        snapshot = deepcopy(result)
        report = render_report(result)
        self.assertLess(report.index("## 状态汇总"), report.index("## 失败与限制"))
        self.assertLess(report.index("## 失败与限制"), report.index("## agent-contracts /"))
        self.assertLess(report.index("## agent-contracts /"), report.index("## 基线"))
        self.assertIn("Fixture entrypoint is missing.", report)
        first_unit = report.index("## agent-contracts /")
        self.assertLess(report.index("Filesystem capability unavailable."), first_unit)
        self.assertLess(report.index("Fixture group could not finish."), first_unit)
        self.assertLess(report.index(units[1]["assessment_basis"]), first_unit)
        self.assertIn("blocked", report)
        self.assertIn("not_run", report)
        self.assertIn("[result.json](result.json)", report)
        self.assertNotIn(fingerprint, report)
        self.assertNotIn("```json", report)
        self.assertTrue(all("check_sources" not in unit["conditions"] for unit in units))
        self.assertEqual(result, snapshot)
        self.assertEqual(result["baseline"]["framework_sources"]["benchmarks/checks/execution/react.py"]["value"], fingerprint)

    def test_research_tables_preserve_unknown_zero_gaps_and_answer_links(self):
        with TemporaryDirectory() as temporary:
            result = review_fixture(Path(temporary))
            unit = result["units"][0]
            unit["metrics"].update(sdk_requests=known(0), input_tokens=known(0),
                                   output_tokens=unknown("endpoint omitted output usage"),
                                   input_tokens_missing_requests=known(1))
            unit["coverage"] = [{"name": "summary_used", "status": "gap",
                                  "reason": "Summary was not observed.", "evidence": []}]
            unit["tracing"] = {"remote": [{"status": "query_failed", "trace_id": "fixture-trace"}]}
            unit["evidence"] = [{"kind": "stderr", "path": str(Path(temporary) / "missing.txt"),
                                 "availability": "missing", "source": "fixture"}]
            snapshot = deepcopy(result)
            report = render_research_report(result)
            self.assertIn("| sdk_requests | 0 |", report)
            self.assertIn("endpoint omitted output usage", report)
            self.assertIn("Summary was not observed.", report)
            self.assertLess(report.index("Summary was not observed."), report.index("## fixture /"))
            self.assertIn("query_failed", report)
            self.assertIn("| 远端核验 | query_failed | 1 |", report)
            self.assertIn("[answer](<", report)
            self.assertIn("missing.txt", report)
            self.assertIn("missing", report)
            self.assertIn("缺失请求字段", report)
            self.assertNotIn("```json", report)
            self.assertEqual(result, snapshot)
            summary = summarize_research(result["units"])
            self.assertEqual(summary["planned_samples"], 1)
            self.assertEqual(summary["metrics"]["sdk_requests"]["observed_sum"]["value"], 0)
            self.assertEqual(summary["metrics"]["output_tokens"]["observed_sum"]["availability"], "unknown")
            self.assertEqual(summary["metrics"]["input_tokens"]["missing_request_fields"]["value"], 1)


if __name__ == "__main__":
    unittest.main()
