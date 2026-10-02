"""V2 orchestration checks using durable fixture events and one real assertion."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.suites.agent_contracts import suite as catalog
from benchmarks.core import runner
from benchmarks.core.executors import control
from benchmarks.core.files import ROOT, atomic_write_json, sha256_file, utc_now
from benchmarks.core.events import GroupJournal, plan_digest
from benchmarks.core.contracts import exit_code, summarize, validate_result
from benchmarks.checks.support import events as support


class FailedEventWrite:
    """Leave a genuine partial record, then fail the recording channel."""

    def __init__(self, stream):
        self.stream = stream

    def write(self, value):
        self.stream.write(value[:len(value) // 2])
        self.stream.flush()
        raise OSError("injected group-event disk failure")

    def close(self):
        self.stream.close()


class V2RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="benchmark-v2-runner-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "scripts").mkdir()
        self.units = {}
        for suffix in ("a", "b", "unselected"):
            identifier = f"fixture-{suffix}"
            script = f"scripts/{identifier}.py"
            (self.root / script).write_text("# Synthetic orchestration fixture.\n", encoding="utf-8")
            self.units[identifier] = {
                "unit_id": identifier, "suite_id": "control-execution",
                "module": "control-execution", "script": script,
                "category": "control_logic", "definition_version": "2",
                "granularity": "script", "execution_mode": "deterministic_stub",
                "timeout_seconds": 120,
                "groups": [
                    {"group_id": f"{identifier}/{name}", "description": "Synthetic fixture group",
                     "source": {"path": script, "symbol": name},
                     "primary_area": "execution", "tags": ["execution", "recording"]}
                    for name in ("first", "second", "third")
                ],
            }
        self.suites = {
            "fixture-all-selected": ["fixture-a", "fixture-b"],
            "fixture-a-only": ["fixture-a"],
            "fixture-b-only": ["fixture-b"],
        }
        self.launched = []

    def launch_fixture(self, modes=None, alterations=None):
        """Replace process launch, but use real journals, file evidence and receipts."""
        modes, alterations = modes or {}, alterations or {}

        def launch(command, **kwargs):
            unit_id = command[command.index("--unit-id") + 1]
            run_id = command[command.index("--run-id") + 1]
            path = Path(command[command.index("--receipt") + 1])
            definition = self.units[unit_id]
            self.launched.append(unit_id)
            kwargs["stdout"].write(b"SYNTHETIC FIXTURE: ALL CHECKS PASSED\n")
            kwargs["stderr"].write(b"synthetic worker evidence\n")
            journal = GroupJournal(path.with_name("groups.jsonl"), run_id, definition)
            outcome, error, code = "passed", None, 0
            try:
                with support.activate(journal), patch.object(
                    support, "_offsets", return_value={"stdout": 0, "stderr": 0}
                ):
                    for index, group in enumerate(definition["groups"]):
                        mode = modes.get(unit_id)
                        if mode == "recording_failure" and index == 1:
                            journal.stream = FailedEventWrite(journal.stream)
                        with support.check_group(group["group_id"]):
                            support.observe_state({"run_id": "shared-observed-run"})
                            support.observe_state({"run_id": "shared-observed-run"})
                            if unit_id == "fixture-b":
                                support.observe_state({"run_id": "second-observed-run"})
                            support.observe_harness("trace_harness", run_id="constructed-run")
                            if index == 1 and mode == "assertion":
                                raise AssertionError("injected fixture assertion")
                            if index == 1 and mode in {"gap", "not_applicable"}:
                                support.coverage_gap("link fixture", "environment unavailable",
                                                     applicable=mode == "gap")
            except support.CheckStopped as stopped:
                outcome, code = "assertion_failed", 1
                error = {"stage": "check_execution", "code": "check_error",
                         "exception_type": type(stopped.cause).__name__}
            except support.RecordingFailure:
                outcome, code = "recording_failed", 2
                error = {"stage": "group_recording", "code": "recording_failed",
                         "exception_type": "RecordingFailure"}
            finally:
                journal.close()
            if journal.coverage_gaps:
                outcome, code = "coverage_gap", 2
            receipt = {
                "schema_version": "3", "adapter_version": "3", "run_id": run_id,
                "unit_id": unit_id, "script_path": str(self.root / definition["script"]),
                "script_sha256": sha256_file(self.root / definition["script"]),
                "plan_digest": plan_digest(definition["groups"]), "phase": "finished",
                "outcome": outcome, "error": error, "started_at": utc_now(), "ended_at": utc_now(),
                "events_count": journal.sequence, "events_sha256": sha256_file(path.with_name("groups.jsonl")),
            }
            atomic_write_json(path, receipt)
            alteration = alterations.get(unit_id)
            if alteration == "journal_suffix":
                with path.with_name("groups.jsonl").open("ab") as stream:
                    stream.write(b'{"partial":')
            elif alteration in {
                "observation_run_id_array", "observation_log_path_number",
                "observation_raw_payload", "observation_raw_identifier", "evidence_invalid_identifiers",
            }:
                events_path = path.with_name("groups.jsonl")
                events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
                # Corrupt the second group, so a fully validated prior group
                # must survive while this record and its suffix are rejected.
                damaged = next(event for event in events if event["event"] == "observation"
                               and event["group_id"] == f"{unit_id}/second")
                if alteration == "observation_run_id_array":
                    damaged["payload"]["identifiers"]["run_id"] = ["invalid", "identity"]
                elif alteration == "observation_log_path_number":
                    damaged["payload"]["identifiers"]["log_path"] = 123
                elif alteration == "observation_raw_payload":
                    damaged["payload"]["raw_response"] = {"content": "UNFILTERED_OBSERVATION_MARKER"}
                elif alteration == "observation_raw_identifier":
                    damaged["payload"]["identifiers"]["raw_response"] = {"content": "UNFILTERED_OBSERVATION_MARKER"}
                else:
                    damaged["event"] = "evidence"
                    damaged["payload"] = {
                        "kind": "artifact", "path": str(path.with_name("stdout.txt")),
                        "identifiers": {"run_id": ["invalid", "identity"]},
                    }
                events_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            elif alteration == "receipt_json":
                path.write_text("not JSON", encoding="utf-8")
            elif alteration == "receipt_missing":
                path.unlink()
            elif alteration == "receipt_identity":
                atomic_write_json(path, {**receipt, "run_id": "previous-run"})
            elif alteration == "receipt_digest":
                atomic_write_json(path, {**receipt, "events_sha256": "not-the-durable-events"})
            elif alteration == "exit_mismatch":
                code = 1
            return SimpleNamespace(returncode=code, wait=lambda **_: code, poll=lambda: code)

        return launch

    def run_fixture(self, selected=None, *, modes=None, alterations=None):
        identifier = (selected or ["fixture-all-selected"])[0]
        registered = {key: (lambda k=key: {"suite_id": k, "version": "fixture", "executor": "contracts",
                       "units": [self.units[i] for i in self.suites[k]]}) for key in self.suites}
        with patch.object(runner, "collect_baseline", return_value={}), patch.object(control.subprocess, "Popen", side_effect=self.launch_fixture(modes, alterations)):
            result, saved = runner.run_benchmark(identifier, root=self.root, registry=registered)
        validate_result(result)
        self.assertEqual({path.name for path in saved}, {"result.json", "report.md"})
        self.assertEqual(json.loads(next(p for p in saved if p.name == "result.json").read_text(encoding="utf-8")),result)
        return result

    def test_catalog_maps_indirect_groups_once(self):
        units, _ = catalog.load_catalog()
        self.assertEqual(len(units), 17)
        groups = [g for unit in units.values() for g in unit["groups"]]
        self.assertEqual(len(groups), 155)
        self.assertEqual(len({g["group_id"] for g in groups}), 155)
        self.assertEqual(sum(g.get("legacy_source", {}).get("path") == "scripts/check_trace_data.py" for g in groups), 2)
        self.assertEqual(sum(g.get("legacy_source", {}).get("path") == "scripts/check_trace_usage.py" for g in groups), 4)

    def test_single_selection_does_not_run_unselected_units(self):
        result = self.run_fixture(["fixture-b-only"])
        self.assertEqual(self.launched, ["fixture-b"])
        self.assertEqual(result["summary"]["control_logic"]["total"], 1)
        self.assertEqual(result["summary"]["control_logic"]["groups"]["planned"], 3)

    def test_group_assertion_stops_its_script_but_next_unit_executes(self):
        result = self.run_fixture(modes={"fixture-a": "assertion"})
        failed, next_unit = result["units"]
        self.assertEqual(self.launched, ["fixture-a", "fixture-b"])
        self.assertEqual((failed["execution_status"], failed["verdict"]), ("completed", "failed"))
        self.assertEqual([g["verdict"] for g in failed["groups"]], ["passed", "failed", "pending_review"])
        self.assertEqual(failed["groups"][2]["execution_status"], "not_run")
        self.assertIsNone(failed["groups"][2]["started_at"])
        self.assertEqual(failed["issues"][0]["group_id"], "fixture-a/second")
        self.assertEqual(next_unit["verdict"], "passed")
        self.assertEqual(exit_code(result), 1)
        self.assertEqual(result["summary"]["control_logic"]["groups"]["finished"], 5)

    def test_group_recording_failure_preserves_prefix_and_stops_scheduling(self):
        result = self.run_fixture(modes={"fixture-a": "recording_failure"})
        first, second = result["units"]
        self.assertEqual(self.launched, ["fixture-a"])
        self.assertEqual(first["execution_status"], "failed")
        self.assertEqual(first["groups"][0]["verdict"], "passed")
        self.assertTrue(all(g["verdict"] != "passed" for g in first["groups"][1:]))
        self.assertEqual(second["execution_status"], "not_run")
        self.assertTrue(all(g["started_at"] is None for g in second["groups"]))
        self.assertEqual(result["framework_errors"][0]["code"], "scheduling_stopped")
        self.assertEqual(exit_code(result), 2)

    def test_corrupt_journal_cannot_use_success_receipt_or_stdout_to_pass(self):
        result = self.run_fixture(alterations={"fixture-a": "journal_suffix"})
        self.assertEqual(self.launched, ["fixture-a"])
        self.assertEqual(result["units"][0]["execution_status"], "failed")
        self.assertEqual(result["units"][0]["verdict"], "pending_review")
        self.assertEqual(exit_code(result), 2)
        self.assertIn("invalid_event_stream", {item["code"] for item in result["units"][0]["issues"]})

    def test_invalid_final_receipts_do_not_erase_confirmed_groups_or_pass(self):
        for alteration in ("receipt_missing", "receipt_json", "receipt_identity", "receipt_digest", "exit_mismatch"):
            with self.subTest(alteration=alteration):
                result = self.run_fixture(["fixture-a-only"], alterations={"fixture-a": alteration})
                unit = result["units"][0]
                self.assertEqual((unit["execution_status"], unit["verdict"]), ("failed", "pending_review"))
                self.assertTrue(all(g["verdict"] == "passed" for g in unit["groups"]))
                self.assertIn("invalid_receipt", {item["code"] for item in unit["issues"]})
                self.assertEqual(exit_code(result), 2)

    def test_malformed_observation_or_evidence_keeps_prefix_and_saves_failure(self):
        for alteration in (
            "observation_run_id_array", "observation_log_path_number", "observation_raw_payload",
            "observation_raw_identifier", "evidence_invalid_identifiers",
        ):
            with self.subTest(alteration=alteration):
                self.launched.clear()
                result = self.run_fixture(alterations={"fixture-a": alteration})
                first, second = result["units"]
                self.assertEqual(self.launched, ["fixture-a"])
                self.assertEqual((first["execution_status"], first["verdict"]), ("failed", "pending_review"))
                self.assertEqual(first["groups"][0]["verdict"], "passed")
                self.assertEqual(first["groups"][1]["execution_status"], "failed")
                self.assertEqual(first["groups"][1]["verdict"], "pending_review")
                self.assertEqual(first["groups"][1]["observations"], [])
                self.assertEqual(first["groups"][1]["evidence"], [])
                self.assertEqual(first["groups"][2]["execution_status"], "not_run")
                self.assertEqual(second["execution_status"], "not_run")
                self.assertEqual(exit_code(result), 2)
                self.assertIn("invalid_event_stream", {item["code"] for item in first["issues"]})
                self.assertNotIn("UNFILTERED_OBSERVATION_MARKER", json.dumps(result))
                output = next(item for item in first["evidence"] if item["kind"] == "stdout")
                self.assertIn("ALL CHECKS PASSED", Path(output["path"]).read_text(encoding="utf-8"))

    def test_applicable_gap_is_not_pass_and_does_not_stop_independent_units(self):
        result = self.run_fixture(modes={"fixture-a": "gap"})
        first, second = result["units"]
        self.assertEqual(self.launched, ["fixture-a", "fixture-b"])
        self.assertEqual((first["execution_status"], first["verdict"]), ("completed", "pending_review"))
        self.assertEqual(first["groups"][1]["verdict"], "pending_review")
        self.assertEqual(second["verdict"], "passed")
        self.assertEqual(result["summary"]["control_logic"]["groups"]["applicable_coverage_gaps"], 1)
        self.assertEqual(exit_code(result), 2)

    def test_nonapplicable_fixture_keeps_explicit_gap_without_blocking_pass(self):
        result = self.run_fixture(["fixture-a-only"], modes={"fixture-a": "not_applicable"})
        group = result["units"][0]["groups"][1]
        self.assertFalse(group["coverage_gaps"][0]["applicable"])
        self.assertEqual(group["verdict"], "passed")
        self.assertEqual(exit_code(result), 0)

    def test_summary_separates_levels_deduplicates_runs_and_excludes_harness(self):
        result = self.run_fixture()
        summary = result["summary"]["control_logic"]
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["modules"], {"control-execution": 2})
        self.assertEqual(summary["groups"]["planned"], 6)
        self.assertEqual(summary["groups"]["started"], 6)
        self.assertEqual(summary["groups"]["finished"], 6)
        self.assertEqual(summary["observed_agent_runs"]["count"], 2)
        self.assertEqual(summary["observed_agent_runs"]["completeness"], "partial")
        self.assertEqual(summary["harness_observations"], {"trace_harness": 6})

    def test_original_review_assertion_stops_before_legacy_catch_can_continue(self):
        """Modify one parsed fake refusal; the original check_refusal assert must fail."""
        definition = deepcopy(catalog.load_catalog()[0]["execution.input_boundaries"])
        unit = control.make_unit(definition, ROOT)
        directory = self.root / "real-review-injection"
        directory.mkdir()
        real_popen = subprocess.Popen
        injection = """
import sys
from lineage_agent.models import responses
from benchmarks.core.executors import control_worker

original = responses.parse_response
def alter_refusal_text(*args, **kwargs):
    result = original(*args, **kwargs)
    if result.get('has_refusal'):
        result['text'] = 'injected deterministic mismatch'
    return result

responses.parse_response = alter_refusal_text
print('TEST FAULT INJECTION: original Review refusal assertion', flush=True)
raise SystemExit(control_worker.main(sys.argv[1:]))
"""

        def launch(command, **kwargs):
            self.assertEqual(command[:3], [sys.executable, "-m", "benchmarks.core.executors.control_worker"])
            self.assertEqual(kwargs["env"]["LANGSMITH_TRACING"], "false")
            self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
            return real_popen([sys.executable, "-c", injection, *command[3:]], **kwargs)

        with patch.object(control.subprocess, "Popen", side_effect=launch):
            stopped = control.execute_unit(unit, definition, "real-review-injection", directory, root=ROOT)
        self.assertFalse(stopped)
        self.assertEqual((unit["execution_status"], unit["verdict"]), ("completed", "failed"), unit["issues"])
        self.assertEqual(unit["groups"][0]["verdict"], "passed")
        failed = unit["groups"][1]
        self.assertEqual(failed["group_id"], "execution.input_boundaries.refusal")
        self.assertEqual(failed["verdict"], "failed")
        self.assertEqual(failed["issues"][0]["stage"], "check_execution")
        self.assertEqual(failed["issues"][0]["exception_type"], "AssertionError")
        self.assertTrue(all(g["execution_status"] == "not_run" for g in unit["groups"][2:]))
        self.assertTrue(all(g["started_at"] is None for g in unit["groups"][2:]))
        folder = directory / definition["unit_id"]
        stdout = (folder / "stdout.txt").read_text(encoding="utf-8")
        stderr = (folder / "stderr.txt").read_text(encoding="utf-8")
        self.assertIn("invalid_output_7", stdout)
        self.assertNotIn("PASS check_direct_config", stdout)
        self.assertNotIn("FAIL check_refusal", stdout)
        self.assertIn("check_refusal", stderr)
        self.assertIn("AssertionError", stderr)
        logs = [e for e in unit["groups"][0]["evidence"] if e["kind"] == "original_log"]
        self.assertTrue(logs)
        self.assertTrue(all(Path(item["path"]).is_file() for item in logs))
        result = {
            "schema_version": "3", "run_id": "real-review-injection", "started_at": utc_now(),
            "ended_at": utc_now(), "selected_suites": ["control-execution"], "baseline": {},
            "units": [unit], "summary": summarize([unit]), "framework_errors": [],
        }
        code, saved = runner.persist_result(result, directory)
        self.assertEqual(code, 1)
        self.assertEqual({path.name for path in saved}, {"result.json", "report.md"})


if __name__ == "__main__":
    unittest.main()
