"""Exercise orchestration faults without launching the real ReAct check."""

from contextlib import redirect_stderr
from copy import deepcopy
from importlib import metadata
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from benchmarks import __main__ as cli
from benchmarks import runner
from benchmarks.results import exit_code
from benchmarks.tests.contract_examples import example_result


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.definition = deepcopy(runner.SUITES["control-react"])
        self.script = self.root / self.definition["script"]
        self.script.parent.mkdir()
        self.script.write_text("# Test fixture; never executed.\n", encoding="utf-8")
        self.directory = self.root / "artifacts"
        self.directory.mkdir()
        self.run_id = "fixture-benchmark-run"

    def unit(self):
        return runner.make_unit(self.definition, self.root)

    def receipt(self, unit):
        check_id = "c0000000-0000-4000-8000-000000000001"
        return {
            "schema_version": "1",
            "adapter_version": "1",
            "run_id": self.run_id,
            "unit_id": self.definition["unit_id"],
            "script_path": str(self.script),
            "script_sha256": unit["conditions"]["materials"]["value"]["script_sha256"]["value"],
            "phase": "finished",
            "outcome": "passed",
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:00:01+00:00",
            "check_id": check_id,
            "check_dir": str(self.root / "logs/checks" / check_id),
            "error": None,
        }

    def fake_launch(self, process, receipt):
        def launch(command, **kwargs):
            # Actual files prove evidence survives, even when the verdict is rejected.
            kwargs["stdout"].write(b"ALL CHECKS PASSED\npartial output retained\n")
            kwargs["stderr"].write(b"fixture diagnostic\n")
            if receipt is not None:
                receipt_path = Path(command[command.index("--receipt") + 1])
                text = receipt if isinstance(receipt, str) else json.dumps(receipt)
                receipt_path.write_text(text, encoding="utf-8")
            return process

        return launch

    def execute(self, unit, directory=None):
        return runner.execute_unit(
            unit,
            self.definition,
            self.run_id,
            directory or self.directory,
            root=self.root,
        )

    def test_unknown_cli_choice_does_not_start_or_create_artifacts(self):
        with (
            patch.object(cli, "run_benchmark") as run,
            patch.object(Path, "mkdir") as mkdir,
            redirect_stderr(io.StringIO()) as stderr,
        ):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["--suite", "unknown-suite"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unknown-suite", stderr.getvalue())
        run.assert_not_called()
        mkdir.assert_not_called()

    def test_worker_environment_excludes_secrets_and_disables_tracing(self):
        environment = runner.safe_environment(
            {
                "Path": "runtime-path",
                "SYSTEMROOT": "system-root",
                "TEMP": "temporary-directory",
                "VIRTUAL_ENV": "runtime-venv",
                "OPENAI_API_KEY": "credential-marker",
                "LINEAGE_API_KEY": "credential-marker",
                "LANGSMITH_API_KEY": "credential-marker",
                "LANGCHAIN_API_KEY": "credential-marker",
                "UNRELATED_PRIVATE_VALUE": "credential-marker",
                "LANGSMITH_TRACING": "true",
                "LANGCHAIN_TRACING": "true",
                "LANGCHAIN_TRACING_V2": "true",
                "PYTHONOPTIMIZE": "2",
                "PYTHONPATH": "untrusted-import-path",
            }
        )
        self.assertEqual(environment["Path"], "runtime-path")
        self.assertEqual(environment["SYSTEMROOT"], "system-root")
        self.assertEqual(environment["VIRTUAL_ENV"], "runtime-venv")
        self.assertNotIn("credential-marker", json.dumps(environment))
        self.assertNotIn("PYTHONOPTIMIZE", environment)
        self.assertNotIn("PYTHONPATH", environment)
        for key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2"):
            self.assertEqual(environment[key], "false")
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")

    def test_missing_entry_is_blocked_and_failed_launch_is_not_an_assertion(self):
        missing = self.unit()
        self.script.unlink()
        with patch.object(runner.subprocess, "Popen") as launch:
            self.assertFalse(self.execute(missing))
        launch.assert_not_called()
        self.assertEqual(missing["execution_status"], "blocked")
        self.assertEqual(missing["verdict"], "pending_review")
        self.assertFalse((self.directory / self.definition["unit_id"]).exists())

        self.script.write_text("# Restored fixture.\n", encoding="utf-8")
        failed = self.unit()
        with patch.object(runner.subprocess, "Popen", side_effect=OSError("fixture launch denied")):
            self.assertFalse(self.execute(failed))
        self.assertEqual(failed["execution_status"], "failed")
        self.assertEqual(failed["verdict"], "pending_review")
        self.assertEqual(failed["issues"][0]["stage"], "launch")
        self.assertEqual(failed["metrics"]["duration_seconds"]["availability"], "unknown")
        self.assertEqual(failed["evidence"][0]["availability"], "available")
        self.assertEqual(failed["evidence"][2]["availability"], "missing")

    def test_timeout_terminates_and_escalates_with_partial_evidence_preserved(self):
        unit = self.unit()
        receipt = self.receipt(unit)
        receipt.update(phase="check_execution", outcome="running", ended_at=None)
        Path(receipt["check_dir"]).mkdir(parents=True)
        process = Mock(returncode=-9)
        process.poll.return_value = None
        process.wait.side_effect = [
            subprocess.TimeoutExpired("fixture-worker", 120),
            subprocess.TimeoutExpired("fixture-worker", 5),
            -9,
        ]
        with patch.object(runner.subprocess, "Popen", side_effect=self.fake_launch(process, receipt)):
            self.assertFalse(self.execute(unit))
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 3)
        self.assertEqual(unit["execution_status"], "failed")
        self.assertEqual(unit["verdict"], "pending_review")
        self.assertEqual(unit["issues"][0]["code"], "timeout")
        stdout = next(item for item in unit["evidence"] if item["kind"] == "stdout")
        self.assertIn("partial output retained", Path(stdout["path"]).read_text())
        logs = next(item for item in unit["evidence"] if item["kind"] == "original_check_logs")
        self.assertEqual(logs["availability"], "available")
        self.assertEqual(Path(logs["path"]), Path(receipt["check_dir"]))

    def test_parent_interrupt_stops_worker_and_preserves_partial_output(self):
        unit = self.unit()
        process = Mock(returncode=-15)
        process.poll.return_value = None
        process.wait.side_effect = [KeyboardInterrupt(), -15]
        with patch.object(runner.subprocess, "Popen", side_effect=self.fake_launch(process, None)):
            self.assertTrue(self.execute(unit))
        process.terminate.assert_called_once()
        process.kill.assert_not_called()
        self.assertEqual(process.wait.call_count, 2)
        self.assertEqual(unit["execution_status"], "failed")
        self.assertEqual(unit["verdict"], "pending_review")
        self.assertEqual(unit["issues"][0]["code"], "interrupted")
        stdout = next(item for item in unit["evidence"] if item["kind"] == "stdout")
        self.assertIn("partial output retained", Path(stdout["path"]).read_text())

    def test_original_react_assertion_failure_reaches_saved_benchmark_result(self):
        """Change only a fake reply, then execute the original assertion and adapter."""
        unit = runner.make_unit(self.definition, runner.ROOT)
        real_popen = subprocess.Popen
        injection = """
import sys
from benchmarks import react_worker

original_loader = react_worker._load_script
def load_with_wrong_fake_reply(path):
    module = original_loader(path)
    original_message = module.message
    module.message = lambda *args, **kwargs: original_message('injected-mismatch')
    print('TEST FAULT INJECTION: deterministic reply text mismatch', flush=True)
    return module

react_worker._load_script = load_with_wrong_fake_reply
sys.argv = ['worker', *sys.argv[1:]]
raise SystemExit(react_worker.main())
"""

        def spawn_injected(command, **kwargs):
            self.assertEqual(command[:3], [sys.executable, "-m", "benchmarks.react_worker"])
            self.assertEqual(kwargs["cwd"], runner.ROOT)
            self.assertEqual(kwargs["env"]["LANGSMITH_TRACING"], "false")
            self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
            return real_popen([sys.executable, "-c", injection, *command[3:]], **kwargs)

        with patch.object(runner.subprocess, "Popen", side_effect=spawn_injected):
            interrupted = runner.execute_unit(
                unit, self.definition, self.run_id, self.directory, root=runner.ROOT
            )
        self.assertFalse(interrupted)
        self.assertEqual(unit["execution_status"], "completed", unit["issues"])
        self.assertEqual(unit["verdict"], "failed")
        self.assertEqual(unit["issues"][0]["exception_type"], "AssertionError")
        evidence = {item["kind"]: item for item in unit["evidence"]}
        stdout = Path(evidence["stdout"]["path"]).read_text(encoding="utf-8")
        stderr = Path(evidence["stderr"]["path"]).read_text(encoding="utf-8")
        self.assertIn("CHECK check_direct_answer", stdout)
        self.assertIn("TEST FAULT INJECTION", stdout)
        self.assertNotIn("PASS check_direct_answer", stdout)
        self.assertIn("check_direct_answer", stderr)
        self.assertIn("AssertionError", stderr)
        receipt = json.loads(Path(evidence["receipt"]["path"]).read_text(encoding="utf-8"))
        self.assertEqual(receipt["outcome"], "assertion_failed")
        self.assertEqual(receipt["script_path"], str((runner.ROOT / self.definition["script"]).resolve()))
        log_directory = Path(evidence["original_check_logs"]["path"])
        self.assertEqual(log_directory.name, receipt["check_id"])
        self.assertTrue(list(log_directory.glob("*.jsonl")))
        self.assertEqual(evidence["original_check_logs"]["availability"], "available")
        result = example_result([unit])
        result["run_id"] = self.run_id
        code, saved = runner.persist_result(result, self.directory)
        self.assertEqual(code, 1)
        self.assertEqual({path.name for path in saved}, {"result.json", "report.md"})
        saved_result = json.loads((self.directory / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(exit_code(saved_result), 1)
        self.assertEqual(saved_result["units"][0]["verdict"], "failed")

    def test_success_text_cannot_rescue_invalid_receipts_or_process_exit(self):
        cases = (
            ("different-run", {"run_id": "previous-run"}, 0),
            ("different-adapter", {"adapter_version": "999"}, 0),
            ("different-script", {"script_sha256": "unrelated-content-hash"}, 0),
            ("exit-disagrees", {}, 1),
            ("malformed-json", None, 0),
        )
        for name, modifications, code in cases:
            with self.subTest(name=name):
                directory = self.directory / name
                directory.mkdir()
                unit = self.unit()
                receipt = self.receipt(unit)
                if modifications is None:
                    receipt = "this is not JSON"
                else:
                    receipt.update(modifications)
                process = Mock(returncode=code)
                process.poll.return_value = code
                process.wait.return_value = code
                with patch.object(runner.subprocess, "Popen", side_effect=self.fake_launch(process, receipt)):
                    self.assertFalse(self.execute(unit, directory))
                self.assertEqual(unit["execution_status"], "failed")
                self.assertEqual(unit["verdict"], "pending_review")
                self.assertEqual(unit["issues"][0]["code"], "invalid_receipt")
                stdout = next(e for e in unit["evidence"] if e["kind"] == "stdout")
                self.assertIn("ALL CHECKS PASSED", Path(stdout["path"]).read_text())

    def test_bound_assertion_receipt_fails_despite_success_text(self):
        unit = self.unit()
        receipt = self.receipt(unit)
        receipt.update(
            outcome="assertion_failed",
            error={
                "stage": "check_execution",
                "code": "assertion_failed",
                "exception_type": "AssertionError",
            },
        )
        process = Mock(returncode=1)
        process.poll.return_value = 1
        process.wait.return_value = 1
        with patch.object(runner.subprocess, "Popen", side_effect=self.fake_launch(process, receipt)):
            self.execute(unit)
        self.assertEqual(unit["execution_status"], "completed")
        self.assertEqual(unit["verdict"], "failed")
        self.assertEqual(exit_code(example_result([unit])), 1)
        self.assertTrue(Path(unit["issues"][0]["evidence"][0]).is_file())

    def test_assertion_verdict_requires_check_stage_and_assertion_exception(self):
        invalid_errors = (
            {},
            {"stage": "import", "code": "assertion_failed", "exception_type": "AssertionError"},
            {"stage": "check_execution", "code": "assertion_failed", "exception_type": "RuntimeError"},
        )
        for index, error in enumerate(invalid_errors):
            with self.subTest(error=error):
                directory = self.directory / f"invalid-assertion-{index}"
                directory.mkdir()
                unit = self.unit()
                receipt = self.receipt(unit)
                receipt.update(outcome="assertion_failed", error=error)
                process = Mock(returncode=1)
                process.poll.return_value = 1
                process.wait.return_value = 1
                with patch.object(runner.subprocess, "Popen", side_effect=self.fake_launch(process, receipt)):
                    self.execute(unit, directory)
                self.assertEqual(unit["execution_status"], "failed")
                self.assertEqual(unit["verdict"], "pending_review")
                self.assertEqual(unit["issues"][0]["code"], "invalid_receipt")

    def test_artifact_save_failures_keep_verdict_and_only_list_saved_files(self):
        real_json = runner.atomic_write_json
        real_text = runner.atomic_write_text
        for failing_name in ("result.json", "report.md"):
            with self.subTest(failing_name=failing_name):
                directory = self.directory / failing_name.replace(".", "-")
                directory.mkdir()
                evidence = directory / "stdout.txt"
                evidence.write_text("original check output", encoding="utf-8")
                unit = self.unit()
                unit.update(execution_status="completed", verdict="passed")
                unit["evidence"] = [{
                    "kind": "stdout", "path": str(evidence),
                    "source": "fixture worker", "availability": "available",
                }]
                result = example_result([unit])

                def write_json(path, value):
                    if path.name == failing_name:
                        raise OSError("injected result write failure")
                    real_json(path, value)

                def write_text(path, value):
                    if path.name == failing_name:
                        raise OSError("injected report write failure")
                    real_text(path, value)

                with (
                    patch.object(runner, "atomic_write_json", side_effect=write_json),
                    patch.object(runner, "atomic_write_text", side_effect=write_text),
                    redirect_stderr(io.StringIO()) as stderr,
                ):
                    code, saved = runner.persist_result(result, directory)
                self.assertEqual(code, 2)
                self.assertEqual(result["units"][0]["verdict"], "passed")
                self.assertTrue(result["framework_errors"])
                self.assertIn("保存失败", stderr.getvalue())
                self.assertEqual(evidence.read_text(), "original check output")
                expected_saved = {"report.md"} if failing_name == "result.json" else {"result.json"}
                self.assertEqual({path.name for path in saved}, expected_saved)
                self.assertTrue(all(path.is_file() for path in saved))
                self.assertFalse((directory / failing_name).exists())
                if failing_name == "report.md":
                    saved_result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
                    self.assertEqual(exit_code(saved_result), 2)
                    self.assertEqual(saved_result["units"][0]["verdict"], "passed")

    def test_unavailable_run_directory_never_starts_check(self):
        with (
            patch.object(runner, "collect_baseline", return_value={}),
            patch.object(Path, "mkdir", side_effect=PermissionError("injected directory failure")),
            patch.object(runner, "execute_unit") as execute,
            redirect_stderr(io.StringIO()) as stderr,
        ):
            result, saved = runner.run_benchmark(["control-react"], root=self.root)
        execute.assert_not_called()
        self.assertEqual(saved, [])
        self.assertEqual(exit_code(result), 2)
        self.assertEqual(result["units"][0]["execution_status"], "not_run")
        self.assertIn("没有已保存的报告", stderr.getvalue())

    def test_missing_git_and_dependency_versions_are_explicitly_unknown(self):
        with (
            patch.object(runner.subprocess, "run", side_effect=FileNotFoundError("git missing")),
            patch.object(runner.metadata, "version", side_effect=metadata.PackageNotFoundError),
        ):
            baseline = runner.collect_baseline(self.root)
        missing = [baseline[key] for key in ("agent_commit", "working_tree_dirty", "lock_sha256")]
        missing.extend(baseline["dependencies"].values())
        self.assertTrue(missing)
        for field in missing:
            self.assertEqual(field["availability"], "unknown")
            self.assertIsNone(field["value"])
            self.assertTrue(field["reason"])
        self.assertEqual(baseline["python_version"]["availability"], "known")


if __name__ == "__main__":
    unittest.main()
