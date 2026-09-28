"""Shared metadata and persistence fault invariants retained from framework v1."""
from contextlib import redirect_stderr
from importlib import metadata
import io,json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from benchmarks.core import metadata as baseline_tools
from benchmarks.core.processes import safe_environment
from benchmarks.reporting import storage as runner
from benchmarks.core.contracts import exit_code
from benchmarks.tests.contract_examples import example_result,research_examples

class RunnerTests(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=self.directory=Path(temp.name)
    def unit(self):
        unit=research_examples()[0]
        unit.update(category="control",execution_mode="deterministic_stub",dimensions=[])
        unit.pop("turns")
        return unit

    def test_worker_environment_excludes_secrets_and_disables_tracing(self):
        environment = safe_environment(
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

    def test_missing_git_and_dependency_versions_are_explicitly_unknown(self):
        with (
            patch.object(baseline_tools.subprocess, "run", side_effect=FileNotFoundError("git missing")),
            patch.object(baseline_tools.metadata, "version", side_effect=metadata.PackageNotFoundError),
        ):
            baseline = baseline_tools.collect_baseline(self.root)
        missing = [baseline[key] for key in ("agent_commit", "working_tree_dirty", "lock_sha256")]
        missing.extend(baseline["dependencies"].values())
        self.assertTrue(missing)
        for field in missing:
            self.assertEqual(field["availability"], "unknown")
            self.assertIsNone(field["value"])
            self.assertTrue(field["reason"])
        self.assertEqual(baseline["python_version"]["availability"], "known")
