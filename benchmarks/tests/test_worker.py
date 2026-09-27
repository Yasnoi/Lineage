"""Fault-injection harness checks; these are not Research evaluations."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from benchmarks import react_worker
from benchmarks.common import atomic_write_json


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="benchmark-worker-harness-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.script = self.directory / "harness_check.py"
        self.receipt = self.directory / "receipt.json"

    def write_script(self, body):
        self.script.write_text(
            "from pathlib import Path\n"
            "CHECK_ID = 'harness-check'\n"
            "CHECK_DIR = Path(__file__).parent / 'harness-logs'\n" + body,
            encoding="utf-8",
        )

    def execute(self):
        output, error = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = react_worker.execute(
                "harness-run", "harness-unit", self.receipt, script_path=self.script
            )
        receipt = json.loads(self.receipt.read_text(encoding="utf-8")) if self.receipt.exists() else None
        return code, receipt, output.getvalue(), error.getvalue()

    def test_success_and_start_receipts(self):
        self.write_script("def main():\n    print('harness check ran')\n")
        snapshots = []

        def record(path, value):
            snapshots.append(dict(value))
            atomic_write_json(path, value)

        with patch.object(react_worker, "atomic_write_json", side_effect=record):
            code, receipt, output, error = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(error, "")
        self.assertIn("harness check ran", output)
        self.assertEqual(receipt["outcome"], "passed")
        self.assertEqual(receipt["phase"], "finished")
        self.assertEqual(receipt["check_id"], "harness-check")
        self.assertEqual(receipt["check_dir"], str(self.directory / "harness-logs"))
        self.assertEqual(receipt["run_id"], "harness-run")
        self.assertEqual(receipt["unit_id"], "harness-unit")
        self.assertEqual(len(receipt["script_sha256"]), 64)
        self.assertIsNotNone(receipt["ended_at"])
        self.assertEqual(
            [item["phase"] for item in snapshots],
            ["preflight", "import", "check_execution", "finished"],
        )
        self.assertEqual(snapshots[2]["outcome"], "started")

    def test_assertion_failure_retains_log_identity_and_traceback(self):
        self.write_script(
            "def main():\n"
            "    CHECK_DIR.mkdir()\n"
            "    (CHECK_DIR / 'partial.jsonl').write_text('harness evidence')\n"
            "    print('all passed')\n"
            "    assert False, 'harness assertion'\n"
        )
        code, receipt, output, error = self.execute()
        self.assertEqual(code, 1)
        self.assertEqual(receipt["outcome"], "assertion_failed")
        self.assertEqual(receipt["error"]["stage"], "check_execution")
        self.assertEqual(receipt["check_id"], "harness-check")
        self.assertTrue((Path(receipt["check_dir"]) / "partial.jsonl").is_file())
        self.assertIn("all passed", output)
        self.assertIn("AssertionError: harness assertion", error)

    def test_unexpected_error_is_not_an_assertion_result(self):
        self.write_script("def main():\n    raise ValueError('harness error')\n")
        code, receipt, _, error = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["outcome"], "error")
        self.assertEqual(receipt["error"]["exception_type"], "ValueError")
        self.assertEqual(receipt["check_id"], "harness-check")
        self.assertIn("ValueError: harness error", error)

    def test_import_assertion_is_adapter_error(self):
        self.write_script("assert False, 'import harness error'\n")
        code, receipt, _, error = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["outcome"], "error")
        self.assertEqual(receipt["error"]["stage"], "import")
        self.assertIsNone(receipt["check_id"])
        self.assertIn("import harness error", error)

    def test_missing_script_is_preflight_error(self):
        code, receipt, _, _ = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["error"]["stage"], "preflight")
        self.assertIsNone(receipt["script_sha256"])

    def test_initial_receipt_write_failure_does_not_run_script(self):
        self.write_script("def main():\n    print('harness ran')\n")
        with patch.object(react_worker, "atomic_write_json", side_effect=OSError("harness disk failure")):
            code, receipt, output, error = self.execute()
        self.assertEqual(code, 2)
        self.assertIsNone(receipt)
        self.assertEqual(output, "")
        self.assertIn("could not save", error)

    def test_final_receipt_write_failure_retains_started_evidence(self):
        self.write_script("def main():\n    print('harness ran')\n")

        def fail_final(path, value):
            if value["phase"] == "finished":
                raise OSError("harness final write failure")
            atomic_write_json(path, value)

        with patch.object(react_worker, "atomic_write_json", side_effect=fail_final):
            code, receipt, output, error = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["outcome"], "started")
        self.assertEqual(receipt["check_id"], "harness-check")
        self.assertIn("harness ran", output)
        self.assertIn("could not save", error)

    def test_existing_receipt_is_never_reused(self):
        self.write_script("def main():\n    print('harness ran')\n")
        self.receipt.write_text('{"previous": true}', encoding="utf-8")
        code, receipt, output, error = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt, {"previous": True})
        self.assertEqual(output, "")
        self.assertIn("existing receipt", error)

    def test_optimized_python_refuses_before_import(self):
        process = subprocess.run(
            [
                sys.executable, "-O", "-m", "benchmarks.react_worker",
                "--run-id", "harness-run", "--unit-id", "harness-unit",
                "--receipt", str(self.receipt),
            ],
            cwd=react_worker.SCRIPT_PATH.parents[1],
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(process.returncode, 2)
        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        self.assertEqual(receipt["outcome"], "error")
        self.assertEqual(receipt["error"]["stage"], "preflight")
        self.assertIsNone(receipt["script_sha256"])
        self.assertIsNone(receipt["check_id"])
        self.assertIn("assertions enabled", process.stderr)

    def test_fixed_adapter_keeps_local_source_first_and_restores_search_path(self):
        self.write_script("def main():\n    import sys\n    print(sys.path[0])\n")
        original = list(sys.path)
        with (
            patch.object(react_worker, "SCRIPT_PATH", self.script),
            patch.object(react_worker, "_require_local_agent") as verify,
        ):
            code, _, output, _ = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(output.strip(), str(react_worker.ROOT / "src"))
        self.assertEqual(sys.path, original)

    def test_cached_external_agent_is_rejected_before_check(self):
        self.write_script("def main():\n    print('harness ran')\n")
        external = ModuleType("lineage_agent")
        external.__file__ = str(self.directory / "foreign_agent/__init__.py")
        original = list(sys.path)
        with (
            patch.object(react_worker, "SCRIPT_PATH", self.script),
            patch.dict(sys.modules, {"lineage_agent": external}),
        ):
            code, receipt, output, error = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["error"]["stage"], "import")
        self.assertEqual(receipt["error"]["exception_type"], "ImportError")
        self.assertEqual(receipt["outcome"], "error")
        self.assertEqual(output, "")
        self.assertIn("outside the current repository", error)
        self.assertEqual(sys.path, original)

    def test_cached_external_runtime_is_rejected(self):
        local = ModuleType("lineage_agent")
        local.__file__ = str(react_worker.ROOT / "src/lineage_agent/__init__.py")
        foreign = ModuleType("lineage_agent.agent.runtime")
        foreign.__file__ = str(self.directory / "foreign_agent/agent/runtime.py")
        with patch.dict(sys.modules, {"lineage_agent": local, "lineage_agent.agent.runtime": foreign}):
            with self.assertRaisesRegex(ImportError, "lineage_agent.agent.runtime"):
                react_worker._require_local_agent()


if __name__ == "__main__":
    unittest.main()
