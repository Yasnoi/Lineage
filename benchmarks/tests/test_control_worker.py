"""Trusted worker lifecycle checks with synthetic scripts, never Research tasks."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from benchmarks import control_worker
from benchmarks.common import ROOT, atomic_write_json, sha256_file
from benchmarks.groups import GroupJournal, plan_digest, read_group_events
from benchmarks.runner import safe_environment


class ControlWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="benchmark-control-worker-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.script = self.directory / "trusted_fixture.py"
        self.receipt_path = self.directory / "receipt.json"
        self.events_path = self.directory / "groups.jsonl"
        self.definition = {
            "unit_id": "trusted-fixture", "script": str(self.script),
            "groups": [self.group("async_main"), self.group("after_main_cli")],
        }
        # Only this in-process test seam extends the catalog. CLI consumers
        # cannot supply a script or command in the production worker.
        self.catalog_patch = patch.object(
            control_worker, "load_catalog", return_value=({"trusted-fixture": self.definition}, {})
        )
        self.catalog_patch.start()
        self.addCleanup(self.catalog_patch.stop)

    def group(self, name):
        return {
            "group_id": f"worker-fixture/{name}", "description": "Synthetic worker check",
            "source": {"path": str(self.script), "symbol": name},
            "primary_area": "execution", "tags": ["execution"],
        }

    def write_script(self, source=None):
        self.script.write_text(source or (
            "import asyncio\n"
            "import sys\n"
            "from check_support import check_group\n"
            "async def main():\n"
            "    with check_group('worker-fixture/async_main'):\n"
            "        await asyncio.sleep(0)\n"
            "        print('async main completed')\n"
            "def cli_check():\n"
            "    with check_group('worker-fixture/after_main_cli'):\n"
            "        assert len(sys.argv) == 1\n"
            "        print('CLI after main completed')\n"
            "if __name__ == '__main__':\n"
            "    asyncio.run(main())\n"
            "    cli_check()\n"
        ), encoding="utf-8")

    def execute(self):
        output, error = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = control_worker.execute("worker-review", "trusted-fixture", self.receipt_path)
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8")) if self.receipt_path.exists() else None
        return code, receipt, output.getvalue(), error.getvalue()

    def read_groups(self):
        return read_group_events(self.events_path, "worker-review", self.definition)

    def test_entire_main_entrypoint_runs_async_main_and_following_cli(self):
        self.write_script()
        original_path, original_argv = list(sys.path), sys.argv
        code, receipt, output, error = self.execute()
        self.assertEqual((code, error), (0, ""))
        self.assertIn("async main completed", output)
        self.assertIn("CLI after main completed", output)
        self.assertEqual(receipt["outcome"], "passed")
        self.assertEqual(receipt["phase"], "finished")
        self.assertEqual(receipt["script_sha256"], sha256_file(self.script))
        self.assertEqual(receipt["plan_digest"], plan_digest(self.definition["groups"]))
        self.assertEqual(receipt["events_sha256"], sha256_file(self.events_path))
        groups, errors, count = self.read_groups()
        self.assertEqual(errors, [])
        self.assertEqual([g["verdict"] for g in groups], ["passed", "passed"])
        self.assertEqual(receipt["events_count"], count)
        self.assertEqual(sys.path, original_path)
        self.assertIs(sys.argv, original_argv)

    def test_initial_journal_failure_is_fatal_recording_failure(self):
        self.write_script()
        with patch.object(control_worker, "GroupJournal", side_effect=OSError("journal unavailable")), \
                patch.object(control_worker, "_execute_script") as script:
            code, receipt, _, _ = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["outcome"], "recording_failed")
        script.assert_not_called()

    def test_worker_interrupt_is_distinct_from_an_independent_script_error(self):
        self.write_script("from check_support import check_group\n"
                          "with check_group('worker-fixture/async_main'):\n"
                          "    raise KeyboardInterrupt()\n")
        code, receipt, _, _ = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["outcome"], "interrupted")
        groups, _, _ = self.read_groups()
        self.assertEqual(groups[0]["verdict"], "pending_review")
        self.assertEqual(groups[1]["execution_status"], "not_run")

    def test_changed_source_cannot_receive_a_passed_receipt(self):
        self.write_script()
        source = self.script.read_text(encoding="utf-8")
        self.script.write_text(source + "\nfrom pathlib import Path\nPath(__file__).write_text('# changed')\n", encoding="utf-8")
        code, receipt, _, _ = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["outcome"], "error")
        self.assertNotEqual(receipt["script_sha256"], sha256_file(self.script))

    def test_cli_rejects_unknown_unit_and_arbitrary_command_options(self):
        for extra in (["--unit-id", "../untrusted.py"],
                      ["--unit-id", "trusted-fixture", "--script", str(self.script)],
                      ["--unit-id", "trusted-fixture", "--command", "python"]):
            with self.subTest(arguments=extra), redirect_stderr(StringIO()), patch.object(control_worker, "execute") as execute:
                with self.assertRaises(SystemExit) as stopped:
                    control_worker.main(["--run-id", "worker-review", "--receipt", str(self.receipt_path), *extra])
                self.assertEqual(stopped.exception.code, 2)
                execute.assert_not_called()
        self.assertFalse(self.receipt_path.exists())
        self.assertFalse(self.events_path.exists())

    def test_foreign_cached_agent_source_is_rejected_before_script(self):
        self.write_script()
        foreign = ModuleType("lineage_agent")
        foreign.__file__ = str(self.directory / "foreign" / "__init__.py")
        with patch.object(control_worker, "import_module", return_value=foreign), patch.object(control_worker, "_execute_script") as execute:
            code, receipt, _, _ = self.execute()
        execute.assert_not_called()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["error"]["stage"], "import")
        self.assertEqual(receipt["error"]["exception_type"], "ImportError")

    def test_foreign_check_support_is_rejected_before_script(self):
        self.write_script()
        foreign = ModuleType("check_support")
        foreign.__file__ = str(self.directory / "foreign_support.py")
        with patch.object(control_worker, "_require_local_agent"), patch.object(
            control_worker, "import_module", return_value=foreign
        ), patch.object(control_worker, "_execute_script") as execute:
            code, receipt, _, _ = self.execute()
        execute.assert_not_called()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["error"]["exception_type"], "ImportError")

    def test_module_assertion_before_first_group_is_execution_error(self):
        self.write_script("assert False, 'synthetic import failure'\n")
        code, receipt, _, error = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["outcome"], "error")
        self.assertEqual(receipt["error"]["exception_type"], "AssertionError")
        self.assertIn("synthetic import failure", error)
        groups, errors, _ = self.read_groups()
        self.assertEqual(errors, [])
        self.assertEqual([g["execution_status"] for g in groups], ["not_run", "not_run"])

    def test_early_system_exit_zero_does_not_confirm_declared_groups(self):
        self.write_script("raise SystemExit(0)\n")
        code, receipt, _, _ = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["outcome"], "error")
        self.assertEqual(receipt["error"]["exception_type"], "SystemExit")
        self.assertTrue(all(group["execution_status"] == "not_run" for group in self.read_groups()[0]))

    def test_early_return_keeps_finished_group_and_marks_remaining_not_run(self):
        self.write_script(
            "from check_support import check_group\n"
            "with check_group('worker-fixture/async_main'):\n"
            "    print('all passed')\n"
        )
        code, receipt, output, _ = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["outcome"], "error")
        self.assertIn("all passed", output)
        groups, errors, _ = self.read_groups()
        self.assertEqual(errors, [])
        self.assertEqual([group["verdict"] for group in groups], ["passed", "pending_review"])
        self.assertEqual(groups[1]["execution_status"], "not_run")

    def test_wrong_group_order_is_recording_failure_before_body(self):
        self.write_script(
            "from check_support import check_group\n"
            "with check_group('worker-fixture/after_main_cli'):\n"
            "    print('must not execute')\n"
        )
        code, receipt, output, _ = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["outcome"], "recording_failed")
        self.assertNotIn("must not execute", output)
        self.assertTrue(all(group["execution_status"] == "not_run" for group in self.read_groups()[0]))

    def test_assertion_stops_remaining_groups_even_with_legacy_exception_catch(self):
        self.write_script(
            "from check_support import check_group\n"
            "try:\n"
            "    with check_group('worker-fixture/async_main'):\n"
            "        print('all passed')\n"
            "        assert False, 'group assertion'\n"
            "except Exception:\n"
            "    print('legacy catch swallowed failure')\n"
            "with check_group('worker-fixture/after_main_cli'):\n"
            "    print('must not execute')\n"
        )
        code, receipt, output, error = self.execute()
        self.assertEqual(code, 1)
        self.assertEqual(receipt["outcome"], "assertion_failed")
        self.assertNotIn("legacy catch swallowed", output)
        self.assertNotIn("must not execute", output)
        self.assertIn("group assertion", error)
        groups, errors, _ = self.read_groups()
        self.assertEqual(errors, [])
        self.assertEqual([group["verdict"] for group in groups], ["failed", "pending_review"])
        self.assertEqual(groups[1]["execution_status"], "not_run")

    def test_group_recording_failure_stops_execution_and_keeps_started_prefix(self):
        self.write_script()
        original_emit = GroupJournal.emit

        def fail_group_end(journal, event, group_id, payload):
            if event == "group_ended":
                raise OSError("synthetic journal write failure")
            return original_emit(journal, event, group_id, payload)

        with patch.object(GroupJournal, "emit", new=fail_group_end):
            code, receipt, output, _ = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["outcome"], "recording_failed")
        self.assertIn("async main completed", output)
        self.assertNotIn("CLI after main completed", output)
        groups, errors, _ = self.read_groups()
        self.assertEqual(errors, [])
        self.assertEqual([group["execution_status"] for group in groups], ["failed", "not_run"])
        self.assertEqual([group["verdict"] for group in groups], ["pending_review", "pending_review"])

    def test_initial_receipt_failure_prevents_script_and_journal_creation(self):
        self.write_script()
        with patch.object(control_worker, "atomic_write_json", side_effect=OSError("synthetic disk failure")), patch.object(
            control_worker, "_execute_script"
        ) as execute:
            code, receipt, output, error = self.execute()
        execute.assert_not_called()
        self.assertEqual(code, 2)
        self.assertIsNone(receipt)
        self.assertEqual(output, "")
        self.assertFalse(self.events_path.exists())
        self.assertIn("Could not save final worker receipt", error)

    def test_final_receipt_failure_preserves_completed_event_evidence(self):
        self.write_script()

        def fail_final(path, receipt):
            if receipt["phase"] == "finished":
                raise OSError("synthetic final receipt failure")
            atomic_write_json(path, receipt)

        with patch.object(control_worker, "atomic_write_json", side_effect=fail_final):
            code, receipt, output, error = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["outcome"], "started")
        self.assertIn("CLI after main completed", output)
        self.assertIn("Could not save final worker receipt", error)
        groups, errors, _ = self.read_groups()
        self.assertEqual(errors, [])
        self.assertTrue(all(group["verdict"] == "passed" for group in groups))

    def test_previous_group_events_are_not_reused(self):
        self.write_script()
        previous = b'{"previous":true}\n'
        self.events_path.write_bytes(previous)
        with patch.object(control_worker, "_execute_script") as execute:
            code, receipt, output, _ = self.execute()
        execute.assert_not_called()
        self.assertEqual(code, 2)
        self.assertIsNone(receipt)
        self.assertEqual(output, "")
        self.assertEqual(self.events_path.read_bytes(), previous)

    def test_environment_gap_is_recorded_as_incomplete_coverage(self):
        self.write_script(
            "from check_support import check_group, coverage_gap\n"
            "with check_group('worker-fixture/async_main'):\n"
            "    coverage_gap('fixture-capability', 'synthetic unavailable capability')\n"
            "with check_group('worker-fixture/after_main_cli'):\n"
            "    pass\n"
        )
        code, receipt, _, _ = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(receipt["outcome"], "coverage_gap")
        groups, errors, _ = self.read_groups()
        self.assertEqual(errors, [])
        self.assertEqual(groups[0]["verdict"], "pending_review")
        self.assertEqual(groups[0]["coverage_gaps"][0]["scenario"], "fixture-capability")
        self.assertEqual(groups[1]["verdict"], "passed")

    def test_sync_decorator_and_async_context_share_correct_group_identity(self):
        self.write_script(
            "import asyncio\n"
            "from check_support import check_group, observe_harness\n"
            "@check_group('worker-fixture/async_main')\n"
            "def decorated():\n"
            "    observe_harness('projection', scenario='sync decorated call')\n"
            "async def continuation():\n"
            "    with check_group('worker-fixture/after_main_cli'):\n"
            "        await asyncio.sleep(0)\n"
            "        observe_harness('trace', scenario='async continuation')\n"
            "decorated()\n"
            "asyncio.run(continuation())\n"
        )
        code, _, _, error = self.execute()
        self.assertEqual((code, error), (0, ""))
        groups, errors, _ = self.read_groups()
        self.assertEqual(errors, [])
        self.assertEqual(groups[0]["observations"][0]["kind"], "projection")
        self.assertEqual(groups[1]["observations"][0]["kind"], "trace")

    def test_child_process_does_not_inherit_parent_observer_activation(self):
        with control_worker._script_context(ROOT / "scripts/check_tracing.py"):
            support = control_worker.import_module("check_support")

        class RejectingObserver:
            def emit(self, *args):
                raise AssertionError("child must not publish into parent observer")

        source = (
            "import sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from check_support import check_group, observe_harness\n"
            "with check_group('child-only-group'):\n"
            "    observe_harness('child_process', returncode=0)\n"
            "print('standalone child completed')\n"
        )
        with support.activate(RejectingObserver()):
            process = subprocess.run(
                [sys.executable, "-c", source, str(ROOT / "scripts")],
                cwd=ROOT, env=safe_environment(dict(os.environ)),
                capture_output=True, text=True, encoding="utf-8", timeout=20,
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "standalone child completed")
        self.assertEqual(process.stderr, "")


if __name__ == "__main__":
    unittest.main()
