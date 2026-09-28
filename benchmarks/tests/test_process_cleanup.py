"""Verify owned process cleanup, including an actual Windows descendant."""

from copy import deepcopy
import ctypes
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

from benchmarks.core import processes as runner
from benchmarks.core.executors import control
from benchmarks.suites.agent_contracts.suite import load_catalog


class ProcessCleanupTests(unittest.TestCase):
    def test_tree_command_failure_still_reaps_direct_child_and_reports_uncertainty(self):
        for failure in (OSError("taskkill unavailable"), subprocess.TimeoutExpired("taskkill", 5), None):
            with self.subTest(failure=type(failure).__name__ if failure else "nonzero-exit"):
                process = Mock(pid=123456)
                process.poll.return_value = None
                command_result = Mock(returncode=1)
                with patch.object(runner.os, "name", "nt"), patch.object(
                    runner.subprocess, "run", side_effect=failure, return_value=command_result,
                ) as terminate_tree:
                    with self.assertRaises(runner.ProcessCleanupError):
                        runner._stop(process)
                terminate_tree.assert_called_once_with(
                    ["taskkill", "/PID", "123456", "/T", "/F"], capture_output=True, timeout=5,
                )
                process.terminate.assert_called_once_with()
                process.wait.assert_called_once_with(timeout=5)
                process.kill.assert_not_called()

    def test_direct_child_kill_fallback_does_not_claim_tree_success(self):
        process = Mock(pid=123456)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 5), 0]
        with patch.object(runner.os, "name", "nt"), patch.object(
            runner.subprocess, "run", side_effect=OSError("taskkill unavailable"),
        ), self.assertRaises(runner.ProcessCleanupError):
            runner._stop(process)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_count, 2)

    def test_successful_tree_termination_waits_for_owned_worker(self):
        process = Mock(pid=123456)
        process.poll.return_value = None
        with patch.object(runner.os, "name", "nt"), patch.object(
            runner.subprocess, "run", return_value=Mock(returncode=0),
        ):
            runner._stop(process)
        process.wait.assert_called_once_with(timeout=5)
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_already_exited_worker_does_not_terminate_an_unowned_pid(self):
        process = Mock(pid=123456)
        process.poll.return_value = 0
        with patch.object(runner.subprocess, "run") as terminate_tree:
            runner._stop(process)
        terminate_tree.assert_not_called()
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_uncertain_cleanup_stops_scheduling_and_retains_timeout_fact(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            definition = deepcopy(load_catalog()[0]["execution.react"])
            script = root / definition["script"]
            script.parent.mkdir(parents=True)
            script.write_text("# Never executed by this orchestration fixture.\n", encoding="utf-8")
            directory = root / "artifacts"
            directory.mkdir()
            unit = control.make_unit(definition, root)
            process = Mock(pid=123456, returncode=-1)
            process.wait.side_effect = subprocess.TimeoutExpired("worker", definition["timeout_seconds"])

            def launch(command, **kwargs):
                kwargs["stdout"].write(b"actual partial output retained\n")
                return process

            with patch.object(runner.subprocess, "Popen", side_effect=launch), patch.object(
                control, "_stop", side_effect=runner.ProcessCleanupError("tree not confirmed"),
            ):
                stop_scheduling = control.execute_unit(unit, definition, "fixture-run", directory, root=root)
            self.assertTrue(stop_scheduling)
            self.assertEqual(unit["execution_status"], "failed")
            self.assertEqual(unit["verdict"], "pending_review")
            self.assertTrue({"timeout", "process_tree_unconfirmed"} <= {item["code"] for item in unit["issues"]})
            output = directory / definition["unit_id"] / "stdout.txt"
            self.assertEqual(output.read_text(encoding="utf-8"), "actual partial output retained\n")

    @unittest.skipUnless(os.name == "nt", "Exercises Windows taskkill /T and process handles")
    def test_real_windows_timeout_terminates_worker_and_sleeping_descendant(self):
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_path = root / "descendant.pid"
            worker_code = (
                "from pathlib import Path; import subprocess, sys, time; "
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
                "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
                "Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8'); time.sleep(60)"
            )
            process = subprocess.Popen(
                [sys.executable, "-c", worker_code, str(pid_path)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            child_handle = None
            try:
                deadline = time.monotonic() + 5
                while not pid_path.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(pid_path.exists(), "worker did not record its own descendant")
                child_pid = int(pid_path.read_text(encoding="utf-8"))
                # Hold an owned-process handle before termination; PID reuse cannot
                # make the test inspect or clean up a different process later.
                child_handle = kernel.OpenProcess(0x00100000 | 0x0001, False, child_pid)
                self.assertTrue(child_handle, f"OpenProcess failed: {ctypes.get_last_error()}")
                self.assertEqual(kernel.WaitForSingleObject(child_handle, 0), 258)
                with self.assertRaises(subprocess.TimeoutExpired):
                    process.wait(timeout=0.05)
                runner._stop(process)
                self.assertIsNotNone(process.poll())
                self.assertEqual(kernel.WaitForSingleObject(child_handle, 5000), 0,
                                 "descendant survived the owned worker's timeout cleanup")
            finally:
                if child_handle:
                    if kernel.WaitForSingleObject(child_handle, 0) == 258:
                        kernel.TerminateProcess(child_handle, 1)
                        kernel.WaitForSingleObject(child_handle, 5000)
                    kernel.CloseHandle(child_handle)
                if process.poll() is None:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, timeout=5)
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
