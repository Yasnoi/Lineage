"""Real process boundaries, crash-released writer locks and explicit recovery."""

import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest


class ConversationProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="conversation-process-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.environment = {**os.environ, "PYTHONUTF8": "1", "LANGSMITH_TRACING": "false",
                            "LANGCHAIN_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}

    def command(self, mode):
        return [sys.executable, "-m", "benchmarks.checks.support.recovery_worker", mode, str(self.root)]

    def worker(self, mode, payload=None):
        result = subprocess.run(self.command(mode), input=json.dumps(payload or {}) + "\n",
                                capture_output=True, text=True, encoding="utf-8", env=self.environment, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 1, result.stdout)
        return json.loads(lines[0])

    def test_published_history_summaries_artifacts_and_cursor_survive_abrupt_process_exit(self):
        created = self.worker("create")
        self.assertEqual(created["status"], "created")
        self.assertEqual(len(created["run_ids"]), 2)
        self.assertTrue(created["summary_digests"])
        self.assertTrue(all(count > 0 for count in created["summary_calls"]))
        continued = self.worker("reopen", created)
        self.assertEqual(continued["status"], "continued")
        self.assertEqual(continued["snapshots"], 3)
        self.assertEqual(continued["old_records"], 2)
        self.assertTrue(continued["cursor_preserved"])
        self.assertGreater(continued["summary_calls"], 0)

    def test_live_process_exclusion_and_killed_run_require_explicit_choice(self):
        process = subprocess.Popen(self.command("interrupt"), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, encoding="utf-8", env=self.environment)
        ready = Queue()
        reader = Thread(target=lambda: ready.put(process.stdout.readline()), daemon=True)
        reader.start()
        try:
            try:
                line = ready.get(timeout=45)
            except Empty:
                process.kill()
                output, error = process.communicate(timeout=15)
                self.fail("worker did not reach the confirmed resource boundary: " + output + error)
            if not line:
                output, error = process.communicate(timeout=15)
                self.fail("worker exited before handshake: " + output + error)
            payload = json.loads(line)
            self.assertEqual(payload["status"], "waiting_for_second_response")
            self.assertTrue(payload["artifact_refs"])
            self.assertEqual(payload["known_tokens"], 12)
            rejected = self.worker("probe", payload)
            self.assertEqual(rejected, {"status": "busy", "error_code": "conversation_busy"})
            process.kill()
            output, error = process.communicate(timeout=15)
            self.assertNotEqual(process.returncode, 0)
            self.assertFalse(output, output)
            self.assertFalse(error, error)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=15)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
        recovered = self.worker("recover-interrupted", payload)
        self.assertEqual(recovered["status"], "recovered_by_choice")
        self.assertEqual(recovered["snapshots"], 2)
        self.assertTrue(recovered["no_reexecution"])
        self.assertTrue(recovered["interrupted_resources_preserved"])
        self.assertEqual(recovered["known_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
