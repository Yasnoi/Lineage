"""Actual OS termination and same-Run recovery across interpreter processes."""

import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest


class ExecutionProcessTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="execution-process-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = {**os.environ, "PYTHONUTF8": "1", "LANGSMITH_TRACING": "false",
                            "LANGCHAIN_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}

    def command(self, mode, scenario):
        return [sys.executable, "-m", "benchmarks.checks.support.execution_worker",
                mode, str(self.root), scenario]

    def worker(self, mode, scenario, payload):
        result = subprocess.run(self.command(mode, scenario), input=json.dumps(payload) + "\n",
            capture_output=True, text=True, encoding="utf-8", env=self.environment, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout[-6000:])
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 1, result.stdout[:2000])
        return json.loads(lines[0])

    def kill_at_boundary(self, scenario, *, check_lock=False, mode="start", payload=None):
        child = subprocess.Popen(self.command(mode, scenario), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", env=self.environment)
        if payload is not None:
            child.stdin.write(json.dumps(payload) + "\n")
            child.stdin.flush()
        ready = Queue()
        Thread(target=lambda: ready.put(child.stdout.readline()), daemon=True).start()
        try:
            try:
                line = ready.get(timeout=90)
            except Empty:
                child.kill()
                output, error = child.communicate(timeout=15)
                self.fail("worker did not reach boundary: " + output + error)
            if not line:
                output, error = child.communicate(timeout=15)
                self.fail("worker exited before boundary: " + output + error)
            payload = json.loads(line)
            self.assertEqual(payload["status"], "at_boundary")
            self.assertEqual(payload["scenario"], scenario)
            if check_lock:
                self.assertEqual(self.worker("probe", scenario, payload),
                                 {"status": "busy", "code": "conversation_busy"})
            child.kill()
            output, error = child.communicate(timeout=15)
            self.assertNotEqual(child.returncode, 0)
            self.assertEqual(output, "")
            self.assertEqual(error, "")
            return payload
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=15)
            for stream in (child.stdin, child.stdout, child.stderr):
                stream.close()

    def assert_completed(self, payload, result):
        self.assertEqual(result["status"], "completed", result.get("reason"))
        self.assertEqual(result["snapshot_status"], "published")
        self.assertEqual(result["conversation_id"], payload["conversation_id"])
        self.assertEqual(result["run_id"], payload["run_id"])
        self.assertEqual(result["published_runs"].count(payload["run_id"]), 1)
        self.assertTrue(result["no_open_effects"])
        self.assertTrue(result["duplicate_resume_no_dispatch"])
        after = {item["id"]: item for item in result["operations"]}
        for operation in payload["operations"]:
            self.assertIn(operation["id"], after)
            if operation["completed"]:
                self.assertEqual(after[operation["id"]], operation,
                                 "a trusted complete result was repeated or changed")

    def test_complete_model_response_reused_without_new_request(self):
        payload = self.kill_at_boundary("response_saved", check_lock=True)
        result = self.worker("resume", "response_saved", payload)
        self.assert_completed(payload, result)
        self.assertEqual(result["events"], payload["events"])
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["usage_totals"]["total_tokens"], 12)
        self.assertEqual(len(result["published_runs"]), 2)

    def test_first_run_can_resume_without_any_sealed_snapshot(self):
        payload = self.kill_at_boundary("first_response")
        self.assertIsNone(payload["source_snapshot_id"])
        result = self.worker("resume", "first_response", payload)
        self.assert_completed(payload, result)
        self.assertEqual(result["events"], payload["events"])
        self.assertEqual(len(result["published_runs"]), 1)

    def test_finalization_candidate_and_confirmed_point_publish_without_execution(self):
        for scenario in ("final_operation", "final_point"):
            with self.subTest(scenario=scenario), TemporaryDirectory(prefix="final-process-") as temporary:
                self.root = Path(temporary)
                payload = self.kill_at_boundary(scenario)
                result = self.worker("resume", scenario, payload)
                self.assert_completed(payload, result)
                self.assertEqual(result["events"], payload["events"])
                self.assertEqual(result["model_calls"], 1)
                self.assertEqual(result["summary_calls"], 0)
                self.assertEqual(result["tool_calls"], 0)

    def test_publication_before_display_is_complete_and_old_execution_cannot_resume(self):
        payload = self.kill_at_boundary("final_published")
        result = self.worker("published", "final_published", payload)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["events"], payload["events"])
        self.assertTrue(result["no_open_effects"])
        self.assertEqual(result["record_run_id"], payload["run_id"])
        self.assertEqual(result["published_runs"].count(payload["run_id"]), 1)
        self.assertTrue(result["old_execution_rejected"])

    def test_unknown_model_attempt_retries_fixed_request_and_preserves_unknown_usage(self):
        payload = self.kill_at_boundary("unknown_model")
        result = self.worker("resume", "unknown_model", payload)
        self.assert_completed(payload, result)
        requests = [event for event in result["events"] if event["kind"] == "model"]
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["request_digest"], requests[1]["request_digest"])
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["usage_totals"]["total_tokens"], 12)
        self.assertEqual(result["usage_totals"]["missing_usage_calls"], 1)
        operation = next(item for item in result["operations"] if item["kind"] == "model")
        self.assertEqual(len(operation["attempt_ids"]), 2)
        self.assertEqual(operation["completed"], 1)

    def test_saved_tool_raw_is_reused_before_observation_and_group_adoption(self):
        payload = self.kill_at_boundary("tool_raw_saved")
        result = self.worker("resume", "tool_raw_saved", payload)
        self.assert_completed(payload, result)
        self.assertEqual([event["key"] for event in result["events"] if event["kind"] == "tool"], ["a", "b", "c"])
        self.assertEqual(result["tool_calls"], 3)
        self.assertEqual(result["tool_outcomes"]["succeeded"], 3)
        saved = next(item for item in payload["operations"] if item["kind"] == "tool" and item["completed"])
        self.assertEqual(len(saved["artifacts"]), 1)
        self.assertIn(saved["artifacts"][0], json.dumps(result["final_input"]))

    def test_second_unknown_attempt_does_not_gain_another_retry_after_restarting_again(self):
        first = self.kill_at_boundary("unknown_model")
        second = self.kill_at_boundary("unknown_model", mode="retry", payload=first)
        self.assertEqual(first["run_id"], second["run_id"])
        result = self.worker("resume", "unknown_model", second)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["reason"], "recovery_retry_exhausted")
        self.assertEqual(len([event for event in result["events"] if event["kind"] == "model"]), 2)
        operation = next(item for item in result["operations"] if item["kind"] == "model")
        self.assertEqual(len(operation["attempt_ids"]), 2)
        self.assertEqual(operation["completed"], 0)
        self.assertEqual(result["inspected_quota"]["usage_totals"]["missing_usage_calls"], 2)

    def test_file_budget_unknown_reservation_and_saved_result_survive_process_boundary(self):
        payload = self.kill_at_boundary("read_budget")
        (self.root / "b.txt").write_bytes(b"new")
        result = self.worker("resume", "read_budget", payload)
        self.assert_completed(payload, result)
        reads = [event for event in result["events"] if event["kind"] == "read"]
        self.assertEqual([item["path"] for item in reads], ["a.txt", "b.txt", "b.txt", "c.txt"])
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(result["inspected_quota"]["reserved"]["read_bytes"], 8)
        self.assertEqual(result["inspected_quota"]["used"]["read_bytes"], 4)
        self.assertEqual(result["final_budget"]["reserved"]["read_bytes"], 8)
        self.assertEqual(result["final_budget"]["used"]["read_bytes"], 11)

    def test_partial_batch_reuses_old_result_and_unknown_retry_observes_new_environment(self):
        payload = self.kill_at_boundary("tool_partial")
        (self.root / "external-observation.txt").write_text("new-observation", encoding="utf-8")
        result = self.worker("resume", "tool_partial", payload)
        self.assert_completed(payload, result)
        self.assertEqual([event["key"] for event in result["events"] if event["kind"] == "tool"], ["a", "b", "b", "c"])
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(result["tool_outcomes"]["succeeded"], 3)
        outputs = [item for item in result["final_input"] if item.get("type") == "function_call_output"]
        self.assertEqual([item["call_id"] for item in outputs], ["a", "b", "c"])
        self.assertIn("old-observation:a", outputs[0]["output"])
        self.assertIn("new-observation:b", outputs[1]["output"])
        self.assertIn("new-observation:c", outputs[2]["output"])

    def test_unknown_external_effect_without_declaration_pauses(self):
        payload = self.kill_at_boundary("unsafe_tool")
        result = self.worker("resume", "unsafe_tool", payload)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["reason"], "tool_recovery_unsupported")
        self.assertEqual(result["events"], payload["events"])
        self.assertNotIn("snapshot_id", result)
        self.assertTrue(result["duplicate_resume_no_dispatch"])

    def test_history_complete_and_truncated_unconfirmed_tail_remains_isolated(self):
        payload = self.kill_at_boundary("history_tail")
        directory = self.root / ".lineage" / payload["conversation_id"]
        self.assertTrue(payload["target_history"])
        target = directory / payload["target_history"][-1]
        tail = b'{"unconfirmed":"never-authorize-this-tail"}\n{"truncated":'
        with target.open("ab") as stream:
            stream.write(tail)
        original = target.read_bytes()
        result = self.worker("resume", "history_tail", payload)
        self.assert_completed(payload, result)
        self.assertEqual(target.read_bytes(), original)
        self.assertTrue(result["original_history_unchanged"])
        self.assertGreater(len(result["history_paths"]), len(payload["history"]))
        self.assertNotIn("never-authorize-this-tail", json.dumps(result["final_input"]))
        self.assertNotIn("never-authorize-this-tail", json.dumps(result["final_record"]))

    def test_compact_h_and_full_candidates_resume_without_repeating_finished_summary(self):
        for scenario, expected_candidates in (("compact_h", 1), ("compact_candidates", 2)):
            with self.subTest(scenario=scenario), TemporaryDirectory(prefix="compact-process-") as temporary:
                self.root = Path(temporary)
                payload = self.kill_at_boundary(scenario)
                self.assertEqual(len(payload["candidate_ids"]), expected_candidates)
                self.assertEqual(payload["candidates_active"], [])
                self.assertEqual(payload["candidates_accessible"], [])
                result = self.worker("resume", scenario, payload)
                self.assert_completed(payload, result)
                summaries = [event for event in result["events"] if event["kind"] == "summary"]
                self.assertEqual(result["summary_calls"], len(summaries))
                self.assertGreater(result["summary_calls"], 0)
                for operation in payload["operations"]:
                    if operation["kind"] == "summary" and operation["completed"]:
                        self.assertEqual(sum(event["request_digest"] == operation["request_digest"]
                                             for event in summaries), 1)


if __name__ == "__main__":
    unittest.main()
