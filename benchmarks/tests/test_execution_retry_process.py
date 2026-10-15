"""Unknown summary retries retain all Run budgets through two hard exits."""

import unittest

from benchmarks.tests import test_execution_process as process_helpers


class ExecutionSummaryRetryProcessTests(unittest.TestCase):
    setUp = process_helpers.ExecutionProcessTests.setUp
    command = process_helpers.ExecutionProcessTests.command
    worker = process_helpers.ExecutionProcessTests.worker
    kill_at_boundary = process_helpers.ExecutionProcessTests.kill_at_boundary

    def test_unknown_r_summary_retries_once_without_partial_h_or_quota_reset(self):
        first = self.kill_at_boundary("unknown_summary")
        second = self.kill_at_boundary("unknown_summary", mode="retry", payload=first)
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["conversation_id"], second["conversation_id"])
        self.assertEqual(first["compact_source"], second["compact_source"])
        self.assertEqual(first["candidate_ids"], second["candidate_ids"])
        self.assertEqual(len(first["candidate_ids"]), 1)
        for payload in (first, second):
            self.assertEqual(payload["boundary"], "summary-R-in-flight")
            self.assertEqual(payload["candidates_active"], [])
            self.assertEqual(payload["candidates_accessible"], [])

        result = self.worker("resume", "unknown_summary", second)
        self.assertEqual(result["status"], "paused", result.get("reason"))
        self.assertEqual(result["reason"], "recovery_retry_exhausted")
        self.assertEqual(result["events"], second["events"], "third restore sent another request or tool")
        self.assertEqual(result["operations"], second["operations"])
        self.assertEqual(result["compact_source"], first["compact_source"])
        self.assertEqual(result["candidate_ids"], first["candidate_ids"])
        self.assertEqual(result["candidates_active"], [])
        self.assertEqual(result["candidates_accessible"], [])
        self.assertTrue(result["no_open_effects"])
        self.assertTrue(result["duplicate_resume_no_dispatch"])
        self.assertNotIn("snapshot_id", result)

        summaries = [event for event in result["events"] if event["kind"] == "summary"]
        self.assertEqual([event["lane"] for event in summaries], ["H", "R", "R"])
        self.assertEqual(summaries[1]["request_digest"], summaries[2]["request_digest"])
        h = next(operation for operation in result["operations"] if operation["lane"] == "H")
        r = next(operation for operation in result["operations"] if operation["lane"] == "R")
        self.assertEqual((len(h["attempt_ids"]), h["completed"]), (1, 1))
        self.assertEqual((len(r["attempt_ids"]), r["completed"]), (2, 0))
        self.assertEqual(len(set(r["attempt_ids"])), 2)

        quota = result["inspected_quota"]
        self.assertEqual(quota["used"]["summary_calls"], 1)
        self.assertEqual(quota["reserved"]["summary_calls"], 2)
        self.assertEqual(quota["charged"]["summary_calls"], 3)
        self.assertEqual(quota["summary_usage_totals"], {
            "input_tokens": 10, "output_tokens": 2, "total_tokens": 12, "missing_usage_calls": 2})
        self.assertEqual(quota["usage_totals"]["missing_usage_calls"], 2)
        self.assertEqual(first["quota"]["reserved"]["summary_calls"], 1)
        for key in ("model_calls", "tool_calls", "read_bytes"):
            self.assertGreater(first["quota"]["used"][key], 0)
            self.assertEqual(first["quota"]["used"][key], second["quota"]["used"][key])
            self.assertEqual(first["quota"]["used"][key], quota["used"][key])
        self.assertEqual(quota["remaining_model_calls"], quota["model_call_limit"] - quota["charged"]["model_calls"])
        self.assertEqual(result["final_budget"], second["quota"])
        known_models = len([event for event in result["events"] if event["kind"] == "model"])
        reads = [event for event in result["events"] if event["kind"] == "read"]
        self.assertEqual(quota["used"]["model_calls"], known_models)
        self.assertEqual(quota["usage_totals"]["total_tokens"], (known_models + 1) * 12)
        self.assertEqual(quota["used"]["read_bytes"], sum(event["bytes_read"] for event in reads))
        self.assertEqual(quota["used"]["tool_calls"], len(reads))


if __name__ == "__main__":
    unittest.main()
