"""Research subprocess boundary faults, using explicitly synthetic receipts."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from benchmarks.common import atomic_write_json, sha256_file, utc_now
from benchmarks.research import batch
from benchmarks.research.definitions import PROFILE, get_case
from benchmarks.research.worker import pending_turn
from benchmarks.results import known


class ResearchProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="research-process-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.case = get_case("wechat-capacity-ocr")
        self.unit = batch.make_unit(self.case, "pilot", 1, "sample-id", PROFILE)
        self.spec = {"run_id": "sample-id", "case_id": self.case["case_id"],
                     "definition_version": self.case["version"], "prompts": self.case["turns"],
                     "ledger_path": str(self.directory / "requests.jsonl")}
        (self.directory / "requests.jsonl").touch()
        self.connection = {"API_KEY": "synthetic-process-marker", "BASE_URL": PROFILE["base_url"], "MODEL_NAME": PROFILE["model_name"]}

    def receipt(self, *, final=True, status="completed", global_stop=False):
        turns = [pending_turn(i) for i in range(1, 3)]
        for turn in turns[:2 if final else 1]:
            turn.update(execution_status="completed", started_at=utc_now(), ended_at=utc_now(),
                        run_id=known(f"agent-run-{turn['turn']}"), session_id=known("real-session-fixture"),
                        checkpoint_id=known(f"checkpoint-{turn['turn']}"), checkpoint_complete=True,
                        agent_outcome=known("completed"))
        return {"schema_version": "1", "run_id": self.spec["run_id"], "case_id": self.spec["case_id"],
                "definition_version": self.spec["definition_version"],
                "spec_sha256": sha256_file(self.directory / "spec.json"), "final": final,
                "started_at": utc_now(), "ended_at": utc_now() if final else None,
                "execution_status": status, "verdict": "pending_review", "turns": turns,
                "agent_outcome": known("completed"), "metrics": {"sdk_requests": known(2)},
                "coverage": [], "issues": [], "evidence": [], "global_stop": global_stop,
                "global_stop_reason": "record_write_failed" if global_stop else None}

    def launch(self, process, *, final=True, status="completed", changes=None, global_stop=False):
        def spawn(command, **kwargs):
            self.assertEqual(command[1:3], ["-m", "benchmarks.research.worker"])
            self.assertEqual(kwargs["env"]["LANGSMITH_TRACING"], "false")
            self.assertNotIn("PYTHONOPTIMIZE", kwargs["env"])
            kwargs["stdout"].write(b"synthetic success text; never an assessment\n")
            kwargs["stderr"].write(b"synthetic diagnostic output\n")
            kwargs["stdout"].flush()
            kwargs["stderr"].flush()
            receipt = self.receipt(final=final, status=status, global_stop=global_stop)
            observations = self.directory / "observations.jsonl"
            observations.write_text('{"fixture": "synthetic process evidence"}\n', encoding="utf-8")
            receipt["observations_sha256"] = sha256_file(observations)
            receipt.update(changes or {})
            atomic_write_json(self.directory / "worker-result.json", receipt)
            return process
        return spawn

    def execute(self):
        return batch.execute_sample(self.unit, self.spec, self.directory, self.connection, timeout=1)

    def test_normal_pending_assessment_and_process_evidence(self):
        process = Mock(returncode=0)
        process.wait.return_value = 0
        with patch.object(batch.subprocess, "Popen", side_effect=self.launch(process)), patch.object(batch, "_stop") as cleanup:
            self.assertFalse(self.execute())
        cleanup.assert_called_once_with(process)
        self.assertEqual(self.unit["execution_status"], "completed")
        self.assertEqual(self.unit["verdict"], "pending_review")
        self.assertEqual(self.unit["turns"][1]["run_id"]["value"], "agent-run-2")
        self.assertTrue(all(Path(e["path"]).is_file() for e in self.unit["evidence"]))

    def test_timeout_retains_confirmed_prefix_and_unstarted_suffix(self):
        process = Mock(returncode=-1)
        process.wait.side_effect = subprocess.TimeoutExpired("synthetic worker", 1)
        with patch.object(batch.subprocess, "Popen", side_effect=self.launch(process, final=False, status="failed")), \
                patch.object(batch, "_stop") as cleanup:
            self.assertFalse(self.execute())
        cleanup.assert_called_once_with(process)
        self.assertEqual(self.unit["execution_status"], "failed")
        self.assertEqual(self.unit["turns"][0]["execution_status"], "completed")
        self.assertEqual(self.unit["turns"][1]["execution_status"], "not_run")
        self.assertEqual(self.unit["issues"][-1]["code"], "sample_timeout")
        self.assertIn("synthetic success", (self.directory / "stdout.txt").read_text())

    def test_timeout_race_keeps_finished_facts_without_restoring_success(self):
        process = Mock(returncode=0)
        process.wait.side_effect = subprocess.TimeoutExpired("synthetic worker", 1)
        with patch.object(batch.subprocess, "Popen", side_effect=self.launch(process)), patch.object(batch, "_stop"):
            self.assertFalse(self.execute())
        self.assertEqual(self.unit["execution_status"], "failed")
        self.assertTrue(all(t["execution_status"] == "completed" for t in self.unit["turns"]))

    def test_interruption_stops_scheduling_and_keeps_prefix(self):
        process = Mock(returncode=-1)
        process.wait.side_effect = KeyboardInterrupt()
        with patch.object(batch.subprocess, "Popen", side_effect=self.launch(process, final=False, status="failed")), \
                patch.object(batch, "_stop") as cleanup:
            self.assertTrue(self.execute())
        cleanup.assert_called_once_with(process)
        self.assertEqual(self.unit["turns"][0]["execution_status"], "completed")
        self.assertEqual(self.unit["issues"][-1]["code"], "interrupted")

    def test_shutdown_failure_stops_scheduling_even_with_final_success(self):
        process = Mock(returncode=0)
        process.wait.return_value = 0
        with patch.object(batch.subprocess, "Popen", side_effect=self.launch(process)), \
                patch.object(batch, "_stop", side_effect=RuntimeError("synthetic cleanup failure")):
            self.assertTrue(self.execute())
        self.assertEqual(self.unit["execution_status"], "failed")
        self.assertEqual(self.unit["issues"][-1]["code"], "process_tree_unconfirmed")
        self.assertEqual(self.unit["turns"][0]["execution_status"], "completed")

    def test_start_failure_is_not_model_or_quality_failure(self):
        with patch.object(batch.subprocess, "Popen", side_effect=OSError("synthetic process launch error")):
            self.assertTrue(self.execute())
        self.assertEqual(self.unit["execution_status"], "failed")
        self.assertEqual(self.unit["verdict"], "pending_review")
        self.assertTrue(all(turn["execution_status"] == "not_run" for turn in self.unit["turns"]))
        self.assertEqual(self.unit["issues"][-1]["code"], "worker_unavailable")

    def test_output_recording_channel_failure_stops_before_launch(self):
        (self.directory / "stdout.txt").write_text("previous artifact", encoding="utf-8")
        with patch.object(batch.subprocess, "Popen") as launch:
            self.assertTrue(self.execute())
        launch.assert_not_called()
        self.assertEqual((self.directory / "stdout.txt").read_text(), "previous artifact")
        self.assertEqual(self.unit["execution_status"], "failed")

    def test_invalid_identity_ignores_success_text_and_untrusted_turns(self):
        process = Mock(returncode=0)
        process.wait.return_value = 0
        with patch.object(batch.subprocess, "Popen", side_effect=self.launch(process, changes={"run_id": "previous-run"})), \
                patch.object(batch, "_stop"):
            self.assertTrue(self.execute())
        self.assertEqual(self.unit["execution_status"], "failed")
        self.assertEqual(self.unit["issues"][-1]["code"], "invalid_research_receipt")
        self.assertTrue(all(turn["run_id"]["availability"] != "known" for turn in self.unit["turns"]))

    def test_process_and_final_receipt_disagreement_cannot_pass(self):
        process = Mock(returncode=2)
        process.wait.return_value = 2
        with patch.object(batch.subprocess, "Popen", side_effect=self.launch(process)), patch.object(batch, "_stop"):
            self.assertTrue(self.execute())
        self.assertEqual(self.unit["execution_status"], "failed")
        self.assertEqual(self.unit["verdict"], "pending_review")

    def test_missing_final_retains_valid_partial_prefix(self):
        process = Mock(returncode=2)
        process.wait.return_value = 2
        with patch.object(batch.subprocess, "Popen", side_effect=self.launch(process, final=False, status="failed")), \
                patch.object(batch, "_stop"):
            self.assertTrue(self.execute())
        self.assertEqual(self.unit["execution_status"], "failed")
        self.assertEqual(self.unit["turns"][0]["execution_status"], "completed")
        self.assertEqual(self.unit["turns"][1]["execution_status"], "not_run")
        self.assertEqual(self.unit["issues"][-1]["code"], "invalid_research_receipt")

    def test_non_object_receipt_is_explicit_adaptation_failure(self):
        process = Mock(returncode=0)
        process.wait.return_value = 0
        original = self.launch(process)
        def spawn(command, **kwargs):
            result = original(command, **kwargs)
            (self.directory / "worker-result.json").write_text("[]", encoding="utf-8")
            return result
        with patch.object(batch.subprocess, "Popen", side_effect=spawn), patch.object(batch, "_stop"):
            self.assertTrue(self.execute())
        self.assertEqual(self.unit["execution_status"], "failed")
        self.assertEqual(self.unit["issues"][-1]["code"], "invalid_research_receipt")

    def test_worker_record_failure_requests_global_stop(self):
        process = Mock(returncode=2)
        process.wait.return_value = 2
        with patch.object(batch.subprocess, "Popen", side_effect=self.launch(process, status="failed", global_stop=True)), \
                patch.object(batch, "_stop"):
            self.assertTrue(self.execute())
        self.assertEqual(self.unit["execution_status"], "failed")
        self.assertEqual(self.unit["turns"][0]["execution_status"], "completed")


if __name__ == "__main__":
    unittest.main()
