"""Exercise real durable group events and legacy-check observation boundaries."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmarks.core.events import GroupJournal, planned_groups, read_group_events
from benchmarks.core.contracts import exit_code, validate_result
from benchmarks.reporting.render import render_report
from benchmarks.tests.contract_examples import example_result
from benchmarks.checks.support import events as support


def definition():
    return {
        "unit_id": "fixture-script",
        "script": "fixture.py",
        "module": "control-fixture",
        "groups": [
            {"group_id": f"fixture/{name}", "description": f"Fixture {name}",
             "source": {"path": "fixture.py", "symbol": name},
             "primary_area": "execution", "tags": ["execution"]}
            for name in ("first", "second", "third")
        ],
    }


class PartialWrite:
    """Retain the genuine stream prefix, then fail one actual artifact write."""

    def __init__(self, stream):
        self.stream = stream

    def write(self, value):
        self.stream.write(value[: max(1, len(value) // 2)])
        self.stream.flush()
        raise OSError("injected partial group-event write")

    def close(self):
        self.stream.close()


class GroupObservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "groups.jsonl"
        self.plan = definition()
        self.offsets = patch.object(support, "_offsets", return_value={"stdout": 0, "stderr": 0})
        self.offset_mock = self.offsets.start()
        self.addCleanup(self.offsets.stop)

    def journal(self):
        journal = GroupJournal(self.path, "benchmark-run", self.plan)
        self.addCleanup(journal.close)
        return journal

    def read(self):
        return read_group_events(self.path, "benchmark-run", self.plan)

    def test_standalone_hooks_preserve_original_exception_and_noop(self):
        original = AssertionError("original assertion")
        with self.assertRaises(AssertionError) as raised:
            with support.check_group("arbitrary-standalone-group"):
                support.observe_state({"run_id": "not-recorded", "input": "PRIVATE"})
                support.evidence(self.path)
                support.coverage_gap("fixture", "not recorded")
                raise original
        self.assertIs(raised.exception, original)
        self.assertFalse(self.path.exists())
        self.offset_mock.assert_not_called()

    def test_declared_plan_preserves_unstarted_groups_without_fabricated_passes(self):
        groups = planned_groups(self.plan)
        self.assertEqual([group["planned_index"] for group in groups], [1, 2, 3])
        self.assertTrue(all(group["execution_status"] == "not_run" for group in groups))
        self.assertTrue(all(group["verdict"] == "pending_review" for group in groups))
        self.assertTrue(all(group["started_at"] is None and group["ended_at"] is None for group in groups))
        groups[0]["tags"].append("sessions")
        self.assertEqual(self.plan["groups"][0]["tags"], ["execution"])

    def test_invalid_end_metadata_preserves_only_completed_prefix(self):
        journal = self.journal()
        with support.activate(journal):
            with support.check_group("fixture/first"):
                pass
            with support.check_group("fixture/second"):
                pass
        journal.close()
        original = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        for field, value in (("output_offsets", None), ("output_offsets", {"stdout": -1, "stderr": 0}),
                             ("elapsed_seconds", 0)):
            with self.subTest(field=field, value=value):
                records = deepcopy(original)
                if field == "elapsed_seconds":
                    records[-1][field] = value
                else:
                    records[-1]["payload"][field] = value
                self.path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
                groups, errors, count = self.read()
                self.assertTrue(errors)
                self.assertEqual(count, len(records) - 1)
                self.assertEqual(groups[0]["verdict"], "passed")
                self.assertEqual(groups[1]["execution_status"], "failed")
                self.assertIsNone(groups[1]["ended_at"])
                self.assertIsNone(groups[1]["duration_seconds"])
                self.assertEqual(groups[2]["execution_status"], "not_run")

    def test_later_log_identity_is_preserved_when_combining_references(self):
        journal = self.journal()
        log = self.path.with_name("observed.jsonl")
        with support.activate(journal):
            with support.check_group("fixture/first"):
                support.evidence(log, kind="run_log", run_id="r1")
                support.evidence(log, kind="run_log", run_id="r1", session_id="s1", last_confirmed_checkpoint_id="c1")
        journal.close()
        groups, _, _ = self.read()
        self.assertEqual(len(groups[0]["evidence"]), 1)
        self.assertEqual(groups[0]["evidence"][0]["identifiers"],
                         {"run_id": "r1", "session_id": "s1", "last_confirmed_checkpoint_id": "c1"})

    def test_actual_group_events_preserve_order_and_original_log_reference(self):
        log = Path(self.temporary.name) / "actual-run.jsonl"
        log.write_text('{"event":"run.started"}\n', encoding="utf-8")
        journal = self.journal()
        with support.activate(journal):
            with support.check_group("fixture/first"):
                support.observe_state({"run_id": "agent-1", "log_path": str(log), "stop_reason": "error"})
                support.observe_state({"run_id": "agent-1", "log_path": str(log), "stop_reason": "error"})
            with support.check_group("fixture/second"):
                support.observe_harness("trace_serializer", run_id="harness-1")
            with support.check_group("fixture/third"):
                pass
        journal.close()
        groups, errors, count = self.read()
        self.assertFalse(errors)
        self.assertEqual(count, 10)
        self.assertEqual([group["group_id"] for group in groups], [g["group_id"] for g in self.plan["groups"]])
        self.assertTrue(all(group["verdict"] == "passed" for group in groups))
        self.assertTrue(all(group["duration_seconds"] >= 0 for group in groups))
        self.assertEqual(len(groups[0]["evidence"]), 1)
        self.assertEqual(groups[0]["evidence"][0]["path"], str(log))
        self.assertEqual(groups[0]["evidence"][0]["availability"], "available")
        self.assertEqual(groups[0]["observations"][0]["identifiers"]["stop_reason"], "error")
        self.assertEqual(groups[1]["observations"][0]["kind"], "trace_serializer")

    def test_out_of_order_group_aborts_without_starting_a_different_group(self):
        journal = self.journal()
        with support.activate(journal), self.assertRaises(support.RecordingFailure):
            with support.check_group("fixture/second"):
                self.fail("unexpected group was allowed to run")
        self.assertTrue(journal.failed)
        journal.close()
        groups, errors, count = self.read()
        self.assertFalse(errors)
        self.assertEqual(count, 1)
        self.assertTrue(all(group["execution_status"] == "not_run" for group in groups))

    def test_real_assertion_bypasses_legacy_except_exception_and_stops_remaining_groups(self):
        journal = self.journal()
        swallowed = []
        original = AssertionError("actual fixture assertion")
        with support.activate(journal), self.assertRaises(support.CheckStopped) as raised:
            with support.check_group("fixture/first"):
                pass
            try:
                with support.check_group("fixture/second"):
                    raise original
            except Exception:
                swallowed.append(True)
            with support.check_group("fixture/third"):
                self.fail("a later group ran after uncertain cleanup")
        self.assertIs(raised.exception.cause, original)
        self.assertFalse(swallowed)
        journal.close()
        groups, errors, _ = self.read()
        self.assertFalse(errors)
        self.assertEqual([group["execution_status"] for group in groups], ["completed", "completed", "not_run"])
        self.assertEqual([group["verdict"] for group in groups], ["passed", "failed", "pending_review"])
        self.assertEqual(groups[1]["issues"][0]["exception_type"], "AssertionError")
        self.assertIsNone(groups[2]["started_at"])

    def test_unexpected_check_error_is_execution_failure_without_failed_assessment(self):
        journal = self.journal()
        original = ValueError("unexpected check error")
        with support.activate(journal), self.assertRaises(support.CheckStopped) as raised:
            with support.check_group("fixture/first"):
                raise original
        self.assertIs(raised.exception.cause, original)
        journal.close()
        groups, errors, _ = self.read()
        self.assertFalse(errors)
        self.assertEqual((groups[0]["execution_status"], groups[0]["verdict"]), ("failed", "pending_review"))
        self.assertEqual(groups[0]["issues"][0]["exception_type"], "ValueError")

    def test_applicable_and_nonapplicable_coverage_gaps_keep_different_verdicts(self):
        journal = self.journal()
        with support.activate(journal):
            with support.check_group("fixture/first"):
                support.coverage_gap("platform-specific", "not applicable on this platform", applicable=False)
            with support.check_group("fixture/second"):
                support.coverage_gap("symlink", "creation permission unavailable", applicable=True)
            with support.check_group("fixture/third"):
                pass
        self.assertTrue(journal.coverage_gaps)
        journal.close()
        groups, errors, _ = self.read()
        self.assertFalse(errors)
        self.assertEqual([group["verdict"] for group in groups], ["passed", "pending_review", "passed"])
        self.assertFalse(groups[0]["coverage_gaps"][0]["applicable"])
        self.assertTrue(groups[1]["coverage_gaps"][0]["applicable"])
        self.assertEqual(groups[1]["execution_status"], "completed")

    def test_unfinished_group_and_unstarted_group_are_distinct(self):
        journal = self.journal()
        with support.activate(journal):
            with support.check_group("fixture/first"):
                pass
        journal.emit("group_started", "fixture/second", {"output_offsets": {"stdout": 10, "stderr": 0}})
        journal.close()
        groups, errors, _ = self.read()
        self.assertFalse(errors)
        self.assertEqual([group["execution_status"] for group in groups], ["completed", "failed", "not_run"])
        self.assertEqual(groups[1]["issues"][0]["code"], "unfinished_group")
        self.assertIsNotNone(groups[1]["started_at"])
        self.assertIsNone(groups[1]["ended_at"])
        self.assertIsNone(groups[2]["started_at"])

    def test_partial_record_write_preserves_prefix_and_cannot_continue(self):
        journal = self.journal()
        with support.activate(journal), self.assertRaises(support.RecordingFailure):
            with support.check_group("fixture/first"):
                pass
            with support.check_group("fixture/second"):
                journal.stream = PartialWrite(journal.stream)
                support.observe_state({"run_id": "actual-run-before-write-fault"})
        self.assertTrue(journal.failed)
        with self.assertRaises(OSError):
            journal.emit("group_ended", "fixture/second", {"outcome": "passed", "error": None,
                         "output_offsets": {"stdout": 0, "stderr": 0}})
        journal.close()
        groups, errors, count = self.read()
        self.assertEqual(count, 4)
        self.assertEqual(errors[0]["stage"], "group_recording")
        self.assertEqual([group["verdict"] for group in groups], ["passed", "pending_review", "pending_review"])
        self.assertEqual([group["execution_status"] for group in groups], ["completed", "failed", "not_run"])

    def test_corrupt_suffix_cannot_be_repaired_by_later_success_events(self):
        journal = self.journal()
        with support.activate(journal):
            for group in self.plan["groups"]:
                with support.check_group(group["group_id"]):
                    pass
        journal.close()
        lines = self.path.read_bytes().splitlines(keepends=True)
        corrupt = json.loads(lines[4])
        corrupt["run_id"] = "old-benchmark-run"
        lines[4] = (json.dumps(corrupt) + "\n").encode()
        self.path.write_bytes(b"".join(lines))
        groups, errors, count = self.read()
        self.assertEqual(count, 4)
        self.assertTrue(errors)
        self.assertEqual([group["execution_status"] for group in groups], ["completed", "failed", "not_run"])
        self.assertEqual(groups[0]["verdict"], "passed")
        self.assertEqual(groups[2]["verdict"], "pending_review")

    def test_plan_mismatch_and_empty_stream_never_infer_success(self):
        journal = self.journal()
        journal.close()
        altered = deepcopy(self.plan)
        altered["groups"][0]["description"] = "different definition"
        groups, errors, count = read_group_events(self.path, "benchmark-run", altered)
        self.assertEqual(count, 0)
        self.assertTrue(errors)
        self.assertTrue(all(group["execution_status"] == "not_run" for group in groups))
        self.path.write_bytes(b"")
        groups, errors, count = self.read()
        self.assertEqual(count, 0)
        self.assertEqual(errors[0]["code"], "empty_event_stream")
        self.assertTrue(all(group["verdict"] == "pending_review" for group in groups))

    def test_observations_filter_payloads_and_preserve_safe_actual_identities(self):
        marker = "SENSITIVE_REQUEST_OR_ENVIRONMENT_BODY"
        journal = self.journal()
        with support.activate(journal):
            with support.check_group("fixture/first"):
                support.observe_state({
                    "run_id": "agent-observed", "stop_reason": "completed", "model_calls": 1,
                    "input": [{"content": marker}], "result": {"raw_response": marker},
                    "api_key": marker, "environment": {"API_KEY": marker},
                    "config": {"secret": marker},
                    "active_context": {"version_id": "version-observed", "document": marker},
                }, session_id="session-observed", api_key=marker, raw_request=marker)
                support.evidence(self.path.with_name("actual.jsonl"), kind="run_log",
                                 run_id="agent-observed", session_id="session-observed", raw_response=marker)
                support.observe_harness("serializer_fixture", run_id="harness-observed", body=marker)
        journal.close()
        self.assertNotIn(marker, self.path.read_text(encoding="utf-8"))
        groups, errors, _ = self.read()
        self.assertFalse(errors)
        agent, harness = groups[0]["observations"]
        self.assertEqual(agent["kind"], "agent")
        self.assertEqual(agent["identifiers"]["session_id"], "session-observed")
        self.assertEqual(agent["identifiers"]["active_version_id"], "version-observed")
        self.assertEqual(harness["kind"], "serializer_fixture")
        self.assertEqual(groups[0]["evidence"][0]["availability"], "missing")

    def test_old_schema_one_contract_remains_readable_without_group_details(self):
        old = example_result()
        self.assertEqual(old["schema_version"], "1")
        self.assertTrue(all("groups" not in unit for unit in old["units"]))
        restored = json.loads(json.dumps(old))
        validate_result(restored)
        self.assertEqual(exit_code(restored), 3)
        report = render_report(restored)
        self.assertIn(old["run_id"], report)
        self.assertIn("pending_review", report)
        self.assertEqual(old, restored)


if __name__ == "__main__":
    unittest.main()
