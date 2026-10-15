"""Trace equivalence preserves behavior and journal geometry across new IDs."""

from copy import deepcopy
import hashlib
import unittest
from uuid import uuid4

from benchmarks.checks.support.tracing import stable


def state():
    previous, current, conversation = (str(uuid4()) for _ in range(3))
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    return {
        "run_id": current, "conversation_id": conversation, "log_path": "generated.log",
        "execution": {"point_id": str(uuid4()), "parent_id": str(uuid4()), "phase": "finalizing"},
        "log_segments": [{"execution_id": str(uuid4()), "path": current + ".jsonl", "complete": True}],
        "input": [{"role": "user", "content": "Exact original 17; do not write files"}],
        "result": {"text": "exact answer"}, "tool_outcomes": {"succeeded": 1},
        "active_context": {
            "version_id": str(uuid4()), "parent_version_id": str(uuid4()),
            "turns": [{"run_id": previous, "item_count": 2, "group_count": 1, "record_end": 5}],
            "current_run_id": current, "groups": [{"group_id": current + "/group-1", "closed": True}],
            "item_refs": [f"history://record/{previous}/0", f"history://record/{current}/0"],
        },
        "record_batch": [{"source_ref": f"history://record/{current}/2", "run_id": current,
                          "sequence": 2, "offset": 200, "length": 100, "sha256": digest(current + "/record")}],
        "record_files": {
            previous: {"format_version": 1, "record_count": 5, "byte_end": 500,
                       "prefix_sha256": digest(previous + "/file")},
            current: {"format_version": 1, "record_count": 3, "byte_end": 300,
                      "prefix_sha256": digest(current + "/file"), "segments": [
                          {"path": "history/" + current + ".jsonl", "start_sequence": 0,
                           "record_count": 3, "byte_end": 300, "prefix_sha256": digest(current + "/file")}]},
        },
        "record_metrics": {"count": 3, "file_bytes": 300, "logical_bytes": 500,
                           "checkpoint_serialized_bytes": 987},
    }


class TraceEquivalenceTests(unittest.TestCase):
    def test_generated_ids_and_digests_normalize_without_mutating_originals(self):
        first, second = state(), state()
        original = deepcopy(first)
        second["record_metrics"]["checkpoint_serialized_bytes"] = 999
        self.assertEqual(stable(first), stable(second))
        self.assertEqual(first, original)
        self.assertNotIn("checkpoint_id", stable(first)["active_context"]["turns"][0])

    def test_business_facts_record_geometry_and_counts_stay_exact(self):
        first = state()
        mutations = [
            lambda value: value["input"][0].update(content="changed input"),
            lambda value: value["result"].update(text="changed answer"),
            lambda value: value["tool_outcomes"].update(succeeded=2),
            lambda value: value["record_batch"][0].update(length=99),
            lambda value: value["record_files"][value["run_id"]].update(record_count=4),
            lambda value: value["record_files"][value["run_id"]].update(byte_end=301),
            lambda value: value["record_metrics"].update(file_bytes=301),
            lambda value: value["execution"].update(phase="request_ready"),
            lambda value: value["log_segments"][0].update(complete=False),
            lambda value: value["record_files"][value["run_id"]]["segments"][0].update(byte_end=301),
        ]
        for mutate in mutations:
            changed = deepcopy(first)
            mutate(changed)
            self.assertNotEqual(stable(first), stable(changed))

    def test_invalid_receipt_identity_hash_or_bounds_is_rejected(self):
        for changes in ({"sha256": "bad"}, {"offset": 250}, {"length": 0}, {"sequence": 3},
                        {"source_ref": "history://record/foreign/2"}):
            value = state()
            value["record_batch"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                stable(value)


if __name__ == "__main__":
    unittest.main()
