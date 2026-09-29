"""Remote trace verification against local identity facts, without networking."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from benchmarks.core import trace_evidence as evidence


class RemoteTraceTests(unittest.TestCase):
    def setUp(self):
        root, child = str(uuid4()), str(uuid4())
        binding = {"benchmark_batch_id": "batch", "sample_run_id": "sample", "case_id": "case", "turn": 1}
        self.expected = {"trace_id": root, "enabled": True, "upload_state": "submission_completed", "spans": [
            {"span_id": root, "trace_id": root, "parent_run_id": None, "run_type": "chain",
             "metadata": {"run_id": root, **binding}, "usage": {}},
            {"span_id": child, "trace_id": root, "parent_run_id": root, "run_type": "llm",
             "metadata": {"run_id": root, **binding, "purpose": "business", "request_id": str(uuid4())},
             "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}},
        ]}
        self.runs = [SimpleNamespace(id=s["span_id"], trace_id=root, parent_run_id=s["parent_run_id"],
                                    run_type=s["run_type"], end_time="finished", extra={"metadata": deepcopy(s["metadata"])},
                                    prompt_tokens=s["usage"].get("input_tokens"), completion_tokens=s["usage"].get("output_tokens"),
                                    total_tokens=s["usage"].get("total_tokens"), url="https://smith.langchain.com/r/example",
                                    inputs={"secret": "must-not-persist"}, outputs={"answer": "must-not-persist"})
                     for s in self.expected["spans"]]

    def test_identity_usage_and_completed_spans_required(self):
        result = evidence.compare_trace(self.expected, self.runs)
        self.assertEqual(result["status"], "verified")
        self.assertNotIn("must-not-persist", str(result))
        for field, value in (("parent_run_id", str(uuid4())), ("end_time", None), ("prompt_tokens", 8)):
            changed = deepcopy(self.runs)
            setattr(changed[1], field, value)
            self.assertEqual(evidence.compare_trace(self.expected, changed)["status"], "incomplete")
        changed = deepcopy(self.runs)
        changed[1].extra["metadata"]["request_id"] = str(uuid4())
        self.assertEqual(evidence.compare_trace(self.expected, changed)["status"], "incomplete")
        self.assertEqual(evidence.compare_trace(self.expected, self.runs[:1])["status"], "incomplete")
        changed[0].url = "https://smith.langchain.com/r/example?api_key=hidden"
        self.assertIsNone(evidence.compare_trace(self.expected, changed)["url"])

    def test_bounded_queries_and_safe_projection(self):
        calls = []
        class Client:
            def list_runs(inner, **kwargs):
                calls.append(kwargs)
                return self.runs
            def close(inner, **kwargs):
                pass
        found, _ = evidence.query_traces([self.expected], None, client_factory=lambda _: Client())
        self.assertEqual(found[0]["status"], "verified")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("inputs", calls[0]["select"])
        self.assertNotIn("outputs", calls[0]["select"])
        class Failing(Client):
            def list_runs(inner, **kwargs):
                calls.append(kwargs)
                raise RuntimeError("private-endpoint-detail")
        calls.clear()
        found, _ = evidence.query_traces([self.expected], None, client_factory=lambda _: Failing())
        self.assertEqual(len(calls), 3)
        self.assertEqual(found[0]["status"], "query_failed")
        self.assertNotIn("private-endpoint-detail", str(found))

    def test_stalled_query_does_not_block_or_mutate_snapshot(self):
        release = threading.Event()
        class Client:
            def list_runs(inner, **kwargs):
                release.wait(1)
                return self.runs
            def close(inner, **kwargs):
                pass
        started = time.monotonic()
        try:
            found, _ = evidence.query_traces([self.expected], None, timeout_seconds=0.02,
                                            client_factory=lambda _: Client())
            snapshot = deepcopy(found)
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            release.set()
        self.assertEqual(found, snapshot)
        self.assertEqual(found[0]["status"], "query_failed")
        self.assertEqual(found[0]["reason"], "query_deadline")

    def test_remote_save_failure_is_not_silenced(self):
        unit = {"turns": [{"tracing": {"traces": [self.expected]}}], "evidence": []}
        spec = {"run_id": "sample", "batch_id": "batch", "tracing": {"enabled": True,
                "endpoint": "https://trace.invalid", "project": "offline"}}
        with TemporaryDirectory() as directory, patch.object(evidence, "query_traces", return_value=([], 0)), \
                patch.object(evidence, "atomic_write_json", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                evidence.verify_sample_traces(unit, spec, Path(directory), {"LANGSMITH_API_KEY": "secret"})
        self.assertEqual(unit["evidence"], [])


if __name__ == "__main__":
    unittest.main()
