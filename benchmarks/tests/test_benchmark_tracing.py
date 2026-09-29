"""Offline evidence for opt-in Benchmark identities and upload observations."""

import asyncio
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from uuid import uuid4

from benchmarks.checks.support.runtime import LocalClient, message, response
from benchmarks.checks.support.tracing import capture, MODEL_SECRET, TRACE_SECRET, SETTINGS
from lineage_agent.agent import runtime
from lineage_agent.agent.sessions import SessionManager
from lineage_agent.models.responses import parse_response
from lineage_agent.recording import tracing
from lineage_agent.recording.trace_data import TraceSanitizer


def binding(sample="sample-id", turn=1):
    return {"benchmark_batch_id": "test-batch", "sample_run_id": sample,
            "case_id": "wechat-capabilities", "turn": turn}


def model_result(text="safe answer"):
    return parse_response(response(message(text)))


class BenchmarkTracingTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_scope_reports_agent_identity_without_client_or_payload(self):
        with capture(LANGSMITH_TRACING="false"), patch.object(
            tracing, "Client", side_effect=AssertionError("disabled tracing constructed a client"),
        ), tracing.trace_scope(binding()) as scope:
            trace = tracing.start_trace(str(uuid4()), SETTINGS)
            observed = tracing.bind_request(str(uuid4()), "business")
            self.assertEqual(observed["agent_run_id"], trace.run_id)
            self.assertIsNone(observed["trace_id"])
            self.assertNotIn("span_id", observed)
            trace.finish()
            await trace.close()
        status = scope.statuses()[0]
        self.assertEqual(status["upload_state"], "disabled")
        self.assertTrue(status["shutdown_confirmed"])
        self.assertEqual(status["spans"], [])
        self.assertEqual(set(status["counts"].values()), {0})
        status["binding"]["case_id"] = "mutated"
        self.assertEqual(scope.statuses()[0]["binding"], binding())
        self.assertEqual(tracing.bind_request(str(uuid4()), "business"), {})

    async def test_disabled_binding_filters_known_credentials(self):
        values = {**binding(), "case_id": MODEL_SECRET}
        with capture(LANGSMITH_TRACING="false"), tracing.trace_scope(values) as scope:
            trace = tracing.start_trace(str(uuid4()), SETTINGS)
            observed = tracing.bind_request(str(uuid4()), "business")
            await trace.close()
        self.assertNotIn(MODEL_SECRET, json.dumps([observed, scope.statuses()]))

    async def test_session_model_summary_and_tool_spans_match_wire_identities(self):
        request_ids = [str(uuid4()), str(uuid4())]
        summary_id, session_id = str(uuid4()), str(uuid4())
        secret_summary = "PRIVATE_SUMMARY_BODY_781a"
        with capture() as (transport, _), tracing.trace_scope(binding()) as scope:
            trace = tracing.start_trace(str(uuid4()), SETTINGS)
            trace.bind_session(session_id)
            trace.set_model_step("step-0001", 1)
            model = trace.model_start("local-check", [{"role": "user", "content": MODEL_SECRET}], None)
            request = tracing.bind_request(request_ids[0], "business")
            trace.model_end(model, result=model_result(TRACE_SECRET))
            trace.model_complete()
            tool = trace.tool_start({"call_id": "call-a", "name": "list_directory", "arguments": '{"path":"."}'},
                                    step_id="step-0001", request_index=1)
            trace.tool_end(tool, output={"ok": False, "error": {"code": "invalid_arguments"}},
                           outcome="rejected", stage="argument_validation")
            summary = trace.summary_start("local-check", {"summary_id": summary_id, "summary_call": 1,
                                                         "step_id": "step-0002"})
            summary_request = tracing.bind_request(request_ids[1], "summary")
            trace.summary_end(summary, result=model_result(secret_summary))
            trace.finish_checkpoint(str(uuid4()))
            trace.finish()
            await trace.close()
        status = scope.statuses()[0]
        self.assertEqual(status["upload_state"], "submission_completed")
        self.assertEqual(status["counts"], {"generated": 4, "queued": 4, "attempted": 4,
                                           "submitted": 4, "failed": 0, "dropped": 0, "unconfirmed": 0})
        self.assertEqual(status["session_id"], session_id)
        wire = {span["id"]: span for span in transport.sent}
        self.assertEqual(set(wire), {span["span_id"] for span in status["spans"]})
        for span in status["spans"]:
            payload = wire[span["span_id"]]
            self.assertEqual(span["trace_id"], payload["trace_id"])
            self.assertEqual(span["parent_run_id"], payload.get("parent_run_id"))
            for key, value in span["metadata"].items():
                self.assertEqual(value, payload["extra"]["metadata"][key])
            self.assertEqual(span["metadata"]["session_id"], session_id)
            self.assertEqual({key: span["metadata"][key] for key in binding()}, binding())
            self.assertNotIn("inputs", span)
            self.assertNotIn("outputs", span)
        self.assertEqual(request["span_id"], model.payload["id"])
        self.assertEqual(request["session_id"], session_id)
        self.assertEqual(summary_request["span_id"], summary.payload["id"])
        self.assertEqual(summary_request["summary_id"], summary_id)
        for secret in (MODEL_SECRET, TRACE_SECRET, secret_summary):
            self.assertNotIn(secret, json.dumps([transport.sent, status]))
        self.assertEqual(wire[summary_request["span_id"]]["outputs"]["content_omitted"], True)
        self.assertEqual(next(s for s in status["spans"] if s["span_id"] == summary_request["span_id"])["usage"],
                         {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})

    async def test_unadmitted_span_is_not_fabricated_as_a_request(self):
        with capture(), tracing.trace_scope(binding()) as scope:
            trace = tracing.start_trace(str(uuid4()), SETTINGS)
            span = trace.summary_start("local-check", {"summary_call": 1})
            # A budget abort occurs before bind_request and before SDK transport.
            trace.summary_end(span, error_code="model_interrupted", stage="model_request")
            trace.finish(error_code="run_interrupted")
            await trace.close()
        status = scope.statuses()[0]
        self.assertEqual(status["counts"]["generated"], 2)
        self.assertFalse(any("request_id" in item["metadata"] for item in status["spans"]))
        self.assertEqual(status["upload_state"], "submission_completed")
        self.assertEqual(next(s for s in status["spans"] if s["run_type"] == "llm")["metadata"]["outcome"], "failed")

    async def test_nested_and_concurrent_scopes_do_not_mix_runs(self):
        async def exercise(sample):
            with tracing.trace_scope(binding(sample)) as scope:
                trace = tracing.start_trace(str(uuid4()), SETTINGS)
                await asyncio.sleep(0)
                observed = tracing.bind_request(str(uuid4()), "business")
                self.assertEqual(observed["sample_run_id"], sample)
                with tracing.trace_scope(binding("nested")) as nested:
                    self.assertEqual(tracing.bind_request(str(uuid4()), "business"), {})
                    inner = tracing.start_trace(str(uuid4()), SETTINGS)
                    await inner.close()
                self.assertEqual(tracing.bind_request(str(uuid4()), "business")["agent_run_id"], trace.run_id)
                await trace.close()
            self.assertEqual(len(scope.statuses()), 1)
            self.assertEqual(len(nested.statuses()), 1)
            return scope.statuses()[0]

        with capture(LANGSMITH_TRACING="false"):
            first, second = await asyncio.gather(exercise("sample-one"), exercise("sample-two"))
        self.assertNotEqual(first["agent_run_id"], second["agent_run_id"])
        self.assertNotEqual(first["binding"]["sample_run_id"], second["binding"]["sample_run_id"])

    async def test_configuration_and_upload_failure_are_separate_safe_states(self):
        with capture(LANGSMITH_API_KEY=""), tracing.trace_scope(binding()) as scope:
            trace = tracing.start_trace(str(uuid4()), SETTINGS)
            await trace.close()
        status = scope.statuses()[0]
        self.assertEqual(status["upload_state"], "configuration_error")
        self.assertEqual(status["warning_codes"], ["configuration"])
        self.assertIsNone(status["trace_id"])
        with capture(401) as (transport, _), tracing.trace_scope(binding()) as scope:
            trace = tracing.start_trace(str(uuid4()), SETTINGS)
            trace.finish()
            await trace.close()
        status = scope.statuses()[0]
        self.assertEqual(status["upload_state"], "incomplete")
        self.assertEqual(status["counts"]["failed"], 1)
        self.assertEqual(status["counts"]["attempted"], 1)
        self.assertEqual(status["counts"]["submitted"], 0)
        self.assertEqual(len(transport.sent), 1)
        self.assertEqual(status["warning_codes"], ["upload"])
        self.assertNotIn(TRACE_SECRET, json.dumps(status))

    async def test_filter_failure_is_not_reported_as_successful_submission(self):
        with capture(), tracing.trace_scope(binding()) as scope:
            trace = tracing.start_trace(str(uuid4()), SETTINGS)
            with patch.object(TraceSanitizer, "cap_payload", side_effect=ValueError("PRIVATE_FILTER_FAILURE")):
                trace.finish()
            await trace.close()
        status = scope.statuses()[0]
        self.assertEqual(status["upload_state"], "incomplete")
        self.assertEqual(status["counts"]["generated"], 1)
        self.assertEqual(status["counts"]["queued"], 0)
        self.assertEqual(status["counts"]["unconfirmed"], 1)
        self.assertEqual(status["warning_codes"], ["recording"])
        self.assertNotIn("PRIVATE_FILTER_FAILURE", json.dumps(status))

    async def test_timeout_and_queue_drop_snapshot_does_not_change_after_late_upload(self):
        entered, release = threading.Event(), threading.Event()

        class BlockedClient:
            def create_run(self, **kwargs):
                entered.set()
                release.wait(3)

            def close(self, **kwargs):
                pass

        trace = None
        try:
            with capture(), patch.object(tracing, "QUEUE_SIZE", 1), patch.object(
                tracing, "CLOSE_TIMEOUT_SECONDS", 0.01,
            ), patch.object(tracing, "_create_client", return_value=BlockedClient()), tracing.trace_scope(binding()) as scope:
                trace = tracing.start_trace(str(uuid4()), SETTINGS)
                first = trace.model_start("local-check", [], None)
                trace.model_end(first, result=model_result())
                deadline = asyncio.get_running_loop().time() + 2
                while not entered.is_set() and asyncio.get_running_loop().time() < deadline:
                    await asyncio.sleep(0.001)
                self.assertTrue(entered.is_set())
                second = trace.summary_start("local-check")
                trace.summary_end(second, result=model_result())
                trace.finish()  # Root is dropped while one span waits in the queue.
                await trace.close()
                before = scope.statuses()[0]
            self.assertEqual(before["upload_state"], "incomplete")
            self.assertFalse(before["shutdown_confirmed"])
            self.assertEqual(before["counts"]["attempted"], 1)
            self.assertEqual(before["counts"]["dropped"], 1)
            self.assertEqual(before["counts"]["unconfirmed"], 2)
            self.assertEqual(before["warning_codes"], ["overflow", "timeout"])
            release.set()
            await asyncio.to_thread(trace._worker.join, 2)
            self.assertFalse(trace._worker.is_alive())
            self.assertEqual(scope.statuses()[0], before)
        finally:
            release.set()
            if trace and trace._worker:
                await asyncio.to_thread(trace._worker.join, 2)

    async def test_real_runtime_session_requests_and_results_are_unchanged(self):
        outcomes, sent_requests = [], []
        with TemporaryDirectory() as temporary:
            settings = replace(SETTINGS, log_dir=Path(temporary))
            for enabled in (False, True):
                clients = []
                mappings = []

                def create_client(_settings):
                    client = LocalClient([response(message("same answer"))])
                    original = client.responses.create

                    async def observed(**kwargs):
                        mappings.append(tracing.bind_request(str(uuid4()), "business"))
                        return await original(**kwargs)

                    client.responses.create = observed
                    clients.append(client)
                    return client

                with capture(LANGSMITH_TRACING=str(enabled).lower()), patch.object(runtime, "create_client", side_effect=create_client):
                    manager = SessionManager()
                    session_id = manager.create_session(settings)
                    turns = []
                    statuses = []
                    for turn in (1, 2):
                        with tracing.trace_scope(binding(turn=turn)) as scope:
                            returned = await manager.run_turn(session_id, "same input")
                        self.assertTrue(returned.checkpoint_complete)
                        turns.append(returned.state)
                        statuses.extend(scope.statuses())
                self.assertEqual([item["agent_run_id"] for item in mappings], [s["run_id"] for s in turns])
                self.assertEqual([item["binding"]["turn"] for item in statuses], [1, 2])
                self.assertTrue(all(client.closed for client in clients))
                self.assertTrue(all(s["stop_reason"] == "completed" for s in turns))
                if enabled:
                    self.assertTrue(all(s["session_id"] == session_id for s in statuses))
                    self.assertTrue(all(s["upload_state"] == "submission_completed" for s in statuses))
                    self.assertTrue(all("span_id" in m for m in mappings))
                outcomes.append([s["result"] for s in turns])
                sent_requests.append([deepcopy(c.requests) for c in clients])
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(sent_requests[0], sent_requests[1])

    async def test_binding_rejects_free_text_and_mismatched_purpose(self):
        for changes in ({"turn": True}, {"case_id": "contains\ncontrol"}, {"extra": "payload"}):
            with self.assertRaises(ValueError):
                with tracing.trace_scope({**binding(), **changes}):
                    pass
        with capture(), tracing.trace_scope(binding()):
            trace = tracing.start_trace(str(uuid4()), SETTINGS)
            span = trace.summary_start("local-check")
            unmatched = tracing.bind_request(str(uuid4()), "business")
            self.assertNotIn("span_id", unmatched)
            with self.assertRaises(ValueError):
                tracing.bind_request(str(uuid4()), "invalid-purpose")
            with self.assertRaises(ValueError):
                tracing.bind_request("arbitrary-data", "summary")
            tracing.bind_request(str(uuid4()), "summary")
            duplicate = tracing.bind_request(str(uuid4()), "summary")
            self.assertNotIn("span_id", duplicate)
            trace.summary_end(span, result=model_result())
            trace.finish()
            await trace.close()
        self.assertEqual(trace.status()["warning_codes"], ["recording"])

    async def test_request_observation_failure_does_not_change_actual_model_execution(self):
        results, requests = [], []
        with TemporaryDirectory() as temporary:
            settings = replace(SETTINGS, log_dir=Path(temporary))
            for failure in (None, ValueError("PRIVATE_ASSOCIATION_FAILURE"), KeyError("PRIVATE_ASSOCIATION_FAILURE")):
                client = LocalClient([response(message("unchanged actual answer"))])
                original = client.responses.create
                associations = []

                async def observed(**kwargs):
                    associations.append(tracing.bind_request(str(uuid4()), "business"))
                    return await original(**kwargs)

                client.responses.create = observed
                inject = (patch.object(tracing.RunTrace, "_bind_request", side_effect=failure)
                          if failure else nullcontext())
                with capture() as (_, stderr), patch.object(runtime, "create_client", return_value=client), \
                        tracing.trace_scope(binding()) as scope, inject:
                    state = await runtime.run_agent("same actual input", settings)
                self.assertEqual(state["stop_reason"], "completed")
                self.assertTrue(client.closed)
                self.assertEqual(len(client.requests), 1)
                self.assertEqual(len(associations), 1)
                if failure:
                    self.assertEqual(set(associations[0]), {"agent_run_id", "request_id", "purpose", *binding()})
                    self.assertEqual(associations[0]["agent_run_id"], state["run_id"])
                    self.assertEqual(scope.statuses()[0]["warning_codes"], ["recording"])
                    self.assertEqual(scope.statuses()[0]["upload_state"], "incomplete")
                    self.assertNotIn("PRIVATE_ASSOCIATION_FAILURE", stderr.getvalue())
                    self.assertNotIn("PRIVATE_ASSOCIATION_FAILURE", json.dumps(scope.statuses()))
                else:
                    self.assertIn("span_id", associations[0])
                results.append(state["result"])
                requests.append(deepcopy(client.requests))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(requests[0], requests[2])


if __name__ == "__main__":
    unittest.main()
