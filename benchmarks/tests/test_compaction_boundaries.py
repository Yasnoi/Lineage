"""Request identity, budget semantics and final/confirmation failure boundaries."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from benchmarks.tests import test_prefix_compaction as fixtures
from benchmarks.checks.support.runtime import message, response, recording_failure
from lineage_agent.agent import compaction, graph, runtime
from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.sessions import SessionManager
from lineage_agent.models.summarization import is_summary_request, summary_request_lane
from lineage_agent.recording.recorder import RunRecorder


class CompactionBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = fixtures.PrefixCompactionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tmp.cleanup)

    async def test_checked_requests_equal_actual_sdk_requests_including_final_call_options(self):
        fixture = self.fixture
        fixture.settings = replace(fixture.settings, max_model_calls=7)
        checks = []
        native = graph.check_context_budget
        def measured(request, risks, settings, **kwargs):
            budget = native(request, risks, settings, **kwargs)
            checks.append((deepcopy(request), deepcopy(budget)))
            return budget
        with patch.object(graph, "check_context_budget", side_effect=measured), patch.object(compaction, "check_context_budget", side_effect=measured):
            state, client = await fixture.standalone(count=6)
        fixture.assert_complete(state)
        self.assertGreater(state["summary_calls"], 0)
        for request in client.requests:
            self.assertTrue(any(request == prepared and budget["decision"] == "allow" for prepared, budget in checks))
        final = client.business_requests[-1]
        self.assertEqual(final["tool_choice"], "none")
        self.assertEqual(final["instructions"], graph.FINAL_ANSWER_INSTRUCTIONS)
        self.assertTrue(all(request["max_output_tokens"] == fixture.settings.max_output_tokens for request in client.summary_requests))

    async def test_summary_above_soft_target_can_commit_with_margin_and_positive_recovery(self):
        def over_soft(request):
            policy = json.loads(request["instructions"].split("Compact request policy: ")[1])
            return fixtures._summary(request, "a" * (policy["target_bytes"] + 10))
        state, _ = await self.fixture.standalone(count=6, summary=over_soft)
        self.fixture.assert_complete(state)
        event = state["compaction_events"][-1]
        self.assertEqual(event["status"], "used")
        self.assertGreater(event["lanes"]["R"]["summary_bytes"], event["lanes"]["R"]["target_bytes"])
        self.assertGreater(event["net_reclaimed"], 0)
        self.assertGreaterEqual(event["recovery_margin"], event["recovery_budget"]["margin_required"])

    async def test_private_final_save_failure_cannot_report_completed(self):
        native = SessionCheckpointer.put
        def fail(saver, config, checkpoint, metadata, versions):
            if metadata.get("source") == "update":
                raise OSError("injected final save")
            return native(saver, config, checkpoint, metadata, versions)
        with patch.object(SessionCheckpointer, "put", new=fail):
            state, client = await self.fixture.standalone(count=0)
        self.assertEqual(state["stop_reason"], "error")
        self.assertEqual(state["error_code"], "checkpoint_write_error")
        self.assertEqual(state["model_calls"], 1)
        self.assertEqual(len(client.business_requests), 1)

    async def test_activation_confirmation_failure_keeps_old_activity_and_paid_facts(self):
        native = SessionCheckpointer.read_confirmed
        async def fail(saver, scope, checkpoint):
            saved = await native(saver, scope, checkpoint)
            if (saved.checkpoint["channel_values"].get("compaction") or {}).get("status") == "pending_save":
                raise CheckpointError("checkpoint_read_error")
            return saved
        with patch.object(SessionCheckpointer, "read_confirmed", new=fail):
            state, client = await self.fixture.standalone(count=6)
        self.assertEqual(state["error_code"], "checkpoint_read_error")
        self.assertIsNone(state["active_context"]["run_summary"])
        self.assertEqual(state["summary_calls"], 1)
        self.assertEqual(len(client.business_requests), 6)
        self.assertEqual(len(self.fixture.executed), 6)

    async def test_summary_logging_boundaries_do_not_send_or_enable_unsaved_work(self):
        for event, expected_calls in (("summary.started", 0), ("summary.returned", 1), ("compaction.activated", 1)):
            with self.subTest(event=event), recording_failure(event) as faults:
                state, client = await self.fixture.standalone(count=6)
            self.assertTrue(faults)
            self.assertEqual(state["summary_calls"], expected_calls)
            self.assertEqual(state["error_code"], "logging_error")
            self.assertEqual(len(client.business_requests), 6)
            self.assertEqual(len(client.summary_requests), expected_calls)
            if event == "summary.started":
                self.assertFalse(state["compaction"]["lanes"]["R"]["model_called"])
                self.assertNotIn("summary_call", state["compaction"]["lanes"]["R"])
                self.assertNotIn("summary_call", state["compaction"])
                self.assertIsNone(state["active_context"]["run_summary"])

        # A failed R start must preserve actual H facts, including on a retry.
        for failed_call in (2, 4):
            with self.subTest(failed_call=failed_call), SessionManager() as manager:
                fixture = self.fixture
                fixture.executed.clear()
                sid = manager.create_session(fixture.settings, tools=[fixtures.TOOL], execute_tool=fixture.tool)
                seed = fixtures._Client([response(message("historical source " + "h" * 4200))])
                with patch.object(runtime, "create_client", return_value=seed):
                    await manager.run_turn(sid, "previous task")
                sent = []
                def summarize(request):
                    sent.append(request)
                    text = "L" * 11000 if failed_call == 4 and len(sent) <= 2 else "facts from the covered prefix"
                    return fixtures._summary(request, text)
                client = fixtures._Client(fixture.replies(6), summarize)
                native_emit = RunRecorder.emit
                faults = []
                def fail_start(recorder, event, **details):
                    if event == "summary.started" and details.get("summary_call") == failed_call:
                        faults.append(details)
                        recorder._fail("write")
                        return False
                    return native_emit(recorder, event, **details)
                with patch.object(runtime, "create_client", return_value=client), patch.object(RunRecorder, "emit", new=fail_start):
                    turn = await manager.run_turn(sid, "current task")
                state = turn.state
                self.assertEqual(len(faults), 1)
                self.assertEqual(state["summary_calls"], failed_call - 1)
                self.assertEqual(len(client.summary_requests), failed_call - 1)
                self.assertEqual(state["compaction"]["summary_call"], failed_call - 1)
                self.assertEqual(state["compaction"]["summary_lane"], "H")
                self.assertTrue(state["compaction"]["lanes"]["H"]["model_called"])
                self.assertFalse(state["compaction"]["lanes"]["R"]["model_called"])
                self.assertNotIn("summary_call", state["compaction"]["lanes"]["R"])
                self.assertIsNone(state["active_context"]["history_summary"])
                self.assertIsNone(state["active_context"]["run_summary"])
                self.assertEqual(fixture.executed, [f"c{index}" for index in range(state["tool_calls"])])
                self.assertEqual(len(client.business_requests), len(fixture.executed))
                record = await manager.get_record(sid, state["run_id"])
                self.assertEqual([value["summary_call"] for value in record["summary_results"]], list(range(1, failed_call)))
                for saved, actual in zip(record["summary_results"], client.summary_requests, strict=True):
                    lane = summary_request_lane(actual)
                    self.assertEqual(saved["compaction"]["summary_lane"], lane)
                    self.assertEqual(saved["compaction"]["summary_id"], saved["compaction"]["lanes"][lane]["summary_id"])
                    self.assertIsNotNone(saved["result"])

    async def test_short_run_has_no_summary_or_history_tools_executed(self):
        state, client = await self.fixture.standalone(count=1)
        self.fixture.assert_complete(state)
        self.assertEqual(state["summary_calls"], 0)
        self.assertFalse(state["compaction_events"])
        self.assertEqual(state["tool_calls"], 1)
        self.assertFalse(client.summary_requests)
