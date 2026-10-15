"""Actual dispatch keeps pre-request log barriers and per-execution trace identity."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from langgraph.errors import NodeCancelledError

from benchmarks.checks.support.runtime import LocalClient, SETTINGS, TOOL, call, message, response, recording_failure
from benchmarks.checks.support.tracing import capture
from benchmarks.tests.test_benchmark_tracing import binding
from lineage_agent.agent import runtime
from lineage_agent.agent.execution import DurableExecution
from lineage_agent.agent.sessions import SessionManager
from lineage_agent.archive import ArchiveStore
from lineage_agent.recording import tracing


class ExecutionDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="execution-diagnostics-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = replace(SETTINGS, application_root=self.root, log_dir=self.root / "logs")
        asyncio.get_event_loop().slow_callback_duration = 10

    async def test_real_dispatch_is_logged_and_bound_before_client(self):
        identities = []
        test = self

        class InspectClient(LocalClient):
            async def create(self, **kwargs):
                events = [json.loads(line) for path in (test.root / "logs").glob("*.jsonl")
                          for line in path.read_text(encoding="utf-8").splitlines()]
                started = [event for event in events if event["event"] == "model.started"]
                test.assertEqual(len(started), 1)
                identity = tracing.bind_request(str(uuid4()), "business")
                test.assertEqual(identity["step_id"], started[0]["step_id"])
                test.assertEqual(identity["model_call"], 1)
                test.assertIn("span_id", identity)
                identities.append(identity)
                return await super().create(**kwargs)

        client = InspectClient([response(message())])
        with capture(), tracing.trace_scope(binding("dispatch")), patch.object(runtime, "create_client", return_value=client):
            state = await runtime.run_agent("input", self.settings)
        self.assertEqual(state["stop_reason"], "completed")
        self.assertEqual(identities[0]["agent_run_id"], state["run_id"])
        self.assertEqual(len(client.requests), 1)

    async def test_failed_started_log_does_not_dispatch_or_reserve_attempt(self):
        client = LocalClient([response(message())])
        with capture(LANGSMITH_TRACING="false"), SessionManager() as manager:
            sid = manager.create_session(self.settings)
            with recording_failure("model.started"), patch.object(runtime, "create_client", return_value=client):
                turn = await manager.run_turn(sid, "input")
            self.assertEqual(turn.state["error_code"], "logging_error")
            self.assertEqual(client.requests, [])
            archive = manager._checkpointer.archive(sid)
            operations = archive.operations(turn.state["run_id"])
            self.assertTrue(operations)
            self.assertTrue(all(not archive.attempts(op["operation_id"]) for op in operations))

    async def test_retry_last_slot_reports_fixed_request_flags(self):
        settings = replace(self.settings, max_model_calls=2)

        class Interrupted(LocalClient):
            async def create(self, **kwargs):
                self.requests.append(kwargs)
                raise asyncio.CancelledError()

        original = Interrupted([])
        with capture(LANGSMITH_TRACING="false"):
            with SessionManager() as manager:
                sid = manager.create_session(settings)
                with patch.object(runtime, "create_client", return_value=original):
                    with self.assertRaises((asyncio.CancelledError, NodeCancelledError)):
                        await manager.run_turn(sid, "input")
                run_id = manager._checkpointer.archive(sid).unfinished_runs()[0]["run_id"]
            with SessionManager() as manager:
                inspection = manager.inspect_run(sid, run_id, application_root=self.root)
                opened = await manager.open_run(sid, run_id, choice=inspection.choice, application_root=self.root)
                client = LocalClient([response(message())])
                with patch.object(runtime, "create_client", return_value=client):
                    resumed = await manager.resume_run(sid, choice=opened.choice)
                self.assertEqual(resumed.status, "completed", resumed)
                self.assertEqual(client.requests, original.requests)
                events = [json.loads(line) for line in Path(resumed.turn.state["log_path"]).read_text(encoding="utf-8").splitlines()]
                started = [event for event in events if event["event"] == "model.started"]
                self.assertEqual(len(started), 1)
                self.assertEqual(started[0]["remaining_model_calls"], 1)
                self.assertFalse(started[0]["final_answer_call"])
                self.assertNotIn("instructions", client.requests[0])

    async def test_saved_response_reuse_keeps_all_interrupted_logs_without_new_send(self):
        native = DurableExecution.send_request

        async def interrupt_after_saved(coordinator, *args, **kwargs):
            await native(coordinator, *args, **kwargs)
            raise asyncio.CancelledError()

        client = LocalClient([response(message("saved response"))])
        with capture(), tracing.trace_scope(binding("resume")) as scope:
            with SessionManager() as manager:
                sid = manager.create_session(self.settings)
                with patch.object(runtime, "create_client", return_value=client), \
                        patch.object(DurableExecution, "send_request", new=interrupt_after_saved):
                    with self.assertRaises((asyncio.CancelledError, NodeCancelledError)):
                        await manager.run_turn(sid, "input")
                run_id = manager._checkpointer.archive(sid).unfinished_runs()[0]["run_id"]
            for should_interrupt in (True, False):
                with SessionManager() as manager:
                    inspection = manager.inspect_run(sid, run_id, application_root=self.root)
                    opened = await manager.open_run(sid, run_id, choice=inspection.choice, application_root=self.root)
                    with patch.object(runtime, "create_client", return_value=LocalClient([])):
                        if should_interrupt:
                            with patch.object(DurableExecution, "send_request", new=interrupt_after_saved):
                                with self.assertRaises((asyncio.CancelledError, NodeCancelledError)):
                                    await manager.resume_run(sid, choice=opened.choice)
                        else:
                            result = await manager.resume_run(sid, choice=opened.choice)
                            self.assertEqual(result.status, "completed", result)
                            state = result.turn.state
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(state["model_calls"], 1)
        segments = state["log_segments"]
        self.assertEqual(len(segments), 3)
        self.assertEqual([segment["complete"] for segment in segments], [False, False, True])
        self.assertEqual(len({segment["path"] for segment in segments}), 3)
        for index, segment in enumerate(segments):
            events = [json.loads(line) for line in Path(segment["path"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(event["event"] == "model.started" for event in events), int(index == 0))
            self.assertTrue(all(event["run_id"] == run_id for event in events))
        statuses = scope.statuses()
        self.assertEqual(len(statuses), 3)
        self.assertEqual({status["agent_run_id"] for status in statuses}, {run_id})
        self.assertEqual(len({status["trace_id"] for status in statuses}), 3)
        self.assertTrue(all(span["metadata"]["run_id"] == run_id
                            for status in statuses for span in status["spans"]))

    async def test_saved_raw_tool_result_is_marked_reused_without_another_tool_start(self):
        executed = []

        async def tool(name, arguments):
            executed.append((name, arguments))
            return "saved observation"

        native = ArchiveStore.finish_attempt

        def stop_after_raw(archive, attempt_id, result):
            saved = native(archive, attempt_id, result)
            if "artifact" in result:
                raise asyncio.CancelledError()
            return saved

        with capture(), tracing.trace_scope(binding("tool-resume")) as scope:
            with SessionManager() as manager:
                sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=tool)
                with patch.object(runtime, "create_client", return_value=LocalClient([response(call())])), \
                        patch.object(ArchiveStore, "finish_attempt", new=stop_after_raw):
                    with self.assertRaises((asyncio.CancelledError, NodeCancelledError)):
                        await manager.run_turn(sid, "input")
                run_id = manager._checkpointer.archive(sid).unfinished_runs()[0]["run_id"]
            with SessionManager() as manager:
                inspection = manager.inspect_run(sid, run_id, application_root=self.root)
                opened = await manager.open_run(sid, run_id, choice=inspection.choice, application_root=self.root,
                                               tools=[TOOL], execute_tool=tool)
                with patch.object(runtime, "create_client", return_value=LocalClient([response(message())])):
                    resumed = await manager.resume_run(sid, choice=opened.choice)
                self.assertEqual(resumed.status, "completed", resumed)
                state = resumed.turn.state
        self.assertEqual(len(executed), 1)
        self.assertEqual(state["tool_calls"], 1)
        events = [json.loads(line) for line in Path(state["log_path"]).read_text(encoding="utf-8").splitlines()]
        self.assertFalse(any(event["event"] == "tool.started" for event in events))
        reused = [event for event in events if event["event"] == "tool.result_reused"]
        self.assertEqual(len(reused), 1)
        self.assertTrue(reused[0]["reused_result"])
        self.assertFalse(reused[0]["executor_entered"])
        facts = state["execution_facts"]
        self.assertEqual(len(facts), 1)
        self.assertTrue(facts[0]["reused_result"])
        self.assertTrue(facts[0]["executor_entered"])
        spans = [span for span in scope.statuses()[-1]["spans"] if span["run_type"] == "tool"]
        self.assertEqual(len(spans), 1)
        self.assertTrue(spans[0]["metadata"]["reused_result"])
        self.assertFalse(spans[0]["metadata"]["executor_entered"])
        self.assertEqual(spans[0]["metadata"]["model_call"], 1)


if __name__ == "__main__":
    unittest.main()
