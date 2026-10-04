"""Keep primary execution facts when observation storage cleanup also fails."""

import asyncio
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmarks.checks.support.runtime import LocalClient, SETTINGS, message, response
from benchmarks.core.budgets import WorkerAbort
from benchmarks.core.executors import agent_worker
from lineage_agent.agent import runtime
from lineage_agent.agent.sessions import SessionManager, SessionError
from lineage_agent.artifacts.store import ArtifactError, ArtifactStore
from lineage_agent.config.settings import ConfigurationError
from lineage_agent.entrypoints import cli


PRIVATE = "private-cleanup-exception-content"


class FailingCloseClient(LocalClient):
    async def __aexit__(self, *_):
        self.closed = True
        raise OSError(PRIVATE)


class ObservationLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="observation-lifecycle-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = replace(SETTINGS, log_dir=self.root)
        self.tracing = patch.dict("os.environ", {
            "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
        })
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def events(self):
        return [json.loads(line) for path in self.root.glob("*.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines()]

    def failing_store_close(self):
        original = ArtifactStore.close
        attempts = []

        def close(store):
            attempts.append(store)
            original(store)
            raise ArtifactError("artifact_cleanup_error")

        return close, attempts

    async def test_two_cleanup_failures_keep_answer_and_first_failure(self):
        client = FailingCloseClient([response(message("completed answer"))])
        close, attempts = self.failing_store_close()
        with patch.object(runtime, "create_client", return_value=client), \
                patch.object(ArtifactStore, "close", close):
            state = await runtime.run_agent("finish", self.settings)

        self.assertEqual(state["result"]["text"], "completed answer")
        self.assertEqual(state["model_calls"], 1)
        self.assertEqual(state["error_code"], "client_close_error")
        self.assertEqual(state["cleanup_error"], {
            "code": "client_close_error", "previous_stop_reason": "completed",
        })
        self.assertTrue(client.closed)
        self.assertEqual(len(attempts), 1)
        self.assertTrue(attempts[0]._directory.exists())
        events = self.events()
        self.assertEqual([event["error_code"] for event in events if event["event"] == "run.cleanup_failed"],
                         ["client_close_error", "artifact_cleanup_error"])
        self.assertNotIn(PRIVATE, json.dumps(events))

    async def test_cancellation_remains_primary_when_both_closes_fail(self):
        entered = asyncio.Event()

        class WaitingClient(FailingCloseClient):
            async def create(self, **kwargs):
                entered.set()
                await asyncio.Event().wait()

        client = WaitingClient([])
        close, attempts = self.failing_store_close()
        with patch.object(runtime, "create_client", return_value=client), \
                patch.object(ArtifactStore, "close", close):
            task = asyncio.create_task(runtime.run_agent("wait", self.settings))
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertTrue(client.closed)
        self.assertEqual(len(attempts), 1)
        self.assertTrue(attempts[0]._directory.exists())
        events = self.events()
        self.assertEqual([event["error_code"] for event in events if event["event"] == "run.cleanup_failed"],
                         ["client_close_error", "artifact_cleanup_error"])
        self.assertTrue(any(event["event"] == "run.aborted" for event in events))
        self.assertNotIn(PRIVATE, json.dumps(events))

    async def test_preexecution_configuration_stays_primary_during_cleanup_failure(self):
        close, attempts = self.failing_store_close()
        with patch.object(runtime, "create_client", side_effect=ConfigurationError(PRIVATE)), \
                patch.object(ArtifactStore, "close", close):
            with self.assertRaises(runtime.PreExecutionConfigurationError):
                await runtime.run_agent("invalid connection", self.settings)
        self.assertEqual(len(attempts), 1)
        self.assertTrue(attempts[0]._directory.exists())
        events = self.events()
        self.assertTrue(any(event["event"] == "run.configuration_rejected" for event in events))
        self.assertNotIn(PRIVATE, json.dumps(events))

    async def test_bind_initialization_failure_closes_owned_store_without_model_call(self):
        stores = []

        def factory():
            store = ArtifactStore()
            stores.append(store)
            return store

        with patch.object(runtime, "ArtifactStore", side_effect=factory), \
                patch.object(ArtifactStore, "bind_session", side_effect=ArtifactError("artifact_write_error")), \
                patch.object(runtime, "create_client") as create_client:
            state = await runtime.run_agent("cannot bind", self.settings)

        self.assertEqual(state["error_code"], "artifact_store_init_error")
        self.assertEqual(state["model_calls"], 0)
        self.assertEqual(state["tool_calls"], 0)
        create_client.assert_not_called()
        self.assertEqual(len(stores), 1)
        self.assertFalse(stores[0]._directory.exists())

    async def test_manager_context_preserves_primary_and_cleanup_can_retry(self):
        manager = SessionManager()
        sid = manager.create_session(self.settings)
        manager.artifacts(sid).put("saved", tool_name="custom")
        original = RuntimeError(PRIVATE)
        close, attempts = self.failing_store_close()
        with patch.object(ArtifactStore, "close", close):
            with self.assertRaises(RuntimeError) as caught:
                with manager:
                    raise original

        self.assertIs(caught.exception, original)
        self.assertEqual(manager.cleanup_error, "artifact_cleanup_error")
        self.assertEqual(len(attempts), 1)
        with self.assertRaises(SessionError) as closed:
            manager.create_session(self.settings)
        self.assertEqual(closed.exception.code, "session_manager_closed")
        manager.close()
        self.assertIsNone(manager.cleanup_error)
        manager.close()

    async def test_interactive_reports_cleanup_failure_without_replacing_interrupt(self):
        async def interrupted_loop(settings, dependencies, manager):
            manager.create_session(settings)
            raise KeyboardInterrupt(PRIVATE)

        close, attempts = self.failing_store_close()
        stderr = StringIO()
        with patch.object(cli, "_interactive_loop", interrupted_loop), \
                patch.object(ArtifactStore, "close", close), redirect_stderr(stderr):
            with self.assertRaises(KeyboardInterrupt):
                await cli.interactive(self.settings, {})
        self.assertEqual(stderr.getvalue().count(runtime.ARTIFACT_CLOSE_MESSAGE), 1)
        self.assertNotIn(PRIVATE, stderr.getvalue())
        self.assertEqual(len(attempts), 1)

    async def test_interactive_leaves_standalone_cleanup_error_for_main_to_report(self):
        async def completed_loop(settings, dependencies, manager):
            manager.create_session(settings)
            return 0

        close, _ = self.failing_store_close()
        stderr = StringIO()
        with patch.object(cli, "_interactive_loop", completed_loop), \
                patch.object(ArtifactStore, "close", close), redirect_stderr(stderr):
            with self.assertRaises(ArtifactError) as caught:
                await cli.interactive(self.settings, {})
        self.assertEqual(caught.exception.code, "artifact_cleanup_error")
        self.assertEqual(stderr.getvalue(), "")

    def worker_spec(self, name):
        directory = self.root / name
        directory.mkdir()
        material, output = directory / "material", directory / "output"
        material.mkdir()
        output.mkdir()
        ledger = directory / "requests.jsonl"
        ledger.touch()
        return {
            "batch_id": "lifecycle-batch", "run_id": name, "case_id": "lifecycle-case",
            "definition_version": "test", "prompts": ["first", "second"],
            "profile": {"model_name": "local-check", "base_url": "https://unused.invalid/v1",
                        "max_model_calls": 3},
            "material_root": str(material), "output_dir": str(output), "ledger_path": str(ledger),
            "remaining_seconds": 300, "max_requests": 20,
        }

    async def test_worker_does_not_swallow_standalone_cleanup_interruptions(self):
        original_close = SessionManager.close
        for interruption in (KeyboardInterrupt, asyncio.CancelledError):
            with self.subTest(interruption=interruption.__name__):
                spec = self.worker_spec(interruption.__name__)

                def interrupted_close(manager):
                    original_close(manager)
                    raise interruption(PRIVATE)

                with patch.object(SessionManager, "close", interrupted_close), \
                        patch.dict("os.environ", {"API_KEY": "local-test-only"}), \
                        patch.object(runtime, "create_client", side_effect=lambda settings: LocalClient([response(message())])):
                    with self.assertRaises(interruption):
                        await agent_worker.execute_sample(spec, "test-spec")
                self.assertEqual(len(list(Path(spec["output_dir"]).glob("turn-*-answer.md"))), 2)

    async def test_worker_keeps_primary_failure_when_cleanup_is_interrupted(self):
        spec = self.worker_spec("primary-failure")
        original_close = SessionManager.close

        async def aborted_turn(manager, session_id, prompt):
            raise WorkerAbort("sample_stopped_for_test")

        def interrupted_close(manager):
            original_close(manager)
            raise asyncio.CancelledError(PRIVATE)

        with patch.object(SessionManager, "close", interrupted_close), \
                patch.object(SessionManager, "run_turn", aborted_turn), \
                patch.dict("os.environ", {"API_KEY": "local-test-only"}):
            result, code = await agent_worker.execute_sample(spec, "test-spec")
        self.assertEqual(code, 2)
        self.assertEqual(result["global_stop_reason"], "sample_stopped_for_test")
        self.assertTrue(any(issue["code"] == "artifact_cleanup_error" for issue in result["issues"]))
        self.assertNotIn(PRIVATE, json.dumps(result))


if __name__ == "__main__":
    unittest.main()
