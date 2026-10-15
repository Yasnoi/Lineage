"""Recovery identity, explicit fallback, and failed binding cleanup contracts."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from langgraph.errors import NodeCancelledError

from benchmarks.checks.support.runtime import LocalClient, SETTINGS, TOOL, call, message, response
from lineage_agent.agent import runtime
from lineage_agent.agent.sessions import RecoveryChoice, SessionError, SessionManager
from lineage_agent.config.settings import ConfigurationError
from lineage_agent.archive import ArchiveError, ArchiveStore
from lineage_agent.tools.types import ToolArgumentError, ToolExecutionError, ToolRecoveryBinding


class ExecutionReviewTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="execution-review-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = replace(SETTINGS, application_root=self.root, log_dir=self.root / "logs")
        asyncio.get_event_loop().slow_callback_duration = 10

    async def interrupted(self, *, seed=False, options=None):
        class Interrupted(LocalClient):
            async def create(self, **kwargs):
                raise asyncio.CancelledError()

        with SessionManager() as manager:
            sid = manager.create_session(self.settings, **(options or {}))
            if seed:
                with patch.object(runtime, "create_client", return_value=LocalClient([response(message("seed"))])):
                    self.assertEqual((await manager.run_turn(sid, "seed")).snapshot_status, "published")
            with patch.object(runtime, "create_client", return_value=Interrupted([])):
                with self.assertRaises((asyncio.CancelledError, NodeCancelledError)):
                    await manager.run_turn(sid, "preserve this interrupted task")
            run_id = manager._checkpointer.archive(sid).unfinished_runs()[0]["run_id"]
        return sid, run_id

    async def test_cached_choice_identity_cannot_be_reused_with_a_different_target(self):
        sid, run_id = await self.interrupted()
        with SessionManager() as manager:
            inspection = manager.inspect_run(sid, run_id, application_root=self.root)
            opened = await manager.open_run(sid, run_id, choice=inspection.choice, application_root=self.root)
            client = LocalClient([response(message("completed"))])
            with patch.object(runtime, "create_client", return_value=client):
                result = await manager.resume_run(sid, choice=opened.choice)
                self.assertEqual(result.status, "completed")
                self.assertEqual(await manager.resume_run(sid, choice=opened.choice), result)
                for changed in (replace(opened.choice, run_id="another-run"),
                                replace(opened.choice, execution_point_id="another-point"),
                                replace(opened.choice, revision=opened.choice.revision + 1)):
                    with self.subTest(choice=changed):
                        with self.assertRaises(SessionError) as rejected:
                            await manager.resume_run(sid, choice=changed)
                        self.assertEqual(rejected.exception.code, "recovery_changed")
            self.assertEqual(len(client.requests), 1)

    async def test_explicit_fallback_preserves_facts_and_permanently_rejects_old_execution(self):
        sid, run_id = await self.interrupted(seed=True)
        with SessionManager() as manager:
            inspected = manager.inspect_run(sid, run_id, application_root=self.root)
            with self.assertRaises(SessionError) as prompt:
                await manager.open_session(sid, application_root=self.root)
            self.assertEqual(prompt.exception.code, "recovery_required")
            details = prompt.exception.details
            choice = RecoveryChoice(details["snapshot_id"], tuple(details["unfinished_run_ids"]))
            with patch.object(runtime, "create_client", side_effect=AssertionError("fallback executed")):
                await manager.open_session(sid, application_root=self.root, recovery=choice)
            archive = manager._checkpointer.archive(sid)
            self.assertEqual(archive.run_status(run_id)["status"], "abandoned")
            self.assertEqual(archive.budget_projection(run_id)["reserved"]["model_calls"], 1)
            manager.close_session(sid)
            self.assertEqual(manager.inspect_run(sid, run_id, application_root=self.root).status, "abandoned")
            with self.assertRaises(SessionError) as rejected:
                await manager.open_run(sid, run_id, choice=inspected.choice, application_root=self.root)
            self.assertEqual(rejected.exception.code, "run_abandoned")
            self.assertFalse(manager._sessions)

    async def test_incompatible_custom_binding_releases_lock_before_valid_open(self):
        async def tool(name, arguments):
            raise AssertionError("open must never execute tools")

        binding = ToolRecoveryBinding("test.review", 1, {TOOL["name"]: "safe_repeat"})
        options = {"tools": [TOOL], "execute_tool": tool, "recovery_binding": binding}
        sid, run_id = await self.interrupted(options=options)
        with SessionManager() as manager:
            inspection = manager.inspect_run(sid, run_id, application_root=self.root)
            with self.assertRaises(ConfigurationError):
                await manager.open_run(sid, run_id, choice=inspection.choice, application_root=self.root,
                    **{**options, "recovery_binding": replace(binding, version=2)})
            self.assertFalse(manager._sessions)
        with SessionManager() as other:
            inspection = other.inspect_run(sid, run_id, application_root=self.root)
            with patch.object(runtime, "create_client", side_effect=AssertionError("open executed")):
                opened = await other.open_run(sid, run_id, choice=inspection.choice,
                                              application_root=self.root, **options)
            self.assertEqual(opened.run_id, run_id)

    async def test_saved_tool_failure_and_rejection_are_reused_without_calling_executor(self):
        for error_type, reason, outcome in ((ToolExecutionError, "file_not_found", "failed"),
                                            (ToolArgumentError, "outside_project_root", "rejected")):
            with self.subTest(outcome=outcome):
                executed = []

                async def tool(name, arguments):
                    executed.append((name, arguments))
                    raise error_type(reason_code=reason)

                binding = ToolRecoveryBinding("test.saved-error", 1, {TOOL["name"]: "safe_repeat"})
                options = {"tools": [TOOL], "execute_tool": tool, "recovery_binding": binding}
                original_finish = ArchiveStore.finish_attempt

                def finish(archive, attempt_id, result):
                    saved = original_finish(archive, attempt_id, result)
                    if "error" in result:
                        raise asyncio.CancelledError()
                    return saved

                with SessionManager() as manager:
                    sid = manager.create_session(self.settings, **options)
                    client = LocalClient([response(call("saved-error", arguments='{"key":"once"}'))])
                    with patch.object(runtime, "create_client", return_value=client), \
                            patch.object(ArchiveStore, "finish_attempt", new=finish):
                        with self.assertRaises((asyncio.CancelledError, NodeCancelledError)):
                            await manager.run_turn(sid, "keep known error")
                    run_id = manager._checkpointer.archive(sid).unfinished_runs()[0]["run_id"]
                self.assertEqual(len(executed), 1)
                with SessionManager() as manager:
                    inspection = manager.inspect_run(sid, run_id, application_root=self.root)
                    opened = await manager.open_run(sid, run_id, choice=inspection.choice,
                                                     application_root=self.root, **options)
                    with patch.object(runtime, "create_client", return_value=LocalClient([response(message("done"))])):
                        resumed = await manager.resume_run(sid, choice=opened.choice)
                    self.assertEqual(resumed.status, "completed", resumed.reason)
                    self.assertEqual(resumed.turn.state["tool_calls"], 1)
                    self.assertEqual(resumed.turn.state["tool_outcomes"][outcome], 1)
                    self.assertEqual(len(executed), 1)

    async def test_tool_result_archive_failure_preserves_real_outcome_in_log_and_trace(self):
        from benchmarks.checks.support.tracing import capture
        for error_type, reason, outcome, stage in (
                (None, None, "succeeded", "execution"),
                (ToolExecutionError, "file_not_found", "failed", "execution"),
                (ToolArgumentError, "outside_project_root", "rejected", "argument_validation")):
            with self.subTest(outcome=outcome):
                executed = []

                async def tool(name, arguments):
                    executed.append(name)
                    if error_type is not None:
                        raise error_type(reason_code=reason)
                    return "saved body"

                original_finish = ArchiveStore.finish_attempt

                def finish(archive, attempt_id, result):
                    if "error" in result or "artifact" in result:
                        raise ArchiveError("archive_write_error")
                    return original_finish(archive, attempt_id, result)

                client = LocalClient([response(call("first"), call("pending"))])
                with capture() as (transport, _), SessionManager() as manager:
                    sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=tool)
                    with patch.object(runtime, "create_client", return_value=client), \
                            patch.object(ArchiveStore, "finish_attempt", new=finish):
                        turn = await manager.run_turn(sid, "preserve failure facts")
                    self.assertEqual(turn.snapshot_status, "not_published")
                    state = turn.state
                self.assertEqual(executed, [TOOL["name"]])
                self.assertEqual(state["tool_outcomes"][outcome], 1)
                self.assertEqual(state["tool_outcomes"]["skipped"], 0)
                self.assertEqual(state["tool_calls"], 1)
                self.assertEqual(len(client.requests), 1)
                self.assertFalse(any(item.get("type") == "function_call_output" for item in state["input"]))
                events = [json.loads(line) for line in Path(state["log_path"]).read_text(encoding="utf-8").splitlines()]
                outcomes = [item for item in events if item["event"] in {"tool.succeeded", "tool.failed", "tool.rejected"}]
                self.assertEqual(len(outcomes), 1)
                self.assertEqual(outcomes[0]["event"], "tool." + outcome)
                self.assertEqual(outcomes[0]["stage"], stage)
                self.assertEqual(outcomes[0]["persistence_error_code"], "archive_write_error")
                self.assertFalse(any(item["event"] == "tool.result_created" for item in events))
                spans = [item for item in transport.sent if item["run_type"] == "tool"]
                self.assertEqual(len(spans), 1)
                self.assertEqual(spans[0]["extra"]["metadata"]["outcome"], outcome)
                self.assertEqual(spans[0]["extra"]["metadata"]["stage"], stage)
                if reason is not None:
                    self.assertEqual(outcomes[0]["reason_code"], reason)
                    self.assertEqual(spans[0]["extra"]["metadata"]["reason_code"], reason)


if __name__ == "__main__":
    unittest.main()
