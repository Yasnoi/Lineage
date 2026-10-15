"""Business archive adoption and reconstruction without historical graph state."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmarks.checks.support.runtime import LocalClient, SETTINGS, TOOL, call, message, response
from benchmarks.tests.test_prefix_compaction import _Client
from lineage_agent.agent import runtime
from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.context import ContextRecords
from lineage_agent.agent.records import RecordStore
from lineage_agent.agent.sessions import SessionManager, SessionError
from lineage_agent.archive import ArchiveError, ArchiveStore
from lineage_agent.artifacts.store import ArtifactStore


class ArchiveRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="archive-runtime-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = replace(SETTINGS, application_root=self.root, log_dir=self.root / "logs",
                                context_budget_tokens=24000, max_output_tokens=256,
                                context_safety_margin_tokens=64, max_model_calls=25,
                                summary_history_max_output_tokens=512, summary_run_max_output_tokens=256)
        self.executed = []
        tracing = patch.dict("os.environ", {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING": "false",
                                           "LANGCHAIN_TRACING_V2": "false"})
        tracing.start()
        self.addCleanup(tracing.stop)

    async def tool(self, name, arguments):
        self.executed.append(arguments["key"])
        return "original-detail=37; " + arguments["key"] + "x" * 3200

    def replies(self, prefix, count=10):
        return [response(call(f"{prefix}{i}", arguments=json.dumps({"key": f"{prefix}{i}"})))
                for i in range(count)] + [response(message("completed"), usage=None)]

    async def test_old_checkpoints_unavailable_rebuild_and_compact_again(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=self.tool)
            artifact = manager.artifacts(sid).put("durable artifact body", tool_name="custom")
            with patch.object(runtime, "create_client", return_value=_Client(self.replies("a"))):
                first = await manager.run_turn(sid, "retain original details")
            self.assertTrue(first.checkpoint_complete, first.checkpoint_error)
            self.assertEqual(first.state["stop_reason"], "completed")
            self.assertGreater(first.state["summary_calls"], 0)
            before = await manager.get_record(sid, first.state["run_id"])
            self.assertEqual(len(before["model_results"]), first.state["model_calls"])
            self.assertEqual(len(before["summary_results"]), first.state["summary_calls"])
            self.assertIsNone(before["model_results"][-1]["result"]["usage"])
            saver = manager._checkpointer
            archived = saver.archive(sid).events(first.state["run_id"])
            states = [event["payload"]["compaction"]["status"] for event in archived if event["kind"] == "compaction"]
            self.assertIn("confirmed", states)
            self.assertIn("used", states)
            for namespace in saver.storage[sid].values():
                for checkpoint_id in list(namespace):
                    if checkpoint_id != first.checkpoint_id:
                        del namespace[checkpoint_id]
            native = saver.read_confirmed
            async def current_only(thread_id, checkpoint_id):
                self.assertEqual(checkpoint_id, saver.last_confirmed(thread_id), "historical checkpoint read")
                return await native(thread_id, checkpoint_id)
            with patch.object(saver, "read_confirmed", side_effect=current_only):
                self.assertEqual(await manager.get_record(sid, first.state["run_id"]), before)
                context = ContextRecords(saver, sid)
                for span in first.state["active_context"]["summary_spans"]:
                    self.assertTrue(await context.summary(span["reference"]))
                with ArchiveStore(self.root / ".lineage" / sid, create=False) as archive:
                    view_id = first.state["resource_view_id"]
                    records = RecordStore(archive=archive, view_id=view_id)
                    try:
                        originals = await records.records(first.state["run_id"])
                        self.assertTrue(any("original-detail=37" in str(r) for r in originals))
                        for ref in archive.view(view_id)["history_scopes"]:
                            self.assertTrue((await records.search(ref, ["original-detail"]))["hits"])
                    finally:
                        records.close()
                    with ArtifactStore() as store:
                        restored = store.restore_session(sid, archive=archive, view_id=view_id)
                        self.assertEqual(restored.read(artifact["artifact_ref"], 0)["text"], "durable artifact body")
                second_client = _Client(self.replies("b"))
                with patch.object(runtime, "create_client", return_value=second_client):
                    second = await manager.run_turn(sid, "continue with a new request")
                self.assertTrue(second.checkpoint_complete, second.checkpoint_error)
                self.assertEqual(second.state["stop_reason"], "completed")
                self.assertGreater(second.state["summary_calls"], 0)
                self.assertEqual(await manager.get_record(sid, first.state["run_id"]), before)

    async def test_same_run_material_is_visible_before_final_record(self):
        with SessionManager() as manager:
            sid = None
            seen = []
            async def probe(name, arguments):
                saver = manager._checkpointer
                records = RecordStore(saver, sid)
                material = await records.records()
                seen.append(material)
                run_id = material[-1]["run_id"]
                with self.assertRaises(ArchiveError):
                    saver.archive(sid).read_run(run_id)
                if arguments["key"] == "second":
                    view_id = saver.confirmed_view(sid)
                    refs = saver.archive(sid).view(view_id)["artifacts"]
                    self.assertEqual(len(refs), 1)
                    artifact = saver.archive(sid).artifact(refs[0], view_id)
                    self.assertEqual(artifact["metadata"]["origin"]["run_id"], run_id)
                    self.assertIn("first confirmed tool detail", manager.artifacts(sid).read(refs[0], 0, view_id=view_id)["text"])
                return "first confirmed tool detail " + "x" * 10000 if arguments["key"] == "first" else "checked"
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=probe)
            replies = [response(call("first", arguments='{"key":"first"}')),
                       response(call("second", arguments='{"key":"second"}')), response(message("done"))]
            with patch.object(runtime, "create_client", return_value=LocalClient(replies)):
                turn = await manager.run_turn(sid, "query confirmed work")
            self.assertTrue(turn.checkpoint_complete, turn.checkpoint_error)
            self.assertEqual(len(seen), 2)
            self.assertTrue(any("first confirmed tool detail" in str(record) for record in seen[1]))

    async def test_artifact_save_failure_stops_batch_without_publishing_resources(self):
        for failure in ("body", "registration"):
            with self.subTest(failure=failure), SessionManager() as manager:
                executed = []
                async def large_tool(name, arguments):
                    executed.append(arguments["key"])
                    return "large executed result " * 2000
                sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=large_tool)
                stable = manager.artifacts(sid).put("previous durable result", tool_name="custom")
                saver = manager._checkpointer
                captured = []
                def failed(*args, **kwargs):
                    captured.append(saver.confirmed_view(sid))
                    if failure == "registration":
                        raise ArchiveError("archive_write_error")
                    raise OSError("injected body write failure")
                rename = Path.rename
                def failed_body(path, destination):
                    if path.suffix == ".tmp":
                        return failed()
                    return rename(path, destination)
                target = patch.object(ArchiveStore, "save_artifact", side_effect=failed) if failure == "registration" \
                    else patch.object(Path, "rename", new=failed_body)
                client = LocalClient([response(call("first", arguments='{"key":"first"}'),
                                               call("second", arguments='{"key":"second"}')),
                                      response(message("must never run"))])
                with target, patch.object(runtime, "create_client", return_value=client):
                    turn = await manager.run_turn(sid, "save large results")
                self.assertEqual(executed, ["first"])
                self.assertFalse(turn.checkpoint_complete)
                self.assertEqual(turn.checkpoint_error, "artifact_write_error")
                self.assertEqual(turn.state["model_calls"], 1)
                self.assertEqual(turn.state["tool_calls"], 1)
                self.assertEqual(turn.state["tool_outcomes"], {"succeeded": 1, "failed": 0, "rejected": 0, "skipped": 0})
                self.assertEqual(saver.confirmed_view(sid), captured[-1])
                self.assertEqual(saver.archive(sid).view(captured[-1])["artifacts"], [stable["artifact_ref"]])
                self.assertEqual(manager.artifacts(sid).read(stable["artifact_ref"], 0)["text"], "previous durable result")
                facts = [event for event in saver.archive(sid).events(turn.state["run_id"])
                         if event["kind"] == "execution_facts"]
                self.assertEqual(facts[-1]["payload"]["facts"][0]["outcome"], "succeeded")
                with self.assertRaises(SessionError):
                    await manager.run_turn(sid, "blocked")

    async def test_archive_failure_boundaries_preserve_facts_and_old_access(self):
        for failure in ("append_event", "prepare_view", "confirm_execution", "publish_snapshot"):
            with self.subTest(failure=failure), SessionManager() as manager:
                sid = manager.create_session(self.settings)
                original = getattr(ArchiveStore, failure)
                captured = []
                def fail(store, *args, **kwargs):
                    saver = manager._checkpointer
                    should_fail = False
                    if failure == "append_event":
                        should_fail = args[1] == "model_result"
                    elif failure == "prepare_view":
                        should_fail = any(w["record_count"] > 1 for w in kwargs["record_files"].values())
                    elif failure == "confirm_execution":
                        point = store.execution(args[0], confirmed=False)
                        should_fail = point["state"].get("result") is not None
                    else:
                        should_fail = True
                    if should_fail:
                        captured.append(saver.confirmed_view(sid))
                        raise ArchiveError("archive_write_error")
                    return original(store, *args, **kwargs)
                with patch.object(ArchiveStore, failure, new=fail), \
                        patch.object(runtime, "create_client", return_value=LocalClient([response(message("paid answer"))])):
                    turn = await manager.run_turn(sid, "request")
                self.assertTrue(captured)
                self.assertFalse(turn.checkpoint_complete)
                self.assertEqual(turn.checkpoint_error, "archive_write_error")
                self.assertEqual(turn.state["result"]["text"], "paid answer")
                self.assertIsNotNone(turn.state["active_context"]["current_run_id"])
                self.assertEqual(manager._checkpointer.confirmed_view(sid), captured[-1])
                self.assertIsNone(manager._sessions[sid].last_checkpoint_id)
                with self.assertRaises(SessionError):
                    await manager.get_record(sid, turn.state["run_id"])
                with self.assertRaises(SessionError):
                    await manager.run_turn(sid, "must not retry")
                events = manager._checkpointer.archive(sid).events(turn.state["run_id"])
                if failure != "append_event":
                    self.assertTrue(any(e["kind"] == "model_result" and e["payload"]["result"]["text"] == "paid answer"
                                        for e in events))


if __name__ == "__main__":
    unittest.main()
