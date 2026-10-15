"""Execution candidates are durable, exact, and absent from business grants."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from lineage_agent.archive import ArchiveError, ArchiveStore
from lineage_agent.artifacts.store import ArtifactError, ArtifactStore
from lineage_agent.artifacts.tools import ArtifactExecutor
from lineage_agent.observations.manager import ObservationManager
from lineage_agent.storage import conversation_paths
from lineage_agent.tools.types import ToolExecutionError


class ExecutionArtifactCandidateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="execution-candidate-")
        self.addCleanup(temporary.cleanup)
        self.paths = conversation_paths(Path(temporary.name), "conversation")
        self.archive = ArchiveStore(self.paths.directory, create=True)
        self.addCleanup(self.archive.close)
        self.store = ArtifactStore()
        self.addCleanup(self.store.close)
        self.handle = self.store.bind_session("session", directory=self.paths.artifacts).bind_origin("run")
        self.identity = {"operation_id": "operation", "attempt_id": "attempt"}
        self.raw = json.dumps({"content": "first KEY " * 2000, "bytes_read": 20000}, ensure_ascii=False)

    def save(self):
        return self.handle.put_candidate(self.raw, tool_name="read_text_file", artifact_id=uuid4().hex,
                                         **self.identity)

    def view(self, refs):
        view = self.archive.prepare_view(record_files={}, history_scopes={}, summaries=[], artifacts=refs)
        self.archive.confirm_view(view, "checkpoint-" + view)
        return view

    async def test_saved_raw_and_preview_do_not_grant_recall_until_explicit_promotion(self):
        saved = self.save()
        ref = saved["artifact_ref"]
        identity = {**self.identity, "sha256": saved["sha256"]}
        self.assertEqual(self.handle.read_candidate_raw(ref, **identity), self.raw)
        self.assertEqual(self.handle.registered_refs(), [])
        current = self.view([])
        executor = ArtifactExecutor(self.handle, view_provider=lambda: current)
        with self.assertRaises(ToolExecutionError):
            await executor("artifact_read", {"artifact_ref": ref, "start": 0, "end": None})
        with patch.object(type(self.handle), "put", side_effect=AssertionError("must not resave")):
            preview = ObservationManager(self.handle).observe_saved(ref, keywords=("KEY",), **identity)
        self.assertEqual(preview["observation"]["artifact_ref"], ref)
        self.assertEqual(preview["observation"]["keyword_status"], "matched")
        self.assertEqual(self.handle.registered_refs(), [])
        self.assertEqual(len(list(self.paths.artifacts.glob("*.raw"))), 1)
        self.handle.promote_candidate(ref, **identity)
        self.handle.promote_candidate(ref, **identity)
        self.assertEqual(self.handle.registered_refs(), [ref])
        with self.assertRaises(ToolExecutionError):
            await executor("artifact_read", {"artifact_ref": ref, "start": 0, "end": None})
        current = self.view([ref])
        result = json.loads(await executor("artifact_read", {"artifact_ref": ref, "start": 0, "end": 5}))
        self.assertEqual(result["text"], "first")

    def test_rebuild_can_load_exact_candidate_without_importing_its_reference(self):
        saved = self.save()
        empty = self.view([])
        self.store.close()
        with ArtifactStore() as rebuilt:
            handle = rebuilt.restore_session("new", archive=self.archive, view_id=empty).bind_origin("run")
            self.assertEqual(handle.registered_refs(), [])
            self.assertEqual(handle.read_candidate_raw(saved["artifact_ref"], **self.identity), self.raw)
            with self.assertRaises(ArtifactError):
                handle.read(saved["artifact_ref"], 0)
            for wrong in ({**self.identity, "attempt_id": "other"},
                          {**self.identity, "operation_id": "other"},
                          {**self.identity, "sha256": "0" * 64}):
                with self.assertRaises(ArtifactError):
                    handle.read_candidate_raw(saved["artifact_ref"], **wrong)
            with self.assertRaises(ArtifactError):
                handle.bind_origin("other-run").read_candidate_raw(saved["artifact_ref"], **self.identity)

    def test_same_candidate_commit_is_idempotent_and_new_attempt_needs_new_identity(self):
        artifact_id = uuid4().hex
        first = self.handle.put_candidate(self.raw, tool_name="read_text_file", artifact_id=artifact_id,
                                          **self.identity)
        repeated = self.handle.put_candidate(self.raw, tool_name="read_text_file", artifact_id=artifact_id,
                                             **self.identity)
        self.assertEqual(first, repeated)
        self.assertEqual(self.handle.registered_refs(), [])
        with self.assertRaises(ArtifactError):
            self.handle.put_candidate("different", tool_name="read_text_file", artifact_id=artifact_id,
                                      **self.identity)
        self.assertEqual((self.paths.artifacts / (artifact_id + ".raw")).read_text(encoding="utf-8"), self.raw)

    def test_missing_or_tampered_candidate_never_becomes_unavailable_preview(self):
        saved = self.save()
        ref = saved["artifact_ref"]
        manager = ObservationManager(self.handle)
        manager.observe_saved(ref, **self.identity)
        body = self.paths.directory / self.archive.artifact(ref)["path"]
        body.write_text("corrupt", encoding="utf-8")
        for action in (lambda: manager.observe_saved(ref, **self.identity),
                       lambda: self.handle.promote_candidate(ref, **self.identity)):
            with self.assertRaises(ArtifactError):
                action()
        body.unlink()
        with self.assertRaises(ArtifactError):
            self.handle.read_candidate_raw(ref, **self.identity)

    def test_release_revokes_candidate_write_and_promotion(self):
        saved = self.save()
        self.handle.release()
        with self.assertRaises(ArtifactError):
            self.handle.promote_candidate(saved["artifact_ref"], **self.identity)
        with self.assertRaises(ArtifactError):
            self.save()
        self.assertEqual(self.handle.read_candidate_raw(saved["artifact_ref"], **self.identity), self.raw)

    def test_unencodable_candidate_is_saved_exactly_but_never_disclosed(self):
        from copy import deepcopy
        from lineage_agent.agent.record_codec import decode_record
        raw = "private\ud800"
        saved = self.handle.put_candidate(raw, tool_name="custom", artifact_id=uuid4().hex, **self.identity)
        registered = self.archive.artifact(saved["artifact_ref"])
        body = (self.paths.directory / registered["path"]).read_bytes()
        self.assertEqual(decode_record(body), {"raw_result": raw})
        self.assertEqual(saved["raw_bytes"], len(body))
        self.assertEqual(saved["raw_encoding"], "record_codec_v1")
        self.assertEqual(self.handle.read_candidate_raw(saved["artifact_ref"], **self.identity), raw)
        output = ObservationManager(self.handle).observe_saved(saved["artifact_ref"], **self.identity)
        self.assertEqual(output["observation"]["stage"], "serialize")
        self.assertNotIn("artifact_ref", output["observation"])
        self.assertEqual(self.handle.registered_refs(), [])
        self.assertNotIn("private", json.dumps(output))
        for changes in ({"raw_encoding": "unsupported"}, {"origin": {"purpose": "ordinary"}}):
            corrupt = deepcopy(registered)
            corrupt["metadata"].update(changes)
            with self.assertRaises(ArchiveError):
                self.archive.save_artifact(corrupt)
        with self.assertRaises(ArtifactError) as ordinary:
            self.handle.put(raw, tool_name="custom")
        self.assertEqual(ordinary.exception.code, "invalid_artifact_result")

    async def test_result_persistence_error_survives_diagnostic_write_failure(self):
        from lineage_agent.recording.recorder import RunRecorder
        from lineage_agent.tools.execution import execute_calls
        recorder = RunRecorder(self.paths.directory / "logs")
        self.addCleanup(recorder.close)
        failure = ArtifactError("artifact_write_error")
        failure.execution_fact = {"reason_code": "artifact_write_error", "outcome": "succeeded", "stage": "observation"}
        calls = []

        async def fail_after_execution(name, arguments):
            calls.append(name)
            # tool.started is durable, but even recording the completed
            # delegate outcome now fails. The primary storage error wins.
            recorder._stream.close()
            raise failure

        requests = [{"type": "function_call", "call_id": identity, "name": "custom", "arguments": "{}"}
                    for identity in ("first", "pending")]
        with self.assertRaises(ArtifactError) as caught:
            await execute_calls(requests, [{"type": "function", "name": "custom"}],
                                fail_after_execution, observation_manager=ObservationManager(self.handle),
                                recorder=recorder, recoverable=True)
        self.assertIs(caught.exception, failure)
        self.assertEqual(calls, ["custom"])
        self.assertTrue(recorder.failed)
        events = [json.loads(line) for line in recorder.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([item["event"] for item in events], ["tool.started"])


if __name__ == "__main__":
    unittest.main()
