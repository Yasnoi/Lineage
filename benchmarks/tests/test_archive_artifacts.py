"""Artifact registrations survive handle replacement without granting by scans."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from lineage_agent.archive import ArchiveStore
from lineage_agent.artifacts.store import ArtifactError, ArtifactStore
from lineage_agent.artifacts.tools import ArtifactExecutor
from lineage_agent.observations.manager import ObservationManager
from lineage_agent.storage import conversation_paths
from lineage_agent.tools.types import ToolExecutionError


class ArchiveArtifactTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="archive-artifact-test-")
        self.addCleanup(self.temporary.cleanup)
        self.paths = conversation_paths(Path(self.temporary.name), "conversation")
        self.archive = ArchiveStore(self.paths.directory, create=True)
        self.addCleanup(self.archive.close)
        self.store = ArtifactStore()
        self.addCleanup(self.store.close)
        self.writer = self.store.bind_session("session", directory=self.paths.artifacts)

    def view(self, refs, *, confirm=True):
        view_id = self.archive.prepare_view(
            record_files={}, history_scopes={}, summaries=[], artifacts=refs,
        )
        if confirm:
            self.archive.confirm_view(view_id, "checkpoint-" + view_id)
        return view_id

    def test_rebuild_keeps_original_ref_and_only_approved_records(self):
        original = self.writer.bind_origin("run-one").put(
            json.dumps({"content": "left KEY right"}), tool_name="read_text_file")
        ref = original["artifact_ref"]
        excluded = self.writer.put("not in this view", tool_name="custom")["artifact_ref"]
        view_id = self.view([ref])
        self.assertEqual(self.archive.artifact(ref)["metadata"]["origin"], {"run_id": "run-one"})
        self.store.close()
        with ArtifactStore() as rebuilt:
            handle = rebuilt.restore_session("new-process-handle", archive=self.archive, view_id=view_id)
            self.assertEqual(handle.registered_refs(), [ref])
            self.assertEqual(handle.read(ref, 5, 8)["text"], "KEY")
            self.assertEqual(handle.search(ref, ["KEY"])["hits"][0]["matches"][0]["start"], 5)
            with self.assertRaises(ArtifactError) as error:
                handle.read(excluded, 0)
            self.assertEqual(error.exception.code, "artifact_not_found")
        # The external archive is borrowed, and close never deletes the body.
        self.assertEqual(self.archive.artifact(ref)["metadata"]["view_kind"], "content")
        self.assertEqual(len(list(self.paths.artifacts.glob("*.raw"))), 2)
        with ArtifactStore() as untrusted:
            handle = untrusted.bind_session("ordinary", directory=self.paths.artifacts)
            self.assertEqual(handle.registered_refs(), [])
            with self.assertRaises(ArtifactError):
                handle.read(ref, 0)

    async def test_tool_access_waits_for_confirmed_view_but_preview_is_immediate(self):
        current = self.view([])
        executor = ArtifactExecutor(self.writer, view_provider=lambda: current).bind_run()
        ref = self.writer.put("new material", tool_name="custom")["artifact_ref"]
        arguments = {"artifact_ref": ref, "start": 0, "end": None}
        self.assertEqual(self.writer.read(ref, 0)["text"], "new material")
        with self.assertRaises(ToolExecutionError) as error:
            await executor("artifact_read", arguments)
        self.assertEqual(error.exception.reason_code, "artifact_not_found")
        candidate = self.view([ref], confirm=False)
        current = candidate
        with self.assertRaises(ToolExecutionError):
            await executor("artifact_read", arguments)
        self.archive.confirm_view(candidate, "checkpoint-" + candidate)
        self.assertEqual(json.loads(await executor("artifact_read", arguments))["text"], "new material")
        current = None
        with self.assertRaises(ToolExecutionError):
            await executor("artifact_read", arguments)

    def test_database_failure_keeps_raw_without_publishing_handle(self):
        retained = self.writer.put("previously confirmed", tool_name="custom")["artifact_ref"]
        confirmed = self.view([retained])
        with patch.object(ArchiveStore, "save_artifact", side_effect=OSError("private disk failure")):
            with self.assertRaises(ArtifactError) as error:
                self.writer.put("body survived metadata failure", tool_name="custom")
        self.assertEqual(error.exception.code, "artifact_write_error")
        self.assertNotIn("private", str(error.exception))
        self.assertEqual(self.writer.registered_refs(), [retained])
        self.assertEqual(self.writer.persistence_error(), "artifact_write_error")
        with self.assertRaises(ArtifactError):
            self.writer.check_persistence()
        with self.assertRaises(ArtifactError):
            self.writer.put("cannot retry silently", tool_name="custom")
        self.assertEqual(self.writer.read(retained, 0, view_id=confirmed)["text"], "previously confirmed")
        self.assertEqual(self.archive.view(confirmed)["artifacts"], [retained])
        files = list(self.paths.artifacts.glob("*.raw"))
        self.assertEqual(len(files), 2)
        self.assertIn("body survived metadata failure", [path.read_text(encoding="utf-8") for path in files])
        self.assertEqual(list(self.paths.artifacts.glob("*.tmp")), [])

    def test_real_body_failure_survives_observation_fallback_and_blocks_new_writes(self):
        manager = ObservationManager(self.writer)
        with patch("lineage_agent.artifacts.store.os.fsync", side_effect=OSError("private disk failure")):
            result = manager.observe("large body" * 5000, tool_name="custom")
        self.assertEqual(result["observation"]["stage"], "save")
        self.assertNotIn("artifact_ref", result["observation"])
        self.assertEqual(self.writer.persistence_error(), "artifact_write_error")
        self.assertEqual(self.writer.registered_refs(), [])
        with self.assertRaises(ArtifactError):
            self.writer.bind_origin("next-run").put("still blocked", tool_name="custom")
        self.assertEqual(list(self.paths.artifacts.iterdir()), [])

    def test_unencodable_value_and_preview_failure_do_not_block_persistence(self):
        with self.assertRaises(ArtifactError):
            self.writer.put("bad\ud800", tool_name="custom")
        self.assertIsNone(self.writer.persistence_error())
        manager = ObservationManager(self.writer)
        with patch.object(type(self.writer), "search", side_effect=OSError("preview failure")):
            result = manager.observe("large KEY" * 5000, tool_name="custom", keywords=("KEY",))
        self.assertEqual(result["observation"]["stage"], "preview")
        self.assertIsNone(self.writer.persistence_error())
        ref = self.writer.put("subsequent write", tool_name="custom")["artifact_ref"]
        self.assertEqual(self.writer.read(ref, 0)["text"], "subsequent write")

    def test_rebuilt_handle_detects_body_changes_after_successful_read(self):
        ref = self.writer.put("original body", tool_name="custom")["artifact_ref"]
        view_id = self.view([ref])
        with ArtifactStore() as rebuilt:
            handle = rebuilt.restore_session("rebuilt", archive=self.archive, view_id=view_id)
            self.assertEqual(handle.read(ref, 0)["text"], "original body")
            path = self.paths.directory / self.archive.artifact(ref)["path"]
            path.write_bytes(b"tampered")
            for action in (lambda: handle.read(ref, 0), lambda: handle.search(ref, ["body"])):
                with self.assertRaises(ArtifactError) as error:
                    action()
                self.assertEqual(error.exception.code, "artifact_read_error")
            path.unlink()
            with self.assertRaises(ArtifactError):
                handle.read(ref, 0)

    def test_wrong_conversation_and_unconfirmed_view_cannot_rebuild(self):
        ref = self.writer.put("private", tool_name="custom")["artifact_ref"]
        pending = self.view([ref], confirm=False)
        with ArtifactStore() as rebuilt:
            with self.assertRaises(ArtifactError):
                rebuilt.restore_session("pending", archive=self.archive, view_id=pending)
        confirmed = self.view([ref])
        other_paths = conversation_paths(Path(self.temporary.name), "other")
        with ArchiveStore(other_paths.directory, create=True) as other, ArtifactStore() as rebuilt:
            with self.assertRaises(ArtifactError):
                rebuilt.restore_session("foreign", archive=other, view_id=confirmed)


if __name__ == "__main__":
    unittest.main()
