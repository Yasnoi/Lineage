"""Released application handles retain reads without regaining write authority."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lineage_agent.archive import ArchiveStore
from lineage_agent.artifacts.store import ArtifactError, ArtifactStore
from lineage_agent.storage import ConversationLease, conversation_paths


class ArtifactGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="artifact-generations-")
        self.addCleanup(self.temp.cleanup)
        self.paths = conversation_paths(Path(self.temp.name), "conversation")
        self.archive = ArchiveStore(self.paths.directory)
        self.addCleanup(self.archive.close)
        self.store = ArtifactStore()
        self.addCleanup(self.store.close)

    def view(self, refs):
        view_id = self.archive.prepare_view(record_files={}, history_scopes={},
                                           summaries=[], artifacts=refs)
        self.archive.confirm_view(view_id, "checkpoint-" + view_id)
        return view_id

    def test_reopening_has_new_generation_and_only_imports_target_view(self):
        old = self.store.bind_session("session", directory=self.paths.artifacts)
        run_writer = old.bind_origin("old-run")
        retained = run_writer.put("published", tool_name="custom")["artifact_ref"]
        view_id = self.view([retained])
        interrupted = old.put("interrupted", tool_name="custom")["artifact_ref"]
        old.release()
        current = self.store.restore_session("session", archive=self.archive, view_id=view_id)
        self.assertEqual(current.registered_refs(), [retained])
        with self.assertRaises(ArtifactError) as error:
            current.read(interrupted, 0)
        self.assertEqual(error.exception.code, "artifact_not_found")
        added = current.put("new generation", tool_name="custom")["artifact_ref"]
        for handle in (old, run_writer, run_writer.bind_origin("new-origin")):
            self.assertEqual(handle.read(retained, 0)["text"], "published")
            self.assertEqual(handle.read(interrupted, 0)["text"], "interrupted")
            with self.assertRaises(ArtifactError) as error:
                handle.put("must remain revoked", tool_name="custom")
            self.assertEqual(error.exception.code, "artifact_write_error")
            with self.assertRaises(ArtifactError):
                handle.read(added, 0)
        # Closing the session's archive does not close the binding's connection.
        current.release()
        self.archive.close()
        self.assertEqual(current.read(retained, 0)["text"], "published")
        self.assertEqual(current.read(added, 0)["text"], "new generation")

    def test_lease_release_revokes_writes_without_closing_borrowed_store(self):
        lease = ConversationLease.acquire(self.paths.directory, create=True)
        self.addCleanup(lease.close)
        writer = self.store.bind_session("one", directory=self.paths.artifacts, lease=lease)
        ref = writer.put("old text", tool_name="custom")["artifact_ref"]
        lease.close()
        self.assertEqual(writer.read(ref, 0)["text"], "old text")
        with self.assertRaises(ArtifactError):
            writer.put("no lease", tool_name="custom")
        writer.release()
        writer.release()
        with ConversationLease.acquire(self.paths.directory) as next_lease:
            fresh = self.store.bind_session("one", directory=self.paths.artifacts, lease=next_lease)
            self.assertEqual(fresh.registered_refs(), [])
            with self.assertRaises(ArtifactError):
                fresh.read(ref, 0)
            fresh.put("new text", tool_name="custom")
            with self.assertRaises(ArtifactError):
                writer.put("old writer never revives", tool_name="custom")

    def test_failed_open_can_discard_only_its_temporary_binding(self):
        original = self.store.bind_session("writer", directory=self.paths.artifacts)
        ref = original.put("existing", tool_name="custom")["artifact_ref"]
        view = self.view([ref])
        temporary = self.store.restore_session("opening", archive=self.archive, view_id=view)
        temporary.discard()
        temporary.discard()
        self.assertEqual(len(self.store._owned_archives), 1)
        self.assertEqual(original.read(ref, 0)["text"], "existing")
        retry = self.store.restore_session("opening", archive=self.archive, view_id=view)
        self.assertEqual(retry.read(ref, 0)["text"], "existing")


if __name__ == "__main__":
    unittest.main()
