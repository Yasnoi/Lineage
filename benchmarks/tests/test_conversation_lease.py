"""Existing conversation paths and operating-system writer exclusion."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from lineage_agent.agent.record_journal import JournalError, RecordJournal
from lineage_agent.agent.records import RecordStore
from lineage_agent.archive import ArchiveStore
from lineage_agent.storage import ConversationLease, StorageError, conversation_paths


class ConversationLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="conversation-lease-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_existing_paths_and_missing_lock_never_create_storage(self):
        with self.assertRaises(StorageError):
            conversation_paths(self.root, "missing", create=False)
        self.assertEqual(list(self.root.iterdir()), [])
        paths = conversation_paths(self.root, "conversation")
        self.assertEqual(conversation_paths(self.root, "conversation", create=False), paths)
        with self.assertRaises(StorageError) as error:
            ConversationLease.acquire(paths.directory)
        self.assertEqual(error.exception.code, "storage_lock_error")
        self.assertFalse(paths.lock.exists())
        self.assertFalse(paths.database.exists())
        paths.history.rmdir()
        with self.assertRaises(JournalError):
            RecordJournal(paths.history, create=False)
        self.assertFalse(paths.history.exists())

    def test_lock_excludes_same_process_and_retains_file_after_release(self):
        paths = conversation_paths(self.root, "conversation")
        lease = ConversationLease.acquire(paths.directory, create=True)
        self.addCleanup(lease.close)
        self.assertTrue(lease.active)
        with self.assertRaises(StorageError) as error:
            ConversationLease.acquire(paths.directory)
        self.assertEqual(error.exception.code, "conversation_busy")
        lease.release()
        lease.close()
        self.assertFalse(lease.active)
        self.assertEqual(paths.lock.read_bytes(), b"\0")
        with ConversationLease.acquire(paths.directory) as next_lease:
            self.assertTrue(next_lease.active)
        self.assertFalse(next_lease.active)

    def test_archive_reader_never_recreates_missing_history_directory(self):
        paths = conversation_paths(self.root, "conversation")
        with ArchiveStore(paths.directory) as archive:
            view = archive.prepare_view(record_files={}, history_scopes={}, summaries=[], artifacts=[])
            archive.confirm_view(view, "confirmed")
            paths.history.rmdir()
            with self.assertRaises(JournalError):
                RecordStore(archive=archive, view_id=view)
            self.assertFalse(paths.history.exists())

    def test_another_process_is_excluded_and_process_exit_releases_lock(self):
        paths = conversation_paths(self.root, "conversation")
        code = (
            "from pathlib import Path; import os,sys; "
            "from lineage_agent.storage import ConversationLease, StorageError; "
            "\ntry: lease=ConversationLease.acquire(Path(sys.argv[1]))"
            "\nexcept StorageError as exc: print(exc.code); sys.exit(7)"
            "\nos._exit(0)"
        )
        with ConversationLease.acquire(paths.directory, create=True):
            denied = subprocess.run([sys.executable, "-c", code, str(paths.directory)],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(denied.returncode, 7, denied.stderr)
            self.assertEqual(denied.stdout.strip(), "conversation_busy")
        exited = subprocess.run([sys.executable, "-c", code, str(paths.directory)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(exited.returncode, 0, exited.stderr)
        with ConversationLease.acquire(paths.directory):
            pass


if __name__ == "__main__":
    unittest.main()
