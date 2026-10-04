"""Conversation paths, persistent Artifact handles and reserved filesystem paths."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from lineage_agent.artifacts.store import ArtifactError, ArtifactStore
from lineage_agent.config.settings import ConfigurationError
from lineage_agent.storage import StorageError, conversation_paths
from lineage_agent.tools.filesystem.directory import create_directory_executor
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.paths import bind_project_root
from lineage_agent.tools.filesystem.text import create_text_executor
from lineage_agent.tools.types import ToolArgumentError


class PersistentStorageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="persistent-storage-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_explicit_bindings_are_isolated_and_survive_close(self):
        first = conversation_paths(self.root, "first")
        second = conversation_paths(self.root, "second")
        with ArtifactStore() as store:
            one = store.bind_session("one", directory=first.artifacts)
            two = store.bind_session("two", directory=second.artifacts)
            artifact = one.put("persistent exact result", tool_name="custom")
            ref = artifact["artifact_ref"]
            self.assertEqual(store.bind_session("one", directory=first.artifacts).read(ref, 0)["text"], "persistent exact result")
            self.assertEqual(store.bind_session("one").read(ref, 0)["text"], "persistent exact result")
            with self.assertRaises(ArtifactError):
                store.bind_session("one", directory=second.artifacts)
            with self.assertRaises(ArtifactError):
                two.read(ref, 0)
            paths = list(first.artifacts.glob("*.raw"))
            self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].read_text(encoding="utf-8"), "persistent exact result")
        with self.assertRaises(ArtifactError) as error:
            one.read(ref, 0)
        self.assertEqual(error.exception.code, "artifact_store_closed")
        with ArtifactStore() as other:
            view = other.bind_session("one", directory=first.artifacts)
            with self.assertRaises(ArtifactError):
                view.read(ref, 0)

    def test_default_binding_keeps_arbitrary_identity_out_of_paths(self):
        with patch("lineage_agent.artifacts.store.Path.cwd", return_value=self.root), ArtifactStore() as store:
            one = store.bind_session("../arbitrary/session")
            two = store.bind_session("second")
            first = one.put("first", tool_name="custom")
            two.put("second", tool_name="custom")
            self.assertEqual(len(store._directories), 2)
            for directory in store._directories.values():
                self.assertEqual(directory.parent.parent, self.root / ".lineage")
                self.assertEqual(directory.name, "artifacts")
            self.assertEqual(one.read(first["artifact_ref"], 0)["text"], "first")
            with self.assertRaises(ArtifactError):
                store.bind_session("bad", directory=self.root / "visible-artifacts")

    async def test_all_builtin_file_tools_reject_lineage_and_listing_omits_it(self):
        # Existing executors must enforce the boundary before storage exists.
        executors = [create_text_executor(self.root), create_filesystem_executor(self.root)]
        directory = create_directory_executor(self.root, max_entries=1)
        location = conversation_paths(self.root, "session")
        source = location.history / "run.jsonl"
        source.write_text("private record", encoding="utf-8")
        (self.root / ".hidden").write_text("ordinary hidden file", encoding="utf-8")
        (self.root / "visible.txt").write_text("ordinary file", encoding="utf-8")
        for executor in executors:
            for path in (".lineage/session/history/run.jsonl", ".LINEAGE/session/history/run.jsonl"):
                with self.assertRaises(ToolArgumentError) as error:
                    await executor("read_text_file", {"path": path})
                self.assertEqual(error.exception.reason_code, "storage_path_forbidden")
        with self.assertRaises(ToolArgumentError):
            await directory("list_directory", {"path": ".lineage", "after": None})
        with self.assertRaises(ConfigurationError):
            bind_project_root(location.history)
        names, cursor = [], None
        while True:
            page = json.loads(await directory("list_directory", {"path": ".", "after": cursor}))
            names.extend(entry["name"] for entry in page["entries"])
            if not page["truncated"]:
                break
            cursor = page["next_after"]
        self.assertEqual(names, [".hidden", "visible.txt"])

    def test_storage_rejects_unsafe_identifiers(self):
        for identifier in ("../escape", "CON", "", "a/b", "a\\b"):
            with self.assertRaises(StorageError):
                conversation_paths(self.root, identifier)

    def test_shared_parent_creation_race_keeps_valid_directory(self):
        native = Path.mkdir
        shared = self.root / ".lineage"
        raced = []
        def mkdir(path, *args, **kwargs):
            if path == shared and not raced:
                raced.append(path)
                native(path)
            return native(path, *args, **kwargs)
        with patch.object(Path, "mkdir", new=mkdir):
            result = conversation_paths(self.root, "concurrent-session")
        self.assertEqual(raced, [shared])
        self.assertTrue(result.history.is_dir())
        self.assertTrue(result.artifacts.is_dir())


if __name__ == "__main__":
    unittest.main()
