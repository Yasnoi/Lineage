"""Independent business archives and immutable, explicitly admitted resources."""

from copy import deepcopy
from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from lineage_agent.archive import ArchiveError, ArchiveStore
from lineage_agent.agent.record_journal import RecordJournal
from lineage_agent.agent.records import make_record_batch
from lineage_agent.storage import conversation_paths


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="archive-test-")
        self.addCleanup(self.temporary.cleanup)
        self.paths = conversation_paths(Path(self.temporary.name), "conversation")
        self.archive = ArchiveStore(self.paths.directory)
        self.addCleanup(self.archive.close)
        self.journal = RecordJournal(self.paths.history)
        self.addCleanup(self.journal.close)

    def original(self, run="run", text="first", start=0):
        batch = make_record_batch(run, start, [{"role": "user", "content": text}])
        _, watermark = self.journal.append(run, batch)
        return watermark

    def scope_record(self, watermark):
        record = {"history_ref": f"history://scope/{uuid4()}", "source_checkpoint_id": "old",
                  "ranges": [{"run_id": "run", "start": 0, "end": 1}], "record_files": {"run": watermark}}
        self.archive.save_scope(record)
        return record

    def summary_record(self, scope, previous=None):
        record = {"summary_id": str(uuid4()), "lane": "H", "body": "summary body", "run_id": "run",
                  "history_ref": scope["history_ref"], "ranges": scope["ranges"], "previous": previous or []}
        self.archive.save_summary(record)
        return record

    def artifact_record(self):
        identity = uuid4().hex
        data = b"raw artifact"
        (self.paths.artifacts / f"{identity}.raw").write_bytes(data)
        ref = f"artifact://tool/{identity}"
        record = {"artifact_ref": ref, "path": f"artifacts/{identity}.raw", "sha256": hashlib.sha256(data).hexdigest(),
                  "metadata": {"artifact_ref": ref, "raw_bytes": len(data), "tool_name": "custom",
                               "view_kind": "raw_result", "total_chars": len(data)}}
        self.archive.save_artifact(record)
        return record

    def view(self, *, files=None, scopes=None, summaries=None, artifacts=None, facts=None, parent=None, confirm=True):
        identity = self.archive.prepare_view(record_files=files or {}, history_scopes=scopes or {},
                                             summaries=summaries or [], artifacts=artifacts or [],
                                             fact_ends=facts or {}, parent_id=parent)
        if confirm:
            self.archive.confirm_view(identity, str(uuid4()))
        return identity

    def test_registration_does_not_authorize_and_rebuild_keeps_identity(self):
        watermark = self.original()
        scope = self.scope_record(watermark)
        summary = self.summary_record(scope)
        artifact = self.artifact_record()
        empty = self.view()
        with self.assertRaises(ArchiveError):
            self.archive.summary(summary["summary_id"], empty)
        with self.assertRaises(ArchiveError):
            self.archive.artifact(artifact["artifact_ref"], empty)
        view = self.view(files={"run": watermark}, scopes={scope["history_ref"]: scope},
                         summaries=[summary["summary_id"]], artifacts=[artifact["artifact_ref"]], confirm=False)
        with self.assertRaises(ArchiveError):
            self.archive.view(view)
        with self.assertRaises(ArchiveError):
            self.archive.summary(summary["summary_id"], view)
        self.assertEqual(self.archive.summary(summary["summary_id"], view, confirmed=False), summary)
        self.archive.confirm_view(view, "saved-state")
        key = self.archive.cursor_key()
        self.archive.close()
        with ArchiveStore(self.paths.directory, create=False) as rebuilt:
            self.assertEqual(rebuilt.cursor_key(), key)
            self.assertEqual(rebuilt.summary(summary["summary_id"], view), summary)
            self.assertEqual(rebuilt.scope(scope["history_ref"], view), scope)
            self.assertEqual(rebuilt.artifact(artifact["artifact_ref"], view), artifact)
            self.assertEqual(rebuilt.view_for_checkpoint("saved-state"), view)
            self.assertEqual(rebuilt.view(view)["record_paths"], {"run": "history/run.jsonl"})

    def test_scope_boundary_is_fixed_after_growth_and_unconfirmed_tail(self):
        first = self.original()
        scope = self.scope_record(first)
        view = self.view(files={"run": first}, scopes={scope["history_ref"]: scope})
        self.original(text="second", start=1)
        with self.journal.path("run").open("ab") as handle:
            handle.write(b"unconfirmed non-record tail")
        self.assertEqual(self.archive.scope(scope["history_ref"], view)["record_files"]["run"], first)
        self.archive.verify_view(view)
        self.assertEqual(self.archive.view(view)["record_files"]["run"]["record_count"], 1)

    def test_history_corruption_cannot_be_masked_by_successful_read(self):
        watermark = self.original()
        scope = self.scope_record(watermark)
        view = self.view(files={"run": watermark}, scopes={scope["history_ref"]: scope})
        self.archive.scope(scope["history_ref"], view)
        path = self.journal.path("run")
        data = path.read_bytes()
        for changed in (data.replace(b"first", b"other"), data[:-1]):
            path.write_bytes(changed)
            with self.assertRaises(ArchiveError):
                self.archive.scope(scope["history_ref"], view)
        path.unlink()
        with self.assertRaises(ArchiveError):
            self.archive.verify_view(view)

    def test_artifact_corruption_and_metadata_mismatch_rejected(self):
        record = self.artifact_record()
        view = self.view(artifacts=[record["artifact_ref"]])
        self.archive.artifact(record["artifact_ref"], view)
        path = self.paths.directory / record["path"]
        path.write_bytes(b"different")
        with self.assertRaises(ArchiveError):
            self.archive.artifact(record["artifact_ref"], view)
        path.write_bytes(b"raw artifact")
        changed = deepcopy(record)
        changed["metadata"]["total_chars"] += 1
        with self.assertRaises(ArchiveError):
            self.archive.save_artifact(changed)

    def test_objects_are_immutable_and_summary_closure_is_required(self):
        watermark = self.original()
        scope = self.scope_record(watermark)
        first = self.summary_record(scope)
        self.archive.save_summary(first)
        altered = {**first, "body": "replacement"}
        with self.assertRaises(ArchiveError):
            self.archive.save_summary(altered)
        second = self.summary_record(scope, [first["summary_id"]])
        with self.assertRaises(ArchiveError):
            self.view(files={"run": watermark}, scopes={scope["history_ref"]: scope}, summaries=[second["summary_id"]])
        view = self.view(files={"run": watermark}, scopes={scope["history_ref"]: scope},
                         summaries=[first["summary_id"], second["summary_id"]])
        self.assertEqual(self.archive.summary(second["summary_id"], view)["previous"], [first["summary_id"]])

    def test_summary_sources_must_belong_to_its_fixed_scope(self):
        watermark = self.original()
        scope = self.scope_record(watermark)
        record = self.summary_record(scope)
        for reference in ("history://record/run/1", "history://record/other/0"):
            with self.assertRaises(ArchiveError):
                self.archive.save_summary({**record, "summary_id": str(uuid4()), "source_refs": [reference]})

    def test_previous_summary_scope_cannot_be_discarded_or_partially_covered(self):
        self.original()
        second = self.original(text="second", start=1)
        original_scope = {"history_ref": f"history://scope/{uuid4()}",
                          "ranges": [{"run_id": "run", "start": 0, "end": 2}],
                          "record_files": {"run": second}}
        self.archive.save_scope(original_scope)
        previous = self.summary_record(original_scope)
        for ranges in ([{"run_id": "run", "start": 0, "end": 1}],
                       [{"run_id": "run", "start": 1, "end": 2}]):
            scope = {**original_scope, "history_ref": f"history://scope/{uuid4()}", "ranges": ranges}
            self.archive.save_scope(scope)
            with self.assertRaises(ArchiveError):
                self.summary_record(scope, [previous["summary_id"]])

    def test_h_summary_can_reuse_r_across_runs_and_adjacent_scope_intervals(self):
        self.original()
        watermark = self.original(text="second", start=1)
        old_scope = {"history_ref": f"history://scope/{uuid4()}",
                     "ranges": [{"run_id": "run", "start": 0, "end": 2}],
                     "record_files": {"run": watermark}}
        self.archive.save_scope(old_scope)
        previous = {"summary_id": str(uuid4()), "lane": "R", "body": "previous run summary", "run_id": "run",
                    "history_ref": old_scope["history_ref"], "ranges": old_scope["ranges"], "previous": []}
        self.archive.save_summary(previous)
        other = self.original(run="other", text="another run")
        scope = {"history_ref": f"history://scope/{uuid4()}",
                 "ranges": [{"run_id": "run", "start": 0, "end": 1},
                            {"run_id": "run", "start": 1, "end": 2},
                            {"run_id": "other", "start": 0, "end": 1}],
                 "record_files": {"run": watermark, "other": other}}
        self.archive.save_scope(scope)
        current = {"summary_id": str(uuid4()), "lane": "H", "body": "combined history", "run_id": "other",
                   "history_ref": scope["history_ref"], "ranges": scope["ranges"], "previous": [previous["summary_id"]]}
        self.archive.save_summary(current)
        view = self.view(files=scope["record_files"], scopes={old_scope["history_ref"]: old_scope, scope["history_ref"]: scope},
                         summaries=[previous["summary_id"], current["summary_id"]])
        self.assertEqual(self.archive.summary(current["summary_id"], view), current)

    def test_scope_rejects_overlapping_or_outside_ranges(self):
        watermark = self.original()
        for ranges in ([{"run_id": "run", "start": 0, "end": 2}],
                       [{"run_id": "run", "start": 0, "end": 1}] * 2):
            with self.assertRaises(ArchiveError):
                self.archive.save_scope({"history_ref": f"history://scope/{uuid4()}", "ranges": ranges,
                                         "record_files": {"run": watermark}})

    def test_parent_view_cannot_regress(self):
        watermark = self.original()
        parent = self.view(files={"run": watermark})
        with self.assertRaises(ArchiveError):
            self.view(parent=parent)
        second = self.original(text="second", start=1)
        child = self.view(files={"run": second}, parent=parent)
        self.assertEqual(self.archive.view(child)["parent_id"], parent)

    def test_record_order_survives_sorted_encoding_and_inheritance(self):
        first = self.original(run="z-first", text="earliest")
        second = self.original(run="a-second", text="later")
        files = {"z-first": first, "a-second": second}
        parent = self.view(files=files)
        saved = self.archive.view(parent)
        self.assertEqual(list(saved["record_files"]), ["a-second", "z-first"])
        self.assertEqual(saved["record_order"], ["z-first", "a-second"])
        third = self.original(run="m-third", text="latest")
        with self.assertRaises(ArchiveError):
            self.view(files={"a-second": second, "z-first": first, "m-third": third}, parent=parent)
        child = self.view(files={**files, "m-third": third}, parent=parent)
        self.archive.close()
        with ArchiveStore(self.paths.directory, create=False) as rebuilt:
            self.assertEqual(rebuilt.view(child)["record_order"], ["z-first", "a-second", "m-third"])
            rebuilt.verify_view(child)

    def test_view_record_order_must_match_unique_registered_runs(self):
        from lineage_agent.agent.record_codec import decode_record, encode_record
        watermark = self.original()
        view = self.view(files={"run": watermark})
        for order in ([], ["run", "run"], ["another"], {"run": 0}):
            with closing(sqlite3.connect(self.paths.database)) as connection, connection:
                data = connection.execute("SELECT payload FROM views WHERE id=?", (view,)).fetchone()[0]
                payload = decode_record(data)
                payload["record_order"] = order
                changed = encode_record(payload)
                connection.execute("UPDATE views SET payload=?,sha256=? WHERE id=?",
                                   (changed, hashlib.sha256(changed).hexdigest(), view))
            with self.assertRaises(ArchiveError):
                self.archive.verify_view(view)

    def test_confirmation_failure_does_not_publish_view_or_fact(self):
        self.archive.append_event("run", "model", {"usage": None, "checkpoint_id": None})
        view = self.view(facts={"run": 1}, confirm=False)
        with patch.object(self.archive, "_verify_view", side_effect=ArchiveError("archive_write_error")):
            with self.assertRaises(ArchiveError):
                self.archive.confirm_view(view, "broken")
        with self.assertRaises(ArchiveError):
            self.archive.view(view)
        self.assertIsNone(self.archive.events("run")[0]["payload"]["checkpoint_id"])
        self.archive.confirm_view(view, "first-confirmed")
        self.archive.confirm_view(view, "later-confirmed")
        self.assertEqual(self.archive.events("run")[0]["payload"]["checkpoint_id"], "first-confirmed")

    def test_database_confirmation_rolls_back_after_partial_updates(self):
        self.archive.append_event("run", "model", {"checkpoint_id": None})
        view = self.view(facts={"run": 1}, confirm=False)
        with closing(sqlite3.connect(self.paths.database)) as connection, connection:
            connection.execute("CREATE TRIGGER fail_fact_confirmation BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with self.assertRaises(ArchiveError) as caught:
            self.archive.confirm_view(view, "not-published")
        self.assertEqual(caught.exception.code, "archive_write_error")
        with self.assertRaises(ArchiveError):
            self.archive.view(view)
        with self.assertRaises(ArchiveError):
            self.archive.view_for_checkpoint("not-published")
        self.assertIsNone(self.archive.events("run")[0]["payload"]["checkpoint_id"])
        with closing(sqlite3.connect(self.paths.database)) as connection, connection:
            connection.execute("DROP TRIGGER fail_fact_confirmation")
        self.archive.confirm_view(view, "published")

    def test_events_preserve_opaque_values_unknown_usage_and_failed_attempts(self):
        first = {"usage": None, "result": {"value": b"bytes", "tuple": (1, 2)}, "checkpoint_id": None}
        self.assertEqual(self.archive.append_event("run", "model", first, key="model-1"), 1)
        self.assertEqual(self.archive.append_event("run", "model", first, key="model-1"), 1)
        self.archive.append_event("run", "summary", {"attempt": 1, "status": "failed", "usage": None})
        self.archive.append_event("run", "summary", {"attempt": 2, "status": "not_adopted", "usage": {"total": 8}})
        self.assertEqual(self.archive.event_end("run"), 3)
        events = self.archive.events("run", end=2)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["payload"], first)
        with self.assertRaises(ArchiveError):
            self.archive.append_event("run", "model", {"usage": 0}, key="model-1")
        with self.assertRaises(ArchiveError):
            self.archive.events("run", end=4)

    def test_only_sealed_run_details_can_be_read_after_rebuild(self):
        self.archive.append_event("run", "model", {"usage": None})
        view = self.view(facts={"run": 1}, confirm=False)
        payload = {"run_id": "run", "fact_end": 1, "final_state": {"stop_reason": "completed", "usage_totals": None}}
        record = self.archive.stage_run("run", payload)
        with self.assertRaises(ArchiveError):
            self.archive.read_run("run")
        with self.assertRaises(ArchiveError):
            self.archive.seal_run(record, view, "final")
        self.archive.confirm_view(view, "final")
        self.archive.seal_run(record, view, "final")
        self.archive.seal_run(record, view, "final")
        # Checkpoint diagnostics are not a prerequisite for business retrieval.
        with closing(sqlite3.connect(self.paths.database)) as connection, connection:
            connection.execute("DELETE FROM checkpoints")
        self.archive.close()
        with ArchiveStore(self.paths.directory, create=False) as rebuilt:
            self.assertEqual(rebuilt.read_run("run"), {**payload, "view_id": view, "checkpoint_id": "final"})
        self.assertTrue(self.paths.database.exists())

    def test_missing_database_does_not_import_files(self):
        paths = conversation_paths(Path(self.temporary.name), "unregistered")
        (paths.artifacts / "unknown.raw").write_text("orphan", encoding="utf-8")
        with self.assertRaises(ArchiveError):
            ArchiveStore(paths.directory, create=False)
        self.assertFalse(paths.database.exists())
        with ArchiveStore(paths.directory) as archive:
            empty = archive.prepare_view(record_files={}, history_scopes={}, summaries=[], artifacts=[])
            archive.confirm_view(empty, "initial")
            self.assertEqual(archive.view(empty)["artifacts"], [])

    def test_foreign_views_and_object_registrations_are_not_authorized(self):
        artifact = self.artifact_record()
        view = self.view(artifacts=[artifact["artifact_ref"]])
        other_paths = conversation_paths(Path(self.temporary.name), "other")
        with ArchiveStore(other_paths.directory) as other:
            with self.assertRaises(ArchiveError):
                other.view(view)
            with self.assertRaises(ArchiveError):
                other.artifact(artifact["artifact_ref"])

    def test_database_sidecar_links_are_rejected_on_each_operation(self):
        sidecar = self.paths.database.with_name("conversation.sqlite-wal")
        native = __import__("lineage_agent.archive.store", fromlist=["_check_components"])._check_components

        def check(path):
            if path == sidecar:
                raise OSError("simulated reparse point")
            return native(path)

        sidecar.write_bytes(b"unsafe sidecar")
        with patch("lineage_agent.archive.store._check_components", side_effect=check):
            with self.assertRaises(ArchiveError):
                self.archive.cursor_key()
        sidecar.unlink()

    def test_unknown_format_and_copied_conversation_are_rejected(self):
        self.archive.close()
        with closing(sqlite3.connect(self.paths.database)) as connection, connection:
            connection.execute("PRAGMA user_version=99")
        with self.assertRaises(ArchiveError) as caught:
            ArchiveStore(self.paths.directory, create=False)
        self.assertEqual(caught.exception.code, "archive_version_error")
        with closing(sqlite3.connect(self.paths.database)) as connection, connection:
            connection.execute("PRAGMA user_version=1")
        other = conversation_paths(Path(self.temporary.name), "other")
        other.database.write_bytes(self.paths.database.read_bytes())
        with self.assertRaises(ArchiveError):
            ArchiveStore(other.directory, create=False)

    def test_corrupted_database_payload_and_unsafe_paths_rejected(self):
        view = self.view()
        with closing(sqlite3.connect(self.paths.database)) as connection, connection:
            connection.execute("UPDATE views SET payload=? WHERE id=?", (b"changed", view))
        with self.assertRaises(ArchiveError):
            self.archive.view(view)
        for relative in ("../outside.raw", "/outside.raw", "C:/outside.raw", "artifacts\\outside.raw"):
            record = self.artifact_record()
            record["path"] = relative
            with self.assertRaises(ArchiveError):
                self.archive.save_artifact(record)

    def test_closed_store_refuses_access_without_removing_files(self):
        self.archive.close()
        with self.assertRaises(ArchiveError):
            self.archive.cursor_key()
        self.assertTrue(self.paths.database.exists())


if __name__ == "__main__":
    unittest.main()
