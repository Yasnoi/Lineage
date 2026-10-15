"""Atomic recovery points, unfinished attempts and publication ambiguity."""

from contextlib import closing
from copy import deepcopy
import hashlib
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from lineage_agent.archive import ArchiveError, ArchiveStore, FORMAT_VERSION, RECOVERY_VERSION
from lineage_agent.agent.record_codec import decode_record, encode_record
from lineage_agent.agent.record_journal import RecordJournal
from lineage_agent.agent.records import make_record_batch
from lineage_agent.storage import conversation_paths


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="snapshot-test-")
        self.addCleanup(self.temporary.cleanup)
        self.paths = conversation_paths(Path(self.temporary.name), "conversation")
        self.archive = ArchiveStore(self.paths.directory)
        self.addCleanup(self.archive.close)
        self.journal = RecordJournal(self.paths.history)
        self.addCleanup(self.journal.close)
        self.configuration = {"settings": {"model": "controlled-model"}, "binding": {"kind": "test"}}
        self.archive.register_conversation(self.configuration, [])

    def candidate(self, run="run", *, parent=None, stop="completed", tail=None):
        self.archive.begin_run(run, parent)
        previous = self.archive.read_snapshot(parent) if parent else None
        files = deepcopy(previous["final_state"]["record_files"]) if previous else {}
        facts = self.archive.view(previous["view_id"])["fact_ends"] if previous else {}
        _, files[run] = self.journal.append(run, make_record_batch(run, 0, [{"role": "user", "content": run}]))
        facts[run] = self.archive.append_event(run, "model", {"usage": None, "checkpoint_id": None})
        view = self.archive.prepare_view(record_files=files, history_scopes={}, summaries=[], artifacts=[],
                                         fact_ends=facts, parent_id=previous["view_id"] if previous else None)
        checkpoint = f"final-{run}"
        self.archive.confirm_view(view, checkpoint)
        items = [{"role": "user", "content": run}]
        state = {"run_id": run, "conversation_id": self.archive.conversation_id,
                 "resource_view_id": view, "record_files": files, "history_scopes": {},
                 "stop_reason": stop, "input": items + (tail or []), "history_end": 1,
                 "active_context": {"turn_start": 0}, "context_risks": [None] * (1 + len(tail or [])),
                 "usage_totals": {"total_tokens": None}}
        record = self.archive.stage_run(run, {"final_state": state, "fact_end": facts[run]})
        return record, view, checkpoint, parent

    def publish(self, run="run", **kwargs):
        return self.archive.publish_snapshot(*self.candidate(run, **kwargs))

    def db(self, sql, args=()):
        with closing(sqlite3.connect(self.paths.database)) as connection, connection:
            return connection.execute(sql, args).fetchall()

    def test_registration_and_no_snapshot_are_explicit(self):
        original = self.archive.conversation()
        self.assertEqual(self.archive.register_conversation(self.configuration, []), original)
        self.assertEqual((original["format_version"], original["recovery_version"]), (2, 1))
        self.assertEqual((FORMAT_VERSION, RECOVERY_VERSION), (2, 1))
        self.assertIsNone(original["current_snapshot_id"])
        with self.assertRaises(ArchiveError) as error:
            self.archive.read_snapshot()
        self.assertEqual(error.exception.code, "archive_no_snapshot")
        with self.assertRaises(ArchiveError):
            self.archive.register_conversation({"settings": {}}, [])
        metadata = self.archive.metadata()
        self.assertIsNone(metadata["head"])
        self.assertEqual(metadata["unfinished_runs"], [])

    def test_low_level_unregistered_archive_remains_available(self):
        paths = conversation_paths(Path(self.temporary.name), "lowlevel")
        with ArchiveStore(paths.directory) as archive:
            view = archive.prepare_view(record_files={}, history_scopes={}, summaries=[], artifacts=[])
            archive.confirm_view(view, "cp")
            archive.verify_view(view)
            with self.assertRaises(ArchiveError) as error:
                archive.metadata()
            self.assertEqual(error.exception.code, "archive_unregistered")

    def test_run_lifecycle_precedes_files_and_preexecution_rejection(self):
        run = self.archive.begin_run("rejected", None)
        self.assertEqual(self.archive.begin_run("rejected", None), run)
        self.assertFalse(self.journal.path("rejected").exists())
        self.assertEqual([row["run_id"] for row in self.archive.unfinished_runs()], ["rejected"])
        with self.assertRaises(ArchiveError):
            self.archive.begin_run("second", None)
        self.archive.reject_run("rejected")
        self.archive.reject_run("rejected")
        self.assertEqual(self.archive.unfinished_runs(), [])
        self.archive.begin_run("paid", None)
        self.archive.append_event("paid", "model", {"usage": None})
        with self.assertRaises(ArchiveError):
            self.archive.reject_run("paid")
        with self.assertRaises(ArchiveError) as error:
            self.archive.accept_recovery("missing", ["paid"])
        self.assertEqual(error.exception.code, "archive_no_snapshot")

    def test_snapshots_are_immutable_ordered_and_idempotent_after_later_publish(self):
        args = self.candidate("z-first")
        first = self.archive.publish_snapshot(*args)
        second = self.publish("a-second", parent=first["snapshot_id"])
        self.assertEqual(second["sequence"], 2)
        self.assertEqual(second["parent_id"], first["snapshot_id"])
        self.assertEqual(list(second["final_state"]["record_files"]), ["z-first", "a-second"])
        self.assertEqual(self.archive.publish_snapshot(*args), first)
        self.assertEqual(self.archive.read_snapshot()["snapshot_id"], second["snapshot_id"])
        self.assertEqual([row["run_id"] for row in self.archive.published_runs(second["snapshot_id"])], ["z-first", "a-second"])
        self.assertEqual(self.archive.unfinished_runs(), [])
        self.assertIsNone(second["final_state"]["usage_totals"]["total_tokens"])
        self.archive.close()
        with ArchiveStore(self.paths.directory, create=False) as rebuilt:
            self.assertEqual(rebuilt.read_snapshot()["final_state"], second["final_state"])
            self.assertEqual(rebuilt.read_snapshot(first["snapshot_id"]), first)

    def test_state_and_events_do_not_require_historical_checkpoint_mapping(self):
        saved = self.publish()
        self.db("DELETE FROM checkpoints")
        self.assertEqual(self.archive.read_snapshot()["snapshot_id"], saved["snapshot_id"])
        self.assertEqual(self.archive.published_runs(saved["snapshot_id"])[0]["checkpoint_id"], "final-run")

    def test_explicit_head_must_be_the_complete_published_chain_end(self):
        first = self.publish("first")
        second = self.publish("second", parent=first["snapshot_id"])
        latest = self.publish("latest", parent=second["snapshot_id"])
        for head in (first["snapshot_id"], second["snapshot_id"], None, "missing"):
            self.db("UPDATE conversation SET current_snapshot_id=?", (head,))
            for read in (self.archive.conversation, self.archive.metadata, self.archive.read_snapshot):
                with self.subTest(head=head, read=read.__name__), self.assertRaises(ArchiveError) as error:
                    read()
                self.assertEqual(error.exception.code, "archive_integrity_error")
            # Validation never repairs the pointer by selecting another row.
            self.assertEqual(self.db("SELECT current_snapshot_id FROM conversation"), [(head,)])
        self.db("UPDATE conversation SET current_snapshot_id=?", (latest["snapshot_id"],))
        self.assertEqual(self.archive.read_snapshot()["snapshot_id"], latest["snapshot_id"])
        self.assertEqual(self.archive.read_snapshot(first["snapshot_id"]), first)

    def test_empty_conversation_cannot_point_at_a_nonexistent_snapshot(self):
        self.db("UPDATE conversation SET current_snapshot_id='missing'")
        with self.assertRaises(ArchiveError) as error:
            self.archive.conversation()
        self.assertEqual(error.exception.code, "archive_integrity_error")

    def test_unfinished_confirmed_view_never_advances_snapshot(self):
        first = self.publish("first")
        pending = self.candidate("unfinished", parent=first["snapshot_id"])
        self.assertEqual(self.archive.read_snapshot()["snapshot_id"], first["snapshot_id"])
        self.assertNotIn("unfinished", self.archive.read_snapshot()["final_state"]["record_files"])
        with self.assertRaises(ArchiveError):
            self.archive.read_run("unfinished")
        for identity, ids in (("different", ["unfinished"]), (first["snapshot_id"], ["other"])):
            with self.assertRaises(ArchiveError):
                self.archive.accept_recovery(identity, ids)
        choice = self.archive.accept_recovery(first["snapshot_id"], ["unfinished"])
        self.assertEqual(self.archive.recovery_choices()[0]["choice_id"], choice)
        self.assertEqual(self.archive.unfinished_runs(), [])
        self.assertEqual(len(self.archive.events("unfinished")), 1)
        self.assertTrue(self.journal.path("unfinished").exists())
        self.archive.verify_view(pending[1])
        with self.assertRaises(ArchiveError):
            self.archive.publish_snapshot(*pending)
        current = self.publish("continued", parent=first["snapshot_id"])
        self.assertEqual([item["run_id"] for item in self.archive.published_runs(current["snapshot_id"])], ["first", "continued"])
        self.assertNotIn("unfinished", current["final_state"]["record_files"])

    def test_error_and_budget_endings_publish_only_valid_inheritable_prefix(self):
        tail = [{"type": "function_call", "name": "tool", "arguments": "{}", "call_id": "unexecuted"}]
        first = self.publish("error", stop="error", tail=tail)
        self.assertEqual(first["final_state"]["input"][1:], tail)
        second = self.publish("budget", parent=first["snapshot_id"], stop="context_budget_exceeded")
        self.assertEqual(second["sequence"], 2)
        args = self.candidate("invalid", parent=second["snapshot_id"], tail=tail)
        row = self.db("SELECT payload FROM runs WHERE id=?", (args[0],))[0][0]
        payload = decode_record(row)
        payload["final_state"]["history_end"] = 2
        changed = encode_record(payload)
        self.db("UPDATE runs SET payload=?,sha256=? WHERE id=?", (changed, hashlib.sha256(changed).hexdigest(), args[0]))
        with self.assertRaises(ArchiveError):
            self.archive.publish_snapshot(*args)
        self.assertEqual(self.archive.read_snapshot()["snapshot_id"], second["snapshot_id"])

    def test_atomic_failure_rolls_back_seal_snapshot_head_and_lifecycle(self):
        args = self.candidate()
        self.db("CREATE TRIGGER fail_head BEFORE UPDATE ON conversation BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with self.assertRaises(ArchiveError) as error:
            self.archive.publish_snapshot(*args)
        self.assertEqual(error.exception.code, "archive_write_error")
        self.assertEqual(self.db("SELECT sealed FROM runs WHERE id=?", (args[0],)), [(0,)])
        self.assertEqual(self.db("SELECT id FROM snapshots"), [])
        self.assertIsNone(self.archive.conversation()["current_snapshot_id"])
        self.assertEqual(self.archive.unfinished_runs()[0]["run_id"], "run")
        self.db("DROP TRIGGER fail_head")
        self.assertEqual(self.archive.publish_snapshot(*args)["snapshot_id"], args[0])

    def test_commit_error_after_success_resolves_same_id_on_fresh_connection(self):
        args = self.candidate()
        previous_connection = self.archive._connection
        real_commit = self.archive._commit

        def committed_then_error():
            real_commit()
            raise OSError("lost commit acknowledgement")

        with patch.object(self.archive, "_commit", side_effect=committed_then_error):
            saved = self.archive.publish_snapshot(*args)
        self.assertEqual(saved["snapshot_id"], args[0])
        self.assertIsNot(self.archive._connection, previous_connection)
        self.assertEqual(self.db("SELECT COUNT(*) FROM snapshots"), [(1,)])
        self.assertEqual(self.archive.unfinished_runs(), [])

    def test_commit_failure_before_success_is_known_unpublished(self):
        args = self.candidate()
        with patch.object(self.archive, "_commit", side_effect=OSError("commit unavailable")):
            with self.assertRaises(ArchiveError) as error:
                self.archive.publish_snapshot(*args)
        self.assertEqual(error.exception.code, "archive_write_error")
        self.assertIsNone(self.archive.resolve_publication(*args))
        self.assertIsNone(self.archive.conversation()["current_snapshot_id"])
        self.assertEqual(self.archive.publish_snapshot(*args)["snapshot_id"], args[0])

    def test_unavailable_fresh_probe_reports_unknown_without_replaying(self):
        args = self.candidate()
        with patch.object(self.archive, "_commit", side_effect=OSError("commit failed")), \
             patch("lineage_agent.archive.store.ArchiveStore", side_effect=OSError("cannot reconnect")):
            with self.assertRaises(ArchiveError) as error:
                self.archive.publish_snapshot(*args)
        self.assertEqual(error.exception.code, "archive_publication_unknown")
        with ArchiveStore(self.paths.directory, create=False) as rebuilt:
            self.assertIsNone(rebuilt.conversation()["current_snapshot_id"])
            self.assertEqual(len(rebuilt.events("run")), 1)
            self.assertEqual(rebuilt.unfinished_runs()[0]["run_id"], "run")

    def test_catalogue_is_lightweight_but_snapshot_checks_bodies_on_each_read(self):
        saved = self.publish()
        self.archive.read_snapshot()
        self.journal.path("run").unlink()
        self.assertEqual(self.archive.metadata()["head"]["snapshot_id"], saved["snapshot_id"])
        with self.assertRaises(ArchiveError):
            self.archive.read_snapshot()

    def test_existing_read_only_open_does_not_create_or_allow_writes(self):
        saved = self.publish()
        self.archive.close()
        with ArchiveStore(self.paths.directory, create=False, read_only=True) as readonly:
            self.assertEqual(readonly.metadata()["current_snapshot_id"], saved["snapshot_id"])
            with self.assertRaises(ArchiveError):
                readonly.begin_run("blocked", saved["snapshot_id"])
        absent = self.paths.directory.parent / "absent"
        with self.assertRaises(ArchiveError):
            ArchiveStore(absent, create=False)
        self.assertFalse(absent.exists())

    def test_legacy_database_and_unknown_recovery_contract_are_rejected(self):
        self.archive.close()
        self.db("PRAGMA user_version=1")
        with self.assertRaises(ArchiveError) as error:
            ArchiveStore(self.paths.directory, create=False)
        self.assertEqual(error.exception.code, "archive_version_error")
        self.db("PRAGMA user_version=2")
        data = self.db("SELECT payload FROM conversation")[0][0]
        payload = decode_record(data)
        payload["recovery_version"] = 99
        changed = encode_record(payload)
        self.db("UPDATE conversation SET payload=?,sha256=?", (changed, hashlib.sha256(changed).hexdigest()))
        with ArchiveStore(self.paths.directory, create=False) as opened:
            with self.assertRaises(ArchiveError) as error:
                opened.metadata()
            self.assertEqual(error.exception.code, "archive_version_error")

    def test_ancestor_lifecycle_and_payload_corruption_refuse_snapshot(self):
        first = self.publish("first")
        second = self.publish("second", parent=first["snapshot_id"])
        self.db("UPDATE run_lifecycle SET status='abandoned' WHERE run_id='first'")
        with self.assertRaises(ArchiveError):
            self.archive.published_runs(second["snapshot_id"])
        self.db("UPDATE run_lifecycle SET status='published' WHERE run_id='first'")
        self.db("UPDATE runs SET payload=? WHERE id=?", (b"corrupt", first["snapshot_id"]))
        with self.assertRaises(ArchiveError):
            self.archive.published_runs(second["snapshot_id"])

    def test_snapshot_rejects_foreign_run_material_and_unrelated_view_lineage(self):
        first = self.publish("first")
        args = self.candidate("current", parent=first["snapshot_id"])
        base = self.archive.view(args[1])
        payload = decode_record(self.db("SELECT payload FROM runs WHERE id=?", (args[0],))[0][0])
        _, orphan = self.journal.append("orphan", make_record_batch("orphan", 0, [{"role": "user", "content": "orphan"}]))
        files = {run: base["record_files"][run] for run in base["record_order"]}
        cases = (({**files, "orphan": orphan}, first["view_id"]), (files, None))
        for index, (files, parent_view) in enumerate(cases):
            view = self.archive.prepare_view(record_files=files, history_scopes={}, summaries=[], artifacts=[],
                                             parent_id=parent_view, fact_ends=base["fact_ends"])
            checkpoint = f"invalid-{index}"
            self.archive.confirm_view(view, checkpoint)
            state = {**payload["final_state"], "resource_view_id": view, "record_files": files}
            record = self.archive.stage_run("current", {**payload, "final_state": state})
            with self.assertRaises(ArchiveError):
                self.archive.publish_snapshot(record, view, checkpoint, first["snapshot_id"])
        self.assertEqual(self.archive.read_snapshot()["snapshot_id"], first["snapshot_id"])

    def test_snapshot_rejects_expanding_published_run_prefix(self):
        first = self.publish("first")
        args = self.candidate("current", parent=first["snapshot_id"])
        payload = decode_record(self.db("SELECT payload FROM runs WHERE id=?", (args[0],))[0][0])
        files = payload["final_state"]["record_files"]
        _, files["first"] = self.journal.append("first", make_record_batch("first", 1, [{"role": "user", "content": "late tail"}]))
        view = self.archive.prepare_view(record_files={"first": files["first"], "current": files["current"]},
                                         history_scopes={}, summaries=[], artifacts=[], parent_id=first["view_id"],
                                         fact_ends=self.archive.view(args[1])["fact_ends"])
        self.archive.confirm_view(view, "expanded")
        payload["final_state"]["resource_view_id"] = view
        record = self.archive.stage_run("current", payload)
        with self.assertRaises(ArchiveError):
            self.archive.publish_snapshot(record, view, "expanded", first["snapshot_id"])
        self.assertEqual(self.archive.read_snapshot()["final_state"]["record_files"]["first"]["record_count"], 1)


if __name__ == "__main__":
    unittest.main()
