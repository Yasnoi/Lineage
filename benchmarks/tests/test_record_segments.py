"""Explicit same-Run writers keep unconfirmed physical tails outside History."""

from copy import deepcopy
from contextlib import ExitStack
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from lineage_agent.agent.record_codec import encode_record
from lineage_agent.agent.record_journal import JournalError, RecordJournal


class RecordSegmentsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="record-segments-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "history"
        self.run = "same-run"
        self.journal = RecordJournal(self.directory)
        self.addCleanup(self.journal.close)

    def batch(self, sequence, text):
        return [{"source_ref": f"history://record/{self.run}/{sequence}", "run_id": self.run,
                 "sequence": sequence, "group_id": None, "kind": "input",
                 "item": {"role": "user", "content": text}, "call_id": None, "execution": None}]

    def reader(self):
        journal = RecordJournal(self.directory, create=False)
        self.addCleanup(journal.close)
        return journal

    def test_resume_excludes_complete_and_truncated_tail_without_touching_original(self):
        first = self.batch(0, "confirmed 中文")
        receipts, trusted = self.journal.append(self.run, first)
        path = self.journal.path(self.run)
        with path.open("ab") as handle:
            handle.write(encode_record(self.batch(1, "unconfirmed")[0]) + b'{"truncated":')
        before = path.read_bytes()
        resumed = self.reader()
        resumed.resume(self.run, trusted)
        self.assertEqual(len(list(self.directory.iterdir())), 1)
        second = self.batch(1, "new confirmed observation")
        new_receipts, current = resumed.append(self.run, second)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(resumed.records(self.run, current), first + second)
        self.assertEqual(resumed.records(self.run, trusted), first)
        self.assertEqual(new_receipts[0]["offset"], trusted["byte_end"])
        self.assertEqual(current["segments"][0], trusted["segments"][0])
        self.assertEqual(current["segments"][1]["start_sequence"], 1)
        self.assertEqual(len(current["segments"]), 2)
        resumed.verify(self.run, current, receipts + new_receipts)
        resumed.verify_prefix(self.run, trusted, current)

    def test_repeated_process_resumes_keep_global_sequence_and_frozen_segments(self):
        _, first = self.journal.append(self.run, self.batch(0, "zero"))
        a = self.reader()
        a.resume(self.run, first)
        _, second = a.append(self.run, self.batch(1, "one"))
        latest_path = a.paths(self.run, second)[-1]
        with latest_path.open("ab") as handle:
            handle.write(b"unconfirmed tail")
        b = self.reader()
        b.resume(self.run, second)
        _, third = b.append(self.run, self.batch(2, "two"))
        _, fourth = b.append(self.run, self.batch(3, "three"))
        self.assertEqual(len(fourth["segments"]), 3)
        self.assertEqual([r["sequence"] for r in b.records(self.run, fourth)], [0, 1, 2, 3])
        self.assertEqual(latest_path.read_bytes()[-16:], b"unconfirmed tail")
        b.verify_prefix(self.run, first, fourth)
        b.verify_prefix(self.run, second, fourth)
        b.verify_prefix(self.run, third, fourth)

    def test_duplicate_adoption_does_not_create_file_or_logical_record(self):
        batch = self.batch(0, "saved result")
        receipts, trusted = self.journal.append(self.run, batch)
        resumed = self.reader()
        resumed.resume(self.run, trusted)
        self.assertEqual(resumed.append(self.run, batch), (receipts, trusted))
        self.assertEqual(len(list(self.directory.iterdir())), 1)
        next_batch = self.batch(1, "next")
        new_receipts, current = resumed.append(self.run, next_batch)
        self.assertEqual(resumed.append(self.run, next_batch), (new_receipts, current))
        self.assertEqual(len(list(self.directory.iterdir())), 2)

    def test_legacy_fixture_watermark_is_readable_but_new_writes_have_segments(self):
        batch = self.batch(0, "fixture")
        _, watermark = self.journal.append(self.run, batch)
        legacy = {key: value for key, value in watermark.items() if key != "segments"}
        resumed = self.reader()
        self.assertEqual(resumed.records(self.run, legacy), batch)
        resumed.resume(self.run, legacy)
        _, current = resumed.append(self.run, self.batch(1, "new"))
        self.assertEqual(len(current["segments"]), 2)
        resumed.verify_prefix(self.run, legacy, current)

    def test_no_scan_authorizes_files_or_allows_implicit_resume(self):
        _, trusted = self.journal.append(self.run, self.batch(0, "trusted"))
        orphan = self.directory / f"{self.run}.{'f' * 32}.jsonl"
        orphan.write_bytes(encode_record(self.batch(1, "orphan")[0]))
        refused = self.reader()
        with self.assertRaises(JournalError):
            refused.append(self.run, self.batch(1, "implicit"))
        resumed = self.reader()
        resumed.resume(self.run, trusted)
        _, current = resumed.append(self.run, self.batch(1, "explicit"))
        self.assertNotIn(orphan, resumed.paths(self.run, current))
        self.assertEqual(resumed.records(self.run, current)[-1]["item"]["content"], "explicit")
        with self.assertRaises(JournalError):
            resumed.resume(self.run, trusted)

    def test_every_segment_is_rechecked_after_cache_population(self):
        _, trusted = self.journal.append(self.run, self.batch(0, "first"))
        resumed = self.reader()
        resumed.resume(self.run, trusted)
        _, current = resumed.append(self.run, self.batch(1, "second"))
        resumed.records(self.run, current)
        for path in resumed.paths(self.run, current):
            original = path.read_bytes()
            path.write_bytes(original[:-1])
            with self.assertRaises(JournalError):
                resumed.records(self.run, current)
            path.write_bytes(original)
        resumed.paths(self.run, current)[-1].unlink()
        with self.assertRaises(JournalError):
            resumed.verify(self.run, current)

    def test_forged_paths_segments_and_aggregate_hash_are_rejected(self):
        _, trusted = self.journal.append(self.run, self.batch(0, "value"))
        bad_values = []
        for path in ("../outside.jsonl", "history/other-run.jsonl", "history/same-run.bad.jsonl"):
            bad = deepcopy(trusted)
            bad["segments"][0]["path"] = path
            bad_values.append(bad)
        bad = deepcopy(trusted)
        bad["segments"][0]["start_sequence"] = 1
        bad_values.append(bad)
        bad = deepcopy(trusted)
        bad["prefix_sha256"] = hashlib.sha256(b"wrong").hexdigest()
        bad_values.append(bad)
        for bad in bad_values:
            with self.assertRaises(JournalError):
                self.journal.verify(self.run, bad)

    def test_resume_write_failure_keeps_frozen_prefix_and_orphan_new_tail(self):
        _, trusted = self.journal.append(self.run, self.batch(0, "trusted"))
        resumed = self.reader()
        resumed.resume(self.run, trusted)
        with patch("lineage_agent.agent.record_journal.os.fsync", side_effect=OSError("injected")):
            with self.assertRaises(JournalError):
                resumed.append(self.run, self.batch(1, "unconfirmed"))
        self.assertEqual(resumed._written[self.run], trusted)
        self.assertEqual(len(list(self.directory.iterdir())), 2)
        with self.assertRaises(JournalError):
            resumed.append(self.run, self.batch(1, "retry prohibited"))
        fresh = self.reader()
        fresh.resume(self.run, trusted)
        _, current = fresh.append(self.run, self.batch(1, "new trusted"))
        self.assertEqual(len(list(self.directory.iterdir())), 3)
        self.assertEqual(fresh.records(self.run, current)[-1]["item"]["content"], "new trusted")

    def test_source_file_descriptions_keep_global_ranges(self):
        _, first = self.journal.append(self.run, self.batch(0, "first"))
        resumed = self.reader()
        resumed.resume(self.run, first)
        _, current = resumed.append(self.run, self.batch(1, "second") + self.batch(2, "third"))
        ranges = resumed.source_files(self.run, current, 0, 3)
        self.assertEqual([(item["start"], item["end"]) for item in ranges], [(0, 1), (1, 3)])
        self.assertEqual([Path(item["path"]) for item in ranges], resumed.paths(self.run, current))
        with self.assertRaises(JournalError):
            resumed.verify_prefix(self.run, current, first)


class RecordSegmentScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_scope_cursor_survives_segment_growth_and_reader_rebuild(self):
        from lineage_agent.agent.records import RecordStore, make_record_batch
        from lineage_agent.archive import ArchiveStore
        with ExitStack() as stack:
            temporary = stack.enter_context(TemporaryDirectory(prefix="segment-scope-"))
            conversation = Path(temporary) / "conversation"
            journal = RecordJournal(conversation / "history")
            archive = stack.enter_context(ArchiveStore(conversation))
            self.addCleanup(journal.close)
            batch = make_record_batch("run", 0, [{"role": "user", "content": "needle " * 25}])
            _, first = journal.append("run", batch)

            def publish(watermark, scopes, parent=None):
                view = archive.prepare_view(record_files={"run": watermark}, history_scopes=scopes,
                                            summaries=[], artifacts=[], parent_id=parent)
                archive.confirm_view(view, view)
                return view

            initial_view = publish(first, {})
            original = RecordStore(archive=archive, view_id=initial_view)
            scope = await original.register_scope([batch[0]["source_ref"]])
            view = publish(first, {scope["history_ref"]: scope}, initial_view)
            original.close()
            original = RecordStore(archive=archive, view_id=view)
            page = await original.search(scope["history_ref"], ["needle"])
            self.assertTrue(page["has_more"])
            cursor = page["next_cursor"]
            original.close()
            with journal.path("run").open("ab") as handle:
                handle.write(encode_record(make_record_batch("run", 1, [{"role": "user", "content": "needle tail"}])[0]))
            resumed = RecordJournal(conversation / "history", create=False)
            self.addCleanup(resumed.close)
            resumed.resume("run", first)
            later = make_record_batch("run", 1, [{"role": "user", "content": "needle new confirmed"}])
            _, watermark = resumed.append("run", later)
            latest = publish(watermark, {scope["history_ref"]: scope}, view)
            rebuilt = RecordStore(archive=archive, view_id=latest)
            self.addCleanup(rebuilt.close)
            continuation = await rebuilt.search(scope["history_ref"], ["needle"], cursor)
            self.assertTrue(continuation["hits"])
            self.assertTrue(all(hit["source_ref"] == batch[0]["source_ref"] for hit in continuation["hits"]))
            self.assertEqual([r["item"]["content"] for r in await rebuilt.records()],
                             [batch[0]["item"]["content"], later[0]["item"]["content"]])
            self.assertEqual((await rebuilt.scope(scope["history_ref"]))["record_files"], {"run": first})


if __name__ == "__main__":
    unittest.main()
