"""Durable originals, immutable receipts, and non-executable opaque encodings."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from lineage_agent.agent.record_codec import decode_record, encode_record
from lineage_agent.agent.record_journal import JournalError, RecordJournal
from lineage_agent.agent.records import make_record_batch


class RecordJournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="original-journal-test-")
        self.addCleanup(self.temporary.cleanup)
        self.run = str(uuid4())
        self.directory = Path(self.temporary.name) / ".lineage" / self.run / "history"
        self.journal = RecordJournal(self.directory)

    def batch(self, start=0, text="original exact requirement 37"):
        return make_record_batch(self.run, start, [{"role": "user", "content": text}])

    def test_append_receipts_and_old_watermark_exclude_unconfirmed_tail(self):
        first = self.batch()
        receipts, first_watermark = self.journal.append(self.run, first)
        second = self.batch(1, "later actual tool fact")
        later_receipts, watermark = self.journal.append(self.run, second)
        self.assertEqual(self.journal.records(self.run, first_watermark), first)
        self.assertEqual(self.journal.records(self.run, watermark), first + second)
        self.assertEqual(self.journal.append(self.run, second), (later_receipts, watermark))
        self.assertNotIn("item", receipts[0])
        self.assertEqual(receipts[0]["offset"], 0)
        self.assertEqual(receipts[0]["length"], first_watermark["byte_end"])
        self.assertEqual(later_receipts[0]["offset"], first_watermark["byte_end"])
        self.journal.verify(self.run, watermark, receipts + later_receipts)
        with self.assertRaises(JournalError):
            self.journal.get(second[0]["source_ref"], first_watermark)
        self.assertEqual(len(self.journal.path(self.run).read_text(encoding="utf-8").splitlines()), 2)

    def test_close_keeps_file_but_does_not_reopen_writable_run(self):
        batch = self.batch()
        _, watermark = self.journal.append(self.run, batch)
        path = self.journal.path(self.run)
        before = path.read_bytes()
        self.journal.close()
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(JournalError):
            self.journal.records(self.run, watermark)
        reader = RecordJournal(self.directory)
        self.assertEqual(reader.records(self.run, watermark), batch)
        with self.assertRaises(JournalError):
            reader.append(self.run, self.batch(1))
        self.assertEqual(path.read_bytes(), before)

    def test_codec_preserves_json_and_fixed_safe_non_json_types(self):
        original = self.batch()[0]
        original["item"].update(a={1, 2}, b=float("nan"), c=float("inf"), d=float("-inf"),
                                e=("tuple", frozenset({3, 4})), f=b"\x00\xff", g=bytearray(b"bytes"),
                                h={1: "integer key", (2, 3): [True, None]}, i="\ud800")
        encoded = encode_record(original)
        envelope = json.loads(encoded)
        self.assertEqual(envelope["record"]["item"]["content"], original["item"]["content"])
        self.assertIsNone(envelope["record"]["item"]["a"])
        decoded = decode_record(encoded)
        self.assertEqual(decoded["item"]["a"], {1, 2})
        self.assertTrue(math.isnan(decoded["item"]["b"]))
        for key in "cdefghi":
            self.assertEqual(decoded["item"][key], original["item"][key])
        self.assertEqual(encode_record(decoded), encoded)
        _, watermark = self.journal.append(self.run, [original])
        self.assertEqual(self.journal.get(original["source_ref"], watermark)["item"]["f"], b"\x00\xff")

    def test_unknown_objects_cycles_and_forged_tags_are_rejected(self):
        for value in (object(), lambda: None):
            bad = self.batch()[0]
            bad["item"]["opaque"] = value
            with self.assertRaises(JournalError) as error:
                self.journal.append(self.run, [bad])
            self.assertEqual(error.exception.code, "record_encoding_error")
            self.assertFalse(self.journal.path(self.run).exists())
        bad = self.batch()[0]
        bad["item"]["cycle"] = bad
        with self.assertRaises(JournalError):
            self.journal.append(self.run, [bad])
        envelope = json.loads(encode_record(self.batch()[0]))
        envelope["record"]["item"]["content"] = None
        envelope["opaque_values"] = [{"path": ["item", "content"], "encoding": ["python_object", "os.system"]}]
        with self.assertRaises(ValueError):
            decode_record(json.dumps(envelope).encode() + b"\n")

    def test_identity_gaps_and_changed_duplicates_are_rejected(self):
        _, watermark = self.journal.append(self.run, self.batch())
        with self.assertRaises(JournalError):
            self.journal.append(self.run, self.batch(2))
        self.assertEqual(len(self.journal.records(self.run, watermark)), 1)
        other = RecordJournal(self.directory / "other")
        other.append(self.run, self.batch())
        with self.assertRaises(JournalError):
            other.append(self.run, self.batch(0, "changed original"))

    def test_missing_truncated_or_changed_file_never_uses_cached_body(self):
        receipts, watermark = self.journal.append(self.run, self.batch())
        self.journal.records(self.run, watermark)
        path = self.journal.path(self.run)
        before = path.read_bytes()
        for data in (before[:-1], before.replace(b"37", b"73")):
            path.write_bytes(data)
            with self.assertRaises(JournalError) as error:
                self.journal.verify(self.run, watermark, receipts)
            self.assertEqual(error.exception.code, "record_integrity_error")
        path.unlink()
        with self.assertRaises(JournalError) as error:
            self.journal.records(self.run, watermark)
        self.assertEqual(error.exception.code, "record_read_error")

    def test_fsync_failure_retains_unpublished_file_and_blocks_more_appends(self):
        with patch("lineage_agent.agent.record_journal.os.fsync", side_effect=OSError("private")):
            with self.assertRaises(JournalError) as error:
                self.journal.append(self.run, self.batch())
        self.assertEqual(error.exception.code, "record_write_error")
        path = self.journal.path(self.run)
        self.assertTrue(path.exists())
        before = path.read_bytes()
        with self.assertRaises(JournalError):
            self.journal.append(self.run, self.batch())
        self.assertEqual(path.read_bytes(), before)

    def test_short_write_keeps_old_confirmed_prefix_readable(self):
        first = self.batch()
        _, watermark = self.journal.append(self.run, first)
        path = self.journal.path(self.run)
        original_open = Path.open

        class ShortWriter:
            def __init__(self, handle):
                self.handle = handle
            def __enter__(self):
                self.handle.__enter__()
                return self
            def __exit__(self, *args):
                return self.handle.__exit__(*args)
            def write(self, data):
                return self.handle.write(data[:len(data) // 2])

        def opened(target, mode="r", *args, **kwargs):
            handle = original_open(target, mode, *args, **kwargs)
            return ShortWriter(handle) if target == path and mode == "ab" else handle

        with patch.object(Path, "open", opened):
            with self.assertRaises(JournalError) as error:
                self.journal.append(self.run, self.batch(1))
        self.assertEqual(error.exception.code, "record_write_error")
        self.assertGreater(path.stat().st_size, watermark["byte_end"])
        self.assertEqual(self.journal.records(self.run, watermark), first)
        with self.assertRaises(JournalError):
            self.journal.append(self.run, self.batch(1))

    def test_flush_failure_does_not_publish_tail_or_resume_writing(self):
        first = self.batch()
        _, watermark = self.journal.append(self.run, first)
        path = self.journal.path(self.run)
        native_open = Path.open

        class FailingFlush:
            def __init__(self, handle):
                self.handle = handle
            def __enter__(self):
                self.handle.__enter__()
                return self
            def __exit__(self, *args):
                return self.handle.__exit__(*args)
            def write(self, data):
                return self.handle.write(data)
            def flush(self):
                raise OSError("private flush failure")

        def opened(target, mode="r", *args, **kwargs):
            handle = native_open(target, mode, *args, **kwargs)
            return FailingFlush(handle) if target == path and mode == "ab" else handle

        with patch.object(Path, "open", opened):
            with self.assertRaises(JournalError) as error:
                self.journal.append(self.run, self.batch(1, "unconfirmed after flush failure"))
        self.assertEqual(error.exception.code, "record_write_error")
        self.assertEqual(self.journal._written[self.run], watermark)
        self.assertEqual(self.journal.records(self.run, watermark), first)
        retained = path.read_bytes()
        self.assertGreater(len(retained), watermark["byte_end"])
        with self.assertRaises(JournalError):
            self.journal.append(self.run, self.batch(1, "must not retry"))
        self.assertEqual(path.read_bytes(), retained)

    def test_append_readback_failure_does_not_publish_tail_or_resume_writing(self):
        first = self.batch()
        _, watermark = self.journal.append(self.run, first)
        path = self.journal.path(self.run)
        native_open = Path.open
        appended = []

        def opened(target, mode="r", *args, **kwargs):
            if target == path and mode == "ab":
                appended.append(True)
            elif target == path and mode == "rb" and appended:
                raise OSError("private append readback failure")
            return native_open(target, mode, *args, **kwargs)

        with patch.object(Path, "open", opened):
            with self.assertRaises(JournalError) as error:
                self.journal.append(self.run, self.batch(1, "unconfirmed after readback failure"))
        self.assertEqual(error.exception.code, "record_read_error")
        self.assertEqual(appended, [True])
        self.assertEqual(self.journal._written[self.run], watermark)
        self.assertEqual(self.journal.records(self.run, watermark), first)
        retained = path.read_bytes()
        self.assertGreater(len(retained), watermark["byte_end"])
        with self.assertRaises(JournalError):
            self.journal.append(self.run, self.batch(1, "must not retry"))
        self.assertEqual(path.read_bytes(), retained)

    def test_whole_batch_encoding_finishes_before_first_write(self):
        batch = self.batch() + self.batch(1)
        batch[1]["item"]["unsupported"] = object()
        with self.assertRaises(JournalError) as error:
            self.journal.append(self.run, batch)
        self.assertEqual(error.exception.code, "record_encoding_error")
        self.assertFalse(self.journal.path(self.run).exists())

    def test_forged_receipt_and_file_with_wrong_identity_fail(self):
        receipts, watermark = self.journal.append(self.run, self.batch())
        changed = deepcopy(receipts)
        changed[0]["offset"] = 1
        with self.assertRaises(JournalError):
            self.journal.verify(self.run, watermark, changed)
        wrong = self.batch()[0]
        wrong["sequence"] = 3
        data = encode_record(wrong)
        self.journal.path(self.run).write_bytes(data)
        forged = {**watermark, "byte_end": len(data), "prefix_sha256": hashlib.sha256(data).hexdigest()}
        with self.assertRaises(JournalError):
            self.journal.verify(self.run, forged)


if __name__ == "__main__":
    unittest.main()
