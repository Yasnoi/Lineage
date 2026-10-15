"""Append-only original records with checkpoint-controlled readable watermarks."""

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import re
import stat
from threading import RLock
from typing import Any

from lineage_agent.agent.record_codec import decode_record, encode_record
from lineage_agent.storage import ensure_storage_directory, existing_storage_directory, storage_identifier
from lineage_agent.tools.filesystem.paths import _check_components


_ERRORS = {
    "record_create_error": "原始记录文件无法创建，已停止本次请求。",
    "record_write_error": "原始记录未能确认追加保存，已停止后续调用。",
    "record_read_error": "原始记录文件无法读取。",
    "record_integrity_error": "原始记录身份、范围或完整性校验失败。",
    "record_encoding_error": "原始记录包含无法安全保存的值。",
}


class JournalError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code if code in _ERRORS else "record_integrity_error"
        super().__init__(_ERRORS[self.code])


def _identity(ref: str) -> tuple[str, int]:
    match = re.fullmatch(r"history://record/([A-Za-z0-9_-]{1,128})/(0|[1-9][0-9]*)", ref) if type(ref) is str else None
    if match is None:
        raise JournalError("record_integrity_error")
    return match[1], int(match[2])


def _empty() -> dict[str, Any]:
    return {"format_version": 1, "record_count": 0, "byte_end": 0,
            "prefix_sha256": hashlib.sha256(b"").hexdigest()}


class RecordJournal:
    """A trusted conversation directory; this object never imports a session.

    Confirmed watermarks come only from the native checkpointer. A physical
    tail after one of those watermarks is an uncommitted fact, not an available
    history range. Neither failures nor close remove that tail or other files.
    """

    def __init__(self, directory: Path, *, create: bool = True) -> None:
        try:
            self.directory = (ensure_storage_directory if create else existing_storage_directory)(directory)
        except Exception:
            raise JournalError("record_create_error") from None
        self._lock = RLock()
        self._closed = False
        self._written: dict[str, dict[str, Any]] = {}
        self._failed_runs: set[str] = set()
        self._indices: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    def path(self, run_id: str) -> Path:
        try:
            return self.directory / (storage_identifier(run_id) + ".jsonl")
        except Exception:
            raise JournalError("record_integrity_error") from None

    def _prefix(self, run_id: str, watermark: dict[str, Any]) -> bytes:
        try:
            if self._closed:
                raise OSError("closed journal")
            if (type(watermark) is not dict or set(watermark) != {"format_version", "record_count", "byte_end", "prefix_sha256"}
                    or type(watermark["format_version"]) is not int or watermark["format_version"] != 1
                    or type(watermark["record_count"]) is not int or watermark["record_count"] < 0
                    or type(watermark["byte_end"]) is not int or watermark["byte_end"] < 0
                    or type(watermark["prefix_sha256"]) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", watermark["prefix_sha256"]) is None):
                raise JournalError("record_integrity_error")
            path = self.path(run_id)
            if not stat.S_ISREG(_check_components(path).st_mode):
                raise JournalError("record_integrity_error")
            with path.open("rb") as handle:
                data = handle.read(watermark["byte_end"])
            if len(data) != watermark["byte_end"] or hashlib.sha256(data).hexdigest() != watermark["prefix_sha256"]:
                raise JournalError("record_integrity_error")
            return data
        except JournalError:
            raise
        except Exception:
            raise JournalError("record_read_error") from None

    def _index(self, run_id: str, watermark: dict[str, Any]) -> tuple[bytes, list[dict[str, Any]]]:
        data = self._prefix(run_id, watermark)
        key = (run_id, watermark["record_count"], watermark["byte_end"], watermark["prefix_sha256"])
        if key in self._indices:
            return data, self._indices[key]
        try:
            lines = data.splitlines(keepends=True)
            if len(lines) != watermark["record_count"]:
                raise ValueError("record count mismatch")
            offset, receipts = 0, []
            for sequence, line in enumerate(lines):
                if not line.endswith(b"\n"):
                    raise ValueError("incomplete record line")
                record = decode_record(line)
                ref = f"history://record/{run_id}/{sequence}"
                if (record.get("source_ref") != ref or record.get("run_id") != run_id
                        or type(record.get("sequence")) is not int or record["sequence"] != sequence):
                    raise ValueError("record identity mismatch")
                receipts.append({"source_ref": ref, "run_id": run_id, "sequence": sequence,
                                 "offset": offset, "length": len(line), "sha256": hashlib.sha256(line).hexdigest()})
                offset += len(line)
        except Exception:
            raise JournalError("record_integrity_error") from None
        self._indices[key] = receipts
        if len(self._indices) > 16:
            self._indices.pop(next(iter(self._indices)))
        return data, receipts

    def verify(self, run_id: str, watermark: dict[str, Any], receipts=()) -> None:
        with self._lock:
            _, index = self._index(run_id, watermark)
            try:
                for receipt in receipts:
                    sequence = receipt["sequence"]
                    if type(sequence) is not int or not 0 <= sequence < len(index) or receipt != index[sequence]:
                        raise ValueError("receipt mismatch")
            except Exception:
                raise JournalError("record_integrity_error") from None

    def append(self, run_id: str, raw_batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        with self._lock:
            if self._closed or run_id in self._failed_runs:
                raise JournalError("record_write_error")
            path = self.path(run_id)
            try:
                if type(raw_batch) is not list:
                    raise ValueError("invalid batch")
                encoded = []
                previous = None
                for record in raw_batch:
                    sequence = record["sequence"]
                    if (record["run_id"] != run_id or type(sequence) is not int or sequence < 0
                            or record["source_ref"] != f"history://record/{run_id}/{sequence}"
                            or (previous is not None and sequence != previous + 1)):
                        raise JournalError("record_integrity_error")
                    encoded.append(encode_record(record))
                    previous = sequence
            except JournalError:
                raise
            except Exception:
                raise JournalError("record_encoding_error") from None
            try:
                if run_id in self._written:
                    watermark = self._written[run_id]
                    prefix, old_receipts = self._index(run_id, watermark)
                    if path.stat().st_size != watermark["byte_end"]:
                        raise JournalError("record_integrity_error")
                else:
                    if path.exists():
                        # No reopen/resume protocol: only checkpoints may admit
                        # old records for reading, never authorize new writes.
                        raise JournalError("record_integrity_error")
                    watermark, prefix, old_receipts = _empty(), b"", []
                count = len(old_receipts)
                new_data, returned, offset = [], [], len(prefix)
                for record, line in zip(raw_batch, encoded, strict=True):
                    sequence = record["sequence"]
                    if sequence < count:
                        receipt = old_receipts[sequence]
                        if prefix[receipt["offset"]:receipt["offset"] + receipt["length"]] != line:
                            raise JournalError("record_integrity_error")
                        returned.append(deepcopy(receipt))
                        continue
                    if sequence != count + len(new_data):
                        raise JournalError("record_integrity_error")
                    receipt = {"source_ref": record["source_ref"], "run_id": run_id, "sequence": sequence,
                               "offset": offset, "length": len(line), "sha256": hashlib.sha256(line).hexdigest()}
                    returned.append(receipt)
                    new_data.append(line)
                    offset += len(line)
                data = b"".join(new_data)
                if data or run_id not in self._written:
                    _check_components(self.directory)
                    mode = "ab" if run_id in self._written else "xb"
                    with path.open(mode) as handle:
                        if handle.write(data) != len(data):
                            raise OSError("short journal write")
                        handle.flush()
                        os.fsync(handle.fileno())
                complete = prefix + data
                next_watermark = {"format_version": 1, "record_count": count + len(new_data), "byte_end": len(complete),
                                  "prefix_sha256": hashlib.sha256(complete).hexdigest()}
                self.verify(run_id, next_watermark, returned)
                self._written[run_id] = next_watermark
                return returned, deepcopy(next_watermark)
            except JournalError:
                self._failed_runs.add(run_id)
                raise
            except Exception:
                self._failed_runs.add(run_id)
                raise JournalError("record_write_error") from None

    def get(self, source_ref: str, watermark: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            run_id, sequence = _identity(source_ref)
            data, receipts = self._index(run_id, watermark)
            if sequence >= len(receipts):
                raise JournalError("record_integrity_error")
            receipt = receipts[sequence]
            return decode_record(data[receipt["offset"]:receipt["offset"] + receipt["length"]])

    def records(self, run_id: str, watermark: dict[str, Any]) -> list[dict[str, Any]]:
        with self._lock:
            data, receipts = self._index(run_id, watermark)
            return [decode_record(data[item["offset"]:item["offset"] + item["length"]]) for item in receipts]

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._indices.clear()
            self._written.clear()
            self._failed_runs.clear()
