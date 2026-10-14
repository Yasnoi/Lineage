"""Immutable business objects and explicitly confirmed resource views.

The database transaction covers only these registrations. File bodies are
written first, and a resource view is confirmed only after the caller's state
save succeeds. Opening this database does not open or resume an Agent session.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sqlite3
import stat
from threading import RLock
from typing import Any
from uuid import uuid4

from lineage_agent.agent.record_codec import decode_record, encode_record
from lineage_agent.agent.record_journal import JournalError, RecordJournal
from lineage_agent.storage import ensure_storage_directory, storage_identifier
from lineage_agent.tools.filesystem.paths import _check_components


_ERRORS = {
    "archive_create_error": "业务归档无法安全创建。",
    "archive_write_error": "业务归档保存未完成，已停止后续调用。",
    "archive_read_error": "业务归档无法读取。",
    "archive_integrity_error": "业务归档身份、引用或完整性校验失败。",
    "archive_version_error": "业务归档格式版本不受支持。",
}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SCOPE = re.compile(r"history://scope/[0-9a-f-]{36}\Z")
_SOURCE = re.compile(r"history://record/([A-Za-z0-9_-]{1,128})/(0|[1-9][0-9]*)\Z")
_ARTIFACT = re.compile(r"artifact://tool/([0-9a-f]{32})\Z")
_SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, payload BLOB NOT NULL, sha256 TEXT NOT NULL);
CREATE TABLE objects (kind TEXT NOT NULL, id TEXT NOT NULL, payload BLOB NOT NULL,
                      sha256 TEXT NOT NULL, PRIMARY KEY(kind,id));
CREATE TABLE views (id TEXT PRIMARY KEY, payload BLOB NOT NULL, sha256 TEXT NOT NULL,
                    confirmed INTEGER NOT NULL DEFAULT 0 CHECK(confirmed IN (0,1)));
CREATE TABLE checkpoints (id TEXT PRIMARY KEY, view_id TEXT NOT NULL REFERENCES views(id));
CREATE TABLE events (run_id TEXT NOT NULL, sequence INTEGER NOT NULL, kind TEXT NOT NULL,
                     event_key TEXT, payload BLOB NOT NULL, sha256 TEXT NOT NULL,
                     checkpoint_id TEXT,
                     PRIMARY KEY(run_id,sequence), UNIQUE(run_id,event_key));
CREATE TABLE runs (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, payload BLOB NOT NULL,
                   sha256 TEXT NOT NULL, sealed INTEGER NOT NULL DEFAULT 0,
                   view_id TEXT, checkpoint_id TEXT);
CREATE UNIQUE INDEX sealed_run ON runs(run_id) WHERE sealed=1;
"""


class ArchiveError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code if code in _ERRORS else "archive_integrity_error"
        super().__init__(_ERRORS[self.code])


def _check(condition: bool) -> None:
    if not condition:
        raise ArchiveError("archive_integrity_error")


def _encoded(value: dict[str, Any]) -> tuple[bytes, str]:
    data = encode_record(value)
    return data, hashlib.sha256(data).hexdigest()


def _decoded(data: bytes, digest: str) -> dict[str, Any]:
    _check(type(data) is bytes and type(digest) is str and hashlib.sha256(data).hexdigest() == digest)
    try:
        return decode_record(data)
    except Exception:
        raise ArchiveError("archive_integrity_error") from None


def _covers(ranges: list[dict[str, Any]], interval: dict[str, Any]) -> bool:
    """Check coverage by the union, without expanding potentially large ranges."""
    position = interval["start"]
    for candidate in sorted((part for part in ranges if part["run_id"] == interval["run_id"]),
                            key=lambda part: part["start"]):
        if candidate["start"] > position:
            break
        position = max(position, candidate["end"])
        if position >= interval["end"]:
            return True
    return False


class ArchiveStore:
    """Trusted conversation-directory binding; no directory discovery or import."""

    def __init__(self, directory: Path, *, create: bool = True) -> None:
        self._lock = RLock()
        self._closed = False
        self._operation_depth = 0
        self._validated_files: set[tuple[str, bytes]] = set()
        self._validated_artifacts: set[bytes] = set()
        self._connection: sqlite3.Connection | None = None
        try:
            self.directory = Path(directory).absolute()
            self.conversation_id = storage_identifier(self.directory.name)
            if create:
                self.directory = ensure_storage_directory(self.directory)
            else:
                _check(stat.S_ISDIR(_check_components(self.directory).st_mode))
            self.path = self.directory / "conversation.sqlite"
            existed = self.path.exists()
            self._check_paths(require_database=existed)
            if not existed and not create:
                raise ArchiveError("archive_read_error")
            # URI mode prevents a supposedly read-existing binding creating a DB.
            uri = self.path.as_uri() + ("?mode=rw" if existed else "?mode=rwc")
            self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA synchronous=FULL")
            if not existed:
                with self._connection:
                    self._connection.executescript(_SCHEMA)
                    data, digest = _encoded({"schema_version": 1, "conversation_id": self.conversation_id,
                                             "cursor_key": os.urandom(32)})
                    self._connection.execute("INSERT INTO metadata VALUES ('identity',?,?)", (data, digest))
                    self._connection.execute("PRAGMA user_version=1")
            version = self._connection.execute("PRAGMA user_version").fetchone()[0]
            if version != 1:
                raise ArchiveError("archive_version_error")
            row = self._connection.execute("SELECT payload,sha256 FROM metadata WHERE key='identity'").fetchone()
            _check(row is not None)
            identity = _decoded(*row)
            _check(identity.get("schema_version") == 1 and identity.get("conversation_id") == self.conversation_id
                   and type(identity.get("cursor_key")) is bytes and len(identity["cursor_key"]) == 32)
        except Exception as error:
            if self._connection is not None:
                self._connection.close()
            self._closed = True
            if isinstance(error, ArchiveError):
                raise
            raise ArchiveError("archive_create_error" if create else "archive_read_error") from None

    def _check_paths(self, *, require_database: bool = True) -> None:
        _check(stat.S_ISDIR(_check_components(self.directory).st_mode))
        for suffix in ("", "-journal", "-wal", "-shm"):
            path = self.path.with_name(self.path.name + suffix)
            # lexists semantics: a dangling link must not be treated as absent.
            if suffix == "" and require_database or path.exists() or path.is_symlink():
                _check(stat.S_ISREG(_check_components(path).st_mode))

    @contextmanager
    def _operation(self, *, write: bool = False):
        with self._lock:
            outermost = self._operation_depth == 0
            if outermost:
                self._validated_files.clear()
                self._validated_artifacts.clear()
            self._operation_depth += 1
            try:
                if self._closed or self._connection is None:
                    raise ArchiveError("archive_write_error" if write else "archive_read_error")
                self._check_paths()
                if write:
                    with self._connection:
                        yield self._connection
                else:
                    yield self._connection
            except ArchiveError:
                raise
            except JournalError as error:
                raise ArchiveError("archive_integrity_error" if error.code == "record_integrity_error"
                                   else "archive_read_error") from None
            except Exception:
                raise ArchiveError("archive_write_error" if write else "archive_read_error") from None
            finally:
                self._operation_depth -= 1
                if outermost:
                    # Deduplicate only within this operation. A later read must
                    # re-open the body even if all registration metadata agrees.
                    self._validated_files.clear()
                    self._validated_artifacts.clear()

    def _path(self, relative: str) -> Path:
        _check(type(relative) is str and bool(relative) and "\\" not in relative)
        parsed = PurePosixPath(relative)
        _check(not parsed.is_absolute() and not PureWindowsPath(relative).drive
               and str(parsed) == relative and all(part not in ("", ".", "..") for part in parsed.parts))
        for part in parsed.parts:
            _check(not PureWindowsPath(part).is_reserved() and not part.endswith((".", " "))
                   and not any(ord(char) < 32 or char in '<>:"|?*' for char in part))
        path = self.directory.joinpath(*parsed.parts)
        _check(stat.S_ISREG(_check_components(path).st_mode))
        return path

    def _object(self, kind: str, identity: str) -> dict[str, Any]:
        row = self._connection.execute("SELECT payload,sha256 FROM objects WHERE kind=? AND id=?", (kind, identity)).fetchone()
        _check(row is not None)
        value = _decoded(*row)
        _check(value.get({"summary": "summary_id", "scope": "history_ref", "artifact": "artifact_ref"}[kind]) == identity)
        return value

    def _save_object(self, kind: str, identity: str, value: dict[str, Any]) -> None:
        data, digest = _encoded(value)
        row = self._connection.execute("SELECT payload,sha256 FROM objects WHERE kind=? AND id=?", (kind, identity)).fetchone()
        if row is not None:
            _decoded(*row)
            _check(row == (data, digest))
            return
        self._connection.execute("INSERT INTO objects VALUES (?,?,?,?)", (kind, identity, data, digest))

    def _files(self, files: dict[str, Any], paths: dict[str, str] | None = None) -> None:
        _check(type(files) is dict)
        if not files:
            return
        _check(stat.S_ISDIR(_check_components(self.directory / "history").st_mode))
        journal = RecordJournal(self.directory / "history")
        try:
            for run_id, watermark in files.items():
                storage_identifier(run_id)
                expected = f"history/{run_id}.jsonl"
                if paths is not None:
                    _check(paths.get(run_id) == expected)
                identity = (run_id, encode_record(watermark))
                if identity in self._validated_files:
                    continue
                self._path(expected)
                journal.verify(run_id, watermark)
                self._validated_files.add(identity)
        finally:
            journal.close()

    def _scope(self, record: dict[str, Any]) -> None:
        _check(type(record) is dict and type(record.get("history_ref")) is str
               and _SCOPE.fullmatch(record["history_ref"]) is not None)
        files, ranges = record.get("record_files"), record.get("ranges")
        self._files(files)
        _check(type(ranges) is list and bool(ranges))
        intervals: dict[str, list[tuple[int, int]]] = {}
        for interval in ranges:
            _check(type(interval) is dict and interval.get("run_id") in files)
            start, end, run_id = interval.get("start"), interval.get("end"), interval["run_id"]
            _check(type(start) is int and type(end) is int and 0 <= start < end <= files[run_id]["record_count"])
            _check(all(end <= before or start >= after for before, after in intervals.get(run_id, [])))
            intervals.setdefault(run_id, []).append((start, end))

    def _summary(self, record: dict[str, Any]) -> None:
        _check(type(record) is dict and type(record.get("summary_id")) is str and bool(record["summary_id"])
               and record.get("lane") in ("H", "R") and type(record.get("body")) is str and bool(record["body"].strip()))
        storage_identifier(record["summary_id"])
        storage_identifier(record["run_id"])
        reference = record.get("history_ref") or (record.get("scope") or {}).get("history_ref")
        _check(type(reference) is str)
        scope = self._object("scope", reference)
        self._scope(scope)
        if "ranges" in record:
            _check(record["ranges"] == scope["ranges"])
        if "scope" in record:
            _check(record["scope"] == scope)
        previous = record.get("previous", [])
        _check(type(previous) is list)
        for old in previous:
            identity = old.get("summary_id") if type(old) is dict else old
            _check(type(identity) is str and identity != record["summary_id"])
            saved = self._object("summary", identity)
            old_ref = saved.get("history_ref") or (saved.get("scope") or {}).get("history_ref")
            old_scope = self._object("scope", old_ref)
            self._scope(old_scope)
            _check(all(_covers(scope["ranges"], interval) for interval in old_scope["ranges"]))
            if type(old) is dict:
                for field in ("lane", "history_ref", "ranges"):
                    if field in old:
                        _check(old[field] == saved.get(field))
        sources = record.get("source_refs", [])
        _check(type(sources) is list and len(sources) == len(set(sources)))
        for ref in sources:
            match = _SOURCE.fullmatch(ref) if type(ref) is str else None
            _check(match is not None)
            _check(any(bounds["run_id"] == match[1] and bounds["start"] <= int(match[2]) < bounds["end"]
                       for bounds in scope["ranges"]))

    def _artifact(self, record: dict[str, Any]) -> None:
        identity = encode_record(record)
        if identity in self._validated_artifacts:
            return
        _check(type(record) is dict and type(record.get("artifact_ref")) is str)
        match = _ARTIFACT.fullmatch(record["artifact_ref"])
        _check(match is not None and record.get("path") == f"artifacts/{match[1]}.raw")
        digest, metadata = record.get("sha256"), record.get("metadata")
        _check(type(digest) is str and _HASH.fullmatch(digest) is not None and type(metadata) is dict
               and metadata.get("artifact_ref") == record["artifact_ref"])
        data = self._path(record["path"]).read_bytes()
        _check(hashlib.sha256(data).hexdigest() == digest and type(metadata.get("raw_bytes")) is int
               and metadata["raw_bytes"] == len(data) and type(metadata.get("tool_name")) is str
               and bool(metadata["tool_name"]))
        # Import only while validating: ArtifactStore itself consumes ArchiveStore.
        from lineage_agent.artifacts.store import tool_result_view
        text, kind, reason = tool_result_view(data.decode("utf-8"), metadata["tool_name"])
        _check(metadata.get("total_chars") == len(text) and metadata.get("view_kind") == kind
               and metadata.get("degradation_reason") == reason)
        self._validated_artifacts.add(identity)

    def save_summary(self, record: dict[str, Any]) -> None:
        with self._operation(write=True):
            self._summary(record)
            self._save_object("summary", record["summary_id"], record)

    def save_scope(self, record: dict[str, Any]) -> None:
        with self._operation(write=True):
            self._scope(record)
            self._save_object("scope", record["history_ref"], record)

    def save_artifact(self, record: dict[str, Any]) -> None:
        with self._operation(write=True):
            self._artifact(record)
            self._save_object("artifact", record["artifact_ref"], record)

    def _view(self, identity: str, confirmed: bool = True) -> dict[str, Any]:
        row = self._connection.execute("SELECT payload,sha256,confirmed FROM views WHERE id=?", (identity,)).fetchone()
        _check(row is not None and (not confirmed or row[2] == 1))
        value = _decoded(row[0], row[1])
        _check(value.get("view_id") == identity and value.get("conversation_id") == self.conversation_id)
        return value

    def view(self, identity: str, confirmed: bool = True) -> dict[str, Any]:
        with self._operation():
            return self._view(identity, confirmed)

    def _verify_view(self, identity: str, confirmed: bool) -> dict[str, Any]:
        view = self._view(identity, confirmed)
        order = view.get("record_order")
        _check(type(order) is list and all(type(run_id) is str for run_id in order)
               and len(order) == len(set(order)) and set(order) == set(view["record_files"]))
        self._files(view["record_files"], view["record_paths"])
        _check(type(view["history_scopes"]) is dict)
        for ref, recorded in view["history_scopes"].items():
            scope = self._object("scope", ref)
            _check(scope == recorded)
            self._scope(scope)
            for run, watermark in scope["record_files"].items():
                _check(run in view["record_files"] and watermark["record_count"] <= view["record_files"][run]["record_count"]
                       and watermark["byte_end"] <= view["record_files"][run]["byte_end"])
        for field in ("summaries", "artifacts"):
            _check(type(view[field]) is list and len(view[field]) == len(set(view[field])))
        for identity in view["summaries"]:
            record = self._object("summary", identity)
            self._summary(record)
            ref = record.get("history_ref") or record["scope"]["history_ref"]
            _check(ref in view["history_scopes"])
            for old in record.get("previous", []):
                _check((old.get("summary_id") if type(old) is dict else old) in view["summaries"])
        for ref in view["artifacts"]:
            self._artifact(self._object("artifact", ref))
        _check(type(view["fact_ends"]) is dict)
        for run_id, end in view["fact_ends"].items():
            _check(type(end) is int and end >= 0)
            events = self._events(run_id, end)
            _check(len(events) == end)
        return view

    def verify_view(self, identity: str, confirmed: bool = True) -> None:
        with self._operation():
            self._verify_view(identity, confirmed)

    def prepare_view(self, *, record_files: dict, history_scopes: dict,
                     summaries, artifacts, parent_id: str | None = None, fact_ends: dict | None = None) -> str:
        with self._operation(write=True):
            identity = str(uuid4())
            value = {"view_id": identity, "conversation_id": self.conversation_id, "parent_id": parent_id,
                     "record_files": deepcopy(record_files),
                     "record_order": list(record_files),
                     "record_paths": {run: f"history/{run}.jsonl" for run in record_files},
                     "history_scopes": deepcopy(history_scopes), "summaries": list(summaries),
                     "artifacts": list(artifacts), "fact_ends": deepcopy(fact_ends or {})}
            if parent_id is not None:
                parent = self._verify_view(parent_id, True)
                _check(value["record_order"][:len(parent["record_order"])] == parent["record_order"])
                for run, watermark in parent["record_files"].items():
                    _check(run in record_files and watermark["record_count"] <= record_files[run]["record_count"]
                           and watermark["byte_end"] <= record_files[run]["byte_end"])
                _check(all(history_scopes.get(ref) == scope for ref, scope in parent["history_scopes"].items()))
                for field in ("summaries", "artifacts"):
                    _check(set(parent[field]).issubset(value[field]))
                _check(all(value["fact_ends"].get(run, -1) >= end for run, end in parent["fact_ends"].items()))
            data, digest = _encoded(value)
            self._connection.execute("INSERT INTO views(id,payload,sha256) VALUES (?,?,?)", (identity, data, digest))
            self._verify_view(identity, False)
            return identity

    def confirm_view(self, identity: str, checkpoint_id: str) -> None:
        with self._operation(write=True):
            _check(type(checkpoint_id) is str and bool(checkpoint_id))
            view = self._verify_view(identity, False)
            row = self._connection.execute("SELECT view_id FROM checkpoints WHERE id=?", (checkpoint_id,)).fetchone()
            _check(row is None or row[0] == identity)
            self._connection.execute("UPDATE views SET confirmed=1 WHERE id=?", (identity,))
            self._connection.execute("INSERT OR IGNORE INTO checkpoints VALUES (?,?)", (checkpoint_id, identity))
            for run_id, end in view["fact_ends"].items():
                self._connection.execute("UPDATE events SET checkpoint_id=? WHERE run_id=? AND sequence<=? AND checkpoint_id IS NULL",
                                         (checkpoint_id, run_id, end))

    def view_for_checkpoint(self, checkpoint_id: str) -> str:
        with self._operation():
            row = self._connection.execute("SELECT view_id FROM checkpoints WHERE id=?", (checkpoint_id,)).fetchone()
            _check(row is not None)
            self._view(row[0])
            return row[0]

    def summary(self, identity: str, view_id: str, *, confirmed: bool = True) -> dict[str, Any]:
        """Read an admitted summary, or validate a staged view before state save."""
        with self._operation():
            view = self._verify_view(view_id, confirmed)
            _check(identity in view["summaries"])
            return self._object("summary", identity)

    def scope(self, ref: str, view_id: str) -> dict[str, Any]:
        with self._operation():
            view = self._verify_view(view_id, True)
            _check(ref in view["history_scopes"])
            return self._object("scope", ref)

    def artifact(self, ref: str, view_id: str | None = None) -> dict[str, Any]:
        with self._operation():
            if view_id is not None:
                _check(ref in self._verify_view(view_id, True)["artifacts"])
            record = self._object("artifact", ref)
            self._artifact(record)
            return record

    def cursor_key(self) -> bytes:
        with self._operation():
            row = self._connection.execute("SELECT payload,sha256 FROM metadata WHERE key='identity'").fetchone()
            _check(row is not None)
            value = _decoded(*row)
            _check(value.get("conversation_id") == self.conversation_id and type(value.get("cursor_key")) is bytes
                   and len(value["cursor_key"]) == 32)
            return value["cursor_key"]

    def append_event(self, run_id: str, kind: str, payload: dict, *, key: str | None = None) -> int:
        with self._operation(write=True):
            storage_identifier(run_id)
            _check(type(kind) is str and bool(kind) and (key is None or type(key) is str and bool(key)))
            data, digest = _encoded(payload)
            if key is not None:
                row = self._connection.execute("SELECT sequence,kind,payload,sha256 FROM events WHERE run_id=? AND event_key=?",
                                               (run_id, key)).fetchone()
                if row is not None:
                    _decoded(row[2], row[3])
                    _check(row[1:] == (kind, data, digest))
                    return row[0]
            sequence = self._connection.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM events WHERE run_id=?", (run_id,)).fetchone()[0]
            self._connection.execute("INSERT INTO events(run_id,sequence,kind,event_key,payload,sha256) VALUES (?,?,?,?,?,?)",
                                     (run_id, sequence, kind, key, data, digest))
            return sequence

    def _events(self, run_id: str, end: int | None) -> list[dict[str, Any]]:
        storage_identifier(run_id)
        _check(end is None or type(end) is int and end >= 0)
        rows = self._connection.execute("SELECT sequence,kind,payload,sha256,checkpoint_id FROM events WHERE run_id=? AND (? IS NULL OR sequence<=?) ORDER BY sequence",
                                       (run_id, end, end)).fetchall()
        values = []
        for expected, (sequence, kind, data, digest, checkpoint_id) in enumerate(rows, 1):
            _check(sequence == expected)
            payload = _decoded(data, digest)
            if "checkpoint_id" in payload and payload["checkpoint_id"] is None:
                payload["checkpoint_id"] = checkpoint_id
            values.append({"sequence": sequence, "kind": kind, "payload": payload})
        if end is not None:
            _check(len(values) == end)
        return values

    def events(self, run_id: str, *, end: int | None = None) -> list[dict[str, Any]]:
        with self._operation():
            return self._events(run_id, end)

    def event_end(self, run_id: str) -> int:
        with self._operation():
            return len(self._events(run_id, None))

    def stage_run(self, run_id: str, payload: dict) -> str:
        with self._operation(write=True):
            storage_identifier(run_id)
            _check("run_id" not in payload or payload["run_id"] == run_id)
            identity = str(uuid4())
            data, digest = _encoded(payload)
            self._connection.execute("INSERT INTO runs(id,run_id,payload,sha256) VALUES (?,?,?,?)", (identity, run_id, data, digest))
            return identity

    def seal_run(self, record_id: str, view_id: str, checkpoint_id: str) -> None:
        with self._operation(write=True):
            view = self._verify_view(view_id, True)
            row = self._connection.execute("SELECT run_id,payload,sha256,sealed,view_id,checkpoint_id FROM runs WHERE id=?", (record_id,)).fetchone()
            _check(row is not None)
            payload = _decoded(row[1], row[2])
            _check(self.view_for_checkpoint(checkpoint_id) == view_id)
            if row[3]:
                _check(row[4:] == (view_id, checkpoint_id))
                return
            if "fact_end" in payload:
                _check(payload["fact_end"] == view["fact_ends"].get(row[0], 0))
                self._events(row[0], payload["fact_end"])
            self._connection.execute("UPDATE runs SET sealed=1,view_id=?,checkpoint_id=? WHERE id=?", (view_id, checkpoint_id, record_id))

    def read_run(self, run_id: str) -> dict[str, Any]:
        with self._operation():
            row = self._connection.execute("SELECT payload,sha256,view_id,checkpoint_id FROM runs WHERE run_id=? AND sealed=1", (run_id,)).fetchone()
            _check(row is not None)
            payload = _decoded(row[0], row[1])
            view = self._verify_view(row[2], True)
            _check("run_id" not in payload or payload["run_id"] == run_id)
            if "fact_end" in payload:
                _check(payload["fact_end"] == view["fact_ends"].get(run_id, 0))
            final = payload.get("final_state", {})
            _check(type(final) is dict and ("run_id" not in final or final["run_id"] == run_id))
            if "resource_view_id" in final:
                _check(final["resource_view_id"] == row[2])
            return {**payload, "view_id": row[2], "checkpoint_id": row[3]}

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
