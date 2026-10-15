"""Immutable business objects and explicitly confirmed resource views.

The database transaction covers only these registrations. File bodies are
written first, and a resource view is confirmed only after the caller's state
save succeeds. Opening this database does not open or resume an Agent session.
"""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
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
    "archive_publication_unknown": "无法确认对话快照是否已经发布；请重新打开并核验会话。",
    "archive_publication_conflict": "对话恢复点或未完成轮次已经变化，本次操作未发布。",
    "archive_no_snapshot": "该会话尚无已发布的恢复点。",
    "archive_unregistered": "该归档尚未登记为可恢复会话。",
    "archive_no_execution": "该轮次尚无已确认的执行恢复点。",
    "archive_attempt_limit": "该操作的执行尝试额度已耗尽，不能再次派发。",
}
FORMAT_VERSION = 3
RECOVERY_VERSION = 2
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
CREATE TABLE conversation (id INTEGER PRIMARY KEY CHECK(id=1), payload BLOB NOT NULL,
                           sha256 TEXT NOT NULL, current_snapshot_id TEXT, updated_at TEXT NOT NULL);
CREATE TABLE snapshots (id TEXT PRIMARY KEY REFERENCES runs(id), sequence INTEGER NOT NULL UNIQUE,
                        parent_id TEXT REFERENCES snapshots(id), run_id TEXT NOT NULL UNIQUE,
                        view_id TEXT NOT NULL REFERENCES views(id), payload BLOB NOT NULL, sha256 TEXT NOT NULL);
CREATE TABLE run_lifecycle (run_id TEXT PRIMARY KEY, source_snapshot_id TEXT REFERENCES snapshots(id),
                            status TEXT NOT NULL CHECK(status IN ('started','published','abandoned','rejected')),
                            started_at TEXT NOT NULL, completed_at TEXT, snapshot_id TEXT REFERENCES snapshots(id),
                            choice_id TEXT, current_execution_id TEXT,
                            revision INTEGER NOT NULL DEFAULT 0);
CREATE TABLE recovery_choices (id TEXT PRIMARY KEY, payload BLOB NOT NULL, sha256 TEXT NOT NULL);
CREATE TABLE execution_points (id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES run_lifecycle(run_id),
                              sequence INTEGER NOT NULL, parent_id TEXT REFERENCES execution_points(id),
                              view_id TEXT NOT NULL REFERENCES views(id), payload BLOB NOT NULL, sha256 TEXT NOT NULL,
                              confirmed INTEGER NOT NULL DEFAULT 0 CHECK(confirmed IN (0,1)), checkpoint_id TEXT);
CREATE UNIQUE INDEX confirmed_execution_sequence ON execution_points(run_id,sequence) WHERE confirmed=1;
CREATE TABLE operations (id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES run_lifecycle(run_id),
                         sequence INTEGER NOT NULL, kind TEXT NOT NULL, payload BLOB NOT NULL, sha256 TEXT NOT NULL,
                         UNIQUE(run_id,sequence));
CREATE TABLE attempts (id TEXT PRIMARY KEY, operation_id TEXT NOT NULL REFERENCES operations(id),
                       sequence INTEGER NOT NULL, payload BLOB NOT NULL, sha256 TEXT NOT NULL,
                       result BLOB, result_sha256 TEXT, UNIQUE(operation_id,sequence));
CREATE TABLE operation_results (operation_id TEXT NOT NULL REFERENCES operations(id), slot TEXT NOT NULL,
                               payload BLOB NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(operation_id,slot));
"""


class ArchiveError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code if code in _ERRORS else "archive_integrity_error"
        super().__init__(_ERRORS[self.code])


def _check(condition: bool) -> None:
    if not condition:
        raise ArchiveError("archive_integrity_error")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conflict(condition: bool) -> None:
    if not condition:
        raise ArchiveError("archive_publication_conflict")


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


from lineage_agent.archive.execution import ExecutionArchiveMixin


class ArchiveStore(ExecutionArchiveMixin):
    """Trusted conversation-directory binding; no directory discovery or import."""

    def __init__(self, directory: Path, *, create: bool = True, read_only: bool = False) -> None:
        self._lock = RLock()
        self._closed = False
        self._operation_depth = 0
        self._validated_files: set[tuple[str, bytes]] = set()
        self._validated_file_data: dict[tuple[str, bytes], bytes] = {}
        self._validated_artifacts: set[bytes] = set()
        self._validated_views: dict[tuple[str, bool], dict] = {}
        self._validated_scopes: set[bytes] = set()
        self._validated_summaries: set[bytes] = set()
        self._connection: sqlite3.Connection | None = None
        self.read_only = read_only
        try:
            _check(not (create and read_only))
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
            uri = self.path.as_uri() + ("?mode=ro" if read_only else "?mode=rw" if existed else "?mode=rwc")
            self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA synchronous=FULL")
            if not existed:
                with self._connection:
                    self._connection.executescript(_SCHEMA)
                    data, digest = _encoded({"schema_version": FORMAT_VERSION, "conversation_id": self.conversation_id,
                                             "cursor_key": os.urandom(32)})
                    self._connection.execute("INSERT INTO metadata VALUES ('identity',?,?)", (data, digest))
                    self._connection.execute(f"PRAGMA user_version={FORMAT_VERSION}")
            version = self._connection.execute("PRAGMA user_version").fetchone()[0]
            if version != FORMAT_VERSION:
                raise ArchiveError("archive_version_error")
            row = self._connection.execute("SELECT payload,sha256 FROM metadata WHERE key='identity'").fetchone()
            _check(row is not None)
            identity = _decoded(*row)
            _check(identity.get("schema_version") == FORMAT_VERSION and identity.get("conversation_id") == self.conversation_id
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
    def _operation(self, *, write: bool = False, immediate: bool = False):
        with self._lock:
            outermost = self._operation_depth == 0
            if outermost:
                self._validated_files.clear()
                self._validated_file_data.clear()
                self._validated_artifacts.clear()
                self._validated_views.clear()
                self._validated_scopes.clear()
                self._validated_summaries.clear()
            self._operation_depth += 1
            try:
                if self._closed or self._connection is None:
                    raise ArchiveError("archive_write_error" if write else "archive_read_error")
                if write and self.read_only:
                    raise ArchiveError("archive_write_error")
                self._check_paths()
                if write:
                    try:
                        if immediate:
                            self._connection.execute("BEGIN IMMEDIATE")
                        yield self._connection
                        self._commit()
                    except BaseException:
                        self._connection.rollback()
                        raise
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
                    self._validated_file_data.clear()
                    self._validated_artifacts.clear()
                    self._validated_views.clear()
                    self._validated_scopes.clear()
                    self._validated_summaries.clear()

    def _commit(self) -> None:
        """One commit boundary, also usable for failure-injection tests."""
        self._connection.commit()

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
        pending = []
        for run_id, watermark in files.items():
            storage_identifier(run_id)
            if paths is not None:
                _check(paths.get(run_id) == f"history/{run_id}.jsonl")
            identity = (run_id, encode_record(watermark))
            if identity not in self._validated_files:
                pending.append((run_id, watermark, identity))
        if not pending:
            return
        _check(stat.S_ISDIR(_check_components(self.directory / "history").st_mode))
        journal = RecordJournal(self.directory / "history", create=False)
        try:
            for run_id, watermark, identity in pending:
                for path in journal.paths(run_id, watermark):
                    self._path(path.relative_to(self.directory).as_posix())
                data, _ = journal._index(run_id, watermark)
                self._validated_files.add(identity)
                self._validated_file_data[identity] = data
        finally:
            journal.close()

    def _file_extension(self, run_id: str, previous: dict, current: dict) -> None:
        """Compare two freshly verified registrations without rereading bodies.

        Identical segment paths mean their verified byte prefixes are the same
        physical prefix. All preceding segments remain frozen; only the last
        old segment may grow. Validation is deduplicated solely in this call.
        """
        self._files({run_id: previous})
        self._files({run_id: current})

        def segments(watermark):
            return watermark.get("segments", [{"path": f"history/{run_id}.jsonl", "start_sequence": 0,
                **{key: watermark[key] for key in ("record_count", "byte_end", "prefix_sha256")}}])

        before, after = segments(previous), segments(current)
        _check(len(before) <= len(after) and previous["record_count"] <= current["record_count"]
               and previous["byte_end"] <= current["byte_end"])
        _check(self._validated_file_data[run_id, encode_record(current)].startswith(
            self._validated_file_data[run_id, encode_record(previous)]))
        for index, old in enumerate(before):
            new = after[index]
            _check(old["path"] == new["path"] and old["start_sequence"] == new["start_sequence"]
                   and old["record_count"] <= new["record_count"] and old["byte_end"] <= new["byte_end"]
                   and (index == len(before) - 1 or old == new))

    def _scope(self, record: dict[str, Any]) -> None:
        identity = encode_record(record)
        if identity in self._validated_scopes:
            return
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
        self._validated_scopes.add(identity)

    def _summary(self, record: dict[str, Any]) -> None:
        identity = encode_record(record)
        if identity in self._validated_summaries:
            return
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
        self._validated_summaries.add(encode_record(record))

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
        from lineage_agent.artifacts.store import decode_artifact_raw, tool_result_view
        text, kind, reason = tool_result_view(decode_artifact_raw(data, metadata), metadata["tool_name"])
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
        cache_key = (identity, confirmed)
        if cache_key in self._validated_views:
            return self._validated_views[cache_key]
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
                _check(run in view["record_files"])
                self._file_extension(run, watermark, view["record_files"][run])
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
        self._validated_views[cache_key] = view
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
                    _check(run in record_files)
                    self._file_extension(run, watermark, record_files[run])
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
            self._confirm_view_rows(identity, checkpoint_id)

    def _confirm_view_rows(self, identity: str, checkpoint_id: str) -> None:
        """Confirm within the caller's transaction, including execution commits."""
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

    def stage_run(self, run_id: str, payload: dict, *, record_id: str | None = None) -> str:
        identity = record_id or str(uuid4())

        def existing():
            row = self._connection.execute("SELECT run_id,payload,sha256 FROM runs WHERE id=?", (identity,)).fetchone()
            if row is not None:
                _decoded(row[1], row[2])
                _check(row == (run_id, *_encoded(payload)))
                return identity
            return None

        def write():
            storage_identifier(run_id)
            _check("run_id" not in payload or payload["run_id"] == run_id)
            storage_identifier(identity)
            data, digest = _encoded(payload)
            if saved := existing():
                return saved
            self._connection.execute("INSERT INTO runs(id,run_id,payload,sha256) VALUES (?,?,?,?)", (identity, run_id, data, digest))
            return identity

        return self._execution_transaction(write, existing)

    def seal_run(self, record_id: str, view_id: str, checkpoint_id: str) -> None:
        with self._operation(write=True):
            self._seal_run_rows(record_id, view_id, checkpoint_id)

    def _seal_run_rows(self, record_id: str, view_id: str, checkpoint_id: str) -> tuple[str, dict, str]:
        """Use the caller's transaction; never commit during snapshot publication."""
        view = self._verify_view(view_id, True)
        row = self._connection.execute("SELECT run_id,payload,sha256,sealed,view_id,checkpoint_id FROM runs WHERE id=?", (record_id,)).fetchone()
        _check(row is not None)
        payload = _decoded(row[1], row[2])
        mapping = self._connection.execute("SELECT view_id FROM checkpoints WHERE id=?", (checkpoint_id,)).fetchone()
        _check(mapping is not None and mapping[0] == view_id)
        if row[3]:
            _check(row[4:] == (view_id, checkpoint_id))
        else:
            if "fact_end" in payload:
                _check(payload["fact_end"] == view["fact_ends"].get(row[0], 0))
                self._events(row[0], payload["fact_end"])
            self._connection.execute("UPDATE runs SET sealed=1,view_id=?,checkpoint_id=? WHERE id=?", (view_id, checkpoint_id, record_id))
        return row[0], payload, row[2]

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

    def staged_run(self, record_id: str) -> dict[str, Any]:
        """Read one explicitly selected final candidate, without treating it as sealed."""
        with self._operation():
            row = self._connection.execute(
                "SELECT run_id,payload,sha256,sealed,view_id,checkpoint_id FROM runs WHERE id=?", (record_id,)).fetchone()
            _check(row is not None)
            payload = _decoded(row[1], row[2])
            _check("run_id" not in payload or payload["run_id"] == row[0])
            final = payload.get("final_state")
            _check(type(final) is dict and final.get("run_id", row[0]) == row[0])
            if view_id := final.get("resource_view_id"):
                view = self._verify_view(view_id, False)
                _check(final.get("record_files", {}) == view["record_files"]
                       and final.get("history_scopes", {}) == view["history_scopes"])
                if row[3]:
                    _check(row[4] == view_id)
            return {"record_id": record_id, "run_id": row[0], "payload": payload,
                    "sealed": bool(row[3]), "view_id": row[4], "checkpoint_id": row[5]}

    def register_conversation(self, configuration: dict, tools: list[dict]) -> dict[str, Any]:
        """Register the caller's allowlisted, non-secret runtime descriptor once."""
        with self._operation(write=True, immediate=True):
            _check(type(configuration) is dict and type(tools) is list and all(type(tool) is dict for tool in tools))
            row = self._connection.execute("SELECT payload,sha256 FROM conversation WHERE id=1").fetchone()
            if row is not None:
                value = _decoded(*row)
                _check(value.get("configuration") == configuration and value.get("tools") == tools)
                return self._conversation()
            now = _now()
            value = {"conversation_id": self.conversation_id, "configuration": deepcopy(configuration),
                     "tools": deepcopy(tools), "format_version": FORMAT_VERSION,
                     "recovery_version": RECOVERY_VERSION, "created_at": now}
            data, digest = _encoded(value)
            self._connection.execute("INSERT INTO conversation VALUES (1,?,?,NULL,?)", (data, digest, now))
            return {**value, "current_snapshot_id": None, "updated_at": now}

    def _conversation(self) -> dict[str, Any]:
        # Read the explicit pointer and row count in one SQLite statement, so
        # read-only catalogues cannot observe different publication commits.
        row = self._connection.execute(
            "SELECT payload,sha256,current_snapshot_id,updated_at,(SELECT COUNT(*) FROM snapshots) "
            "FROM conversation WHERE id=1").fetchone()
        if row is None:
            raise ArchiveError("archive_unregistered")
        value = _decoded(row[0], row[1])
        _check(value.get("conversation_id") == self.conversation_id and value.get("format_version") == FORMAT_VERSION
               and type(value.get("configuration")) is dict and type(value.get("tools")) is list)
        if value.get("recovery_version") != RECOVERY_VERSION:
            raise ArchiveError("archive_version_error")
        if row[4] == 0:
            _check(row[2] is None)
        else:
            _check(type(row[2]) is str and len(self._snapshot_chain(row[2])) == row[4])
        return {**value, "current_snapshot_id": row[2], "updated_at": row[3]}

    def conversation(self) -> dict[str, Any]:
        with self._operation():
            return self._conversation()

    def _lifecycle(self, run_id: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT run_id,source_snapshot_id,status,started_at,completed_at,snapshot_id,choice_id,current_execution_id,revision "
                                       "FROM run_lifecycle WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        return dict(zip(("run_id", "source_snapshot_id", "status", "started_at", "completed_at", "snapshot_id", "choice_id", "current_execution_id", "revision"), row))

    def _unfinished_runs(self) -> list[dict[str, Any]]:
        rows = self._connection.execute("SELECT run_id FROM run_lifecycle WHERE status='started' ORDER BY started_at,run_id").fetchall()
        return [self._lifecycle(row[0]) for row in rows]

    def unfinished_runs(self) -> list[dict[str, Any]]:
        with self._operation():
            self._conversation()
            return self._unfinished_runs()

    def begin_run(self, run_id: str, source_snapshot_id: str | None) -> dict[str, Any]:
        """Durably identify an attempt before its first body or business call."""
        with self._operation(write=True, immediate=True):
            storage_identifier(run_id)
            current = self._conversation()["current_snapshot_id"]
            _conflict(current == source_snapshot_id)
            existing = self._lifecycle(run_id)
            if existing is not None:
                _conflict(existing["status"] == "started" and existing["source_snapshot_id"] == source_snapshot_id)
                return existing
            _conflict(not self._unfinished_runs())
            if source_snapshot_id is not None:
                self._snapshot_chain(source_snapshot_id)
            self._connection.execute("INSERT INTO run_lifecycle(run_id,source_snapshot_id,status,started_at) VALUES (?,?,'started',?)",
                                     (run_id, source_snapshot_id, _now()))
            return self._lifecycle(run_id)

    def reject_run(self, run_id: str) -> None:
        """Close a pre-execution configuration rejection, never a paid attempt."""
        with self._operation(write=True, immediate=True):
            run = self._lifecycle(run_id)
            _check(run is not None)
            if run["status"] == "rejected":
                return
            _conflict(run["status"] == "started")
            _check(not self._events(run_id, None))
            _check(self._connection.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone() is None)
            _check(self._connection.execute("SELECT 1 FROM operations WHERE run_id=?", (run_id,)).fetchone() is None)
            self._connection.execute("UPDATE run_lifecycle SET status='rejected',completed_at=?,revision=revision+1 WHERE run_id=?", (_now(), run_id))

    def accept_recovery(self, snapshot_id: str, unfinished_run_ids: list[str]) -> str:
        """Record an explicit choice of the exact current snapshot and pending set."""
        with self._operation(write=True, immediate=True):
            _check(type(unfinished_run_ids) is list and bool(unfinished_run_ids)
                   and all(type(run) is str for run in unfinished_run_ids)
                   and len(unfinished_run_ids) == len(set(unfinished_run_ids)))
            current = self._conversation()["current_snapshot_id"]
            if current is None:
                raise ArchiveError("archive_no_snapshot")
            _conflict(snapshot_id == current)
            _conflict(set(unfinished_run_ids) == {run["run_id"] for run in self._unfinished_runs()})
            self._read_snapshot(snapshot_id)
            identity, now = str(uuid4()), _now()
            data, digest = _encoded({"choice_id": identity, "snapshot_id": snapshot_id,
                                     "unfinished_run_ids": sorted(unfinished_run_ids), "created_at": now})
            self._connection.execute("INSERT INTO recovery_choices VALUES (?,?,?)", (identity, data, digest))
            self._connection.executemany("UPDATE run_lifecycle SET status='abandoned',completed_at=?,choice_id=?,revision=revision+1 WHERE run_id=?",
                                         [(now, identity, run) for run in unfinished_run_ids])
            return identity

    def recovery_choices(self) -> list[dict[str, Any]]:
        with self._operation():
            return [_decoded(data, digest) for data, digest in self._connection.execute(
                "SELECT payload,sha256 FROM recovery_choices ORDER BY rowid").fetchall()]

    def _snapshot_meta(self, snapshot_id: str) -> dict[str, Any]:
        row = self._connection.execute("SELECT sequence,parent_id,run_id,view_id,payload,sha256 FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
        _check(row is not None)
        value = _decoded(row[4], row[5])
        _check(value.get("snapshot_id") == snapshot_id and value.get("conversation_id") == self.conversation_id
               and (value.get("sequence"), value.get("parent_id"), value.get("run_id"), value.get("view_id")) == row[:4]
               and type(row[0]) is int and row[0] > 0)
        if value.get("recovery_version") != RECOVERY_VERSION:
            raise ArchiveError("archive_version_error")
        return value

    def _snapshot_chain(self, snapshot_id: str) -> list[dict[str, Any]]:
        chain, seen = [], set()
        expected = None
        while snapshot_id is not None:
            _check(snapshot_id not in seen)
            seen.add(snapshot_id)
            current = self._snapshot_meta(snapshot_id)
            _check(expected is None or current["sequence"] == expected)
            lifecycle = self._lifecycle(current["run_id"])
            _check(lifecycle is not None and lifecycle["status"] == "published"
                   and lifecycle["snapshot_id"] == snapshot_id and lifecycle["source_snapshot_id"] == current["parent_id"])
            chain.append(current)
            expected = current["sequence"] - 1
            snapshot_id = current["parent_id"]
        _check(expected == 0)
        return list(reversed(chain))

    def _validate_snapshot_state(self, payload: dict, run_id: str, view_id: str) -> dict[str, Any]:
        view = self._verify_view(view_id, True)
        final = payload.get("final_state")
        _check(type(final) is dict and final.get("run_id") == run_id
               and final.get("conversation_id") == self.conversation_id and final.get("resource_view_id") == view_id
               and final.get("record_files") == view["record_files"] and final.get("history_scopes") == view["history_scopes"]
               and final.get("stop_reason") in {"completed", "max_model_calls", "context_budget_exceeded", "error"}
               and type(payload.get("fact_end")) is int and payload["fact_end"] == view["fact_ends"].get(run_id, 0))
        items, boundary, active = final.get("input"), final.get("history_end"), final.get("active_context")
        _check(type(items) is list and type(boundary) is int and 0 <= boundary <= len(items)
               and type(active) is dict and type(active.get("turn_start")) is int and 0 <= active["turn_start"] <= boundary
               and type(final.get("context_risks")) is list and len(final["context_risks"]) == len(items))
        # The unadmitted tail may intentionally contain unmatched tool calls.
        # Existing runtime validators define only the inheritable prefix here.
        from lineage_agent.agent.runtime import _input_association_error, _input_shape_error
        _check(all(type(item) is dict for item in items[:boundary])
               and _input_shape_error(items[:boundary]) is None and _input_association_error(items[:boundary]) is None)
        restored = deepcopy(final)
        restored["record_files"] = {run: deepcopy(view["record_files"][run]) for run in view["record_order"]}
        return restored

    def _validate_snapshot_lineage(self, run_id: str, view_id: str, parent_id: str | None) -> None:
        """A recovery point may authorize only its published ancestors and Run."""
        ancestors = self._snapshot_chain(parent_id) if parent_id is not None else []
        allowed = {item["run_id"] for item in ancestors} | {run_id}
        view = self._view(view_id)
        _check(set(view["record_files"]).issubset(allowed) and set(view["fact_ends"]).issubset(allowed))
        source_view = ancestors[-1]["view_id"] if ancestors else None
        if source_view is not None:
            parent = self._view(source_view)
            _check(view["record_order"][:len(parent["record_order"])] == parent["record_order"]
                   and all(view["record_files"].get(run) == watermark for run, watermark in parent["record_files"].items())
                   and all(view["fact_ends"].get(run) == end for run, end in parent["fact_ends"].items())
                   and all(view["history_scopes"].get(ref) == scope for ref, scope in parent["history_scopes"].items())
                   and all(set(parent[field]).issubset(view[field]) for field in ("summaries", "artifacts")))
        seen, cursor = set(), view_id
        while cursor != source_view:
            _check(cursor is not None and cursor not in seen)
            seen.add(cursor)
            cursor = self._view(cursor)["parent_id"]
        for identity in view["summaries"]:
            _check(self._object("summary", identity)["run_id"] in allowed)
        for ref in view["artifacts"]:
            origin = self._object("artifact", ref)["metadata"].get("origin", {})
            _check(type(origin) is dict and (origin.get("run_id") is None or origin["run_id"] in allowed))

    def _read_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        chain = self._snapshot_chain(snapshot_id)
        current = chain[-1]
        row = self._connection.execute("SELECT run_id,payload,sha256,sealed,view_id,checkpoint_id FROM runs WHERE id=?", (snapshot_id,)).fetchone()
        _check(row is not None and row[0] == current["run_id"] and row[2] == current["run_sha256"]
               and row[3] == 1 and row[4] == current["view_id"])
        payload = _decoded(row[1], row[2])
        restored = self._validate_snapshot_state(payload, row[0], row[4])
        self._validate_snapshot_lineage(row[0], row[4], current["parent_id"])
        return {**current, "final_state": restored, "fact_end": payload["fact_end"], "checkpoint_id": row[5]}

    def read_snapshot(self, snapshot_id: str | None = None) -> dict[str, Any]:
        with self._operation():
            identity = snapshot_id or self._conversation()["current_snapshot_id"]
            if identity is None:
                raise ArchiveError("archive_no_snapshot")
            return self._read_snapshot(identity)

    def published_runs(self, snapshot_id: str) -> list[dict[str, Any]]:
        with self._operation():
            result = []
            for snapshot in self._snapshot_chain(snapshot_id):
                saved = self._read_snapshot(snapshot["snapshot_id"])
                result.append({key: saved[key] for key in ("run_id", "checkpoint_id", "snapshot_id", "view_id", "sequence")})
            return result

    def metadata(self) -> dict[str, Any]:
        """Catalogue data only: never reconstruct or authorize body resources."""
        with self._operation():
            conversation = self._conversation()
            head = conversation["current_snapshot_id"]
            unfinished = self._unfinished_runs()
            for run in unfinished:
                point = self._connection.execute(
                    "SELECT payload,sha256 FROM execution_points WHERE id=? AND run_id=? AND confirmed=1",
                    (run["current_execution_id"], run["run_id"]),
                ).fetchone() if run["current_execution_id"] else None
                _check(run["current_execution_id"] is None or point is not None)
                value = _decoded(*point) if point is not None else None
                _check(value is None or value.get("point_id") == run["current_execution_id"]
                       and value.get("run_id") == run["run_id"] and type(value.get("phase")) is str)
                run["execution_phase"] = value["phase"] if value is not None else None
            return {**conversation, "head": self._snapshot_meta(head) if head is not None else None,
                    "unfinished_runs": unfinished}

    def _published_candidate(self, record_id: str, view_id: str, checkpoint_id: str,
                             expected_snapshot_id: str | None) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT 1 FROM snapshots WHERE id=?", (record_id,)).fetchone()
        if row is None:
            return None
        snapshot = self._read_snapshot(record_id)
        _check(snapshot["view_id"] == view_id and snapshot["checkpoint_id"] == checkpoint_id
               and snapshot["parent_id"] == expected_snapshot_id)
        head = self._conversation()["current_snapshot_id"]
        _check(head is not None and record_id in {item["snapshot_id"] for item in self._snapshot_chain(head)})
        return snapshot

    def publish_snapshot(self, record_id: str, view_id: str, checkpoint_id: str,
                         expected_snapshot_id: str | None) -> dict[str, Any]:
        """Atomically seal a final Run and publish its immutable recovery point."""
        try:
            with self._operation(write=True, immediate=True):
                if existing := self._published_candidate(record_id, view_id, checkpoint_id, expected_snapshot_id):
                    return existing
                conversation = self._conversation()
                _conflict(conversation["current_snapshot_id"] == expected_snapshot_id)
                row = self._connection.execute("SELECT run_id,payload,sha256 FROM runs WHERE id=?", (record_id,)).fetchone()
                _check(row is not None)
                lifecycle = self._lifecycle(row[0])
                _conflict(lifecycle is not None and lifecycle["status"] == "started"
                          and lifecycle["source_snapshot_id"] == expected_snapshot_id)
                payload = _decoded(row[1], row[2])
                self._validate_snapshot_state(payload, row[0], view_id)
                self._validate_snapshot_lineage(row[0], view_id, expected_snapshot_id)
                sequence = self._snapshot_meta(expected_snapshot_id)["sequence"] + 1 if expected_snapshot_id else 1
                now = _now()
                snapshot = {"snapshot_id": record_id, "conversation_id": self.conversation_id,
                            "run_id": row[0], "view_id": view_id, "parent_id": expected_snapshot_id,
                            "sequence": sequence, "recovery_version": RECOVERY_VERSION,
                            "created_at": now, "run_sha256": row[2]}
                data, digest = _encoded(snapshot)
                self._seal_run_rows(record_id, view_id, checkpoint_id)
                self._connection.execute("INSERT INTO snapshots VALUES (?,?,?,?,?,?,?)",
                                         (record_id, sequence, expected_snapshot_id, row[0], view_id, data, digest))
                self._connection.execute("UPDATE conversation SET current_snapshot_id=?,updated_at=? WHERE id=1", (record_id, now))
                self._connection.execute("UPDATE run_lifecycle SET status='published',completed_at=?,snapshot_id=?,revision=revision+1 WHERE run_id=?",
                                         (now, record_id, row[0]))
                result = self._read_snapshot(record_id)
            return result
        except ArchiveError as error:
            if error.code not in {"archive_write_error", "archive_read_error"}:
                raise
            # A COMMIT error does not prove rollback. Resolve the same immutable
            # ID with a fresh connection; never retry a model or mint another ID.
            confirmed = self.resolve_publication(record_id, view_id, checkpoint_id, expected_snapshot_id)
            if confirmed is not None:
                return confirmed
            raise error

    def resolve_publication(self, record_id: str, view_id: str, checkpoint_id: str,
                            expected_snapshot_id: str | None) -> dict[str, Any] | None:
        with self._lock:
            try:
                _check(self._operation_depth == 0 and not self._closed)
                # Closing also discards any transaction whose rollback itself
                # failed. The replacement cannot observe connection-local data.
                self._connection.close()
                self._connection = None
                replacement = ArchiveStore(self.directory, create=False, read_only=self.read_only)
                self._connection = replacement._connection
                replacement._connection = None
                replacement._closed = True
                with self._operation():
                    result = self._published_candidate(record_id, view_id, checkpoint_id, expected_snapshot_id)
                    if result is None:
                        _check(self._conversation()["current_snapshot_id"] == expected_snapshot_id)
                    return result
            except Exception:
                if self._connection is None:
                    self._closed = True
                raise ArchiveError("archive_publication_unknown") from None

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
