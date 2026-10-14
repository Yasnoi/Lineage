"""Immutable raw snapshots and bounded character-based retrieval.

Trusted handles and confirmed resource views grant access; files are never
discovered or implicitly imported.
The filesystem boundary assumes stable local paths, like the project tools. It
does not provide an OS sandbox against a hostile process replacing paths.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from threading import RLock
from typing import Any, NoReturn
from uuid import uuid4

from lineage_agent.archive import ArchiveStore
from lineage_agent.tools.filesystem import paths
from lineage_agent.storage import conversation_paths, ensure_storage_directory, storage_identifier
from lineage_agent.tools.types import (
    MAX_RECALL_BYTES, inline_tool_output, serialize_tool_output, tool_output_bytes,
)


MAX_OUTPUT_BYTES = MAX_RECALL_BYTES
MAX_KEYWORDS = 8
MAX_KEYWORD_CHARS = 128
MAX_HITS = 10
CONTEXT_CHARS = 80
_REF = re.compile(r"artifact://tool/([0-9a-f]{32})\Z")
_BOUNDARIES = frozenset("\r\n。！？；.!?;")
_ERRORS = {
    "artifact_write_error": "Artifact 保存失败，未发布引用。",
    "artifact_read_error": "Artifact 无法读取或快照完整性校验失败。",
    "artifact_not_found": "当前会话中不存在该 Artifact。",
    "artifact_store_closed": "ArtifactStore 已关闭，引用不再可用。",
    "artifact_cleanup_error": "ArtifactStore 未能完成关闭。",
    "invalid_artifact_ref": "Artifact 引用格式无效。",
    "invalid_artifact_range": "字符范围必须是正文内的有效左闭右开范围。",
    "invalid_artifact_keywords": "检索需要 1 至 8 个关键词，每个包含 1 至 128 个字符。",
    "invalid_artifact_result": "原始结果及来源工具必须是可编码的字符串，来源工具不能为空。",
    "invalid_artifact_session": "应用必须提供非空的会话标识。",
}
ARTIFACT_ERROR_CODES = frozenset(_ERRORS)


class ArtifactError(Exception):
    """A fixed application diagnostic, never an underlying exception message."""

    def __init__(self, code: str) -> None:
        self.code = code if code in _ERRORS else "artifact_read_error"
        super().__init__(_ERRORS[self.code])


@dataclass(frozen=True)
class _Record:
    path: Path
    sha256: bytes
    metadata: dict[str, Any]


def _check_path(path: Path, *, directory: bool) -> None:
    info = paths._check_components(path)
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise OSError("unexpected storage entry type")


def _reject_constant(value: str) -> NoReturn:
    raise ValueError("Non-standard JSON constants are not supported.")


def tool_result_view(raw: str, tool_name: str) -> tuple[str, str, str | None]:
    """Select the same text view for stored recall and in-memory previews."""
    if tool_name == "read_text_file":
        try:
            value = json.loads(raw, parse_constant=_reject_constant)
            if isinstance(value, dict) and isinstance(value.get("content"), str):
                # Escaped lone surrogates can occur in otherwise UTF-8 JSON.
                # They cannot be disclosed as valid UTF-8 content.
                value["content"].encode("utf-8")
                return value["content"], "content", None
        except (ValueError, UnicodeError, RecursionError):
            pass
        return raw, "raw_result", "invalid_text_result"
    return raw, "raw_result", None


def _fits(value: dict[str, Any]) -> bool:
    result = serialize_tool_output(value)
    return tool_output_bytes(inline_tool_output(result)) <= MAX_OUTPUT_BYTES


def _base(record: _Record) -> dict[str, Any]:
    result = {key: record.metadata[key] for key in ("artifact_ref", "view_kind", "total_chars")}
    if "degradation_reason" in record.metadata:
        result["degradation_reason"] = record.metadata["degradation_reason"]
    return result


def _index(value: object, length: int) -> bool:
    return type(value) is int and 0 <= value <= length


class ArtifactStore:
    """Persistent registrations with explicit handles and session isolation."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._closed = False
        self._records: dict[str, dict[str, _Record]] = {}
        self._application_root = Path.cwd()
        self._directories: dict[str, Path] = {}
        self._archives: dict[str, ArchiveStore] = {}
        self._owned_archives: dict[str, ArchiveStore] = {}
        self._write_failures: set[str] = set()

    @property
    def _directory(self) -> Path:
        # Compatibility for diagnostics that inspected the first binding. All
        # operations use their own binding's path, never this convenience view.
        if not self._directories:
            return self._application_root / ".lineage" / "unbound" / "artifacts"
        return next(iter(self._directories.values()))

    def _ensure_open(self) -> None:
        if self._closed:
            raise ArtifactError("artifact_store_closed")

    def bind_session(self, session_id: str, *, directory: Path | None = None) -> "SessionArtifacts":
        with self._lock:
            self._ensure_open()
            if type(session_id) is not str or not session_id:
                raise ArtifactError("invalid_artifact_session")
            try:
                if directory is None:
                    path = self._directories.get(session_id)
                    if path is None:
                        path = conversation_paths(self._application_root, str(uuid4())).artifacts
                else:
                    path = Path(directory).absolute()
                    if path.name != "artifacts" or path.parent.parent.name.casefold() != ".lineage":
                        raise ValueError("invalid artifact directory")
                    storage_identifier(path.parent.name)
                    path = ensure_storage_directory(path)
                if session_id in self._directories and self._directories[session_id] != path:
                    raise ValueError("session directory changed")
                if session_id not in self._archives:
                    archive = ArchiveStore(path.parent, create=True)
                    self._archives[session_id] = archive
                    self._owned_archives[session_id] = archive
                self._directories[session_id] = path
                self._records.setdefault(session_id, {})
            except Exception:
                raise ArtifactError("artifact_write_error") from None
            return SessionArtifacts(self, session_id)

    def restore_session(self, session_id: str, *, archive: ArchiveStore,
                        view_id: str) -> "SessionArtifacts":
        """Rebuild only registrations authorized by an explicitly supplied view.

        The archive is borrowed. Closing this store does not close that handle.
        Normal ``bind_session`` deliberately never performs this import.
        """
        with self._lock:
            self._ensure_open()
            if type(session_id) is not str or not session_id:
                raise ArtifactError("invalid_artifact_session")
            try:
                directory = Path(archive.directory).absolute() / "artifacts"
                if (directory.parent.parent.name.casefold() != ".lineage"
                        or session_id in self._directories):
                    raise ValueError("invalid or existing binding")
                storage_identifier(directory.parent.name)
                _check_path(directory, directory=True)
                view = archive.view(view_id)
                records = {}
                for ref in view["artifacts"]:
                    item = archive.artifact(ref, view_id=view_id)
                    records[ref] = self._record_from_archive(archive, item)
                self._directories[session_id] = directory
                self._archives[session_id] = archive
                self._records[session_id] = records
            except Exception:
                raise ArtifactError("artifact_read_error") from None
            return SessionArtifacts(self, session_id)

    @staticmethod
    def _record_from_archive(archive: ArchiveStore, item: dict[str, Any]) -> _Record:
        return _Record(Path(archive.directory) / item["path"],
                       bytes.fromhex(item["sha256"]), deepcopy(item["metadata"]))

    def _put(self, session_id: str, raw: str, tool_name: str,
             origin: dict[str, str] | None = None) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            if session_id in self._write_failures:
                raise ArtifactError("artifact_write_error")
            try:
                if type(raw) is not str or type(tool_name) is not str or not tool_name:
                    raise ValueError
                data = raw.encode("utf-8")
                tool_name.encode("utf-8")
            except (ValueError, UnicodeError):
                raise ArtifactError("invalid_artifact_result") from None
            content, view, reason = tool_result_view(raw, tool_name)
            identifier = uuid4().hex
            ref = f"artifact://tool/{identifier}"
            metadata = {
                "artifact_ref": ref, "tool_name": tool_name,
                "raw_bytes": len(data), "view_kind": view, "total_chars": len(content),
            }
            if reason is not None:
                metadata["degradation_reason"] = reason
            registered_metadata = deepcopy(metadata)
            if origin:
                registered_metadata["origin"] = deepcopy(origin)
            directory = self._directories[session_id]
            destination = directory / f"{identifier}.raw"
            temporary = directory / f"{identifier}.tmp"
            temporary_owned = False
            try:
                _check_path(directory, directory=True)
                # Fresh names and exclusive creation prevent replacing snapshots.
                if destination.exists() or temporary.exists():
                    raise OSError("storage name collision")
                with temporary.open("xb") as handle:
                    temporary_owned = True
                    if handle.write(data) != len(data):
                        raise OSError("short storage write")
                    handle.flush()
                    os.fsync(handle.fileno())
                # Windows rename refuses an existing destination. The explicit
                # precheck covers ordinary collisions on other supported hosts.
                temporary.rename(destination)
                _check_path(destination, directory=False)
                digest = hashlib.sha256(data).digest()
                archive = self._archives[session_id]
                archive.save_artifact({
                    "artifact_ref": ref,
                    "path": destination.relative_to(archive.directory).as_posix(),
                    "sha256": digest.hex(), "metadata": registered_metadata,
                })
                record = _Record(destination, digest, registered_metadata)
                self._records[session_id][ref] = record
            except Exception:
                self._write_failures.add(session_id)
                # Never register an incomplete result, even if cleanup also fails.
                try:
                    if temporary_owned:
                        temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                raise ArtifactError("artifact_write_error") from None
            return deepcopy(metadata)

    def _load(self, session_id: str, ref: str,
              view_id: str | None = None) -> tuple[_Record, str]:
        self._ensure_open()
        if type(ref) is not str or _REF.fullmatch(ref) is None:
            raise ArtifactError("invalid_artifact_ref")
        record = self._records.get(session_id, {}).get(ref)
        if record is None:
            raise ArtifactError("artifact_not_found")
        try:
            archive = self._archives[session_id]
            if view_id is not None and ref not in archive.view(view_id)["artifacts"]:
                raise ArtifactError("artifact_not_found")
            registered = archive.artifact(ref, view_id=view_id)
            if self._record_from_archive(archive, registered) != record:
                raise ValueError("registration differs")
            _check_path(record.path, directory=False)
            data = record.path.read_bytes()
            if hashlib.sha256(data).digest() != record.sha256:
                raise ValueError("snapshot differs")
            raw = data.decode("utf-8")
            content, _, _ = tool_result_view(raw, record.metadata["tool_name"])
            return record, content
        except ArtifactError:
            raise
        except Exception:
            raise ArtifactError("artifact_read_error") from None

    def _read(self, session_id: str, ref: str, start: int, end: int | None,
              view_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            record, content = self._load(session_id, ref, view_id)
            requested_end = len(content) if end is None else end
            if not (_index(start, len(content)) and _index(requested_end, len(content))
                    and start <= requested_end):
                raise ArtifactError("invalid_artifact_range")

            def response(stop: int) -> dict[str, Any]:
                return {**_base(record), "start": start, "end": stop,
                        "text": content[start:stop],
                        "next_start": stop if stop < requested_end else None}

            # No returned character takes fewer than one encoded byte. Limit the
            # search itself, not just the final string, for very large snapshots.
            upper = min(requested_end, start + MAX_OUTPUT_BYTES)
            if _fits(response(upper)):
                actual_end = upper
            else:
                low, high = start, upper
                while low < high:
                    middle = (low + high + 1) // 2
                    if _fits(response(middle)):
                        low = middle
                    else:
                        high = middle - 1
                actual_end = low
            if actual_end < requested_end:
                for position in range(actual_end - 1, max(start, actual_end - 128) - 1, -1):
                    if content[position] in _BOUNDARIES:
                        actual_end = position + 1
                        break
            result = response(actual_end)
            if not _fits(result) or (start < requested_end and actual_end == start):
                raise ArtifactError("artifact_read_error")
            return result

    def _search(self, session_id: str, ref: str, keywords: list[str], start: int,
                view_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            record, content = self._load(session_id, ref, view_id)
            if not _index(start, len(content)):
                raise ArtifactError("invalid_artifact_range")
            if (type(keywords) is not list or not 1 <= len(keywords) <= MAX_KEYWORDS
                    or any(type(word) is not str or not 1 <= len(word) <= MAX_KEYWORD_CHARS
                           for word in keywords)):
                raise ArtifactError("invalid_artifact_keywords")
            try:
                for word in keywords:
                    word.encode("utf-8")
            except UnicodeError:
                raise ArtifactError("invalid_artifact_keywords") from None
            result = {**_base(record), "hits": [], "has_more": False, "next_start": None}
            positions = [content.find(word, start) for word in keywords]
            last_start = None
            while any(position >= 0 for position in positions):
                match_start = min(position for position in positions if position >= 0)
                indices = [index for index, position in enumerate(positions) if position == match_start]
                matches = [{"keyword_index": index, "start": match_start,
                            "end": match_start + len(keywords[index])} for index in indices]
                match_end = max(match["end"] for match in matches)
                snippet_start, snippet_end = max(0, match_start - CONTEXT_CHARS), min(len(content), match_end + CONTEXT_CHARS)
                hit = {"matches": matches, "start": snippet_start, "end": snippet_end,
                       "text": content[snippet_start:snippet_end]}
                # Either final state must fit: false/null can be longer than
                # true/a small cursor. One look-ahead determines which is sent.
                candidate = {**result, "hits": result["hits"] + [hit],
                             "has_more": True, "next_start": match_start + 1}
                finished = {**candidate, "has_more": False, "next_start": None}
                if len(result["hits"]) == MAX_HITS or not (_fits(candidate) and _fits(finished)):
                    if last_start is None:
                        raise ArtifactError("artifact_read_error")
                    result.update(has_more=True, next_start=last_start + 1)
                    break
                result["hits"].append(hit)
                last_start = match_start
                for index in indices:
                    positions[index] = content.find(keywords[index], match_start + 1)
            if not _fits(result):
                raise ArtifactError("artifact_read_error")
            return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._records.clear()
            failed = False
            for archive in self._owned_archives.values():
                try:
                    archive.close()
                except Exception:
                    failed = True
            self._owned_archives.clear()
            self._archives.clear()
            # Published files belong to the conversation, not the handle. No
            # implicit import, deletion, or cross-process recovery occurs here.
            if failed:
                raise ArtifactError("artifact_cleanup_error")

    def __enter__(self) -> "ArtifactStore":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class SessionArtifacts:
    """Trusted application handle; model tools never accept its session ID."""

    def __init__(self, store: ArtifactStore, session_id: str,
                 origin: dict[str, str] | None = None) -> None:
        self._store, self._session_id = store, session_id
        self._origin = origin

    def bind_origin(self, run_id: str) -> "SessionArtifacts":
        """Return a per-run writer without changing the shared session handle."""
        storage_identifier(run_id)
        return SessionArtifacts(self._store, self._session_id, {"run_id": run_id})

    def registered_refs(self) -> list[str]:
        with self._store._lock:
            self._store._ensure_open()
            return list(self._store._records.get(self._session_id, {}))

    def persistence_error(self) -> str | None:
        """Report a real body/registration failure even if preview policy caught it."""
        with self._store._lock:
            return "artifact_write_error" if self._session_id in self._store._write_failures else None

    def check_persistence(self) -> None:
        if code := self.persistence_error():
            raise ArtifactError(code)

    def put(self, raw_result: str, *, tool_name: str) -> dict[str, Any]:
        return self._store._put(self._session_id, raw_result, tool_name, self._origin)

    def read(self, artifact_ref: str, start: int, end: int | None = None, *,
             view_id: str | None = None) -> dict[str, Any]:
        return self._store._read(self._session_id, artifact_ref, start, end, view_id)

    def search(self, artifact_ref: str, keywords: list[str], start: int = 0, *,
               view_id: str | None = None) -> dict[str, Any]:
        return self._store._search(self._session_id, artifact_ref, keywords, start, view_id)
