"""Immutable raw snapshots and bounded character-based retrieval.

Only the in-memory index grants access; files are never discovered or imported.
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
import shutil
import stat
import tempfile
from threading import RLock
from typing import Any
from uuid import uuid4

from lineage_agent.tools.filesystem import paths
from lineage_agent.tools.types import serialize_tool_output


MAX_OUTPUT_BYTES = 8192
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
    "artifact_cleanup_error": "ArtifactStore 已关闭，但自身临时文件未能完整清理。",
    "invalid_artifact_ref": "Artifact 引用格式无效。",
    "invalid_artifact_range": "字符范围必须是正文内的有效左闭右开范围。",
    "invalid_artifact_keywords": "检索需要 1 至 8 个关键词，每个包含 1 至 128 个字符。",
    "invalid_artifact_result": "原始结果及来源工具必须是可编码的字符串，来源工具不能为空。",
    "invalid_artifact_session": "应用必须提供非空的会话标识。",
}


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


def _content(raw: str, tool_name: str) -> tuple[str, str, str | None]:
    if tool_name == "read_text_file":
        try:
            value = json.loads(raw)
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
    envelope = serialize_tool_output({"ok": True, "result": result})
    return len(envelope.encode("utf-8")) <= MAX_OUTPUT_BYTES


def _base(record: _Record) -> dict[str, Any]:
    result = {key: record.metadata[key] for key in ("artifact_ref", "view_kind", "total_chars")}
    if "degradation_reason" in record.metadata:
        result["degradation_reason"] = record.metadata["degradation_reason"]
    return result


def _index(value: object, length: int) -> bool:
    return type(value) is int and 0 <= value <= length


class ArtifactStore:
    """Application-owned temporary storage; call close or use a with block."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._closed = False
        self._records: dict[str, dict[str, _Record]] = {}
        self._root = paths.ARTIFACT_CACHE_ROOT
        try:
            _check_path(self._root.parent, directory=True)
            self._root.mkdir(exist_ok=True)
            _check_path(self._root, directory=True)
            self._directory = Path(tempfile.mkdtemp(prefix="store-", dir=self._root))
            _check_path(self._directory, directory=True)
        except Exception:
            raise ArtifactError("artifact_write_error") from None

    def _ensure_open(self) -> None:
        if self._closed:
            raise ArtifactError("artifact_store_closed")

    def bind_session(self, session_id: str) -> "SessionArtifacts":
        with self._lock:
            self._ensure_open()
            if type(session_id) is not str or not session_id:
                raise ArtifactError("invalid_artifact_session")
            self._records.setdefault(session_id, {})
            return SessionArtifacts(self, session_id)

    def _put(self, session_id: str, raw: str, tool_name: str) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            try:
                if type(raw) is not str or type(tool_name) is not str or not tool_name:
                    raise ValueError
                data = raw.encode("utf-8")
                tool_name.encode("utf-8")
            except (ValueError, UnicodeError):
                raise ArtifactError("invalid_artifact_result") from None
            content, view, reason = _content(raw, tool_name)
            identifier = uuid4().hex
            ref = f"artifact://tool/{identifier}"
            metadata = {
                "artifact_ref": ref, "tool_name": tool_name,
                "raw_bytes": len(data), "view_kind": view, "total_chars": len(content),
            }
            if reason is not None:
                metadata["degradation_reason"] = reason
            destination = self._directory / f"{identifier}.raw"
            temporary = self._directory / f"{identifier}.tmp"
            temporary_owned = False
            try:
                _check_path(self._directory, directory=True)
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
                record = _Record(destination, hashlib.sha256(data).digest(), metadata)
                self._records[session_id][ref] = record
            except Exception:
                # Never register an incomplete result, even if cleanup also fails.
                try:
                    if temporary_owned:
                        temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                raise ArtifactError("artifact_write_error") from None
            return deepcopy(metadata)

    def _load(self, session_id: str, ref: str) -> tuple[_Record, str]:
        self._ensure_open()
        if type(ref) is not str or _REF.fullmatch(ref) is None:
            raise ArtifactError("invalid_artifact_ref")
        record = self._records.get(session_id, {}).get(ref)
        if record is None:
            raise ArtifactError("artifact_not_found")
        try:
            _check_path(record.path, directory=False)
            data = record.path.read_bytes()
            if hashlib.sha256(data).digest() != record.sha256:
                raise ValueError("snapshot differs")
            raw = data.decode("utf-8")
            content, _, _ = _content(raw, record.metadata["tool_name"])
            return record, content
        except Exception:
            raise ArtifactError("artifact_read_error") from None

    def _read(self, session_id: str, ref: str, start: int, end: int | None) -> dict[str, Any]:
        with self._lock:
            record, content = self._load(session_id, ref)
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

    def _search(self, session_id: str, ref: str, keywords: list[str], start: int) -> dict[str, Any]:
        with self._lock:
            record, content = self._load(session_id, ref)
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
            try:
                # Keep a retryable cleanup path after failure, but never discover
                # or delete a different Store's directory or a replacement link.
                _check_path(self._root, directory=True)
                if self._directory.parent != self._root or not self._directory.name.startswith("store-"):
                    raise OSError("invalid cleanup target")
                try:
                    self._directory.lstat()
                except FileNotFoundError:
                    return
                _check_path(self._directory, directory=True)
                shutil.rmtree(self._directory)
            except Exception:
                raise ArtifactError("artifact_cleanup_error") from None

    def __enter__(self) -> "ArtifactStore":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class SessionArtifacts:
    """Trusted application handle; model tools never accept its session ID."""

    def __init__(self, store: ArtifactStore, session_id: str) -> None:
        self._store, self._session_id = store, session_id

    def put(self, raw_result: str, *, tool_name: str) -> dict[str, Any]:
        return self._store._put(self._session_id, raw_result, tool_name)

    def read(self, artifact_ref: str, start: int, end: int | None = None) -> dict[str, Any]:
        return self._store._read(self._session_id, artifact_ref, start, end)

    def search(self, artifact_ref: str, keywords: list[str], start: int = 0) -> dict[str, Any]:
        return self._store._search(self._session_id, artifact_ref, keywords, start)
