"""Immutable interaction records addressed independently of active model input."""

import base64
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import hmac
import json
import re
from typing import Any
from uuid import uuid4

from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.record_journal import JournalError, RecordJournal
from lineage_agent.archive import ArchiveError, ArchiveStore
from lineage_agent.tools.types import (
    MAX_RECALL_BYTES, ToolArgumentError, ToolExecutionError, inline_tool_output,
    serialize_tool_output, tool_output_bytes,
)


_SOURCE = re.compile(r"history://record/([A-Za-z0-9_-]{1,128})/(0|[1-9][0-9]*)\Z")
_SCOPE = re.compile(r"history://scope/[0-9a-f-]{36}\Z")
_TEXT_TYPES = {"input_text", "output_text", "refusal", "reasoning_text", "summary_text", "text"}


def source_ref(run_id: str, sequence: int) -> str:
    ref = f"history://record/{run_id}/{sequence}"
    parse_source_ref(ref)
    return ref


def parse_source_ref(value: str) -> tuple[str, int]:
    match = _SOURCE.fullmatch(value) if type(value) is str and len(value) <= 200 else None
    if match is None:
        raise ValueError("invalid original record reference")
    return match[1], int(match[2])


def make_record_batch(
    run_id: str, start_sequence: int, items: list[dict[str, Any]], *,
    group_id: Any = None, kind: str = "input", call_facts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Allocate immutable identities before saving; never fabricate protocol items.

    Outcomes and executions without a result still get original records. Their
    ``item`` is None, so callers must filter it when rebuilding protocol history.
    """
    if type(start_sequence) is not int or start_sequence < 0:
        raise ValueError("invalid original record sequence")
    facts = {fact["call_id"]: fact for fact in call_facts or [] if type(fact.get("call_id")) is str}
    batch: list[dict[str, Any]] = []
    emitted: set[str] = set()

    def append(item: dict[str, Any] | None, item_kind: str, *,
               fact: dict[str, Any] | None = None, outcome: dict[str, Any] | None = None) -> None:
        sequence = start_sequence + len(batch)
        call_id = item.get("call_id") if item is not None else (fact or {}).get("call_id")
        # Actual protocol output already holds the observed body. Keep execution
        # facts beside it without storing a second copy of that body.
        execution = {key: deepcopy(value) for key, value in fact.items() if key != "output"} if fact else None
        batch.append({
            "source_ref": source_ref(run_id, sequence), "run_id": run_id,
            "sequence": sequence, "group_id": deepcopy(group_id), "kind": item_kind,
            "item": deepcopy(item), "call_id": call_id, "execution": execution,
            **({"outcome": deepcopy(outcome)} if outcome is not None else {}),
        })

    for item in items:
        if kind == "outcome":
            append(None, kind, outcome=item)
        else:
            call_id = item.get("call_id")
            append(item, kind, fact=facts.get(call_id) if type(call_id) is str else None)
            if type(call_id) is str:
                emitted.add(call_id)
    for fact in call_facts or []:
        if type(fact.get("call_id")) is not str or fact["call_id"] not in emitted:
            append(None, "execution", fact=fact)
    return batch


def _text_content(value: Any) -> tuple[Any, bool]:
    if type(value) is str:
        return value, False
    if not isinstance(value, list):
        return None, value is not None
    parts = []
    omitted = False
    for part in value:
        if not isinstance(part, dict) or part.get("type") not in _TEXT_TYPES:
            omitted = True
            continue
        clean = {key: part[key] for key in ("type", "text", "refusal")
                 if key in part and type(part[key]) is str}
        if any(key not in clean for key in part):
            omitted = True
        parts.append(clean)
    return parts, omitted


def record_projection(record: dict[str, Any]) -> tuple[str, bool]:
    """Return the fixed searchable view; offsets are never source-file offsets."""
    item = record.get("item")
    projected: dict[str, Any] = {}
    omitted = False
    if item is not None:
        kind = item.get("type", "message" if "role" in item else None)
        if kind == "message":
            projected = {key: item[key] for key in ("type", "role", "id", "status") if key in item}
            projected["content"], omitted = _text_content(item.get("content"))
            omitted |= any(key not in {"type", "role", "id", "status", "content"} for key in item)
        elif kind == "function_call":
            projected = {key: item[key] for key in ("type", "id", "call_id", "name", "arguments", "status") if key in item}
            omitted = any(key not in projected for key in item)
        elif kind == "function_call_output":
            projected = {key: item[key] for key in ("type", "id", "call_id", "status") if key in item}
            projected["output"], omitted = _text_content(item.get("output"))
            omitted |= any(key not in {"type", "id", "call_id", "status", "output"} for key in item)
        elif kind == "reasoning":
            projected = {key: item[key] for key in ("type", "id", "status") if key in item}
            for field in ("summary", "content"):
                if field in item:
                    projected[field], removed = _text_content(item[field])
                    omitted |= removed
            omitted |= any(key not in projected for key in item)
        else:
            projected = {"type": kind, "opaque_content_omitted": True}
            omitted = True
    result = {"item": projected if item is not None else None}
    if record.get("execution") is not None:
        result["execution"] = record["execution"]
    if "outcome" in record:
        result["outcome"] = record["outcome"]
    return serialize_tool_output(result), omitted


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _fits(value: dict[str, Any]) -> bool:
    return tool_output_bytes(inline_tool_output(serialize_tool_output(value))) <= MAX_RECALL_BYTES


@dataclass(frozen=True)
class _Location:
    run_id: str
    watermark: dict[str, Any]


@dataclass
class _Index:
    locations: dict[str, _Location]
    ordered_refs: list[str]
    scopes: dict[str, dict[str, Any]]


class RecordStore:
    """File-backed bodies admitted by an explicit confirmed resource view.

    A live store follows the saver’s confirmed view. A rebuilt store is pinned
    to the supplied view and needs neither a graph nor historical checkpoints.
    """

    def __init__(self, saver: SessionCheckpointer | None = None, scope_id: str | None = None,
                 *, archive: ArchiveStore | None = None, view_id: str | None = None) -> None:
        if archive is None and (saver is None or scope_id is None):
            raise ValueError("a live storage binding or an archive and view is required")
        if archive is not None and view_id is None:
            raise ValueError("rebuilt readers require an explicit confirmed view")
        self.saver, self.scope_id = saver, scope_id
        self._archive_store, self._view_id = archive, view_id
        self._journal_store = RecordJournal(archive.directory / "history", create=False) if archive is not None else None
        if archive is not None:
            archive.view(view_id)
        self._indices: dict[str, _Index] = {}

    @property
    def archive(self) -> ArchiveStore:
        return self._archive_store if self._archive_store is not None else self.saver.archive(self.scope_id)

    @property
    def journal(self) -> RecordJournal:
        return self._journal_store if self._journal_store is not None else self.saver.history_journal(self.scope_id)

    def close(self) -> None:
        self.clear_index()
        if self._journal_store is not None:
            self._journal_store.close()

    def clear_index(self) -> None:
        self._indices.clear()

    def _head(self, checkpoint_id: str | None = None, *, view_id: str | None = None) -> str:
        if checkpoint_id is not None and view_id is not None:
            raise CheckpointError("checkpoint_read_error")
        try:
            head = (view_id or (self.archive.view_for_checkpoint(checkpoint_id) if checkpoint_id else None)
                    or self._view_id or (self.saver.confirmed_view(self.scope_id) if self.saver else None))
        except ArchiveError as error:
            raise CheckpointError(error.code) from None
        if head is None:
            raise CheckpointError("checkpoint_read_error")
        return head

    def _file_index(self, files: dict, scopes: dict | None = None,
                    order: list[str] | None = None) -> _Index:
        if type(files) is not dict:
            raise ValueError("invalid file manifest")
        order = list(files) if order is None else order
        if (type(order) is not list or len(order) != len(set(order))
                or set(order) != set(files)):
            raise ValueError("invalid file order")
        locations, ordered = {}, []
        for run_id in order:
            watermark = files[run_id]
            self.journal.verify(run_id, watermark)
            for sequence in range(watermark["record_count"]):
                ref = source_ref(run_id, sequence)
                ordered.append(ref)
                locations[ref] = _Location(run_id, deepcopy(watermark))
        return _Index(locations, ordered, deepcopy(scopes or {}))

    async def _index(self, checkpoint_id: str | None = None, *, view_id: str | None = None) -> _Index:
        head = self._head(checkpoint_id, view_id=view_id)
        try:
            values = self.archive.view(head)
            files = values["record_files"]
            if type(files) is not dict:
                raise ValueError("invalid file manifest")
            # A cached metadata index never bypasses fresh file integrity checks.
            for run_id, watermark in files.items():
                self.journal.verify(run_id, watermark)
            if head in self._indices:
                return self._indices[head]
            index = self._file_index(files, values.get("history_scopes", {}), values["record_order"])
        except (JournalError, ArchiveError) as error:
            raise CheckpointError(error.code) from None
        except CheckpointError:
            raise
        except Exception:
            raise CheckpointError("checkpoint_read_error") from None
        self._indices[head] = index
        if len(self._indices) > 16:
            self._indices.pop(next(iter(self._indices)))
        return index

    async def _load(self, ref: str, index: _Index) -> dict[str, Any]:
        try:
            location = index.locations[ref]
            return self.journal.get(ref, location.watermark)
        except JournalError as error:
            raise CheckpointError(error.code) from None
        except CheckpointError:
            raise
        except Exception:
            raise CheckpointError("checkpoint_read_error") from None

    async def records(self, run_id: str | None = None, checkpoint_id: str | None = None,
                      *, view_id: str | None = None) -> list[dict[str, Any]]:
        index = await self._index(checkpoint_id, view_id=view_id)
        return [await self._load(ref, index) for ref in index.ordered_refs
                if run_id is None or parse_source_ref(ref)[0] == run_id]

    async def get(self, ref: str, checkpoint_id: str | None = None,
                  *, view_id: str | None = None) -> dict[str, Any]:
        return await self._load(ref, await self._index(checkpoint_id, view_id=view_id))

    async def register_scope(
        self, source_refs: list[str], *, checkpoint_id: str | None = None,
        history_ref: str | None = None, view_id: str | None = None,
    ) -> dict[str, Any]:
        """Build verified metadata; the caller publishes it in a saved candidate."""
        head = self._head(checkpoint_id, view_id=view_id)
        index = await self._index(view_id=head)
        selected = set(source_refs)
        if not selected or not selected.issubset(index.locations):
            raise CheckpointError("checkpoint_read_error")
        ranges = []
        for ref in index.ordered_refs:
            if ref not in selected:
                continue
            await self._load(ref, index)
            run_id, sequence = parse_source_ref(ref)
            if ranges and ranges[-1]["run_id"] == run_id and ranges[-1]["end"] == sequence:
                ranges[-1]["end"] = sequence + 1
            else:
                ranges.append({"run_id": run_id, "start": sequence, "end": sequence + 1})
        history_ref = history_ref or f"history://scope/{uuid4()}"
        if type(history_ref) is not str or _SCOPE.fullmatch(history_ref) is None:
            raise CheckpointError("checkpoint_read_error")
        files = {interval["run_id"]: deepcopy(index.locations[source_ref(interval["run_id"], interval["start"])].watermark)
                 for interval in ranges}
        result = {"history_ref": history_ref, "source_checkpoint_id": checkpoint_id,
                  "source_view_id": head, "ranges": ranges, "record_files": files}
        if history_ref in index.scopes and index.scopes[history_ref] != result:
            raise CheckpointError("checkpoint_read_error")
        try:
            self.archive.save_scope(result)
        except ArchiveError as error:
            if self.saver is not None:
                self.saver.block(self.scope_id, error.code)
            raise CheckpointError(error.code) from None
        return result

    async def scope(self, history_ref: str, checkpoint_id: str | None = None,
                    *, view_id: str | None = None) -> dict[str, Any]:
        head = self._head(checkpoint_id, view_id=view_id)
        index = await self._index(view_id=head)
        try:
            metadata = self.archive.scope(history_ref, head)
            if metadata != index.scopes[history_ref]:
                raise ValueError("scope registration mismatch")
            if metadata["history_ref"] != history_ref:
                raise ValueError("scope mismatch")
            source_index = self._file_index(metadata["record_files"])
            seen = set()
            for interval in metadata["ranges"]:
                start, end = interval["start"], interval["end"]
                if type(start) is not int or type(end) is not int or not 0 <= start < end:
                    raise ValueError("invalid scope range")
                if end - start > len(source_index.locations):
                    raise ValueError("invalid scope range")
                for sequence in range(start, end):
                    ref = source_ref(interval["run_id"], sequence)
                    if ref not in source_index.locations or ref in seen:
                        raise ValueError("scope record missing or duplicated")
                    seen.add(ref)
            if not seen:
                raise ValueError("empty scope")
            return metadata
        except (ArchiveError, JournalError) as error:
            raise CheckpointError(error.code) from None
        except CheckpointError:
            raise
        except Exception:
            raise CheckpointError("checkpoint_read_error") from None

    @staticmethod
    def _base(record: dict[str, Any], text: str, omitted: bool) -> dict[str, Any]:
        return {key: record[key] for key in ("source_ref", "run_id", "sequence", "group_id", "kind", "call_id")} | {
            "view_kind": "record_json", "total_chars": len(text), "opaque_content_omitted": omitted,
        }

    async def read(self, ref: str, start: int, end: int | None = None) -> dict[str, Any]:
        try:
            parse_source_ref(ref)
        except ValueError:
            raise ToolArgumentError(reason_code="invalid_history_ref") from None
        try:
            record = await self.get(ref)
            text, omitted = record_projection(record)
        except (CheckpointError, ValueError, TypeError):
            raise ToolExecutionError(reason_code="history_read_error") from None
        requested_end = len(text) if end is None else end
        if (type(start) is not int or type(requested_end) is not int
                or not 0 <= start <= requested_end <= len(text)):
            raise ToolArgumentError(reason_code="invalid_history_range")

        def response(stop: int) -> dict[str, Any]:
            return {**self._base(record, text, omitted), "start": start, "end": stop,
                    "requested_end": requested_end, "text": text[start:stop],
                    "complete": start == 0 and stop == len(text),
                    "next_start": stop if stop < requested_end else None}

        low, high = start, min(requested_end, start + MAX_RECALL_BYTES)
        while low < high:
            middle = (low + high + 1) // 2
            if _fits(response(middle)):
                low = middle
            else:
                high = middle - 1
        result = response(low)
        if not _fits(result) or (start < requested_end and low == start):
            raise ToolExecutionError(reason_code="history_read_error")
        return result

    def _cursor(self, history_ref: str, fingerprint: str, index: int, position: int) -> str:
        # Fixed-width positions make every cursor in a scope/query equally
        # sized, so the reserved continuation envelope also fits after moving
        # across a record boundary or to a larger character offset.
        payload = _canonical([1, history_ref, fingerprint, f"{index:020d}", f"{position:020d}"])
        signature = hmac.new(self.archive.cursor_key(), payload, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(payload + signature).rstrip(b"=").decode("ascii")

    def _decode_cursor(self, cursor: str, history_ref: str, fingerprint: str) -> tuple[int, int]:
        try:
            if type(cursor) is not str or not 1 <= len(cursor) <= 1024:
                raise ValueError("invalid cursor")
            raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
            payload, signature = raw[:-32], raw[-32:]
            if not hmac.compare_digest(signature, hmac.new(self.archive.cursor_key(), payload, hashlib.sha256).digest()):
                raise ValueError("invalid cursor signature")
            version, scope, query, index, position = json.loads(payload)
            if (version != 1 or scope != history_ref or query != fingerprint
                    or type(index) is not str or type(position) is not str
                    or re.fullmatch(r"[0-9]{20}", index) is None
                    or re.fullmatch(r"[0-9]{20}", position) is None):
                raise ValueError("cursor mismatch")
            return int(index), int(position)
        except (ArchiveError, CheckpointError):
            raise ToolExecutionError(reason_code="history_read_error") from None
        except Exception:
            raise ToolArgumentError(reason_code="invalid_history_cursor") from None

    async def search(self, history_ref: str, keywords: list[str], cursor: str | None = None) -> dict[str, Any]:
        if type(history_ref) is not str or _SCOPE.fullmatch(history_ref) is None:
            raise ToolArgumentError(reason_code="invalid_history_ref")
        if (type(keywords) is not list or not 1 <= len(keywords) <= 8
                or any(type(word) is not str or not 1 <= len(word) <= 128 for word in keywords)):
            raise ToolArgumentError(reason_code="invalid_history_keywords")
        try:
            fingerprint = _digest(keywords)
        except (ValueError, UnicodeError):
            raise ToolArgumentError(reason_code="invalid_history_keywords") from None
        first_index, first_position = self._decode_cursor(cursor, history_ref, fingerprint) if cursor is not None else (0, 0)
        try:
            head = self._head()
            metadata = await self.scope(history_ref, view_id=head)
            snapshot = self._file_index(metadata["record_files"])
            refs = [source_ref(interval["run_id"], sequence) for interval in metadata["ranges"]
                    for sequence in range(interval["start"], interval["end"])]
            if first_index >= len(refs):
                raise ToolArgumentError(reason_code="invalid_history_cursor")
            result = {"history_ref": history_ref, "view_kind": "record_json", "query_status": "no_match",
                      "hits": [], "has_more": False, "next_cursor": None}
            for record_index in range(first_index, len(refs)):
                record = await self._load(refs[record_index], snapshot)
                text, omitted = record_projection(record)
                position = first_position if record_index == first_index else 0
                if position > len(text):
                    raise ToolArgumentError(reason_code="invalid_history_cursor")
                positions = [text.find(word, position) for word in keywords]
                while any(offset >= 0 for offset in positions):
                    match_start = min(offset for offset in positions if offset >= 0)
                    indices = [index for index, offset in enumerate(positions) if offset == match_start]
                    matches = [{"keyword_index": index, "start": match_start, "end": match_start + len(keywords[index])} for index in indices]
                    end = max(match["end"] for match in matches)
                    next_cursor = self._cursor(history_ref, fingerprint, record_index, match_start)
                    if len(result["hits"]) >= 10:
                        result.update(has_more=True, next_cursor=next_cursor)
                        return result
                    # Preserve an entire same-position matching group; only its
                    # snippet context can be reduced to fit the envelope.
                    candidate = None
                    for context in range(80, -1, -1):
                        left, right = max(0, match_start - context), min(len(text), end + context)
                        hit = {**self._base(record, text, omitted), "matches": matches,
                               "start": left, "end": right, "text": text[left:right]}
                        trial = {**result, "query_status": "matched", "hits": result["hits"] + [hit]}
                        continuing = {**trial, "has_more": True, "next_cursor": self._cursor(history_ref, fingerprint, record_index, match_start + 1)}
                        if _fits(trial) and _fits(continuing):
                            candidate = trial
                            break
                    if candidate is None:
                        if not result["hits"]:
                            raise ToolExecutionError(reason_code="history_read_error")
                        result.update(has_more=True, next_cursor=next_cursor)
                        if not _fits(result):
                            raise ToolExecutionError(reason_code="history_read_error")
                        return result
                    result = candidate
                    for index in indices:
                        positions[index] = text.find(keywords[index], match_start + 1)
            return result
        except (CheckpointError, JournalError, ArchiveError):
            raise ToolExecutionError(reason_code="history_read_error") from None
