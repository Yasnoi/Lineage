"""Allowlisted copies for tracing; never serialize State or SDK response objects."""

import json
from copy import deepcopy
from pathlib import Path
import re
from typing import Any

from lineage_agent.models.types import ModelResult
from lineage_agent.recording.summaries import summarize_value
from lineage_agent.tools.types import safe_tool_argument_reason, safe_tool_execution_reason


_MAX_TEXT = 4000
_MAX_ITEMS = 128
_MAX_LINEAGE_ITEMS = 32
_MAX_PAYLOAD = 65536
_MAX_JSON = 262144
_CREDENTIAL_KEY = re.compile(
    r"(?:api.?key|secret|password|passwd|authorization|cookie|private.?key|"
    r"access.?key|session.?key|access.?token|refresh.?token|^token$)", re.I,
)
_TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|lsv2_[A-Za-z0-9_-]{8,}|"
    r"gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,})\b"
)
_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_ -]?key|langsmith[_ -]?api[_ -]?key|secret|password|"
    r"passwd|authorization|access[_ -]?token|refresh[_ -]?token|token|cookie)"
    r"\s*[\"']?\s*[:=]\s*)"
)
_BEARER = re.compile(r"(?i)\bBearer\s+([^\s\"',;]+)")
_ERROR_CODES = {
    "unknown_tool", "invalid_arguments", "execution_error", "logging_error", "tool_interrupted",
}
_ENTRY_TYPES = {"file", "directory", "symlink", "junction", "reparse_point", "other"}
_FILESYSTEM_TOOLS = ("list_directory", "read_text_file")
_STOP_REASONS = ("completed", "max_model_calls", "context_budget_exceeded", "error")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate key")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise ValueError("non-finite value")


class TraceSanitizer:
    """Project approved fields, redact credentials, and bound upload size."""

    def __init__(self, secrets: tuple[str, ...] | list[str], project_root: Path | None) -> None:
        variants: set[str] = set()
        for secret in secrets:
            if isinstance(secret, str) and secret:
                variants.update((secret, json.dumps(secret)[1:-1]))
        self._secrets = sorted(variants, key=len, reverse=True)
        self._root: re.Pattern[str] | None = None
        if project_root is not None:
            root = str(project_root).rstrip("/\\")
            if root:
                parts = re.split(r"[/\\]+", root)
                self._root = re.compile(r"[/\\]+".join(map(re.escape, parts)), re.I)

    def text(self, value: str) -> str:
        """Clean readable text without examining environment or filesystem."""
        value = value.encode("utf-8", errors="replace").decode("utf-8")
        if (not any(secret in value for secret in self._secrets)
                and (self._root is None or self._root.search(value) is None)
                and _TOKEN.search(value) is None and _BEARER.search(value) is None
                and _ASSIGNMENT.search(value) is None):
            return self._clip(value, _MAX_TEXT)
        return self._text_fragments([value])[0]

    @staticmethod
    def _clip(value: str, limit: int) -> str:
        marker = "[truncated]"
        return value if len(value) <= limit else value[:max(0, limit - len(marker))] + marker

    def _path(self, value: str) -> str:
        cleaned = self.text(value)
        if re.match(r"^(?:[A-Za-z]:|[/\\])", value) and "[project_root]" not in cleaned:
            return "[absolute_path]"
        return cleaned

    @staticmethod
    def _decode(value: Any) -> tuple[Any, bool]:
        if not isinstance(value, str) or len(value) > _MAX_JSON:
            return None, False
        try:
            return json.loads(
                value, object_pairs_hook=_strict_object, parse_constant=_reject_constant,
            ), True
        except (ValueError, TypeError, RecursionError):
            return None, False

    @staticmethod
    def _omissions(result: dict[str, Any], source: dict[str, Any], allowed: set[str]) -> None:
        count = len(source.keys() - allowed)
        if count:
            result["omitted_fields"] = count

    def _arguments(self, name: Any, raw: Any) -> dict[str, Any]:
        decoded, valid = self._decode(raw)
        if not valid:
            return {"summary": summarize_value(raw), "omitted": "invalid_or_oversized_json"}
        if name not in _FILESYSTEM_TOOLS or not isinstance(decoded, dict):
            return {"summary": summarize_value(decoded), "omitted": "unsupported_arguments"}
        result: dict[str, Any] = {}
        if isinstance(decoded.get("path"), str):
            result["path"] = self._path(decoded["path"])
        elif "path" in decoded:
            result["path"] = {"summary": summarize_value(decoded["path"])}
        allowed = {"path"}
        if name == "list_directory":
            allowed.add("after")
            if "after" in decoded and (decoded["after"] is None or isinstance(decoded["after"], str)):
                result["after"] = self.text(decoded["after"]) if decoded["after"] is not None else None
        self._omissions(result, decoded, allowed)
        return result

    def tool_inputs(self, call: dict[str, Any]) -> dict[str, Any]:
        """Retain call association and approved filesystem path arguments."""
        result: dict[str, Any] = {"arguments": self._arguments(call.get("name"), call.get("arguments"))}
        for key in ("name", "call_id"):
            if isinstance(call.get(key), str):
                result[key] = self.text(call[key])
        return result

    def _directory(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"summary": summarize_value(value), "omitted": "invalid_directory_result"}
        result: dict[str, Any] = {}
        if isinstance(value.get("path"), str):
            result["path"] = self._path(value["path"])
        if type(value.get("limit")) is int and value["limit"] >= 0:
            result["limit"] = value["limit"]
        if type(value.get("truncated")) is bool:
            result["truncated"] = value["truncated"]
        for key in ("after", "next_after"):
            if key in value and (value[key] is None or isinstance(value[key], str)):
                result[key] = self.text(value[key]) if value[key] is not None else None
        entries = value.get("entries")
        if isinstance(entries, list):
            result["entries"] = []
            for entry in entries[:_MAX_ITEMS]:
                if not isinstance(entry, dict):
                    result["entries"].append({"summary": summarize_value(entry)})
                    continue
                item: dict[str, Any] = {}
                if isinstance(entry.get("name"), str):
                    item["name"] = self.text(entry["name"])
                if isinstance(entry.get("path"), str):
                    item["path"] = self._path(entry["path"])
                if isinstance(entry.get("type"), str) and entry["type"] in _ENTRY_TYPES:
                    item["type"] = entry["type"]
                self._omissions(item, entry, {"name", "path", "type"})
                result["entries"].append(item)
            self._omitted(result, "entries", len(entries), list(range(1, len(result["entries"]) + 1)))
        self._omissions(result, value, {"path", "entries", "limit", "truncated", "after", "next_after"})
        return result

    def _text_file(self, value: Any) -> dict[str, Any]:
        """Project read metadata only; file content is never a tool preview."""
        if not isinstance(value, dict):
            return {"summary": summarize_value(value), "omitted": "invalid_text_result"}
        result: dict[str, Any] = {"content_omitted": True}
        if isinstance(value.get("path"), str):
            result["path"] = self._path(value["path"])
        if value.get("encoding") in ("utf-8", "utf-8-sig"):
            result["encoding"] = value["encoding"]
        if type(value.get("bytes_read")) is int and value["bytes_read"] >= 0:
            result["bytes_read"] = value["bytes_read"]
        if type(value.get("complete")) is bool:
            result["complete"] = value["complete"]
        self._omissions(result, value, {"path", "encoding", "bytes_read", "complete", "content"})
        return result

    def tool_outputs(self, output: dict[str, Any] | None, name: str | None = None) -> dict[str, Any]:
        """Decode the application envelope; unknown tools never expose results."""
        if not isinstance(output, dict):
            return {"summary": summarize_value(output)}
        result: dict[str, Any] = {}
        if type(output.get("ok")) is bool:
            result["ok"] = output["ok"]
        if output.get("ok") is True:
            raw = output.get("result")
            decoded, valid = self._decode(raw)
            if name == "list_directory" and valid:
                result["result"] = self._directory(decoded)
            elif name == "read_text_file" and valid:
                result["result"] = self._text_file(decoded)
            else:
                result["result"] = {"summary": summarize_value(raw), "omitted": "unsupported_result"}
        elif output.get("ok") is False:
            error = output.get("error")
            code = error.get("code") if isinstance(error, dict) else None
            # External messages are never trusted. Regenerate any reason from
            # the same application-owned tables used by the executor and JSONL.
            result["error"] = {"code": code if isinstance(code, str) and code in _ERROR_CODES else "tool_error"}
            if isinstance(error, dict) and "reason_code" in error:
                reason = error["reason_code"]
                if code == "invalid_arguments":
                    reason_code, reason_message = safe_tool_argument_reason(reason)
                elif code == "execution_error":
                    reason_code, reason_message = safe_tool_execution_reason(reason)
                else:
                    reason_code = None
                if reason_code is not None:
                    result["error"].update(reason_code=reason_code, reason_message=reason_message)
        self._omissions(result, output, {"ok", "result", "error"})
        return result

    def _text_fragments(self, texts: list[str]) -> list[str]:
        """Mask one logical text stream before mapping it back to copied leaves."""
        texts = [text.encode("utf-8", errors="replace").decode("utf-8") for text in texts]
        joined = "".join(texts)
        intervals: list[tuple[int, int, str]] = []
        for secret in self._secrets:
            offset = 0
            while (start := joined.find(secret, offset)) >= 0:
                intervals.append((start, start + len(secret), "[redacted]"))
                offset = start + len(secret)
        intervals.extend((match.start(), match.end(), "[redacted]") for match in _TOKEN.finditer(joined))
        intervals.extend((match.start(1), match.end(1), "[redacted]") for match in _BEARER.finditer(joined))
        if self._root is not None:
            intervals.extend((match.start(), match.end(), "[project_root]") for match in self._root.finditer(joined))
        assignment_end = 0
        for match in _ASSIGNMENT.finditer(joined):
            if match.start() < assignment_end:
                continue
            start = match.end()
            if start == len(joined):
                continue
            quote = joined[start] if joined[start] in "\"'" else None
            if quote:
                start += 1
                end = start
                escaped = False
                while end < len(joined):
                    char = joined[end]
                    if char == quote and not escaped:
                        break
                    escaped = char == "\\" and not escaped
                    end += 1
            else:
                end = start
                while end < len(joined) and joined[end] not in "\r\n":
                    end += 1
            if end > start:
                intervals.append((start, end, "[redacted]"))
                assignment_end = end
        merged: list[tuple[int, int, str]] = []
        for start, stop, marker in sorted(intervals):
            if merged and start < merged[-1][1]:
                old_start, old_stop, old_marker = merged.pop()
                marker = "[redacted]" if "[redacted]" in (marker, old_marker) else marker
                merged.append((old_start, max(stop, old_stop), marker))
            else:
                merged.append((start, stop, marker))
        cleaned = []
        offset = 0
        interval_index = 0
        for text in texts:
            end = offset + len(text)
            chunks = []
            cursor = 0
            while interval_index < len(merged) and merged[interval_index][1] <= offset:
                interval_index += 1
            index = interval_index
            while index < len(merged) and merged[index][0] < end:
                start, stop, marker = merged[index]
                if start < end and stop > offset:
                    start, stop = max(start, offset) - offset, min(stop, end) - offset
                    chunks.extend((text[cursor:start], marker))
                    cursor = stop
                index += 1
            chunks.append(text[cursor:])
            cleaned.append(self._clip("".join(chunks), _MAX_TEXT))
            offset = end
        return cleaned

    @staticmethod
    def _ranges(indices: list[int]) -> list[list[int]]:
        ranges: list[list[int]] = []
        for index in indices:
            if ranges and index == ranges[-1][1] + 1:
                ranges[-1][1] = index
            else:
                ranges.append([index, index])
        return ranges

    def _omitted(self, owner: dict[str, Any], key: str, count: int, retained: list[int],
                 summaries: list[dict[str, Any]] | None = None) -> None:
        """Keep source counts outside protocol arrays, using inclusive 1-based ranges."""
        if count == len(retained):
            return
        retained_set = set(retained)
        omitted = [index for index in range(1, count + 1) if index not in retained_set]
        details: dict[str, Any] = {
            "original_count": count, "retained_count": len(retained),
            "omitted_count": count - len(retained), "omitted_ranges": self._ranges(omitted),
            "retained_ranges": self._ranges(retained),
        }
        if summaries:
            details["summarized_groups"] = summaries[:_MAX_ITEMS]
            if len(summaries) > _MAX_ITEMS:
                details["summarized_groups_omitted_count"] = len(summaries) - _MAX_ITEMS
        owner.setdefault("trace_omissions", {})[key] = details

    @staticmethod
    def _source_indices(owner: dict[str, Any], key: str) -> tuple[int, list[int], list[dict[str, Any]]]:
        details = owner.get("trace_omissions", {}).get(key)
        if details is None:
            return len(owner[key]), list(range(1, len(owner[key]) + 1)), []
        return details["original_count"], [index for first, last in details["retained_ranges"]
                                           for index in range(first, last + 1)], details.get("summarized_groups", [])

    def _retain(self, owner: dict[str, Any], key: str, positions: list[int],
                *, summarized: list[int] | None = None, reason: str = "byte_limit") -> None:
        old_unlisted = owner.get("trace_omissions", {}).get(key, {}).get("summarized_groups_omitted_count", 0)
        count, sources, summaries = self._source_indices(owner, key)
        if summarized:
            summaries = [*summaries, {"range": [sources[summarized[0]], sources[summarized[-1]]], "reason": reason}]
        owner[key] = [owner[key][index] for index in positions]
        self._omitted(owner, key, count, [sources[index] for index in positions], summaries)
        if old_unlisted:
            details = owner["trace_omissions"][key]
            details["summarized_groups_omitted_count"] = old_unlisted + details.get("summarized_groups_omitted_count", 0)

    @staticmethod
    def _context_groups(items: list[dict[str, Any]]) -> tuple[list[int], list[tuple[list[int], bool]]]:
        """Keep initial instructions/request and each complete serial tool batch together."""
        anchors: list[int] = []
        user_seen = False
        for index, item in enumerate(items):
            role = item.get("role")
            if role in ("system", "developer") or (role == "user" and not user_seen):
                anchors.append(index)
                user_seen |= role == "user"
            else:
                break
        groups: list[tuple[list[int], bool]] = []
        current: list[int] = []
        pending: set[str] = set()
        calls = False
        unmatched = False
        for index in range(len(anchors), len(items)):
            item = items[index]
            if item.get("role") == "user" and current and not pending:
                groups.extend([(current, not unmatched)] if calls else [([entry], True) for entry in current])
                current, calls, unmatched = [], False, False
            current.append(index)
            kind, call_id = item.get("type"), item.get("call_id")
            if kind == "function_call":
                calls = True
                if isinstance(call_id, str) and call_id not in pending:
                    pending.add(call_id)
                else:
                    unmatched = True
            elif kind == "function_call_output":
                if call_id in pending:
                    pending.remove(call_id)
                else:
                    unmatched = True
                if calls and not pending:
                    groups.append((current, not unmatched))
                    current, calls, unmatched = [], False, False
        if current:
            groups.extend([(current, not pending and not unmatched)] if calls or unmatched
                          else [([entry], True) for entry in current])
        return anchors, groups

    def _limit_context(self, owner: dict[str, Any]) -> None:
        anchors, groups = self._context_groups(owner["input"])
        retained = anchors if len(anchors) <= _MAX_ITEMS else []
        count, sources, summaries = self._source_indices(owner, "input")
        summaries = list(summaries)
        for indices, complete in reversed(groups):
            if complete and len(retained) + len(indices) <= _MAX_ITEMS:
                retained.extend(indices)
            else:
                summaries.append({"range": [sources[indices[0]], sources[indices[-1]]],
                                  "reason": "item_limit" if complete else "incomplete_group"})
        if len(anchors) > _MAX_ITEMS:
            summaries.append({"range": [sources[anchors[0]], sources[anchors[-1]]], "reason": "item_limit"})
        retained.sort()
        owner["input"] = [owner["input"][index] for index in retained]
        self._omitted(owner, "input", count, [sources[index] for index in retained], summaries)

    def _items(self, items: Any, *, key: str) -> dict[str, Any]:
        if isinstance(items, str):
            items = [{"role": "user", "content": items}]
        if not isinstance(items, list):
            return {key: [], "trace_omissions": {key: {"summary": summarize_value(items)}}}
        names: dict[str, Any] = {}
        leaves: list[tuple[int, int | None]] = []
        texts: list[str] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            if isinstance(item, dict) and item.get("type") == "function_call" and isinstance(item.get("call_id"), str):
                call_id = item["call_id"]
                names[call_id] = item.get("name") if call_id not in names else None
            if item.get("type") == "message" or (item.get("type") is None and item.get("role") in ("system", "developer", "user", "assistant")):
                content = item.get("content")
                if isinstance(content, str):
                    leaves.append((index, None))
                    texts.append(content)
                elif isinstance(content, list):
                    for part_index, part in enumerate(content):
                        field = "refusal" if isinstance(part, dict) and part.get("type") == "refusal" else "text"
                        if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "refusal") and isinstance(part.get(field), str):
                            leaves.append((index, part_index))
                            texts.append(part[field])
        # Match across every readable message before any item/part/text limit.
        cleaned = dict(zip(leaves, self._text_fragments(texts)))
        projected = []
        retained = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "function_call":
                safe = self.tool_inputs(item)
                safe["type"] = "function_call"
            elif kind == "function_call_output":
                call_id = item.get("call_id")
                decoded, valid = self._decode(item.get("output"))
                envelope = self.tool_outputs(decoded, names.get(call_id)) if valid and isinstance(call_id, str) else {"omitted": "invalid_tool_result"}
                safe = {"type": kind, "output": envelope}
                if isinstance(call_id, str):
                    safe["call_id"] = self.text(call_id)
            elif kind == "message" or (kind is None and item.get("role") in ("system", "developer", "user", "assistant")):
                safe = {"type": "message"}
                if item.get("role") in ("system", "developer", "user", "assistant"):
                    safe["role"] = item["role"]
                content = item.get("content")
                if isinstance(content, str):
                    safe["content"] = cleaned[index, None]
                elif isinstance(content, list):
                    safe["content"] = []
                    part_indices = []
                    for part_index, part in enumerate(content):
                        field = "refusal" if isinstance(part, dict) and part.get("type") == "refusal" else "text"
                        if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "refusal") and isinstance(part.get(field), str):
                            if len(safe["content"]) < _MAX_ITEMS:
                                safe["content"].append({"type": part["type"], field: cleaned[index, part_index]})
                                part_indices.append(part_index + 1)
                    self._omitted(safe, "content", len(content), part_indices)
                else:
                    safe["content"] = {"summary": summarize_value(content)}
            else:
                continue
            projected.append(safe)
            retained.append(index + 1)
        owner = {key: projected}
        self._omitted(owner, key, len(items), retained)
        if key == "input":
            self._limit_context(owner)
        elif len(projected) > _MAX_ITEMS:
            self._retain(owner, key, list(range(_MAX_ITEMS)), reason="item_limit")
        return owner

    def _tool_definition(self, tool: Any) -> dict[str, Any]:
        if not isinstance(tool, dict) or tool.get("type") != "function" or tool.get("name") not in _FILESYSTEM_TOOLS:
            return {"summary": summarize_value(tool), "omitted": "unsupported_tool"}
        result: dict[str, Any] = {"type": "function", "name": tool["name"]}
        if type(tool.get("strict")) is bool:
            result["strict"] = tool["strict"]
        # Only retain actual recognized schema values; never synthesize a schema
        # for a custom executor which happens to use the built-in tool's name.
        parameters = tool.get("parameters")
        if isinstance(parameters, dict):
            schema: dict[str, Any] = {}
            if parameters.get("type") == "object":
                schema["type"] = "object"
            if type(parameters.get("additionalProperties")) is bool:
                schema["additionalProperties"] = parameters["additionalProperties"]
            if parameters.get("required") == ["path"] or tool["name"] == "list_directory" and parameters.get("required") == ["path", "after"]:
                schema["required"] = list(parameters["required"])
            properties = parameters.get("properties")
            path = properties.get("path") if isinstance(properties, dict) else None
            if isinstance(path, dict):
                path_schema: dict[str, Any] = {}
                if path.get("type") == "string":
                    path_schema["type"] = "string"
                if type(path.get("minLength")) is int and path["minLength"] >= 0:
                    path_schema["minLength"] = path["minLength"]
                self._omissions(path_schema, path, {"type", "minLength"})
                schema["properties"] = {"path": path_schema}
            after = properties.get("after") if isinstance(properties, dict) else None
            if tool["name"] == "list_directory" and isinstance(after, dict):
                after_schema: dict[str, Any] = {}
                if after.get("type") == ["string", "null"]:
                    after_schema["type"] = ["string", "null"]
                if type(after.get("minLength")) is int and after["minLength"] >= 0:
                    after_schema["minLength"] = after["minLength"]
                self._omissions(after_schema, after, {"type", "minLength"})
                schema.setdefault("properties", {})["after"] = after_schema
            self._omissions(schema, parameters, {"type", "properties", "required", "additionalProperties"})
            result["parameters"] = schema
        self._omissions(result, tool, {"type", "name", "parameters", "strict"})
        return result

    def model_inputs(self, model_name: str, input: Any, tools: Any, *, instructions=None, tool_choice=None, active_context=None) -> dict[str, Any]:
        """Project Responses messages and nested tool envelopes before upload."""
        summary = active_context.get("summary") if isinstance(active_context, dict) else None
        if isinstance(summary, dict) and isinstance(input, list) and input:
            # Explicit application provenance, not a text-prefix heuristic.
            # Exclude the derived body before even gathering readable leaves.
            input = [
                {"role": "assistant", "content": "[summary content omitted]"},
                *input[1:],
            ]
        result: dict[str, Any] = {"model": self.text(model_name), **self._items(input, key="input")}
        if isinstance(summary, dict):
            result["active_context"] = self.active_context(active_context)
            result["summary_content_omitted"] = True
        if isinstance(instructions, str):
            result["instructions"] = self.text(instructions)
        if tool_choice in ("none", "auto", "required"):
            result["tool_choice"] = tool_choice
        if isinstance(tools, list) and tools:
            # Do not include arbitrary descriptions or schema extension values.
            result["tools"] = [self._tool_definition(tool) for tool in tools[:_MAX_ITEMS]]
            self._omitted(result, "tools", len(tools), list(range(1, len(result["tools"]) + 1)))
        return result

    @staticmethod
    def _identifier(value: Any) -> str | None:
        """Application UUIDs and source references cannot carry body previews."""
        if type(value) is str and re.fullmatch(
            r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}(?::(?:[0-9]{1,16}|outcome))?",
            value,
        ):
            return value
        return None

    @classmethod
    def _reference(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, Any] = {}
        for key in ("run_id", "checkpoint_id", "summary_id", "source_id"):
            identifier = cls._identifier(value.get(key))
            if identifier is not None:
                result[key] = identifier
        for key in ("item_count", "start", "end"):
            number = value.get(key)
            if type(number) is int and 0 <= number <= 2**63 - 1:
                result[key] = number
        return result

    @classmethod
    def active_context(cls, value: Any) -> dict[str, Any]:
        """Project lineage identifiers without copying an active message."""
        if not isinstance(value, dict):
            return {}
        result: dict[str, Any] = {}
        for key in ("version_id", "parent_version_id"):
            identifier = cls._identifier(value.get(key))
            if identifier is not None:
                result[key] = identifier
        if isinstance(value.get("summary"), dict):
            result["summary"] = cls._reference(value["summary"])
        turns = value.get("turns")
        if isinstance(turns, list):
            result["turns"] = [cls._reference(turn) for turn in turns[:_MAX_LINEAGE_ITEMS]]
            result["turn_count"] = len(turns)
            if len(turns) > _MAX_LINEAGE_ITEMS:
                result["turns_omitted"] = len(turns) - _MAX_LINEAGE_ITEMS
        return result

    @classmethod
    def compaction(cls, value: Any) -> dict[str, Any]:
        """Only lineage, numeric diagnostics and fixed outcomes; never material."""
        if not isinstance(value, dict):
            return {}
        result: dict[str, Any] = {}
        for key in ("summary_id", "candidate_version_id", "confirmed_checkpoint_id", "summary_checkpoint_id"):
            identifier = cls._identifier(value.get(key))
            if identifier is not None:
                result[key] = identifier
        for key in ("request_no", "source_bytes", "summary_bytes", "message_bytes", "summary_call"):
            number = value.get(key)
            if type(number) is int and 0 <= number <= 2**63 - 1:
                result[key] = number
        step = value.get("step_id")
        if type(step) is str and re.fullmatch(r"step-[0-9]{4,16}", step):
            result["step_id"] = step
        if value.get("status") in (
            "preparing", "summary_budget_rejected", "candidate", "pending_save",
            "confirmed", "used", "stopped", "failed",
        ):
            result["status"] = value["status"]
        reason = value.get("reason")
        if type(reason) is str and reason in {
            "no_eligible_history", "protected_history", "protected_context_too_large",
            "summary_budget_exceeded", "summary_too_large", "summary_not_effective",
            "recovery_exhausted", "summary_refusal", "summary_tool_request", "invalid_summary",
            "context_measurement_error", "summary_model_error", "logging_error", "checkpoint_error",
        }:
            result["reason"] = reason
        sources = value.get("sources")
        if isinstance(sources, dict):
            source_view: dict[str, Any] = {}
            previous = sources.get("previous_summary")
            if isinstance(previous, dict):
                source_view["previous_summary"] = cls._reference(previous)
            turns = sources.get("added_turns")
            if isinstance(turns, list):
                source_view["added_turns"] = [cls._reference(turn) for turn in turns[:_MAX_LINEAGE_ITEMS]]
                source_view["turn_count"] = len(turns)
                if len(turns) > _MAX_LINEAGE_ITEMS:
                    source_view["turns_omitted"] = len(turns) - _MAX_LINEAGE_ITEMS
            result["sources"] = source_view
        for key in ("summary_budget", "candidate_budget"):
            budget = cls.context_budget(value.get(key))
            if budget:
                result[key] = budget
        return result

    def summary_outputs(self, result: ModelResult | None) -> dict[str, Any]:
        """No summary text, source material, tool requests or reasoning previews."""
        output: dict[str, Any] = {"content_omitted": True}
        if not isinstance(result, dict):
            return output
        text = result.get("text")
        if isinstance(text, str):
            output["text_length"] = len(text)
        output["has_refusal"] = result.get("has_refusal") is True
        calls = result.get("tool_calls")
        if isinstance(calls, list):
            output["tool_request_count"] = len(calls)
        usage = self._usage(result.get("usage"))
        if usage:
            output["usage_metadata"] = usage
        return output

    @staticmethod
    def _usage(value: Any) -> dict[str, int]:
        if not isinstance(value, dict):
            return {}
        return {key: value[key] for key in ("input_tokens", "output_tokens", "total_tokens") if type(value.get(key)) is int and value[key] >= 0}

    def model_outputs(self, result: ModelResult | None) -> dict[str, Any]:
        """Keep text/tool outputs and known usage fields, never the raw response."""
        if result is None:
            return {}
        output: dict[str, Any] = self._items(result.get("output_items"), key="output")
        output["has_refusal"] = result.get("has_refusal") is True
        usage: dict[str, Any] = self._usage(result.get("usage"))
        raw = result.get("raw_response")
        raw_usage = raw.get("usage") if isinstance(raw, dict) else None
        if isinstance(raw_usage, dict):
            for source, target, field, label in (
                ("input_tokens_details", "input_token_details", "cached_tokens", "cache_read"),
                ("output_tokens_details", "output_token_details", "reasoning_tokens", "reasoning"),
            ):
                detail = raw_usage.get(source)
                count = detail.get(field) if isinstance(detail, dict) else None
                if type(count) is int and count >= 0:
                    usage[target] = {label: count}
        if usage:
            output["usage_metadata"] = usage
        return output

    @staticmethod
    def context_budget(diagnostic: Any) -> dict[str, Any]:
        """Project only bounded budget facts, never context or risk provenance."""
        if not isinstance(diagnostic, dict):
            return {}
        result: dict[str, Any] = {}
        if diagnostic.get("method") == "utf8_json_opaque_v1":
            result["method"] = "utf8_json_opaque_v1"
        step = diagnostic.get("step_id")
        if type(step) is str and re.fullmatch(r"step-[0-9]{4,16}", step):
            result["step_id"] = step
        for key in (
            "transparent_tokens", "opaque_items", "opaque_risk_tokens",
            "required_tokens", "limit_tokens", "output_reserve_tokens",
            "safety_margin_tokens", "trigger_percent", "trigger_tokens",
        ):
            value = diagnostic.get(key)
            if type(value) is int and 0 <= value <= 2**63 - 1:
                result[key] = value
        decision = diagnostic.get("decision")
        if type(decision) is str and decision in {"allow", "exceeded", "error"}:
            result["decision"] = decision
        if type(diagnostic.get("request_sent")) is bool:
            result["request_sent"] = diagnostic["request_sent"]
        code = diagnostic.get("error_code")
        if code is None or type(code) is str and code in {
            "context_budget_exceeded", "context_measurement_error",
        }:
            result["error_code"] = code
        return result

    def root_outputs(self, state: dict[str, Any] | None) -> dict[str, Any]:
        """Summarize final behavior without copying State or local log paths."""
        if state is None:
            return {}
        result: dict[str, Any] = {}
        final = state.get("result")
        cleanup = state.get("cleanup_error")
        completed = state.get("stop_reason") == "completed" or isinstance(cleanup, dict) and cleanup.get("previous_stop_reason") == "completed"
        if completed and isinstance(final, dict) and isinstance(final.get("text"), str):
            result["text"] = self.text(final["text"])
        if isinstance(final, dict):
            result["has_refusal"] = final.get("has_refusal") is True
        if isinstance(cleanup, dict) and cleanup.get("code") == "client_close_error":
            previous = cleanup.get("previous_stop_reason")
            result["cleanup_error"] = {
                "code": "client_close_error",
                "previous_stop_reason": previous if previous in _STOP_REASONS else None,
            }
        for key in ("model_calls", "summary_calls", "tool_requests", "tool_calls"):
            if type(state.get(key)) is int and state[key] >= 0:
                result[key] = state[key]
        if state.get("stop_reason") in _STOP_REASONS:
            result["stop_reason"] = state["stop_reason"]
        budget = self.context_budget(state.get("context_budget"))
        if budget:
            result["context_budget"] = budget
        active = self.active_context(state.get("active_context"))
        if active:
            result["active_context"] = active
        compaction = self.compaction(state.get("compaction"))
        if compaction:
            result["compaction"] = compaction
        outcomes = state.get("tool_outcomes")
        if isinstance(outcomes, dict):
            result["tool_outcomes"] = {key: outcomes[key] for key in ("succeeded", "rejected", "failed", "skipped") if type(outcomes.get(key)) is int and outcomes[key] >= 0}
        for key in ("usage_totals", "summary_usage_totals"):
            totals = state.get(key)
            if isinstance(totals, dict):
                result[key] = self._usage(totals)
                count = totals.get("missing_usage_calls")
                if type(count) is int and count >= 0:
                    result[key]["missing_usage_calls"] = count
        return result

    def _copy(self, value: Any, depth: int = 0, key: str = "") -> Any:
        if depth > 16:
            return {"trace_truncated": True, "summary": summarize_value(value)}
        if isinstance(value, str):
            if key in ("arguments", "output"):
                # Finalized protocol JSON is already clean. Never reinterpret it
                # as ordinary text or serialize it for a second time.
                return value
            return self.text(value)
        if value is None or type(value) in (bool, int):
            return value
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    continue
                result[self.text(key)] = "[redacted]" if _CREDENTIAL_KEY.search(key) else self._copy(item, depth + 1, key)
            return result
        if isinstance(value, (list, tuple)):
            return [self._copy(item, depth + 1) for item in value]
        return {"summary": summarize_value(value)}

    @classmethod
    def _wire(cls, value: Any) -> Any:
        """Encode only the two Responses JSON fields, once at the upload boundary."""
        if isinstance(value, list):
            return [cls._wire(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: cls._wire(item) for key, item in value.items()}
        kind = value.get("type")
        field = {"function_call": "arguments", "function_call_output": "output"}.get(kind) if isinstance(kind, str) else None
        if field and isinstance(result.get(field), dict):
            result[field] = json.dumps(result[field], ensure_ascii=False, separators=(",", ":"))
        return result

    def _size(self, value: Any) -> int:
        return len(json.dumps(self._wire(value), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    def _shrink(self, value: Any, limit: int, key: str = "") -> Any:
        if isinstance(value, str):
            if key in ("arguments", "output"):
                return value
            if key in ("call_id", "name", "type", "role"):
                return value
            return self._clip(value, limit)
        if isinstance(value, list):
            return [self._shrink(item, limit) for item in value]
        if isinstance(value, dict):
            return {key: deepcopy(item) if key in ("usage_metadata", "usage_totals", "trace_omissions", "context_budget")
                    else self._shrink(item, limit, key) for key, item in value.items()}
        return value

    def _reduce_input(self, result: dict[str, Any], inputs: dict[str, Any]) -> None:
        items = inputs.get("input")
        if isinstance(items, list):
            # Drop complete old rounds first. Keep initial instructions/request
            # and the latest complete round while they fit.
            while self._size(result) > _MAX_PAYLOAD:
                _, groups = self._context_groups(inputs["input"])
                if len(groups) <= 1:
                    break
                removed = set(groups[0][0])
                self._retain(inputs, "input", [index for index in range(len(inputs["input"])) if index not in removed])
        for limit in (512, 128, 32):
            if self._size(result) <= _MAX_PAYLOAD:
                return
            reduced = self._shrink(inputs, limit)
            inputs.clear()
            inputs.update(reduced)
            inputs["trace_truncated"] = True
        if self._size(result) > _MAX_PAYLOAD and isinstance(inputs.get("tools"), list):
            self._retain(inputs, "tools", [])
        if isinstance(inputs.get("input"), list):
            while self._size(result) > _MAX_PAYLOAD and inputs["input"]:
                anchors, groups = self._context_groups(inputs["input"])
                removed = groups[0][0] if groups else anchors
                removed_set = set(removed)
                self._retain(inputs, "input", [index for index in range(len(inputs["input"])) if index not in removed_set],
                             summarized=removed)

    def _reduce_output(self, result: dict[str, Any], outputs: dict[str, Any]) -> None:
        for limit in (512, 128, 32):
            if self._size(result) <= _MAX_PAYLOAD:
                return
            reduced = self._shrink(outputs, limit)
            outputs.clear()
            outputs.update(reduced)
            outputs["trace_truncated"] = True
        if self._size(result) > _MAX_PAYLOAD and isinstance(outputs.get("output"), list):
            self._retain(outputs, "output", [], summarized=list(range(len(outputs["output"]))))
        directory = outputs.get("result")
        if self._size(result) > _MAX_PAYLOAD and isinstance(directory, dict) and isinstance(directory.get("entries"), list):
            self._retain(directory, "entries", [])

    def cap_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Budget input before output and finalize protocol JSON on a clean copy."""
        result = self._copy(payload)
        wire = self._wire(result)
        if len(json.dumps(wire, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= _MAX_PAYLOAD:
            return wire
        inputs = result.get("inputs") if "inputs" in result else result if "input" in result else None
        outputs = result.get("outputs") if "outputs" in result else result if "output" in result else None
        if isinstance(inputs, dict) and self._size(result) > _MAX_PAYLOAD:
            self._reduce_input(result, inputs)
        if isinstance(outputs, dict) and self._size(result) > _MAX_PAYLOAD:
            self._reduce_output(result, outputs)
        if self._size(result) > _MAX_PAYLOAD:
            # Normal run metadata and association IDs are small; do not trade
            # them or usage for unbounded custom data on a malformed envelope.
            raise ValueError("trace envelope exceeds payload limit")
        return self._wire(result)
