"""Allowlisted copies for tracing; never serialize State or SDK response objects."""

import json
from pathlib import Path
import re
from typing import Any

from lineage_agent.models.types import ModelResult
from lineage_agent.recording.summaries import summarize_value


_MAX_TEXT = 4000
_MAX_ITEMS = 128
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
    r"\s*[\"']?\s*[:=]\s*)([\"']?)[^\r\n,;}\"']+"
)
_ERROR_CODES = {
    "unknown_tool", "invalid_arguments", "execution_error", "logging_error", "tool_interrupted",
}
_ENTRY_TYPES = {"file", "directory", "symlink", "junction", "reparse_point", "other"}


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
        for secret in self._secrets:
            value = value.replace(secret, "[redacted]")
        if self._root is not None:
            value = self._root.sub("[project_root]", value)
        value = _TOKEN.sub("[redacted]", value)
        value = re.sub(r"(?i)\bBearer\s+[^\s\"',;]+", "Bearer [redacted]", value)
        value = _ASSIGNMENT.sub(r"\1\2[redacted]", value)
        if len(value) > _MAX_TEXT:
            value = value[:_MAX_TEXT] + "[truncated]"
        return value

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
        if name != "list_directory" or not isinstance(decoded, dict):
            return {"summary": summarize_value(decoded), "omitted": "unsupported_arguments"}
        result: dict[str, Any] = {}
        if isinstance(decoded.get("path"), str):
            result["path"] = self._path(decoded["path"])
        elif "path" in decoded:
            result["path"] = {"summary": summarize_value(decoded["path"])}
        self._omissions(result, decoded, {"path"})
        return result

    def tool_inputs(self, call: dict[str, Any]) -> dict[str, Any]:
        """Retain call association and only approved directory arguments."""
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
            if len(entries) > _MAX_ITEMS:
                result.update(trace_truncated=True, omitted_entries=len(entries) - _MAX_ITEMS)
        self._omissions(result, value, {"path", "entries", "limit", "truncated"})
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
            result["result"] = (
                self._directory(decoded) if name == "list_directory" and valid else
                {"summary": summarize_value(raw), "omitted": "unsupported_result"}
            )
        elif output.get("ok") is False:
            error = output.get("error")
            code = error.get("code") if isinstance(error, dict) else None
            # The error message may be supplied by external callers. Only codes
            # belong in this projection; application-owned reasons are metadata.
            result["error"] = {"code": code if isinstance(code, str) and code in _ERROR_CODES else "tool_error"}
        self._omissions(result, output, {"ok", "result", "error"})
        return result

    def _text_fragments(self, texts: list[str]) -> list[str]:
        """Redact known secrets even when a message splits one across text parts."""
        joined = "".join(texts)
        intervals: list[tuple[int, int]] = []
        for secret in self._secrets:
            offset = 0
            while (start := joined.find(secret, offset)) >= 0:
                intervals.append((start, start + len(secret)))
                offset = start + len(secret)
        cleaned = []
        offset = 0
        for text in texts:
            end = offset + len(text)
            overlaps = sorted((max(start, offset) - offset, min(stop, end) - offset)
                              for start, stop in intervals if start < end and stop > offset)
            chunks = []
            cursor = 0
            for start, stop in overlaps:
                if stop > cursor:
                    chunks.extend((text[cursor:max(start, cursor)], "[redacted]"))
                    cursor = stop
            chunks.append(text[cursor:])
            cleaned.append(self.text("".join(chunks)))
            offset = end
        return cleaned

    def _items(self, items: Any) -> list[dict[str, Any]]:
        if isinstance(items, str):
            return [{"role": "user", "content": self.text(items)}]
        if not isinstance(items, list):
            return [{"summary": summarize_value(items)}]
        names: dict[str, Any] = {}
        for item in items:
            if isinstance(item, dict) and item.get("type") == "function_call" and isinstance(item.get("call_id"), str):
                call_id = item["call_id"]
                # Ambiguous associations must not grant permission to expose a result.
                names[call_id] = item.get("name") if call_id not in names else None
        projected = []
        for item in items[:_MAX_ITEMS]:
            if not isinstance(item, dict):
                projected.append({"summary": summarize_value(item)})
                continue
            kind = item.get("type")
            if kind == "function_call":
                safe = self.tool_inputs(item)
                safe["type"] = "function_call"
                safe["arguments"] = json.dumps(safe["arguments"], ensure_ascii=False)
            elif kind == "function_call_output":
                call_id = item.get("call_id")
                decoded, valid = self._decode(item.get("output"))
                envelope = self.tool_outputs(decoded, names.get(call_id)) if valid and isinstance(call_id, str) else {"omitted": "invalid_tool_result"}
                safe = {"type": kind, "output": json.dumps(envelope, ensure_ascii=False)}
                if isinstance(call_id, str):
                    safe["call_id"] = self.text(call_id)
            elif kind == "message" or (kind is None and item.get("role") in ("system", "developer", "user", "assistant")):
                safe = {"type": "message"}
                if item.get("role") in ("system", "developer", "user", "assistant"):
                    safe["role"] = item["role"]
                content = item.get("content")
                if isinstance(content, str):
                    safe["content"] = self.text(content)
                elif isinstance(content, list):
                    safe["content"] = []
                    text_parts = [part for part in content[:_MAX_ITEMS] if isinstance(part, dict)
                                  and part.get("type") in ("input_text", "output_text")
                                  and isinstance(part.get("text"), str)]
                    texts = iter(self._text_fragments([part["text"] for part in text_parts]))
                    for part in content[:_MAX_ITEMS]:
                        if isinstance(part, dict) and part.get("type") in ("input_text", "output_text") and isinstance(part.get("text"), str):
                            safe["content"].append({"type": part["type"], "text": next(texts)})
                        else:
                            safe["content"].append({"omitted": "unsupported_content", "summary": summarize_value(part)})
                    if len(content) > _MAX_ITEMS:
                        safe["trace_truncated"] = True
                else:
                    safe["content"] = {"summary": summarize_value(content)}
            else:
                safe = {"omitted": "unsupported_protocol_item", "summary": summarize_value(item)}
            projected.append(safe)
        if len(items) > _MAX_ITEMS:
            projected.append({"trace_truncated": True, "omitted_items": len(items) - _MAX_ITEMS})
        return projected

    def _tool_definition(self, tool: Any) -> dict[str, Any]:
        if not isinstance(tool, dict) or tool.get("type") != "function" or tool.get("name") != "list_directory":
            return {"summary": summarize_value(tool), "omitted": "unsupported_tool"}
        result: dict[str, Any] = {"type": "function", "name": "list_directory"}
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
            if parameters.get("required") == ["path"]:
                schema["required"] = ["path"]
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
            self._omissions(schema, parameters, {"type", "properties", "required", "additionalProperties"})
            result["parameters"] = schema
        self._omissions(result, tool, {"type", "name", "parameters", "strict"})
        return result

    def model_inputs(self, model_name: str, input: Any, tools: Any) -> dict[str, Any]:
        """Project Responses messages and nested tool envelopes before upload."""
        result: dict[str, Any] = {"model": self.text(model_name), "input": self._items(input)}
        if isinstance(tools, list) and tools:
            # Do not include arbitrary descriptions or schema extension values.
            result["tools"] = [self._tool_definition(tool) for tool in tools[:_MAX_ITEMS]]
            if len(tools) > _MAX_ITEMS:
                result["trace_truncated"] = True
        return result

    @staticmethod
    def _usage(value: Any) -> dict[str, int]:
        if not isinstance(value, dict):
            return {}
        return {key: value[key] for key in ("input_tokens", "output_tokens", "total_tokens") if type(value.get(key)) is int and value[key] >= 0}

    def model_outputs(self, result: ModelResult | None) -> dict[str, Any]:
        """Keep text/tool outputs and known usage fields, never the raw response."""
        if result is None:
            return {}
        output: dict[str, Any] = {"output": self._items(result.get("output_items"))}
        if isinstance(result.get("text"), str):
            output["text"] = self.text(result["text"])
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

    def root_outputs(self, state: dict[str, Any] | None) -> dict[str, Any]:
        """Summarize final behavior without copying State or local log paths."""
        if state is None:
            return {}
        result: dict[str, Any] = {}
        final = state.get("result")
        if state.get("stop_reason") == "completed" and isinstance(final, dict) and isinstance(final.get("text"), str):
            result["text"] = self.text(final["text"])
        for key in ("model_calls", "tool_requests", "tool_calls"):
            if type(state.get(key)) is int and state[key] >= 0:
                result[key] = state[key]
        if state.get("stop_reason") in ("completed", "max_model_calls", "error"):
            result["stop_reason"] = state["stop_reason"]
        outcomes = state.get("tool_outcomes")
        if isinstance(outcomes, dict):
            result["tool_outcomes"] = {key: outcomes[key] for key in ("succeeded", "rejected", "failed", "skipped") if type(outcomes.get(key)) is int and outcomes[key] >= 0}
        totals = state.get("usage_totals")
        if isinstance(totals, dict):
            result["usage_totals"] = self._usage(totals)
            count = totals.get("missing_usage_calls")
            if type(count) is int and count >= 0:
                result["usage_totals"]["missing_usage_calls"] = count
        return result

    def _copy(self, value: Any, depth: int = 0, key: str = "") -> Any:
        if depth > 16:
            return {"trace_truncated": True, "summary": summarize_value(value)}
        if isinstance(value, str):
            if key in ("arguments", "output"):
                decoded, valid = self._decode(value)
                if valid:
                    return json.dumps(self._copy(decoded, depth + 1), ensure_ascii=False)
                return json.dumps({"trace_truncated": True, "summary": summarize_value(value)})
            return self.text(value)
        if value is None or type(value) in (bool, int):
            return value
        if isinstance(value, dict):
            result = {}
            for key, item in list(value.items())[:_MAX_ITEMS]:
                if not isinstance(key, str):
                    continue
                result[self.text(key)] = "[redacted]" if _CREDENTIAL_KEY.search(key) else self._copy(item, depth + 1, key)
            if len(value) > _MAX_ITEMS:
                result["trace_truncated"] = True
            return result
        if isinstance(value, (list, tuple)):
            result = [self._copy(item, depth + 1) for item in value[:_MAX_ITEMS]]
            if len(value) > _MAX_ITEMS:
                result.append({"trace_truncated": True, "omitted_items": len(value) - _MAX_ITEMS})
            return result
        return {"summary": summarize_value(value)}

    @staticmethod
    def _size(value: Any) -> int:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    def _shrink(self, value: Any, limit: int, key: str = "") -> Any:
        if isinstance(value, str):
            if key in ("arguments", "output"):
                decoded, valid = self._decode(value)
                if valid:
                    return json.dumps(self._shrink(decoded, limit), ensure_ascii=False)
                return json.dumps({"trace_truncated": True, "summary": summarize_value(value)})
            if key in ("call_id", "name", "type", "role"):
                return value
            return value if len(value) <= limit else value[:limit] + "[truncated]"
        if isinstance(value, list):
            return [self._shrink(item, limit) for item in value]
        if isinstance(value, dict):
            return {key: self._shrink(item, limit, key) for key, item in value.items()}
        return value

    def cap_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Bound a clean copy, preserving transport identifiers on full run records."""
        result = self._copy(payload)
        if self._size(result) <= _MAX_PAYLOAD:
            return result
        targets = [key for key in ("inputs", "outputs") if key in result]
        for limit in (512, 128, 32):
            if targets:
                for key in targets:
                    result[key] = self._shrink(result[key], limit)
                    if isinstance(result[key], dict):
                        result[key]["trace_truncated"] = True
            else:
                result = self._shrink(result, limit)
                result["trace_truncated"] = True
            if self._size(result) <= _MAX_PAYLOAD:
                return result
        if targets:
            for key in targets:
                result[key] = {"trace_truncated": True, "summary": summarize_value(result[key])}
            if self._size(result) > _MAX_PAYLOAD and "extra" in result:
                result["extra"] = {"metadata": {"trace_truncated": True}}
            if self._size(result) > _MAX_PAYLOAD:
                # Normal transport identifiers are short. An invalid envelope
                # cannot bypass the byte limit via arbitrary top-level values.
                result = {key: result[key] for key in (
                    "id", "trace_id", "parent_run_id", "dotted_order", "name",
                    "run_type", "start_time", "end_time", "project_name", "extra",
                    "error", "inputs", "outputs",
                ) if key in result}
                result = self._shrink(result, 512)
            if self._size(result) > _MAX_PAYLOAD:
                raise ValueError("trace envelope exceeds payload limit")
        else:
            result = {"trace_truncated": True, "summary": summarize_value(result)}
        return result
