"""Summarize shape only; values and exception text never become previews."""

import json
import re
from itertools import islice
from typing import Any

from lineage_agent.models.types import ToolDefinition


_MAX_FIELDS = 16
_MAX_NAME_CHARS = 64
_MAX_SUMMARY_BYTES = 1024
_SENSITIVE_NAMES = (
    "apikey", "secret", "password", "passwd", "token", "credential",
    "authorization", "cookie", "privatekey", "accesskey", "sessionkey",
)


def summarize_value(value: Any) -> dict[str, Any]:
    """Return bounded, deterministic type and size metadata without values."""
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, str):
        return {"type": "string", "length": len(value)}
    if isinstance(value, dict):
        return {"type": "object", "count": len(value)}
    if isinstance(value, (list, tuple)):
        return {"type": "array", "count": len(value)}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    return {"type": "other"}


def _field_name(name: str, properties: dict[str, Any]) -> dict[str, Any]:
    # Classify before applying length limits; truncation is not redaction.
    normalized = re.sub(r"[^a-z0-9]", "", name.casefold())
    if name not in properties:
        return {"name": "[unknown]"}
    if any(fragment in normalized for fragment in _SENSITIVE_NAMES):
        return {"name": "[redacted]"}
    # Surrogates cannot be emitted as UTF-8; normalize only the log label.
    safe_name = name.encode("utf-8", errors="replace").decode("utf-8")
    result: dict[str, Any] = {"name": safe_name[:_MAX_NAME_CHARS]}
    if len(safe_name) > _MAX_NAME_CHARS:
        result["name_truncated"] = True
    return result


def summarize_arguments(arguments: dict[str, Any], tool: ToolDefinition) -> dict[str, Any]:
    parameters = tool.get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    if not isinstance(properties, dict):
        properties = {}
    fields = [
        {**_field_name(name, properties), **summarize_value(value)}
        for name, value in islice(arguments.items(), _MAX_FIELDS)
    ]
    summary: dict[str, Any] = {"type": "object", "count": len(arguments), "fields": fields}
    omitted = len(arguments) - len(fields)
    if omitted:
        summary.update(omitted_fields=omitted, truncated=True)
    while len(json.dumps(summary, ensure_ascii=False).encode("utf-8")) > _MAX_SUMMARY_BYTES:
        fields.pop()
        omitted += 1
        summary.update(omitted_fields=omitted, truncated=True)
    return summary


def tool_identity(call: dict[str, Any], request_index: int) -> dict[str, Any]:
    """Keep real association identifiers, never manufacture missing ones."""
    identity: dict[str, Any] = {"request_index": request_index}
    call_id = call.get("call_id")
    name = call.get("name")
    if isinstance(call_id, str) and call_id.strip():
        identity["call_id"] = call_id
    if isinstance(name, str) and name.strip():
        identity["tool_name"] = name
    return identity
