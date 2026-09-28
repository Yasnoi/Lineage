"""Reusable fixtures and evidence assertions for projection."""


import json
from lineage_agent.models.responses import parse_response

SECRET = "SYNTHETIC-KNOWN-MODEL-KEY-8562"

def text_item(text, role="assistant"):
    return {"type": "message", "role": role, "content": [{"type": "output_text", "text": text}]}

def readable(items):
    """Reassemble visible text, not just search for contiguous JSON substrings."""
    parts = []
    for item in items:
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(part["text"] for part in content if isinstance(part.get("text"), str))
    return "".join(parts)

def projected(sanitizer, items, output=None):
    payload = {"inputs": sanitizer.model_inputs("synthetic", items, None)}
    if output is not None:
        payload["outputs"] = sanitizer.model_outputs(parse_response({"status": "completed", "output": output, "usage": {}}))
    return sanitizer.cap_payload(payload)

def directory_pair(identifier, *, entries=0, width=8):
    return [
        {"type": "function_call", "call_id": identifier, "name": "list_directory", "arguments": '{"path":"."}'},
        {"type": "function_call_output", "call_id": identifier, "output": json.dumps({
            "ok": True, "result": json.dumps({"path": ".", "entries": [
                {"name": f"{index}-" + "n" * width, "path": f"{index}-" + "n" * width, "type": "file"}
                for index in range(entries)], "limit": entries, "truncated": False})})},
    ]

def assert_omission(owner, field, original_count):
    actual = len(owner[field])
    if actual == original_count:
        assert field not in owner.get("trace_omissions", {})
        return
    metadata = owner["trace_omissions"][field]
    assert metadata["original_count"] == original_count
    assert metadata["retained_count"] == actual
    assert metadata["omitted_count"] == original_count - actual
    retained = [index for lo, hi in metadata["retained_ranges"] for index in range(lo, hi + 1)]
    omitted = [index for lo, hi in metadata["omitted_ranges"] for index in range(lo, hi + 1)]
    assert len(retained) == actual and len(omitted) == original_count - actual
    assert sorted(retained + omitted) == list(range(1, original_count + 1))
