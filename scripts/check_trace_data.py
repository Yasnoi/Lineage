"""Synthetic boundary checks for trace projection; no credentials or network."""

from copy import deepcopy
import json

from lineage_agent.models.responses import parse_response
from lineage_agent.recording.trace_data import TraceSanitizer
from check_support import check_group, observe_harness


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


@check_group("check_trace_data/check_redaction_boundaries")
def check_redaction_boundaries():
    observe_harness("trace_projection")
    sanitizer = TraceSanitizer([SECRET], None)
    samples = [
        'password="SYNTHETIC_HEAD,SYNTHETIC_TAIL"',
        "password='SYNTHETIC_HEAD;SYNTHETIC_TAIL'",
        'password="SYNTHETIC_HEAD space SYNTHETIC_TAIL"',
        r'password="SYNTHETIC_HEAD\";SYNTHETIC_TAIL"',
        r'password="SYNTHETIC_HEAD\\"; harmless',
        'password="SYNTHETIC_HEAD;SYNTHETIC_TAIL',
        "password=SYNTHETIC_HEAD space SYNTHETIC_TAIL\npublic text",
        '"password": "SYNTHETIC_HEAD,SYNTHETIC_TAIL"',
    ]
    for sample in samples:
        cleaned = sanitizer.text(sample)
        assert "SYNTHETIC_HEAD" not in cleaned and "SYNTHETIC_TAIL" not in cleaned, cleaned
        assert "[redacted]" in cleaned
        assert sanitizer.text(cleaned) == cleaned, "redaction must be stable"
    assert "public text" in sanitizer.text(samples[-2])
    assert "harmless" in sanitizer.text(samples[4]), "an even backslash count closes the quote"

    fragments = [SECRET[:10], SECRET[10:20], SECRET[20:]]
    items = [text_item(part) for part in fragments]
    before = deepcopy(items)
    payload = projected(sanitizer, items, items)
    for key, field in (("inputs", "input"), ("outputs", "output")):
        text = readable(payload[key][field])
        assert SECRET not in text and all(part not in text for part in fragments)
    assert items == before and "text" not in payload["outputs"]

    # Matching must precede both text and item limits. Each variant exposes one
    # half if only retained items/characters are searched.
    crossing = [text_item(SECRET[:10])] + [text_item("")] * 128 + [text_item(SECRET[10:])]
    for key, field in (("inputs", "input"), ("outputs", "output")):
        value = projected(sanitizer, crossing, crossing)[key]
        joined = readable(value[field])
        assert SECRET[:10] not in joined and SECRET[10:] not in joined
    long_parts = [{"type": "output_text", "text": "x" * 3995 + SECRET[:10]},
                  {"type": "output_text", "text": SECRET[10:]}]
    long_items = [{"type": "message", "role": "assistant", "content": long_parts}]
    value = projected(sanitizer, long_items)["inputs"]
    assert SECRET[10:] not in readable(value["input"])

    call = {"type": "function_call", "call_id": "safe-id", "name": "list_directory",
            "arguments": json.dumps({"path": samples[0], "private": "SYNTHETIC_UNKNOWN"})}
    output = {"type": "function_call_output", "call_id": "safe-id", "output": json.dumps({
        "ok": True, "result": json.dumps({"path": ".", "entries": [
            {"name": samples[1], "path": samples[3], "type": "file"}], "truncated": False})})}
    source = [call, output]
    before = deepcopy(source)
    value = projected(sanitizer, source)
    assert source == before
    assert all(marker not in json.dumps(value) for marker in ("SYNTHETIC_HEAD", "SYNTHETIC_TAIL", "SYNTHETIC_UNKNOWN"))
    assert isinstance(json.loads(value["inputs"]["input"][0]["arguments"]), dict)
    assert json.loads(value["inputs"]["input"][1]["output"])["ok"] is True
    assert sanitizer.cap_payload(value) == value
    assert sanitizer.text("password=value " * 2000) == "password=[redacted]"


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


@check_group("check_trace_data/check_capacity_and_rounds")
def check_capacity_and_rounds():
    observe_harness("trace_projection")
    sanitizer = TraceSanitizer([], None)
    for count in (127, 128, 129, 130):
        items = [text_item(f"item-{index}", "user") for index in range(count)]
        value = projected(sanitizer, items)
        owner = value["inputs"]
        assert len(owner["input"]) == min(128, count)
        assert readable(owner["input"][:1]) == "item-0"
        assert readable(owner["input"][-1:]) == f"item-{count - 1}"
        assert all("omitted_items" not in item for item in owner["input"])
        assert_omission(owner, "input", count)
        if count > 128:
            indices = [i for lo, hi in owner["trace_omissions"]["input"]["retained_ranges"] for i in range(lo, hi + 1)]
            assert [readable([item]) for item in owner["input"]] == [f"item-{i - 1}" for i in indices]
        assert sanitizer.cap_payload(value) == value
    assert owner["trace_omissions"]["input"]["omitted_count"] == 2

    assistant_only = [text_item(f"assistant-{index}") for index in range(130)]
    value = projected(sanitizer, assistant_only)
    assert len(value["inputs"]["input"]) == 128
    assert readable(value["inputs"]["input"][-1:]) == "assistant-129"
    assert_omission(value["inputs"], "input", 130)

    many = [text_item(f"user-{index}", "user") for index in range(5001)]
    value = projected(sanitizer, many)
    assert_omission(value["inputs"], "input", len(many))
    assert len(value["inputs"]["input"]) == 128
    details = value["inputs"]["trace_omissions"]["input"]
    assert len(details["summarized_groups"]) <= 128 and details["summarized_groups_omitted_count"] > 0
    assert sanitizer.cap_payload(value) == value

    anchors = [text_item("system instructions", "system"), text_item("initial user request", "user")]
    items = list(anchors)
    for index in range(40):
        first, second = directory_pair(f"a-{index}"), directory_pair(f"b-{index}")
        items.extend([first[0], second[0], first[1], second[1]])
    value = projected(sanitizer, items)
    kept = value["inputs"]["input"]
    assert readable(kept[:2]) == "system instructionsinitial user request"
    calls = [item["call_id"] for item in kept if item.get("type") == "function_call"]
    outputs = [item["call_id"] for item in kept if item.get("type") == "function_call_output"]
    assert calls == outputs and calls[-2:] == ["a-39", "b-39"]
    assert len(calls) % 2 == 0 and len(kept) <= 128
    assert all((f"a-{index}" in calls) == (f"b-{index}" in calls) for index in range(40))
    assert_omission(value["inputs"], "input", len(items))

    # A single 130-item batch cannot fit. Summarize it as a whole, keeping no
    # misleading half-call/result context. Dedicated tool spans are unaffected.
    pairs = [directory_pair(f"large-{index}") for index in range(65)]
    oversized = anchors + [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
    value = projected(sanitizer, oversized)
    assert len(value["inputs"]["input"]) == 2
    assert_omission(value["inputs"], "input", len(oversized))
    assert value["inputs"]["trace_omissions"]["input"]["summarized_groups"]

    # Input reduction must not consume a small complete answer and actual usage.
    history = list(anchors)
    for index in range(20):
        history.extend(directory_pair(f"bytes-{index}", entries=30, width=60))
    result = parse_response({"status": "completed", "output": [text_item("answer " * 140)],
                             "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}})
    small_output = sanitizer.model_outputs(result)
    value = sanitizer.cap_payload({"inputs": sanitizer.model_inputs("synthetic", history, None),
                                   "outputs": small_output, "extra": {"metadata": {"run_id": "synthetic-run", "duration_ms": 12}}})
    assert value["outputs"] == sanitizer.cap_payload(small_output)
    assert readable(value["outputs"]["output"]) == "answer " * 140
    assert value["outputs"]["usage_metadata"] == {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    assert len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()) <= 65536
    assert_omission(value["inputs"], "input", len(history))
    assert sanitizer.cap_payload(value) == value
    assert value["extra"]["metadata"] == {"run_id": "synthetic-run", "duration_ms": 12}

    huge = parse_response({"status": "completed", "output": [text_item("界" * 6000) for _ in range(130)],
                           "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}})
    value = sanitizer.cap_payload({"inputs": sanitizer.model_inputs("synthetic", anchors, None),
                                   "outputs": sanitizer.model_outputs(huge)})
    assert value["outputs"]["usage_metadata"] == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()) <= 65536
    assert_omission(value["outputs"], "output", 130)
    assert sanitizer.cap_payload(value) == value


if __name__ == "__main__":
    check_redaction_boundaries()
    print("PASS trace redaction boundaries")
    check_capacity_and_rounds()
    print("PASS trace capacity, omissions and complete rounds")
