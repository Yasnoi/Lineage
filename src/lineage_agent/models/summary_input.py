"""Render selected, confirmed history as inert text; never select or fetch sources.

Receipts describe character positions in the rendered message for local evidence.
They are not sent to the model and contain no duplicate source bodies.
"""

from dataclasses import dataclass
from html import escape
from typing import Any

from lineage_agent.models.responses import compactable_group_items


INPUT_FORMAT = "compact_conversation_v1"


@dataclass(frozen=True)
class SummaryInput:
    input: list[dict[str, Any]]
    source_spans: list[dict[str, int]]


def record_details(record: dict, *, background: bool = False) -> dict:
    """Only application-owned identities and actual execution facts label sources."""
    return {"kind": "background" if background else "record", "run_id": record["run_id"],
            "group_id": record.get("group_id"), "execution": record.get("execution")}


def serialize_conversation(items: list[dict], *, item_details: list[dict] | None = None) -> SummaryInput:
    """Preserve readable content/order/pairing, replacing only opaque reasoning.

The caller supplies trusted scope/summary identities. Similar headings in source
text have no special meaning. Unknown protocol dependencies remain barriers.
"""
    if not isinstance(items, list) or not items or any(not isinstance(item, dict) for item in items):
        raise ValueError("empty summary sources")
    details = [{} for _ in items] if item_details is None else item_details
    if len(details) != len(items) or any(not isinstance(meta, dict) for meta in details):
        raise ValueError("source metadata alignment differs")
    # Calls are unique within a Run, not across all historical Runs. Check each
    # contiguous owner before rendering; never repair incomplete protocol pairs.
    bucket, owner = [], None
    seen_owners = set()
    for index, (item, meta) in enumerate(zip(items, details, strict=True)):
        kind = meta.get("kind", "record")
        if kind not in {"scope", "record", "background", "summary", "candidate"}:
            raise ValueError("unknown source kind")
        if kind == "scope":
            if index != 0 or item.get("role") != "user" or type(item.get("content")) is not str:
                raise ValueError("invalid application scope")
            continue
        next_owner = meta.get("run_id")
        if next_owner != owner and bucket:
            if not compactable_group_items(bucket, require_response=False):
                raise ValueError("unsupported summary boundary")
            seen_owners.add(owner)
            bucket = []
            if next_owner in seen_owners:
                raise ValueError("noncontiguous source owner")
        owner = next_owner
        bucket.append(item)
        if kind in {"summary", "candidate"} and (
                item.get("role") != "assistant" or type(item.get("content")) is not str
                or meta.get("lane") not in {"H", "R"}):
            raise ValueError("invalid derived source")
    if not bucket or not compactable_group_items(bucket, require_response=False):
        raise ValueError("unsupported summary boundary")

    pieces, spans = [], []
    length = 0

    def add(value: str) -> None:
        nonlocal length
        value.encode("utf-8")
        pieces.append(value)
        length += len(value)

    def attrs(**values) -> str:
        return "".join(f' {key}="{escape(str(value), quote=True)}"'
                       for key, value in values.items() if value is not None)

    def block(tag: str, text: str, **values) -> tuple[int, int]:
        add(f"<{tag}{attrs(**values)}>\n")
        start = length
        add(escape(text, quote=False))
        end = length
        add(f"\n</{tag}>\n")
        return start, end

    def content(value: Any, *, reasoning: bool = False) -> None:
        if isinstance(value, str):
            add(escape(value, quote=False))
            add("\n")
        else:
            for part in value or []:
                kind = part["type"]
                text = part["refusal" if kind == "refusal" else "text"]
                tag = {"reasoning_text": "text" if reasoning else "assistant_thinking",
                       "summary_text": "text" if reasoning else "reasoning_summary",
                       "refusal": "refusal"}.get(kind, "text")
                block(tag, text)

    first = 0
    if details[0].get("kind") == "scope":
        start, end = block("summary_scope", items[0]["content"])
        spans.append({"index": 0, "body_start": start, "body_end": end})
        first = 1
    add("<conversation>\n")
    owner, group, turn_open, group_open = None, None, False, False
    calls = {}
    for index in range(first, len(items)):
        item, meta = items[index], details[index]
        next_owner, next_group = meta.get("run_id"), meta.get("group_id")
        if not turn_open or owner != next_owner:
            if group_open:
                add("</interaction_group>\n")
                group_open = False
            if turn_open:
                add("</turn>\n")
            add(f"<turn{attrs(run_id=next_owner)}>\n")
            owner, group, turn_open = next_owner, None, True
        if next_group != group:
            if group_open:
                add("</interaction_group>\n")
                group_open = False
            if next_group is not None:
                add(f"<interaction_group{attrs(id=next_group)}>\n")
                group_open = True
            group = next_group
        receipt = {"index": index, "start": length}
        kind = meta.get("kind", "record")
        shape = item.get("type") or "message"
        if kind in {"summary", "candidate"}:
            start, end = block("previous_summary" if kind == "summary" else "candidate_summary",
                               item["content"], lane=meta["lane"])
            receipt.update(body_start=start, body_end=end)
        elif shape == "message":
            tag = "current_request_background" if kind == "background" else item["role"] + "_message"
            add(f"<{tag}{attrs(phase=item.get('phase'))}>\n")
            content(item["content"])
            add(f"</{tag}>\n")
        elif shape == "reasoning":
            if item.get("content"):
                add("<assistant_thinking>\n")
                content(item["content"], reasoning=True)
                add("</assistant_thinking>\n")
            if item.get("summary"):
                add("<reasoning_summary>\n")
                content(item["summary"], reasoning=True)
                add("</reasoning_summary>\n")
            if item.get("encrypted_content") is not None:
                add("<opaque_reasoning>Unreadable reasoning exists; preserved in original records.</opaque_reasoning>\n")
            elif not item.get("content") and not item.get("summary"):
                add("<reasoning_note>No readable reasoning was supplied.</reasoning_note>\n")
        elif shape == "function_call":
            calls[(owner, item["call_id"])] = item["name"]
            block("tool_call", item["arguments"], call_id=item["call_id"], name=item["name"])
        elif shape == "function_call_output":
            add("<tool_result" + attrs(call_id=item["call_id"], name=calls[(owner, item["call_id"])]) + ">\n")
            execution = meta.get("execution")
            if execution is not None:
                if execution.get("call_id") != item["call_id"]:
                    raise ValueError("execution identity differs")
                add("<execution" + attrs(outcome=execution.get("outcome"), stage=execution.get("stage"),
                                          reason=execution.get("reason_code")) + " />\n")
            else:
                add("<execution>Execution metadata unavailable; do not infer success from protocol completion.</execution>\n")
            content(item["output"])
            add("</tool_result>\n")
        receipt["end"] = length
        spans.append(receipt)
    if group_open:
        add("</interaction_group>\n")
    add("</turn>\n</conversation>")
    return SummaryInput([{"role": "user", "content": "".join(pieces)}], spans)
