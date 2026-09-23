"""Immutable checkpoint references and bounded, contiguous context selection."""

from copy import deepcopy
import json
from typing import Any, TypedDict
from uuid import uuid4

from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.budget import check_context_budget


class TurnReference(TypedDict):
    run_id: str
    checkpoint_id: str
    item_count: int


class ActiveContext(TypedDict):
    version_id: str
    parent_version_id: str | None
    summary: dict[str, str] | None
    turns: list[TurnReference]
    turn_start: int


def new_context() -> ActiveContext:
    return {"version_id": str(uuid4()), "parent_version_id": None,
            "summary": None, "turns": [], "turn_start": 0}


def json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


async def read_record(saver: SessionCheckpointer, session_id: str, ref: dict) -> dict:
    """Read only confirmed snapshots; never resume, reset, or heal a failed thread."""
    saved = await saver.read_confirmed(session_id, ref["checkpoint_id"])
    final = saved.checkpoint["channel_values"]
    if final.get("run_id") != ref["run_id"]:
        raise CheckpointError("checkpoint_read_error")
    active = final.get("active_context")
    start = active.get("turn_start") if isinstance(active, dict) else None
    boundary, items = final.get("history_end"), final.get("input")
    if (type(start) is not int or type(boundary) is not int or not isinstance(items, list)
            or not 0 <= start <= boundary <= len(items)):
        raise CheckpointError("checkpoint_read_error")
    model_results, summary_results = {}, {}
    cursor = saved
    seen = set()
    while cursor is not None:
        checkpoint_id = cursor.config["configurable"]["checkpoint_id"]
        if checkpoint_id in seen:
            raise CheckpointError("checkpoint_read_error")
        seen.add(checkpoint_id)
        values = cursor.checkpoint["channel_values"]
        if values.get("run_id") != ref["run_id"]:
            break
        count = values.get("model_calls", 0)
        if count:
            # Walking backwards ends at the node that produced this attempt;
            # later tool/summary errors must not relabel the model's outcome.
            model_results[count] = {
                "model_call": count, "result": values.get("result"),
                "checkpoint_id": checkpoint_id, "error_code": values.get("error_code"),
            }
        count = values.get("summary_calls", 0)
        attempt = values.get("compaction") or {}
        if count and attempt.get("summary_call") == count:
            summary_results[count] = {
                "summary_call": count, "result": values.get("summary_result"),
                "compaction": attempt, "checkpoint_id": checkpoint_id,
            }
        parent = cursor.parent_config
        cursor = (await saver.read_confirmed(session_id, parent["configurable"]["checkpoint_id"])) if parent else None
    return deepcopy({
        "run_id": ref["run_id"], "checkpoint_id": ref["checkpoint_id"],
        "input": items[start:], "admitted_input": items[start:boundary],
        "history_end": boundary - start,
        **{name: final.get(name) for name in (
            "stop_reason", "error_code", "tool_outcomes", "model_calls", "summary_calls", "usage_totals",
        )},
        "model_results": [model_results[key] for key in sorted(model_results)],
        "summary_results": [summary_results[key] for key in sorted(summary_results)],
    })


def _without_encrypted(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_encrypted(child) for key, child in value.items() if key != "encrypted_content"}
    if isinstance(value, list):
        return [_without_encrypted(child) for child in value]
    return value


def _has_encrypted(value: Any) -> bool:
    if isinstance(value, dict):
        return "encrypted_content" in value or any(_has_encrypted(child) for child in value.values())
    return isinstance(value, list) and any(_has_encrypted(child) for child in value)


def readable_item(item: dict) -> tuple[dict, bool] | None:
    """Unknown/multimodal/authority-bearing items are whole-turn barriers."""
    kind = item.get("type")
    message = kind == "message" or (kind is None and "role" in item)
    if message:
        if item.get("role") not in {"user", "assistant"} or _has_encrypted(item):
            return None
        content = item.get("content")
        if not isinstance(content, str) and not _text_parts(content):
            return None
    elif kind == "reasoning":
        if any(key in item and not _text_parts(item[key]) for key in ("content", "summary")):
            return None
    elif kind == "function_call":
        if _has_encrypted(item) or not isinstance(item.get("arguments"), str):
            return None
    elif kind == "function_call_output":
        value = item.get("output")
        if _has_encrypted(item) or (not isinstance(value, str) and not _text_parts(value)):
            return None
    else:
        return None
    return deepcopy(_without_encrypted(item)), _has_encrypted(item)


def _text_parts(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(part, dict) and part.get("type") in {
            "input_text", "output_text", "refusal", "reasoning_text", "summary_text", "text",
        } and not _has_encrypted(part)
        for part in value
    )


def _source_ids(document: dict) -> set[str]:
    return {source for section in document.values() for entry in section for source in entry["source_ids"]}


class ContextRecords:
    """Run-bound access to the session's confirmed originals; no body cache."""

    def __init__(self, saver: SessionCheckpointer, session_id: str):
        self.saver, self.session_id = saver, session_id

    def confirmed(self) -> str:
        if error := self.saver.failure(self.session_id):
            raise error
        checkpoint_id = self.saver.last_confirmed(self.session_id)
        if checkpoint_id is None:
            raise CheckpointError("checkpoint_read_error")
        return checkpoint_id

    async def summary(self, ref: dict | None) -> dict | None:
        if ref is None:
            return None
        saved = await self.saver.read_confirmed(self.session_id, ref["checkpoint_id"])
        compaction = saved.checkpoint["channel_values"].get("compaction")
        if not isinstance(compaction, dict) or compaction.get("summary_id") != ref["summary_id"]:
            raise CheckpointError("checkpoint_read_error")
        document = compaction.get("document")
        if not isinstance(document, dict):
            raise CheckpointError("checkpoint_read_error")
        return deepcopy(document)

    async def select(self, state: dict, settings: ModelSettings, prepare_request, instructions: str,
                     tool_choice: str | None) -> tuple[dict | None, str | None]:
        """Choose enough whole old turns for the target with a full summary reserve."""
        active = state["active_context"]
        turns = active["turns"]
        eligible = len(turns) - settings.context_keep_recent_turns
        if eligible <= 0:
            return None, "no_eligible_history"
        old_summary = await self.summary(active["summary"])
        material = {"previous_summary": old_summary, "sources": [], "turn_outcomes": [], "opaque_omissions": []}
        allowed = _source_ids(old_summary) if old_summary is not None else set()
        offset = 1 if active["summary"] else 0
        removed = offset
        placeholder = {"role": "assistant", "content": ""}
        for index, ref in enumerate(turns[:eligible]):
            record = await read_record(self.saver, self.session_id, ref)
            items = record["admitted_input"]
            if len(items) != ref["item_count"] or items != state["input"][offset:offset + len(items)]:
                raise CheckpointError("checkpoint_read_error")
            readable = [readable_item(item) for item in items]
            if any(item is None for item in readable):
                return None, "protected_history"
            for item_index, converted in enumerate(readable):
                payload, omitted = converted
                source_id = f"{ref['run_id']}:{item_index}"
                allowed.add(source_id)
                material["sources"].append({
                    "source_id": source_id, "role": payload.get("role", "tool" if payload.get("type") == "function_call_output" else "assistant"),
                    "kind": payload.get("type", "message"), "payload": payload,
                })
                if omitted:
                    material["opaque_omissions"].append({"source_id": source_id, "kind": "encrypted_reasoning"})
            outcome_id = f"{ref['run_id']}:outcome"
            allowed.add(outcome_id)
            material["turn_outcomes"].append({
                "source_id": outcome_id, "run_id": ref["run_id"],
                **{key: record[key] for key in ("stop_reason", "error_code", "tool_outcomes")},
            })
            removed += len(items)
            offset += len(items)
            candidate_input = [placeholder] + state["input"][removed:]
            risks = [None] + state["context_risks"][removed:]
            request = prepare_request(candidate_input, instructions=instructions, tool_choice=tool_choice)
            budget = check_context_budget(request, risks, settings)
            if budget["decision"] == "error":
                return None, "context_measurement_error"
            upper_bound = budget["required_tokens"] - json_bytes(placeholder) + settings.summary_max_bytes
            if upper_bound < settings.context_budget_tokens * settings.context_target_percent // 100:
                return {
                    "sources": {"previous_summary": deepcopy(active["summary"]), "added_turns": deepcopy(turns[:index + 1])},
                    "material": material, "allowed_source_ids": sorted(allowed),
                    "removed_items": removed, "removed_turns": index + 1,
                    "reserved_candidate_tokens": upper_bound,
                }, None
        return None, "protected_context_too_large"
