"""Immutable checkpoint references and bounded, contiguous context selection."""

from copy import deepcopy
import json
import re
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.checkpoint.base import CheckpointTuple

from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.records import RecordStore, record_projection
from lineage_agent.agent.state import (
    ActiveContext, AgentState, RecordReference, SummaryReference,
)
from lineage_agent.config.settings import ModelSettings
from lineage_agent.tools.types import ToolOutcomes


class TurnSlice(TypedDict):
    """Only the confirmed facts needed to select and describe source material."""

    admitted_input: list[dict[str, Any]]
    stop_reason: Literal["completed", "max_model_calls", "context_budget_exceeded", "error"] | None
    error_code: str | None
    tool_outcomes: ToolOutcomes


def new_context() -> ActiveContext:
    return {"version_id": str(uuid4()), "parent_version_id": None,
            "history_summary": None, "run_summary": None, "turns": [], "turn_start": 0,
            "initial_input_count": 0, "groups": [], "item_refs": [], "summary_spans": []}


def json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


async def _read_final_turn(
    saver: SessionCheckpointer, session_id: str, ref: RecordReference,
) -> tuple[CheckpointTuple, int, int]:
    """Load one fixed final checkpoint and validate its run and turn boundaries."""
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
    return saved, start, boundary


async def read_turn_slice(
    saver: SessionCheckpointer, session_id: str, ref: RecordReference,
) -> TurnSlice:
    """Select facts without reconstructing the run's model/summary call archive.

    The native saver still deserializes the final checkpoint. Do not traverse
    its parents or copy unrelated raw responses just to select this turn.
    """
    saved, start, boundary = await _read_final_turn(saver, session_id, ref)
    final = saved.checkpoint["channel_values"]
    return deepcopy({
        "admitted_input": final["input"][start:boundary],
        "stop_reason": final.get("stop_reason"),
        "error_code": final.get("error_code"),
        "tool_outcomes": final.get("tool_outcomes"),
    })


async def read_record(
    saver: SessionCheckpointer, session_id: str, ref: RecordReference,
) -> dict[str, Any]:
    """Read only confirmed snapshots; never resume, reset, or heal a failed thread."""
    saved, start, boundary = await _read_final_turn(saver, session_id, ref)
    final = saved.checkpoint["channel_values"]
    originals = await RecordStore(saver, session_id).records(ref["run_id"], checkpoint_id=ref["checkpoint_id"])
    items = [record["item"] for record in originals if record.get("item") is not None]
    admitted = [record["item"] for record in originals if record.get("item") is not None
                and record["sequence"] < final.get("record_admitted_through", 0)]
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
        "input": items, "admitted_input": admitted,
        "history_end": len(admitted), "records": originals,
        "groups": sorted({record["group_id"] for record in originals if record.get("group_id")}),
        **{name: final.get(name) for name in (
            "stop_reason", "error_code", "tool_outcomes", "model_calls", "summary_calls", "usage_totals",
        )},
        "model_results": [model_results[key] for key in sorted(model_results)],
        "summary_results": [summary_results[key] for key in sorted(summary_results)],
    })


def _source_ids(document: dict) -> set[str]:
    return {source for section in document.values() for entry in section for source in entry["source_ids"]}


class ContextRecords:
    """Confirmed original facts and effective views, bound to one storage scope."""

    def __init__(self, saver: SessionCheckpointer, session_id: str):
        self.saver, self.session_id = saver, session_id
        self.store = RecordStore(saver, session_id)

    def confirmed(self) -> str:
        if error := self.saver.failure(self.session_id):
            raise error
        checkpoint_id = self.saver.last_confirmed(self.session_id)
        if checkpoint_id is None:
            raise CheckpointError("checkpoint_read_error")
        return checkpoint_id

    async def snapshot(self, checkpoint_id: str) -> dict:
        saved = await self.saver.read_confirmed(self.session_id, checkpoint_id)
        return saved.checkpoint["channel_values"]

    async def summary(self, ref: SummaryReference | None) -> dict | None:
        if ref is None:
            return None
        values = await self.snapshot(ref["checkpoint_id"])
        lane = (values.get("compaction") or {}).get("lanes", {}).get(ref["lane"], {})
        if lane.get("summary_id") != ref["summary_id"] or not isinstance(lane.get("document"), dict):
            raise CheckpointError("checkpoint_read_error")
        return deepcopy(lane["document"])

    async def select(self, state: AgentState, settings: ModelSettings) -> tuple[dict | None, str | None]:
        from lineage_agent.models.responses import compactable_group_items
        active = state["active_context"]
        checkpoint_id = self.confirmed()
        saved = await self.snapshot(checkpoint_id)
        if (saved.get("record_revision") != state["record_revision"]
                or saved.get("active_context", {}).get("version_id") != active["version_id"]):
            return None, "source_changed"
        groups = active["groups"]
        closed = [g for g in groups if g["closed"]]
        protected = settings.context_keep_recent_groups
        remove_groups = closed[:-protected] if len(closed) > protected else []
        need = max(0, protected - len(closed))
        keep_turn_at = len(active["turns"])
        if need:
            for i in range(len(active["turns"]) - 1, -1, -1):
                keep_turn_at = i
                need -= active["turns"][i]["group_count"]
                if need <= 0:
                    break
        turns = active["turns"][:keep_turn_at]
        old_start = 1 if active["history_summary"] else 0
        history_end = old_start + sum(turn["item_count"] for turn in turns)
        history_indices = list(range(old_start, history_end))
        current_indices = [i for group in remove_groups for i in range(group["start"], group["end"])]
        # Protocol legality is checked on complete groups/turns, never by making up results.
        offset = old_start
        spans = {span["index"]: span for span in active["summary_spans"]}
        for turn in turns:
            original_items = [state["input"][i] for i in range(offset, offset + turn["item_count"]) if i not in spans]
            if not compactable_group_items(original_items, require_response=False):
                return None, "unsupported_protocol_boundary"
            offset += turn["item_count"]
        for group in remove_groups:
            if not compactable_group_items(state["input"][group["start"]:group["end"]]):
                return None, "unsupported_protocol_boundary"
        all_records = await self.store.records(checkpoint_id=checkpoint_id)
        by_ref = {r["source_ref"]: r for r in all_records}
        lanes = {}
        for lane, indices, previous in (
            ("H", history_indices, active["history_summary"]),
            ("R", current_indices, active["run_summary"]),
        ):
            prior = [previous] if previous else []
            prior += [spans[i]["reference"] for i in indices if i in spans]
            visible = [active["item_refs"][i] for i in indices if active["item_refs"][i]]
            covered = set(visible)
            # Historical outcomes and already summarized ranges remain searchable.
            ranges = [r for ref in prior for r in ref["ranges"]]
            if lane == "H":
                ranges += [{"run_id": turn["run_id"], "start": 0, "end": turn["record_end"]} for turn in turns]
                visible += [r["source_ref"] for r in all_records if r["kind"] == "outcome"
                            and r["run_id"] in {turn["run_id"] for turn in turns}]
            else:
                ranges += [{"run_id": state["run_id"], "start": g["record_start"], "end": g["record_end"]}
                           for g in remove_groups]
            covered.update(r["source_ref"] for r in all_records if any(
                r["run_id"] == bounds["run_id"] and bounds["start"] <= r["sequence"] < bounds["end"]
                for bounds in ranges))
            covered.update(visible)
            documents = [await self.summary(ref) for ref in prior]
            allowed = set(visible)
            for document in documents:
                allowed.update(_source_ids(document))
            if not allowed.issubset(covered) or not covered.issubset(by_ref):
                raise CheckpointError("checkpoint_read_error")
            if not covered:
                lanes[lane] = {"status": "empty", "source_refs": [], "previous": [], "allowed_source_ids": []}
                continue
            scope = await self.store.register_scope(sorted(covered), checkpoint_id=checkpoint_id)
            visible_refs = set(visible)
            lanes[lane] = {
                "status": "selected", "source_refs": [r["source_ref"] for r in all_records
                                                       if r["source_ref"] in visible_refs], "previous": prior,
                "allowed_source_ids": sorted(allowed), "scope": scope,
                "summary_id": str(uuid4()),
                "source_metadata": {ref: _artifact_metadata(by_ref[ref]) for ref in allowed
                                    if _artifact_metadata(by_ref[ref])},
                "source_bytes": sum(len(record_projection(by_ref[ref])[0].encode("utf-8")) for ref in visible_refs)
                    + sum(json_bytes(doc) for doc in documents),
            }
        if all(lane["status"] == "empty" for lane in lanes.values()):
            return None, "no_compressible_range"
        removed = set(history_indices + current_indices)
        if active["history_summary"]:
            removed.add(0)
        if active["run_summary"]:
            removed.update(span["index"] for span in active["summary_spans"]
                           if span["run_id"] == state["run_id"] and span["lane"] == "R")
        return {
            "snapshot": {"run_id": state["run_id"], "source_checkpoint_id": checkpoint_id,
                         "record_revision": state["record_revision"], "active_version_id": active["version_id"]},
            "lanes": lanes, "removed_indices": sorted(removed), "removed_turns": len(turns),
            "removed_groups": [g["group_id"] for g in remove_groups],
        }, None

    async def material(self, compaction: dict, lane: str, state: AgentState) -> dict:
        selected = compaction["lanes"][lane]
        checkpoint_id = compaction["snapshot"]["source_checkpoint_id"]
        documents = [await self.summary(ref) for ref in selected["previous"]]
        sources = []
        for source_ref in selected["source_refs"]:
            record = await self.store.get(source_ref, checkpoint_id=checkpoint_id)
            projection, opaque = record_projection(record)
            sources.append({"source_id": source_ref, "kind": record["kind"],
                            "group_id": record.get("group_id"), "payload": json.loads(projection),
                            "opaque_omitted": opaque})
        material = {"sources": sources, "previous_summary": documents,
                    "turn_outcomes": [], "allowed_source_ids": selected["allowed_source_ids"]}
        if lane == "R":
            active = state["active_context"]
            start = active["turn_start"]
            material["current_request_readonly"] = deepcopy(state["input"][start:start + active["initial_input_count"]])
        return material


def _artifact_metadata(record: dict) -> dict:
    """Only program-created observation/recall fields confer Artifact identity."""
    item = record.get("item") or {}
    if item.get("type") != "function_call_output" or not isinstance(item.get("output"), str):
        return {}
    try:
        envelope = json.loads(item["output"])
        observation = envelope.get("observation", {})
        value = observation
        if observation.get("mode") == "inline" and (record.get("execution") or {}).get("name") in {"artifact_read", "artifact_search"}:
            value = json.loads(observation["raw_result"])
        ref = value.get("artifact_ref")
        if not isinstance(ref, str) or re.fullmatch(r"artifact://tool/[0-9a-f]{32}", ref) is None:
            return {}
        return {key: deepcopy(value[key]) for key in ("artifact_ref", "view_kind", "total_chars", "complete", "start", "end") if key in value}
    except (ValueError, TypeError, KeyError, AttributeError):
        return {}
