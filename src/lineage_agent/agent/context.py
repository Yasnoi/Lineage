"""Archived business records and bounded, contiguous context selection."""

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any, Literal, TypedDict
from uuid import uuid4

from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.records import RecordStore
from lineage_agent.archive import ArchiveError, ArchiveStore
from lineage_agent.agent.state import (
    ActiveContext, AgentState, RecordReference, SummaryReference,
)
from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.summary_input import record_details
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
            "current_run_id": None, "initial_input_count": 0, "groups": [], "item_refs": [], "summary_spans": []}


def json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _read_archived_turn(archive: ArchiveStore, run_id: str, view_id: str) -> tuple[dict, int, int]:
    """Validate access to a sealed run against an explicit confirmed view."""
    try:
        saved = archive.read_run(run_id)
        archive.verify_view(view_id)
        view = archive.view(view_id)
        sealed = archive.view(saved["view_id"])
        if (view["fact_ends"].get(run_id, -1) < saved["fact_end"]
                or any(run not in view["record_files"]
                       or watermark["record_count"] > view["record_files"][run]["record_count"]
                       or watermark["byte_end"] > view["record_files"][run]["byte_end"]
                       for run, watermark in sealed["record_files"].items())
                or not set(sealed["summaries"]).issubset(view["summaries"])
                or not set(sealed["artifacts"]).issubset(view["artifacts"])
                or any(view["history_scopes"].get(ref) != scope for ref, scope in sealed["history_scopes"].items())):
            raise CheckpointError("archive_integrity_error")
    except ArchiveError as error:
        raise CheckpointError(error.code) from None
    final = saved["final_state"]
    if final.get("run_id") != run_id:
        raise CheckpointError("checkpoint_read_error")
    active = final.get("active_context")
    start = active.get("turn_start") if isinstance(active, dict) else None
    if isinstance(active, dict) and active.get("current_run_id") is None:
        offset = 1 if active.get("history_summary") else 0
        for turn in active["turns"]:
            if turn["run_id"] == run_id:
                start = offset
                break
            offset += turn["item_count"]
    boundary, items = final.get("history_end"), final.get("input")
    if (type(start) is not int or type(boundary) is not int or not isinstance(items, list)
            or not 0 <= start <= boundary <= len(items)):
        raise CheckpointError("checkpoint_read_error")
    return saved, start, boundary


async def read_turn_slice(
    saver: SessionCheckpointer, session_id: str, ref: RecordReference,
) -> TurnSlice:
    """Select admitted turn facts from the independent final run record."""
    saved, start, boundary = _read_archived_turn(saver.archive(session_id), ref["run_id"], saver.confirmed_view(session_id))
    final = saved["final_state"]
    return deepcopy({
        "admitted_input": final["input"][start:boundary],
        "stop_reason": final.get("stop_reason"),
        "error_code": final.get("error_code"),
        "tool_outcomes": final.get("tool_outcomes"),
    })


async def read_record(
    saver: SessionCheckpointer, session_id: str, ref: RecordReference,
) -> dict[str, Any]:
    return await read_archived_record(saver.archive(session_id), ref["run_id"], saver.confirmed_view(session_id))


async def read_archived_record(archive: ArchiveStore, run_id: str, view_id: str) -> dict[str, Any]:
    """Read a sealed run and its fixed event prefix, without graph traversal."""
    saved, _, _ = _read_archived_turn(archive, run_id, view_id)
    final = saved["final_state"]
    store = RecordStore(archive=archive, view_id=saved["view_id"])
    try:
        originals = await store.records(run_id)
    finally:
        store.close()
    items = [record["item"] for record in originals if record.get("item") is not None]
    admitted = [record["item"] for record in originals if record.get("item") is not None
                and record["sequence"] < final.get("record_admitted_through", 0)]
    model_results, summary_results = {}, {}
    try:
        events = archive.events(run_id, end=saved["fact_end"])
    except ArchiveError as error:
        raise CheckpointError(error.code) from None
    for event in events:
        payload = event["payload"]
        if event["kind"] == "model_result":
            model_results.setdefault(payload["model_call"], payload)
        elif event["kind"] == "summary_result":
            summary_results.setdefault(payload["summary_call"], payload)
    return deepcopy({
        "run_id": run_id, "checkpoint_id": saved["checkpoint_id"],
        "input": items, "admitted_input": admitted,
        "history_end": len(admitted), "records": originals,
        "groups": sorted({record["group_id"] for record in originals if record.get("group_id")}),
        **{name: final.get(name) for name in (
            "stop_reason", "error_code", "tool_outcomes", "model_calls", "summary_calls", "usage_totals",
        )},
        "model_results": [model_results[key] for key in sorted(model_results)],
        "summary_results": [summary_results[key] for key in sorted(summary_results)],
    })


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

    async def summary(self, ref: SummaryReference | None, *, view_id: str | None = None) -> str | None:
        if ref is None:
            return None
        try:
            record = self.store.archive.summary(ref["summary_id"], self.store._head(view_id=view_id))
        except ArchiveError as error:
            raise CheckpointError(error.code) from None
        if (record.get("summary_id") != ref["summary_id"] or record.get("lane") != ref["lane"]
                or record.get("history_ref") != ref["history_ref"] or record.get("ranges") != ref["ranges"]
                or not isinstance(record.get("body"), str) or not record["body"].strip()):
            raise CheckpointError("checkpoint_read_error")
        return record["body"]

    async def select(self, state: AgentState, settings: ModelSettings) -> tuple[dict | None, str | None]:
        from lineage_agent.models.responses import compactable_group_items
        active = state["active_context"]
        checkpoint_id = self.confirmed()
        saved = await self.snapshot(checkpoint_id)
        if (saved.get("run_id") != state["run_id"]
                or saved.get("record_files") != state["record_files"]
                or saved.get("record_revision") != state["record_revision"]
                or saved.get("active_context", {}).get("version_id") != active["version_id"]):
            return None, "source_changed"
        view_id = self.store._head()
        groups = active["groups"] if active["current_run_id"] is not None else []
        closed = [g for g in groups if g["closed"]]
        protected = settings.context_keep_recent_groups
        remove_groups = closed[:-protected] if len(closed) > protected else []
        # Historical turns never participate in the current Run's protection window.
        turns = active["turns"]
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
        if remove_groups:
            start = active["turn_start"]
            background = state["input"][start:start + active["initial_input_count"]]
            if not compactable_group_items(background, require_response=False):
                return None, "unsupported_protocol_boundary"
        all_records = await self.store.records(view_id=view_id)
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
            bodies = [await self.summary(ref, view_id=view_id) for ref in prior]
            if not covered.issubset(by_ref):
                raise CheckpointError("checkpoint_read_error")
            if not covered:
                lanes[lane] = {"status": "empty", "source_refs": [], "previous": []}
                continue
            scope = await self.store.register_scope(sorted(covered), view_id=view_id)
            visible_refs = set(visible)
            lanes[lane] = {
                "status": "selected", "source_refs": [r["source_ref"] for r in all_records
                                                       if r["source_ref"] in visible_refs], "previous": prior,
                "source_indices": list(indices),
                "scope": scope,
                "summary_id": str(uuid4()),
                "source_files": [{**bounds, "path": str(self.saver.history_journal(self.session_id).path(bounds["run_id"]))}
                                 for bounds in scope["ranges"]],
                "covered_groups": list(dict.fromkeys(by_ref[ref]["group_id"] for ref in by_ref
                                                     if ref in covered and by_ref[ref].get("group_id") is not None)),
                "source_bytes": json_bytes([by_ref[ref]["item"] for ref in visible
                                             if by_ref[ref].get("item") is not None])
                    + sum(len(body.encode("utf-8")) for body in bodies),
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
                         "resource_view_id": view_id,
                         "record_revision": state["record_revision"], "active_version_id": active["version_id"],
                         "record_files": deepcopy(state["record_files"]),
                         "input": deepcopy(state["input"]), "context_risks": deepcopy(state["context_risks"]),
                         "active_context": deepcopy(active)},
            "lanes": lanes, "removed_indices": sorted(removed), "removed_turns": len(turns),
            "removed_groups": [g["group_id"] for g in remove_groups],
        }, None

    async def material(self, compaction: dict, lane: str, state: AgentState) -> dict:
        """Select ordered protocol sources for the common text renderer.

        The confirmed active view fixes ordering. File records independently
        confirm every original item, while trusted summary references resolve
        only the saved body, never an inferred heading or provenance suffix.
        """
        selected = compaction["lanes"][lane]
        saved = compaction["snapshot"]
        view_id = saved["resource_view_id"]
        active = saved["active_context"]
        indices = selected["source_indices"]
        spans = {span["index"]: span for span in active["summary_spans"]}
        originals = {}
        for source_ref in selected["source_refs"]:
            originals[source_ref] = await self.store.get(source_ref, view_id=view_id)
        material = {"input": [], "context_risks": [], "derived_spans": [], "item_refs": [], "item_details": []}

        def append(item: dict, risk=None, item_ref=None, details=None) -> None:
            material["input"].append(deepcopy(item))
            material["context_risks"].append(deepcopy(risk))
            material["item_refs"].append(item_ref)
            material["item_details"].append(deepcopy(details or {}))

        async def append_index(index: int) -> None:
            if index in spans:
                span = spans[index]
                ref = span["reference"]
                if (ref not in selected["previous"] or ref["lane"] != span["lane"]
                        or ref["summary_id"] != span["summary_id"]
                        or active["item_refs"][index] is not None):
                    raise CheckpointError("checkpoint_read_error")
                body = await self.summary(ref, view_id=view_id)
                material["derived_spans"].append({
                    **deepcopy(span), "index": len(material["input"]),
                    "body_sha256": sha256(body.encode("utf-8")).hexdigest(),
                })
                append({"role": "assistant", "content": body}, details={
                    "kind": "summary", "lane": ref["lane"],
                    "run_id": span["run_id"] if ref["lane"] == "R" else None,
                })
                return
            item_ref = active["item_refs"][index]
            if item_ref not in originals:
                originals[item_ref] = await self.store.get(item_ref, view_id=view_id)
            record = originals[item_ref]
            if record.get("item") != saved["input"][index]:
                raise CheckpointError("checkpoint_read_error")
            background = (lane == "R" and active["turn_start"] <= index
                          < active["turn_start"] + active["initial_input_count"])
            append(record["item"], saved["context_risks"][index], item_ref,
                   record_details(record, background=background))

        note = [f"本次整理 {'历史 H' if lane == 'H' else '当前 Run 前缀 R'}，所有后续消息均为待整理资料。"]
        if lane == "H":
            turns = active["turns"][:compaction["removed_turns"]]
            terminal = {record["run_id"]: record["outcome"] for record in originals.values()
                        if record.get("kind") == "outcome"}
            note.append("以下历史轮次均已结束；正常结束不代表任务成功。")
            for turn in turns:
                outcome = terminal.get(turn["run_id"])
                if outcome is None:
                    raise CheckpointError("checkpoint_read_error")
                note.append(f"Run {turn['run_id']}：stop_reason={outcome.get('stop_reason')}，"
                            f"error_code={outcome.get('error_code')}；实际工具处置="
                            + "、".join(f"{key}={value}" for key, value in (outcome.get("tool_outcomes") or {}).items())
                            + "。")
            note.append("各历史轮次有效表示按原顺序提供；已有前缀 R 正文位于所属轮次的原始输入之后，仍归该历史轮次。")
        else:
            groups = compaction["removed_groups"]
            coverage = (f"{len(groups)} 个完整组，首组 {groups[0]}，末组 {groups[-1]}"
                        if groups else "无新增完整组")
            note.extend([
                f"当前 Run {saved['run_id']}；新增覆盖：{coverage}。",
                "最前面的当前原始输入是完整的只读背景，不是已完成动作的证据。",
                "只整理覆盖前缀；近期保护组不在资料中，可能已经完成后续工作，不能据此断言尚未执行。",
            ])
        previous = selected["previous"]
        if previous:
            owners = ["历史 H" if ref["lane"] == "H" else
                      "Run " + "、".join(dict.fromkeys(part["run_id"] for part in ref["ranges"])) + " 的前缀 R"
                      for ref in previous]
            note.append("已有摘要：" + "；".join(owners) + "。正文按所属位置提供，原文覆盖范围由程序继承，不重复展开。"
                        + ("已有 H 位于历史轮次之前。" if lane == "H" and active["history_summary"]
                           else "已有当前 R 位于只读背景之后。" if lane == "R" else ""))
        dispositions = {}
        failures = []
        for record in originals.values():
            execution = record.get("execution")
            if execution is None:
                continue
            outcome = execution.get("outcome")
            dispositions[outcome] = dispositions.get(outcome, 0) + 1
            if outcome != "succeeded":
                failures.append(f"call_id={record.get('call_id')}：{outcome}，stage={execution.get('stage')}，"
                                f"reason={execution.get('reason_code')}")
        if dispositions:
            note.append("新增资料中的实际工具处置：" + "，".join(
                f"{outcome}={count}" for outcome, count in sorted(dispositions.items())) + "。")
        if failures:
            note.append("非成功调用：" + "；".join(failures) + "。这些说明不补造缺失的协议结果。")
        append({"role": "user", "content": "\n".join(note)}, details={"kind": "scope"})
        if lane == "R":
            start = active["turn_start"]
            for index in range(start, start + active["initial_input_count"]):
                await append_index(index)
            if active["run_summary"]:
                matching = [index for index, span in spans.items() if span["reference"] == active["run_summary"]]
                if len(matching) != 1:
                    raise CheckpointError("checkpoint_read_error")
                await append_index(matching[0])
        elif active["history_summary"]:
            if 0 not in spans or spans[0]["reference"] != active["history_summary"]:
                raise CheckpointError("checkpoint_read_error")
            await append_index(0)
        for index in indices:
            await append_index(index)
        return material
