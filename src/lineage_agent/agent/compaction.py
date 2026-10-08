"""Transactional two-lane compaction of confirmed history and current groups."""

from collections.abc import Awaitable, Callable
from copy import deepcopy
import json
from hashlib import sha256
from typing import Any
from uuid import uuid4

from langgraph.types import Overwrite

from lineage_agent.agent.checkpointing import CheckpointError
from lineage_agent.agent.context import ContextRecords, json_bytes
from lineage_agent.agent.record_journal import JournalError
from lineage_agent.agent.state import AgentState
from lineage_agent.models.budget import annotate_context_risks, check_context_budget
from lineage_agent.models.responses import ModelError
from lineage_agent.models.summarization import (
    SummaryError, parse_summary, prepare_summary_request, summary_budget_settings,
)
from lineage_agent.models.summary_input import INPUT_FORMAT, serialize_conversation
from lineage_agent.models.types import ModelResult
from lineage_agent.recording.recorder import LOG_FAILURE_MESSAGE


type SummaryCall = Callable[[dict[str, Any], dict[str, Any]], Awaitable[ModelResult]]
MAX_COMPACT_PASSES = 2


def safe_compaction_error(reason: str) -> str:
    return "上下文恢复未通过检查，已保留原活动视图与实际执行记录。原因：" + reason


def _message(lane: str, selected: dict, body: str) -> dict:
    """Render range provenance only from trusted metadata, never from model text."""
    ranges = selected["scope"]["ranges"]
    runs = list(dict.fromkeys(bounds["run_id"] for bounds in ranges))
    coverage = ("覆盖旧轮次：" + "、".join(runs) if lane == "H" else
                "覆盖当前 Run：" + "、".join(runs) + "；完整交互组：" + "、".join(selected["covered_groups"]))
    files = [f"- {entry['path']}：记录序号 [{entry['start']}, {entry['end']})"
             for entry in selected["source_files"]]
    note = "\n".join([
        "来源与回读〔程序添加〕", coverage,
        "本次整理使用覆盖范围的有效上下文，其中可能包含已有摘要。以下为范围级溯源，不代表逐句事实已验证。",
        "原文文件（记录序号从 0 开始，每行一条记录）：", *files,
        "检索入口：" + selected["scope"]["history_ref"],
        "需要细节时使用 history_search(history_ref, keywords, cursor)，再用命中的 source_ref 调用 history_read(source_ref, start, end)。",
        "检索覆盖该范围内全部原始记录，包括摘要未提及的细节。回读返回资料，不重放工具。",
    ])
    return {"role": "assistant", "content": f"{'历史摘要 H' if lane == 'H' else '当前进展摘要 R'}\n\n{body}\n\n---\n{note}"}


def _compose(state: AgentState, compaction: dict, documents: dict, checkpoint_id: str = "") -> tuple[list, dict]:
    old = state["active_context"]
    removed = set(compaction["removed_indices"])
    current_end = old["turn_start"] + old["initial_input_count"]
    items, refs, spans, mapping = [], [], [], {}
    active = deepcopy(old)
    active.update(parent_version_id=old["version_id"], version_id=compaction["candidate_version_id"],
                  turns=deepcopy(old["turns"][compaction["removed_turns"]:]),
                  history_summary=None, run_summary=None)

    def insert(lane: str) -> None:
        selected = compaction["lanes"][lane]
        if selected["status"] == "empty":
            return
        reference = {"summary_id": selected["summary_id"], "checkpoint_id": checkpoint_id,
                     "lane": lane, "history_ref": selected["scope"]["history_ref"],
                     "ranges": deepcopy(selected["scope"]["ranges"])}
        active["history_summary" if lane == "H" else "run_summary"] = reference
        spans.append({"index": len(items), "lane": lane, "summary_id": selected["summary_id"],
                      "run_id": state["run_id"], "reference": reference})
        items.append(_message(lane, selected, documents[lane]))
        refs.append(None)

    insert("H")
    old_spans = {span["index"]: span for span in old["summary_spans"]}
    for index, item in enumerate(state["input"]):
        if index == current_end:
            insert("R")
        if index in removed:
            continue
        mapping[index] = len(items)
        if index in old_spans:
            span = deepcopy(old_spans[index]); span["index"] = len(items); spans.append(span)
        items.append(deepcopy(item)); refs.append(old["item_refs"][index])
    if current_end == len(state["input"]):
        insert("R")
    active["turn_start"] = mapping.get(old["turn_start"], len(items))
    active["groups"] = []
    for group in old["groups"]:
        if group["group_id"] in compaction["removed_groups"]:
            continue
        new_group = deepcopy(group)
        new_group["start"] = mapping[group["start"]]
        new_group["end"] = mapping[group["end"] - 1] + 1
        active["groups"].append(new_group)
    active["item_refs"], active["summary_spans"] = refs, spans
    return items, active


def build_compaction_nodes(*, settings, records: ContextRecords | None, call_summary,
                           prepare_model_request, recorder, tracer, finish, accumulate_usage,
                           validate_input) -> dict[str, Callable]:
    def emit(event, step_id, **details):
        return recorder is None or recorder.emit(event, step_id=step_id, node="summary", **details)

    def event_for(compaction):
        return {key: deepcopy(compaction.get(key)) for key in (
            "request_no", "attempt_no", "status", "reason", "original_required", "candidate_required",
            "net_reclaimed", "remaining_budget",
        )} | {"lanes": {lane: {key: selected.get(key) for key in (
            "summary_id", "status", "model_called", "max_output_tokens", "summary_bytes", "summary_call",
        )} for lane, selected in compaction.get("lanes", {}).items()},
              "history_scopes": [deepcopy(selected["scope"]) for selected in compaction.get("lanes", {}).values()
                                 if "scope" in selected]}

    def stop(state, compaction, reason, *, failed=False, update=None, node="summary"):
        compaction.update(status="failed" if failed else "stopped", reason=reason)
        values = dict(update or {})
        values.update(compaction=compaction, stop_reason="error" if failed else "context_budget_exceeded",
                      error_code=reason if failed else "context_budget_exceeded", error=safe_compaction_error(reason),
                      compaction_events=state["compaction_events"] + [event_for(compaction)])
        emit("compaction.stopped", compaction.get("step_id"), request_no=compaction["request_no"],
             attempt_no=compaction["attempt_no"], reason=reason)
        if recorder is not None and recorder.failed:
            values.update(stop_reason="error", error_code="logging_error", error=LOG_FAILURE_MESSAGE)
        if tracer is not None:
            tracer.observe_compaction(compaction)
        return finish(state, values, node, compaction.get("step_id"))

    def measure(state, compaction, docs):
        items, active = _compose(state, compaction, docs)
        risks = annotate_context_risks(items, fallback_tokens=settings.opaque_fallback_tokens, run_id=state["run_id"])
        # Retained opaque estimates preserve the evidence-based original reserve.
        old_risks = dict(zip(state["active_context"]["item_refs"], state["context_risks"]))
        risks = [deepcopy(old_risks[ref]) if ref is not None and ref in old_risks else risk
                 for ref, risk in zip(active["item_refs"], risks)]
        budget = check_context_budget(prepare_model_request(items, **compaction["request_options"]), risks, settings)
        if budget["decision"] == "error":
            raise ValueError("measurement")
        return budget, items, risks, active

    async def unchanged(state, compaction):
        assert records is not None
        saved = await records.snapshot(records.confirmed())
        snapshot = compaction["snapshot"]
        return (state["run_id"] == saved.get("run_id") == snapshot["run_id"]
                and state["record_revision"] == saved.get("record_revision") == snapshot["record_revision"]
                and state["active_context"]["version_id"] == saved.get("active_context", {}).get("version_id")
                == snapshot["active_version_id"]
                and state["record_files"] == saved.get("record_files") == snapshot["record_files"])

    async def source_changed(state, compaction):
        # A newer confirmed view wins. Paid summary facts still belong to this Run.
        latest = await records.snapshot(records.confirmed())
        update = {field: deepcopy(latest[field]) for field in (
            "context_risks", "active_context", "history_end", "record_revision", "record_sequence",
            "record_admitted_through", "execution_facts", "run_call_ids", "history_scopes", "record_files",
        )}
        update["input"] = Overwrite(deepcopy(latest["input"]))
        return stop(state, compaction, "source_changed", failed=True, update=update)

    async def prepare(state):
        compaction = deepcopy(state["compaction"])
        compaction["step_id"] = recorder.next_step() if recorder else None
        if records is None:
            return stop(state, compaction, "no_compressible_range")
        retry = compaction["status"] == "retry_pending"
        try:
            if retry:
                if compaction["attempt_no"] != 1 or not await unchanged(state, compaction):
                    return await source_changed(state, compaction)
                compaction["attempt_no"] = 2
                for selected in compaction["lanes"].values():
                    if selected["status"] == "empty":
                        continue
                    selected["candidate_input"] = {
                        "body": selected["body"], "summary_id": selected["summary_id"],
                        "checkpoint_id": records.confirmed(),
                    }
                    selected["summary_id"] = str(uuid4())
                    selected["status"] = "prepared"
            else:
                selected, reason = await records.select(state, settings)
                if selected is None:
                    if reason == "source_changed":
                        return await source_changed(state, compaction)
                    return stop(state, compaction, reason)
                compaction.update(selected)
                compaction["original_required"] = state["context_budget"]["required_tokens"]
                compaction["candidate_version_id"] = str(uuid4())
            lanes = {lane: selected for lane, selected in compaction["lanes"].items()
                     if selected["status"] != "empty"}
            # This is only an impossibility precheck: no generation target or headroom policy.
            base = measure(state, compaction, {lane: "" for lane in lanes})[0]
            if base["required_tokens"] + len(lanes) > settings.context_budget_tokens:
                return stop(state, compaction, "protected_context_too_large")
            for lane, selected in lanes.items():
                selected["max_output_tokens"] = summary_budget_settings(settings, lane).max_output_tokens
                if not retry:
                    if not selected["source_refs"] and len(selected["previous"]) == 1:
                        selected["body"] = await records.summary(selected["previous"][0])
                        selected["status"] = "reused"
                        selected["summary_bytes"] = json_bytes(_message(lane, selected, selected["body"]))
                    else:
                        selected["status"] = "prepared"
                selected["model_called"] = False
                selected.pop("summary_call", None)
            compaction.update(status="preparing", pass_no=compaction["attempt_no"])
            scopes = deepcopy(state["history_scopes"])
            scopes.update({selected["scope"]["history_ref"]: selected["scope"] for selected in lanes.values()})
            emit("compaction.started", compaction["step_id"], request_no=compaction["request_no"],
                 attempt_no=compaction["attempt_no"])
            return finish(state, {"compaction": compaction, "history_scopes": scopes},
                          "prepare_compaction", compaction["step_id"])
        except CheckpointError:
            raise
        except Exception:
            return stop(state, compaction, "context_measurement_error", failed=True)

    async def summarize_lane(state, lane):
        compaction = deepcopy(state["compaction"])
        selected = compaction["lanes"][lane]
        step_id = recorder.next_step() if recorder else None
        compaction["step_id"] = step_id
        if not await unchanged(state, compaction):
            return await source_changed(state, compaction)
        next_status = "history_ready" if lane == "H" else "current_ready"
        if selected["status"] in ("empty", "reused"):
            compaction["status"] = next_status
            return finish(state, {"compaction": compaction}, "summarize_" + lane, step_id)
        try:
            if compaction["attempt_no"] == 2:
                candidate = selected["candidate_input"]
                material = {
                    "input": [
                        {"role": "user", "content": (
                            f"精简本次 {lane} 候选摘要。它尚未启用；覆盖范围和归属不变。"
                            "下文仅为第一份候选正文，不含原始历史；保留重要条件、失败、修正和不确定性。")},
                        {"role": "assistant", "content": candidate["body"]},
                    ],
                    "context_risks": [None, None], "item_refs": [None, None],
                    "item_details": [{"kind": "scope"}, {"kind": "candidate", "lane": lane,
                        "run_id": state["run_id"] if lane == "R" else None}],
                    "derived_spans": [{"index": 1, "lane": lane, "run_id": state["run_id"],
                        "summary_id": candidate["summary_id"],
                        "reference": {key: candidate[key] for key in ("summary_id", "checkpoint_id")},
                        "body_sha256": sha256(candidate["body"].encode("utf-8")).hexdigest()}],
                }
            else:
                material = await records.material(compaction, lane, state)
            try:
                rendered = serialize_conversation(material["input"], item_details=material["item_details"])
            except (TypeError, ValueError, KeyError):
                return stop(state, compaction, "unsupported_protocol_boundary")
            request = prepare_summary_request(settings, rendered, lane=lane, attempt_no=compaction["attempt_no"])
            if validate_input(request["input"]) is not None:
                return stop(state, compaction, "unsupported_protocol_boundary")
            # Only text is sent. Original opaque reserves still belong to the
            # retained business view, never to this rendered summary request.
            risks = [None] * len(request["input"])
            budget = check_context_budget(request, risks, summary_budget_settings(settings, lane), step_id=step_id)
            selected["summary_request_sha256"] = sha256(json.dumps(
                request, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest()
            selected["attempt_no"] = compaction["attempt_no"]
            selected["request_no"] = compaction["request_no"]
            selected["summary_budget"] = budget
            compaction["summary_budget"] = budget
            emit("context_budget.checked", step_id, purpose="summary", lane=lane,
                 **{key: value for key, value in budget.items() if key != "step_id"})
            if budget["decision"] != "allow":
                return stop(state, compaction, "summary_budget_exceeded" if budget["decision"] == "exceeded" else "context_measurement_error")
        except CheckpointError:
            raise
        except Exception:
            return stop(state, compaction, "context_measurement_error", failed=True)
        count = state["summary_calls"] + 1
        if not emit("summary.started", step_id, lane=lane, summary_call=count, summary_id=selected["summary_id"],
                    request_no=compaction["request_no"], attempt_no=compaction["attempt_no"]):
            return stop(state, compaction, "logging_error", failed=True)
        compaction.update(summary_call=count, summary_lane=lane, summary_id=selected["summary_id"])
        selected.update(summary_call=count, model_called=True)
        budget["request_sent"] = True
        failure = None
        try:
            result = await call_summary(request, {**compaction, "lane": lane,
                "source_bytes": json_bytes(request["input"]),
                # This provenance is application metadata, never SDK request input.
                "summary_input_metadata": {
                    "format": INPUT_FORMAT, "context_risks": risks,
                    "source_spans": deepcopy(rendered.source_spans),
                    **{key: deepcopy(material[key]) for key in ("derived_spans", "item_refs", "item_details")},
                }})
        except ModelError as exc:
            result, failure = exc.result, "summary_model_error"
        except Exception:
            result, failure = None, "summary_model_error"
        usage = result["usage"] if result is not None else None
        update = {"summary_calls": count, "summary_result": result,
                  "summary_usage_totals": accumulate_usage(state["summary_usage_totals"], usage),
                  "usage_totals": accumulate_usage(state["usage_totals"], usage)}
        if failure:
            emit("summary.failed", step_id, lane=lane, summary_call=count, error_code=failure)
            return stop(state, compaction, failure, failed=True, update=update)
        try:
            body = parse_summary(result)
        except SummaryError as exc:
            return stop(state, compaction, exc.code, failed=True, update=update)
        except Exception:
            return stop(state, compaction, "invalid_summary", failed=True, update=update)
        selected.update(body=body, status="generated",
                        summary_bytes=json_bytes(_message(lane, selected, body)))
        compaction["status"] = next_status
        emit("summary.returned", step_id, summary_id=selected["summary_id"], lane=lane,
             summary_call=count, usage=usage, summary_bytes=selected["summary_bytes"])
        if recorder is not None and recorder.failed:
            return stop(state, compaction, "logging_error", failed=True, update=update)
        update["compaction"] = compaction
        return finish(state, update, "summarize_" + lane, step_id)

    async def validate(state):
        compaction = deepcopy(state["compaction"])
        if not await unchanged(state, compaction):
            return await source_changed(state, compaction)
        docs = {lane: selected["body"] for lane, selected in compaction["lanes"].items() if selected["status"] != "empty"}
        try:
            budget, items, _, _ = measure(state, compaction, docs)
            if validate_input(items) is not None:
                return stop(state, compaction, "unsupported_protocol_boundary")
        except Exception:
            return stop(state, compaction, "context_measurement_error", failed=True)
        required = budget["required_tokens"]
        compaction.update(candidate_budget=budget, candidate_required=required,
                          net_reclaimed=compaction["original_required"] - required,
                          remaining_budget=settings.context_budget_tokens - required)
        reason = ("candidate_budget_exceeded" if budget["decision"] != "allow" else
                  "summary_not_effective" if compaction["net_reclaimed"] <= 0 else None)
        emit("context_budget.checked", compaction.get("step_id"), purpose="candidate",
             **{key: value for key, value in budget.items() if key != "step_id"})
        if reason and (reason != "candidate_budget_exceeded" or compaction["attempt_no"] == MAX_COMPACT_PASSES):
            return stop(state, compaction, reason)
        compaction.update(status="retry_pending" if reason else "validated", reason=reason)
        return finish(state, {"compaction": compaction,
                              "compaction_events": state["compaction_events"] + [event_for(compaction)]},
                      "validate_candidate", compaction.get("step_id"))

    async def activate(state):
        compaction = deepcopy(state["compaction"])
        if not await unchanged(state, compaction):
            return await source_changed(state, compaction)
        checkpoint_id = records.confirmed()
        saved = await records.snapshot(checkpoint_id)
        if saved.get("compaction") != compaction or compaction["status"] != "validated":
            raise CheckpointError("checkpoint_read_error")
        for selected in compaction["lanes"].values():
            if selected["status"] != "empty":
                await records.store.scope(selected["scope"]["history_ref"], checkpoint_id=checkpoint_id)
                for bounds in selected["scope"]["ranges"]:
                    try:
                        records.saver.history_journal(records.session_id).verify(
                            bounds["run_id"], compaction["snapshot"]["record_files"][bounds["run_id"]])
                    except JournalError as exc:
                        raise CheckpointError(exc.code) from exc
        docs = {lane: selected["body"] for lane, selected in compaction["lanes"].items() if selected["status"] != "empty"}
        _, _, risks, _ = measure(state, compaction, docs)
        items, active = _compose(state, compaction, docs, checkpoint_id)
        delta = len(items) - len(state["input"])
        compaction.update(status="pending_save", summary_checkpoint_id=checkpoint_id)
        return finish(state, {"input": Overwrite(items), "context_risks": risks, "active_context": active,
                              "history_end": state["history_end"] + delta, "compaction": compaction},
                      "activate_context", compaction.get("step_id"))

    async def summarize_history(state):
        return await summarize_lane(state, "H")

    async def summarize_current(state):
        return await summarize_lane(state, "R")

    return {"prepare_compaction": prepare, "summarize_history": summarize_history,
            "summarize_current": summarize_current,
            "validate_candidate": validate, "activate_context": activate}
