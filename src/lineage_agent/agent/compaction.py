"""Transactional two-lane compaction of confirmed history and current groups."""

from collections.abc import Awaitable, Callable
from copy import deepcopy
import json
import math
from hashlib import sha256
from typing import Any
from uuid import uuid4

from langgraph.types import Overwrite

from lineage_agent.agent.checkpointing import CheckpointError
from lineage_agent.agent.context import ContextRecords, json_bytes
from lineage_agent.agent.state import AgentState
from lineage_agent.models.budget import annotate_context_risks, check_context_budget
from lineage_agent.models.responses import ModelError
from lineage_agent.models.summarization import (
    SUMMARY_FIELDS, SummaryError, parse_summary, prepare_summary_request, summary_budget_settings,
)
from lineage_agent.models.types import ModelResult
from lineage_agent.recording.recorder import LOG_FAILURE_MESSAGE


type SummaryCall = Callable[[dict[str, Any], dict[str, Any]], Awaitable[ModelResult]]
MAX_COMPACT_PASSES = 2


def safe_compaction_error(reason: str) -> str:
    return "上下文恢复未通过检查，已保留原活动视图与实际执行记录。原因：" + reason


def _document_ids(document: dict) -> set[str]:
    return {ref for entries in document.values() for entry in entries for ref in entry["source_ids"]}


def _message(lane: str, selected: dict, document: dict) -> dict:
    # All machine entry points are supplied by the program, never by the summary model.
    content = {"kind": "derived_context_summary", "lane": lane,
               "summary_id": selected["summary_id"], "document": document,
               "history": {"history_ref": selected["scope"]["history_ref"],
                           "read_tool": "history_read", "search_tool": "history_search",
                           "scope": "compressed original interactions; includes uncited records"}}
    references = {ref: selected["source_metadata"][ref] for ref in _document_ids(document)
                  if ref in selected.get("source_metadata", {})}
    if references:
        content["source_references"] = references
    return {"role": "assistant", "content": json.dumps(content, ensure_ascii=False, separators=(",", ":"))}


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
            "net_reclaimed", "recovery_margin", "recovery_budget",
        )} | {"lanes": {lane: {key: selected.get(key) for key in (
            "summary_id", "status", "model_called", "target_bytes", "summary_bytes", "summary_call",
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
                == snapshot["active_version_id"])

    async def source_changed(state, compaction):
        # A newer confirmed view wins. Paid summary facts still belong to this Run.
        latest = await records.snapshot(records.confirmed())
        update = {field: deepcopy(latest[field]) for field in (
            "context_risks", "active_context", "history_end", "record_revision", "record_sequence",
            "record_admitted_through", "execution_facts", "run_call_ids", "history_scopes",
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
                for lane, selected in compaction["lanes"].items():
                    if selected["status"] == "empty":
                        continue
                    selected["candidate_input"] = {"document": deepcopy(selected["document"]),
                        "summary_id": selected["summary_id"], "checkpoint_id": records.confirmed()}
                    selected["summary_id"] = str(uuid4())
                    selected["allowed_source_ids"] = sorted(_document_ids(selected["document"]))
                    selected["model_called"] = False
            else:
                selected, reason = await records.select(state, settings)
                if selected is None:
                    if reason == "source_changed":
                        return await source_changed(state, compaction)
                    return stop(state, compaction, reason)
                compaction.update(selected)
                compaction["original_required"] = state["context_budget"]["required_tokens"]
                compaction["candidate_version_id"] = str(uuid4())
            lanes = {lane: selected for lane, selected in compaction["lanes"].items() if selected["status"] != "empty"}
            empty = {lane: {field: [] for field in SUMMARY_FIELDS} for lane in lanes}
            base = measure(state, compaction, empty)[0]["required_tokens"]
            margin = math.ceil(settings.context_budget_tokens * settings.context_recovery_margin_percent / 100)
            capacity = min(settings.context_budget_tokens - margin - base, compaction["original_required"] - base - 1)
            soft = (settings.context_budget_tokens - base) * settings.summary_target_percent // 100
            minima, weights = {}, {}
            for lane, selected in lanes.items():
                if not selected["allowed_source_ids"]:
                    return stop(state, compaction, "invalid_summary", failed=True)
                doc = deepcopy(empty[lane])
                doc["facts"] = [{"text": "x", "source_ids": [min(selected["allowed_source_ids"], key=len)]}]
                minima[lane] = json_bytes(_message(lane, selected, doc)) - json_bytes(_message(lane, selected, empty[lane]))
                weights[lane] = max(1, selected["source_bytes"])
            if capacity < sum(minima.values()):
                return stop(state, compaction, "protected_context_too_large")
            total = min(capacity, max(sum(minima.values()), soft))
            if retry:
                total = min(total, max(sum(minima.values()), sum(
                    json_bytes(_message(lane, selected, selected["document"])) - json_bytes(_message(lane, selected, empty[lane]))
                    for lane, selected in lanes.items()) * 3 // 4))
            remaining = total - sum(minima.values())
            allocations = {lane: minima[lane] + remaining * weights[lane] // sum(weights.values()) for lane in lanes}
            allocations["R" if "R" in allocations else "H"] += total - sum(allocations.values())
            for lane, selected in lanes.items():
                selected["target_bytes"] = allocations[lane] + json_bytes(_message(lane, selected, empty[lane]))
                if not retry and not selected["source_refs"] and len(selected["previous"]) == 1:
                    selected["document"] = await records.summary(selected["previous"][0])
                selected["status"] = "reused" if ("document" in selected and
                    json_bytes(_message(lane, selected, selected["document"])) <= selected["target_bytes"]) else "prepared"
                selected["model_called"] = False
            reusable = {lane: selected for lane, selected in lanes.items() if selected["status"] == "reused"}
            updating = {lane: selected for lane, selected in lanes.items() if selected["status"] != "reused"}
            if reusable and updating:
                reused_cost = sum(json_bytes(_message(lane, selected, selected["document"]))
                                  - json_bytes(_message(lane, selected, empty[lane])) for lane, selected in reusable.items())
                left = max(0, total - reused_cost - sum(minima[lane] for lane in updating))
                denominator = sum(weights[lane] for lane in updating)
                shares = {lane: minima[lane] + left * weights[lane] // denominator for lane in updating}
                shares["R" if "R" in updating else "H"] += total - reused_cost - sum(shares.values())
                for lane, selected in updating.items():
                    selected["target_bytes"] = shares[lane] + json_bytes(_message(lane, selected, empty[lane]))
            if retry and all(selected["status"] == "reused" for selected in lanes.values()):
                return stop(state, compaction, "no_shortening_space")
            compaction.update(status="preparing", pass_no=compaction["attempt_no"],
                              recovery_budget={"base_required": base, "hard_capacity": capacity, "soft_target": soft,
                                               "margin_required": margin, "target_bytes": total})
            scopes = deepcopy(state["history_scopes"])
            scopes.update({selected["scope"]["history_ref"]: selected["scope"] for selected in lanes.values()})
            emit("compaction.started", compaction["step_id"], request_no=compaction["request_no"],
                 attempt_no=compaction["attempt_no"], recovery_budget=compaction["recovery_budget"])
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
                material = {"candidate_input": selected["candidate_input"],
                            "sources": [{"source_id": ref, "kind": "prior_candidate_reference"}
                                        for ref in selected["allowed_source_ids"]]}
            else:
                material = await records.material(compaction, lane, state)
            # Optional H background is made only of whole entries, never evidence for R.
            if lane == "R" and compaction["lanes"]["H"]["status"] != "empty":
                hdoc = compaction["lanes"]["H"]["document"]
                background = {field: [] for field in SUMMARY_FIELDS}
                for field, entries in hdoc.items():
                    for entry in entries:
                        proposed = deepcopy(background); proposed[field].append(entry)
                        if json_bytes(proposed) <= 8192:
                            background = proposed
                material["history_background_readonly"] = {"document": background, "complete": background == hdoc}
            request = prepare_summary_request(settings, material, target_bytes=selected["target_bytes"], lane=lane)
            risks = annotate_context_risks(request["input"], fallback_tokens=settings.opaque_fallback_tokens,
                                           run_id=state["run_id"])
            budget = check_context_budget(request, risks, summary_budget_settings(settings), step_id=step_id)
            if budget["decision"] == "exceeded" and "history_background_readonly" in material:
                # Background is optional; remove whole entries until the actual request fits.
                background = material["history_background_readonly"]
                for field in reversed(SUMMARY_FIELDS):
                    while background["document"][field] and budget["decision"] == "exceeded":
                        background["document"][field].pop(); background["complete"] = False
                        request = prepare_summary_request(settings, material, target_bytes=selected["target_bytes"], lane=lane)
                        risks = annotate_context_risks(request["input"], fallback_tokens=settings.opaque_fallback_tokens, run_id=state["run_id"])
                        budget = check_context_budget(request, risks, summary_budget_settings(settings), step_id=step_id)
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
        compaction.update(summary_call=count, summary_lane=lane, summary_id=selected["summary_id"])
        selected.update(summary_call=count, model_called=True)
        if not emit("summary.started", step_id, lane=lane, summary_call=count, summary_id=selected["summary_id"],
                    request_no=compaction["request_no"], attempt_no=compaction["attempt_no"]):
            return stop(state, compaction, "logging_error", failed=True)
        budget["request_sent"] = True
        failure = None
        try:
            result = await call_summary(request, {**compaction, "lane": lane,
                                                "source_bytes": json_bytes(material)})
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
            parsed = parse_summary(result, allowed_source_ids=set(selected["allowed_source_ids"]),
                                   summary_id=selected["summary_id"], lane=lane)
            for source in _document_ids(parsed["document"]):
                await records.store.get(source, checkpoint_id=compaction["snapshot"]["source_checkpoint_id"])
        except CheckpointError:
            return stop(state, compaction, "invalid_summary", failed=True, update=update)
        except SummaryError as exc:
            return stop(state, compaction, exc.code, failed=True, update=update)
        except Exception:
            return stop(state, compaction, "invalid_summary", failed=True, update=update)
        selected.update(document=parsed["document"], status="generated",
                        summary_bytes=json_bytes(_message(lane, selected, parsed["document"])))
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
        docs = {lane: selected["document"] for lane, selected in compaction["lanes"].items() if selected["status"] != "empty"}
        try:
            budget, items, _, _ = measure(state, compaction, docs)
            if validate_input(items) is not None:
                return stop(state, compaction, "unsupported_protocol_boundary")
        except Exception:
            return stop(state, compaction, "context_measurement_error", failed=True)
        required = budget["required_tokens"]
        compaction.update(candidate_budget=budget, candidate_required=required,
                          net_reclaimed=compaction["original_required"] - required,
                          recovery_margin=settings.context_budget_tokens - required)
        reason = ("candidate_budget_exceeded" if budget["decision"] != "allow" else
                  "summary_not_effective" if compaction["net_reclaimed"] <= 0 else
                  "recovery_margin_insufficient" if compaction["recovery_margin"] < compaction["recovery_budget"]["margin_required"] else None)
        emit("context_budget.checked", compaction.get("step_id"), purpose="candidate",
             **{key: value for key, value in budget.items() if key != "step_id"})
        if reason and compaction["attempt_no"] == MAX_COMPACT_PASSES:
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
                for source in selected["allowed_source_ids"]:
                    await records.store.get(source, checkpoint_id=compaction["snapshot"]["source_checkpoint_id"])
        docs = {lane: selected["document"] for lane, selected in compaction["lanes"].items() if selected["status"] != "empty"}
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
