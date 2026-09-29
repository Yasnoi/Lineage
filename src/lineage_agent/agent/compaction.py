"""At most two saved summary attempts per over-budget business request."""

from collections.abc import Awaitable, Callable
from copy import deepcopy
from hashlib import sha256
import json
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langgraph.types import Overwrite

from lineage_agent.agent.checkpointing import CheckpointError
from lineage_agent.agent.context import ContextRecords, budget_for_retained
from lineage_agent.agent.state import ActiveContext, AgentState, CompactionState, UsageTotals
from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.budget import annotate_context_risks, check_context_budget
from lineage_agent.models.responses import ModelError
from lineage_agent.models.summarization import (
    SummaryError, minimum_summary_message_bytes, parse_summary,
    prepare_summary_request, summary_budget_settings,
)
from lineage_agent.models.types import ModelResult, TokenUsage
from lineage_agent.recording.recorder import LOG_FAILURE_MESSAGE, RunRecorder

if TYPE_CHECKING:
    from lineage_agent.recording.tracing import RunTrace


type SummaryCall = Callable[[dict[str, Any], dict[str, Any]], Awaitable[ModelResult]]
type CompactionNode = Callable[[AgentState], Awaitable[dict[str, Any]]]

MAX_COMPACT_PASSES = 2


def _source_headers(material: dict[str, Any], allowed: list[str]) -> list[dict[str, Any]]:
    """Keep role provenance for retrying a candidate without caching originals."""
    headers = [{key: source[key] for key in ("source_id", "role", "kind")}
               for source in material["sources"]]
    identified = {header["source_id"] for header in headers}
    identified.update(outcome["source_id"] for outcome in material["turn_outcomes"])
    headers.extend({"source_id": source, "kind": "prior_summary_reference"}
                   for source in allowed if source not in identified)
    return headers


def safe_compaction_error(reason: str) -> str:
    return {
        "context_measurement_error": "摘要或候选请求无法完成预算计量，已停止。",
        "summary_model_error": "摘要模型调用失败，已有调用与消耗予以保留。",
        "invalid_summary": "摘要响应未通过结构或来源校验，未启用候选版本。",
        "summary_refusal": "摘要模型拒绝生成摘要，未启用候选版本。",
        "summary_tool_request": "摘要响应包含工具请求，未执行该请求或启用候选版本。",
    }.get(reason, "上下文预算无法通过本次有界压缩恢复，未发起对应业务请求。")


def build_compaction_nodes(
    *, settings: ModelSettings, records: ContextRecords | None,
    call_summary: SummaryCall | None,
    prepare_model_request: Callable[..., dict[str, Any]],
    recorder: RunRecorder | None, tracer: "RunTrace | None",
    finish: Callable[[AgentState, dict[str, Any], str, str | None], dict[str, Any]],
    accumulate_usage: Callable[[UsageTotals, TokenUsage | None], UsageTotals],
    validate_input: Callable[[list[dict[str, Any]]], tuple[str, int] | None],
) -> tuple[CompactionNode, CompactionNode]:
    """Nodes preserve the old version until a validated activation is saved."""
    def emit(event: str, step_id: str | None, *, node: str = "summary", **details: Any) -> bool:
        return recorder is None or recorder.emit(event, step_id=step_id, node=node, **details)

    def stop(
        state: AgentState, compaction: CompactionState, reason: str, step_id: str | None,
        *, failed: bool = False, update: dict[str, Any] | None = None, node: str = "summary",
    ) -> dict[str, Any]:
        compaction.update(status="failed" if failed else "stopped", reason=reason)
        values = dict(update or {})
        values.update(
            compaction=compaction,
            stop_reason="error" if failed else "context_budget_exceeded",
            error_code=reason if failed else "context_budget_exceeded",
            error=safe_compaction_error(reason),
        )
        emit("compaction.stopped", step_id, node=node, summary_id=compaction["summary_id"],
             request_no=compaction["request_no"], pass_no=compaction["pass_no"], reason=reason)
        if recorder is not None and recorder.failed:
            values.update(stop_reason="error", error_code="logging_error", error=LOG_FAILURE_MESSAGE)
        if tracer is not None:
            tracer.observe_compaction(compaction)
        return finish(state, values, node, step_id)

    async def summarize(state: AgentState) -> dict[str, Any]:
        step_id = recorder.next_step() if recorder is not None else None
        compaction = deepcopy(state["compaction"])
        assert compaction is not None
        compaction["step_id"] = step_id
        options = compaction["request_options"]
        if records is None:
            return stop(state, compaction, "no_eligible_history", step_id)
        try:
            if compaction["pass_no"] == 1:
                selected, reason = await records.select(
                    state, settings, prepare_model_request, summary_id=compaction["summary_id"], **options,
                )
                if selected is None:
                    assert reason is not None
                    return stop(state, compaction, reason, step_id, failed=reason == "context_measurement_error")
                material = selected["material"]
                compaction.update(
                    sources=selected["sources"], allowed_source_ids=selected["allowed_source_ids"],
                    removed_items=selected["removed_items"], removed_turns=selected["removed_turns"],
                    recovery_budget=selected["recovery_budget"],
                    source_headers=_source_headers(material, selected["allowed_source_ids"]),
                    outcome_headers=deepcopy(material["turn_outcomes"]),
                    opaque_omissions=deepcopy(material["opaque_omissions"]),
                )
            else:
                assert compaction["pass_no"] == MAX_COMPACT_PASSES
                ref = compaction["candidate_input"]
                saved = await records.saver.read_confirmed(records.session_id, ref["checkpoint_id"])
                previous = saved.checkpoint["channel_values"].get("compaction", {})
                if (saved.checkpoint["channel_values"].get("run_id") != state["run_id"]
                        or previous.get("summary_id") != ref["summary_id"]
                        or previous.get("request_no") != compaction["request_no"]
                        or previous.get("pass_no") != 1 or previous.get("status") != "candidate"
                        or not isinstance(previous.get("document"), dict)):
                    raise CheckpointError("checkpoint_read_error")
                document = deepcopy(previous["document"])
                allowed = {source for section in document.values() for entry in section for source in entry["source_ids"]}
                compaction["allowed_source_ids"] = sorted(allowed)
                recovery = budget_for_retained(
                    state, settings, prepare_model_request, compaction["removed_items"], options,
                )
                compaction["recovery_budget"] = recovery
                minimum = minimum_summary_message_bytes(compaction["summary_id"], allowed)
                minimum += int(bool(state["input"][compaction["removed_items"]:]))
                if recovery["available_tokens"] <= 0 or recovery["target_tokens"] < minimum:
                    return stop(state, compaction, "protected_context_too_large", step_id)
                # Only this confirmed candidate supplies body content. Original
                # lineage remains unchanged and the old active summary is not
                # mislabelled as having been replaced by the first candidate.
                material = {
                    "candidate_input": {**ref, "document": document},
                    "sources": [deepcopy(header) for header in compaction["source_headers"]
                                if header["source_id"] in allowed],
                    "turn_outcomes": [deepcopy(header) for header in compaction["outcome_headers"]
                                      if header["source_id"] in allowed],
                    "opaque_omissions": [deepcopy(header) for header in compaction["opaque_omissions"]
                                         if header["source_id"] in allowed],
                }
        except CheckpointError:
            raise
        except Exception:
            return stop(state, compaction, "context_measurement_error", step_id, failed=True)
        if not emit("compaction.started", step_id, summary_id=compaction["summary_id"],
                    request_no=compaction["request_no"], pass_no=compaction["pass_no"],
                    sources=compaction["sources"], candidate_input=compaction.get("candidate_input"),
                    recovery_budget=compaction["recovery_budget"],
                    old_version_id=state["active_context"]["version_id"]):
            return stop(state, compaction, "logging_error", step_id, failed=True)
        try:
            target = compaction["recovery_budget"]["target_tokens"]
            request = prepare_summary_request(settings, material, target_tokens=target)
            compaction["summary_request_sha256"] = sha256(json.dumps(
                request, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest()
            risks = annotate_context_risks(request["input"], fallback_tokens=settings.opaque_fallback_tokens,
                                           run_id=state["run_id"])
            budget = check_context_budget(request, risks, summary_budget_settings(settings, target), step_id=step_id)
        except Exception:
            return stop(state, compaction, "context_measurement_error", step_id, failed=True)
        compaction["summary_budget"] = budget
        if not emit("context_budget.checked", step_id, purpose="summary", summary_id=compaction["summary_id"],
                    **{key: value for key, value in budget.items() if key != "step_id"}):
            return stop(state, compaction, "logging_error", step_id, failed=True)
        if budget["decision"] != "allow":
            return stop(state, compaction,
                        "summary_budget_exceeded" if budget["decision"] == "exceeded" else "context_measurement_error",
                        step_id, failed=budget["decision"] == "error")
        count = state["summary_calls"] + 1
        compaction["summary_call"] = count
        if not emit("summary.started", step_id, summary_id=compaction["summary_id"],
                    summary_call=count, request_no=compaction["request_no"], pass_no=compaction["pass_no"],
                    sources=compaction["sources"], candidate_input=compaction.get("candidate_input")):
            return stop(state, compaction, "logging_error", step_id, failed=True)
        budget["request_sent"] = True
        failure = None
        metadata = {**compaction, "summary_call": count,
                    "source_bytes": len(request["input"][0]["content"].encode("utf-8"))}
        try:
            assert call_summary is not None
            result = await call_summary(request, metadata)
        except ModelError as exc:
            result, failure = exc.result, "summary_model_error"
            compaction["cause_code"] = exc.code
        except Exception:
            result, failure = None, "summary_model_error"
        usage = result["usage"] if result is not None else None
        update: dict[str, Any] = {
            "summary_calls": count, "summary_result": result,
            "summary_usage_totals": accumulate_usage(state["summary_usage_totals"], usage),
            "usage_totals": accumulate_usage(state["usage_totals"], usage),
        }
        if result is not None:
            emit("summary.returned", step_id, summary_id=compaction["summary_id"],
                 summary_call=count, usage=usage, text_length=len(result["text"]),
                 has_refusal=result.get("has_refusal", False))
        if failure is not None:
            emit("summary.failed", step_id, summary_id=compaction["summary_id"],
                 summary_call=count, error_code=failure, cause_code=compaction.get("cause_code"))
            return stop(state, compaction, failure, step_id, failed=True, update=update)
        try:
            parsed = parse_summary(result, allowed_source_ids=set(compaction["allowed_source_ids"]),
                                   summary_id=compaction["summary_id"])
        except SummaryError as exc:
            reason = "context_measurement_error" if exc.code == "summary_measurement_error" else exc.code
            emit("summary.failed", step_id, summary_id=compaction["summary_id"],
                 summary_call=count, error_code=reason)
            return stop(state, compaction, reason, step_id, failed=True, update=update)
        except Exception:
            return stop(state, compaction, "invalid_summary", step_id, failed=True, update=update)
        compaction.update(status="candidate", document=parsed["document"], message=parsed["message"],
                          summary_bytes=parsed["bytes"], candidate_version_id=str(uuid4()))
        emit("compaction.candidate", step_id, summary_id=compaction["summary_id"],
             candidate_version_id=compaction["candidate_version_id"], summary_bytes=parsed["bytes"])
        if recorder is not None and recorder.failed:
            return stop(state, compaction, "logging_error", step_id, failed=True, update=update)
        update["compaction"] = compaction
        if tracer is not None:
            tracer.observe_compaction(compaction)
        return finish(state, update, "summary", step_id)

    async def activate(state: AgentState) -> dict[str, Any]:
        step_id = recorder.next_step() if recorder is not None else None
        compaction = deepcopy(state["compaction"])
        assert compaction is not None and records is not None
        summary_checkpoint_id = records.confirmed()
        saved = await records.saver.read_confirmed(records.session_id, summary_checkpoint_id)
        saved_candidate = saved.checkpoint["channel_values"].get("compaction")
        if (saved.checkpoint["channel_values"].get("run_id") != state["run_id"]
                or not isinstance(saved_candidate, dict)
                or saved_candidate.get("summary_id") != compaction["summary_id"]
                or saved_candidate.get("request_no") != compaction["request_no"]
                or saved_candidate.get("pass_no") != compaction["pass_no"]
                or saved_candidate.get("status") != "candidate"):
            raise CheckpointError("checkpoint_read_error")
        removed = compaction["removed_items"]
        items = [deepcopy(compaction["message"])] + deepcopy(state["input"][removed:])
        risks = annotate_context_risks(items[:1], fallback_tokens=settings.opaque_fallback_tokens,
                                      run_id=state["run_id"]) + deepcopy(state["context_risks"][removed:])
        if validate_input(items) is not None:
            return stop(state, compaction, "invalid_summary", step_id, failed=True, node="activate_context")
        try:
            request = prepare_model_request(items, **compaction["request_options"])
            budget = check_context_budget(request, risks, settings, step_id=step_id)
        except Exception:
            return stop(state, compaction, "context_measurement_error", step_id, failed=True, node="activate_context")
        compaction["candidate_budget"] = budget
        if not emit("context_budget.checked", step_id, node="activate_context", purpose="candidate", summary_id=compaction["summary_id"],
                    **{key: value for key, value in budget.items() if key != "step_id"}):
            return stop(state, compaction, "logging_error", step_id, failed=True, node="activate_context")
        if budget["decision"] == "error":
            return stop(state, compaction, "context_measurement_error", step_id, failed=True, node="activate_context")
        if budget["decision"] == "exceeded":
            if compaction["pass_no"] >= MAX_COMPACT_PASSES:
                return stop(state, compaction, "recovery_exhausted", step_id, node="activate_context")
            try:
                recovery = budget_for_retained(
                    state, settings, prepare_model_request, removed, compaction["request_options"],
                )
            except Exception:
                return stop(state, compaction, "context_measurement_error", step_id, failed=True, node="activate_context")
            if recovery["available_tokens"] <= 0:
                return stop(state, compaction, "protected_context_too_large", step_id, node="activate_context")
            compaction.update(status="retry_pending", summary_checkpoint_id=summary_checkpoint_id)
            if not emit("compaction.retry_pending", step_id, node="activate_context",
                        summary_id=compaction["summary_id"], request_no=compaction["request_no"],
                        pass_no=1, checkpoint_id=summary_checkpoint_id):
                return stop(state, compaction, "logging_error", step_id, failed=True, node="activate_context")
            if tracer is not None:
                tracer.observe_compaction(compaction)
            return finish(state, {"compaction": compaction}, "activate_context", step_id)
        if budget["required_tokens"] >= state["context_budget"]["required_tokens"]:
            return stop(state, compaction, "summary_not_effective", step_id, node="activate_context")
        previous = state["active_context"]
        active: ActiveContext = {
            "version_id": compaction["candidate_version_id"], "parent_version_id": previous["version_id"],
            "summary": {"summary_id": compaction["summary_id"], "checkpoint_id": summary_checkpoint_id},
            "turns": deepcopy(previous["turns"][compaction["removed_turns"]:]),
            "turn_start": previous["turn_start"] + 1 - removed,
        }
        compaction.update(status="pending_save", summary_checkpoint_id=summary_checkpoint_id)
        if tracer is not None:
            tracer.observe_compaction(compaction)
        return finish(state, {
            "input": Overwrite(items), "context_risks": risks,
            "active_context": active, "compaction": compaction,
            "history_end": state["history_end"] + 1 - removed,
        }, "activate_context", step_id)

    return summarize, activate
