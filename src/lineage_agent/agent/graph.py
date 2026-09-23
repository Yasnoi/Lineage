"""Orchestrate one bounded ReAct loop with injected model and tool access."""

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Overwrite

from lineage_agent.agent.state import AgentState, CompactionState, UsageTotals
from lineage_agent.agent.compaction import SummaryCall, build_compaction_nodes
from lineage_agent.agent.checkpointing import CheckpointError
from lineage_agent.agent.context import ContextRecords
from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.budget import (
    annotate_context_risks, budget_measurement_error, check_context_budget,
)
from lineage_agent.models.responses import ModelError, safe_model_error
from lineage_agent.models.types import ModelCall, TokenUsage, ToolDefinition
from lineage_agent.recording.recorder import LOG_FAILURE_MESSAGE, RunRecorder
from lineage_agent.recording.summaries import summarize_value, tool_identity
from lineage_agent.tools.execution import execute_calls
from lineage_agent.tools.types import ToolExecutor, read_budget_snapshot

if TYPE_CHECKING:
    from lineage_agent.recording.tracing import RunTrace


def _accumulate_usage(totals: UsageTotals, usage: TokenUsage | None) -> UsageTotals:
    """Add only reported values, once per model attempt in this user request."""
    updated = totals.copy()
    complete = True
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(field) if usage is not None else None
        if value is None:
            complete = False
        else:
            previous = updated[field]
            updated[field] = value if previous is None else previous + value
    if not complete:
        updated["missing_usage_calls"] += 1
    return updated


def build_graph(
    call_model: ModelCall,
    *,
    prepare_model_request: Callable[..., dict[str, Any]],
    settings: ModelSettings,
    tools: list[ToolDefinition] | None = None,
    execute_tool: ToolExecutor | None = None,
    max_model_calls: int = 5,
    recorder: RunRecorder | None = None,
    tracer: "RunTrace | None" = None,
    checkpointer: BaseCheckpointSaver | None = None,
    observe_state: Callable[[AgentState, str, str | None], None] | None = None,
    call_summary: SummaryCall | None = None,
    context_records: ContextRecords | None = None,
    validate_input: Callable[[list[dict[str, Any]]], tuple[str, int] | None] | None = None,
) -> CompiledStateGraph:
    """Keep protocol parsing in models and execute each tool batch serially."""
    if type(max_model_calls) is not int or max_model_calls <= 0:
        raise ValueError("MAX_MODEL_CALLS 必须是正整数。")
    available_tools = tools or []

    def finish(state: AgentState, update: dict[str, Any], node: str, step_id: str | None) -> dict[str, Any]:
        # Checkpoint writes can fail before LangGraph yields a node's update.
        # Keep only this run's actual facts, never a second history store.
        if observe_state is not None:
            observed = {**state, **update}
            new_input = update.get("input", [])
            observed["input"] = new_input.value if isinstance(new_input, Overwrite) else state["input"] + new_input
            observe_state(deepcopy(observed), node, step_id)
        return update

    def emit(event: str, step_id: str | None, node: str, **details: Any) -> bool:
        return recorder is None or recorder.emit(
            event, step_id=step_id, node=node, **details
        )

    async def model_node(state: AgentState) -> dict[str, Any]:
        step_id = recorder.next_step() if recorder is not None else None
        if tracer is not None:
            tracer.bind_active_context(state["active_context"])
        compaction: CompactionState | None = deepcopy(state["compaction"])
        if compaction is not None and compaction["status"] == "pending_save":
            assert context_records is not None
            checkpoint_id = context_records.confirmed()
            saved = await context_records.saver.read_confirmed(context_records.session_id, checkpoint_id)
            saved_active = saved.checkpoint["channel_values"].get("active_context", {})
            if saved_active.get("version_id") != compaction["candidate_version_id"]:
                raise CheckpointError("checkpoint_read_error")
            compaction.update(status="confirmed", confirmed_checkpoint_id=checkpoint_id)
            # The fact observer also retains this confirmed acknowledgement if
            # the next log write fails before a business request is attempted.
            state = {**state, "compaction": compaction}
            if tracer is not None:
                tracer.observe_compaction(compaction)
            if not emit("compaction.activated", step_id, "model", summary_id=compaction["summary_id"],
                        version_id=state["active_context"]["version_id"], checkpoint_id=checkpoint_id):
                return finish(state, {"compaction": compaction, "stop_reason": "error",
                                     "error_code": "logging_error", "error": LOG_FAILURE_MESSAGE}, "model", step_id)
        if state["model_calls"] >= max_model_calls:
            return finish(state, {"stop_reason": "max_model_calls", "error": None, "error_code": None}, "model", step_id)
        remaining = max_model_calls - state["model_calls"]
        budget_error = None

        def budget_unavailable() -> None:
            nonlocal budget_error
            budget_error = "read_budget_unavailable"

        budget = read_budget_snapshot(execute_tool, on_unavailable=budget_unavailable)
        allowances = {
            "remaining_model_calls": remaining, "final_answer_call": remaining == 1,
            "read_budget": budget, "read_budget_error": budget_error,
        }
        instructions = f"本次用户请求剩余模型调用次数（包含当前调用）：{remaining}。"
        if budget is not None:
            instructions += f"本次运行剩余成功读取额度：{budget['remaining_bytes']} 字节。"
        if remaining == 1:
            instructions += "这是最后一次调用，不得请求工具；请依据已有资料回答，资料不足时明确说明。"
        try:
            request = prepare_model_request(
                state["input"], instructions=instructions,
                tool_choice="none" if remaining == 1 else None,
            )
            context_budget = check_context_budget(
                request, state["context_risks"], settings, step_id=step_id,
            )
        except Exception:
            context_budget = budget_measurement_error(settings, step_id=step_id)
        if tracer is not None:
            tracer.observe_read_budget("model", budget_error)
            tracer.observe_context_budget(context_budget)
        candidate_result_ids = [
            item["call_id"] for item in state["input"]
            if item.get("type") == "function_call_output"
        ]
        checked = emit(
            "context_budget.checked", step_id, "model",
            **{key: value for key, value in context_budget.items() if key != "step_id"},
            model_call_candidate=state["model_calls"] + 1,
            input_tool_result_call_ids=candidate_result_ids,
        )
        if not checked or context_budget["decision"] != "allow":
            exceeded = context_budget["decision"] == "exceeded"
            if checked and exceeded and context_records is not None and (
                compaction is None or compaction["request_no"] != state["model_calls"] + 1
            ):
                compaction = {
                    "request_no": state["model_calls"] + 1, "summary_id": str(uuid4()),
                    "status": "preparing", "reason": None,
                    "request_options": {"instructions": instructions, "tool_choice": "none" if remaining == 1 else None},
                    "old_version_id": state["active_context"]["version_id"],
                }
                return finish(state, {
                    "context_budget": context_budget,
                    "compaction": compaction,
                }, "model", step_id)
            if checked and exceeded:
                if compaction is None:
                    compaction = {"request_no": state["model_calls"] + 1, "status": "stopped", "reason": "no_eligible_history"}
                else:
                    compaction.update(reason="recovery_exhausted")
            return finish(state, {
                "context_budget": context_budget,
                "compaction": compaction,
                "stop_reason": "context_budget_exceeded" if checked and exceeded else "error",
                "error_code": context_budget["error_code"] if checked else "logging_error",
                "error": (
                    LOG_FAILURE_MESSAGE if not checked else
                    "上下文预算未恢复至可发送范围，本次模型请求未发送。" if exceeded else
                    "无法完成请求预算计量，本次模型请求未发送。"
                ),
            }, "model", step_id)
        if not emit(
            "model.started", step_id, "model",
            model_call=state["model_calls"] + 1,
            active_version_id=state["active_context"]["version_id"],
            **allowances,
            input_item_count=len(state["input"]),
            input_tool_result_call_ids=candidate_result_ids,
        ):
            return finish(state, {
                "context_budget": context_budget, "stop_reason": "error",
                "compaction": compaction,
                "error": LOG_FAILURE_MESSAGE, "error_code": "logging_error",
            }, "model", step_id)

        failure: tuple[str, str] | None = None
        context_budget["request_sent"] = True
        if compaction is not None and compaction.get("request_no") == state["model_calls"] + 1 and compaction["status"] == "confirmed":
            compaction["status"] = "used"
        if tracer is not None:
            tracer.observe_context_budget(context_budget)
            tracer.set_model_step(step_id, state["model_calls"] + 1, **allowances)
        try:
            result = await call_model(request)
        except ModelError as exc:
            result = exc.result
            failure = (exc.code, exc.stage)
        except Exception:
            # Injected callables can fail outside the SDK's exception types.
            result = None
            failure = ("model_error", "model_request")

        model_calls = state["model_calls"] + 1
        output_items = result["output_items"] if result is not None else []
        raw_items = result["raw_response"].get("output", []) if result is not None else []
        requests = [
            item for item in raw_items if isinstance(item, dict) and item.get("type") == "function_call"
        ] if isinstance(raw_items, list) else []
        update: dict[str, Any] = {
            "input": output_items,
            "context_budget": context_budget,
            "compaction": compaction,
            "context_risks": state["context_risks"] + annotate_context_risks(
                output_items, fallback_tokens=settings.opaque_fallback_tokens,
                run_id=state["run_id"], model_call=model_calls,
                usage=result["usage"] if result is not None else None,
            ),
            # New user input becomes inheritable only once actually dispatched.
            # Later tool batches advance the boundary as complete protocol groups.
            "history_end": len(state["input"]),
            "result": result,
            "model_calls": model_calls,
            "tool_requests": state["tool_requests"] + len(requests),
            "tool_outcomes": state["tool_outcomes"].copy(),
            "usage_totals": _accumulate_usage(
                state["usage_totals"], result["usage"] if result is not None else None
            ),
            "stop_reason": None,
            "error": None,
            "error_code": None,
        }
        if result is not None:
            emit(
                "model.returned", step_id, "model", model_call=model_calls,
                text_summary=summarize_value(result["text"]),
                output_item_count=len(output_items), tool_request_count=len(requests),
                tool_requests=[
                    {**tool_identity(call, index),
                     "arguments_summary": summarize_value(call.get("arguments"))}
                    for index, call in enumerate(requests, 1)
                ],
                usage=result["usage"],
                has_refusal=result.get("has_refusal", False),
            )

        if failure is None and result is None:
            failure = ("invalid_response", "response_validation")
        if failure is None and result is not None:
            calls = result["tool_calls"]
            if calls:
                previous_ids = {
                    item["call_id"] for item in state["input"]
                    if item.get("type") == "function_call"
                }
                ids = [call["call_id"] for call in calls]
                if len(set(ids)) != len(ids) or any(value in previous_ids for value in ids):
                    failure = ("duplicate_call_id", "response_validation")
                elif model_calls >= max_model_calls:
                    update["stop_reason"] = "max_model_calls"
            elif result["text"].strip():
                update["stop_reason"] = "completed"
                update["history_end"] = len(state["input"]) + len(output_items)
            else:
                failure = ("empty_response", "response_validation")

        if failure is not None:
            code, explanation, stage = safe_model_error(failure[0])
            emit("model.failed", step_id, "model", error_code=code, stage=stage)
            update.update(
                stop_reason="error",
                error=explanation, error_code=code,
            )
        if recorder is not None and recorder.failed:
            update.update(stop_reason="error", error=LOG_FAILURE_MESSAGE, error_code="logging_error")

        if update["stop_reason"] is not None and requests:
            reason = (
                "logging_error" if recorder is not None and recorder.failed
                else "max_model_calls" if update["stop_reason"] == "max_model_calls"
                else "protocol_error"
            )
            for index, call in enumerate(requests, 1):
                update["tool_outcomes"]["skipped"] += 1
                emit(
                    "tool.skipped", step_id, "model",
                    **tool_identity(call, index), stage="routing", reason=reason,
                )
                if tracer is not None:
                    tracer.tool_skipped(
                        call, step_id=step_id, request_index=index, reason=reason,
                    )
        if recorder is not None and recorder.failed:
            update.update(stop_reason="error", error=LOG_FAILURE_MESSAGE, error_code="logging_error")
        if tracer is not None:
            trace_failure = failure
            if trace_failure is None and recorder is not None and recorder.failed:
                trace_failure = ("logging_error", "recording")
            tracer.model_complete(
                error_code=trace_failure[0] if trace_failure else None,
                stage=trace_failure[1] if trace_failure else None,
            )
        if compaction is not None and compaction.get("request_no") == model_calls and compaction["status"] == "used":
            emit("compaction.used", step_id, "model", summary_id=compaction["summary_id"],
                 version_id=state["active_context"]["version_id"],
                 checkpoint_id=compaction["confirmed_checkpoint_id"], model_call=model_calls)
            if tracer is not None:
                tracer.observe_compaction(compaction)
            if recorder is not None and recorder.failed:
                update.update(stop_reason="error", error=LOG_FAILURE_MESSAGE, error_code="logging_error")
        return finish(state, update, "model", step_id)

    async def tools_node(state: AgentState) -> dict[str, Any]:
        step_id = recorder.next_step() if recorder is not None else None
        result = state["result"]
        assert result is not None
        batch = await execute_calls(
            result["tool_calls"], available_tools, execute_tool,
            recorder=recorder, step_id=step_id,
            tracer=tracer,
        )
        update: dict[str, Any] = {
            "input": batch["output_items"],
            "context_risks": state["context_risks"] + annotate_context_risks(
                batch["output_items"], fallback_tokens=settings.opaque_fallback_tokens,
                run_id=state["run_id"],
            ),
            "tool_calls": state["tool_calls"] + batch["tool_calls"],
            "tool_outcomes": {
                name: state["tool_outcomes"][name] + count
                for name, count in batch["tool_outcomes"].items()
            },
        }
        if batch["error"] is not None:
            update.update(stop_reason="error", error=batch["error"], error_code="logging_error")
        # A complete batch includes genuine rejection/failure results. A partial
        # batch must not carry its entire response group into the next turn.
        if [item["call_id"] for item in batch["output_items"]] == [
            call["call_id"] for call in result["tool_calls"]
        ]:
            update["history_end"] = len(state["input"]) + len(batch["output_items"])
        return finish(state, update, "tools", step_id)

    def after_model(state: AgentState) -> str:
        if state["stop_reason"] is not None:
            return END
        if (state["compaction"] or {}).get("status") == "preparing":
            return "summary"
        return "tools"

    def after_tools(state: AgentState) -> str:
        return END if state["stop_reason"] is not None else "model"

    builder = StateGraph(AgentState)
    builder.add_node("model", model_node)
    builder.add_node("tools", tools_node)
    summarize, activate = build_compaction_nodes(
        settings=settings, records=context_records, call_summary=call_summary,
        prepare_model_request=prepare_model_request, recorder=recorder, tracer=tracer,
        finish=finish, accumulate_usage=_accumulate_usage,
        validate_input=validate_input or (lambda items: None),
    )
    builder.add_node("summary", summarize)
    builder.add_node("activate_context", activate)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", after_model, {"tools": "tools", "summary": "summary", END: END})
    builder.add_conditional_edges("tools", after_tools, {"model": "model", END: END})
    builder.add_conditional_edges("summary", lambda state: END if state["stop_reason"] else "activate_context")
    builder.add_conditional_edges("activate_context", lambda state: END if state["stop_reason"] else "model")
    return builder.compile(checkpointer=checkpointer)
