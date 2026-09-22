"""Orchestrate one bounded ReAct loop with injected model and tool access."""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from lineage_agent.agent.state import AgentState, UsageTotals
from lineage_agent.models.responses import ModelError
from lineage_agent.models.types import ModelCall, TokenUsage, ToolDefinition
from lineage_agent.recording.recorder import LOG_FAILURE_MESSAGE, RunRecorder
from lineage_agent.recording.summaries import summarize_value, tool_identity
from lineage_agent.tools.execution import execute_calls
from lineage_agent.tools.types import ToolExecutor


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
    tools: list[ToolDefinition] | None = None,
    execute_tool: ToolExecutor | None = None,
    max_model_calls: int = 5,
    recorder: RunRecorder | None = None,
) -> CompiledStateGraph:
    """Keep protocol parsing in models and execute each tool batch serially."""
    if type(max_model_calls) is not int or max_model_calls <= 0:
        raise ValueError("MAX_MODEL_CALLS 必须是正整数。")
    available_tools = tools or []

    def emit(event: str, step_id: str | None, node: str, **details: Any) -> bool:
        return recorder is None or recorder.emit(
            event, step_id=step_id, node=node, **details
        )

    async def model_node(state: AgentState) -> dict[str, Any]:
        step_id = recorder.next_step() if recorder is not None else None
        if state["model_calls"] >= max_model_calls:
            return {"stop_reason": "max_model_calls", "error": None}
        if not emit(
            "model.started", step_id, "model",
            model_call=state["model_calls"] + 1,
            input_item_count=len(state["input"]),
            input_tool_result_call_ids=[
                item["call_id"] for item in state["input"]
                if item.get("type") == "function_call_output"
            ],
        ):
            return {"stop_reason": "error", "error": LOG_FAILURE_MESSAGE}

        failure: tuple[str, str] | None = None
        try:
            result = await call_model(state["input"])
        except ModelError as exc:
            result = exc.result
            failure = (exc.code, exc.stage)
        except Exception:
            # Injected callables can fail outside the SDK's exception types.
            result = None
            failure = ("model_error", "model_request")

        model_calls = state["model_calls"] + 1
        output_items = result["output_items"] if result is not None else []
        requests = [item for item in output_items if item.get("type") == "function_call"]
        update: dict[str, Any] = {
            "input": output_items,
            "result": result,
            "model_calls": model_calls,
            "tool_requests": state["tool_requests"] + len(requests),
            "tool_outcomes": state["tool_outcomes"].copy(),
            "usage_totals": _accumulate_usage(
                state["usage_totals"], result["usage"] if result is not None else None
            ),
            "stop_reason": None,
            "error": None,
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
            else:
                failure = ("empty_response", "response_validation")

        if failure is not None:
            code, stage = failure
            emit("model.failed", step_id, "model", error_code=code, stage=stage)
            update.update(
                stop_reason="error",
                error="模型响应未通过协议校验。" if stage == "response_validation"
                else "模型调用失败。",
            )
        if recorder is not None and recorder.failed:
            update.update(stop_reason="error", error=LOG_FAILURE_MESSAGE)

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
        if recorder is not None and recorder.failed:
            update.update(stop_reason="error", error=LOG_FAILURE_MESSAGE)
        return update

    async def tools_node(state: AgentState) -> dict[str, Any]:
        step_id = recorder.next_step() if recorder is not None else None
        result = state["result"]
        assert result is not None
        batch = await execute_calls(
            result["tool_calls"], available_tools, execute_tool,
            recorder=recorder, step_id=step_id,
        )
        update: dict[str, Any] = {
            "input": batch["output_items"],
            "tool_calls": state["tool_calls"] + batch["tool_calls"],
            "tool_outcomes": {
                name: state["tool_outcomes"][name] + count
                for name, count in batch["tool_outcomes"].items()
            },
        }
        if batch["error"] is not None:
            update.update(stop_reason="error", error=batch["error"])
        return update

    def after_model(state: AgentState) -> str:
        return END if state["stop_reason"] is not None else "tools"

    def after_tools(state: AgentState) -> str:
        return END if state["stop_reason"] is not None else "model"

    builder = StateGraph(AgentState)
    builder.add_node("model", model_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", after_model, {"tools": "tools", END: END})
    builder.add_conditional_edges("tools", after_tools, {"model": "model", END: END})
    return builder.compile()
