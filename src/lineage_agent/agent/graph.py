"""Orchestrate one bounded ReAct loop with injected model and tool access."""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from lineage_agent.agent.state import AgentState, UsageTotals
from lineage_agent.models.responses import ModelError
from lineage_agent.models.types import ModelCall, TokenUsage, ToolDefinition
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
) -> CompiledStateGraph:
    """Keep protocol parsing in models and execute each tool batch serially."""
    if type(max_model_calls) is not int or max_model_calls <= 0:
        raise ValueError("MAX_MODEL_CALLS 必须是正整数。")
    available_tools = tools or []

    async def model_node(state: AgentState) -> dict[str, Any]:
        if state["model_calls"] >= max_model_calls:
            return {"stop_reason": "max_model_calls", "error": None}

        failure = None
        try:
            result = await call_model(state["input"])
        except ModelError as exc:
            result = exc.result
            failure = str(exc)

        model_calls = state["model_calls"] + 1
        output_items = result["output_items"] if result is not None else []
        # Count received requests even when their protocol metadata is invalid.
        request_count = sum(item.get("type") == "function_call" for item in output_items)
        update: dict[str, Any] = {
            "input": output_items,
            "result": result,
            "model_calls": model_calls,
            "tool_requests": state["tool_requests"] + request_count,
            "usage_totals": _accumulate_usage(
                state["usage_totals"], result["usage"] if result is not None else None
            ),
            "stop_reason": None,
            "error": None,
        }
        if failure is not None:
            update.update(stop_reason="error", error=failure)
            return update

        if result is None:
            update.update(stop_reason="error", error="模型未返回有效结果。")
            return update

        calls = result["tool_calls"]
        if calls:
            previous_ids = {
                item.get("call_id")
                for item in state["input"]
                if item.get("type") == "function_call"
            }
            ids = [call["call_id"] for call in calls]
            if len(set(ids)) != len(ids) or any(call_id in previous_ids for call_id in ids):
                update.update(stop_reason="error", error="模型重复使用 call_id，无法关联工具结果。")
            elif model_calls >= max_model_calls:
                # Do not execute tools whose results cannot reach another model call.
                update["stop_reason"] = "max_model_calls"
        elif result["text"].strip():
            update["stop_reason"] = "completed"
        else:
            update.update(stop_reason="error", error="模型既未给出回答，也未请求工具。")
        return update

    async def tools_node(state: AgentState) -> dict[str, Any]:
        result = state["result"]
        assert result is not None  # Only successful responses with tool calls route here.
        batch = await execute_calls(result["tool_calls"], available_tools, execute_tool)
        return {
            "input": batch["output_items"],
            "tool_calls": state["tool_calls"] + batch["tool_calls"],
        }

    def after_model(state: AgentState) -> str:
        return END if state["stop_reason"] is not None else "tools"

    builder = StateGraph(AgentState)
    builder.add_node("model", model_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", after_model, {"tools": "tools", END: END})
    builder.add_edge("tools", "model")
    return builder.compile()
