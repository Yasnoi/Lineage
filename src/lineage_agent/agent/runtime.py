"""Assemble dependencies and run one agent execution."""

from copy import deepcopy
from functools import partial
from typing import Any

from langsmith import tracing_context

from lineage_agent.agent.graph import build_graph
from lineage_agent.agent.state import AgentState
from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.models.responses import create_client, generate_response
from lineage_agent.models.types import ModelInput, ToolDefinition
from lineage_agent.recording.recorder import LOG_FAILURE_MESSAGE, RunRecorder
from lineage_agent.recording.tracing import start_trace
from lineage_agent.tools.types import ToolExecutor


def _input_association_error(items: list[dict[str, Any]]) -> tuple[str, int] | None:
    """Validate only the local function call/result associations before sending."""
    requested: dict[str, int] = {}
    returned: set[str] = set()
    for index, item in enumerate(items, 1):
        kind = item.get("type")
        if kind not in ("function_call", "function_call_output"):
            continue
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id.strip():
            return "invalid_call_id", index
        if kind == "function_call":
            if call_id in requested:
                return "duplicate_call_id", index
            requested[call_id] = index
        else:
            if call_id not in requested:
                return "unmatched_tool_result", index
            if call_id in returned:
                return "duplicate_tool_result", index
            returned.add(call_id)
    for call_id, index in requested.items():
        if call_id not in returned:
            return "missing_tool_result", index
    return None


async def run_agent(
    input: ModelInput,
    settings: ModelSettings,
    *,
    tools: list[ToolDefinition] | None = None,
    execute_tool: ToolExecutor | None = None,
) -> AgentState:
    """Handle one user request; all counters and usage start fresh on every call."""
    if isinstance(input, str):
        if not input.strip():
            raise ValueError("用户输入不能为空。")
        input_items = [{"role": "user", "content": input}]
    elif isinstance(input, list) and input and all(isinstance(item, dict) for item in input):
        input_items = deepcopy(input)
    else:
        raise ValueError("input 必须是非空字符串或非空 Responses 协议项列表。")

    available_tools = deepcopy(tools) if tools is not None else []
    names: set[str] = set()
    for tool in available_tools:
        if (
            not isinstance(tool, dict)
            or tool.get("type") != "function"
            or not isinstance(tool.get("name"), str)
            or not tool["name"].strip()
            or not isinstance(tool.get("parameters"), dict)
        ):
            raise ConfigurationError("工具定义必须使用包含名称和 JSON Schema 参数的 Responses function 格式。")
        if tool["name"] in names:
            raise ConfigurationError("同一次运行中的工具名称不能重复。")
        names.add(tool["name"])
    if available_tools and not callable(execute_tool):
        raise ConfigurationError("提供工具定义时必须同时提供工具执行器。")

    recorder = RunRecorder(settings.log_dir)
    state: AgentState = {
        "input": input_items,
        "run_id": recorder.run_id,
        "log_path": str(recorder.path),
        "log_complete": False,
        "result": None,
        "model_calls": 0,
        "tool_requests": 0,
        "tool_calls": 0,
        "tool_outcomes": {"succeeded": 0, "rejected": 0, "failed": 0, "skipped": 0},
        "usage_totals": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "missing_usage_calls": 0,
        },
        "stop_reason": None,
        "error": None,
    }
    tracer = start_trace(recorder.run_id, settings)
    try:
        if not recorder.emit(
            "run.started", input_item_count=len(input_items),
            max_model_calls=settings.max_model_calls,
        ):
            state.update(stop_reason="error", error=LOG_FAILURE_MESSAGE)
        elif invalid := _input_association_error(input_items):
            code, index = invalid
            recorder.emit(
                "run.input_rejected", stage="input_validation",
                error_code=code, input_index=index,
            )
            state.update(stop_reason="error", error="初始上下文中的工具请求与结果无法唯一关联。")
        else:
            async with create_client(settings) as client:
                call_model = partial(
                    generate_response, client, settings.model_name,
                    tools=available_tools or None, tracer=tracer,
                )
                graph = build_graph(
                    call_model,
                    tools=available_tools,
                    execute_tool=execute_tool,
                    max_model_calls=settings.max_model_calls,
                    recorder=recorder,
                    tracer=tracer,
                )
                # Only our explicit, filtered spans may leave the process. Environment
                # flags must not enable LangGraph's automatic full-State recording.
                with tracing_context(enabled=False, parent=False):
                    state = await graph.ainvoke(
                        state, config={"recursion_limit": 2 * settings.max_model_calls + 1},
                    )
    except BaseException:
        # Preserve cancellation/configuration exceptions without logging their text.
        # There is no completed record if execution did not return a final State.
        recorder.emit("run.aborted", stage="runtime", error_code="run_interrupted")
        recorder.close()
        tracer.finish(error_code="run_interrupted")
        await tracer.close()
        raise

    if recorder.failed:
        state.update(stop_reason="error", error=LOG_FAILURE_MESSAGE)
    recorder.emit(
        "run.ended", stop_reason=state["stop_reason"],
        model_calls=state["model_calls"], tool_requests=state["tool_requests"],
        tool_calls=state["tool_calls"], tool_outcomes=state["tool_outcomes"],
        last_usage=state["result"]["usage"] if state["result"] is not None else None,
        usage_totals=state["usage_totals"],
    )
    state["log_complete"] = recorder.close()
    if not state["log_complete"]:
        state.update(stop_reason="error", error=LOG_FAILURE_MESSAGE)
    tracer.finish(state)
    await tracer.close()
    return state
