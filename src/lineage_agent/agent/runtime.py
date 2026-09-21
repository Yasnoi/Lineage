"""Assemble dependencies and run one agent execution."""

from copy import deepcopy
from functools import partial

from lineage_agent.agent.graph import build_graph
from lineage_agent.agent.state import AgentState
from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.models.responses import create_client, generate_response
from lineage_agent.models.types import ModelInput, ToolDefinition
from lineage_agent.tools.types import ToolExecutor


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

    initial_state: AgentState = {
        "input": input_items,
        "result": None,
        "model_calls": 0,
        "tool_requests": 0,
        "tool_calls": 0,
        "usage_totals": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "missing_usage_calls": 0,
        },
        "stop_reason": None,
        "error": None,
    }
    async with create_client(settings) as client:
        call_model = partial(
            generate_response, client, settings.model_name, tools=available_tools or None
        )
        graph = build_graph(
            call_model,
            tools=available_tools,
            execute_tool=execute_tool,
            max_model_calls=settings.max_model_calls,
        )
        return await graph.ainvoke(
            initial_state,
            config={"recursion_limit": 2 * settings.max_model_calls + 1},
        )
