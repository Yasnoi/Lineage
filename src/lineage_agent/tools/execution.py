"""Execute function calls sequentially and produce Responses protocol items."""

import json
import math
from typing import Any, NoReturn

from lineage_agent.models.types import FunctionCall, ToolDefinition
from lineage_agent.tools.types import ToolArgumentError, ToolBatchResult, ToolExecutor


def _reject_constant(value: str) -> NoReturn:
    raise ValueError("Non-finite numbers are not JSON values.")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Non-finite numbers are not supported.")
    return parsed


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


async def execute_calls(
    calls: list[FunctionCall],
    tools: list[ToolDefinition],
    execute_tool: ToolExecutor | None,
) -> ToolBatchResult:
    """Return one correlated output per call without aborting on tool errors."""
    available_names = {
        tool["name"]
        for tool in tools
        if tool.get("type") == "function" and isinstance(tool.get("name"), str)
    }
    output_items: list[dict[str, Any]] = []
    tool_calls = 0

    for call in calls:
        if call["name"] not in available_names:
            output = _error("unknown_tool", "请求的工具未在本次运行中提供。")
        else:
            try:
                arguments = json.loads(
                    call["arguments"],
                    parse_constant=_reject_constant,
                    parse_float=_parse_float,
                )
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
            except (TypeError, ValueError):
                output = _error("invalid_arguments", "工具参数必须是有效的 JSON 对象。")
            else:
                if execute_tool is None:
                    output = _error("execution_error", "本次运行未配置工具执行器。")
                else:
                    tool_calls += 1
                    try:
                        result = await execute_tool(call["name"], arguments)
                        if not isinstance(result, str):
                            raise TypeError("Tool executors must return a string.")
                    except ToolArgumentError:
                        output = _error("invalid_arguments", "工具参数未通过执行器校验。")
                    except Exception:
                        output = _error("execution_error", "工具执行失败。")
                    else:
                        output = {"ok": True, "result": result}

        output_items.append(
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": json.dumps(output, ensure_ascii=False, allow_nan=False),
            }
        )

    return {"output_items": output_items, "tool_calls": tool_calls}
