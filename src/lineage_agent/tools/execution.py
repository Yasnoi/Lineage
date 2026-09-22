"""Execute function calls sequentially and produce Responses protocol items."""

import json
import math
from typing import Any, NoReturn

from lineage_agent.models.types import FunctionCall, ToolDefinition
from lineage_agent.recording.recorder import LOG_FAILURE_MESSAGE, RunRecorder
from lineage_agent.recording.summaries import (
    summarize_arguments,
    summarize_value,
    tool_identity,
)
from lineage_agent.tools.types import (
    ToolArgumentError,
    ToolBatchResult,
    ToolExecutor,
    ToolOutcomes,
)


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
    *,
    recorder: RunRecorder | None = None,
    step_id: str | None = None,
) -> ToolBatchResult:
    """Recover from tool errors; stop remaining calls if their record fails."""
    available_tools = {
        tool["name"]: tool
        for tool in tools
        if tool.get("type") == "function" and isinstance(tool.get("name"), str)
    }
    output_items: list[dict[str, Any]] = []
    tool_calls = 0
    outcomes: ToolOutcomes = {"succeeded": 0, "rejected": 0, "failed": 0, "skipped": 0}
    error: str | None = None
    if recorder is not None and step_id is None:
        step_id = recorder.next_step()

    def record(event: str, **details: Any) -> bool:
        if recorder is None:
            return True
        return recorder.emit(event, step_id=step_id, node="tools", **details)

    def skip_remaining(first_index: int) -> None:
        outcomes["skipped"] += len(calls) - first_index
        for index in range(first_index, len(calls)):
            record(
                "tool.skipped",
                **tool_identity(calls[index], index + 1),
                stage="execution",
                reason="logging_error",
                error_code="logging_error",
            )

    for index, call in enumerate(calls):
        identity = tool_identity(call, index + 1)
        arguments_summary = summarize_value(call["arguments"])
        outcome = "rejected"
        stage = "name_validation"
        if call["name"] not in available_tools:
            output = _error("unknown_tool", "请求的工具未在本次运行中提供。")
        else:
            stage = "argument_parsing"
            try:
                arguments = json.loads(
                    call["arguments"],
                    parse_constant=_reject_constant,
                    parse_float=_parse_float,
                )
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
            except (TypeError, ValueError, RecursionError):
                output = _error("invalid_arguments", "工具参数必须是有效的 JSON 对象。")
            else:
                arguments_summary = summarize_arguments(arguments, available_tools[call["name"]])
                if execute_tool is None:
                    stage = "executor_configuration"
                    output = _error("execution_error", "本次运行未配置工具执行器。")
                else:
                    stage = "execution"
                    if not record(
                        "tool.started",
                        **identity,
                        stage=stage,
                        arguments_summary=arguments_summary,
                    ):
                        skip_remaining(index)
                        error = LOG_FAILURE_MESSAGE
                        break
                    tool_calls += 1
                    try:
                        result = await execute_tool(call["name"], arguments)
                        if not isinstance(result, str):
                            raise TypeError("Tool executors must return a string.")
                    except ToolArgumentError:
                        stage = "argument_validation"
                        output = _error("invalid_arguments", "工具参数未通过执行器校验。")
                    except Exception:
                        outcome = "failed"
                        output = _error("execution_error", "工具执行失败。")
                    else:
                        outcome = "succeeded"
                        output = {"ok": True, "result": result}

        outcomes[outcome] += 1
        details = {**identity, "stage": stage, "arguments_summary": arguments_summary}
        if output["ok"]:
            details["result_summary"] = summarize_value(output["result"])
        else:
            details["error_code"] = output["error"]["code"]
        outcome_recorded = record(f"tool.{outcome}", **details)

        # Creating a protocol output is independent of recording execution success.
        # Preserve this real result even if its outcome event could not be written.
        item = {
            "type": "function_call_output",
            "call_id": call["call_id"],
            "output": json.dumps(output, ensure_ascii=False, allow_nan=False),
        }
        output_items.append(item)
        result_details = {
            **identity,
            "stage": "result_generation",
            "outcome": outcome,
            "ok": output["ok"],
            "output_summary": summarize_value(item["output"]),
        }
        if not output["ok"]:
            result_details["error_code"] = output["error"]["code"]
        result_recorded = record("tool.result_created", **result_details)
        if not outcome_recorded or not result_recorded:
            skip_remaining(index + 1)
            error = LOG_FAILURE_MESSAGE
            break

    return {
        "output_items": output_items,
        "tool_calls": tool_calls,
        "tool_outcomes": outcomes,
        "error": error,
    }
