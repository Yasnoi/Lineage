"""Execute function calls sequentially and produce Responses protocol items."""

import json
import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any, NoReturn

from lineage_agent.models.types import FunctionCall, ToolDefinition
from lineage_agent.observations.hints import split_observation
from lineage_agent.recording.recorder import LOG_FAILURE_MESSAGE, RunRecorder
from lineage_agent.recording.summaries import (
    summarize_arguments,
    summarize_value,
    tool_identity,
)
from lineage_agent.tools.types import (
    ToolArgumentError,
    ToolBatchResult,
    ToolExecutionError,
    ToolExecutor,
    ToolOutcomes,
    safe_tool_argument_reason,
    safe_tool_execution_reason,
    serialize_tool_output,
)

if TYPE_CHECKING:
    from lineage_agent.observations import ObservationManager
    from lineage_agent.recording.tracing import RunTrace


def _reject_constant(value: str) -> NoReturn:
    raise ValueError("Non-finite numbers are not JSON values.")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Non-finite numbers are not supported.")
    return parsed


def _parse_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate decoded keys separately in every JSON object."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate object keys are not supported.")
        result[key] = value
    return result


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


async def execute_calls(
    calls: list[FunctionCall],
    tools: list[ToolDefinition],
    execute_tool: ToolExecutor | None,
    *,
    observation_manager: "ObservationManager",
    observation_hint_tools: frozenset[str] = frozenset(),
    native_recall_tools: frozenset[str] = frozenset(),
    recorder: RunRecorder | None = None,
    step_id: str | None = None,
    tracer: "RunTrace | None" = None,
) -> ToolBatchResult:
    """Recover from tool errors; stop remaining calls if their record fails."""
    available_tools = {
        tool["name"]: tool
        for tool in tools
        if tool.get("type") == "function" and isinstance(tool.get("name"), str)
    }
    output_items: list[dict[str, Any]] = []
    call_facts: list[dict[str, Any]] = []
    tool_calls = 0
    outcomes: ToolOutcomes = {"succeeded": 0, "rejected": 0, "failed": 0, "skipped": 0}
    error: str | None = None
    if recorder is not None and step_id is None:
        step_id = recorder.next_step()

    def record(event: str, **details: Any) -> bool:
        if recorder is None:
            return True
        return recorder.emit(event, step_id=step_id, node="tools", **details)

    def skip_remaining(first_index: int, *, first_traced: bool = False) -> None:
        outcomes["skipped"] += len(calls) - first_index
        for index in range(first_index, len(calls)):
            call_facts.append({
                "call_id": calls[index]["call_id"], "name": calls[index]["name"],
                "request_index": index + 1, "outcome": "skipped", "stage": "execution",
                "reason_code": "logging_error", "executor_entered": False,
                "protocol_output_created": False, "output": None,
            })
            record(
                "tool.skipped",
                **tool_identity(calls[index], index + 1),
                stage="execution",
                reason="logging_error",
                error_code="logging_error",
            )
            if tracer is not None and not (first_traced and index == first_index):
                tracer.tool_skipped(
                    calls[index], step_id=step_id, request_index=index + 1,
                    reason="logging_error",
                )

    for index, call in enumerate(calls):
        token = (
            tracer.tool_start(call, step_id=step_id, request_index=index + 1)
            if tracer is not None else None
        )
        identity = tool_identity(call, index + 1)
        arguments_summary = summarize_value(call["arguments"])
        reason_code = reason_message = None
        outcome = "rejected"
        stage = "name_validation"
        executor_entered = False
        if call["name"] not in available_tools:
            output = _error("unknown_tool", "请求的工具未在本次运行中提供。")
        else:
            stage = "argument_parsing"
            try:
                arguments = json.loads(
                    call["arguments"],
                    parse_constant=_reject_constant,
                    parse_float=_parse_float,
                    object_pairs_hook=_parse_object,
                )
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
            except (TypeError, ValueError, RecursionError):
                output = _error(
                    "invalid_arguments", "工具参数必须是有效的 JSON 对象，且所有对象内的键名不得重复。"
                )
            else:
                arguments_summary = summarize_arguments(arguments, available_tools[call["name"]])
                keywords: tuple[str, ...] = ()
                if call["name"] in observation_hint_tools:
                    arguments, keywords = split_observation(arguments)
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
                        if tracer is not None:
                            tracer.tool_end(
                                token, output=None, outcome="skipped", stage=stage,
                                reason_code="logging_error", reason_message=LOG_FAILURE_MESSAGE,
                            )
                        skip_remaining(index, first_traced=True)
                        error = LOG_FAILURE_MESSAGE
                        break
                    tool_calls += 1
                    executor_entered = True
                    try:
                        result = await execute_tool(call["name"], arguments)
                        if not isinstance(result, str):
                            raise TypeError("Tool executors must return a string.")
                    except ToolArgumentError as exc:
                        stage = "argument_validation"
                        reason_code, reason_message = safe_tool_argument_reason(
                            getattr(exc, "reason_code", None)
                        )
                        output = _error("invalid_arguments", "工具参数未通过执行器校验。")
                    except ToolExecutionError as exc:
                        outcome = "failed"
                        reason_code, reason_message = safe_tool_execution_reason(
                            getattr(exc, "reason_code", None)
                        )
                        output = _error("execution_error", "工具执行失败。")
                    except Exception:
                        outcome = "failed"
                        output = _error("execution_error", "工具执行失败。")
                    except BaseException as interruption:
                        # Attach actual facts to the original interruption. The
                        # graph may save them while still propagating it; none
                        # of these absent results becomes a protocol output.
                        outcomes["failed"] += 1
                        call_facts.append({
                            "call_id": call["call_id"], "name": call["name"],
                            "request_index": index + 1, "outcome": "interrupted", "stage": "execution",
                            "reason_code": "tool_interrupted", "executor_entered": True,
                            "protocol_output_created": False, "output": None,
                        })
                        for skipped_index in range(index + 1, len(calls)):
                            skipped = calls[skipped_index]
                            outcomes["skipped"] += 1
                            call_facts.append({
                                "call_id": skipped["call_id"], "name": skipped["name"],
                                "request_index": skipped_index + 1, "outcome": "skipped", "stage": "execution",
                                "reason_code": "tool_interrupted", "executor_entered": False,
                                "protocol_output_created": False, "output": None,
                            })
                        interruption.tool_batch = {
                            "output_items": deepcopy(output_items), "tool_calls": tool_calls,
                            "tool_outcomes": outcomes.copy(), "call_facts": deepcopy(call_facts),
                            "error": None,
                        }
                        if tracer is not None:
                            tracer.tool_end(
                                token, output=_error("tool_interrupted", "工具执行被中断。"),
                                outcome="failed", stage="execution",
                            )
                        raise
                    else:
                        outcome = "succeeded"
                        # Only governed observations become protocol items. Keep
                        # execution success separate from delivery failures.
                        output = observation_manager.observe(
                            result, tool_name=call["name"], keywords=keywords,
                            native_recall=call["name"] in native_recall_tools,
                        )

        if reason_code is not None:
            output["error"].update(reason_code=reason_code, reason_message=reason_message)
        outcomes[outcome] += 1
        if tracer is not None:
            tracer.tool_end(
                token, output=output, outcome=outcome, stage=stage,
                reason_code=reason_code, reason_message=reason_message,
            )
        details = {**identity, "stage": stage, "arguments_summary": arguments_summary}
        if output["ok"]:
            observation = output["observation"]
            details["result_summary"] = summarize_value(result)
            details["observation_status"] = observation["status"]
            if observation.get("mode") is not None:
                details["observation_mode"] = observation["mode"]
            if observation["status"] == "unavailable":
                details["observation_stage"] = observation["stage"]
                details["observation_error_code"] = observation["error"]["code"]
        else:
            details["error_code"] = output["error"]["code"]
        if reason_code is not None:
            details.update(reason_code=reason_code, reason_message=reason_message)
        outcome_recorded = record(f"tool.{outcome}", **details)

        # Creating a protocol output is independent of recording execution success.
        # Preserve this real result even if its outcome event could not be written.
        item = {
            "type": "function_call_output",
            "call_id": call["call_id"],
            "output": serialize_tool_output(output),
        }
        output_items.append(item)
        call_facts.append({
            "call_id": call["call_id"], "name": call["name"], "request_index": index + 1,
            "outcome": outcome, "stage": stage,
            "reason_code": reason_code or (output.get("error") or {}).get("code"),
            "executor_entered": executor_entered, "protocol_output_created": True,
            "output": deepcopy(output),
        })
        result_details = {
            **identity,
            "stage": "result_generation",
            "outcome": outcome,
            "ok": output["ok"],
            "output_summary": summarize_value(item["output"]),
        }
        if not output["ok"]:
            result_details["error_code"] = output["error"]["code"]
        if reason_code is not None:
            result_details.update(reason_code=reason_code, reason_message=reason_message)
        result_recorded = record("tool.result_created", **result_details)
        if not outcome_recorded or not result_recorded:
            skip_remaining(index + 1)
            error = LOG_FAILURE_MESSAGE
            break

    return {
        "output_items": output_items,
        "tool_calls": tool_calls,
        "tool_outcomes": outcomes,
        "call_facts": call_facts,
        "error": error,
    }
