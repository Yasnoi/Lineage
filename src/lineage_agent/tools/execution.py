"""Execute function calls sequentially and produce Responses protocol items."""

import json
import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, NoReturn

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
    verify_persistence: Callable[[], None] | None = None,
    recoverable: bool = False,
    request_index_offset: int = 0,
    reused_result: bool = False,
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
        return recorder.emit(event, step_id=step_id, node="tools",
                             **({"reused_result": True, "executor_entered": False} if reused_result else {}), **details)

    def skip_remaining(first_index: int, *, first_traced: bool = False, reason: str = "logging_error") -> None:
        outcomes["skipped"] += len(calls) - first_index
        for index in range(first_index, len(calls)):
            call_facts.append({
                "call_id": calls[index]["call_id"], "name": calls[index]["name"],
                "request_index": request_index_offset + index + 1, "outcome": "skipped", "stage": "execution",
                "reason_code": reason, "executor_entered": False,
                "protocol_output_created": False, "output": None,
            })
            record(
                "tool.skipped",
                **tool_identity(calls[index], request_index_offset + index + 1),
                stage="execution",
                reason=reason,
                error_code=reason,
            )
            if tracer is not None and not (first_traced and index == first_index):
                tracer.tool_skipped(
                    calls[index], step_id=step_id, request_index=request_index_offset + index + 1,
                    reason=reason,
                )

    for index, call in enumerate(calls):
        token = (
            tracer.tool_start(call, step_id=step_id, request_index=request_index_offset + index + 1,
                              **({"reused_result": True} if reused_result else {}))
            if tracer is not None else None
        )
        identity = tool_identity(call, request_index_offset + index + 1)
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
                        "tool.result_reused" if reused_result else "tool.started",
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
                    tool_calls += int(not reused_result)
                    # This logical fact includes the original trusted attempt;
                    # only this process's diagnostics describe result replay.
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
                    except Exception as exc:
                        if getattr(exc, "code", "").startswith(("archive_", "checkpoint_", "record_", "artifact_")):
                            if fact := getattr(exc, "execution_fact", None):
                                # The delegate completed before durable result
                                # storage failed. Record that real outcome, but
                                # do not invent a model-visible protocol result.
                                from lineage_agent.observations.manager import safe_observation_error
                                actual_outcome, actual_stage = fact["outcome"], fact["stage"]
                                details = {**identity, "stage": actual_stage,
                                           "arguments_summary": arguments_summary,
                                           "persistence_error_code": getattr(exc, "code", "archive_write_error")}
                                tool_reason = tool_message = None
                                if actual_outcome == "succeeded":
                                    code, message = safe_observation_error(
                                        "artifact_write_error" if fact["reason_code"] == "artifact_write_error"
                                        else "observation_save_error")
                                    diagnostic = {"ok": True, "observation": {
                                        "status": "unavailable", "stage": "save",
                                        "error": {"code": code, "message": message}}}
                                    details.update(observation_status="unavailable",
                                                   observation_stage="save", observation_error_code=code)
                                else:
                                    if actual_outcome == "rejected":
                                        tool_reason, tool_message = safe_tool_argument_reason(fact.get("tool_reason_code"))
                                        diagnostic = _error("invalid_arguments", "工具参数未通过执行器校验。")
                                    else:
                                        tool_reason, tool_message = safe_tool_execution_reason(fact.get("tool_reason_code"))
                                        diagnostic = _error("execution_error", "工具执行失败。")
                                    diagnostic["error"].update(reason_code=tool_reason, reason_message=tool_message)
                                    details.update(error_code=diagnostic["error"]["code"],
                                                   reason_code=tool_reason, reason_message=tool_message)
                                if tracer is not None:
                                    tracer.tool_end(token, output=diagnostic, outcome=actual_outcome, stage=actual_stage,
                                                    reason_code=tool_reason, reason_message=tool_message)
                                record(f"tool.{actual_outcome}", **details)
                            raise
                        outcome = "failed"
                        output = _error("execution_error", "工具执行失败。")
                    except BaseException as interruption:
                        if recoverable:
                            if tracer is not None:
                                tracer.tool_end(token, output={"status": "unknown"},
                                                outcome="unknown", stage="execution")
                            raise
                        # Attach actual facts to the original interruption. The
                        # graph may save them while still propagating it; none
                        # of these absent results becomes a protocol output.
                        outcomes["failed"] += 1
                        call_facts.append({
                            "call_id": call["call_id"], "name": call["name"],
                            "request_index": request_index_offset + index + 1, "outcome": "interrupted", "stage": "execution",
                            "reason_code": "tool_interrupted", "executor_entered": True,
                            "protocol_output_created": False, "output": None,
                        })
                        for skipped_index in range(index + 1, len(calls)):
                            skipped = calls[skipped_index]
                            outcomes["skipped"] += 1
                            call_facts.append({
                                "call_id": skipped["call_id"], "name": skipped["name"],
                                "request_index": request_index_offset + skipped_index + 1, "outcome": "skipped", "stage": "execution",
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
            "call_id": call["call_id"], "name": call["name"], "request_index": request_index_offset + index + 1,
            **({"reused_result": True} if reused_result else {}),
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
        if verify_persistence is not None:
            try:
                verify_persistence()
            except Exception as failure:
                # The tool has already executed. Retain that fact and stop the
                # remaining batch before graph adoption can authorize new data.
                skip_remaining(index + 1, reason=getattr(failure, "code", "artifact_write_error"))
                failure.tool_batch = {
                    "output_items": deepcopy(output_items), "tool_calls": tool_calls,
                    "tool_outcomes": outcomes.copy(), "call_facts": deepcopy(call_facts), "error": None,
                }
                raise
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
