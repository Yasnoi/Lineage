"""Assemble dependencies and run one agent execution."""

from copy import deepcopy
from typing import Any

from langsmith import tracing_context
from langgraph.types import Overwrite

from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.context import ContextRecords, new_context
from lineage_agent.agent.graph import build_graph
from lineage_agent.agent.state import AgentState
from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.models.budget import annotate_context_risks
from lineage_agent.models.requests import prepare_request
from lineage_agent.models.responses import create_client, send_prepared_response
from lineage_agent.models.types import ModelInput, ToolDefinition
from lineage_agent.recording.recorder import LOG_FAILURE_MESSAGE, RunRecorder
from lineage_agent.recording.summaries import tool_identity
from lineage_agent.recording.tracing import start_trace
from lineage_agent.tools.types import ToolExecutor, read_budget_snapshot


CLIENT_CLOSE_MESSAGE = "模型客户端关闭失败；已取得的回答与统计予以保留。"
INVALID_INPUT_MESSAGE = "新增输入不符合支持的消息或工具协议项格式。"


class PreExecutionConfigurationError(ConfigurationError):
    """Only raised at explicit synchronous boundaries before graph execution."""

    def __init__(self) -> None:
        super().__init__("执行前配置检查未通过，请检查工具绑定、API_KEY 和连接配置；修正后可继续当前会话。")


def _valid_content(value: Any) -> bool:
    if isinstance(value, str):
        return True
    if not isinstance(value, list):
        return False
    for part in value:
        if not isinstance(part, dict):
            return False
        kind = part.get("type")
        if not isinstance(kind, str) or not kind.strip():
            return False
        if kind in ("input_text", "output_text") and not isinstance(part.get("text"), str):
            return False
        if kind == "refusal" and not isinstance(part.get("refusal"), str):
            return False
    return True


def _input_shape_error(items: list[dict[str, Any]]) -> tuple[str, int] | None:
    """Check supported containers, preserving opaque typed protocol extensions."""
    for index, item in enumerate(items, 1):
        kind = item.get("type")
        if kind == "message" or "role" in item:
            valid = (
                ("type" not in item or kind == "message")
                and item.get("role") in ("user", "assistant", "system", "developer")
                and _valid_content(item.get("content"))
            )
        elif not isinstance(kind, str) or not kind.strip():
            valid = False
        elif kind == "function_call":
            valid = (
                isinstance(item.get("name"), str) and bool(item["name"].strip())
                and isinstance(item.get("arguments"), str)
            )
        elif kind == "function_call_output":
            valid = _valid_content(item.get("output"))
        else:
            valid = True
        if not valid:
            return "invalid_input", index
    return None


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


def normalize_input(input: ModelInput) -> list[dict[str, Any]]:
    """Copy new protocol items before any await or access to saved history."""
    if isinstance(input, str):
        if not input.strip():
            raise ValueError("用户输入不能为空。")
        return [{"role": "user", "content": input}]
    elif isinstance(input, list) and input and all(isinstance(item, dict) for item in input):
        return deepcopy(input)
    else:
        raise ValueError("input 必须是非空字符串或非空 Responses 协议项列表。")


def validate_tools(
    tools: list[ToolDefinition] | None, execute_tool: ToolExecutor | None,
) -> list[ToolDefinition]:
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
    return available_tools


async def run_agent(
    input: ModelInput,
    settings: ModelSettings,
    *,
    tools: list[ToolDefinition] | None = None,
    execute_tool: ToolExecutor | None = None,
) -> AgentState:
    """Handle one independent request without retaining a conversation."""
    state, _, _ = await _run_agent(input, settings, tools=tools, execute_tool=execute_tool)
    return state


async def _run_agent(
    input: ModelInput,
    settings: ModelSettings,
    *,
    tools: list[ToolDefinition] | None = None,
    execute_tool: ToolExecutor | None = None,
    session_id: str | None = None,
    checkpointer: SessionCheckpointer | None = None,
    checkpoint_id: str | None = None,
) -> tuple[AgentState, str | None, str | None]:
    """Share execution and cleanup; sessions add only the checkpoint lifecycle."""
    input_items = normalize_input(input)
    try:
        available_tools = validate_tools(tools, execute_tool)
        bind_run = getattr(execute_tool, "bind_run", None)
        if callable(bind_run):
            execute_tool = bind_run()
            if not callable(execute_tool):
                raise ConfigurationError("工具执行器 bind_run 必须返回可调用的运行实例。")
    except ConfigurationError:
        raise PreExecutionConfigurationError() from None

    recorder = RunRecorder(settings.log_dir)
    tracer = None
    propagating_error = False
    try:
        state: AgentState = {
            "input": input_items,
            "history_end": 0,
            "context_risks": annotate_context_risks(
                input_items, fallback_tokens=settings.opaque_fallback_tokens,
                run_id=recorder.run_id,
            ),
            "context_budget": None,
            "active_context": new_context(),
            "compaction": None,
            "summary_calls": 0,
            "summary_result": None,
            "summary_usage_totals": {
                "input_tokens": None, "output_tokens": None, "total_tokens": None,
                "missing_usage_calls": 0,
            },
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
            "error_code": None,
            "cleanup_error": None,
        }
        tracer = start_trace(recorder.run_id, settings)
        if session_id is not None:
            assert checkpointer is not None
            checkpointer.bind_run(session_id, recorder)
            tracer.bind_session(session_id)
        config: dict[str, Any] = {"recursion_limit": 5 * settings.max_model_calls + 1}
        if session_id is not None:
            config["configurable"] = {"thread_id": session_id}
            if checkpoint_id is not None:
                config["configurable"]["checkpoint_id"] = checkpoint_id
            config["metadata"] = {"session_id": session_id, "run_id": recorder.run_id}
        checkpoint_error: str | None = None
        final_checkpoint_id: str | None = None
        client_manager = None
        client = None
        last_node: str | None = None
        last_step_id: str | None = None
        primary_error: BaseException | None = None

        def prepare_model_request(model_input, **options):
            return prepare_request(
                settings.model_name, model_input, tools=available_tools or None,
                max_output_tokens=settings.max_output_tokens, **options,
            )

        async def call_model(request):
            return await send_prepared_response(client, request, tracer=tracer)

        async def call_summary(request, metadata):
            return await send_prepared_response(
                client, request, tracer=tracer, purpose="summary", summary_metadata=metadata,
            )

        def observe_state(observed: AgentState, node: str, step_id: str | None) -> None:
            nonlocal state, last_node, last_step_id
            state = observed
            last_node, last_step_id = node, step_id

        try:
            graph = build_graph(
                call_model, tools=available_tools, execute_tool=execute_tool,
                prepare_model_request=prepare_model_request, settings=settings,
                max_model_calls=settings.max_model_calls, recorder=recorder, tracer=tracer,
                checkpointer=checkpointer, observe_state=observe_state,
                call_summary=call_summary,
                context_records=ContextRecords(checkpointer, session_id) if checkpointer is not None and session_id is not None else None,
                validate_input=lambda items: _input_shape_error(items) or _input_association_error(items),
            )
            start_budget_error = None

            def start_budget_unavailable():
                nonlocal start_budget_error
                start_budget_error = "read_budget_unavailable"

            start_budget = read_budget_snapshot(execute_tool, on_unavailable=start_budget_unavailable)
            tracer.observe_read_budget("run_start", start_budget_error)
            if not recorder.emit(
                "run.started", input_item_count=len(input_items),
                max_model_calls=settings.max_model_calls,
                read_budget=start_budget, read_budget_error=start_budget_error,
            ):
                state.update(stop_reason="error", error=LOG_FAILURE_MESSAGE, error_code="logging_error")
            # Only our explicit, filtered spans may leave the process. Even if log
            # creation failed, load the old history before saving this failed turn.
            with tracing_context(enabled=False, parent=False):
                if checkpointer is not None:
                    saved = await graph.aget_state(config)
                    if checkpoint_id is not None and not saved.values:
                        raise CheckpointError("checkpoint_read_error")
                    if saved.values:
                        history = saved.values.get("input")
                        risks = saved.values.get("context_risks")
                        boundary = saved.values.get("history_end")
                        if (
                            not isinstance(history, list) or type(boundary) is not int
                            or not 0 <= boundary <= len(history)
                            or not all(isinstance(item, dict) for item in history[:boundary])
                            or not isinstance(risks, list) or len(risks) != len(history)
                            or _input_association_error(history[:boundary]) is not None
                        ):
                            raise CheckpointError("checkpoint_read_error")
                        state["input"] = deepcopy(history[:boundary]) + input_items
                        state["context_risks"] = deepcopy(risks[:boundary]) + state["context_risks"]
                        state["history_end"] = boundary
                        previous_active = saved.values.get("active_context")
                        if not isinstance(previous_active, dict):
                            raise CheckpointError("checkpoint_read_error")
                        active = deepcopy(previous_active)
                        start = active.get("turn_start")
                        if type(start) is not int or not 0 <= start <= boundary:
                            raise CheckpointError("checkpoint_read_error")
                        if boundary > start:
                            active["turns"].append({
                                "run_id": saved.values["run_id"],
                                "checkpoint_id": saved.config["configurable"]["checkpoint_id"],
                                "item_count": boundary - start,
                            })
                        active["turn_start"] = boundary
                        if (1 if active["summary"] else 0) + sum(ref["item_count"] for ref in active["turns"]) != boundary:
                            raise CheckpointError("checkpoint_read_error")
                        state["active_context"] = active
                if recorder.failed:
                    state.update(stop_reason="error", error=LOG_FAILURE_MESSAGE, error_code="logging_error")
                elif invalid := _input_shape_error(state["input"]) or _input_association_error(state["input"]):
                    code, index = invalid
                    recorder.emit("run.input_rejected", stage="input_validation", error_code=code, input_index=index)
                    state.update(
                        stop_reason="error", error_code=code,
                        error=INVALID_INPUT_MESSAGE if code == "invalid_input" else "初始上下文中的工具请求与结果无法唯一关联。",
                    )
                else:
                    try:
                        client_manager = create_client(settings)
                    except ConfigurationError:
                        raise PreExecutionConfigurationError() from None
                    client = await client_manager.__aenter__()
                    graph_input = {**state, "input": Overwrite(deepcopy(state["input"]))} if checkpointer is not None else state
                    state = await graph.ainvoke(
                        graph_input, config=config,
                        **({"durability": "sync"} if checkpointer is not None else {}),
                    )
        except CheckpointError as exc:
            checkpoint_error = exc.code
            recorder.emit("run.checkpoint_failed", error_code=exc.code)
            if last_node == "model" and state["stop_reason"] is None and state["result"] is not None:
                # A model node can also prepare compaction without a new call.
                # Its latest result may belong to tools that already completed.
                # Only unmatched requests are known to have remained unexecuted.
                result_ids = {
                    item.get("call_id") for item in state["input"]
                    if item.get("type") == "function_call_output"
                }
                calls = [
                    call for call in state["result"]["tool_calls"]
                    if call["call_id"] not in result_ids
                ]
                state["tool_outcomes"]["skipped"] += len(calls)
                for index, call in enumerate(calls, 1):
                    recorder.emit(
                        "tool.skipped", step_id=last_step_id, node="model",
                        **tool_identity(call, index), stage="routing",
                        reason="checkpoint_error", error_code=exc.code,
                    )
                    tracer.tool_skipped(
                        call, step_id=last_step_id, request_index=index, reason="checkpoint_error",
                    )
            if state["stop_reason"] is None:
                state.update(stop_reason="error", error=str(exc), error_code=exc.code)
        except BaseException as exc:
            primary_error = exc
        finally:
            if client_manager is not None:
                try:
                    await client_manager.__aexit__(
                        type(primary_error) if primary_error is not None else None,
                        primary_error, primary_error.__traceback__ if primary_error is not None else None,
                    )
                except BaseException as exc:
                    if isinstance(exc, Exception):
                        previous = state["stop_reason"]
                        state["cleanup_error"] = {"code": "client_close_error", "previous_stop_reason": previous}
                        recorder.emit("run.cleanup_failed", stage="client_close", error_code="client_close_error")
                        if state["stop_reason"] != "error":
                            state.update(stop_reason="error", error=CLIENT_CLOSE_MESSAGE, error_code="client_close_error")
                    elif primary_error is None:
                        primary_error = exc
        if primary_error is not None:
            # Preserve cancellation/configuration exceptions without logging their text.
            # There is no completed record if execution did not return a final State.
            configuration_rejected = isinstance(primary_error, PreExecutionConfigurationError)
            failure_code = "pre_execution_configuration_error" if configuration_rejected else "run_interrupted"
            try:
                recorder.emit(
                    "run.configuration_rejected" if configuration_rejected else "run.aborted",
                    stage="pre_execution" if configuration_rejected else "runtime", error_code=failure_code,
                )
            except BaseException:
                pass  # Reporting must not replace the already captured failure.
            try:
                tracer.finish(error_code=failure_code)
            except BaseException:
                pass
            # The outer finally independently closes all remaining resources.
            raise primary_error

        if recorder.failed:
            state.update(stop_reason="error", error=LOG_FAILURE_MESSAGE, error_code="logging_error")
        end_budget_error = None

        def end_budget_unavailable():
            nonlocal end_budget_error
            end_budget_error = "read_budget_unavailable"

        end_budget = read_budget_snapshot(execute_tool, on_unavailable=end_budget_unavailable)
        tracer.observe_read_budget("run_end", end_budget_error)
        recorder.emit(
            "run.ended", stop_reason=state["stop_reason"],
            model_calls=state["model_calls"], tool_requests=state["tool_requests"],
            tool_calls=state["tool_calls"], tool_outcomes=state["tool_outcomes"],
            last_usage=state["result"]["usage"] if state["result"] is not None else None,
            usage_totals=state["usage_totals"],
            error_code=state["error_code"], cleanup_error=state["cleanup_error"],
            context_budget=state["context_budget"],
            summary_calls=state["summary_calls"], summary_usage_totals=state["summary_usage_totals"],
            active_version_id=state["active_context"]["version_id"],
            compaction_status=(state["compaction"] or {}).get("status"),
            summary_id=(state["compaction"] or {}).get("summary_id"),
            has_refusal=state["result"].get("has_refusal", False) if state["result"] is not None else False,
            read_budget=end_budget, read_budget_error=end_budget_error,
            **({
                "checkpoint_finalization": "failed" if checkpoint_error else "pending",
                "last_confirmed_checkpoint_id": checkpointer.last_confirmed(session_id),
                "checkpoint_error": checkpoint_error,
            } if checkpointer is not None and session_id is not None else {}),
        )
        state["log_complete"] = recorder.close()
        if not state["log_complete"]:
            state.update(stop_reason="error", error=LOG_FAILURE_MESSAGE, error_code="logging_error")
        if checkpointer is not None and session_id is not None:
            # The JSONL intentionally ends before this final acknowledgement. Its
            # completion, including close errors, is part of the final saved State.
            checkpointer.unbind_run(session_id)
            if checkpoint_error is None:
                final_config = {
                    "configurable": {"thread_id": session_id},
                    "metadata": {"session_id": session_id, "run_id": recorder.run_id},
                }
                try:
                    with tracing_context(enabled=False, parent=False):
                        saved_config = await graph.aupdate_state(
                            final_config, {**deepcopy(state), "input": Overwrite(deepcopy(state["input"]))},
                            as_node="model",
                        )
                    final_checkpoint_id = saved_config["configurable"]["checkpoint_id"]
                except CheckpointError as exc:
                    checkpoint_error = exc.code
                except Exception:
                    checkpoint_error = "checkpoint_write_error"
                except BaseException:
                    # Keep the interruption primary; independent cleanup in the
                    # outer finally also runs if tracing diagnostics fail.
                    try:
                        tracer.finish_checkpoint(None, "checkpoint_write_error")
                        tracer.finish(state, error_code="run_interrupted")
                    except BaseException:
                        pass
                    raise
            tracer.finish_checkpoint(final_checkpoint_id, checkpoint_error)
        tracer.finish(state, read_budget=end_budget, read_budget_error=end_budget_error)
        await tracer.close()
        return deepcopy(state), final_checkpoint_id, checkpoint_error
    except BaseException:
        propagating_error = True
        raise
    finally:
        # Independently attempt every cleanup even if execution, final saving,
        # or an earlier cleanup failed. Never replace an in-flight primary error.
        cleanup_error = None
        try:
            recorder.close()
        except BaseException as exc:
            cleanup_error = exc
        try:
            if checkpointer is not None and session_id is not None:
                checkpointer.unbind_run(session_id)
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if tracer is not None:
            try:
                tracer.finish(error_code="run_interrupted")
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
            try:
                await tracer.close()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if cleanup_error is not None and not propagating_error:
            raise cleanup_error
