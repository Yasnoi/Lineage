"""Assemble dependencies and run one agent execution."""

from copy import deepcopy
from typing import Any

from langsmith import tracing_context
from langgraph.types import Overwrite

from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.graph import build_graph
from lineage_agent.agent.state import AgentState
from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.models.responses import create_client, generate_response
from lineage_agent.models.types import ModelInput, ToolDefinition
from lineage_agent.recording.recorder import LOG_FAILURE_MESSAGE, RunRecorder
from lineage_agent.recording.summaries import tool_identity
from lineage_agent.recording.tracing import start_trace
from lineage_agent.tools.types import ToolExecutor, read_budget_snapshot


CLIENT_CLOSE_MESSAGE = "模型客户端关闭失败；已取得的回答与统计予以保留。"


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
    available_tools = validate_tools(tools, execute_tool)
    bind_run = getattr(execute_tool, "bind_run", None)
    if callable(bind_run):
        execute_tool = bind_run()
        if not callable(execute_tool):
            raise ConfigurationError("工具执行器 bind_run 必须返回可调用的运行实例。")

    recorder = RunRecorder(settings.log_dir)
    state: AgentState = {
        "input": input_items,
        "history_end": 0,
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
    config: dict[str, Any] = {"recursion_limit": 2 * settings.max_model_calls + 1}
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

    async def call_model(model_input, **options):
        return await generate_response(
            client, settings.model_name, model_input,
            tools=available_tools or None, tracer=tracer, **options,
        )

    def observe_state(observed: AgentState, node: str, step_id: str | None) -> None:
        nonlocal state, last_node, last_step_id
        state = observed
        last_node, last_step_id = node, step_id

    try:
        graph = build_graph(
            call_model, tools=available_tools, execute_tool=execute_tool,
            max_model_calls=settings.max_model_calls, recorder=recorder, tracer=tracer,
            checkpointer=checkpointer, observe_state=observe_state,
        )
        if not recorder.emit(
            "run.started", input_item_count=len(input_items),
            max_model_calls=settings.max_model_calls,
            read_budget=read_budget_snapshot(execute_tool),
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
                    boundary = saved.values.get("history_end")
                    if (
                        not isinstance(history, list) or type(boundary) is not int
                        or not 0 <= boundary <= len(history)
                        or not all(isinstance(item, dict) for item in history[:boundary])
                        or _input_association_error(history[:boundary]) is not None
                    ):
                        raise CheckpointError("checkpoint_read_error")
                    state["input"] = deepcopy(history[:boundary]) + input_items
                    state["history_end"] = boundary
            if recorder.failed:
                state.update(stop_reason="error", error=LOG_FAILURE_MESSAGE, error_code="logging_error")
            elif invalid := _input_association_error(state["input"]):
                code, index = invalid
                recorder.emit("run.input_rejected", stage="input_validation", error_code=code, input_index=index)
                state.update(stop_reason="error", error="初始上下文中的工具请求与结果无法唯一关联。", error_code=code)
            else:
                state["history_end"] = len(state["input"])
                client_manager = create_client(settings)
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
            # The requests were received, but their tools node never ran. Keep
            # this known outcome even if its record also fails; create no output.
            calls = state["result"]["tool_calls"]
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
        recorder.emit("run.aborted", stage="runtime", error_code="run_interrupted")
        recorder.close()
        if checkpointer is not None and session_id is not None:
            checkpointer.unbind_run(session_id)
        tracer.finish(error_code="run_interrupted")
        await tracer.close()
        raise primary_error

    if recorder.failed:
        state.update(stop_reason="error", error=LOG_FAILURE_MESSAGE, error_code="logging_error")
    recorder.emit(
        "run.ended", stop_reason=state["stop_reason"],
        model_calls=state["model_calls"], tool_requests=state["tool_requests"],
        tool_calls=state["tool_calls"], tool_outcomes=state["tool_outcomes"],
        last_usage=state["result"]["usage"] if state["result"] is not None else None,
        usage_totals=state["usage_totals"],
        error_code=state["error_code"], cleanup_error=state["cleanup_error"],
        has_refusal=state["result"].get("has_refusal", False) if state["result"] is not None else False,
        read_budget=read_budget_snapshot(execute_tool),
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
                # Execution resources and JSONL are already closed. Do not
                # leave a tracing worker alive if finalization is interrupted.
                tracer.finish_checkpoint(None, "checkpoint_write_error")
                tracer.finish(state, error_code="run_interrupted")
                await tracer.close()
                raise
        tracer.finish_checkpoint(final_checkpoint_id, checkpoint_error)
    tracer.finish(state, read_budget=read_budget_snapshot(execute_tool))
    await tracer.close()
    return deepcopy(state), final_checkpoint_id, checkpoint_error
