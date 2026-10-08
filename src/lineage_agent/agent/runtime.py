"""Assemble dependencies and run one agent execution."""

from copy import deepcopy
from typing import Any
from uuid import uuid4
import json

from langsmith import tracing_context
from langgraph.types import Overwrite

from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.context import ContextRecords, new_context
from lineage_agent.agent.records import make_record_batch
from lineage_agent.agent.record_journal import JournalError, RecordJournal
from lineage_agent.agent.graph import build_graph
from lineage_agent.agent.state import AgentState
from lineage_agent.artifacts.store import ArtifactStore, SessionArtifacts
from lineage_agent.artifacts.tools import ARTIFACT_TOOLS, ArtifactExecutor
from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.models.budget import annotate_context_risks
from lineage_agent.models.requests import prepare_request
from lineage_agent.models.responses import create_client, send_prepared_response
from lineage_agent.models.types import ModelInput, ToolDefinition
from lineage_agent.observations.hints import enhance_tools
from lineage_agent.observations.manager import ObservationManager
from lineage_agent.recording.recorder import LOG_FAILURE_MESSAGE, RunRecorder
from lineage_agent.recording.summaries import tool_identity
from lineage_agent.recording.tracing import start_trace
from lineage_agent.tools.types import ToolExecutor, read_budget_snapshot
from lineage_agent.tools.history import HISTORY_TOOLS, HISTORY_TOOL_NAMES, HistoryExecutor
from lineage_agent.storage import StorageError, conversation_paths


CLIENT_CLOSE_MESSAGE = "模型客户端关闭失败；已取得的回答与统计予以保留。"
ARTIFACT_INIT_MESSAGE = "工具结果存储无法初始化，未发起模型或工具调用。"
ARTIFACT_CLOSE_MESSAGE = "工具结果存储关闭失败；已取得的回答与统计予以保留。"
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


def inherit_context(previous: dict, new_input: list, new_risks: list, *, run_id: str) -> dict:
    """Compose the next Run from the confirmed effective view, without registering it twice."""
    history, risks, boundary = previous.get("input"), previous.get("context_risks"), previous.get("history_end")
    if (not isinstance(history, list) or type(boundary) is not int or not 0 <= boundary <= len(history)
            or not all(isinstance(item, dict) for item in history[:boundary])
            or not isinstance(risks, list) or len(risks) != len(history)
            or _input_association_error(history[:boundary]) is not None):
        raise CheckpointError("checkpoint_read_error")
    active = deepcopy(previous.get("active_context"))
    if not isinstance(active, dict):
        raise CheckpointError("checkpoint_read_error")
    start = active.get("turn_start")
    if type(start) is not int or not 0 <= start <= boundary:
        raise CheckpointError("checkpoint_read_error")
    # Only non-completed turns retain the previous admission-at-next-turn rule.
    if active.get("current_run_id") is not None and boundary > start:
        if any(turn["run_id"] == previous["run_id"] for turn in active["turns"]):
            raise CheckpointError("checkpoint_read_error")
        active["turns"].append({"run_id": previous["run_id"], "item_count": boundary - start,
                                "group_count": sum(g["closed"] for g in active["groups"]),
                                "record_end": previous["record_sequence"]})
    active.update(turn_start=boundary, current_run_id=run_id, run_summary=None, groups=[],
                  initial_input_count=len(new_input), parent_version_id=active["version_id"], version_id=str(uuid4()),
                  item_refs=active["item_refs"][:boundary] + [f"history://record/{run_id}/{i}" for i in range(len(new_input))],
                  summary_spans=[span for span in active["summary_spans"] if span["index"] < boundary])
    return {"input": deepcopy(history[:boundary]) + deepcopy(new_input),
            "context_risks": deepcopy(risks[:boundary]) + deepcopy(new_risks), "history_end": boundary,
            "active_context": active, "history_scopes": deepcopy(previous.get("history_scopes", {})),
            "record_files": deepcopy(previous.get("record_files", {}))}


def completed_context(state: AgentState) -> dict:
    """Prepare history ownership for the same final checkpoint, without a self reference."""
    active = deepcopy(state["active_context"])
    if (active["current_run_id"] != state["run_id"] or state["history_end"] != len(state["input"])
            or any(not group["closed"] for group in active["groups"])
            or any(turn["run_id"] == state["run_id"] for turn in active["turns"])):
        raise CheckpointError("checkpoint_read_error")
    active["turns"].append({"run_id": state["run_id"], "item_count": state["history_end"] - active["turn_start"],
                            "group_count": len(active["groups"]), "record_end": state["record_sequence"]})
    active.update(parent_version_id=active["version_id"], version_id=str(uuid4()),
                  current_run_id=None, run_summary=None, groups=[], initial_input_count=0,
                  turn_start=state["history_end"])
    return active


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
    """Handle one independent request; release execution state and retain its files."""
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
    artifacts: SessionArtifacts | None = None,
) -> tuple[AgentState, str | None, str | None]:
    """Share execution and cleanup; sessions add only the checkpoint lifecycle."""
    input_items = normalize_input(input)
    try:
        available_tools = validate_tools(tools, execute_tool)
        if session_id is None:
            native_names = {tool["name"] for tool in ARTIFACT_TOOLS}
            if any(tool["name"] in native_names for tool in available_tools):
                raise ConfigurationError("已提供的工具名称与 Artifact 工具冲突。")
            available_tools.extend(deepcopy(ARTIFACT_TOOLS))
        if any(tool["name"] in HISTORY_TOOL_NAMES for tool in available_tools):
            raise ConfigurationError("已提供的工具名称与历史工具冲突。")
        available_tools.extend(deepcopy(HISTORY_TOOLS))
        available_tools, observation_hint_tools = enhance_tools(available_tools)
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
    owned_store = None
    store_close_attempted = False
    private_checkpointer = checkpointer is None
    scope_id = session_id or "run:" + recorder.run_id
    conversation_id = session_id or recorder.run_id
    try:
        initial_batch = make_record_batch(recorder.run_id, 0, input_items)
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
            "compaction_events": [],
            "record_batch": [], "record_files": {},
            "record_sequence": len(input_items), "record_revision": 1,
            "record_admitted_through": 0, "history_scopes": {}, "record_metrics": {},
            "execution_facts": [], "run_call_ids": [],
            "summary_calls": 0,
            "summary_result": None,
            "summary_usage_totals": {
                "input_tokens": None, "output_tokens": None, "total_tokens": None,
                "missing_usage_calls": 0,
            },
            "run_id": recorder.run_id,
            "conversation_id": conversation_id,
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
        state["active_context"].update(initial_input_count=len(input_items), current_run_id=recorder.run_id,
                                       item_refs=[r["source_ref"] for r in initial_batch])
        if checkpointer is None:
            checkpointer = SessionCheckpointer()
        checkpointer.bind_run(scope_id, recorder)
        try:
            paths = conversation_paths(settings.application_root, conversation_id)
            journal = RecordJournal(paths.history)
            checkpointer.bind_history(scope_id, journal)
            receipts, watermark = journal.append(recorder.run_id, initial_batch)
            state["record_batch"] = receipts
            state["record_files"] = {recorder.run_id: watermark}
        except (StorageError, JournalError) as exc:
            code = exc.code if isinstance(exc, JournalError) else "record_create_error"
            state.update(stop_reason="error", error_code=code, error=str(CheckpointError(code)))
            recorder.emit("run.initialization_failed", stage="history_init", error_code=code)
            state["log_complete"] = recorder.close()
            return deepcopy(state), None, code
        context_records = ContextRecords(checkpointer, scope_id)
        execute_tool = HistoryExecutor(context_records.store, execute_tool)
        tracer = start_trace(recorder.run_id, settings)
        if session_id is not None:
            recorder.bind_session(session_id)
            tracer.bind_session(session_id)
        config: dict[str, Any] = {"recursion_limit": 16 * settings.max_model_calls + 8,
            "configurable": {"thread_id": scope_id}, "metadata": {"run_id": recorder.run_id}}
        if checkpoint_id is not None:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        if session_id is not None:
            config["metadata"]["session_id"] = session_id
        checkpoint_error: str | None = None
        final_checkpoint_id: str | None = None
        client_manager = None
        client = None
        last_node: str | None = None
        last_step_id: str | None = None
        primary_error: BaseException | None = None
        observation_manager = None
        artifact_initialization_failed = False
        try:
            if artifacts is None:
                owned_store = ArtifactStore()
                artifacts = owned_store.bind_session(recorder.run_id, directory=paths.artifacts)
                execute_tool = ArtifactExecutor(artifacts, execute_tool)
            observation_manager = ObservationManager(artifacts)
        except Exception:
            artifact_initialization_failed = True
            state.update(stop_reason="error", error=ARTIFACT_INIT_MESSAGE,
                         error_code="artifact_store_init_error")

        def prepare_model_request(model_input, **options):
            return prepare_request(
                settings.model_name, model_input, tools=available_tools or None,
                max_output_tokens=settings.max_output_tokens, **options,
                reasoning_effort=settings.reasoning_effort,
            )

        async def call_model(request):
            return await send_prepared_response(client, request, tracer=tracer)

        async def call_summary(request, metadata):
            return await send_prepared_response(
                client, request, tracer=tracer, purpose="summary", summary_metadata=metadata,
            )

        def observe_state(observed: AgentState, node: str, step_id: str | None) -> None:
            nonlocal state, last_node, last_step_id
            if node == "activate_context" and (observed["compaction"] or {}).get("status") == "pending_save":
                # LangGraph still saves the full activation update. This local
                # failure fallback cannot claim it is active before that save
                # is confirmed by the next model node. Retain paid-call facts.
                for field in ("input", "context_risks", "active_context", "history_end"):
                    observed[field] = deepcopy(state[field])
            state = observed
            last_node, last_step_id = node, step_id

        try:
            graph = build_graph(
                call_model, tools=available_tools, execute_tool=execute_tool,
                prepare_model_request=prepare_model_request, settings=settings,
                max_model_calls=settings.max_model_calls, recorder=recorder, tracer=tracer,
                checkpointer=checkpointer, observe_state=observe_state,
                call_summary=call_summary,
                context_records=context_records,
                validate_input=lambda items: _input_shape_error(items) or _input_association_error(items),
                observation_manager=observation_manager,
                observation_hint_tools=observation_hint_tools,
                native_recall_tools=frozenset(tool["name"] for tool in ARTIFACT_TOOLS) | HISTORY_TOOL_NAMES,
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
                        own_files = state["record_files"]
                        state.update(inherit_context(saved.values, input_items, state["context_risks"], run_id=recorder.run_id))
                        state["record_files"].update(own_files)
                if recorder.failed:
                    state.update(stop_reason="error", error=LOG_FAILURE_MESSAGE, error_code="logging_error")
                elif artifact_initialization_failed:
                    recorder.emit("run.initialization_failed", stage="artifact_init",
                                  error_code="artifact_store_init_error")
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
                result_ids = {fact["call_id"] for fact in state["execution_facts"]}
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
            if state["stop_reason"] is None or private_checkpointer:
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
            if owned_store is not None:
                store_close_attempted = True
                try:
                    owned_store.close()
                except BaseException as exc:
                    if isinstance(exc, Exception):
                        previous = state["stop_reason"]
                        if state["cleanup_error"] is None:
                            state["cleanup_error"] = {
                                "code": "artifact_cleanup_error", "previous_stop_reason": previous,
                            }
                        try:
                            recorder.emit("run.cleanup_failed", stage="artifact_close",
                                          error_code="artifact_cleanup_error")
                        except BaseException:
                            pass  # Cleanup diagnostics never replace a primary failure.
                        if state["stop_reason"] != "error":
                            state.update(stop_reason="error", error=ARTIFACT_CLOSE_MESSAGE,
                                         error_code="artifact_cleanup_error")
                    elif primary_error is None:
                        primary_error = exc
        if primary_error is not None:
            # Preserve cancellation/configuration exceptions without logging their text.
            # There is no completed record if execution did not return a final State.
            configuration_rejected = isinstance(primary_error, PreExecutionConfigurationError)
            failure_code = "pre_execution_configuration_error" if configuration_rejected else "run_interrupted"
            if not configuration_rejected and checkpointer.failure(scope_id) is None and last_node is not None:
                # A cancelled tool batch may have produced real results before
                # LangGraph could checkpoint its update. Save those facts once;
                # this is never a resume, and the original exception still wins.
                state.update(stop_reason="error", error_code="run_interrupted", error="运行被中断。")
                terminal = make_record_batch(state["run_id"], state["record_sequence"], [{
                    "stop_reason": "error", "error_code": "run_interrupted",
                    "tool_outcomes": state["tool_outcomes"],
                }], kind="outcome")
                try:
                    receipts, watermark = journal.append(state["run_id"], terminal)
                    state.update(record_batch=state["record_batch"] + receipts,
                                 record_sequence=state["record_sequence"] + len(terminal),
                                 record_revision=state["record_revision"] + 1)
                    state["record_files"][state["run_id"]] = watermark
                    with tracing_context(enabled=False, parent=False):
                        await graph.aupdate_state(
                            {"configurable": {"thread_id": scope_id}},
                            {**deepcopy(state), "input": Overwrite(deepcopy(state["input"]))}, as_node="model",
                        )
                except BaseException:
                    pass  # Failed acknowledgement cannot replace the interruption.
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
                "last_confirmed_checkpoint_id": checkpointer.last_confirmed(scope_id),
                "checkpoint_error": checkpoint_error,
            } if checkpointer is not None else {}),
        )
        state["log_complete"] = recorder.close()
        if not state["log_complete"]:
            state.update(stop_reason="error", error=LOG_FAILURE_MESSAGE, error_code="logging_error")
        if checkpointer is not None:
            outcome = make_record_batch(state["run_id"], state["record_sequence"], [{
                "stop_reason": state["stop_reason"], "error_code": state["error_code"],
                "tool_outcomes": state["tool_outcomes"], "model_calls": state["model_calls"],
            }], kind="outcome")
            if checkpoint_error is None:
                try:
                    receipts, watermark = journal.append(state["run_id"], outcome)
                    state.update(record_batch=state["record_batch"] + receipts,
                                 record_sequence=state["record_sequence"] + len(outcome),
                                 record_revision=state["record_revision"] + 1)
                    state["record_files"][state["run_id"]] = watermark
                except JournalError as exc:
                    checkpoint_error = exc.code
                    state.update(stop_reason="error", error_code=exc.code, error=str(CheckpointError(exc.code)))
            try:
                originals = journal.records(state["run_id"], state["record_files"][state["run_id"]])
                try:
                    logical_bytes = sum(len(json.dumps(r, ensure_ascii=False, allow_nan=False).encode("utf-8"))
                                        for r in originals)
                except (TypeError, ValueError, UnicodeError):
                    # Tagged opaque values have no ordinary JSON byte measure.
                    logical_bytes = None
                # Snapshot before final serialization. Count serialized values, not Python heap memory.
                def binary_bytes(value):
                    if isinstance(value, bytes):
                        return len(value)
                    if isinstance(value, dict):
                        return sum(binary_bytes(child) for child in value.values())
                    if isinstance(value, (tuple, list)):
                        return sum(binary_bytes(child) for child in value)
                    return 0
                state["record_metrics"] = {
                    "count": len(originals),
                    "logical_bytes": logical_bytes,
                    "checkpoint_serialized_bytes": binary_bytes(checkpointer.storage.get(scope_id, {}))
                        + sum(binary_bytes(v) for k, v in checkpointer.blobs.items() if k[0] == scope_id)
                        + sum(binary_bytes(v) for k, v in checkpointer.writes.items() if k[0] == scope_id),
                    "file_bytes": state["record_files"][state["run_id"]]["byte_end"],
                    "measured_before_final_save": True,
                }
            except Exception:
                state["record_metrics"] = {"count": state["record_sequence"], "logical_bytes": None,
                    "file_bytes": state["record_files"].get(state["run_id"], {}).get("byte_end"),
                    "measured_before_final_save": True}
            # The JSONL intentionally ends before this final acknowledgement. Its
            # completion, including close errors, is part of the final saved State.
            checkpointer.unbind_run(scope_id)
            if checkpoint_error is None:
                final_config = {
                    "configurable": {"thread_id": scope_id},
                    "metadata": {"run_id": recorder.run_id, **({"session_id": session_id} if session_id else {})},
                }
                try:
                    final_state = deepcopy(state)
                    if state["stop_reason"] == "completed":
                        final_state["active_context"] = completed_context(state)
                    with tracing_context(enabled=False, parent=False):
                        saved_config = await graph.aupdate_state(
                            final_config, {**final_state, "input": Overwrite(deepcopy(final_state["input"]))},
                            as_node="model",
                        )
                    candidate_id = saved_config["configurable"]["checkpoint_id"]
                    confirmed = await checkpointer.read_confirmed(scope_id, candidate_id)
                    saved_values = confirmed.checkpoint["channel_values"]
                    if any(saved_values.get(key) != final_state[key] for key in (
                            "run_id", "active_context", "record_files", "record_revision", "stop_reason")):
                        raise CheckpointError("checkpoint_read_error")
                    final_checkpoint_id = candidate_id
                    state = final_state
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
            if checkpoint_error is not None and private_checkpointer:
                state.update(stop_reason="error", error_code=checkpoint_error,
                             error=str(CheckpointError(checkpoint_error)))
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
        if owned_store is not None and not store_close_attempted:
            try:
                owned_store.close()
            except BaseException as exc:
                cleanup_error = exc
        try:
            recorder.close()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        try:
            if checkpointer is not None:
                checkpointer.unbind_run(scope_id)
                if private_checkpointer:
                    checkpointer.release(scope_id)
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
