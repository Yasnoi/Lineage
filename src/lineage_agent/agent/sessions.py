"""Process-local conversations backed only by LangGraph checkpoints."""

from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock, RLock
from uuid import uuid4

from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.context import read_record
from lineage_agent.agent.runtime import (
    PreExecutionConfigurationError, _run_agent, normalize_input, validate_tools,
)
from lineage_agent.agent.state import AgentState
from lineage_agent.artifacts.store import ArtifactError, ArtifactStore, SessionArtifacts
from lineage_agent.artifacts.tools import ARTIFACT_TOOLS, ArtifactExecutor
from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.models.types import ModelInput, ToolDefinition
from lineage_agent.observations.hints import enhance_tools
from lineage_agent.tools.types import ToolExecutor


_SESSION_ERRORS = {
    "unknown_session": "会话不存在，请使用当前管理器创建的会话标识。",
    "session_busy": "该会话正在处理请求，本次提交未执行。",
    "session_blocked": "该会话的保存或执行完整性未确认，请新建会话；不会自动恢复或重试。",
    "unknown_record": "此会话中没有该运行的已确认记录。",
    "artifacts_unavailable": "此会话未启用 ArtifactStore。",
    "session_manager_closed": "会话管理器已关闭，不能继续使用。",
}


class SessionError(RuntimeError):
    """A rejected submission; no new run or model call has started."""

    def __init__(self, code: str):
        self.code = code if code in _SESSION_ERRORS else "unknown_session"
        super().__init__(_SESSION_ERRORS[self.code])


@dataclass(frozen=True)
class SessionTurn:
    """Execution facts and the separate acknowledgement of final state saving."""

    session_id: str
    state: AgentState
    checkpoint_id: str | None
    checkpoint_complete: bool
    checkpoint_error: str | None


@dataclass
class _Session:
    settings: ModelSettings
    tools: list[ToolDefinition]
    execute_tool: ToolExecutor | None
    artifacts: SessionArtifacts | None = None
    last_checkpoint_id: str | None = None
    blocked: bool = False
    lock: Lock = field(default_factory=Lock)
    records: dict[str, str] = field(default_factory=dict)


class SessionManager:
    """Own checkpoints and trusted bindings, without a second history cache.

    Session identifiers belong to this manager. Dropping the manager or exiting
    the process discards all saved conversations. No resume API is exposed.
    """

    def __init__(self) -> None:
        try:
            self._checkpointer = SessionCheckpointer()
        except Exception:
            raise CheckpointError("checkpoint_create_error") from None
        self._sessions: dict[str, _Session] = {}
        self._lifecycle_lock = RLock()
        self._active_operations = 0
        self._closed = False
        self._owned_store: ArtifactStore | None = None
        self.cleanup_error: str | None = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise SessionError("session_manager_closed")

    def close(self) -> None:
        """Close owned storage when idle; borrowed stores remain caller-owned."""
        with self._lifecycle_lock:
            if self._active_operations:
                raise SessionError("session_busy")
            self._closed = True
            if self._owned_store is not None:
                try:
                    self._owned_store.close()
                except Exception:
                    self.cleanup_error = "artifact_cleanup_error"
                    raise ArtifactError("artifact_cleanup_error") from None
                self._owned_store = None
            self.cleanup_error = None

    def __enter__(self) -> "SessionManager":
        with self._lifecycle_lock:
            self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except BaseException:
            if exc_type is None:
                raise
            # The original exception/cancellation stays primary. close() records
            # a safe cleanup code and keeps the owned store available for retry.

    def create_session(
        self,
        settings: ModelSettings,
        *,
        tools: list[ToolDefinition] | None = None,
        execute_tool: ToolExecutor | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> str:
        """Freeze the available capabilities; each request binds fresh budgets."""
        with self._lifecycle_lock:
            self._ensure_open()
            available_tools = validate_tools(tools, execute_tool)
            artifact_names = {tool["name"] for tool in ARTIFACT_TOOLS}
            if any(tool["name"] in artifact_names for tool in available_tools):
                raise ConfigurationError("已提供的工具名称与 Artifact 工具冲突。")
            # Reject incompatible reserved schemas before allocating storage.
            enhance_tools(available_tools)
            session_id = str(uuid4())
            if artifact_store is None:
                if self._owned_store is None:
                    self._owned_store = ArtifactStore()
                artifact_store = self._owned_store
            artifacts = artifact_store.bind_session(session_id)
            execute_tool = ArtifactExecutor(artifacts, execute_tool)
            available_tools = validate_tools([*available_tools, *ARTIFACT_TOOLS], execute_tool)
            self._sessions[session_id] = _Session(
                deepcopy(settings), available_tools, execute_tool,
                artifacts=artifacts,
            )
            return session_id

    def artifacts(self, session_id: str) -> SessionArtifacts:
        """Expose trusted storage access for this manager's enabled session."""
        with self._lifecycle_lock:
            self._ensure_open()
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionError("unknown_session")
            if session.artifacts is None:
                raise SessionError("artifacts_unavailable")
            return session.artifacts

    async def run_turn(self, session_id: str, input: ModelInput) -> SessionTurn:
        with self._lifecycle_lock:
            self._ensure_open()
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionError("unknown_session")
            if not session.lock.acquire(blocking=False):
                raise SessionError("session_busy")
            self._active_operations += 1
        try:
            if session.blocked:
                raise SessionError("session_blocked")
            # Bad caller input has not touched the history and need not block it.
            new_input = normalize_input(input)
            try:
                state, checkpoint_id, error = await _run_agent(
                    new_input, session.settings, tools=session.tools,
                    execute_tool=session.execute_tool,
                    session_id=session_id, checkpointer=self._checkpointer,
                    checkpoint_id=session.last_checkpoint_id,
                    artifacts=session.artifacts,
                )
            except PreExecutionConfigurationError:
                # This signal is produced only before entering the client or
                # graph. No new checkpoint or execution must be resumed.
                raise
            except BaseException:
                # Cancellation is not a new request or an implicit resume. Its
                # potentially partial writes cannot be used as a committed turn.
                session.blocked = True
                raise
            if error is not None:
                session.blocked = True
            else:
                session.last_checkpoint_id = checkpoint_id
                if checkpoint_id is not None:
                    session.records[state["run_id"]] = checkpoint_id
            return SessionTurn(
                session_id, deepcopy(state), checkpoint_id,
                checkpoint_id is not None and error is None, error,
            )
        finally:
            with self._lifecycle_lock:
                session.lock.release()
                self._active_operations -= 1

    async def get_record(self, session_id: str, run_id: str) -> dict:
        """Retrieve a confirmed run's original facts without executing the graph."""
        with self._lifecycle_lock:
            self._ensure_open()
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionError("unknown_session")
            checkpoint_id = session.records.get(run_id)
            if checkpoint_id is None:
                raise SessionError("unknown_record")
            self._active_operations += 1
        try:
            return await read_record(self._checkpointer, session_id, {
                "run_id": run_id, "checkpoint_id": checkpoint_id,
            })
        finally:
            with self._lifecycle_lock:
                self._active_operations -= 1
