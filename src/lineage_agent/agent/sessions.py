"""Process-local conversations backed only by LangGraph checkpoints."""

from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from uuid import uuid4

from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.runtime import _run_agent, normalize_input, validate_tools
from lineage_agent.agent.state import AgentState
from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.types import ModelInput, ToolDefinition
from lineage_agent.tools.types import ToolExecutor


_SESSION_ERRORS = {
    "unknown_session": "会话不存在，请使用当前管理器创建的会话标识。",
    "session_busy": "该会话正在处理请求，本次提交未执行。",
    "session_blocked": "该会话的保存或执行完整性未确认，请新建会话；不会自动恢复或重试。",
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
    last_checkpoint_id: str | None = None
    blocked: bool = False
    lock: Lock = field(default_factory=Lock)


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

    def create_session(
        self,
        settings: ModelSettings,
        *,
        tools: list[ToolDefinition] | None = None,
        execute_tool: ToolExecutor | None = None,
    ) -> str:
        """Freeze the available capabilities; each request binds fresh budgets."""
        available_tools = validate_tools(tools, execute_tool)
        session_id = str(uuid4())
        self._sessions[session_id] = _Session(
            deepcopy(settings), available_tools, execute_tool,
        )
        return session_id

    async def run_turn(self, session_id: str, input: ModelInput) -> SessionTurn:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionError("unknown_session")
        if not session.lock.acquire(blocking=False):
            raise SessionError("session_busy")
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
                )
            except BaseException:
                # Cancellation is not a new request or an implicit resume. Its
                # potentially partial writes cannot be used as a committed turn.
                session.blocked = True
                raise
            if error is not None:
                session.blocked = True
            else:
                session.last_checkpoint_id = checkpoint_id
            return SessionTurn(
                session_id, deepcopy(state), checkpoint_id,
                checkpoint_id is not None and error is None, error,
            )
        finally:
            session.lock.release()
