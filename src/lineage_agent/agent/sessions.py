"""Exclusive conversation ownership, immutable publication and explicit reopening."""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import logging
from threading import Lock, RLock
from typing import Literal
from uuid import uuid4

from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.context import read_record
from lineage_agent.agent.runtime import (
    PreExecutionConfigurationError, _run_agent, normalize_input, validate_tools,
)
from lineage_agent.agent.state import AgentState
from lineage_agent.agent.recovery import validate_snapshot_state
from lineage_agent.archive import ArchiveError, ArchiveStore
from lineage_agent.artifacts.store import ArtifactError, ArtifactStore, SessionArtifacts
from lineage_agent.artifacts.tools import ARTIFACT_TOOLS, ArtifactExecutor
from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.config.recovery import (
    serialize_settings, restore_settings, describe_tools, restore_tools, configuration_warnings,
)
from lineage_agent.models.types import ModelInput, ToolDefinition
from lineage_agent.observations.hints import enhance_tools
from lineage_agent.tools.types import ToolExecutor
from lineage_agent.tools.history import HISTORY_TOOL_NAMES
from lineage_agent.storage import StorageError, conversation_paths, ConversationLease


_SESSION_ERRORS = {
    "unknown_session": "会话不存在，请使用当前管理器创建的会话标识。",
    "session_busy": "该会话正在处理请求，本次提交未执行。",
    "session_blocked": "该会话已阻断；请显式关闭并重新打开，选择已发布恢复点，不会自动重试。",
    "unknown_record": "此会话中没有该运行的已确认记录。",
    "artifacts_unavailable": "此会话未启用 ArtifactStore。",
    "session_manager_closed": "会话管理器已关闭，不能继续使用。",
    "session_already_open": "该会话已在此管理器打开；请切换，或先关闭后显式重新打开。",
    "conversation_busy": "另一个会话实例正持有此对话的写入保护。",
    "conversation_not_found": "指定对话不存在或无法安全定位。",
    "no_snapshot": "此对话尚无已发布恢复点，不能恢复为空白会话。",
    "recovery_required": "存在未封存轮次；须明确选择从最后已发布快照开始新轮次。",
    "recovery_changed": "恢复点或未封存轮次已变化，请重新检查后选择。",
    "archive_version_error": "保存格式不受支持，本版本不迁移旧对话。",
    "archive_integrity_error": "快照或必需资源的身份、关联或完整性校验失败。",
    "archive_read_error": "会话归档无法读取。",
    "archive_write_error": "会话登记未能保存。",
    "archive_publication_unknown": "快照发布结果无法确认，当前实例已阻断。",
}


class SessionError(RuntimeError):
    """A rejected submission; no new run or model call has started."""

    def __init__(self, code: str, *, details: dict | None = None):
        self.code = code if code in _SESSION_ERRORS else "unknown_session"
        self.details = deepcopy(details or {})
        super().__init__(_SESSION_ERRORS[self.code])


@dataclass(frozen=True)
class SessionTurn:
    """Execution facts and the separate acknowledgement of final state saving."""

    session_id: str
    state: AgentState
    checkpoint_id: str | None
    checkpoint_complete: bool
    checkpoint_error: str | None
    snapshot_id: str | None = None
    snapshot_status: Literal["published", "not_published", "unknown"] = "not_published"


@dataclass(frozen=True)
class RecoveryChoice:
    snapshot_id: str
    unfinished_run_ids: tuple[str, ...]


@dataclass(frozen=True)
class SessionOpened:
    session_id: str
    snapshot_id: str
    settings: ModelSettings
    warnings: tuple[str, ...] = ()


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
    snapshot_id: str | None = None
    lease: ConversationLease | None = None


class SessionManager:
    """Own checkpoints and trusted bindings, without a second history cache.

    Reopening imports a published terminal state. It never resumes pending tasks.
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
            failure = None
            for session_id in list(self._sessions):
                try:
                    self._release_session(session_id)
                except BaseException as error:
                    failure = failure or error
            if self._owned_store is not None:
                try:
                    self._owned_store.close()
                except Exception:
                    self.cleanup_error = "artifact_cleanup_error"
                    raise ArtifactError("artifact_cleanup_error") from None
                self._owned_store = None
            if failure is not None:
                self.cleanup_error = "artifact_cleanup_error"
                raise failure
            self.cleanup_error = None

    def _discard_failed_open(self, session_id, artifacts, archive, lease, *, bootstrapped):
        operations = []
        if artifacts is not None:
            operations.append(artifacts.discard)
        if bootstrapped:
            operations.append(lambda: self._checkpointer.release(session_id))
        if archive is not None:
            operations.append(archive.close)
        if lease is not None:
            operations.append(lease.close)
        for operation in operations:
            try:
                operation()
            except BaseException:
                # All handles get a release attempt; cleanup cannot replace the
                # opening failure or prevent the OS lease from being released.
                logging.getLogger(__name__).warning("会话打开失败；部分临时句柄清理未完成。")

    def _release_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id)
        try:
            if session.artifacts is not None:
                session.artifacts.release()
        finally:
            try:
                self._checkpointer.release(session_id)
            finally:
                if session.lease is not None:
                    session.lease.close()

    def close_session(self, session_id: str) -> None:
        """Revoke this binding's writers before relinquishing process ownership."""
        with self._lifecycle_lock:
            self._ensure_open()
            if session_id not in self._sessions:
                raise SessionError("unknown_session")
            if self._active_operations:
                raise SessionError("session_busy")
            self._release_session(session_id)

    def settings(self, session_id: str) -> ModelSettings:
        with self._lifecycle_lock:
            self._ensure_open()
            if session_id not in self._sessions:
                raise SessionError("unknown_session")
            return deepcopy(self._sessions[session_id].settings)

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
            artifact_names = {tool["name"] for tool in ARTIFACT_TOOLS} | HISTORY_TOOL_NAMES
            if any(tool["name"] in artifact_names for tool in available_tools):
                raise ConfigurationError("已提供的工具名称与内置召回工具冲突。")
            # Reject incompatible reserved schemas before allocating storage.
            enhance_tools(available_tools)
            configuration = {"settings": serialize_settings(settings),
                             "binding": describe_tools(available_tools, execute_tool)}
            session_id = str(uuid4())
            if artifact_store is None:
                if self._owned_store is None:
                    self._owned_store = ArtifactStore()
                artifact_store = self._owned_store
            lease = archive = artifacts = None
            try:
                paths = conversation_paths(settings.application_root, session_id)
                lease = ConversationLease.acquire(paths.directory, create=True)
                archive = ArchiveStore(paths.directory)
                archive.register_conversation(configuration, available_tools)
                self._checkpointer.bind_archive(session_id, archive)
                artifacts = artifact_store.bind_session(session_id, directory=paths.artifacts, lease=lease)
            except BaseException:
                self._discard_failed_open(session_id, artifacts, archive, lease, bootstrapped=True)
                raise
            execute_tool = ArtifactExecutor(artifacts, execute_tool)
            available_tools = validate_tools([*available_tools, *ARTIFACT_TOOLS], execute_tool)
            self._sessions[session_id] = _Session(
                deepcopy(settings), available_tools, execute_tool,
                artifacts=artifacts, lease=lease,
            )
            return session_id

    @staticmethod
    def list_sessions(application_root: Path) -> list[dict]:
        """Discover registrations only; listing does not authorize body resources."""
        from lineage_agent.tools.filesystem.paths import _check_components
        import stat
        root = Path(application_root).absolute() / ".lineage"
        if not root.exists():
            return []
        try:
            if not stat.S_ISDIR(_check_components(root).st_mode):
                raise ValueError
            directories = sorted(root.iterdir(), key=lambda path: path.name)
        except Exception:
            raise SessionError("archive_read_error") from None
        result = []
        for directory in directories:
            if not (directory / "conversation.sqlite").exists():
                continue
            item = {"conversation_id": directory.name, "current_snapshot_id": None,
                    "unfinished_runs": [], "status": "damaged", "updated_at": None, "error_code": None}
            try:
                paths = conversation_paths(application_root, directory.name, create=False)
                with ArchiveStore(paths.directory, create=False, read_only=True) as archive:
                    metadata = archive.metadata()
                    item.update(metadata)
                    item["status"] = ("no_snapshot" if not item["current_snapshot_id"] else
                                      "recovery_required" if item["unfinished_runs"] else "ready")
            except (ArchiveError, StorageError) as error:
                item["error_code"] = error.code
                item["status"] = "unsupported" if error.code == "archive_version_error" else "damaged"
            result.append(item)
        return result

    async def open_session(
        self, conversation_id: str, *, application_root: Path,
        current_project_root: Path | None = None, log_dir: Path | None = None,
        tools: list[ToolDefinition] | None = None, execute_tool: ToolExecutor | None = None,
        artifact_store: ArtifactStore | None = None, recovery: RecoveryChoice | None = None,
        current_limits: dict | None = None,
    ) -> SessionOpened:
        """Validate and import the current published snapshot without executing a Run."""
        with self._lifecycle_lock:
            self._ensure_open()
            if conversation_id in self._sessions:
                raise SessionError("session_already_open")
            self._active_operations += 1
        lease = archive = artifacts = None
        registered = bootstrapped = False
        try:
            paths = conversation_paths(application_root, conversation_id, create=False)
            try:
                lease = ConversationLease.acquire(paths.directory, create=False)
            except StorageError as error:
                if error.code != "conversation_busy":
                    # Old formats have no lease file. Identify their version
                    # read-only rather than reporting them as nonexistent.
                    with ArchiveStore(paths.directory, create=False, read_only=True):
                        pass
                raise
            archive = ArchiveStore(paths.directory, create=False)
            metadata = archive.conversation()
            if metadata["current_snapshot_id"] is None:
                raise SessionError("no_snapshot")
            snapshot = archive.read_snapshot()
            snapshot_id = snapshot["snapshot_id"]
            state = deepcopy(snapshot["final_state"])
            view = archive.view(snapshot["view_id"])
            state["record_files"] = {run: state["record_files"][run] for run in view["record_order"]}
            validate_snapshot_state(state, archive, snapshot["view_id"])
            trusted_runs = archive.published_runs(snapshot_id)
            unfinished = tuple(item["run_id"] for item in archive.unfinished_runs())
            details = {"snapshot_id": snapshot_id, "unfinished_run_ids": list(unfinished)}
            if unfinished and recovery is None:
                raise SessionError("recovery_required", details=details)
            if recovery is not None and (recovery.snapshot_id != snapshot_id
                                        or set(recovery.unfinished_run_ids) != set(unfinished)
                                        or len(recovery.unfinished_run_ids) != len(unfinished)):
                raise SessionError("recovery_changed", details=details)
            configuration = metadata["configuration"]
            settings = restore_settings(configuration["settings"], application_root=application_root, log_dir=log_dir)
            if configuration["binding"]["tools"] != metadata["tools"]:
                raise ArchiveError("archive_integrity_error")
            dependencies = restore_tools(configuration["binding"], current_project_root=current_project_root,
                                         tools=tools, execute_tool=execute_tool)
            available_tools = validate_tools(dependencies["tools"], dependencies["execute_tool"])
            reserved = {tool["name"] for tool in ARTIFACT_TOOLS} | HISTORY_TOOL_NAMES
            if any(tool["name"] in reserved for tool in available_tools):
                raise ArchiveError("archive_integrity_error")
            enhance_tools(available_tools)
            warnings = tuple(configuration_warnings(settings, configuration["binding"], current_limits=current_limits))
            if artifact_store is None:
                if self._owned_store is None:
                    self._owned_store = ArtifactStore()
                artifact_store = self._owned_store
            artifacts = artifact_store.restore_session(conversation_id, archive=archive, view_id=snapshot["view_id"], lease=lease)
            checkpoint_id = await self._checkpointer.restore_state(conversation_id, state, archive=archive, artifacts=artifacts)
            bootstrapped = True
            if unfinished:
                archive.accept_recovery(snapshot_id, list(unfinished))
            executor = ArtifactExecutor(artifacts, dependencies["execute_tool"])
            session = _Session(settings, [*available_tools, *deepcopy(ARTIFACT_TOOLS)], executor,
                               artifacts=artifacts, last_checkpoint_id=checkpoint_id,
                               records={item["run_id"]: item["checkpoint_id"] for item in trusted_runs},
                               snapshot_id=snapshot_id, lease=lease)
            with self._lifecycle_lock:
                self._ensure_open()
                self._sessions[conversation_id] = session
                registered = True
            return SessionOpened(conversation_id, snapshot_id, deepcopy(settings), warnings)
        except StorageError as error:
            code = ("conversation_busy" if error.code == "conversation_busy" else
                    "archive_integrity_error" if error.code == "storage_lock_error" else "conversation_not_found")
            raise SessionError(code) from None
        except (ArchiveError, CheckpointError) as error:
            raise SessionError(error.code if error.code in _SESSION_ERRORS else "archive_integrity_error") from None
        except (SessionError, ConfigurationError):
            raise
        except Exception:
            raise SessionError("archive_integrity_error") from None
        finally:
            try:
                if not registered:
                    self._discard_failed_open(conversation_id, artifacts, archive, lease, bootstrapped=bootstrapped)
            finally:
                with self._lifecycle_lock:
                    self._active_operations -= 1

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
            publication = {}
            try:
                state, checkpoint_id, error = await _run_agent(
                    new_input, session.settings, tools=session.tools,
                    execute_tool=session.execute_tool,
                    session_id=session_id, checkpointer=self._checkpointer,
                    checkpoint_id=session.last_checkpoint_id,
                    artifacts=session.artifacts,
                    source_snapshot_id=session.snapshot_id, publication=publication,
                )
            except PreExecutionConfigurationError:
                # This signal is produced only before entering the client or
                # graph. No new checkpoint or execution must be resumed.
                raise
            except BaseException:
                # Cancellation is not a new request or an implicit resume. Its
                # potentially partial writes cannot be used as a committed turn.
                if publication.get("snapshot_status") == "published":
                    session.snapshot_id = publication["snapshot_id"]
                    session.last_checkpoint_id = publication["checkpoint_id"]
                    session.records[publication["run_id"]] = publication["checkpoint_id"]
                session.blocked = True
                raise
            if error is not None:
                session.blocked = True
            else:
                session.last_checkpoint_id = checkpoint_id
                if checkpoint_id is not None:
                    session.records[state["run_id"]] = checkpoint_id
                    session.snapshot_id = publication.get("snapshot_id")
            return SessionTurn(
                session_id, deepcopy(state), checkpoint_id,
                checkpoint_id is not None and error is None, error,
                publication.get("snapshot_id"), publication.get("snapshot_status", "not_published"),
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
