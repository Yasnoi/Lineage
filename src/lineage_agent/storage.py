"""Small, trusted path helpers for persistent conversation files."""

from dataclasses import dataclass
import os
from pathlib import Path, PureWindowsPath
import re
import stat

from lineage_agent.tools.filesystem.paths import _check_components


class StorageError(Exception):
    code = "storage_create_error"

    def __init__(self, code: str = "storage_create_error") -> None:
        messages = {
            "storage_create_error": "会话存储目录无法安全创建或访问。",
            "storage_lock_error": "会话写锁无法安全打开或释放。",
            "conversation_busy": "该会话已由其他活动句柄占用。",
        }
        self.code = code if code in messages else "storage_create_error"
        super().__init__(messages[self.code])


def storage_identifier(value: str) -> str:
    if (type(value) is not str or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) is None
            or PureWindowsPath(value).is_reserved()):
        raise StorageError()
    return value


def ensure_storage_directory(directory: Path) -> Path:
    """Create components individually and reject links before following them."""
    try:
        path = Path(directory).absolute()
        for component in (*reversed(path.parents), path):
            if not component.exists():
                component.mkdir(exist_ok=True)
            if not stat.S_ISDIR(_check_components(component).st_mode):
                raise ValueError("not a directory")
        return path
    except Exception:
        raise StorageError() from None


def existing_storage_directory(directory: Path) -> Path:
    """Validate an existing directory without creating any missing component."""
    try:
        path = Path(directory).absolute()
        if not stat.S_ISDIR(_check_components(path).st_mode):
            raise ValueError("not a directory")
        return path
    except Exception:
        raise StorageError() from None


@dataclass(frozen=True)
class ConversationPaths:
    directory: Path
    history: Path
    artifacts: Path

    @property
    def database(self) -> Path:
        return self.directory / "conversation.sqlite"

    @property
    def lock(self) -> Path:
        return self.directory / "conversation.lock"


def conversation_paths(application_root: Path, conversation_id: str, *,
                       create: bool = True) -> ConversationPaths:
    try:
        directory = Path(application_root).absolute() / ".lineage" / storage_identifier(conversation_id)
        resolve = ensure_storage_directory if create else existing_storage_directory
        return ConversationPaths(directory, resolve(directory / "history"),
                                 resolve(directory / "artifacts"))
    except Exception:
        raise StorageError() from None


class ConversationLease:
    """Process-owned exclusive writer lock; closing the OS handle releases it.

    The lock file is persistent, so releasing a lease never creates an unlink /
    replacement race. An interrupted process also releases its operating-system
    lock without needing a stale PID heuristic.
    """

    def __init__(self, handle) -> None:
        self._handle = handle

    @classmethod
    def acquire(cls, directory: Path, *, create: bool = False) -> "ConversationLease":
        handle = None
        try:
            path = existing_storage_directory(directory) / "conversation.lock"
            if create:
                try:
                    handle = path.open("x+b")
                except FileExistsError:
                    pass
                else:
                    handle.write(b"\0")
                    handle.flush()
                    os.fsync(handle.fileno())
            if handle is None:
                if not stat.S_ISREG(_check_components(path).st_mode):
                    raise ValueError("invalid lock file")
                handle = path.open("r+b")
            if not stat.S_ISREG(_check_components(path).st_mode) or os.fstat(handle.fileno()).st_size != 1:
                raise ValueError("invalid lock file")
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise StorageError("conversation_busy") from None
            return cls(handle)
        except Exception as exc:
            if handle is not None:
                handle.close()
            if isinstance(exc, StorageError) and exc.code == "conversation_busy":
                raise
            raise StorageError("storage_lock_error") from None

    @property
    def active(self) -> bool:
        return self._handle is not None

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            # Closing releases either platform's lock, even if callers did not
            # explicitly unlock first. Do not delete the stable lock file.
            try:
                handle.close()
            except Exception:
                raise StorageError("storage_lock_error") from None

    release = close

    def __enter__(self) -> "ConversationLease":
        if not self.active:
            raise StorageError("storage_lock_error")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
