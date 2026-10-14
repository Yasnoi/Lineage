"""Small, trusted path helpers for persistent conversation files."""

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import re
import stat

from lineage_agent.tools.filesystem.paths import _check_components


class StorageError(Exception):
    code = "storage_create_error"

    def __init__(self) -> None:
        super().__init__("会话存储目录无法安全创建或访问。")


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


@dataclass(frozen=True)
class ConversationPaths:
    directory: Path
    history: Path
    artifacts: Path

    @property
    def database(self) -> Path:
        return self.directory / "conversation.sqlite"


def conversation_paths(application_root: Path, conversation_id: str) -> ConversationPaths:
    try:
        directory = Path(application_root).absolute() / ".lineage" / storage_identifier(conversation_id)
        return ConversationPaths(directory, ensure_storage_directory(directory / "history"),
                                 ensure_storage_directory(directory / "artifacts"))
    except Exception:
        raise StorageError() from None
