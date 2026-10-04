"""Check project-relative paths without permitting links or reparse traversal."""

import os
from pathlib import Path, PureWindowsPath
import stat
import tempfile

from lineage_agent.config.settings import ConfigurationError
from lineage_agent.tools.types import ToolArgumentError


# This location is reserved even before a Store is opened and after it closes.
# It must not be inferred from settings.project_root: injected executors can
# have different roots, and old executors must observe the same boundary.
ARTIFACT_CACHE_ROOT = Path(tempfile.gettempdir()).absolute() / "lineage-agent-artifacts"


def _check_artifact_boundary(path: Path) -> None:
    if any(component.casefold() == ".lineage" for component in path.parts):
        raise ToolArgumentError(reason_code="storage_path_forbidden")
    if path.is_relative_to(ARTIFACT_CACHE_ROOT):
        raise ToolArgumentError(reason_code="artifact_path_forbidden")


def is_link_or_reparse(info: os.stat_result) -> bool:
    """Use no-follow metadata, including Windows reparse types beyond junctions."""
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _check_components(path: Path) -> os.stat_result:
    # Check ancestors too: inspecting only the final component misses linked parents.
    for component in (*reversed(path.parents), path):
        info = component.lstat()
        if is_link_or_reparse(info):
            raise ToolArgumentError(
                "不允许通过链接或重解析点访问项目目录。", reason_code="link_or_reparse_forbidden"
            )
    return info


def bind_project_root(root: str | Path) -> Path:
    """Validate a trusted root before resolving it, without changing cwd."""
    try:
        if isinstance(root, str) and not root.strip():
            raise ValueError("empty root")
        path = Path(root).absolute()
        _check_artifact_boundary(path)
        info = _check_components(path)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("not a directory")
        resolved = path.resolve(strict=True)
        _check_artifact_boundary(resolved)
        return resolved
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ConfigurationError(
            "PROJECT_ROOT 必须是存在且可访问的目录，其自身及祖先不能包含链接或重解析点，"
            "且不能位于应用保留目录。"
        ) from exc


def _relative_parts(value: str) -> list[str]:
    if not isinstance(value, str) or not value:
        raise ToolArgumentError("path 必须是非空项目内相对路径。", reason_code="empty_path")
    windows_path = PureWindowsPath(value)
    if windows_path.drive or windows_path.root:
        raise ToolArgumentError("path 必须是项目内相对路径。", reason_code="non_relative_path")
    parts: list[str] = []
    for part in value.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if (
            part == ".."
            or part.endswith((".", " "))
            or any(character in '<>:"|?*' for character in part)
            or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in part)
            or PureWindowsPath(part).is_reserved()
        ):
            raise ToolArgumentError("path 包含不允许的路径组件。", reason_code="disallowed_path_component")
        parts.append(part)
    return parts


def resolve_project_path(root: Path, path: str) -> Path:
    """Resolve a target under an already bound root; its type is tool-specific.

    Component checks and filesystem use are not atomic. This is an application
    boundary for stable paths, not an OS sandbox against concurrent replacement.
    """
    target = root.joinpath(*_relative_parts(path))
    _check_artifact_boundary(target)
    _check_components(target)
    resolved = target.resolve(strict=True)
    _check_artifact_boundary(resolved)
    if not resolved.is_relative_to(root):
        raise ToolArgumentError("path 超出本次绑定的项目目录。", reason_code="outside_project_root")
    return resolved
