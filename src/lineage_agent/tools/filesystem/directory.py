"""List only the direct children of a directory inside a bound project."""

import json
from heapq import nsmallest
import os
from pathlib import Path
import stat
from typing import Any

from lineage_agent.config.settings import DEFAULT_DIRECTORY_MAX_ENTRIES, ConfigurationError
from lineage_agent.models.types import ToolDefinition
from lineage_agent.tools.filesystem.paths import (
    bind_project_root,
    is_link_or_reparse,
    resolve_project_path,
)
from lineage_agent.tools.types import ToolArgumentError, ToolExecutionError, ToolExecutor


DIRECTORY_TOOL: ToolDefinition = {
    "type": "function",
    "name": "list_directory",
    "description": (
        "列举绑定项目中指定相对目录的直接子项；path='.' 表示根目录。"
        "包含隐藏项，但不公开应用保留目录；不递归、不读取文件正文、不进入链接。"
        "第一页 after=null；truncated=true 时用 next_after 继续列举同一目录。"
        "按名称排序分页，未返回条目不代表不存在；目录变化时需从第一页重新发现。"
        "目录名称和结构不代表已经理解代码。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "after": {"type": ["string", "null"], "minLength": 1},
        },
        "required": ["path", "after"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _entry_type(info: os.stat_result) -> str:
    tag = getattr(info, "st_reparse_tag", 0)
    if stat.S_ISLNK(info.st_mode) or tag == stat.IO_REPARSE_TAG_SYMLINK:
        return "symlink"
    if tag == stat.IO_REPARSE_TAG_MOUNT_POINT:
        return "junction"
    if is_link_or_reparse(info):
        return "reparse_point"
    if stat.S_ISDIR(info.st_mode):
        return "directory"
    if stat.S_ISREG(info.st_mode):
        return "file"
    return "other"


def create_directory_executor(root: str | Path, *, max_entries: int = DEFAULT_DIRECTORY_MAX_ENTRIES) -> ToolExecutor:
    """Bind access once for an agent run; this tool never reads configuration."""
    return directory_executor(bind_project_root(root), max_entries=max_entries)


def directory_executor(project_root: Path, *, max_entries: int) -> ToolExecutor:
    """Construct from a validated root shared with other filesystem tools."""
    return DirectoryExecutor(project_root, max_entries)


class DirectoryExecutor:
    """A directory binding whose actual parameters can be described safely."""

    def __init__(self, project_root: Path, max_entries: int) -> None:
        if type(max_entries) is not int or max_entries <= 0:
            raise ConfigurationError("DIRECTORY_MAX_ENTRIES 必须是正整数。")
        self._root, self._max_entries = project_root, max_entries

    def describe_binding(self) -> dict[str, Any]:
        return {"provider": "builtin.directory", "version": 1,
                "options": {"root": str(self._root), "max_entries": self._max_entries}}

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        project_root, max_entries = self._root, self._max_entries
        if (
            name != DIRECTORY_TOOL["name"]
            or not isinstance(arguments, dict)
            or not {"path"} <= set(arguments) <= {"path", "after"}
            or not isinstance(arguments["path"], str)
        ):
            raise ToolArgumentError(reason_code="invalid_argument_shape")
        after = arguments.get("after")
        if after is not None and (
            not isinstance(after, str) or not after or "/" in after or "\\" in after
        ):
            raise ToolArgumentError(reason_code="invalid_directory_cursor")
        cursor = (after.casefold(), after) if after is not None else None
        try:
            target = resolve_project_path(project_root, arguments["path"])
            info = target.lstat()
            if is_link_or_reparse(info):
                raise ToolArgumentError(reason_code="link_or_reparse_forbidden")
            if not stat.S_ISDIR(info.st_mode):
                raise ToolArgumentError(reason_code="target_not_directory")
            relative = target.relative_to(project_root)
            # Scan each page fully; the generator + heap retains at most N+1
            # candidates, without first materializing an unbounded directory.
            with os.scandir(target) as iterator:
                candidates = nsmallest(
                    max_entries + 1,
                    (entry for entry in iterator if entry.name.casefold() != ".lineage"
                     and (cursor is None or (entry.name.casefold(), entry.name) > cursor)),
                    key=lambda entry: (entry.name.casefold(), entry.name),
                )
                truncated = len(candidates) > max_entries
                entries = [{
                    "name": entry.name,
                    "path": (relative / entry.name).as_posix(),
                    "type": _entry_type(entry.stat(follow_symlinks=False)),
                } for entry in candidates[:max_entries]]
        except FileNotFoundError as exc:
            raise ToolExecutionError(reason_code="file_not_found") from exc
        except PermissionError as exc:
            raise ToolExecutionError(reason_code="permission_denied") from exc
        except OSError as exc:
            raise ToolExecutionError(reason_code="io_error") from exc
        return json.dumps({
            "path": relative.as_posix(), "entries": entries,
            "limit": max_entries, "truncated": truncated,
            "after": after, "next_after": entries[-1]["name"] if truncated else None,
        }, ensure_ascii=False, allow_nan=False)
