"""List only the direct children of a directory inside a bound project."""

import json
import os
from pathlib import Path
import stat
from typing import Any

from lineage_agent.config.settings import ConfigurationError
from lineage_agent.models.types import ToolDefinition
from lineage_agent.tools.filesystem.paths import (
    bind_project_root,
    is_link_or_reparse,
    resolve_project_path,
)
from lineage_agent.tools.types import ToolArgumentError, ToolExecutor


DIRECTORY_TOOL: ToolDefinition = {
    "type": "function",
    "name": "list_directory",
    "description": (
        "列举绑定项目中指定相对目录的直接子项；path='.' 表示根目录。"
        "包含隐藏项，不递归、不读取文件正文、不进入链接。"
        "truncated=true 时结果不完整，未返回条目不代表不存在；"
        "排序仅作用于返回子集。目录名称和结构不代表已经理解代码。"
    ),
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string", "minLength": 1}},
        "required": ["path"],
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


def create_directory_executor(root: str | Path, *, max_entries: int = 100) -> ToolExecutor:
    """Bind access once for an agent run; this tool never reads configuration."""
    if type(max_entries) is not int or max_entries <= 0:
        raise ConfigurationError("DIRECTORY_MAX_ENTRIES 必须是正整数。")
    project_root = bind_project_root(root)

    async def execute_tool(name: str, arguments: dict[str, Any]) -> str:
        if (
            name != DIRECTORY_TOOL["name"]
            or not isinstance(arguments, dict)
            or set(arguments) != {"path"}
            or not isinstance(arguments["path"], str)
        ):
            raise ToolArgumentError("list_directory 仅接受一个字符串字段 path。")
        target = resolve_project_path(project_root, arguments["path"])
        info = target.lstat()
        if is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
            raise ToolArgumentError("path 必须指向不经过链接的项目内目录。")
        relative = target.relative_to(project_root)
        entries = []
        truncated = False
        # Keep at most N entries, consuming one extra only to detect truncation.
        # Any iteration/stat/close error discards this partial result by raising.
        with os.scandir(target) as iterator:
            for index, entry in enumerate(iterator):
                if index == max_entries:
                    truncated = True
                    break
                entries.append({
                    "name": entry.name,
                    "path": (relative / entry.name).as_posix(),
                    "type": _entry_type(entry.stat(follow_symlinks=False)),
                })
        entries.sort(key=lambda entry: (entry["name"].casefold(), entry["name"]))
        return json.dumps({
            "path": relative.as_posix(), "entries": entries,
            "limit": max_entries, "truncated": truncated,
        }, ensure_ascii=False, allow_nan=False)

    return execute_tool
