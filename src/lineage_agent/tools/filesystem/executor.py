"""Combine the two filesystem tools without changing the generic executor."""

from pathlib import Path
from typing import Any

from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL, create_directory_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL, create_text_executor
from lineage_agent.tools.types import ToolArgumentError, ToolExecutor


def create_filesystem_executor(
    root: str | Path, *, max_entries: int = 100, max_bytes: int = 65536,
) -> ToolExecutor:
    """Bind both capabilities; run_agent's supplied schemas control exposure."""
    directory = create_directory_executor(root, max_entries=max_entries)
    text = create_text_executor(root, max_bytes=max_bytes)

    async def execute_tool(name: str, arguments: dict[str, Any]) -> str:
        if name == DIRECTORY_TOOL["name"]:
            return await directory(name, arguments)
        if name == TEXT_TOOL["name"]:
            return await text(name, arguments)
        raise ToolArgumentError()

    return execute_tool
