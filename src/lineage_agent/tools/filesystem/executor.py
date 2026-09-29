"""Combine the two filesystem tools without changing the generic executor."""

from pathlib import Path
from typing import Any

from lineage_agent.config.settings import (
    DEFAULT_DIRECTORY_MAX_ENTRIES, DEFAULT_TEXT_MAX_BYTES, DEFAULT_TEXT_TOTAL_MAX_BYTES,
)
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL, directory_executor
from lineage_agent.tools.filesystem.paths import bind_project_root
from lineage_agent.tools.filesystem.text import TEXT_TOOL, TextExecutor
from lineage_agent.tools.types import ToolArgumentError, ToolExecutor


def create_filesystem_executor(
    root: str | Path, *, max_entries: int = DEFAULT_DIRECTORY_MAX_ENTRIES,
    max_bytes: int = DEFAULT_TEXT_MAX_BYTES, max_total_bytes: int = DEFAULT_TEXT_TOTAL_MAX_BYTES,
) -> ToolExecutor:
    """Bind both capabilities; run_agent's supplied schemas control exposure."""
    project_root = bind_project_root(root)
    return FilesystemExecutor(
        directory_executor(project_root, max_entries=max_entries),
        TextExecutor(project_root, max_bytes, max_total_bytes),
    )


class FilesystemExecutor:
    """Name dispatch with a separate text budget for each bound run."""

    def __init__(self, directory: ToolExecutor, text: TextExecutor) -> None:
        self._directory = directory
        self._text = text

    def bind_run(self) -> "FilesystemExecutor":
        return FilesystemExecutor(self._directory, self._text.bind_run())

    @property
    def read_budget(self) -> dict[str, int]:
        return self._text.read_budget

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        if name == DIRECTORY_TOOL["name"]:
            return await self._directory(name, arguments)
        if name == TEXT_TOOL["name"]:
            return await self._text(name, arguments)
        raise ToolArgumentError()
