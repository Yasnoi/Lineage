"""Callable tool interface and the result of one sequential batch."""

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict


type ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]


TOOL_ARGUMENT_REASONS = {
    "argument_validation_failed": "工具参数未通过执行器校验。",
    "invalid_argument_shape": "list_directory 仅接受一个字符串字段 path。",
    "empty_path": "path 必须是非空项目内相对路径。",
    "non_relative_path": "path 必须是项目内相对路径。",
    "disallowed_path_component": "path 包含不允许的路径组件。",
    "link_or_reparse_forbidden": "不允许通过链接或重解析点访问项目目录。",
    "outside_project_root": "path 超出本次绑定的项目目录。",
    "target_not_directory": "path 的目标不是目录。",
}


def safe_tool_argument_reason(value: object) -> tuple[str, str]:
    """Accept only application-owned diagnostic codes, never exception text."""
    code = (
        value if type(value) is str and value in TOOL_ARGUMENT_REASONS
        else "argument_validation_failed"
    )
    return code, TOOL_ARGUMENT_REASONS[code]


class ToolArgumentError(ValueError):
    """The executor rejected the arguments for a known tool."""

    def __init__(self, message: str = "", *, reason_code: str = "argument_validation_failed") -> None:
        super().__init__(message)
        self.reason_code, _ = safe_tool_argument_reason(reason_code)


class ToolOutcomes(TypedDict):
    """Request dispositions, distinct from attempts to enter the executor."""

    succeeded: int
    rejected: int
    failed: int
    skipped: int


class ToolBatchResult(TypedDict):
    """Protocol outputs and actual executor attempts for one batch."""

    output_items: list[dict[str, Any]]
    tool_calls: int
    tool_outcomes: ToolOutcomes
    error: str | None
