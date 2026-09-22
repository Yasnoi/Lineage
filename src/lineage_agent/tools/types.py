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
    "invalid_text_argument_shape": "read_text_file 仅接受一个字符串字段 path。",
    "target_not_regular_file": "path 的目标不是普通文件。",
    "unsupported_extension": "仅支持扩展名为 .md 或 .txt 的文件。",
    "file_too_large": "文件超过本次配置的字节上限，未返回正文。",
    "non_text_content": "文件包含不支持的控制字符，未返回正文。",
}


TOOL_EXECUTION_REASONS = {
    "execution_failed": "工具执行失败。",
    "decode_error": "文件无法按严格 UTF-8 解码，未返回正文。",
    "file_not_found": "请求的文件或其中间目录不存在。",
    "permission_denied": "没有访问请求文件所需的权限。",
    "io_error": "读取文件时发生 I/O 错误，未返回正文。",
}


def safe_tool_argument_reason(value: object) -> tuple[str, str]:
    """Accept only application-owned diagnostic codes, never exception text."""
    code = (
        value if type(value) is str and value in TOOL_ARGUMENT_REASONS
        else "argument_validation_failed"
    )
    return code, TOOL_ARGUMENT_REASONS[code]


def safe_tool_execution_reason(value: object) -> tuple[str, str]:
    """Keep execution diagnostics within the application-owned vocabulary."""
    code = (
        value if type(value) is str and value in TOOL_EXECUTION_REASONS
        else "execution_failed"
    )
    return code, TOOL_EXECUTION_REASONS[code]


class ToolArgumentError(ValueError):
    """The executor rejected the arguments for a known tool."""

    def __init__(self, message: str = "", *, reason_code: str = "argument_validation_failed") -> None:
        super().__init__(message)
        self.reason_code, _ = safe_tool_argument_reason(reason_code)


class ToolExecutionError(Exception):
    """An execution failure with a safe, explicitly classified reason."""

    def __init__(self, *, reason_code: str = "execution_failed") -> None:
        self.reason_code, message = safe_tool_execution_reason(reason_code)
        super().__init__(message)


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
