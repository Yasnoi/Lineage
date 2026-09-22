"""Callable tool interface and the result of one sequential batch."""

from collections.abc import Awaitable, Callable
from inspect import getattr_static
from typing import Any, TypedDict


type ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]


TOOL_ARGUMENT_REASONS = {
    "argument_validation_failed": "工具参数未通过执行器校验。",
    "invalid_argument_shape": "list_directory 需要字符串 path 及可空字符串 after，不允许其他字段。",
    "invalid_directory_cursor": "after 必须为空值或不含路径分隔符的非空条目名称。",
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
    "read_budget_exceeded": "文件超过本次运行剩余的成功读取额度，未返回正文；可尝试较小文件。",
    "non_text_content": "文件包含不支持的控制字符，未返回正文。",
}


TOOL_EXECUTION_REASONS = {
    "execution_failed": "工具执行失败。",
    "decode_error": "文件无法按严格 UTF-8 解码，未返回正文。",
    "file_not_found": "请求的文件、目录或中间路径不存在。",
    "permission_denied": "没有访问请求文件或目录所需的权限。",
    "io_error": "访问文件或目录时发生 I/O 错误，未返回部分结果。",
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
        self.reason_code, explanation = safe_tool_argument_reason(reason_code)
        super().__init__(message or explanation)


def read_budget_snapshot(
    executor: ToolExecutor | None,
    *,
    on_unavailable: Callable[[], None] | None = None,
) -> dict[str, int] | None:
    """Observe optional metadata; the executor still enforces its own budget."""
    try:
        try:
            snapshot = getattr(executor, "read_budget")
        except AttributeError:
            # A declared property can itself raise AttributeError. Distinguish
            # that failed getter from an executor without this optional field.
            if getattr_static(executor, "read_budget", None) is None:
                return None
            raise
        if snapshot is None:
            return None
        keys = ("limit_bytes", "used_bytes", "remaining_bytes")
        if isinstance(snapshot, dict):
            values = {key: snapshot[key] for key in keys}
            if (
                all(type(value) is int and value >= 0 for value in values.values())
                and values["used_bytes"] + values["remaining_bytes"] == values["limit_bytes"]
            ):
                return values
    except Exception:
        # Never carry caller exception text or an old numeric snapshot into
        # logs, prompts or tracing. Cancellation remains a BaseException.
        pass
    if on_unavailable is not None:
        on_unavailable()
    return None


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
