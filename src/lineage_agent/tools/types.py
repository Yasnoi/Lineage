"""Callable tool interface and the result of one sequential batch."""

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict


type ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]


class ToolArgumentError(ValueError):
    """The executor rejected the arguments for a known tool."""


class ToolBatchResult(TypedDict):
    """Protocol outputs and actual executor attempts for one batch."""

    output_items: list[dict[str, Any]]
    tool_calls: int
