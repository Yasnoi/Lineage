"""Data shared by model access and graph orchestration."""

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict


type ModelInput = str | list[dict[str, Any]]
type ToolDefinition = dict[str, Any]


class FunctionCall(TypedDict):
    """A Responses function request, before its JSON arguments are decoded."""

    call_id: str
    name: str
    arguments: str


class TokenUsage(TypedDict):
    """Token counts reported for one model call; absent counts stay unknown."""

    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


class ModelResult(TypedDict):
    """One model call's output and usage, including replayable protocol items."""

    text: str
    raw_response: dict[str, Any]
    output_items: list[dict[str, Any]]
    tool_calls: list[FunctionCall]
    usage: TokenUsage | None


type ModelCall = Callable[[ModelInput], Awaitable[ModelResult]]
