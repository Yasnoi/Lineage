"""Data shared by model access and graph orchestration."""

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict


class ModelResult(TypedDict):
    """Display text alongside the JSON-serializable response body."""

    text: str
    raw_response: dict[str, Any]


type ModelCall = Callable[[str], Awaitable[ModelResult]]
