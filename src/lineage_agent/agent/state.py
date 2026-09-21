"""State scoped to one user request, including its internal model/tool rounds."""

from operator import add
from typing import Annotated, Any, Literal, TypedDict

from lineage_agent.models.types import ModelResult


class UsageTotals(TypedDict):
    """Known token subtotals for this run only; None means no reported value."""

    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    missing_usage_calls: int


class AgentState(TypedDict):
    """Nodes append protocol items and replace the latest result and counters."""

    input: Annotated[list[dict[str, Any]], add]
    result: ModelResult | None
    model_calls: int
    tool_requests: int
    tool_calls: int
    usage_totals: UsageTotals
    stop_reason: Literal["completed", "max_model_calls", "error"] | None
    error: str | None
