"""Protocol history plus execution data scoped to the latest user request."""

from operator import add
from typing import Annotated, Any, Literal, TypedDict

from lineage_agent.models.budget import ContextBudget, ContextRisk
from lineage_agent.agent.context import ActiveContext
from lineage_agent.models.types import ModelResult
from lineage_agent.tools.types import ToolOutcomes


class UsageTotals(TypedDict):
    """Known token subtotals for this run only; None means no reported value."""

    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    missing_usage_calls: int


class AgentState(TypedDict):
    """Nodes append protocol items and replace the latest result and counters."""

    input: Annotated[list[dict[str, Any]], add]
    history_end: int
    context_risks: list[ContextRisk | None]
    context_budget: ContextBudget | None
    active_context: ActiveContext
    compaction: dict[str, Any] | None
    summary_calls: int
    summary_result: ModelResult | None
    summary_usage_totals: UsageTotals
    run_id: str
    log_path: str
    log_complete: bool
    result: ModelResult | None
    model_calls: int
    tool_requests: int
    tool_calls: int
    tool_outcomes: ToolOutcomes
    usage_totals: UsageTotals
    stop_reason: Literal["completed", "max_model_calls", "context_budget_exceeded", "error"] | None
    error: str | None
    error_code: str | None
    cleanup_error: dict[str, str | None] | None
