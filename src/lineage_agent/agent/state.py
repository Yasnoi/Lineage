"""Protocol history plus execution data scoped to the latest user request."""

from operator import add
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from lineage_agent.models.budget import ContextBudget, ContextRisk
from lineage_agent.models.types import ModelResult
from lineage_agent.tools.types import ToolOutcomes


class RecordReference(TypedDict):
    run_id: str
    checkpoint_id: str


class TurnReference(RecordReference):
    item_count: int


class SummaryReference(TypedDict):
    summary_id: str
    checkpoint_id: str


class ActiveContext(TypedDict):
    version_id: str
    parent_version_id: str | None
    summary: SummaryReference | None
    turns: list[TurnReference]
    turn_start: int


class CompactionRequestOptions(TypedDict):
    instructions: str
    tool_choice: Literal["none"] | None


class CompactionSources(TypedDict):
    previous_summary: SummaryReference | None
    added_turns: list[TurnReference]


CompactionStatus = Literal[
    "preparing", "summary_budget_rejected", "candidate", "pending_save",
    "confirmed", "used", "stopped", "failed",
]


class CompactionState(TypedDict):
    """One recovery's cumulative evidence, including evidence retained on failure.

    An early stop can precede creation of a summary ID. Later fields become
    available at the milestones below and survive a subsequent stop/failure.
    These annotations do not validate checkpoints or acknowledge their saving.
    """

    request_no: int
    status: CompactionStatus
    reason: str | None

    # Preparing an actual attempt.
    summary_id: NotRequired[str]
    old_version_id: NotRequired[str]
    request_options: NotRequired[CompactionRequestOptions]
    step_id: NotRequired[str | None]
    # Source selection and summary request checking.
    sources: NotRequired[CompactionSources]
    allowed_source_ids: NotRequired[list[str]]
    removed_items: NotRequired[int]
    removed_turns: NotRequired[int]
    reserved_candidate_tokens: NotRequired[int]
    summary_budget: NotRequired[ContextBudget]
    summary_call: NotRequired[int]
    cause_code: NotRequired[str]
    # Validated candidate; the raw model response remains in summary_result.
    document: NotRequired[dict[str, list[dict[str, Any]]]]
    message: NotRequired[dict[str, Any]]
    summary_bytes: NotRequired[int]
    candidate_version_id: NotRequired[str]
    candidate_budget: NotRequired[ContextBudget]
    # pending_save refers to the confirmed candidate, not a confirmed activation.
    summary_checkpoint_id: NotRequired[str]
    # Added only after activation saving is confirmed; retained when used.
    confirmed_checkpoint_id: NotRequired[str]


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
    compaction: CompactionState | None
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
