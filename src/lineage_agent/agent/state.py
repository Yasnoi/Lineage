"""Protocol history plus execution data scoped to the latest user request."""

from operator import add
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from lineage_agent.models.budget import ContextBudget, ContextRisk
from lineage_agent.models.types import ModelResult
from lineage_agent.tools.types import ToolOutcomes


class RecordReference(TypedDict):
    run_id: str
    checkpoint_id: str


class TurnDescriptor(TypedDict):
    run_id: str
    item_count: int
    group_count: int
    record_end: int


class SummaryReference(TypedDict):
    summary_id: str
    checkpoint_id: str
    lane: str
    history_ref: str
    ranges: list[dict[str, Any]]


class ActiveContext(TypedDict):
    version_id: str
    parent_version_id: str | None
    history_summary: SummaryReference | None
    run_summary: SummaryReference | None
    current_run_id: str | None
    turns: list[TurnDescriptor]
    turn_start: int
    initial_input_count: int
    groups: list[dict[str, Any]]
    item_refs: list[str | None]
    summary_spans: list[dict[str, Any]]


class CompactionRequestOptions(TypedDict):
    instructions: str | None
    tool_choice: Literal["none"] | None


CompactionStatus = Literal[
    "preparing", "summary_budget_rejected", "candidate", "retry_pending", "pending_save",
    "confirmed", "used", "stopped", "failed", "history_ready", "current_ready", "validated",
]


class CompactionState(TypedDict):
    """One complete H/R attempt; original facts have independent business archives."""

    request_no: int
    attempt_no: int
    status: CompactionStatus
    reason: str | None
    request_options: CompactionRequestOptions
    pass_no: NotRequired[int]
    step_id: NotRequired[str | None]
    old_version_id: NotRequired[str]
    snapshot: NotRequired[dict[str, Any]]
    lanes: NotRequired[dict[str, dict[str, Any]]]
    removed_indices: NotRequired[list[int]]
    removed_turns: NotRequired[int]
    removed_groups: NotRequired[list[str]]
    candidate_version_id: NotRequired[str]
    original_required: NotRequired[int]
    candidate_required: NotRequired[int]
    net_reclaimed: NotRequired[int]
    remaining_budget: NotRequired[int]
    candidate_budget: NotRequired[ContextBudget]
    summary_budget: NotRequired[ContextBudget]
    summary_call: NotRequired[int]
    summary_lane: NotRequired[str]
    summary_id: NotRequired[str]
    summary_checkpoint_id: NotRequired[str]
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
    resource_view_id: str | None
    compaction: CompactionState | None
    compaction_events: list[dict[str, Any]]
    record_batch: list[dict[str, Any]]
    record_files: dict[str, dict[str, Any]]
    record_sequence: int
    record_revision: int
    record_admitted_through: int
    history_scopes: dict[str, dict[str, Any]]
    record_metrics: dict[str, int]
    execution_facts: list[dict[str, Any]]
    run_call_ids: list[str]
    summary_calls: int
    summary_result: ModelResult | None
    summary_usage_totals: UsageTotals
    run_id: str
    conversation_id: str
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
