"""State used by a single agent execution."""

from typing import NotRequired, TypedDict

from lineage_agent.models.types import ModelResult


class AgentState(TypedDict):
    """Keep the input unchanged; write result after a successful model call."""

    user_input: str
    result: NotRequired[ModelResult]
