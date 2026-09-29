"""Prepare one isolated Responses request for both measurement and delivery."""

from copy import deepcopy
from typing import Any

from lineage_agent.config.settings import DEFAULT_MAX_OUTPUT_TOKENS
from lineage_agent.models.types import ModelInput, ToolDefinition


def prepare_request(
    model_name: str,
    input: ModelInput,
    *,
    tools: list[ToolDefinition] | None = None,
    instructions: str | None = None,
    tool_choice: str | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    """Do not measure a filtered tracing projection or reconstruct after checking."""
    request: dict[str, Any] = {
        "model": model_name,
        "input": input,
        "stream": False,
        "store": False,
        "truncation": "disabled",
        "max_output_tokens": max_output_tokens,
    }
    if tools:
        request["tools"] = tools
    if instructions is not None:
        request["instructions"] = instructions
    if tool_choice is not None:
        request["tool_choice"] = tool_choice
    return deepcopy(request)
