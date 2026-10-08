"""Prepare one isolated Responses request for both measurement and delivery."""

from copy import deepcopy
from typing import Any

from lineage_agent.config.settings import DEFAULT_MAX_OUTPUT_TOKENS, validate_reasoning_effort
from lineage_agent.models.types import ModelInput, ToolDefinition


def prepare_request(
    model_name: str,
    input: ModelInput,
    *,
    tools: list[ToolDefinition] | None = None,
    instructions: str | None = None,
    tool_choice: str | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Do not measure a filtered tracing projection or reconstruct after checking."""
    validate_reasoning_effort(reasoning_effort)
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
    if reasoning_effort is not None:
        request["reasoning"] = {"effort": reasoning_effort}
    return deepcopy(request)
