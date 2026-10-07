"""Call an OpenAI-compatible Responses endpoint without graph dependencies."""

import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI, APIConnectionError, APIError, APIStatusError, APITimeoutError
from openai.types.responses import Response

from lineage_agent.config.settings import DEFAULT_MAX_OUTPUT_TOKENS, ConfigurationError, ModelSettings
from lineage_agent.models.requests import prepare_request
from lineage_agent.models.types import ModelInput, ModelResult, TokenUsage, ToolDefinition

if TYPE_CHECKING:
    from lineage_agent.recording.tracing import RunTrace


MODEL_ERRORS = {
    "model_error": ("模型调用失败。", "model_request"),
    "model_timeout": ("模型请求超时，请检查连接或调整 TIMEOUT_SECONDS。", "model_request"),
    "connection_error": ("无法连接模型端点，请检查 BASE_URL 和网络连接。", "model_request"),
    "http_error": ("模型端点返回 HTTP 错误，请求失败。", "model_request"),
    "invalid_response": ("模型端点返回的响应格式无效。", "response_validation"),
    "invalid_output_role": ("模型输出消息具有不允许的角色或消息格式。", "response_validation"),
    "unexpected_tool_result": ("模型响应包含未经应用执行的工具结果。", "response_validation"),
    "invalid_call_id": ("模型工具请求缺少有效且唯一的 call_id。", "response_validation"),
    "duplicate_call_id": ("模型工具请求重复使用 call_id，无法关联结果。", "response_validation"),
    "incomplete_response": ("模型响应未正常完成。", "response_validation"),
    "empty_response": ("模型没有返回文本回答、拒答或工具请求。", "response_validation"),
}


def safe_model_error(code: object) -> tuple[str, str, str]:
    """Derive both message and stage from application-owned codes."""
    code = code if type(code) is str and code in MODEL_ERRORS else "model_error"
    message, stage = MODEL_ERRORS[code]
    return code, message, stage


class ModelError(RuntimeError):
    """A failed attempt; legacy message/stage arguments are never trusted."""

    def __init__(
        self,
        message: str = "",
        *,
        result: ModelResult | None = None,
        code: str = "model_error",
        stage: str = "model_request",
    ) -> None:
        code, message, stage = safe_model_error(code)
        super().__init__(message)
        self.result = result
        self.code = code
        self.stage = stage


def create_client(settings: ModelSettings) -> AsyncOpenAI:
    """Read the credential only when constructing the model client."""
    api_key = os.environ.get("API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError("缺少 API_KEY，请在项目根目录的 .env 中配置。")

    return AsyncOpenAI(
        base_url=settings.base_url,
        api_key=api_key,
        timeout=settings.timeout_seconds,
        max_retries=0,
    )


async def generate_response(
    client: AsyncOpenAI,
    model_name: str,
    input: ModelInput,
    *,
    tools: list[ToolDefinition] | None = None,
    tracer: "RunTrace | None" = None,
    instructions: str | None = None,
    tool_choice: str | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> ModelResult:
    """Compatibility entry point for a standalone Responses call."""
    request = prepare_request(
        model_name, input, tools=tools, instructions=instructions,
        tool_choice=tool_choice, max_output_tokens=max_output_tokens,
    )
    return await send_prepared_response(client, request, tracer=tracer)


async def send_prepared_response(
    client: AsyncOpenAI,
    request: dict[str, Any],
    *,
    tracer: "RunTrace | None" = None,
    purpose: str = "business",
    summary_metadata: dict[str, Any] | None = None,
) -> ModelResult:
    """Send the same prepared kwargs that the graph has already measured."""
    is_summary = purpose == "summary"
    # Native summary input can contain original tool bodies and derived text.
    # Keep it on the content-free summary tracing path.
    span, end = None, None
    if tracer is not None and is_summary:
        start = getattr(tracer, "summary_start", None)
        end = getattr(tracer, "summary_end", None)
        if start is not None:
            span = start(request["model"], summary_metadata)
    elif tracer is not None:
        span = tracer.model_start(
            request["model"], request["input"], request.get("tools"),
            instructions=request.get("instructions"), tool_choice=request.get("tool_choice"),
        )
        end = tracer.model_end
    try:
        result = await _request_response(client, request)
    except ModelError as exc:
        if end is not None:
            end(span, result=exc.result, error_code=exc.code, stage=exc.stage)
        raise
    except BaseException as exc:
        if end is not None:
            end(
                span, error_code="model_error" if isinstance(exc, Exception) else "model_interrupted",
                stage="model_request",
            )
        raise
    if end is not None:
        end(span, result=result)
    return result


async def _request_response(
    client: AsyncOpenAI,
    request: dict[str, Any],
) -> ModelResult:
    """Make one non-streaming request and preserve its full response body."""
    try:
        response = await client.responses.create(**request)
        return parse_response(response)
    except APITimeoutError as exc:
        raise ModelError(code="model_timeout") from exc
    except APIConnectionError as exc:
        raise ModelError(code="connection_error") from exc
    except APIStatusError as exc:
        raise ModelError(code="http_error") from exc
    except APIError as exc:
        raise ModelError() from exc
    except (AttributeError, TypeError, ValueError) as exc:
        raise ModelError(code="invalid_response") from exc


def _parse_usage(raw_usage: Any) -> TokenUsage | None:
    if not isinstance(raw_usage, dict):
        return None

    def count(name: str) -> int | None:
        value = raw_usage.get(name)
        # bool is an int subclass; neither it nor negative values are token counts.
        return value if type(value) is int and value >= 0 else None

    return {
        "input_tokens": count("input_tokens"),
        "output_tokens": count("output_tokens"),
        "total_tokens": count("total_tokens"),
    }


def parse_response(response: Response | dict[str, Any]) -> ModelResult:
    """Parse one response without deciding whether to execute tools or finish."""
    try:
        raw = (
            deepcopy(response)
            if isinstance(response, dict)
            else response.model_dump(mode="json", by_alias=True, exclude_unset=True)
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ModelError(code="invalid_response") from exc

    if not isinstance(raw, dict):
        raise ModelError(code="invalid_response")

    result: ModelResult = {
        "text": "",
        "has_refusal": False,
        "raw_response": raw,
        "output_items": [],
        "tool_calls": [],
        "usage": _parse_usage(raw.get("usage")),
    }
    output = raw.get("output")
    if not isinstance(output, list):
        raise ModelError(result=result, code="invalid_response")

    texts: list[str] = []
    call_ids: set[str] = set()
    protocol_error: str | None = None
    for item in output:
        if not isinstance(item, dict):
            protocol_error = protocol_error or "invalid_response"
            continue
        kind = item.get("type")
        if kind == "function_call_output":
            protocol_error = protocol_error or "unexpected_tool_result"
            continue
        if (
            (kind == "message" and item.get("role") != "assistant")
            or (kind != "message" and "role" in item)
            or not isinstance(kind, str)
            or kind in ("input_message", "input_text", "input_image", "input_file")
        ):
            protocol_error = protocol_error or "invalid_output_role"
            continue
        if kind == "message":
            content = item.get("content")
            if not isinstance(content, list) or any(
                not isinstance(part, dict)
                or part.get("type") not in ("output_text", "refusal")
                or not isinstance(part.get("refusal" if part.get("type") == "refusal" else "text"), str)
                for part in content
            ):
                protocol_error = protocol_error or "invalid_response"
                continue
            if item.get("status") not in (None, "completed"):
                protocol_error = protocol_error or "incomplete_response"
            for part in content:
                refusal = part["type"] == "refusal"
                result["has_refusal"] |= refusal
                texts.append(part["refusal" if refusal else "text"])
        # Preserve valid messages, reasoning and endpoint extensions. Retain
        # malformed calls as evidence, but the graph will not execute them.
        result["output_items"].append(item)
        if kind == "function_call":
            call_id, name, arguments = item.get("call_id"), item.get("name"), item.get("arguments")
            if not isinstance(call_id, str) or not call_id.strip():
                protocol_error = protocol_error or "invalid_call_id"
                continue
            if not isinstance(name, str) or not name.strip() or not isinstance(arguments, str):
                protocol_error = protocol_error or "invalid_response"
                continue
            result["tool_calls"].append({"call_id": call_id, "name": name, "arguments": arguments})
            if call_id in call_ids:
                protocol_error = protocol_error or "invalid_call_id"
            call_ids.add(call_id)
            if item.get("status") not in (None, "completed"):
                protocol_error = protocol_error or "incomplete_response"

    result["text"] = "".join(texts)
    if raw.get("status") != "completed" or raw.get("error") is not None:
        raise ModelError(result=result, code="incomplete_response")
    if protocol_error is not None:
        raise ModelError(result=result, code=protocol_error)
    if not result["text"].strip() and not result["tool_calls"]:
        raise ModelError(result=result, code="empty_response")
    return result


def compactable_group_items(items: list[dict[str, Any]], *, require_response: bool = True) -> bool:
    """Check a complete text/tool protocol group without rewriting its contents.

    Known reasoning moves only with its whole group, including opaque fields.
    Native summary input preserves those fields without decoding them.
    Unknown extensions and non-text items remain protocol barriers.
    """
    def has_encrypted(value: Any) -> bool:
        if isinstance(value, dict):
            return "encrypted_content" in value or any(has_encrypted(child) for child in value.values())
        return isinstance(value, list) and any(has_encrypted(child) for child in value)

    def text_parts(value: Any) -> bool:
        return isinstance(value, list) and all(
            isinstance(part, dict) and part.get("type") in {
                "input_text", "output_text", "refusal", "reasoning_text", "summary_text", "text",
            } and isinstance(part.get("refusal" if part.get("type") == "refusal" else "text"), str)
            and not has_encrypted(part) for part in value
        )

    if not isinstance(items, list) or not items:
        return False
    pending: set[str] = set()
    seen: set[str] = set()
    has_response = False
    for item in items:
        if not isinstance(item, dict) or item.get("status") not in (None, "completed"):
            return False
        kind = item.get("type")
        allowed_keys = {
            "message": {"type", "id", "role", "status", "content", "phase"},
            "reasoning": {"type", "id", "status", "summary", "content", "encrypted_content"},
            "function_call": {"type", "id", "status", "call_id", "name", "arguments", "caller", "namespace", "async"},
            "function_call_output": {"type", "id", "status", "call_id", "output"},
        }
        shape = "message" if kind is None and "role" in item else kind
        if shape not in allowed_keys or set(item) - allowed_keys[shape]:
            return False
        if kind == "message" or kind is None and "role" in item:
            if item.get("role") not in {"user", "assistant"} or has_encrypted(item):
                return False
            if item.get("phase") not in (None, "commentary", "final_answer"):
                return False
            content = item.get("content")
            if not isinstance(content, str) and not text_parts(content):
                return False
            has_response |= item.get("role") == "assistant"
        elif kind == "reasoning":
            if (item.get("encrypted_content") is not None and not isinstance(item["encrypted_content"], str)
                    or any(item.get(key) is not None and not text_parts(item[key]) for key in ("content", "summary"))):
                return False
        elif kind == "function_call":
            call_id = item.get("call_id")
            # SDK-compatible direct calls may include explicit absent metadata.
            # Program callers, namespaces and asynchronous dependencies need a
            # different adapter and cannot be silently reconstructed here.
            if (item.get("caller") not in (None, {"type": "direct"})
                    or item.get("namespace") is not None or item.get("async") not in (None, False)):
                return False
            if (has_encrypted(item) or not isinstance(call_id, str) or not call_id.strip()
                    or call_id in seen or not isinstance(item.get("name"), str) or not item["name"].strip()
                    or not isinstance(item.get("arguments"), str)):
                return False
            seen.add(call_id)
            pending.add(call_id)
            has_response = True
        elif kind == "function_call_output":
            call_id, output = item.get("call_id"), item.get("output")
            if (has_encrypted(item) or not isinstance(call_id, str) or call_id not in pending
                    or not isinstance(output, str) and not text_parts(output)):
                return False
            pending.remove(call_id)
        else:
            return False
    return (has_response or not require_response) and not pending
