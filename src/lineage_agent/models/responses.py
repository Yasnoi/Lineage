"""Call an OpenAI-compatible Responses endpoint without graph dependencies."""

import os
from typing import Any

from openai import AsyncOpenAI, APIConnectionError, APIError, APIStatusError, APITimeoutError
from openai.types.responses import Response

from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.models.types import ModelInput, ModelResult, TokenUsage, ToolDefinition


class ModelError(RuntimeError):
    """A failed model attempt, optionally retaining its received response."""

    def __init__(self, message: str, *, result: ModelResult | None = None) -> None:
        super().__init__(message)
        self.result = result


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
) -> ModelResult:
    """Make one non-streaming request and preserve its full response body."""
    optional: dict[str, Any] = {}
    if tools:
        optional["tools"] = tools

    try:
        response = await client.responses.create(
            model=model_name,
            input=input,
            stream=False,
            store=False,
            **optional,
        )
        return parse_response(response)
    except APITimeoutError as exc:
        raise ModelError("模型请求超时，请检查连接或调整 TIMEOUT_SECONDS。") from exc
    except APIConnectionError as exc:
        raise ModelError("无法连接模型端点，请检查 BASE_URL 和网络连接。") from exc
    except APIStatusError as exc:
        raise ModelError(f"模型端点返回 HTTP {exc.status_code}，请求失败。") from exc
    except APIError as exc:
        raise ModelError("模型请求失败，或端点返回了无法解析的响应。") from exc
    except (AttributeError, TypeError, ValueError) as exc:
        # Compatible endpoints may return malformed JSON or incomplete objects.
        raise ModelError("模型端点返回的响应格式无效，无法读取完整回答。") from exc


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
            response
            if isinstance(response, dict)
            else response.model_dump(mode="json", by_alias=True, exclude_unset=True)
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ModelError("模型端点返回的响应格式无效，无法读取完整回答。") from exc

    if not isinstance(raw, dict):
        raise ModelError("模型端点返回的响应格式无效，无法读取完整回答。")

    result: ModelResult = {
        "text": "",
        "raw_response": raw,
        "output_items": [],
        "tool_calls": [],
        "usage": _parse_usage(raw.get("usage")),
    }
    output = raw.get("output")
    if not isinstance(output, list):
        raise ModelError("模型响应缺少有效的 output 列表。", result=result)

    texts: list[str] = []
    call_ids: set[str] = set()
    protocol_error: str | None = None
    for item in output:
        if not isinstance(item, dict):
            protocol_error = "模型响应包含无效的协议项。"
            continue
        # Preserve all fields, including reasoning and endpoint-specific extensions.
        result["output_items"].append(item)
        if item.get("type") == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments")
            if (
                not isinstance(call_id, str)
                or not call_id.strip()
                or not isinstance(name, str)
                or not name.strip()
                or not isinstance(arguments, str)
            ):
                protocol_error = "模型工具请求缺少有效的 call_id、名称或参数字符串。"
                continue
            result["tool_calls"].append(
                {"call_id": call_id, "name": name, "arguments": arguments}
            )
            if call_id in call_ids:
                protocol_error = "模型工具请求包含重复的 call_id，无法关联执行结果。"
            call_ids.add(call_id)
            if item.get("status") not in (None, "completed"):
                protocol_error = "模型工具请求尚未完成，不能执行。"
        elif item.get("type") == "message":
            content = item.get("content")
            if not isinstance(content, list):
                protocol_error = "模型消息缺少有效的内容列表。"
                continue
            if item.get("status") not in (None, "completed"):
                protocol_error = "模型消息尚未完成，不能作为最终回答。"
            for part in content:
                if not isinstance(part, dict):
                    protocol_error = "模型消息包含无效的内容项。"
                elif part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str):
                        texts.append(text)
                    else:
                        protocol_error = "模型消息中的文本格式无效。"

    result["text"] = "".join(texts)
    if raw.get("status") != "completed" or raw.get("error") is not None:
        raise ModelError("模型响应未正常完成。", result=result)
    if protocol_error is not None:
        raise ModelError(protocol_error, result=result)
    if not result["text"].strip() and not result["tool_calls"]:
        raise ModelError("模型没有返回文本回答或工具请求。", result=result)
    return result
