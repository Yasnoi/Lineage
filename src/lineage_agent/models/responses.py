"""Call an OpenAI-compatible Responses endpoint without graph dependencies."""

import os

from openai import AsyncOpenAI, APIConnectionError, APIError, APIStatusError, APITimeoutError

from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.models.types import ModelResult


class ModelError(RuntimeError):
    """A model call failed or did not produce a complete text response."""


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
    user_input: str,
) -> ModelResult:
    """Make one non-streaming request and preserve its full response body."""
    try:
        response = await client.responses.create(
            model=model_name,
            input=[{"role": "user", "content": user_input}],
            stream=False,
            store=False,
        )

        if response.status != "completed" or response.error is not None:
            raise ModelError("模型响应未正常完成，未生成最终回答。")

        text = response.output_text
        if not text.strip():
            raise ModelError("模型没有返回可展示的文本回答。")

        return {
            "text": text,
            "raw_response": response.model_dump(mode="json"),
        }
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
