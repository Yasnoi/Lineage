"""Load and validate the configuration for one agent run."""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """A required configuration value is missing or invalid."""


@dataclass(frozen=True)
class ModelSettings:
    """Model connection settings, excluding the API key."""

    base_url: str
    model_name: str
    timeout_seconds: float = 60.0


def load_settings(env_file: str | Path | None = None) -> ModelSettings:
    """Read the project .env without overriding existing environment values."""
    dotenv_path = Path(env_file) if env_file is not None else Path.cwd() / ".env"
    try:
        load_dotenv(dotenv_path=dotenv_path, override=False, encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError("无法读取 .env 配置文件。") from exc

    base_url = os.environ.get("BASE_URL", "").strip()
    model_name = os.environ.get("MODEL_NAME", "").strip()
    if not base_url:
        raise ConfigurationError("请在 .env 或环境变量中设置 BASE_URL。")
    if not model_name:
        raise ConfigurationError("请在 .env 或环境变量中设置 MODEL_NAME。")

    try:
        parsed_url = urlsplit(base_url)
        valid_url = (
            parsed_url.scheme in {"http", "https"}
            and bool(parsed_url.hostname)
            and not any(character.isspace() for character in base_url)
        )
        # Accessing port also rejects malformed or out-of-range port values.
        parsed_url.port
    except ValueError:
        valid_url = False
    if not valid_url:
        raise ConfigurationError("BASE_URL 必须是包含有效主机名的 HTTP 或 HTTPS 地址。")

    try:
        timeout_seconds = float(os.environ.get("TIMEOUT_SECONDS", "60"))
    except ValueError as exc:
        raise ConfigurationError("TIMEOUT_SECONDS 必须是有限正数。") from exc
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ConfigurationError("TIMEOUT_SECONDS 必须是有限正数。")

    return ModelSettings(
        base_url=base_url,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
    )
