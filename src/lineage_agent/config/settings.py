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
    """Model connection and local run settings, excluding the API key."""

    base_url: str
    model_name: str
    timeout_seconds: float = 60.0
    max_model_calls: int = 5
    log_dir: Path = Path("logs")
    project_root: Path | None = None
    directory_max_entries: int = 100
    text_max_bytes: int = 65536

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_model_calls, int)
            or isinstance(self.max_model_calls, bool)
            or self.max_model_calls <= 0
        ):
            raise ConfigurationError("MAX_MODEL_CALLS 必须是正整数。")
        if (
            not isinstance(self.directory_max_entries, int)
            or isinstance(self.directory_max_entries, bool)
            or self.directory_max_entries <= 0
        ):
            raise ConfigurationError("DIRECTORY_MAX_ENTRIES 必须是正整数。")
        if type(self.text_max_bytes) is not int or self.text_max_bytes <= 0:
            raise ConfigurationError("TEXT_MAX_BYTES 必须是正整数。")
        try:
            if isinstance(self.log_dir, str) and not self.log_dir.strip():
                raise ValueError("empty log directory")
            object.__setattr__(self, "log_dir", Path(self.log_dir).resolve())
        except (OSError, TypeError, ValueError) as exc:
            raise ConfigurationError("LOG_DIR 必须是有效的非空目录路径。") from exc
        try:
            project_root = self.project_root
            if isinstance(project_root, str) and not project_root.strip():
                project_root = None
            if project_root is not None:
                # Keep link components visible for the filesystem boundary checks.
                project_root = Path(project_root).absolute()
            object.__setattr__(self, "project_root", project_root)
        except (OSError, TypeError, ValueError) as exc:
            raise ConfigurationError("PROJECT_ROOT 必须是有效的目录路径。") from exc


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

    try:
        max_model_calls = int(os.environ.get("MAX_MODEL_CALLS", "5"))
    except ValueError as exc:
        raise ConfigurationError("MAX_MODEL_CALLS 必须是正整数。") from exc

    try:
        directory_max_entries = int(os.environ.get("DIRECTORY_MAX_ENTRIES", "100"))
    except ValueError as exc:
        raise ConfigurationError("DIRECTORY_MAX_ENTRIES 必须是正整数。") from exc

    try:
        text_max_bytes = int(os.environ.get("TEXT_MAX_BYTES", "65536"))
    except ValueError as exc:
        raise ConfigurationError("TEXT_MAX_BYTES 必须是正整数。") from exc

    log_dir_value = os.environ.get("LOG_DIR", "logs").strip()
    if not log_dir_value:
        raise ConfigurationError("LOG_DIR 必须是有效的非空目录路径。")
    log_dir = Path(log_dir_value)
    if not log_dir.is_absolute():
        log_dir = dotenv_path.parent / log_dir

    project_root_value = os.environ.get("PROJECT_ROOT", "").strip()
    project_root = Path(project_root_value) if project_root_value else None
    if project_root is not None and not project_root.is_absolute():
        project_root = dotenv_path.parent / project_root

    return ModelSettings(
        base_url=base_url,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
        max_model_calls=max_model_calls,
        log_dir=log_dir,
        project_root=project_root,
        directory_max_entries=directory_max_entries,
        text_max_bytes=text_max_bytes,
    )
