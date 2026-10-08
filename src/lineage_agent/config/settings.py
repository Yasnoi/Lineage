"""Load and validate the configuration for one agent run."""

import math
import os
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_MODEL_CALLS = 30
DEFAULT_DIRECTORY_MAX_ENTRIES = 1000
DEFAULT_TEXT_MAX_BYTES = 131072
DEFAULT_TEXT_TOTAL_MAX_BYTES = 1048576
DEFAULT_CONTEXT_BUDGET_TOKENS = 250000
DEFAULT_MAX_OUTPUT_TOKENS = 16384
DEFAULT_SUMMARY_HISTORY_MAX_OUTPUT_TOKENS = 16384
DEFAULT_SUMMARY_RUN_MAX_OUTPUT_TOKENS = 16384
DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS = 8192
REASONING_EFFORTS = ("none", "low", "high", "max")


def validate_reasoning_effort(value: str | None) -> None:
    if value is not None and (type(value) is not str or value not in REASONING_EFFORTS):
        raise ConfigurationError("REASONING_EFFORT 必须为 none、low、high、max，或不设置以使用端点默认值。")


class ConfigurationError(ValueError):
    """A required configuration value is missing or invalid."""


_REMOVED_OPTION = object()
_REMOVED_CONTEXT_OPTIONS = (
    "context_trigger_percent", "context_target_percent", "summary_max_output_tokens", "summary_max_bytes",
    "context_keep_recent_turns", "summary_target_percent", "context_recovery_margin_percent",
)


def _removed_configuration(names: list[str]) -> ConfigurationError:
    return ConfigurationError(
        f"{', '.join(name.upper() for name in names)} 已移除。请删除旧配置；"
        "摘要输出上限改用 SUMMARY_HISTORY_MAX_OUTPUT_TOKENS（默认 16384）"
        "和 SUMMARY_RUN_MAX_OUTPUT_TOKENS（默认 16384），"
        "保护窗口改用 CONTEXT_KEEP_RECENT_GROUPS（默认 3），"
        "上下文触发上限固定为 CONTEXT_BUDGET_TOKENS。"
    )


@dataclass(frozen=True)
class ModelSettings:
    """Model connection and local run settings, excluding the API key."""

    base_url: str
    model_name: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS
    log_dir: Path = Path("logs")
    project_root: Path | None = None
    application_root: Path = field(default_factory=Path.cwd)
    directory_max_entries: int = DEFAULT_DIRECTORY_MAX_ENTRIES
    text_max_bytes: int = DEFAULT_TEXT_MAX_BYTES
    text_total_max_bytes: int = DEFAULT_TEXT_TOTAL_MAX_BYTES
    context_budget_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    context_safety_margin_tokens: int = DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS
    opaque_fallback_tokens: int = 4096
    context_keep_recent_groups: int = 3
    summary_history_max_output_tokens: int = DEFAULT_SUMMARY_HISTORY_MAX_OUTPUT_TOKENS
    summary_run_max_output_tokens: int = DEFAULT_SUMMARY_RUN_MAX_OUTPUT_TOKENS
    reasoning_effort: str | None = None
    # Accept obsolete keywords only to give a precise migration error. InitVars
    # are not dataclass fields and never appear in asdict or saved configuration.
    context_trigger_percent: InitVar[object] = _REMOVED_OPTION
    context_target_percent: InitVar[object] = _REMOVED_OPTION
    summary_max_output_tokens: InitVar[object] = _REMOVED_OPTION
    summary_max_bytes: InitVar[object] = _REMOVED_OPTION
    context_keep_recent_turns: InitVar[object] = _REMOVED_OPTION
    summary_target_percent: InitVar[object] = _REMOVED_OPTION
    context_recovery_margin_percent: InitVar[object] = _REMOVED_OPTION

    def __post_init__(self, context_trigger_percent: object, context_target_percent: object,
                      summary_max_output_tokens: object, summary_max_bytes: object,
                      context_keep_recent_turns: object, summary_target_percent: object,
                      context_recovery_margin_percent: object) -> None:
        removed = [name for name, value in zip(_REMOVED_CONTEXT_OPTIONS, (
            context_trigger_percent, context_target_percent, summary_max_output_tokens, summary_max_bytes,
            context_keep_recent_turns, summary_target_percent, context_recovery_margin_percent,
        ), strict=True) if value is not _REMOVED_OPTION]
        if removed:
            raise _removed_configuration(removed)
        validate_reasoning_effort(self.reasoning_effort)
        try:
            if not isinstance(self.base_url, str) or not self.base_url.strip():
                raise ValueError
            base_url = self.base_url.strip()
            parsed = urlsplit(base_url)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or any(char.isspace() or ord(char) < 32 for char in base_url)):
                raise ValueError
            parsed.port
        except ValueError as exc:
            raise ConfigurationError("BASE_URL 必须是包含有效主机名的 HTTP 或 HTTPS 地址。") from exc
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ConfigurationError("MODEL_NAME 必须是非空字符串。")
        try:
            valid_timeout = (
                type(self.timeout_seconds) in (int, float)
                and math.isfinite(self.timeout_seconds) and self.timeout_seconds > 0
            )
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            raise ConfigurationError("TIMEOUT_SECONDS 必须是有限正数。")
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model_name", self.model_name.strip())
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
        if type(self.text_total_max_bytes) is not int or self.text_total_max_bytes <= 0:
            raise ConfigurationError("TEXT_TOTAL_MAX_BYTES 必须是正整数。")
        for name in (
            "context_budget_tokens", "max_output_tokens", "opaque_fallback_tokens",
            "context_keep_recent_groups", "summary_history_max_output_tokens", "summary_run_max_output_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigurationError(f"{name.upper()} 必须是正整数。")
        if (type(self.context_safety_margin_tokens) is not int
                or self.context_safety_margin_tokens < 0):
            raise ConfigurationError("CONTEXT_SAFETY_MARGIN_TOKENS 必须是非负整数。")
        if self.context_budget_tokens <= self.max_output_tokens + self.context_safety_margin_tokens:
            raise ConfigurationError("上下文额度必须大于业务输出预留与安全余量之和。")
        try:
            if isinstance(self.application_root, str) and not self.application_root.strip():
                raise ValueError("empty application root")
            object.__setattr__(self, "application_root", Path(self.application_root).resolve())
        except (OSError, TypeError, ValueError) as exc:
            raise ConfigurationError("APPLICATION_ROOT 必须是有效的非空目录路径。") from exc
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

    removed = [name for name in _REMOVED_CONTEXT_OPTIONS if name.upper() in os.environ]
    if removed:
        raise _removed_configuration(removed)

    base_url = os.environ.get("BASE_URL", "").strip()
    model_name = os.environ.get("MODEL_NAME", "").strip()
    try:
        timeout_seconds = float(os.environ.get("TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    except ValueError as exc:
        raise ConfigurationError("TIMEOUT_SECONDS 必须是有限正数。") from exc

    try:
        max_model_calls = int(os.environ.get("MAX_MODEL_CALLS", str(DEFAULT_MAX_MODEL_CALLS)))
    except ValueError as exc:
        raise ConfigurationError("MAX_MODEL_CALLS 必须是正整数。") from exc

    try:
        directory_max_entries = int(os.environ.get("DIRECTORY_MAX_ENTRIES", str(DEFAULT_DIRECTORY_MAX_ENTRIES)))
    except ValueError as exc:
        raise ConfigurationError("DIRECTORY_MAX_ENTRIES 必须是正整数。") from exc

    try:
        text_max_bytes = int(os.environ.get("TEXT_MAX_BYTES", str(DEFAULT_TEXT_MAX_BYTES)))
    except ValueError as exc:
        raise ConfigurationError("TEXT_MAX_BYTES 必须是正整数。") from exc
    try:
        text_total_max_bytes = int(os.environ.get("TEXT_TOTAL_MAX_BYTES", str(DEFAULT_TEXT_TOTAL_MAX_BYTES)))
    except ValueError as exc:
        raise ConfigurationError("TEXT_TOTAL_MAX_BYTES 必须是正整数。") from exc

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

    budget_values: dict[str, int] = {}
    for name, default in (
        ("CONTEXT_BUDGET_TOKENS", DEFAULT_CONTEXT_BUDGET_TOKENS),
        ("MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS),
        ("CONTEXT_SAFETY_MARGIN_TOKENS", DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS),
        ("OPAQUE_FALLBACK_TOKENS", 4096),
        ("CONTEXT_KEEP_RECENT_GROUPS", 3),
        ("SUMMARY_HISTORY_MAX_OUTPUT_TOKENS", DEFAULT_SUMMARY_HISTORY_MAX_OUTPUT_TOKENS),
        ("SUMMARY_RUN_MAX_OUTPUT_TOKENS", DEFAULT_SUMMARY_RUN_MAX_OUTPUT_TOKENS),
    ):
        try:
            budget_values[name.lower()] = int(os.environ.get(name, str(default)))
        except ValueError as exc:
            raise ConfigurationError(f"{name} 必须是有效整数。") from exc

    return ModelSettings(
        base_url=base_url,
        model_name=model_name,
        reasoning_effort=os.environ.get("REASONING_EFFORT"),
        timeout_seconds=timeout_seconds,
        max_model_calls=max_model_calls,
        log_dir=log_dir,
        project_root=project_root,
        application_root=dotenv_path.parent,
        directory_max_entries=directory_max_entries,
        text_max_bytes=text_max_bytes,
        text_total_max_bytes=text_total_max_bytes,
        **budget_values,
    )
