"""Explicit, credential-free run configuration; no project-specific profiles."""
from dataclasses import fields
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit
from unittest.mock import patch
from lineage_agent.config.settings import ModelSettings, load_settings
from benchmarks.core.diagnostics import ConfigurationError

MODEL_KEYS = {item.name for item in fields(ModelSettings)} - {"project_root", "log_dir", "application_root"}
RETIRED_COMPACT_KEYS = {"context_trigger_percent", "context_target_percent", "summary_max_output_tokens",
                        "summary_max_bytes", "context_keep_recent_turns", "summary_target_percent",
                        "context_recovery_margin_percent"}
CONNECTION_KEYS = {"API_KEY", "BASE_URL", "MODEL_NAME"}
TRACE_ENV_KEYS = {"LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT",
                  "LANGSMITH_PROJECT", "LANGSMITH_WORKSPACE_ID"}


def configuration_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

def _fields(values, allowed, path):
    if not isinstance(values, dict):
        raise ConfigurationError(path, "object_required", "必须是 JSON 对象。", "使用配置样例中的对象结构。")
    if set(values) - allowed:
        raise ConfigurationError(path, "unknown_fields", "包含不支持的字段。",
                                 "只保留以下字段：" + ", ".join(sorted(allowed)) + "。")


def _retired_compact(values, path):
    if isinstance(values, dict) and set(values) & RETIRED_COMPACT_KEYS:
        name = sorted(set(values) & RETIRED_COMPACT_KEYS)[0]
        raise ConfigurationError(f"{path}.{name}", "retired_compact_setting", "此旧 Compact 配置已移除。",
                                 "删除旧字段；当前组保护使用 context_keep_recent_groups，H/R 输出上限分别使用 summary_history_max_output_tokens 和 summary_run_max_output_tokens。")


def _settings_error(exc, path):
    # Match trusted production error categories, but never publish exception text.
    if path == "project_env" and "无法读取 .env" in str(exc):
        return ConfigurationError("project_env", "file_unreadable", "项目 .env 必须是可读取的 UTF-8 文件。",
                                  "检查项目根目录 .env 的读取权限与编码。")
    names = sorted(MODEL_KEYS | RETIRED_COMPACT_KEYS | {"log_dir", "project_root", "application_root"})
    name = next((key for key in names if key.upper() in str(exc)), None)
    if name is None and "上下文额度" in str(exc):
        name = "context_budget_tokens"
    field = f"{path}.{name}" if name else path
    constraint = "模型配置值必须满足生产配置的类型、范围及字段间约束。"
    if name == "base_url":
        constraint = "必须是含有效主机名的 HTTP(S) 地址，且不含凭据、查询或片段。"
    elif name == "model_name":
        constraint = "必须是非空字符串。"
    elif name == "context_budget_tokens":
        constraint = "必须是正整数，且大于 max_output_tokens 与 context_safety_margin_tokens 之和。"
    elif name in RETIRED_COMPACT_KEYS:
        constraint = "此旧 Compact 配置已移除；须删除，不能映射为新配置。"
    elif name == "context_safety_margin_tokens":
        constraint = "必须是非负整数。"
    elif name == "timeout_seconds":
        constraint = "必须是有限正数。"
    elif name in MODEL_KEYS:
        constraint = "必须是正整数。"
    hint = (f"修改进程环境或项目 .env 中的 {name.upper()}。" if path == "project_env" and name
            else "核对该字段与相关模型设置；删除已移除的配置字段。")
    return ConfigurationError(field, "invalid_model_settings", constraint, hint)


def checked_model(values, *, path="model"):
    _retired_compact(values, path)
    _fields(values, MODEL_KEYS, path)
    for key in ("base_url", "model_name"):
        if key not in values:
            raise ConfigurationError(f"{path}.{key}", "required", "必须显式提供。", "填写该模型连接字段。")
    try:
        settings = ModelSettings(**values)
    except (ValueError, TypeError, OverflowError) as exc:
        raise _settings_error(exc, path) from None
    url = urlsplit(settings.base_url)
    if url.username or url.password or url.query or url.fragment:
        raise ConfigurationError(f"{path}.base_url", "unsafe_endpoint", "地址不能包含凭据、查询参数或片段。",
                                 "使用不含密钥的服务地址；凭据放入指定环境变量。")
    return {key: getattr(settings, key) for key in sorted(MODEL_KEYS)}


def selected_units(config, available=None):
    names = config.get("units")
    if not isinstance(names, list) or not names:
        raise ConfigurationError("units", "nonempty_list_required", "必须是非空任务 ID 列表。",
                                 "用 benchmarks list <评测集> 查询可选 ID，并按所需顺序填写。")
    seen = set()
    for index, name in enumerate(names):
        path = f"units[{index}]"
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError(path, "invalid_unit_id", "任务 ID 必须是非空字符串。",
                                     "用 benchmarks list <评测集> 查询可选 ID。")
        if name in seen:
            raise ConfigurationError(path, "duplicate_unit_id", "同一任务 ID 只能出现一次。",
                                     "删除重复项；用 repeats 设置重复次数。")
        if available is not None and name not in available:
            raise ConfigurationError(path, "unknown_unit_id", "任务 ID 必须属于所选评测集。",
                                     "用 benchmarks list <评测集> 查询可选 ID 后修正此项。")
        seen.add(name)
    return names


def checked_config(config):
    if not isinstance(config, dict):
        raise ConfigurationError("configuration", "configuration_required", "必须提供运行配置对象。",
                                 "复制 run.example.json 为本地配置，通过 --config 指定。")
    if config.get("schema_version") != "2":
        raise ConfigurationError("schema_version", "unsupported_version", '仅支持运行配置版本 "2"。',
                                 "从当前 run.example.json 重新配置；旧版 model/credential_env 已移除。")
    allowed = {"schema_version", "enabled", "source_repository", "repeats", "limits",
        "agent_source", "model_overrides", "units", "gate_first_sample", "tracing",
    }
    _fields(config, allowed, "configuration")
    if config.get("enabled") is not True:
        raise ConfigurationError("enabled", "run_disabled", "必须为 true 才会运行。",
                                 "确认模型、材料和预算后，将 enabled 设为 true。")
    if not isinstance(config.get("source_repository"), str) or not config["source_repository"].strip():
        raise ConfigurationError("source_repository", "repository_required", "必须是非空仓库路径。",
                                 "填写含评测所需固定提交的本地材料仓库路径。")
    if type(config.get("repeats")) is not int or not 1 <= config["repeats"] <= 20:
        raise ConfigurationError("repeats", "out_of_range", "必须是 1 到 20 的整数。", "填写每个已选任务的重复次数。")
    limits = config.get("limits", {})
    maxima = {"requests": 400, "execution_seconds": 7200, "sample_seconds": 1200}
    _fields(limits, set(maxima), "limits")
    for key, maximum in maxima.items():
        value = limits.get(key)
        valid = (type(value) is int if key == "requests" else type(value) in {int, float})
        if not valid or not 0 < value <= maximum or not math.isfinite(value):
            kind = "整数" if key == "requests" else "有限数值"
            raise ConfigurationError(f"limits.{key}", "out_of_range", f"必须是大于 0 且不超过 {maximum} 的{kind}。",
                                     "按本次允许的预算填写，0 不表示无限制。")
    if config.get("agent_source") != "project_env":
        raise ConfigurationError("agent_source", "unsupported_source", '必须是 "project_env"。',
                                 "模型连接来自进程环境或项目 .env；个别设置使用 model_overrides。")
    overrides = config.get("model_overrides", {})
    _retired_compact(overrides, "model_overrides")
    _fields(overrides, MODEL_KEYS, "model_overrides")
    units = selected_units(config)
    gate = config.get("gate_first_sample", False)
    if type(gate) is not bool:
        raise ConfigurationError("gate_first_sample", "boolean_required", "必须是 true 或 false。", "使用 JSON 布尔值。")
    tracing = config.get("tracing", {})
    _fields(tracing, {"enabled", "unit_ids"}, "tracing")
    enabled, traced = tracing.get("enabled", False), tracing.get("unit_ids", [])
    if type(enabled) is not bool:
        raise ConfigurationError("tracing.enabled", "boolean_required", "必须是 true 或 false。", "使用 JSON 布尔值。")
    if not isinstance(traced, list) or enabled and not traced:
        raise ConfigurationError("tracing.unit_ids", "invalid_trace_selection", "必须是任务 ID 列表；开启追踪时不能为空。",
                                 "填写本次 units 中需要上传追踪的任务 ID。")
    for index, name in enumerate(traced):
        if not isinstance(name, str) or name not in units or name in traced[:index]:
            raise ConfigurationError(f"tracing.unit_ids[{index}]", "invalid_trace_selection", "必须是 units 中不重复的任务 ID。",
                                     "修正此项，追踪范围只能来自本次已选任务。")
    return {**deepcopy(config), "model_overrides": deepcopy(overrides), "gate_first_sample": gate,
            "tracing": {"enabled": enabled, "unit_ids": list(traced)}}

def load_connection(config, environment=None, *, credential_path="credential_env"):
    environment = os.environ if environment is None else environment
    name = config.get("credential_env")
    if not isinstance(name, str) or not name.isidentifier():
        raise ConfigurationError(credential_path, "invalid_environment_name", "必须是有效的环境变量名。",
                                 "只填写环境变量名；不要填写密钥。")
    key = environment.get(name, "").strip()
    if not key:
        hint = ("在进程环境或项目 .env 中设置非空的 API_KEY。" if credential_path == "environment.API_KEY" else
                "在启动命令的进程环境中设置 credential_env 指定的凭据；此读取不会自动加载 .env。")
        raise ConfigurationError(credential_path, "missing_credentials", "指定的凭据环境变量必须为非空。",
                                 hint)
    return {"API_KEY": key, "BASE_URL": config["model"]["base_url"], "MODEL_NAME": config["model"]["model_name"]}


def resolve_run_config(config, root, environment=None):
    """Freeze production-derived settings once; return secrets separately.

    The temporary environment preserves load_settings' normal process-over-file
    precedence without leaking dotenv additions into later runs or the Judge.
    """
    config = checked_config(config)
    initial = dict(os.environ if environment is None else environment)
    path = Path(root).resolve() / ".env"
    with patch.dict(os.environ, initial, clear=True):
        try:
            settings = load_settings(path)
        except (ValueError, TypeError, OSError, OverflowError) as exc:
            raise _settings_error(exc, "project_env") from None
        production = checked_model({key: getattr(settings, key) for key in MODEL_KEYS}, path="project_env")
        sources = {key: "environment" if key.upper() in initial else
                   "project_env" if key.upper() in os.environ else "default" for key in MODEL_KEYS}
        profile = checked_model({**production, **config["model_overrides"]}, path="model_overrides")
        connection = load_connection({"model": profile, "credential_env": "API_KEY"}, credential_path="environment.API_KEY")
        trace = {"enabled": config["tracing"]["enabled"], "unit_ids": config["tracing"]["unit_ids"],
                 "field_sources": {"enabled": "benchmark", "unit_ids": "benchmark"}}
        if trace["enabled"]:
            # Validate before filtering: unsupported SDK destination/disk knobs
            # must be rejected, not silently dropped to appear safe.
            from lineage_agent.recording.tracing import _read_settings
            os.environ["LANGSMITH_TRACING"] = "true"
            try:
                traced = _read_settings()
                if traced is None:
                    raise ValueError
            except (ValueError, TypeError):
                raise ConfigurationError("tracing", "invalid_trace_environment", "追踪环境配置必须有效。",
                                         "检查 LANGSMITH_API_KEY、LANGSMITH_ENDPOINT、LANGSMITH_PROJECT、LANGSMITH_WORKSPACE_ID；删除不支持的 SDK 上传/磁盘覆盖项。") from None
            trace.update(endpoint=traced.endpoint, project=traced.project, workspace_id=traced.workspace_id)
            trace["field_sources"].update({
                key: "environment" if name in initial else "project_env" if name in os.environ else "default"
                for key, name in (("endpoint", "LANGSMITH_ENDPOINT"), ("project", "LANGSMITH_PROJECT"),
                                  ("workspace_id", "LANGSMITH_WORKSPACE_ID"))
            })
            connection.update(LANGSMITH_TRACING="true", LANGSMITH_API_KEY=traced.api_key,
                              LANGSMITH_ENDPOINT=traced.endpoint, LANGSMITH_PROJECT=traced.project,
                              LANGSMITH_WORKSPACE_ID=traced.workspace_id)
    snapshot = {"agent_source": "project_env", "path": str(path), "settings": production,
                "field_sources": sources, "model_overrides": deepcopy(config["model_overrides"]),
                "tracing": deepcopy(trace)}
    final_sources = {key: "model_override" if key in config["model_overrides"] else source
                     for key, source in sources.items()}
    return {**config, "model": profile, "tracing": trace, "source_snapshot": snapshot,
            "source_snapshot_sha256": configuration_hash(snapshot), "field_sources": final_sources}, connection
