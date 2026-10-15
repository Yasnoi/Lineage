"""Explicit saved configuration and trusted tool rebinding for conversation open."""

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.models.types import ToolDefinition
from lineage_agent.tools.filesystem.directory import (
    DIRECTORY_TOOL, DirectoryExecutor, create_directory_executor,
)
from lineage_agent.tools.filesystem.executor import FilesystemExecutor, create_filesystem_executor
from lineage_agent.tools.filesystem.paths import bind_project_root
from lineage_agent.tools.filesystem.text import TEXT_TOOL, TextExecutor, create_text_executor
from lineage_agent.tools.types import BUILTIN_RECOVERY_POLICIES, ToolExecutor, ToolRecoveryBinding


_SETTINGS = (
    "base_url", "model_name", "timeout_seconds", "max_model_calls",
    "directory_max_entries", "text_max_bytes", "text_total_max_bytes",
    "context_budget_tokens", "max_output_tokens", "context_safety_margin_tokens",
    "opaque_fallback_tokens", "context_keep_recent_groups",
    "summary_history_max_output_tokens", "summary_run_max_output_tokens", "reasoning_effort", "project_root",
)
_BUILTINS = {tool["name"]: tool for tool in (DIRECTORY_TOOL, TEXT_TOOL)}
_OPTIONS = {
    "builtin.directory": {"root", "max_entries"},
    "builtin.text": {"root", "max_bytes", "max_total_bytes"},
    "builtin.filesystem": {"root", "max_entries", "max_bytes", "max_total_bytes"},
    "none": set(), "external": set(),
}


def _require(condition: bool, message: str = "保存的会话配置或工具绑定不兼容。") -> None:
    if not condition:
        raise ConfigurationError(message)


def _safe_endpoint(value: str) -> None:
    parsed = urlsplit(value)
    _require(parsed.username is None and parsed.password is None and "?" not in value and "#" not in value,
             "可恢复会话的 BASE_URL 不能包含用户凭据、查询参数或片段；请使用普通 API 基础地址。")


def serialize_settings(settings: ModelSettings) -> dict[str, Any]:
    """Persist a fixed whitelist; credentials and local storage paths are absent."""
    _safe_endpoint(settings.base_url)
    saved = {name: deepcopy(getattr(settings, name)) for name in _SETTINGS}
    saved["project_root"] = str(settings.project_root) if settings.project_root is not None else None
    return {"version": 1, **saved}


def restore_settings(saved: dict, *, application_root: Path,
                     log_dir: Path | None = None) -> ModelSettings:
    _require(type(saved) is dict and set(saved) == {"version", *_SETTINGS}
             and type(saved.get("version")) is int and saved["version"] == 1)
    _require(saved["project_root"] is None or (type(saved["project_root"]) is str
                                              and Path(saved["project_root"]).is_absolute()))
    settings = ModelSettings(**{name: deepcopy(saved[name]) for name in _SETTINGS},
                             application_root=application_root,
                             log_dir=log_dir if log_dir is not None else Path(application_root) / "logs")
    _safe_endpoint(settings.base_url)
    return settings


def _tool_definitions(tools) -> list[ToolDefinition]:
    _require(type(tools) is list)
    names = set()
    for tool in tools:
        _require(type(tool) is dict and tool.get("type") == "function"
                 and type(tool.get("name")) is str and bool(tool["name"].strip())
                 and type(tool.get("parameters")) is dict and tool["name"] not in names)
        names.add(tool["name"])
    try:
        json.dumps(tools, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ConfigurationError("工具定义必须能够安全保存为 JSON。") from None
    return deepcopy(tools)


def _recovery_description(binding: ToolRecoveryBinding | None, definitions: list) -> dict | None:
    if binding is None:
        return None
    _require(type(binding) is ToolRecoveryBinding, "工具恢复声明必须由应用显式提供。")
    try:
        description = binding.describe()
    except (TypeError, ValueError):
        raise ConfigurationError("工具恢复声明无效。") from None
    _require(set(description["tools"]) == {tool["name"] for tool in definitions},
             "工具恢复声明必须精确覆盖应用提供的工具。")
    return description


def _builtin_recovery(provider: str, definitions: list) -> dict:
    return ToolRecoveryBinding(provider, 1,
        {tool["name"]: BUILTIN_RECOVERY_POLICIES[tool["name"]] for tool in definitions},
        stateless=provider not in {"builtin.text", "builtin.filesystem"}).describe()


def describe_tools(tools: list[ToolDefinition] | None,
                   execute_tool: ToolExecutor | None, *,
                   recovery_binding: ToolRecoveryBinding | None = None) -> dict[str, Any]:
    """Describe actual known factories; tool names never identify an executor."""
    definitions = _tool_definitions(tools if tools is not None else [])
    if not definitions:
        _require(recovery_binding is None)
        return {"version": 2, "provider": "none", "options": {}, "tools": [], "recovery": None}
    _require(callable(execute_tool), "提供工具定义时必须同时提供工具执行器。")
    descriptor = None
    if type(execute_tool) in (DirectoryExecutor, TextExecutor, FilesystemExecutor):
        descriptor = execute_tool.describe_binding()
        if any(tool != _BUILTINS.get(tool["name"]) for tool in definitions):
            descriptor = None
        elif descriptor is not None:
            allowed = ({"list_directory"} if descriptor["provider"] == "builtin.directory" else
                       {"read_text_file"} if descriptor["provider"] == "builtin.text" else set(_BUILTINS))
            if any(tool["name"] not in allowed for tool in definitions):
                descriptor = None
    if descriptor is None:
        descriptor = {"version": 1, "provider": "external", "options": {}}
        recovery = _recovery_description(recovery_binding, definitions)
    else:
        _require(recovery_binding is None, "内置工具使用应用内置的恢复声明，不接受替换。")
        recovery = _builtin_recovery(descriptor["provider"], definitions)
    return {**deepcopy(descriptor), "version": 2, "tools": definitions, "recovery": recovery}


def restore_tools(saved: dict, *, current_project_root: Path | None,
                  tools: list[ToolDefinition] | None = None,
                  execute_tool: ToolExecutor | None = None,
                  recovery_binding: ToolRecoveryBinding | None = None) -> dict[str, Any]:
    """Recreate known bindings under current directory authority.

    An external binding is an application's explicit compatibility declaration;
    this function cannot sandbox arbitrary application-supplied Python code.
    """
    _require(type(saved) is dict and set(saved) == {"version", "provider", "options", "tools", "recovery"}
             and type(saved.get("version")) is int and saved["version"] == 2)
    provider, options = saved["provider"], saved["options"]
    _require(type(provider) is str and provider in _OPTIONS and type(options) is dict
             and set(options) == _OPTIONS[provider])
    definitions = _tool_definitions(saved["tools"])
    if provider == "external":
        _require(bool(definitions) and tools is not None and callable(execute_tool),
                 "此会话需要应用显式提供兼容的自定义工具定义和执行器。")
        _require(_tool_definitions(tools) == definitions,
                 "重新绑定的自定义工具定义与保存定义不一致。")
        _require(_recovery_description(recovery_binding, definitions) == saved["recovery"],
                 "重新绑定的工具恢复声明与保存声明不一致。")
        return {"tools": definitions, "execute_tool": execute_tool}
    _require(tools is None and execute_tool is None and recovery_binding is None,
             "内置工具按保存描述重建，不接受替换执行器。")
    if provider == "none":
        _require(not definitions and saved["recovery"] is None)
        return {"tools": [], "execute_tool": None}
    _require(bool(definitions) and all(tool == _BUILTINS.get(tool["name"]) for tool in definitions))
    allowed = ({"list_directory"} if provider == "builtin.directory" else
               {"read_text_file"} if provider == "builtin.text" else set(_BUILTINS))
    _require(all(tool["name"] in allowed for tool in definitions))
    _require(saved["recovery"] == _builtin_recovery(provider, definitions))
    _require(current_project_root is not None,
             "此会话包含项目文件工具；当前 PROJECT_ROOT 未授予目录访问权限。")
    _require(type(options["root"]) is str and Path(options["root"]).is_absolute())
    for name, value in options.items():
        if name != "root":
            _require(type(value) is int and value > 0)
    authorized = bind_project_root(current_project_root)
    root = bind_project_root(options["root"])
    _require(root.is_relative_to(authorized),
             "当前 PROJECT_ROOT 未覆盖保存的工具根目录；不会重新解释历史相对路径。")
    limits = {name: value for name, value in options.items() if name != "root"}
    factory = {"builtin.directory": create_directory_executor,
               "builtin.text": create_text_executor,
               "builtin.filesystem": create_filesystem_executor}[provider]
    return {"tools": definitions, "execute_tool": factory(root, **limits)}


def configuration_warnings(settings: ModelSettings, binding: dict, *,
                           current_limits: dict | None = None) -> list[str]:
    """Explain numeric differences without replacing saved execution settings."""
    if current_limits is None:
        return []
    _require(type(current_limits) is dict)
    effective = {name: getattr(settings, name) for name in _SETTINGS
                 if type(getattr(settings, name)) in (int, float)}
    for option, field in (("max_entries", "directory_max_entries"),
                          ("max_bytes", "text_max_bytes"), ("max_total_bytes", "text_total_max_bytes")):
        if option in binding.get("options", {}):
            effective[field] = binding["options"][option]
    warnings = []
    for name, current in current_limits.items():
        _require(name in effective and type(current) in (int, float) and math.isfinite(current)
                 and (current >= 0 if name == "context_safety_margin_tokens" else current > 0),
                 "当前额度比较参数无效。")
        if current != effective[name]:
            warnings.append(f"{name.upper()}：当前配置 {current}，此会话继续使用保存值 {effective[name]}。")
    return warnings
