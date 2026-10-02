"""Optional preview hints on the canonical built-in text-reader schema."""

from copy import deepcopy
from typing import Any

from lineage_agent.config.settings import ConfigurationError
from lineage_agent.models.types import ToolDefinition
from lineage_agent.tools.filesystem.text import TEXT_TOOL


_HINT_SCHEMA = {
    "type": "object",
    "description": "本次读取的可选观测提示；仅用于选择大结果预览，不改变文件读取范围。",
    "properties": {
        "intent": {
            "type": ["string", "null"], "maxLength": 512,
            "description": "简短说明调用目的；无需提示时为 null，不要求提供内部推理。",
        },
        "keywords": {
            "type": "array", "maxItems": 8,
            "items": {"type": "string", "minLength": 1, "maxLength": 128},
            "description": "用于区分大小写字面匹配的关键词；无需检索时为空数组。",
        },
    },
    "required": ["intent", "keywords"],
    "additionalProperties": False,
}


def enhance_tools(tools: list[ToolDefinition]) -> tuple[list[ToolDefinition], frozenset[str]]:
    """Copy schemas; only the canonical read_text_file shape reserves hints."""
    enhanced = deepcopy(tools)
    names: set[str] = set()
    for tool in enhanced:
        if tool.get("name") != TEXT_TOOL["name"]:
            continue
        if tool.get("parameters") != TEXT_TOOL["parameters"]:
            raise ConfigurationError("read_text_file 名称保留给内置参数结构；观测提示不能覆盖自定义业务字段。")
        parameters = tool["parameters"]
        parameters["properties"]["observation"] = deepcopy(_HINT_SCHEMA)
        parameters["required"] = [*parameters["required"], "observation"]
        names.add(tool["name"])
    return enhanced, frozenset(names)


def split_observation(arguments: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Consume hints only for a registered enhanced tool; never mutate a call."""
    business = arguments.copy()
    hint = business.pop("observation", None)
    if not isinstance(hint, dict) or set(hint) != {"intent", "keywords"}:
        return business, ()
    intent, keywords = hint["intent"], hint["keywords"]
    if (intent is not None and (type(intent) is not str or len(intent) > 512)
            or type(keywords) is not list or len(keywords) > 8
            or any(type(word) is not str or not 1 <= len(word) <= 128 for word in keywords)):
        return business, ()
    try:
        if intent is not None:
            intent.encode("utf-8")
        for word in keywords:
            word.encode("utf-8")
    except UnicodeError:
        return business, ()
    return business, tuple(keywords)
