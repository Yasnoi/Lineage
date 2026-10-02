"""Observation-aware assertions shared by explicit real-endpoint diagnostics."""

import json

from lineage_agent.artifacts.tools import ARTIFACT_TOOLS
from lineage_agent.observations.hints import enhance_tools, split_observation


def inline_result(item):
    """Require actual inline disclosure; never recover hidden data for a check."""
    envelope = json.loads(item["output"])
    assert envelope.get("ok") is True, "工具执行未成功。"
    observation = envelope.get("observation")
    assert isinstance(observation, dict), "缺少统一工具观测。"
    assert observation.get("status") == "available" and observation.get("mode") == "inline", (
        "该小型诊断样例需要完整 inline 结果，不能把 Artifact 预览当作原始结果。"
    )
    assert isinstance(observation.get("raw_result"), str)
    return observation["raw_result"]


def business_arguments(call):
    arguments = json.loads(call["arguments"])
    return split_observation(arguments)[0] if call["name"] == "read_text_file" else arguments


def runtime_tools(tools):
    """Mirror registration without changing the caller's original schemas."""
    enhanced, _ = enhance_tools([*tools, *ARTIFACT_TOOLS])
    return enhanced
