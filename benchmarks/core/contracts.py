"""Versioned evaluation results: execution, assessment and Agent facts are separate.

Optional multi-turn records hold turn, run_id, session_id, checkpoint_id,
agent_outcome, answer_evidence and process_evidence; context can reference active_version, summary and
mechanism evidence. These fields describe facts supplied by an adapter, never
infer them from a successful answer. No research executor or scorer lives here.
"""

from collections import Counter
import json
from typing import Any, Literal, NotRequired, TypedDict

from benchmarks.core.diagnostics import ConfigurationError


EXECUTION_STATUSES = ("completed", "failed", "blocked", "not_run")
VERDICTS = ("passed", "failed", "pending_review")


class AvailableValue(TypedDict):
    availability: Literal["known", "unknown", "not_applicable"]
    value: Any
    reason: NotRequired[str]


class Evidence(TypedDict):
    kind: str
    path: str
    source: str
    availability: Literal["available", "missing", "unknown"]


class UnitResult(TypedDict):
    suite_id: str
    unit_id: str
    category: str
    definition_version: str
    granularity: str
    execution_mode: str
    conditions: dict[str, AvailableValue]
    execution_status: Literal["completed", "failed", "blocked", "not_run"]
    verdict: Literal["passed", "failed", "pending_review"]
    agent_outcome: AvailableValue
    assessment_basis: str
    dimensions: list[dict]
    issues: list[dict]
    evidence: list[Evidence]
    metrics: dict[str, AvailableValue]
    turns: NotRequired[list[dict]]
    groups: NotRequired[list[dict]]


def known(value: Any) -> AvailableValue:
    return {"availability": "known", "value": value}


def unknown(reason: str) -> AvailableValue:
    return {"availability": "unknown", "value": None, "reason": reason}


def not_applicable(reason: str) -> AvailableValue:
    return {"availability": "not_applicable", "value": None, "reason": reason}


def summarize(units: list[dict]) -> dict:
    summary = {}
    for category in sorted({unit["category"] for unit in units}):
        selected = [unit for unit in units if unit["category"] == category]
        execution = Counter(unit["execution_status"] for unit in selected)
        verdict = Counter(unit["verdict"] for unit in selected)
        summary[category] = {
            "total": len(selected),
            "execution_status": {key: execution[key] for key in EXECUTION_STATUSES},
            "verdict": {key: verdict[key] for key in VERDICTS},
        }
        if any("groups" in unit for unit in selected):
            groups = [group for unit in selected for group in unit.get("groups", [])]
            runs = {observation["identifiers"]["run_id"] for group in groups for observation in group["observations"]
                    if observation["kind"] == "agent" and observation["identifiers"].get("run_id")}
            summary[category]["groups"] = {
                "planned": len(groups), "started": sum(g["started_at"] is not None for g in groups),
                "finished": sum(g["ended_at"] is not None for g in groups),
                "execution_status": {key: sum(g["execution_status"] == key for g in groups) for key in EXECUTION_STATUSES},
                "verdict": {key: sum(g["verdict"] == key for g in groups) for key in VERDICTS},
                "applicable_coverage_gaps": sum(gap["applicable"] for g in groups for gap in g["coverage_gaps"]),
            }
            summary[category]["observed_agent_runs"] = {"count": len(runs), "completeness": "partial",
                "basis": "按显式观测到的真实 run_id 去重；无返回状态的中断等可能仅有日志引用，不声称完整运行总数。"}
            summary[category]["harness_observations"] = dict(Counter(
                observation["kind"] for group in groups for observation in group["observations"] if observation["kind"] != "agent"))
            summary[category]["modules"] = dict(Counter(unit.get("module", "unknown") for unit in selected))
    return summary


def exit_code(result: dict) -> int:
    require_current_result(result)
    units = result["units"]
    if result["framework_errors"] or not units or any(
        unit["execution_status"] != "completed" for unit in units
    ):
        return 2
    if any(unit.get("execution_mode") == "real_model" and any(
        item.get("status") == "gap" for item in unit.get("coverage", [])
    ) for unit in units):
        return 2
    groups = [group for unit in units for group in unit.get("groups", [])]
    if any(group["execution_status"] in {"failed", "blocked"} or any(gap["applicable"] for gap in group["coverage_gaps"])
           for group in groups):
        return 2
    if any(unit["verdict"] == "passed" and any(g["verdict"] != "passed" for g in unit.get("groups", [])) for unit in units):
        return 2
    if any(unit["verdict"] == "failed" for unit in units):
        return 1
    if any(unit["verdict"] == "pending_review" for unit in units):
        return 3
    return 0 if all(unit["verdict"] == "passed" for unit in units) else 2


def require_current_result(result: dict) -> None:
    if not isinstance(result, dict) or result.get("schema_version") != "3":
        raise ConfigurationError("result.schema_version", "unsupported_version",
                                 '仅支持 schema_version="3"。',
                                 "旧产物请直接查看已保存的 JSON 或 Markdown；新评审请使用当前运行结果。")


def validate_result(result: dict) -> None:
    """Validate the small common contract before persisting or reporting it."""
    require_current_result(result)
    required = {"schema_version", "run_id", "started_at", "ended_at", "selected_suites",
                "baseline", "units", "summary", "framework_errors"}
    if not required <= result.keys() or result["schema_version"] != "3":
        raise ValueError("invalid result envelope")
    identities = set()
    for unit in result["units"]:
        if not UnitResult.__required_keys__ <= unit.keys():
            raise ValueError("incomplete evaluation unit")
        identity = (unit["suite_id"], unit["unit_id"])
        if identity in identities:
            raise ValueError("duplicate evaluation unit")
        identities.add(identity)
        if unit["execution_status"] not in EXECUTION_STATUSES or unit["verdict"] not in VERDICTS:
            raise ValueError("invalid evaluation status")
        if unit["execution_status"] != "completed" and unit["verdict"] != "pending_review":
            raise ValueError("an assessment requires completed execution")
        for value in [unit["agent_outcome"], *unit["conditions"].values(), *unit["metrics"].values()]:
            if "value" not in value or value.get("availability") not in {"known", "unknown", "not_applicable"}:
                raise ValueError("invalid availability")
            if value["availability"] != "known" and (value.get("value") is not None or not value.get("reason")):
                raise ValueError("missing data must have a reason, not a fabricated value")
        for dimension in unit["dimensions"]:
            if dimension["verdict"] not in VERDICTS:
                raise ValueError("invalid dimension verdict")
        for evidence in unit["evidence"]:
            if evidence.get("availability") not in {"available", "missing", "unknown"}:
                raise ValueError("invalid evidence availability")
        if unit.get("groups") is not None:
            if not unit.get("groups"):
                raise ValueError("version 3 control units require a complete group plan")
            group_ids = set()
            for index, group in enumerate(unit["groups"], 1):
                if group["group_id"] in group_ids or group["planned_index"] != index:
                    raise ValueError("duplicate or reordered group")
                group_ids.add(group["group_id"])
                if group["execution_status"] not in EXECUTION_STATUSES or group["verdict"] not in VERDICTS:
                    raise ValueError("invalid group status")
                if group["execution_status"] != "completed" and group["verdict"] != "pending_review":
                    raise ValueError("unfinished group cannot have a verdict")
                if group["verdict"] == "passed" and (group["ended_at"] is None or any(gap["applicable"] for gap in group["coverage_gaps"])):
                    raise ValueError("false group pass")
            if unit["verdict"] == "passed" and any(g["verdict"] != "passed" for g in unit["groups"]):
                raise ValueError("false unit pass")
    if result["summary"] != summarize(result["units"]):
        raise ValueError("summary differs from unit facts")
    for unit in result["units"]:
        if unit.get("turns") is not None:
            for index, turn in enumerate(unit["turns"], 1):
                if turn.get("turn") != index or turn.get("execution_status") not in EXECUTION_STATUSES:
                    raise ValueError("invalid turn plan")
            if unit["verdict"] == "passed" and any(
                t["execution_status"] != "completed" or t.get("agent_outcome", {}).get("value") != "completed"
                for t in unit["turns"]):
                raise ValueError("unfinished task cannot pass")
        for assessment in unit.get("assessments", []):
            if assessment.get("verdict") not in VERDICTS or not assessment.get("evaluator_id"):
                raise ValueError("invalid assessment")
        if unit["verdict"] == "passed" and any(d["verdict"] != "passed" for d in unit["dimensions"]):
            raise ValueError("unfinished grading cannot pass")
    from benchmarks.core.tasks import validate_research_result
    validate_research_result(result)
    # Reject non-finite numbers and non-JSON values rather than silently altering facts.
    json.dumps(result, allow_nan=False)
