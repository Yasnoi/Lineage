"""Versioned evaluation results: execution, assessment and Agent facts are separate.

Optional multi-turn records hold turn, run_id, session_id, checkpoint_id,
agent_outcome, answer_evidence and process_evidence; context can reference active_version, summary and
mechanism evidence. These fields describe facts supplied by an adapter, never
infer them from a successful answer. No research executor or scorer lives here.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict


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
            summary[category]["primary_suites"] = dict(Counter(unit["primary_suite"] for unit in selected))
    return summary


def exit_code(result: dict) -> int:
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


def validate_result(result: dict) -> None:
    """Validate the small common contract before persisting or reporting it."""
    required = {"schema_version", "run_id", "started_at", "ended_at", "selected_suites",
                "baseline", "units", "summary", "framework_errors"}
    if not required <= result.keys() or result["schema_version"] not in {"1", "2"}:
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
        if result["schema_version"] == "2" and unit["category"] == "control_logic":
            if not unit.get("groups"):
                raise ValueError("version 2 control units require a complete group plan")
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
    if any(unit.get("execution_mode") == "real_model" for unit in result["units"]):
        from benchmarks.compatibility.research_v2 import validate_research_result
        validate_research_result(result)
    # Reject non-finite numbers and non-JSON values rather than silently altering facts.
    json.dumps(result, allow_nan=False)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def render_report(result: dict) -> str:
    validate_result(result)
    lines = [
        "# Benchmark 运行报告", "", f"运行 ID：`{result['run_id']}`", "",
        f"时间：{result['started_at']} → {result['ended_at']}", "",
        f"退出码：`{exit_code(result)}`（执行状态、评测判定与 Agent 终态分别记录）", "",
        "## 基线", "", "```json", _json(result["baseline"]), "```", "",
        "工作区有修改时，提交号不能完整复现本次源码；本框架不保存源码快照。", "",
        "## 状态汇总", "",
        "| 类别 | 单元数 | completed | failed（执行） | blocked | not_run | passed | failed（判定） | pending_review |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, counts in result["summary"].items():
        numbers = [counts["total"],
                   *(counts["execution_status"][key] for key in EXECUTION_STATUSES),
                   *(counts["verdict"][key] for key in VERDICTS)]
        lines.append(f"| {category} | " + " | ".join(str(number) for number in numbers) + " |")
    lines.extend(["", "各类别分别报告，不计算能够相互抵消的总分。", ""])
    if result["schema_version"] == "2":
        lines.extend(["### 覆盖层级", "", "评测集选择、脚本单元、行为组和实际 Agent 运行分别统计，不相加为测试总数。", "",
                      f"所选集合：{', '.join(result['selected_suites'])}；未选单元：{', '.join(result.get('unselected_units', [])) or '无'}。", ""])
        for category, counts in result["summary"].items():
            lines.extend(["```json", _json({key: value for key, value in counts.items() if key in {
                "groups", "observed_agent_runs", "harness_observations", "primary_suites"}}), "```", ""])
    for unit in result["units"]:
        lines.extend([
            f"## {unit['suite_id']} / {unit['unit_id']}", "",
            f"定义版本：`{unit['definition_version']}`；粒度：`{unit['granularity']}`；执行方式：`{unit['execution_mode']}`。", "",
            f"执行：**{unit['execution_status']}**；判定：**{unit['verdict']}**。", "",
            unit["assessment_basis"], "",
        ])
        if unit["execution_mode"] == "contract_example":
            lines.extend(["契约样例：仅验证结果表达，不是实际 Research 评测。", ""])
        if unit["execution_mode"] == "deterministic_stub":
            lines.extend(["确定性替身检查；不是实际 Research 效果评测，替身 usage 不是实际模型消耗。", ""])
        for title, key in (("运行条件", "conditions"), ("Agent 终态", "agent_outcome"),
                           ("效率", "metrics"), ("分项判定", "dimensions"), ("失败及限制", "issues")):
            lines.extend([f"### {title}", "", "```json", _json(unit[key]), "```", ""])
        lines.extend(["### 证据", ""])
        for evidence in unit["evidence"]:
            location = f"`{evidence['path']}`"
            if evidence["availability"] == "available":
                location = f"[{evidence['kind']}](<{Path(evidence['path']).absolute().as_posix()}>)"
            lines.append(f"- {evidence['kind']}：{location}；{evidence['availability']}；来源：{evidence['source']}")
        lines.append("")
        if "turns" in unit:
            lines.extend(["### 轮次与过程引用", "", "```json", _json(unit["turns"]), "```", ""])
        if "groups" in unit:
            lines.extend(["### 检查组", "", "主归属和标签描述覆盖方向，不证明根本原因。组内请求/State/检查点由原断言直接核对；内存对象未作为完整副本保存。", "",
                          "| 组 ID | 行为／主要归属 | 执行 | 判定 | 定位／覆盖缺口 |", "| --- | --- | --- | --- | --- |"])
            for group in unit["groups"]:
                location = "; ".join(f"{item.get('stage', '')}:{item.get('exception_type', item.get('code', ''))}" for item in group["issues"])
                location += "; ".join(f"{gap['scenario']}: {gap['reason']} ({'阻塞' if gap['applicable'] else '不适用'})" for gap in group["coverage_gaps"])
                if group["execution_status"] == "not_run":
                    location = group["assessment_basis"]
                lines.append(f"| `{group['group_id']}` | {group['description']} / {group['primary_area']} | {group['execution_status']} | {group['verdict']} | {location or '组事件与 JSON 明细'} |")
            lines.extend(["", "组事件、输出字节范围、实际运行及检查点身份、各组证据位置见 result.json 的 groups；group_events 文件保留持久化顺序。", ""])
    lines.extend(["## 框架错误", "", "```json", _json(result["framework_errors"]), "```", ""])
    return "\n".join(lines)
