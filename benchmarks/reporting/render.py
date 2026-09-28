"""Render facts without interpreting project-specific content."""
import json
from pathlib import Path
from typing import Any
from benchmarks.core.contracts import validate_result, exit_code, EXECUTION_STATUSES, VERDICTS

def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def render_report(result: dict) -> str:
    if result.get("schema_version") in {"1", "2"}:
        if any(unit.get("execution_mode") == "real_model" for unit in result["units"]):
            from benchmarks.compatibility.research_v2 import render_research_report
            return render_research_report(result)
        from benchmarks.compatibility.v2 import render_report as legacy_render
        return legacy_render(result)
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
    if result["schema_version"] == "3":
        lines.extend(["### 覆盖层级", "", "评测集、隔离行为链或任务、行为组和实际 Agent 运行分别统计，不相加为测试总数。", "",
                      f"所选集合：{', '.join(result['selected_suites'])}。仅统计本次计划；其他评测集不计为通过。", ""])
        for category, counts in result["summary"].items():
            lines.extend(["```json", _json({key: value for key, value in counts.items() if key in {
                "groups", "observed_agent_runs", "harness_observations", "modules"}}), "```", ""])
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
        if unit.get("assessments"):
            lines.extend(["### 验证方式记录", "", "```json", _json(unit["assessments"]), "```", ""])
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
