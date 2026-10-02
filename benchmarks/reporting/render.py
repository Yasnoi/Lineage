"""Human-readable results; complete identities and configuration stay in JSON."""

from pathlib import Path

from benchmarks.core.contracts import validate_result, exit_code, EXECUTION_STATUSES, VERDICTS


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _available(item: dict | None) -> str:
    if not item:
        return "未知（未记录）"
    state = item["availability"]
    if state != "known":
        label = "不适用" if state == "not_applicable" else "未知"
        return f"{label}（{item.get('reason', '未记录原因')}）"
    value = item["value"]
    if isinstance(value, (dict, list)):
        return "已记录；详见 result.json"
    if type(value) is float:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _issue_text(item: dict) -> str:
    location = "/".join(str(item[key]) for key in ("stage", "code") if item.get(key))
    message = item.get("message", "详见 result.json 的问题记录。")
    exception = f"（{item['exception_type']}）" if item.get("exception_type") else ""
    return f"{location}：{message}{exception}"


def _evidence_link(evidence: dict) -> str:
    kind = evidence.get("kind", "证据")
    path = evidence["path"]
    if evidence["availability"] == "available":
        return f"[{kind}](<{Path(path).absolute().as_posix()}>)"
    return f"{kind}：`{path}`（{evidence['availability']}）"


def _evidence_list(value) -> list[dict]:
    return [value] if isinstance(value, dict) else value


def _conditions(unit: dict) -> str:
    model = unit["conditions"].get("model")
    if model and model["availability"] == "known" and isinstance(model["value"], dict):
        details = model["value"]
        model_text = str(details.get("name", details.get("kind", "已记录")))
    else:
        model_text = _available(model)
    configuration = unit["conditions"].get("configuration", {})
    details = configuration.get("value") if configuration.get("availability") == "known" else None
    parts = [f"模型：{model_text}"]
    if isinstance(details, dict):
        limits = details.get("limits", {})
        for key, label in (("requests", "批次请求上限"), ("execution_seconds", "批次执行上限（秒）"),
                           ("sample_seconds", "实例上限（秒）")):
            if key in limits:
                parts.append(f"{label}：{limits[key]}")
        if "timeout_seconds" in details:
            parts.append(f"超时（秒）：{details['timeout_seconds']}")
        tracing = details.get("tracing")
        if isinstance(tracing, dict):
            parts.append("外部追踪：" + ("启用" if tracing.get("enabled") else "关闭"))
    return "；".join(parts) + "。完整条件与版本见 result.json。"


def _metrics(unit: dict) -> list[str]:
    lines = ["### 效率", "", "| 指标 | 观测值 |", "| --- | --- |"]
    for name, item in unit["metrics"].items():
        if item["availability"] == "known" and isinstance(item["value"], (dict, list)):
            continue
        lines.append(f"| {_cell(name)} | {_cell(_available(item))} |")
    lines.extend(["", "缺失值不补零；摘要 usage 是总量的子集，不再次相加。", ""])
    return lines


def _turns(unit: dict) -> list[str]:
    lines = ["### 轮次", "", "| 轮次 | 执行 | Agent 终态 | 输入、回答与过程证据 |",
             "| ---: | --- | --- | --- |"]
    for turn in unit["turns"]:
        evidence = ([turn["prompt_evidence"]] if turn.get("prompt_evidence") else [])
        evidence += _evidence_list(turn["answer_evidence"]) + _evidence_list(turn["process_evidence"])
        links = "；".join(_evidence_link(item) for item in evidence) or "尚未取得证据"
        lines.append(f"| {turn['turn']} | {turn['execution_status']} | "
                     f"{_cell(_available(turn['agent_outcome']))} | {_cell(links)} |")
    lines.extend(["", "运行、会话、检查点及摘要版本的完整关联见 result.json 的 turns。", ""])
    return lines


def _groups(unit: dict) -> list[str]:
    lines = ["### 检查组", "", "主归属和标签描述覆盖方向，不证明根本原因。", "",
             "| 组 ID | 行为／主要归属 | 执行 | 判定 | 定位／覆盖缺口 |",
             "| --- | --- | --- | --- | --- |"]
    for group in unit["groups"]:
        notes = [_issue_text(item) for item in group["issues"]]
        notes.extend(f"{gap['scenario']}：{gap['reason']}（{'阻塞' if gap['applicable'] else '不适用'}）"
                     for gap in group["coverage_gaps"])
        if group["execution_status"] == "not_run":
            notes.append(group["assessment_basis"])
        location = "；".join(notes) or "—"
        lines.append(f"| `{group['group_id']}` | {_cell(group['description'])} / {_cell(group['primary_area'])} | "
                     f"{group['execution_status']} | {group['verdict']} | {_cell(location)} |")
    lines.extend(["", "各组证据、输出范围和运行身份见 result.json 的 groups；group_events 保留记录顺序。", ""])
    return lines


def _problems(result: dict) -> list[str]:
    lines = [f"- 框架：{_issue_text(item)}" for item in result["framework_errors"]]
    for unit in result["units"]:
        owner = unit["unit_id"]
        if unit["execution_status"] != "completed" or unit["verdict"] != "passed":
            lines.append(f"- {owner}：{unit['execution_status']} / {unit['verdict']}。{unit['assessment_basis']}")
        lines.extend(f"- {owner}：{_issue_text(item)}" for item in unit["issues"])
        for group in unit.get("groups", []):
            group_owner = f"{owner} / {group['group_id']}"
            lines.extend(f"- {group_owner}：{_issue_text(item)}" for item in group["issues"])
            if not group["issues"] and (group["execution_status"] in {"failed", "blocked"} or group["verdict"] == "failed"):
                lines.append(f"- {group_owner}：{group['execution_status']} / {group['verdict']}。{group['assessment_basis']}")
            lines.extend(f"- {group_owner}：适用覆盖缺口 {gap['scenario']}。{gap['reason']}"
                         for gap in group["coverage_gaps"] if gap["applicable"])
        lines.extend(f"- {owner}：机制覆盖缺口 {item['name']}。{item['reason']}"
                     for item in unit.get("coverage", []) if item["status"] == "gap")
    return lines or ["没有失败、阻塞、待评审或适用覆盖缺口记录。"]


def render_report(result: dict, *, additional_sections: list[str] | None = None) -> str:
    validate_result(result)
    lines = [
        "# Benchmark 运行报告", "", f"运行 ID：`{result['run_id']}`", "",
        f"时间：{result['started_at']} → {result['ended_at']}", "",
        f"评测结果退出码：`{exit_code(result)}`。执行状态、评测判定与 Agent 终态分别记录。", "",
        "完整配置、源码指纹、身份关联及原始判定见 [result.json](result.json)。", "",
        "## 状态汇总", "",
        "| 类别 | 单元数 | completed | failed（执行） | blocked | not_run | passed | failed（判定） | pending_review |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, counts in result["summary"].items():
        numbers = [counts["total"], *(counts["execution_status"][key] for key in EXECUTION_STATUSES),
                   *(counts["verdict"][key] for key in VERDICTS)]
        lines.append(f"| {_cell(category)} | " + " | ".join(str(number) for number in numbers) + " |")
    lines.extend(["", f"所选集合：{', '.join(result['selected_suites'])}。仅统计本次计划；其他范围不计为通过。", "",
                  "各类别分别报告，不计算能够相互抵消的总分。行为链、任务、组和 Agent 运行不相加为测试总数。", ""])
    for category, counts in result["summary"].items():
        if "groups" in counts:
            groups = counts["groups"]
            lines.extend([f"{category}：计划 {groups['planned']} 组，开始 {groups['started']} 组，"
                          f"结束 {groups['finished']} 组；适用覆盖缺口 {groups['applicable_coverage_gaps']} 项。"
                          f"已观察 Agent 运行 {counts['observed_agent_runs']['count']} 次（记录可能不完整）。", ""])
    lines.extend(["## 失败与限制", ""])
    lines.extend(_problems(result))
    lines.append("")
    for unit in result["units"]:
        lines.extend([
            f"## {unit['suite_id']} / {unit['unit_id']}", "",
            f"执行：**{unit['execution_status']}**；判定：**{unit['verdict']}**；"
            f"Agent 终态：{_available(unit['agent_outcome'])}。", "", unit["assessment_basis"], "",
            f"定义版本：`{unit['definition_version']}`；粒度：`{unit['granularity']}`；执行方式：`{unit['execution_mode']}`。", "",
            _conditions(unit), "",
        ])
        if unit["execution_mode"] == "contract_example":
            lines.extend(["契约样例：仅验证结果表达，不是实际 Research 评测。", ""])
        if unit["execution_mode"] == "deterministic_stub":
            lines.extend(["确定性替身检查；不是实际 Research 效果评测，替身 usage 不是实际模型消耗。", ""])
        if unit["metrics"]:
            lines.extend(_metrics(unit))
        if unit["dimensions"]:
            lines.extend(["### 分项判定", "", "| 维度 | 判定 | 未通过／待评依据 |", "| --- | --- | --- |"])
            for item in unit["dimensions"]:
                reasons = "；".join(f"{condition['id']}：{condition.get('basis') or '待评审'}"
                                    for condition in item.get("conditions", []) if condition["verdict"] != "passed")
                lines.append(f"| {_cell(item['id'])} | {item['verdict']} | {_cell(reasons or '—')} |")
            for item in unit.get("review", {}).get("severe_errors", []):
                if item.get("present") is True:
                    lines.append(f"\n严重错误 `{item['id']}`：{item['basis']}")
            lines.append("")
        if unit.get("assessments"):
            lines.extend(["验证方式：" + "；".join(f"{item['evaluator_id']}：{item['verdict']}"
                                                  for item in unit["assessments"]) + "。", ""])
        if unit["evidence"]:
            lines.extend(["### 证据", ""])
            lines.extend(f"- {_evidence_link(item)}；{item['availability']}；来源：{item['source']}"
                         for item in unit["evidence"])
            lines.append("")
        if "turns" in unit:
            lines.extend(_turns(unit))
        if "groups" in unit:
            lines.extend(_groups(unit))
    if additional_sections:
        lines.extend(additional_sections)
    baseline = result["baseline"]
    lines.extend(["## 基线", "", "| 项目 | 值 |", "| --- | --- |"])
    for key, label in (("agent_commit", "Agent 提交"), ("working_tree_dirty", "工作区有修改"),
                       ("python_version", "Python")):
        if key in baseline:
            lines.append(f"| {label} | {_cell(_available(baseline[key]))} |")
    for name, version in baseline.get("dependencies", {}).items():
        lines.append(f"| {_cell(name)} | {_cell(_available(version))} |")
    lines.extend(["", f"源码指纹共 {len(baseline.get('framework_sources', {}))} 项，与锁文件摘要统一保存在 result.json 的 baseline。", "",
                  "工作区有修改时，提交号不能完整复现本次源码；本框架不保存源码快照。", ""])
    return "\n".join(lines)
