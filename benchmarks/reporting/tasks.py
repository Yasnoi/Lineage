"""Descriptive Research summaries; mechanism facts and human assessment stay separate."""

from collections import Counter

from benchmarks.core.contracts import require_current_result
from benchmarks.core.tasks import STATUSES, VERDICTS, COVERAGE_STATUSES, research_units, validate_research_result


def summarize_research(units: list[dict]) -> dict:
    """Keep planned samples, turns and dimensions as separate denominators."""
    samples = [unit for unit in units if unit.get("execution_mode") == "real_model"]
    turns = [turn for unit in samples for turn in unit["turns"]]
    dimensions: dict[str, Counter] = {}
    for unit in samples:
        for item in unit["dimensions"]:
            dimensions.setdefault(item["id"], Counter())[item["verdict"]] += 1
    summary = {
        "planned_samples": len(samples),
        "execution_status": {status: sum(unit["execution_status"] == status for unit in samples)
                             for status in sorted(STATUSES)},
        "verdict": {status: sum(unit["verdict"] == status for unit in samples)
                    for status in sorted(VERDICTS)},
        "reviewed_samples": sum(bool(unit.get("review", {}).get("complete")) for unit in samples),
        "planned_turns": len(turns),
        "turn_execution_status": {status: sum(turn["execution_status"] == status for turn in turns)
                                  for status in sorted(STATUSES)},
        "coverage": {status: sum(item["status"] == status for unit in samples
                                 for item in unit.get("coverage", []))
                     for status in sorted(COVERAGE_STATUSES)},
        "dimensions": {name: {"applicable_samples": sum(counts.values()),
                              **{status: counts[status] for status in sorted(VERDICTS)}}
                       for name, counts in dimensions.items()},
    }
    metrics = {}
    for name in ("duration_seconds", "sdk_requests", "business_requests", "summary_requests",
                 "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"):
        observations = [unit.get("metrics", {}).get(name, {}) for unit in samples]
        values = [item["value"] for item in observations if item.get("availability") == "known"
                  and type(item.get("value")) in (int, float)]
        missing_requests = [unit.get("metrics", {}).get(name + "_missing_requests", {}) for unit in samples]
        counts = [item["value"] for item in missing_requests if item.get("availability") == "known"
                  and type(item.get("value")) is int]
        metrics[name] = {
            "observed_sum": {"availability": "known", "value": sum(values)} if values else {
                "availability": "unknown", "value": None, "reason": "没有可取得的样本实测值。"},
            "known_samples": len(values), "unknown_samples": len(samples) - len(values),
        }
        if name.endswith("tokens"):
            metrics[name]["missing_request_fields"] = (
                {"availability": "known", "value": sum(counts)} if counts else
                {"availability": "unknown", "value": None, "reason": "没有已核对的请求 usage 缺失计数。"})
            metrics[name]["basis"] = "仅相加已观测字段；缺失不补零，业务与摘要均已包含，不再累加摘要子集。"
    summary["metrics"] = metrics
    return summary


def research_exit_code(result: dict) -> int:
    """Preserve shared precedence, adding Research-specific coverage gates."""
    from benchmarks.core.contracts import exit_code

    require_current_result(result)
    if any(item["status"] == "gap" for unit in research_units(result) for item in unit.get("coverage", [])):
        return 2
    code = exit_code(result)
    if code == 0 and any(item["status"] == "pending_review" for unit in research_units(result)
                         for item in unit.get("coverage", [])):
        return 3
    return code


def render_research_report(result: dict) -> str:
    from benchmarks.reporting.render import render_report, _available, _cell

    require_current_result(result)
    validate_research_result(result)
    summary = summarize_research(result["units"])
    lines = ["## Research 样本与机制", "",
             "质量、机制覆盖和消耗分别报告。未评分样本不计通过；重复次数由本次配置确定；当前项目样本不代表通用研究能力。", "",
             f"计划样本 {summary['planned_samples']}；已评审样本 {summary['reviewed_samples']}；"
             f"计划轮次 {summary['planned_turns']}。", "",
             "| 轮次执行 | completed | failed | blocked | not_run |", "| --- | ---: | ---: | ---: | ---: |",
             "| 数量 | " + " | ".join(str(summary["turn_execution_status"][key])
                                      for key in ("completed", "failed", "blocked", "not_run")) + " |", "",
             "| 机制覆盖 | verified | gap | pending_review |", "| --- | ---: | ---: | ---: |",
             "| 数量 | " + " | ".join(str(summary["coverage"][key])
                                      for key in ("verified", "gap", "pending_review")) + " |", ""]
    if summary["dimensions"]:
        lines.extend(["| 维度 | 适用样本 | passed | failed | pending_review |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        lines.extend(f"| {_cell(name)} | {counts['applicable_samples']} | {counts['passed']} | "
                     f"{counts['failed']} | {counts['pending_review']} |"
                     for name, counts in summary["dimensions"].items())
        lines.append("")
    lines.extend(["### 实测消耗", "", "| 指标 | 已观测合计 | 已知样本 | 未知样本 | 缺失请求字段 |",
                  "| --- | --- | ---: | ---: | --- |"])
    for name, metric in summary["metrics"].items():
        missing = _available(metric["missing_request_fields"]) if "missing_request_fields" in metric else "不适用"
        lines.append(f"| {_cell(name)} | {_cell(_available(metric['observed_sum']))} | "
                     f"{metric['known_samples']} | {metric['unknown_samples']} | {_cell(missing)} |")
    lines.extend(["", "仅相加已观测字段，缺失不补零；摘要 usage 已包含在总量中，不再次累加。", ""])
    if result.get("batch"):
        batch = result["batch"]
        lines.extend([f"批次累计执行 {batch['execution_seconds']:.3f} 秒；"
                      f"SDK 请求记账：{_available(batch['sdk_requests'])}。", ""])
    for unit in research_units(result):
        lines.extend([f"### {unit['case_id']} / 重复 {unit['repeat']}", ""])
        if unit.get("coverage"):
            lines.extend(["| 机制 | 状态 | 依据／缺口 |", "| --- | --- | --- |"])
            lines.extend(f"| {_cell(item['name'])} | {item['status']} | {_cell(item['reason'])} |"
                         for item in unit["coverage"])
            lines.append("")
        else:
            lines.extend(["本任务没有特定机制前置条件。", ""])
        local = [trace for turn in unit["turns"] for trace in turn.get("tracing", {}).get("traces", [])]
        remote = unit.get("tracing", {}).get("remote", [])
        if local or remote:
            local_counts = Counter(trace["upload_state"] for trace in local)
            remote_counts = Counter(trace["status"] for trace in remote)
            lines.extend(["| 追踪范围 | 状态 | 条数 |", "| --- | --- | ---: |"])
            for label, counts in (("本地提交", local_counts), ("远端核验", remote_counts)):
                if counts:
                    lines.extend(f"| {label} | {_cell(state)} | {count} |" for state, count in counts.items())
                else:
                    lines.append(f"| {label} | 未取得观测 | 未知 |")
            lines.append("")
        else:
            tracing = unit["conditions"].get("configuration", {}).get("value")
            tracing = tracing.get("tracing") if isinstance(tracing, dict) else None
            if isinstance(tracing, dict) and not tracing.get("enabled"):
                lines.extend(["外部追踪未启用。", ""])
    lines.extend(["追踪提交和远端核验不作为回答质量判定；关联、计数及证据位置见 result.json。", ""])
    if result.get("unselected_units"):
        lines.extend(["### 未选择范围", "", "以下任务未选择，不计为通过：", ""])
        lines.extend(f"- `{name}`" for name in result["unselected_units"])
        lines.append("")
    return render_report(result, additional_sections=lines)
