"""Additional validation and descriptive reporting for real Research samples.

This module never judges answer meaning. Mechanism observations and human
assessment are separate, and absent observations cannot manufacture a pass.
"""

from collections import Counter
from typing import Any


from benchmarks.core.tasks import STATUSES, VERDICTS, COVERAGE_STATUSES, research_units, validate_research_result


def summarize_research(units: list[dict]) -> dict:
    """Keep planned samples, turns and dimensions as separate denominators."""
    selected = [unit for unit in units if unit.get("execution_mode") == "real_model"]
    summary = {}
    for phase in sorted({unit.get("phase", "samples") for unit in selected}):
        samples = [unit for unit in selected if unit.get("phase", "samples") == phase]
        turns = [turn for unit in samples for turn in unit["turns"]]
        dimensions: dict[str, Counter] = {}
        for unit in samples:
            for item in unit["dimensions"]:
                name = item.get("id", item.get("name"))
                dimensions.setdefault(name, Counter())[item["verdict"]] += 1
        summary[phase] = {
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
        summary[phase]["metrics"] = metrics
    return summary


def research_exit_code(result: dict) -> int:
    """Preserve shared precedence, adding Research-specific coverage gates."""
    from benchmarks.core.contracts import exit_code

    if any(item["status"] == "gap" for unit in research_units(result) for item in unit.get("coverage", [])):
        return 2
    code = exit_code(result)
    if code == 0 and any(item["status"] == "pending_review" for unit in research_units(result)
                         for item in unit.get("coverage", [])):
        return 3
    return code


def render_research_report(result: dict) -> str:
    from benchmarks.reporting.render import render_report
    import json

    validate_research_result(result)
    report = render_report(result)
    lines = [report, "## Research 样本与机制", "",
             "质量、机制覆盖和消耗分别报告。未评分样本不计通过；重复次数由本次配置确定；当前项目样本不代表通用研究能力。", "",
             "```json", json.dumps(summarize_research(result["units"]), ensure_ascii=False, indent=2), "```", ""]
    if result.get("batch"):
        batch = result["batch"]
        lines.extend(["### 批次执行范围", "",
                      f"计划样本 {batch['planned_samples']}；"
                      f"计划轮次 {batch['planned_turns']}。", "",
                      f"累计实际执行时间：{batch['execution_seconds']:.3f} 秒。", "",
                      "实际 SDK 请求记账：", "", "```json",
                      json.dumps(batch["sdk_requests"], ensure_ascii=False, indent=2), "```", ""])
    for unit in research_units(result):
        lines.extend([f"### {unit['case_id']} / {unit.get('phase', 'samples')} / {unit['repeat']}", "",
                      "```json", json.dumps(unit.get("coverage", []), ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)
