"""Additional validation and descriptive reporting for real Research samples.

This module never judges answer meaning. Mechanism observations and human
assessment are separate, and absent observations cannot manufacture a pass.
"""

from collections import Counter
from typing import Any


STATUSES = {"completed", "failed", "blocked", "not_run"}
VERDICTS = {"passed", "failed", "pending_review"}
COVERAGE_STATUSES = {"verified", "gap", "pending_review"}


def research_units(result: dict) -> list[dict]:
    return [unit for unit in result["units"] if unit.get("execution_mode") == "real_model"]


def _available(value: Any, label: str) -> None:
    if not isinstance(value, dict) or value.get("availability") not in {
        "known", "unknown", "not_applicable"
    } or "value" not in value:
        raise ValueError(f"invalid Research {label} availability")
    if value["availability"] != "known" and (value["value"] is not None or not value.get("reason")):
        raise ValueError(f"invalid missing Research {label}")


def validate_research_result(result: dict) -> None:
    """Validate only real-model extensions; old contract examples stay valid.

The common validator may call this function, so it deliberately does not call
the common validator itself. File existence belongs to evidence collection or
review binding, not historical result parsing.
"""
    for unit in research_units(result):
        if result.get("schema_version") != "2" or unit.get("category") != "research":
            raise ValueError("real Research requires schema 2 research category")
        if not unit.get("case_id") or not unit.get("definition_version"):
            raise ValueError("Research case identity is missing")
        if unit.get("phase") not in {"pilot", "formal"}:
            raise ValueError("Research phase must be pilot or formal")
        if type(unit.get("repeat")) is not int or unit["repeat"] < 1:
            raise ValueError("invalid Research repetition")
        for name in ("model", "configuration", "materials", "rubric"):
            _available(unit.get("conditions", {}).get(name), name)
        turns = unit.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError("Research needs a complete planned turn list")
        for number, turn in enumerate(turns, 1):
            if turn.get("turn") != number or turn.get("execution_status") not in STATUSES:
                raise ValueError("invalid or reordered Research turn")
            for name in ("run_id", "session_id", "checkpoint_id", "agent_outcome"):
                _available(turn.get(name), name)
                if turn[name]["availability"] == "known" and not isinstance(turn[name]["value"], str):
                    raise ValueError("Research turn identity or outcome must be text")
            for field in ("answer_evidence", "process_evidence"):
                items = turn.get(field)
                if field == "answer_evidence" and isinstance(items, dict):
                    items = [items]
                if not isinstance(items, list):
                    raise ValueError("Research turn evidence must be an object or list")
                for item in items:
                    if not isinstance(item, dict) or item.get("availability") not in {
                        "available", "missing", "unknown"
                    } or not item.get("path"):
                        raise ValueError("invalid Research turn evidence")
            if turn["execution_status"] == "not_run" and turn["run_id"]["availability"] == "known":
                raise ValueError("unstarted Research turn cannot claim an Agent run")
            if turn["execution_status"] == "completed" and turn["run_id"]["availability"] != "known":
                raise ValueError("completed Research turn requires an observed Agent run")
        coverage = unit.get("coverage", [])
        if not isinstance(coverage, list):
            raise ValueError("invalid Research coverage")
        names = set()
        for item in coverage:
            if not isinstance(item, dict) or not item.get("name") or item["name"] in names:
                raise ValueError("duplicate or missing Research mechanism identity")
            names.add(item["name"])
            if item.get("status") not in COVERAGE_STATUSES or not item.get("reason"):
                raise ValueError("invalid Research mechanism observation")
        if unit.get("verdict") != "pending_review" and not unit.get("review"):
            raise ValueError("Research quality verdict needs an independent human review")
        if unit.get("verdict") == "passed":
            review = unit.get("review")
            if not isinstance(review, dict) or review.get("complete") is not True or not review.get("reviewer"):
                raise ValueError("Research pass requires a completed named human review")
            if any(item["status"] != "verified" for item in coverage):
                raise ValueError("Research pass cannot hide a mechanism gap or pending review")
            if any(turn["execution_status"] != "completed" for turn in turns):
                raise ValueError("Research pass requires all planned turns")
            if any(dimension["verdict"] != "passed" for dimension in unit["dimensions"]):
                raise ValueError("Research pass requires all applicable dimensions")


def summarize_research(units: list[dict]) -> dict:
    """Keep planned samples, turns and dimensions as separate denominators."""
    selected = [unit for unit in units if unit.get("execution_mode") == "real_model"]
    summary = {}
    for phase in ("pilot", "formal"):
        samples = [unit for unit in selected if unit["phase"] == phase]
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
                     "input_tokens", "output_tokens", "total_tokens"):
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
    from benchmarks.compatibility.v2 import exit_code

    if any(item["status"] == "gap" for unit in research_units(result) for item in unit.get("coverage", [])):
        return 2
    code = exit_code(result)
    if code == 0 and any(item["status"] == "pending_review" for unit in research_units(result)
                         for item in unit.get("coverage", [])):
        return 3
    return code


def render_research_report(result: dict) -> str:
    from benchmarks.compatibility.v2 import render_report
    import json

    validate_research_result(result)
    report = render_report(result)
    lines = [report, "## Research 样本与机制", "",
             "质量、机制覆盖和消耗分别报告。未评分样本不计通过；一次项目上的三次重复仅支持小样本观察。", "",
             "```json", json.dumps(summarize_research(result["units"]), ensure_ascii=False, indent=2), "```", ""]
    if result.get("batch"):
        batch = result["batch"]
        lines.extend(["### 批次执行范围", "",
                      f"计划样本 {batch['planned_samples']}；正式固定分母 {batch['planned_formal_samples']}；"
                      f"计划轮次 {batch['planned_turns']}。", "",
                      f"累计实际执行时间：{batch['execution_seconds']:.3f} 秒。", "",
                      "实际 SDK 请求记账：", "", "```json",
                      json.dumps(batch["sdk_requests"], ensure_ascii=False, indent=2), "```", ""])
    for unit in research_units(result):
        lines.extend([f"### {unit['case_id']} / {unit['phase']} / {unit['repeat']}", "",
                      "```json", json.dumps(unit.get("coverage", []), ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)
