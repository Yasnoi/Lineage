"""Suite-independent orchestration; task content belongs to suite definitions."""
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from uuid import uuid4
from benchmarks.core.contracts import summarize
from benchmarks.core.files import ROOT, atomic_write_json, utc_now
from benchmarks.core.metadata import collect_baseline, issue
from benchmarks.core.registry import load_suite, EXECUTORS
from benchmarks.reporting.storage import persist_result

def run_benchmark(suite_id, *, config=None, root=ROOT, output_root=None, registry=None):
    suite = load_suite(suite_id, registry=registry)
    executor = import_module(EXECUTORS[suite["executor"]])
    definitions = executor.plan(suite, config) if hasattr(executor, "plan") else suite["units"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + str(uuid4())
    result = {"schema_version": "3", "run_id": run_id, "started_at": utc_now(), "ended_at": None,
              "selected_suites": [suite_id], "suite_version": suite["version"], "baseline": collect_baseline(root),
              "units": [executor.make_unit(item, root) for item in definitions],
              "summary": {}, "framework_errors": [], "unselected_units": [
                  task["unit_id"] for task in suite["units"]
                  if task["unit_id"] not in {item.get("definition_unit_id", item["unit_id"]) for item in definitions}
              ]}
    directory = Path(output_root or root / "logs/benchmarks") / run_id
    try:
        directory.mkdir(parents=True, exist_ok=False)
        atomic_write_json(directory / "plan.json", result)
    except OSError as exc:
        result["framework_errors"].append(issue("saving", "run_directory_unavailable", "无法保存执行计划；没有启动评测。", exc))
        result["ended_at"] = utc_now()
        result["summary"] = summarize(result["units"])
        return result, []
    try:
        context = executor.prepare(suite, config, directory, result, root=root) if hasattr(executor, "prepare") else None
        for definition, unit in zip(definitions, result["units"], strict=True):
            if unit["execution_status"] == "blocked":
                continue
            kwargs = {"root": root}
            if context is not None:
                kwargs["context"] = context
            stop = executor.execute_unit(unit, definition, run_id, directory, **kwargs)
            atomic_write_json(directory / "progress.json", {**result, "summary": summarize(result["units"])})
            if stop:
                result["framework_errors"].append(issue("execution", "scheduling_stopped", "记录、预算、中断或收尾状态要求停止后续调度。"))
                break
    except (Exception, KeyboardInterrupt) as exc:
        result["framework_errors"].append(issue("framework", "runner_error", "执行未完成；保留已经取得的事实。", exc))
    result["ended_at"] = utc_now()
    _, saved = persist_result(result, directory)
    if any(unit.get("turns") is not None for unit in result["units"]) and directory / "result.json" in saved:
        try:
            from benchmarks.evaluators.review import make_review_packet
            saved.extend(make_review_packet(result, directory))
        except (Exception, KeyboardInterrupt) as exc:
            result["framework_errors"].append(issue("saving", "review_packet_failed", "评审包保存失败。", exc))
            _, files = persist_result(result, directory)
            saved = list(dict.fromkeys(saved + files))
    return result, saved
