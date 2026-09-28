"""Atomic report saving with explicit partial-failure reporting."""
from pathlib import Path
import sys
from benchmarks.core.files import atomic_write_json, atomic_write_text
from benchmarks.core.contracts import summarize, validate_result, exit_code
from benchmarks.core.metadata import issue
from benchmarks.reporting.render import render_report
from benchmarks.reporting.tasks import render_research_report

def persist_result(result: dict, directory: Path) -> tuple[int, list[Path]]:
    """Best-effort preservation; a save failure never rewrites a unit's verdict."""
    saved = []
    result["summary"] = summarize(result["units"])
    validate_result(result)
    for name in ("result.json", "report.md"):
        path = directory / name
        try:
            if name == "result.json":
                atomic_write_json(path, result)
            else:
                report = (render_research_report(result) if any(u.get("turns") for u in result["units"]) else render_report(result))
                atomic_write_text(path, report)
            saved.append(path)
        except Exception as exc:
            result["framework_errors"].append(issue("saving", "artifact_write_failed", f"无法保存 {name}。", exc))
            print(f"Benchmark 保存失败：{name}（{type(exc).__name__}）；已有证据保留。", file=sys.stderr)
    if result["framework_errors"]:
        # Reconcile an earlier JSON success with a later Markdown failure.
        try:
            atomic_write_json(directory / "result.json", result)
            if directory / "result.json" not in saved:
                saved.append(directory / "result.json")
        except Exception:
            print("Benchmark 错误状态无法写回 JSON；以本次非零退出及 stderr 为准。", file=sys.stderr)
    return exit_code(result), saved