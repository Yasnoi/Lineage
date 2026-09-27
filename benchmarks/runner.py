"""Explicit local suite mapping and subprocess lifecycle; no plugin discovery."""

from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from uuid import UUID, uuid4

from benchmarks.common import ROOT, atomic_write_json, atomic_write_text, sha256_file, utc_now
from benchmarks.catalog import load_catalog, resolve_selection
from benchmarks.groups import planned_groups, plan_digest, read_group_events
from benchmarks.results import (
    exit_code, known, not_applicable, render_report, summarize, unknown, validate_result,
)


UNITS, SUITES = load_catalog()
DEPENDENCIES = ("langgraph", "langsmith", "openai", "python-dotenv", "urllib3")


def safe_environment(source: dict[str, str]) -> dict[str, str]:
    """Only runtime plumbing crosses into this offline, deterministic worker."""
    allowed = {"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "TEMP", "TMP", "TMPDIR",
               "COMSPEC", "SYSTEMDRIVE", "HOME", "USERPROFILE", "LOCALAPPDATA",
               "APPDATA", "VIRTUAL_ENV", "LANG", "LC_ALL"}
    env = {key: value for key, value in source.items() if key.upper() in allowed}
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               PYTHONNOUSERSITE="1", LANGSMITH_TRACING="false", LANGCHAIN_TRACING="false",
               LANGCHAIN_TRACING_V2="false")
    return env


def issue(stage: str, code: str, message: str, exc: BaseException | None = None) -> dict:
    value = {"stage": stage, "code": code, "message": message, "evidence": []}
    if exc is not None:
        value["exception_type"] = type(exc).__name__
    return value


def _digest(path: Path) -> dict:
    try:
        return known(sha256_file(path))
    except OSError:
        return unknown("文件内容摘要无法取得。")


def collect_baseline(root: Path) -> dict:
    def git(*args: str) -> dict:
        try:
            result = subprocess.run(["git", *args], cwd=root, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=5)
            if result.returncode == 0:
                return known(result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
        return unknown("Git 元数据无法取得；未保存源码快照。")

    dirty = git("status", "--porcelain", "--untracked-files=normal")
    if dirty["availability"] == "known":
        dirty = known(bool(dirty["value"]))
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = known(metadata.version(name))
        except metadata.PackageNotFoundError:
            versions[name] = unknown("当前解释器未找到已安装的分发版本。")
    return {
        "agent_commit": git("rev-parse", "HEAD"), "working_tree_dirty": dirty,
        "python_version": known(platform.python_version()), "dependencies": versions,
        "lock_sha256": _digest(root / "uv.lock"),
        "framework_sources": {name: _digest(path) for name, path in (
            ("runner", Path(__file__)), ("results", ROOT / "benchmarks/results.py"),
            ("worker", ROOT / "benchmarks/react_worker.py"),
            ("control_worker", ROOT / "benchmarks/control_worker.py"),
            ("group_events", ROOT / "benchmarks/groups.py"),
            ("check_support", ROOT / "scripts/check_support.py"),
        )},
    }


def make_unit(definition: dict, root: Path) -> dict:
    script_version = _digest(root / definition["script"])
    unit = {
        **{key: definition[key] for key in (
            "suite_id", "unit_id", "category", "definition_version", "granularity", "execution_mode",
        )},
        "conditions": {
            "model": known({"kind": "deterministic_stub", "profile": definition["unit_id"], "version": script_version}),
            "configuration": known({
                "profile": Path(definition["script"]).stem, "script_sha256": script_version,
                "settings_sha256": _digest(root / "src/lineage_agent/config/settings.py"),
                "timeout_seconds": definition["timeout_seconds"], "parent_tracing": "disabled",
                "check_tracing": "explicit synthetic environment and fake HTTP when enabled",
                "assertions": "enabled", "scope": "脚本内按检查覆写配置；版本摘要覆盖定义，不代表所有运行使用相同配置。",
            }),
            "materials": known({"kind": "inline_fixtures", "script_sha256": script_version}),
            "rubric": known({"kind": "python_assertions", "version": "1", "script_sha256": script_version}),
        },
        "execution_status": "not_run", "verdict": "pending_review",
        "agent_outcome": not_applicable("一个脚本包含多个被测运行；具体终态见原检查输出。"),
        "assessment_basis": "尚未执行，没有有效评测判定。",
        "dimensions": [], "issues": [], "evidence": [],
        "metrics": {"duration_seconds": unknown("检查尚未启动。"),
                    "usage": not_applicable("替身 usage 用于断言，不是实际模型消耗。")},
    }
    if "groups" in definition:
        unit.update(groups=planned_groups(definition), selected_by=definition.get("selected_by", []),
                    primary_suite=definition["primary_suite"])
        unit["conditions"]["definition"] = known({"version": "2", "plan_digest": plan_digest(definition["groups"])})
        # Shared helpers contain fixtures/assertions too; version them without loading them.
        unit["conditions"]["check_sources"] = known({
            str(path.relative_to(root)): _digest(path)
            for path in [root / "scripts" / (name + ".py") for name in (
                "check_react", "check_filesystem", "check_text", "check_review_fixes", "check_sessions",
                "check_session_review_fixes", "check_context_budget", "check_compaction", "check_tracing",
                "check_trace_data", "check_trace_usage", "check_text_tracing", "check_session_tracing",
                "check_compaction_tracing", "check_support",
            )]
        })
    return unit


def _evidence(kind: str, path: Path, source: str) -> dict:
    try:
        availability = "available" if path.exists() else "missing"
    except OSError:
        availability = "unknown"
    return {"kind": kind, "path": str(path.absolute()), "source": source, "availability": availability}


def read_receipt(path: Path, *, run_id: str, definition: dict, root: Path,
                 expected_sha256: str | None, returncode: int) -> dict:
    """A successful-looking stdout line is deliberately never an input here."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid receipt object")
    for key, expected in (("schema_version", "1"), ("adapter_version", "1"),
                          ("run_id", run_id), ("unit_id", definition["unit_id"])):
        if data.get(key) != expected:
            raise ValueError("receipt identity mismatch")
    if Path(data.get("script_path", "")).resolve() != (root / definition["script"]).resolve():
        raise ValueError("receipt script mismatch")
    outcome = data.get("outcome")
    expected_codes = {"passed": 0, "assertion_failed": 1, "error": 2}
    if outcome not in expected_codes or returncode != expected_codes[outcome]:
        raise ValueError("receipt and process outcome disagree")
    if data.get("phase") not in {"preflight", "import", "check_execution", "finished"}:
        raise ValueError("invalid receipt phase")
    if not data.get("started_at") or not data.get("ended_at"):
        raise ValueError("receipt not finalized")
    if outcome != "error" and (data["phase"] != "finished" or data.get("script_sha256") != expected_sha256):
        raise ValueError("check not confirmed finished at expected version")
    if data.get("script_sha256") not in {None, expected_sha256}:
        raise ValueError("script changed during evaluation")
    check_id, check_dir = data.get("check_id"), data.get("check_dir")
    if check_id is not None or check_dir is not None:
        UUID(check_id)
        if Path(check_dir).resolve() != (root / "logs/checks" / check_id).resolve():
            raise ValueError("receipt log directory mismatch")
    elif outcome != "error":
        raise ValueError("check identity missing")
    if outcome == "passed" and data.get("error") is not None:
        raise ValueError("passed receipt contains an error")
    if outcome != "passed":
        error = data.get("error")
        if (not isinstance(error, dict) or error.get("stage") not in {"preflight", "import", "check_execution"}
                or not isinstance(error.get("code"), str) or not error["code"]
                or not isinstance(error.get("exception_type"), str) or not error["exception_type"]):
            raise ValueError("failure receipt lacks a reason")
        if outcome == "assertion_failed" and error != {
            "stage": "check_execution", "code": "assertion_failed", "exception_type": "AssertionError",
        }:
            raise ValueError("assertion failure was not confirmed by the check")
    return data


class ProcessCleanupError(RuntimeError):
    """The direct child was handled, but its process tree could not be confirmed."""


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        tree_error = None
        if os.name == "nt" and type(process.pid) is int:
            # The owned process is still alive; kill its descendants before its PID disappears.
            try:
                stopped = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                         capture_output=True, timeout=5)
                if stopped.returncode == 0:
                    process.wait(timeout=5)
                    return
                tree_error = ProcessCleanupError("process tree termination was not confirmed")
            except (OSError, subprocess.SubprocessError) as exc:
                tree_error = exc
        elif os.name != "nt" and type(process.pid) is int:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                return
            except (OSError, subprocess.SubprocessError) as exc:
                tree_error = exc
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if tree_error is not None:
            raise ProcessCleanupError("process tree cleanup remains unconfirmed") from tree_error


def execute_unit(unit: dict, definition: dict, run_id: str, directory: Path,
                 *, root: Path = ROOT) -> bool:
    """Return whether interruption, recording, or uncertain cleanup stops scheduling."""
    script = root / definition["script"]
    if not script.is_file():
        unit.update(execution_status="blocked", assessment_basis="检查入口缺失，检查未启动。")
        unit["issues"].append(issue("preflight", "missing_entrypoint", "固定检查脚本不存在。"))
        return False
    unit_dir = directory / definition["unit_id"]
    unit_dir.mkdir()
    stdout_path, stderr_path = unit_dir / "stdout.txt", unit_dir / "stderr.txt"
    receipt_path = unit_dir / "receipt.json"
    worker = "benchmarks.control_worker" if "groups" in definition else "benchmarks.react_worker"
    command = [sys.executable, "-m", worker, "--run-id", run_id,
               "--unit-id", definition["unit_id"], "--receipt", str(receipt_path)]
    process = None
    interruption = False
    failure = None
    started = time.perf_counter()
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(command, cwd=root, env=safe_environment(dict(os.environ)),
                                       stdout=stdout, stderr=stderr, **({"start_new_session": True} if os.name != "nt" else {}))
            try:
                process.wait(timeout=definition["timeout_seconds"])
            except subprocess.TimeoutExpired as exc:
                failure = issue("execution", "timeout", "检查超时，已请求终止进程树；已有输出保留。", exc)
            except KeyboardInterrupt:
                interruption = True
                failure = issue("execution", "interrupted", "运行被中断，已取得证据保留。")
            finally:
                try:
                    _stop(process)
                except Exception as exc:
                    interruption = True
                    unit["issues"].append(issue("cleanup", "process_tree_unconfirmed", "进程树收尾未确认，停止后续调度。", exc))
                    if failure is None:
                        failure = issue("cleanup", "process_tree_unconfirmed", "进程树收尾未确认。", exc)
    except KeyboardInterrupt:
        interruption = True
        failure = issue("execution", "interrupted", "运行被中断，已取得证据保留。")
        if process is not None:
            try:
                _stop(process)
            except Exception as exc:
                unit["issues"].append(issue("cleanup", "process_tree_unconfirmed", "进程树收尾未确认，停止后续调度。", exc))
    except OSError as exc:
        failure = issue("launch" if process is None else "execution", "process_error",
                        "无法启动检查或保存进程输出；没有有效判定。", exc)
    except Exception as exc:
        failure = issue("launch" if process is None else "execution", "runner_error",
                        "进程管理发生异常，没有有效判定；已取得输出保留。", exc)
    finally:
        if process is not None:
            unit["metrics"]["duration_seconds"] = known(time.perf_counter() - started)
        unit["evidence"].extend([
            _evidence("stdout", stdout_path, "worker process"),
            _evidence("stderr", stderr_path, "worker process"),
            _evidence("receipt", receipt_path, "control adapter v2" if "groups" in definition else "ReAct adapter v1"),
        ])
    if "groups" in definition:
        return finish_control_unit(unit, definition, run_id, root, unit_dir, process, failure, interruption)
    # A started receipt can still locate partial logs after timeout/interruption.
    # Validate those references independently; an unfinished receipt is never a verdict.
    try:
        partial = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (partial.get("run_id") == run_id and partial.get("unit_id") == definition["unit_id"]
                and partial.get("check_id") is not None):
            check_id = str(UUID(partial["check_id"]))
            log_dir = root / "logs/checks" / check_id
            if Path(partial["check_dir"]).resolve() == log_dir.resolve():
                unit["evidence"].append(_evidence("original_check_logs", log_dir, f"check_react:{check_id}"))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    if failure is None:
        try:
            receipt = read_receipt(
                receipt_path, run_id=run_id, definition=definition, root=root,
                expected_sha256=unit["conditions"]["materials"]["value"]["script_sha256"]["value"],
                returncode=process.returncode,
            )
            if receipt["outcome"] == "passed":
                unit.update(execution_status="completed", verdict="passed",
                            assessment_basis="原 main() 完整返回；绑定本次运行的回执和进程退出码一致。")
            elif receipt["outcome"] == "assertion_failed":
                unit.update(execution_status="completed", verdict="failed",
                            assessment_basis="原检查执行中触发 AssertionError；判定为未通过，详见 stderr。")
                unit["issues"].append({**receipt["error"], "message": "原检查断言失败。", "evidence": [str(stderr_path)]})
            else:
                failure = {**receipt["error"], "message": "worker 未正常完成检查；尚无有效判定。", "evidence": [str(stderr_path)]}
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            failure = issue("adaptation", "invalid_receipt", "回执缺失、损坏、身份或退出行为不一致；不从输出补判通过。", exc)
    if failure is not None:
        unit.update(execution_status="failed", verdict="pending_review",
                    assessment_basis="执行或适配故障，无法形成有效评测判定。")
        unit["issues"].append(failure)
    return interruption


def finish_control_unit(unit, definition, run_id, root, directory, process, failure, interruption):
    events = directory / "groups.jsonl"
    receipt_path = directory / "receipt.json"
    groups, event_errors, count = read_group_events(events, run_id, definition)
    unit["groups"] = groups
    unit["evidence"].append(_evidence("group_events", events, "control adapter v2"))
    fatal_recording = bool(event_errors) and process is not None and events.exists()
    unit["issues"].extend(event_errors)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = {"schema_version": "2", "adapter_version": "2", "run_id": run_id,
                    "unit_id": definition["unit_id"], "plan_digest": plan_digest(definition["groups"])}
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("receipt identity mismatch")
        if (Path(receipt["script_path"]).resolve() != (root / definition["script"]).resolve()
                or receipt["script_sha256"] != unit["conditions"]["materials"]["value"]["script_sha256"]["value"]):
            raise ValueError("receipt source mismatch")
        if receipt["phase"] != "finished" or not receipt["ended_at"] or not receipt["started_at"]:
            raise ValueError("unfinished receipt")
        outcome = receipt["outcome"]
        codes = {"passed": 0, "assertion_failed": 1, "error": 2, "coverage_gap": 2, "recording_failed": 2, "interrupted": 2}
        if outcome not in codes or process is None or process.returncode != codes[outcome]:
            raise ValueError("receipt disagrees with process")
        if outcome in {"recording_failed", "interrupted"}:
            fatal_recording = True
        if outcome not in {"recording_failed", "error", "interrupted"}:
            if event_errors or receipt["events_count"] != count or receipt["events_sha256"] != sha256_file(events):
                raise ValueError("receipt disagrees with durable events")
        failed_groups = [g for g in groups if g["verdict"] == "failed"]
        gaps = [gap for group in groups for gap in group["coverage_gaps"] if gap["applicable"]]
        if failure is None:
            if outcome in {"passed", "coverage_gap"}:
                if (any(g["execution_status"] != "completed" for g in groups) or failed_groups
                        or (outcome == "passed" and gaps) or (outcome == "coverage_gap" and not gaps)
                        or receipt["error"] is not None):
                    raise ValueError("false script completion")
                unit.update(execution_status="completed", verdict="pending_review" if gaps else "passed",
                            assessment_basis="完整原入口与全部计划组已结束。" + ("存在环境覆盖缺口。" if gaps else "断言通过。"))
            elif outcome == "assertion_failed":
                if len(failed_groups) != 1 or not isinstance(receipt["error"], dict) or receipt["error"].get("exception_type") != "AssertionError":
                    raise ValueError("unconfirmed assertion failure")
                failure_index = groups.index(failed_groups[0])
                if (any(g["verdict"] not in {"passed", "pending_review"} or g["execution_status"] != "completed" for g in groups[:failure_index])
                        or any(g["started_at"] is not None for g in groups[failure_index + 1:])):
                    raise ValueError("check continued after a group failure")
                unit.update(execution_status="completed", verdict="failed",
                            assessment_basis="真实检查组断言失败；余组按约定未执行，其他独立单元可以继续。")
                unit["issues"].append({**receipt["error"], "group_id": failed_groups[0]["group_id"], "evidence": [str(directory / "stderr.txt")]})
            else:
                failure = {**(receipt.get("error") or {}), "message": "完整入口未正常结束，没有有效脚本级判定。"}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        if failure is None:
            failure = issue("adaptation", "invalid_receipt", "最终回执缺失或与实际组事件不一致。", exc)
    if event_errors and failure is None:
        failure = issue("group_recording", "invalid_event_stream", "组记录不能确认为完整有效。")
    if failure is not None:
        unit.update(execution_status="failed", verdict="pending_review", assessment_basis="执行、组记录或适配故障；此前已确认组保留。")
        unit["issues"].append(failure)
    return interruption or fatal_recording


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
                if any(unit.get("execution_mode") == "real_model" for unit in result["units"]):
                    from benchmarks.research.contracts import render_research_report
                    report = render_research_report(result)
                else:
                    report = render_report(result)
                atomic_write_text(path, report)
            saved.append(path)
        except (OSError, ValueError, TypeError) as exc:
            result["framework_errors"].append(issue("saving", "artifact_write_failed", f"无法保存 {name}。", exc))
            print(f"Benchmark 保存失败：{name}（{type(exc).__name__}）；已有证据保留。", file=sys.stderr)
    if result["framework_errors"]:
        # Reconcile an earlier JSON success with a later Markdown failure.
        try:
            atomic_write_json(directory / "result.json", result)
            if directory / "result.json" not in saved:
                saved.append(directory / "result.json")
        except (OSError, ValueError, TypeError):
            print("Benchmark 错误状态无法写回 JSON；以本次非零退出及 stderr 为准。", file=sys.stderr)
    return exit_code(result), saved


def run_benchmark(selected: list[str], *, root: Path = ROOT,
                  output_root: Path | None = None) -> tuple[dict, list[Path]]:
    if not selected or any(name not in SUITES for name in selected):
        raise ValueError("unknown or empty suite selection")
    selected = list(dict.fromkeys(selected))
    definitions = resolve_selection(selected, UNITS, SUITES)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + str(uuid4())
    result = {
        "schema_version": "2", "run_id": run_id, "started_at": utc_now(), "ended_at": None,
        "selected_suites": selected, "baseline": collect_baseline(root),
        "units": [make_unit(definition, root) for definition in definitions],
        "unselected_units": [key for key in UNITS if key not in {d["unit_id"] for d in definitions}],
        "summary": {}, "framework_errors": [],
    }
    directory = (output_root or root / "logs/benchmarks") / run_id
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        result["framework_errors"].append(issue("saving", "run_directory_unavailable", "无法创建本次产物目录，检查未启动。", exc))
        result["ended_at"] = utc_now()
        result["summary"] = summarize(result["units"])
        print("Benchmark 无法创建产物目录；检查未执行，没有已保存的报告。", file=sys.stderr)
        return result, []
    try:
        atomic_write_json(directory / "plan.json", result)
        for definition, unit in zip(definitions, result["units"]):
            if execute_unit(unit, definition, run_id, directory, root=root):
                result["framework_errors"].append(issue("execution", "scheduling_stopped", "中断、记录故障或进程树收尾未确认，后续单元未执行。"))
                break
    except (Exception, KeyboardInterrupt) as exc:
        result["framework_errors"].append(issue("framework", "runner_error", "框架未完成执行，保留已取得的事实。", exc))
    result["ended_at"] = utc_now()
    _, saved = persist_result(result, directory)
    return result, saved
