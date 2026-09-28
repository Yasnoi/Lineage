"""Run trusted contract chains in separate offline processes."""
import json, os, subprocess, sys, time
from pathlib import Path
from benchmarks.core.files import ROOT, sha256_file
from benchmarks.core.contracts import known, unknown, not_applicable
from benchmarks.core.events import planned_groups, plan_digest, read_group_events
from benchmarks.core.metadata import issue, _digest, _evidence
from benchmarks.core.processes import safe_environment, _stop

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
                "assertions": "enabled", "scope": "行为链内按检查覆写配置；版本摘要覆盖定义，不代表所有运行使用相同配置。",
            }),
            "materials": known({"kind": "inline_fixtures", "script_sha256": script_version}),
            "rubric": known({"kind": "python_assertions", "version": "1", "script_sha256": script_version}),
        },
        "execution_status": "not_run", "verdict": "pending_review",
        "agent_outcome": not_applicable("一个行为链包含多个被测运行；具体终态见检查输出。"),
        "assessment_basis": "尚未执行，没有有效评测判定。",
        "dimensions": [], "issues": [], "evidence": [],
        "metrics": {"duration_seconds": unknown("检查尚未启动。"),
                    "usage": not_applicable("替身 usage 用于断言，不是实际模型消耗。")},
    }
    if "groups" in definition:
        unit.update(groups=planned_groups(definition))
        unit["conditions"]["definition"] = known({"version": "3", "plan_digest": plan_digest(definition["groups"])})
        unit["conditions"]["check_sources"] = known({
            str(p.relative_to(root)): _digest(p) for p in sorted((root / "benchmarks/checks").rglob("*.py"))})
        unit["module"] = definition["module"]
        unit["assessments"] = []
    return unit

def execute_unit(unit: dict, definition: dict, run_id: str, directory: Path,
                 *, root: Path = ROOT) -> bool:
    """Return whether interruption, recording, or uncertain cleanup stops scheduling."""
    script = root / definition["script"]
    if not script.is_file():
        unit.update(execution_status="blocked", assessment_basis="检查入口缺失，检查未启动。")
        unit["issues"].append(issue("preflight", "missing_entrypoint", "固定检查行为链不存在。"))
        return False
    unit_dir = directory / definition["unit_id"]
    unit_dir.mkdir()
    stdout_path, stderr_path = unit_dir / "stdout.txt", unit_dir / "stderr.txt"
    receipt_path = unit_dir / "receipt.json"
    worker = "benchmarks.core.executors.control_worker"
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
            _evidence("receipt", receipt_path, "control adapter v3" if "groups" in definition else "ReAct adapter v1"),
        ])
    if "groups" in definition:
        return finish_control_unit(unit, definition, run_id, root, unit_dir, process, failure, interruption)

def finish_control_unit(unit, definition, run_id, root, directory, process, failure, interruption):
    events = directory / "groups.jsonl"
    receipt_path = directory / "receipt.json"
    groups, event_errors, count = read_group_events(events, run_id, definition)
    unit["groups"] = groups
    unit["evidence"].append(_evidence("group_events", events, "control adapter v3"))
    fatal_recording = bool(event_errors) and process is not None and events.exists()
    unit["issues"].extend(event_errors)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = {"schema_version": "3", "adapter_version": "3", "run_id": run_id,
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
                            assessment_basis="完整行为链与全部计划组已结束。" + ("存在环境覆盖缺口。" if gaps else "断言通过。"))
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
                failure = {**(receipt.get("error") or {}), "message": "完整入口未正常结束，没有有效行为链级判定。"}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        if failure is None:
            failure = issue("adaptation", "invalid_receipt", "最终回执缺失或与实际组事件不一致。", exc)
    if event_errors and failure is None:
        failure = issue("group_recording", "invalid_event_stream", "组记录不能确认为完整有效。")
    if failure is not None:
        unit.update(execution_status="failed", verdict="pending_review", assessment_basis="执行、组记录或适配故障；此前已确认组保留。")
        unit["issues"].append(failure)
    from benchmarks.evaluators.rules import control_assessment
    unit["assessments"] = [control_assessment(unit)]
    return interruption or fatal_recording
