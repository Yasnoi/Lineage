"""Serial, bounded research batches with independent samples and durable plans."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from benchmarks.core.files import ROOT, atomic_write_json, sha256_file
from benchmarks.core.contracts import known, unknown, summarize, validate_result
from benchmarks.core.metadata import issue, _evidence
from benchmarks.core.processes import safe_environment, _stop
from benchmarks.core.configuration import checked_config, checked_model, load_connection
from benchmarks.core.materials import prepare_materials, instantiate_materials, verify_materials
from benchmarks.core.budgets import read_ledger


def research_environment(connection):
    return {**safe_environment(dict(os.environ)), **connection}


def plan(suite, config):
    # Invalid/missing configuration still produces the complete one-repeat blocked plan.
    repeats = config.get("repeats", 1) if isinstance(config, dict) else 1
    if type(repeats) is not int or not 1 <= repeats <= 20:
        repeats = 1
    return [{**deepcopy(task), "suite_id": suite["suite_id"], "repeat": repeat,
             "unit_id": f"{task['unit_id']}.r{repeat}", "sample_run_id": str(uuid4()),
             "rubric": deepcopy(suite["rubric"])}
            for repeat in range(1, repeats + 1) for task in suite["units"]]


def make_unit(task, root=ROOT):
    from benchmarks.core.executors.agent_worker import pending_turn, planned_coverage
    return {"suite_id": task["suite_id"], "unit_id": task["unit_id"], "case_id": task["case_id"],
            "sample_run_id": task["sample_run_id"], "repeat": task["repeat"], "category": "research",
            "definition_version": task["version"], "granularity": "task", "execution_mode": "real_model",
            "conditions": {"model": unknown("运行配置尚未核验。"), "configuration": unknown("运行配置尚未核验。"),
                           "materials": unknown("材料尚未准备。"), "rubric": known(task["rubric"]),
                           "task": known({key: value for key, value in task.items() if key != "rubric"})},
            "execution_status": "not_run", "verdict": "pending_review", "agent_outcome": unknown("尚未执行。"),
            "assessment_basis": "计划任务；尚未执行或评分。", "dimensions": [
                {"id": name, "verdict": "pending_review"} for name in task["dimensions"]],
            "issues": [], "evidence": [], "assessments": [], "metrics": {"duration_seconds": unknown("尚未执行。")},
            "turns": [pending_turn(index) for index in range(1, len(task["turns"]) + 1)],
            "coverage": planned_coverage(task.get("mechanism"))}


def prepare(suite, config, directory, result, *, root=ROOT):
    stage = "configuration"
    try:
        config = checked_config(config)
        stage = "credentials"
        connection = load_connection(config)
        stage = "task_configuration"
        # Validate every task override before allowing the first request.
        for unit in result["units"]:
            profile = model_profile(config["model"], unit["conditions"]["task"]["value"])
            unit["conditions"]["model"] = known({"name": profile["model_name"], "endpoint": profile["base_url"],
                                                "weights_version": unknown("服务端未提供固定权重版本。")})
            unit["conditions"]["configuration"] = known({"settings": profile, "limits": config["limits"],
                                                         "repeats": config["repeats"], "tracing": False, "retries": 0})
        stage = "materials"
        prepared = root / "logs/benchmark-materials" / result["run_id"]
        manifest = prepare_materials(root, Path(config["source_repository"]), prepared, suite["materials"])
        stage = "request_recording"
        ledger = directory / "requests.jsonl"
        ledger.touch(exist_ok=False)
        result["batch"] = {"planned_samples": len(result["units"]),
                           "planned_turns": sum(len(u["turns"]) for u in result["units"]),
                           "execution_seconds": 0.0, "sdk_requests": known(0)}
        return {"config": config, "connection": connection, "prepared": prepared, "manifest": manifest,
                "ledger": ledger, "snapshot": [], "result": result, "elapsed": 0.0}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        for unit in result["units"]:
            unit.update(execution_status="blocked", assessment_basis="显式配置、凭据或固定材料前置条件未满足；未调用模型。")
            unit["issues"].append(issue("preflight", stage + "_unavailable", "前置条件未满足；没有替换模型或回退替身。", exc))
        return None


def execute_unit(unit, task, run_id, directory, *, root=ROOT, context):
    config, result = context["config"], context["result"]
    limits = config["limits"]
    ids = {item["sample_run_id"] for item in result["units"]}
    events = read_ledger(context["ledger"], limits["requests"], batch_id=run_id, run_ids=ids)
    if events != context["snapshot"]:
        raise ValueError("request ledger changed between isolated tasks")
    if len(events) >= limits["requests"] or context["elapsed"] >= limits["execution_seconds"]:
        unit["issues"].append(issue("budget", "batch_exhausted", "整批预算耗尽，本任务未执行。"))
        return True
    if verify_materials(context["prepared"]) != context["manifest"]:
        raise ValueError("prepared material changed")
    sample = directory / unit["unit_id"]
    sample.mkdir(exist_ok=False)
    manifest = instantiate_materials(context["prepared"], sample / "materials", task)
    unit["conditions"]["materials"] = known(manifest)
    timeout = min(limits["sample_seconds"], limits["execution_seconds"] - context["elapsed"])
    spec = {"schema_version": "1", "batch_id": run_id, "run_id": unit["sample_run_id"],
            "case_id": task["case_id"], "definition_version": task["version"], "prompts": task["turns"],
            "mechanism": task.get("mechanism"), "profile": model_profile(config["model"], task),
            "material_root": manifest["material_root"], "output_dir": str(sample.absolute()),
            "ledger_path": str(context["ledger"].absolute()), "max_requests": limits["requests"], "remaining_seconds": timeout}
    started = time.monotonic()
    result["batch"]["sdk_requests"] = unknown("本实例的请求记账尚未完成核验。")
    try:
        stop = execute_sample(unit, spec, sample, context["connection"], root=root, timeout=timeout)
    finally:
        context["elapsed"] += time.monotonic() - started
        result["batch"]["execution_seconds"] = context["elapsed"]
    if verify_materials(sample / "materials") != manifest:
        unit.update(execution_status="failed", verdict="pending_review")
        unit["issues"].append(issue("materials", "material_changed", "材料在执行期间发生变化。"))
        stop = True
    events = read_ledger(context["ledger"], limits["requests"], batch_id=run_id, run_ids=ids)
    if events[:len(context["snapshot"])] != context["snapshot"]:
        raise ValueError("earlier request accounting lost")
    context["snapshot"] = events
    result["batch"]["sdk_requests"] = known(len(events))
    from benchmarks.evaluators.rules import mechanism_assessment
    unit["assessments"].append(mechanism_assessment(unit))
    return stop


def model_profile(profile: dict, case: dict) -> dict:
    from dataclasses import fields
    from lineage_agent.config.settings import ModelSettings
    allowed = {item.name for item in fields(ModelSettings)} - {"project_root", "log_dir"}
    values = {key: value for key, value in profile.items() if key in allowed}
    values.update(case.get("settings_overrides", {}))
    # Persist all resolved numeric defaults; credentials are never ModelSettings fields.
    return checked_model(values)


def _receipt(path: Path, spec: dict, spec_path: Path, *, final: bool, returncode: int | None) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid research receipt object")
    if (data.get("schema_version") != "1" or data.get("run_id") != spec["run_id"]
            or data.get("case_id") != spec["case_id"] or data.get("spec_sha256") != sha256_file(spec_path)
            or data.get("definition_version") != spec.get("definition_version")):
        raise ValueError("research receipt identity mismatch")
    if data.get("final") is not final:
        raise ValueError("research receipt finality mismatch")
    if data.get("verdict") != "pending_review" or type(data.get("global_stop")) is not bool:
        raise ValueError("worker cannot assign Research quality verdicts")
    if data.get("execution_status") not in {"completed", "failed", "blocked", "not_run"}:
        raise ValueError("invalid worker execution status")
    if final:
        expected = 0 if data["execution_status"] == "completed" else 2
        if returncode != expected or not data.get("ended_at"):
            raise ValueError("research receipt/process mismatch")
    if not isinstance(data.get("turns"), list) or len(data["turns"]) != len(spec["prompts"]):
        raise ValueError("incomplete worker turn plan")
    if final and data.get("observations_sha256"):
        if sha256_file(path.parent / "observations.jsonl") != data["observations_sha256"]:
            raise ValueError("worker observations changed")
    return data


def _accept_receipt(unit: dict, data: dict, *, failed: bool) -> None:
    candidate = deepcopy(unit)
    for key in ("turns", "agent_outcome", "coverage", "issues", "metrics", "evidence"):
        if key in data:
            candidate[key] = data[key]
    candidate["execution_status"] = "failed" if failed else data["execution_status"]
    envelope = {"schema_version": "3", "run_id": data["run_id"], "started_at": data.get("started_at"),
                "ended_at": data.get("ended_at"), "selected_suites": [unit["suite_id"]], "baseline": {},
                "units": [candidate], "summary": summarize([candidate]), "framework_errors": []}
    validate_result(envelope)
    for turn in candidate["turns"]:
        answer = turn.get("answer_evidence")
        if isinstance(answer, dict) and answer.get("availability") == "available":
            if not turn.get("answer_sha256") or sha256_file(Path(answer["path"])) != turn["answer_sha256"]:
                raise ValueError("answer file differs from worker receipt")
    # Validation precedes mutation so damaged records cannot destroy the saved plan.
    unit.update(candidate)


def execute_sample(unit: dict, spec: dict, directory: Path, connection: dict,
                   *, root: Path = ROOT, timeout: float = 300) -> bool:
    """Return a global stop only for unsafe bookkeeping/interruption/cleanup."""
    spec_path = directory / "spec.json"
    atomic_write_json(spec_path, spec)
    stdout_path, stderr_path = directory / "stdout.txt", directory / "stderr.txt"
    receipt_path = directory / "worker-result.json"
    process = None
    failure = None
    stop = False
    started = time.perf_counter()
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-m", "benchmarks.core.executors.agent_worker", "--spec", str(spec_path)],
                cwd=root, env=research_environment(connection), stdout=stdout, stderr=stderr,
                **({"start_new_session": True} if os.name != "nt" else {}))
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                failure = issue("execution", "sample_timeout", "样本超过执行上限；已取得证据保留。")
            except KeyboardInterrupt:
                failure = issue("execution", "interrupted", "用户中断；后续样本未执行。")
                stop = True
            finally:
                try:
                    _stop(process)
                except Exception as exc:
                    failure = issue("cleanup", "process_tree_unconfirmed", "进程树收尾未确认。", exc)
                    stop = True
    except (OSError, KeyboardInterrupt) as exc:
        failure = issue("launch", "worker_unavailable", "无法启动研究 worker 或保存输出。", exc)
        stop = True
    elapsed = time.perf_counter() - started
    try:
        data = _receipt(receipt_path, spec, spec_path, final=failure is None,
                        returncode=process.returncode if process else None)
        _accept_receipt(unit, data, failed=failure is not None)
        if failure is None:
            unit["execution_status"] = data["execution_status"]
            unit["assessment_basis"] = "真实模型执行事实已记录；语义质量等待用户人工评分。"
        stop = stop or bool(data.get("global_stop"))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        # A finalized receipt may race with a timeout; retain it as facts, not success.
        if failure is not None:
            try:
                data = _receipt(receipt_path, spec, spec_path, final=True, returncode=0)
                _accept_receipt(unit, data, failed=True)
                stop = stop or bool(data.get("global_stop"))
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pass
        else:
            failure = issue("adaptation", "invalid_research_receipt", "研究回执缺失、损坏或身份/进程退出不一致。", exc)
            stop = True
            try:
                data = _receipt(receipt_path, spec, spec_path, final=False, returncode=None)
                _accept_receipt(unit, data, failed=True)
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pass
    unit["metrics"]["duration_seconds"] = known(elapsed)
    unit["metrics"]["cost"] = unknown("没有可核验账单；不从 token 估算金额。")
    unit["evidence"].extend([_evidence("stdout", stdout_path, "research worker"),
                             _evidence("stderr", stderr_path, "research worker"),
                             _evidence("worker_receipt", receipt_path, "research worker"),
                             _evidence("observations", directory / "observations.jsonl", "partial or final actual observations"),
                             _evidence("request_ledger", Path(spec["ledger_path"]), "batch request reservations")])
    if failure:
        unit.update(execution_status="failed", assessment_basis="执行或证据适配故障；此前确认事实保留。")
        unit["issues"].append(failure)
    return stop
