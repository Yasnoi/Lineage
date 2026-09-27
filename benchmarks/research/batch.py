"""Serial, bounded research batches with independent samples and durable plans."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from dotenv import dotenv_values

from benchmarks.common import ROOT, atomic_write_json, atomic_write_text, sha256_file, utc_now
from benchmarks.results import known, unknown, summarize, validate_result
from benchmarks.runner import collect_baseline, safe_environment, issue, _stop, _evidence, persist_result


SUITE = "research-wechat"


def new_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + str(uuid4())


def load_connection(root: Path, profile: dict, environment: dict | None = None) -> dict:
    """Read only the approved connection; never mutate the parent environment."""
    environment = os.environ if environment is None else environment
    local = dotenv_values(root / ".env", interpolate=False)
    selected = {key: environment.get(key, local.get(key) or "").strip()
                for key in ("API_KEY", "BASE_URL", "MODEL_NAME")}
    if not selected["API_KEY"]:
        raise ValueError("missing_credentials")
    if (selected["BASE_URL"] != profile["base_url"]
            or selected["MODEL_NAME"] != profile["model_name"]):
        raise ValueError("unapproved_connection")
    return selected


def research_environment(connection: dict) -> dict:
    env = safe_environment(dict(os.environ))
    env.update({key: connection[key] for key in ("API_KEY", "BASE_URL", "MODEL_NAME")})
    return env


def model_profile(profile: dict, case: dict) -> dict:
    from dataclasses import fields
    from lineage_agent.config.settings import ModelSettings
    allowed = {item.name for item in fields(ModelSettings)} - {"project_root", "log_dir"}
    values = {key: value for key, value in profile.items() if key in allowed}
    values.update(case.get("settings_overrides", {}))
    # Persist all resolved numeric defaults; credentials are never ModelSettings fields.
    settings = ModelSettings(**values)
    return {key: getattr(settings, key) for key in sorted(allowed)}


def make_unit(case: dict, phase: str, repeat: int, sample_id: str, profile: dict) -> dict:
    from benchmarks.research.worker import planned_coverage
    return {
        "suite_id": SUITE, "unit_id": f"{phase}-{repeat}-{case['case_id']}",
        "case_id": case["case_id"], "sample_run_id": sample_id,
        "phase": phase, "repeat": repeat, "category": "research",
        "definition_version": case["version"], "granularity": "research_case",
        "execution_mode": "real_model",
        "conditions": {
            "model": known({"kind": "real_model", "model_name": profile["model_name"],
                            "base_url": profile["base_url"], "weights_version": unknown("服务端未提供固定权重版本。")}),
            "configuration": known({"profile_id": profile["profile_id"], "version": profile["version"],
                                    "settings": model_profile(profile, case), "tracing": "disabled", "retries": 0}),
            "materials": unknown("材料尚未准备。"), "rubric": unknown("评分规则尚未绑定。"),
        },
        "execution_status": "not_run", "verdict": "pending_review",
        "agent_outcome": unknown("Agent 尚未执行。"),
        "assessment_basis": "计划样本；尚未执行与人工评分。",
        "dimensions": [{"id": key, "verdict": "pending_review", "basis": "等待用户人工评审。"}
                       for key in case["dimensions"]],
        "turns": [{"turn": index, "execution_status": "not_run", "verdict": "pending_review",
                   "started_at": None, "ended_at": None, "run_id": unknown("尚未启动。"),
                   "session_id": unknown("尚未启动。"), "checkpoint_id": unknown("尚未启动。"),
                   "agent_outcome": unknown("尚未启动。"), "answer_evidence": [], "process_evidence": []}
                  for index in range(1, len(case["turns"]) + 1)],
        "coverage": planned_coverage(case["case_id"]), "issues": [], "evidence": [],
        "metrics": {"duration_seconds": unknown("样本尚未启动。"),
                    "usage": unknown("尚无真实 usage。"), "cost": unknown("没有可核验账单。")},
    }


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
    envelope = {"schema_version": "2", "run_id": data["run_id"], "started_at": data.get("started_at"),
                "ended_at": data.get("ended_at"), "selected_suites": [SUITE], "baseline": {},
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
                [sys.executable, "-m", "benchmarks.research.worker", "--spec", str(spec_path)],
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


def _report_batch(result: dict, directory: Path, planned: list[dict], elapsed: float,
                  request_count: int | None) -> list[Path]:
    """The normal result contract remains authoritative; add a compact batch index."""
    from benchmarks.research.review import make_review_packet
    result["ended_at"] = utc_now()
    result["batch"] = {"planned_samples": 24, "planned_formal_samples": 18,
                       "planned_turns": 56, "execution_seconds": elapsed,
                       "sdk_requests": known(request_count) if request_count is not None else unknown("账本无法核验。"),
                       "samples": planned}
    _, saved = persist_result(result, directory)
    if directory / "result.json" in saved:
        try:
            saved.extend(make_review_packet(result, directory))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["framework_errors"].append(issue("saving", "review_packet_failed", "人工评审包生成失败。", exc))
            partial = getattr(exc, "saved", {})
            if partial:
                result["framework_errors"][-1]["partial_artifacts"] = [str(path) for path in partial.values()]
            print("人工评审包生成失败；部分模板可能绑定错误记录前的结果，不能作为完整评审包使用。", file=sys.stderr)
            _, resaved = persist_result(result, directory)
            saved = list(dict.fromkeys(saved + resaved))
    return saved


def run_research(profile_id: str, *, root: Path = ROOT, source_repo: Path | None = None,
                 output_root: Path | None = None) -> tuple[dict, list[Path]]:
    from benchmarks.research.definitions import CASES, PROFILE, definition_fingerprint, load_rubric
    from benchmarks.research.materials import prepare_materials, instantiate_materials, verify_materials
    from benchmarks.research.budget import read_ledger

    if profile_id != PROFILE["profile_id"]:
        raise ValueError("unknown research profile")
    output_root = output_root or root / "logs/benchmarks"
    batch_id = new_id()
    directory = output_root / batch_id
    baseline = collect_baseline(root)
    baseline["research_sources"] = known({str(path.relative_to(root)): sha256_file(path)
                                          for path in sorted((root / "benchmarks/research").rglob("*"))
                                          if path.is_file() and "__pycache__" not in path.parts})
    planned, units = [], []
    for phase, repeats in (("pilot", 1), ("formal", 3)):
        for repeat in range(1, repeats + 1):
            for case in CASES:
                sample_id = new_id()
                units.append(make_unit(case, phase, repeat, sample_id, PROFILE))
                planned.append({"run_id": sample_id, "case_id": case["case_id"], "phase": phase,
                                "repeat": repeat, "directory": str((output_root / sample_id).absolute())})
    result = {"schema_version": "2", "run_id": batch_id, "started_at": utc_now(), "ended_at": None,
              "selected_suites": [SUITE], "baseline": baseline, "units": units, "summary": {},
              "unselected_units": [], "framework_errors": []}
    elapsed, request_count = 0.0, 0
    try:
        directory.mkdir(parents=True, exist_ok=False)
        atomic_write_json(directory / "plan.json", {**result, "samples": planned})
    except OSError as exc:
        result["framework_errors"].append(issue("saving", "run_directory_unavailable", "无法保存完整批次计划；未发送请求。", exc))
        result["summary"] = summarize(units)
        result["ended_at"] = utc_now()
        return result, []
    try:
        connection = load_connection(root, PROFILE)
        prepared_root = root / "logs/benchmark-materials" / batch_id
        manifest = prepare_materials(root, source_repo or root.parent / "WeChat Reporting", prepared_root)
        rubric_definition = load_rubric()
        for unit in units:
            unit["conditions"]["materials"] = known(manifest)
            unit["conditions"]["rubric"] = known(rubric_definition)
            unit["conditions"]["definition"] = known({"sha256": definition_fingerprint()})
        ledger_path = directory / "requests.jsonl"
        ledger_path.touch(exist_ok=False)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        for unit in units:
            unit.update(execution_status="blocked", assessment_basis="连接配置或固定材料前置条件不满足；未调用模型。")
        result["framework_errors"].append(issue("preflight", "research_blocked", "连接配置或固定材料准备失败；没有回退。", exc))
        return result, _report_batch(result, directory, planned, elapsed, request_count)
    try:
        ledger_snapshot = []
        for index, (item, unit) in enumerate(zip(planned, units)):
            events = read_ledger(ledger_path, max_requests=400, batch_id=batch_id,
                                 run_ids={item["run_id"] for item in planned})
            if events != ledger_snapshot:
                raise ValueError("request ledger changed between isolated samples")
            request_count = len(events)
            if request_count >= 400 or elapsed >= 7200:
                result["framework_errors"].append(issue("budget", "batch_limit", "已达到批准的整批执行预算；后续样本未执行。"))
                break
            case = next(case for case in CASES if case["case_id"] == item["case_id"])
            sample_dir = Path(item["directory"])
            sample_dir.mkdir(parents=True, exist_ok=False)
            sample_manifest = instantiate_materials(prepared_root, sample_dir / "materials", case["case_id"])
            if verify_materials(sample_dir / "materials") != sample_manifest:
                raise ValueError("sample material changed before launch")
            # The sample manifest explicitly identifies its isolated Agent-visible root.
            material_root = Path(sample_manifest["material_root"])
            unit["conditions"]["materials"] = known(sample_manifest)
            timeout = min(300.0, 7200.0 - elapsed)
            spec = {"schema_version": "1", "batch_id": batch_id, "run_id": item["run_id"],
                    "case_id": case["case_id"], "definition_version": case["version"],
                    "prompts": case["turns"], "profile": model_profile(PROFILE, case),
                    "material_root": str(material_root.absolute()), "output_dir": str(sample_dir.absolute()),
                    "ledger_path": str(ledger_path.absolute()), "max_requests": 400,
                    "remaining_seconds": timeout}
            sample_result = {**result, "run_id": item["run_id"], "started_at": utc_now(), "ended_at": None,
                             "units": [unit], "framework_errors": [], "summary": {}}
            atomic_write_json(sample_dir / "plan.json", sample_result)
            print(f"Research {index + 1}/24: {unit['unit_id']} ({item['run_id']})", flush=True)
            stop = execute_sample(unit, spec, sample_dir, connection, root=root, timeout=timeout)
            elapsed += unit["metrics"]["duration_seconds"]["value"]
            try:
                if verify_materials(sample_dir / "materials") != sample_manifest:
                    raise ValueError("sample material changed during execution")
            except (OSError, ValueError) as exc:
                unit["execution_status"] = "failed"
                unit["issues"].append(issue("materials", "material_changed", "样本材料与固定版本不符；不能视为约定版本运行。", exc))
                stop = True
            events = read_ledger(ledger_path, max_requests=400, batch_id=batch_id,
                                 run_ids={item["run_id"] for item in planned})
            if events[:len(ledger_snapshot)] != ledger_snapshot:
                raise ValueError("request ledger lost earlier requests")
            ledger_snapshot = events
            request_count = len(events)
            sample_result["ended_at"] = utc_now()
            _, sample_saved = persist_result(sample_result, sample_dir)
            item.update(execution_status=unit["execution_status"], verdict=unit["verdict"],
                        saved=[str(path) for path in sample_saved])
            print(f"  {unit['execution_status']} / {unit['verdict']}; batch requests={request_count}", flush=True)
            # Save batch progress before advancing, including model/Agent failures.
            result["summary"] = summarize(units)
            atomic_write_json(directory / "progress.json", {**result, "samples": planned})
            if sample_result["framework_errors"] or stop:
                result["framework_errors"].append(issue("execution", "scheduling_stopped", "记录、全局前置条件或收尾故障；后续样本未执行。"))
                break
            if index == 5:
                atomic_write_json(directory / "pilot-gate.json", {
                    "checked_at": utc_now(), "definition_sha256": definition_fingerprint(),
                    "status": "technical_recording_complete", "sdk_requests": request_count,
                    "basis": "六个试运行保留有效执行回执与账本；不以回答质量或机制通过作为准入条件。"})
    except (Exception, KeyboardInterrupt) as exc:
        result["framework_errors"].append(issue("framework", "research_batch_stopped", "研究批次中断；已保存事实保留，不自动重跑。", exc))
        try:
            final_ledger = read_ledger(ledger_path, max_requests=400, batch_id=batch_id,
                                      run_ids={item["run_id"] for item in planned})
            if final_ledger[:len(ledger_snapshot)] != ledger_snapshot:
                raise ValueError("request ledger lost earlier requests")
            request_count = len(final_ledger)
        except Exception:
            request_count = None
    return result, _report_batch(result, directory, planned, elapsed, request_count)
