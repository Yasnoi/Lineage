"""One real Research sample in an isolated process; no semantic self-scoring."""

import argparse
import asyncio
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

from benchmarks.core.files import atomic_write_json, atomic_write_text, sha256_file, utc_now
from benchmarks.core.processes import require_local_agent as _require_local_agent
from benchmarks.core.contracts import known, not_applicable, unknown
from benchmarks.core.budgets import RequestBudget, WorkerAbort
from benchmarks.core.evidence import ObservedTools, Observer, digest
from lineage_agent.agent import runtime
from lineage_agent.agent.sessions import SessionManager
from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.models.budget import check_context_budget
from lineage_agent.models.requests import prepare_request
from lineage_agent.models.summarization import is_summary_request
from lineage_agent.recording.tracing import trace_scope
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL


TOOL_DEFINITIONS = [DIRECTORY_TOOL, TEXT_TOOL]
COMPACTION_CHECKS = (
    "setup_without_compaction", "summary_saved_and_activated", "old_record_selected", "old_user_requirements_selected",
    "following_uses_saved_version", "old_tool_originals_removed", "old_user_originals_removed", "following_keeps_selected_version",
    "no_recent_tool_reads", "original_record_unchanged", "first_turn_requested_files_read", "first_turn_background_read",
)


def planned_coverage(mechanism: dict | None) -> list[dict]:
    kind = (mechanism or {}).get("kind")
    names = COMPACTION_CHECKS if kind == "compaction" else ("actual_read_limit",) if kind == "read_limit" else ()
    return [{"name": name, "status": "gap", "reason": "mechanism not yet observed", "evidence": []} for name in names]


def evidence(path: Path, kind: str, source: str) -> dict:
    return {"kind": kind, "path": str(path.absolute()), "source": source,
            "availability": "available" if path.is_file() else "missing"}


def safe_issue(stage: str, code: str, **extra) -> dict:
    return {"stage": stage, "code": code, "message": code, **extra}


def value_or_unknown(value, reason="not observed") -> dict:
    return known(value) if value is not None else unknown(reason)


def make_settings(spec: dict) -> ModelSettings:
    profile = spec["profile"]
    retired = set(profile) & {"context_trigger_percent", "context_target_percent", "summary_max_output_tokens", "summary_max_bytes"}
    if retired:
        raise ConfigurationError("请移除旧 Compact 配置并使用 SUMMARY_TARGET_PERCENT：" + ", ".join(sorted(retired)))
    allowed = {field.name for field in fields(ModelSettings)} - {"log_dir", "project_root"}
    values = {key: value for key, value in profile.items() if key in allowed}
    return ModelSettings(**values, project_root=Path(spec["material_root"]),
                         log_dir=Path(spec["output_dir"]) / "agent-logs")


def pending_turn(index: int) -> dict:
    return {"turn": index, "execution_status": "not_run", "verdict": "pending_review",
            "started_at": None, "ended_at": None, "run_id": unknown("turn not started"),
            "session_id": unknown("turn not started"), "checkpoint_id": unknown("turn not started"),
            "checkpoint_complete": None, "agent_outcome": unknown("turn not started"),
            "metrics": {}, "issues": [], "answer_evidence": [], "process_evidence": []}


def usage_metrics(observer: Observer) -> dict:
    metrics = {"sdk_requests": known(len(observer.request_events)),
               "business_requests": known(sum(e["purpose"] == "business" for e in observer.request_events)),
               "summary_requests": known(sum(e["purpose"] == "summary" for e in observer.request_events)),
               "cost": unknown("no verified billing data; no price-derived estimate")}
    for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"):
        values = [usage[key] for usage in observer.request_usage if usage.get(key) is not None]
        metrics[key] = known(sum(values)) if values else unknown("endpoint usage not observed")
        metrics[key + "_missing_requests"] = known(len(observer.request_events) - len(values))
    metrics["mean_request_seconds"] = (known(sum(observer.request_durations) / len(observer.request_durations))
                                       if observer.request_durations else unknown("no completed transport observation"))
    complete = bool(observer.request_events) and all(
        metrics[key + "_missing_requests"]["value"] == 0 for key in ("input_tokens", "cached_input_tokens"))
    if complete and metrics["input_tokens"]["value"] > 0 and metrics["cached_input_tokens"]["value"] <= metrics["input_tokens"]["value"]:
        metrics["cache_input_fraction"] = known(metrics["cached_input_tokens"]["value"] / metrics["input_tokens"]["value"])
    else:
        metrics["cache_input_fraction"] = unknown("complete, consistent endpoint input/cache usage with positive denominator required")
    return metrics


def pressure_prompt(base: str, previous: dict, requests: list[dict], settings: ModelSettings,
                    max_bytes: int = 16384) -> tuple[str, dict]:
    if type(max_bytes) is not int or not 1 <= max_bytes <= 32768:
        raise WorkerAbort("pressure_limit_invalid")
    business = [request for request in requests if not is_summary_request(request)]
    if not business:
        raise WorkerAbort("pressure_request_evidence_missing")
    history_end = previous["history_end"]
    history = previous["input"][:history_end]
    prepared = prepare_request(settings.model_name, history + [{"role": "user", "content": base}],
                               tools=business[-1].get("tools", TOOL_DEFINITIONS),
                               instructions=business[-1].get("instructions"),
                               max_output_tokens=settings.max_output_tokens)
    risks = previous["context_risks"][:history_end] + [None]
    measured = check_context_budget(prepared, risks, settings)
    if measured["decision"] == "error":
        raise WorkerAbort("pressure_measurement_failed")
    desired = max(1, measured["limit_tokens"] - measured["required_tokens"] + 512)
    actual = min(max_bytes, desired)
    return base + "p" * actual, {"formula_version": "utf8-window-v3", "desired_bytes": desired,
                                "max_padding_bytes": max_bytes,
                                "padding_bytes": actual, "capped": desired > actual,
                                "trigger_tokens": measured["trigger_tokens"],
                                "required_tokens": measured["required_tokens"]}


def context_projection(state: dict) -> dict:
    active = state.get("active_context") or {}
    compaction = state.get("compaction") or {}
    return {"active_version": value_or_unknown(active.get("version_id")),
            "summary": known(active["summary"]) if active.get("summary") else not_applicable("no active summary"),
            "compaction": {key: deepcopy(compaction[key]) for key in (
                "status", "reason", "summary_id", "old_version_id", "candidate_version_id", "sources",
                "removed_items", "removed_turns", "summary_checkpoint_id", "confirmed_checkpoint_id",
                "pass_no", "candidate_input", "summary_request_sha256", "recovery_budget",
            ) if key in compaction}}


def read_limit_coverage(observer: Observer, mechanism: dict) -> dict:
    hits = [event for event in observer.tool_events
            if event["name"] == "read_text_file"
            and event.get("path") == mechanism["path"]
            and event.get("reason_code") == "file_too_large"]
    return {"name": "actual_read_limit", "status": "verified" if hits else "gap",
            "reason": "target read rejected by actual byte limit" if hits else "required target rejection not observed",
            "evidence": [evidence(observer.output_dir / "observations.jsonl", "process", "actual tool executor")],
            "event_sequences": [event["seq"] for event in hits]}


def compaction_coverage(observer: Observer, states: list[dict], old_record: dict | None,
                        unchanged_record: bool | None, prompts: list[str], mechanism: dict,
                        summary_record: dict | None = None) -> list[dict]:
    source_turn = mechanism["source_turn"]
    pressure_turn = mechanism["pressure_turn"]
    follow_up_turn = mechanism["follow_up_turn"]
    proofs = {}
    proofs["setup_without_compaction"] = len(states) >= pressure_turn - 1 and all(s.get("summary_calls") == 0 for s in states[:pressure_turn - 1])
    summary_requests = [r for r in observer.requests.get(pressure_turn, []) if is_summary_request(r)]
    compacted = states[pressure_turn - 1] if len(states) >= pressure_turn else {}
    before = states[pressure_turn - 2] if len(states) >= pressure_turn - 1 else {}
    compaction = compacted.get("compaction") or {}
    active = compacted.get("active_context") or {}
    attempts = (summary_record or {}).get("summary_results", [])
    lineage_valid = bool(1 <= len(attempts) <= 2 and len(attempts) == len(summary_requests)
                         and (summary_record or {}).get("run_id") == compacted.get("run_id"))
    for index, attempt in enumerate(attempts):
        entry = attempt.get("compaction") or {}
        lineage_valid = lineage_valid and (
            entry.get("pass_no") == index + 1
            and entry.get("request_no") == compaction.get("request_no")
            and index < len(summary_requests)
            and entry.get("summary_request_sha256") == digest(summary_requests[index])
        )
        if index:
            previous = attempts[index - 1]
            previous_entry = previous.get("compaction") or {}
            lineage_valid = lineage_valid and entry.get("candidate_input") == {
                "summary_id": previous_entry.get("summary_id"), "checkpoint_id": previous.get("checkpoint_id"),
            } and entry.get("sources") == previous_entry.get("sources")
            try:
                material = json.loads(summary_requests[index]["input"][0]["content"])
                lineage_valid = lineage_valid and material["candidate_input"] == {
                    **entry["candidate_input"], "document": previous_entry["document"],
                } and "previous_summary" not in material and all("payload" not in item for item in material["sources"])
            except (KeyError, IndexError, TypeError, ValueError):
                lineage_valid = False
    last = (attempts[-1].get("compaction") or {}) if attempts else {}
    lineage_valid = lineage_valid and last.get("summary_id") == compaction.get("summary_id")
    proofs["summary_saved_and_activated"] = bool(
        lineage_valid and compaction.get("status") == "used"
        and compaction.get("summary_checkpoint_id") and compaction.get("confirmed_checkpoint_id")
        and active.get("summary") and active.get("version_id") != (before.get("active_context") or {}).get("version_id"))
    source_items = []
    try:
        source_items = json.loads(summary_requests[0]["input"][0]["content"])["sources"]
    except (IndexError, KeyError, TypeError, ValueError):
        pass
    first_run = states[source_turn - 1].get("run_id") if len(states) >= source_turn else None
    selected = [source for source in source_items if str(source.get("source_id", "")).startswith(str(first_run) + ":")]
    admitted = (old_record or {}).get("admitted_input", [])
    old_tools = [item for item in admitted if item.get("type") == "function_call_output"]
    proofs["old_record_selected"] = bool(old_tools and selected and all(
        any(source.get("payload") == item for source in selected) for item in old_tools))
    old_user_items = [item for item in admitted if item.get("role") == "user"]
    proofs["old_user_requirements_selected"] = bool(old_user_items and all(
        any(source.get("payload") == item for source in selected) for item in old_user_items))
    following = [r for r in observer.requests.get(follow_up_turn, []) if not is_summary_request(r)]
    message = compaction.get("message")
    proofs["following_uses_saved_version"] = bool(
        following and message and following[0].get("input") == compacted.get("input", []) + [{"role": "user", "content": prompts[follow_up_turn - 1]}]
        and following[0]["input"][0] == message)
    proofs["old_tool_originals_removed"] = bool(following and old_tools and all(
        item not in following[0].get("input", []) for item in old_tools))
    proofs["old_user_originals_removed"] = bool(following and old_user_items and all(
        item not in following[0].get("input", []) for item in old_user_items))
    proofs["following_keeps_selected_version"] = bool(len(states) >= follow_up_turn and states[follow_up_turn - 1].get("summary_calls") == 0
        and (states[follow_up_turn - 1].get("active_context") or {}).get("version_id") == active.get("version_id"))
    proofs["no_recent_tool_reads"] = bool(len(states) >= follow_up_turn and all(state.get("tool_calls") == 0 for state in states[mechanism["no_tools_from_turn"] - 1:]))
    proofs["original_record_unchanged"] = unchanged_record is True
    proofs["first_turn_requested_files_read"] = any(
        event["turn"] == source_turn and event["name"] == "read_text_file" and event["outcome"] == "succeeded"
        and event.get("path") == mechanism["source_path"] for event in observer.tool_events)
    proofs["first_turn_background_read"] = any(
        event["turn"] == source_turn and event["name"] == "read_text_file" and event["outcome"] == "succeeded"
        and event.get("path") == mechanism["background_path"] and event.get("bytes_read") == mechanism["background_bytes"]
        for event in observer.tool_events)
    process = evidence(observer.output_dir / "mechanism.json", "mechanism", "in-memory actual SDK comparisons")
    details = {"checks": proofs, "first_run_id": first_run,
               "summary_source_ids": [item.get("source_id") for item in selected],
               "old_tool_item_sha256": [digest(item) for item in old_tools],
               "summary_request_sha256": [digest(item) for item in summary_requests],
               "following_request_sha256": [digest(item) for item in following],
               "semantic_boundary": "Recent answers require separate human review for restated facts; no semantic pass is inferred."}
    atomic_write_json(observer.output_dir / "mechanism.json", details)
    process["availability"] = "available"
    return [{"name": name, "status": "verified" if passed else "gap",
             "reason": "actual request/checkpoint comparison satisfied" if passed else "required mechanism evidence absent or differs",
             "evidence": [process]} for name, passed in proofs.items()]


async def execute_sample(spec: dict, spec_sha256: str) -> tuple[dict, int]:
    """Execute approved fixed prompts; callers may inject test factories, never endpoint fallback."""
    mechanism = spec.get("mechanism") or {}
    output = Path(spec["output_dir"])
    started = time.monotonic()
    result = {"schema_version": "1", "spec_sha256": spec_sha256, "run_id": spec["run_id"],
              "case_id": spec["case_id"], "definition_version": spec["definition_version"],
              "started_at": utc_now(), "ended_at": None, "final": False,
              "execution_status": "failed", "verdict": "pending_review",
              "agent_outcome": unknown("no Agent outcome yet"),
              "turns": [pending_turn(i) for i in range(1, len(spec["prompts"]) + 1)],
              "metrics": {}, "issues": [], "evidence": [], "coverage": planned_coverage(mechanism),
              "global_stop": False, "global_stop_reason": None}
    observer = None
    current = None
    states = []
    old_record = None
    unchanged_record = None
    manager = None
    sid = None
    code = 2
    primary_failure = False

    def save():
        result["metrics"] = usage_metrics(observer) if observer else {}
        result["metrics"]["duration_seconds"] = known(time.monotonic() - started)
        atomic_write_json(output / "worker-result.json", result)

    if (output / "worker-result.json").exists() or (output / "observations.jsonl").exists():
        print("Research worker refuses existing sample artifacts.", file=sys.stderr)
        return result, 2
    try:
        save()
        tracing = spec.get("tracing", {"enabled": False})
        if not isinstance(tracing, dict) or type(tracing.get("enabled")) is not bool:
            raise WorkerAbort("tracing_configuration_invalid")
        os.environ["LANGSMITH_TRACING"] = "true" if tracing["enabled"] else "false"
        for name in ("LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2"):
            os.environ[name] = "false"
        if not tracing["enabled"]:
            for name in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
                os.environ.pop(name, None)
        if not spec.get("case_id") or not spec["prompts"] or any(not isinstance(p, str) or not p.strip() for p in spec["prompts"]):
            raise ValueError("invalid fixed research definition")
        if not __debug__:
            raise WorkerAbort("assertions_disabled")
        try:
            _require_local_agent()
        except ImportError as exc:
            raise WorkerAbort("agent_source_mismatch") from exc
        if not os.environ.get("API_KEY", "").strip():
            result["execution_status"] = "blocked"
            raise WorkerAbort("credentials_missing")
        try:
            settings = make_settings(spec)
        except (ValueError, TypeError, KeyError) as exc:
            result["execution_status"] = "blocked"
            raise WorkerAbort("research_profile_invalid") from exc
        budget = RequestBudget(Path(spec["ledger_path"]), spec["run_id"], spec["case_id"],
                               max_requests=spec.get("max_requests", 400), remaining_seconds=spec.get("remaining_seconds", 1200),
                               batch_id=spec.get("batch_id", spec["run_id"]))
        observer = Observer(budget, output, Path(spec["material_root"]))
        executor = ObservedTools(create_filesystem_executor(
            settings.project_root, max_entries=settings.directory_max_entries,
            max_bytes=settings.text_max_bytes, max_total_bytes=settings.text_total_max_bytes), observer)
        multiple = len(spec["prompts"]) > 1
        if multiple:
            manager = SessionManager()
            sid = manager.create_session(settings, tools=TOOL_DEFINITIONS, execute_tool=executor)
        with patch.object(runtime, "create_client", side_effect=observer.client_factory(runtime.create_client)):
            for index, original_prompt in enumerate(spec["prompts"], 1):
                observer.turn = index
                current = result["turns"][index - 1]
                current["started_at"] = utc_now()
                current["execution_status"] = "failed"
                current["session_id"] = known(sid) if sid else not_applicable("independent single run")
                prompt = original_prompt
                if mechanism.get("kind") == "compaction" and index == mechanism["pressure_turn"]:
                    prompt, padding = pressure_prompt(prompt, states[-1], observer.requests.get(index - 1, []), settings,
                                                      mechanism.get("pressure_max_bytes", 16384))
                    current["pressure"] = padding
                prompt_path = output / f"turn-{index:02d}-prompt.txt"
                atomic_write_text(prompt_path, prompt)
                current["prompt_evidence"] = evidence(prompt_path, "prompt", "fixed definition and pressure formula")
                current["prompt_sha256"] = sha256_file(prompt_path)
                current["answer_evidence"] = evidence(output / f"turn-{index:02d}-answer.md", "answer", "planned Agent answer")
                save()
                request_start = len(observer.request_events)
                turn_started = time.monotonic()
                checkpoint_complete = None
                with trace_scope({"benchmark_batch_id": spec.get("batch_id", spec["run_id"]),
                                  "sample_run_id": spec["run_id"], "case_id": spec["case_id"],
                                  "turn": index}) as scope:
                    try:
                        if multiple:
                            session_turn = await manager.run_turn(sid, prompt)
                            state = session_turn.state
                            checkpoint_complete = session_turn.checkpoint_complete
                            current["checkpoint_complete"] = checkpoint_complete
                            current["checkpoint_id"] = value_or_unknown(session_turn.checkpoint_id, "checkpoint not confirmed")
                            if checkpoint_complete and index == mechanism.get("source_turn", 1):
                                old_record = await manager.get_record(sid, state["run_id"])
                        else:
                            state = await runtime.run_agent(prompt, settings, tools=TOOL_DEFINITIONS, execute_tool=executor)
                            current["checkpoint_id"] = not_applicable("independent single run")
                    finally:
                        current["tracing"] = {"requested": tracing["enabled"], "traces": scope.statuses(),
                                              "remote_status": "not_checked"}
                        trace_path = output / f"turn-{index:02d}-tracing.json"
                        atomic_write_json(trace_path, current["tracing"])
                        current["process_evidence"].append(evidence(trace_path, "tracing", "production trace status snapshot"))
                states.append(deepcopy(state))
                current["run_id"] = known(state["run_id"])
                current["agent_outcome"] = value_or_unknown(state.get("stop_reason"))
                result["agent_outcome"] = deepcopy(current["agent_outcome"])
                current["context"] = context_projection(state)
                current["metrics"] = {"duration_seconds": known(time.monotonic() - turn_started),
                                      "sdk_requests": known(len(observer.request_events) - request_start)}
                for key in ("model_calls", "summary_calls", "tool_calls", "tool_requests", "tool_outcomes", "usage_totals", "summary_usage_totals"):
                    current["metrics"][key] = value_or_unknown(state.get(key))
                current["metrics"]["usage_basis"] = known("usage_totals already includes summary; summary_usage_totals is a subset")
                answer = (state.get("result") or {}).get("text")
                if isinstance(answer, str):
                    answer_path = output / f"turn-{index:02d}-answer.md"
                    atomic_write_text(answer_path, answer)
                    current["answer_evidence"] = evidence(answer_path, "answer", f"Agent run {state['run_id']}")
                    current["answer_sha256"] = sha256_file(answer_path)
                process = {"run_id": state["run_id"], "session_id": sid, "checkpoint_id": current["checkpoint_id"],
                           "checkpoint_complete": checkpoint_complete, "log_complete": state.get("log_complete"),
                           "stop_reason": state.get("stop_reason"), "error_code": state.get("error_code"),
                           "context": current["context"], "metrics": current["metrics"],
                           "observed_request_hashes": [digest(request) for request in observer.requests.get(index, [])]}
                if state.get("log_path") and Path(state["log_path"]).is_file():
                    log_events = [json.loads(line) for line in Path(state["log_path"]).read_text(encoding="utf-8").splitlines()]
                    process["log_identifiers"] = {
                        key: list(dict.fromkeys(event[key] for event in log_events if key in event))
                        for key in ("step_id", "call_id")}
                compaction = state.get("compaction") or {}
                if compaction.get("document"):
                    summary_path = output / f"turn-{index:02d}-summary.json"
                    atomic_write_json(summary_path, {"summary_id": compaction.get("summary_id"), "document": compaction["document"]})
                    current["process_evidence"].append(evidence(summary_path, "summary", "validated saved summary candidate"))
                process_path = output / f"turn-{index:02d}-process.json"
                atomic_write_json(process_path, process)
                current["process_evidence"].append(evidence(process_path, "process", "returned Agent facts and transport observation"))
                if state.get("log_path"):
                    current["process_evidence"].append(evidence(Path(state["log_path"]), "agent_log", "production safe JSONL"))
                current["execution_status"] = "completed"
                current["ended_at"] = utc_now()
                if state.get("log_complete") is not True or multiple and checkpoint_complete is not True:
                    raise WorkerAbort("agent_evidence_not_confirmed")
                save()
                if observer.global_stop:
                    raise WorkerAbort(observer.global_stop_reason)
                if state.get("stop_reason") != "completed":
                    for pending in result["turns"][index:]:
                        pending["issues"].append(safe_issue("dependency", "previous_agent_turn_not_completed"))
                    result["issues"].append(safe_issue("agent_execution", "agent_terminal_stopped_sample", turn=index))
                    break
        if mechanism.get("kind") == "compaction":
            summary_record = None
            if manager and old_record:
                unchanged_record = await manager.get_record(sid, states[mechanism.get("source_turn", 1) - 1]["run_id"]) == old_record
                pressure = mechanism["pressure_turn"]
                if len(states) >= pressure:
                    summary_record = await manager.get_record(sid, states[pressure - 1]["run_id"])
            result["coverage"] = compaction_coverage(observer, states, old_record, unchanged_record, spec["prompts"], mechanism, summary_record)
        elif mechanism.get("kind") == "read_limit":
            result["coverage"] = [read_limit_coverage(observer, mechanism)]
        result["execution_status"] = "completed"
        code = 0
    except BaseException as exc:
        primary_failure = True
        error = exc.code if isinstance(exc, WorkerAbort) else ("interrupted" if isinstance(exc, KeyboardInterrupt) else "worker_execution_failed")
        result["issues"].append(safe_issue("worker", error, error_type=type(exc).__name__))
        # Safe hard failures stop scheduling; ordinary API errors are returned as Agent facts.
        result["global_stop"] = isinstance(exc, (WorkerAbort, OSError, KeyboardInterrupt))
        result["global_stop_reason"] = error if result["global_stop"] else None
        if current and current["execution_status"] != "completed":
            current["execution_status"] = "failed"
            current["ended_at"] = utc_now()
            current["issues"].append(safe_issue("execution", error))
        for pending in result["turns"]:
            if pending["execution_status"] == "not_run":
                pending["issues"].append(safe_issue("dependency", "sample_stopped"))
        if observer and mechanism.get("kind") == "read_limit":
            result["coverage"] = [read_limit_coverage(observer, mechanism)]
    finally:
        if manager is not None:
            try:
                manager.close()
            except BaseException as exc:
                if not isinstance(exc, Exception) and not primary_failure:
                    raise
                # Preserve all completed answers and the primary stop reason.
                # Storage cleanup diagnostics never include paths or exception text.
                result["issues"].append(safe_issue("cleanup", "artifact_cleanup_error"))
                result["execution_status"] = "failed"
                code = 2
    result["ended_at"] = utc_now()
    result["final"] = True
    observation_path = output / "observations.jsonl"
    if observation_path.is_file():
        result["evidence"].append(evidence(observation_path, "process", "research observation events"))
        result["observations_sha256"] = sha256_file(observation_path)
    try:
        save()
    except BaseException:
        # Only previously completed atomic files may be advertised by the parent.
        print("Research worker result save failed; existing evidence is partial.", file=sys.stderr)
        return result, 2
    return result, code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        raw = args.spec.read_bytes()
        spec = json.loads(raw)
        _, code = asyncio.run(execute_sample(spec, hashlib.sha256(raw).hexdigest()))
        return code
    except KeyboardInterrupt:
        print("Research worker interrupted; inspect existing partial evidence.", file=sys.stderr)
        return 2
    except BaseException:
        print("Research worker startup or result recording failed.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
