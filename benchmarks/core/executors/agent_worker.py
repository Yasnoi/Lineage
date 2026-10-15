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
from benchmarks.core.configuration import RETIRED_COMPACT_KEYS
from benchmarks.core.evidence import ObservedTools, Observer, digest
from benchmarks.core.compact_evidence import (
    HISTORY_CHECKS, capture_summary_context, history_coverage, native_binding, summary_context,
    summary_lane, turn_compact_evidence,
)
from lineage_agent.agent import runtime
from lineage_agent.agent.sessions import SessionManager
from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.models.budget import check_context_budget
from lineage_agent.models.responses import ModelError, parse_response
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
    "all_prior_turns_selected", "pressure_has_no_current_summary", "completed_registered_without_summary",
)


def planned_coverage(mechanism: dict | None) -> list[dict]:
    kind = (mechanism or {}).get("kind")
    names = (COMPACTION_CHECKS if kind == "compaction" else HISTORY_CHECKS if kind == "history_recall"
             else ("actual_read_limit",) if kind == "read_limit" else ())
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
    retired = set(profile) & RETIRED_COMPACT_KEYS
    if retired:
        raise ConfigurationError("请移除旧 Compact 配置；H/R 上限使用 SUMMARY_HISTORY_MAX_OUTPUT_TOKENS / SUMMARY_RUN_MAX_OUTPUT_TOKENS：" + ", ".join(sorted(retired)))
    allowed = {field.name for field in fields(ModelSettings)} - {"log_dir", "project_root", "application_root"}
    values = {key: value for key, value in profile.items() if key in allowed}
    return ModelSettings(**values, project_root=Path(spec["material_root"]),
                         application_root=Path(spec.get("application_root", Path.cwd())).resolve(),
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
    for purpose in ("summary", "business"):
        metrics[purpose + "_request_seconds"] = known(sum(duration for event, duration in
            zip(observer.request_events, observer.request_durations) if event["purpose"] == purpose))
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
    def measure(padding):
        inherited = runtime.inherit_context(previous, [{"role": "user", "content": base + "p" * padding}],
                                            [None], run_id="benchmark-pressure")
        prepared = {**deepcopy(business[-1]), "input": inherited["input"]}
        measured = check_context_budget(prepared, inherited["context_risks"], settings)
        if measured["decision"] == "error":
            raise WorkerAbort("pressure_measurement_failed")
        return measured

    initial = measure(0)
    largest = measure(max_bytes)
    capped = largest["decision"] != "exceeded"
    low, high = 0, max_bytes
    if capped:
        low = max_bytes
    else:
        while low < high:
            middle = (low + high) // 2
            if measure(middle)["decision"] == "exceeded":
                high = middle
            else:
                low = middle + 1
    final = measure(low)
    return base + "p" * low, {"formula_version": "actual-inherited-request-v4",
                              "max_padding_bytes": max_bytes, "padding_bytes": low, "capped": capped,
                              "base_required_tokens": initial["required_tokens"],
                              "required_tokens": final["required_tokens"], "decision": final["decision"],
                              "limit_tokens": final["limit_tokens"],
                              "preceding_required_tokens": measure(low - 1)["required_tokens"] if low else None}


def context_projection(state: dict) -> dict:
    active = state.get("active_context") or {}
    compaction = state.get("compaction") or {}
    return {"active_version": value_or_unknown(active.get("version_id")),
            "history_summary": known(active["history_summary"]) if active.get("history_summary") else not_applicable("no historical summary"),
            "run_summary": known(active["run_summary"]) if active.get("run_summary") else not_applicable("no current summary"),
            "compaction": {key: deepcopy(compaction[key]) for key in (
                "status", "reason", "request_no", "attempt_no", "old_version_id", "candidate_version_id",
                "removed_items", "removed_turns", "confirmed_checkpoint_id",
                "candidate_budget", "net_reclaimed", "remaining_budget", "lanes",
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
    source_turn, pressure_turn, follow_up_turn = (mechanism[key] for key in ("source_turn", "pressure_turn", "follow_up_turn"))
    proofs = {}
    proofs["setup_without_compaction"] = len(states) >= pressure_turn - 1 and all(s.get("summary_calls") == 0 for s in states[:pressure_turn - 1])
    summary_requests = [r for r in observer.requests.get(pressure_turn, []) if is_summary_request(r)]
    compacted = states[pressure_turn - 1] if len(states) >= pressure_turn else {}
    before = states[pressure_turn - 2] if len(states) >= pressure_turn - 1 else {}
    compaction, active = compacted.get("compaction") or {}, compacted.get("active_context") or {}
    attempts = (summary_record or {}).get("summary_results", [])
    lineage_valid = bool(1 <= len(attempts) <= 4 and len(attempts) == len(summary_requests)
                         and (summary_record or {}).get("run_id") == compacted.get("run_id"))
    seen, prior_lanes = set(), {}
    for index, attempt in enumerate(attempts):
        entry = attempt.get("compaction") or {}
        lane = summary_lane(summary_requests[index]) if index < len(summary_requests) else None
        value = entry.get("lanes", {}).get(lane, {})
        identity = (entry.get("request_no"), entry.get("attempt_no"), lane)
        lineage_valid = lineage_valid and (
            lane in {"H", "R"} and entry.get("attempt_no") in {1, 2}
            and entry.get("request_no") == compaction.get("request_no") and identity not in seen
            and value.get("summary_request_sha256") == digest(summary_requests[index]))
        candidate_body = None
        try:
            candidate_body = attempt["result"]["text"]
            if entry.get("provenance_phase") == "request_fixed":
                # The immutable response receipt now keeps the input provenance,
                # before parsing/activation creates a candidate. Bind its body to
                # the actual transport response, then verify the candidate below.
                sends = [event for event in observer.request_events
                         if event["turn"] == pressure_turn and event["purpose"] == "summary"
                         and event["request_sha256"] == digest(summary_requests[index])]
                outputs = [response for response in observer.response_outputs
                           if len(sends) == 1 and response["request_id"] == sends[0]["request_id"]]
                response = parse_response({"status": "completed", "output": outputs[0]["output"]}) if len(outputs) == 1 else None
                lineage_valid = lineage_valid and (
                    response is not None and value.get("body") == prior_lanes.get(lane, {}).get("body")
                    and candidate_body == response["text"]
                    and attempt["result"]["output_items"] == outputs[0]["output"]
                    and attempt["result"]["raw_response"]["output"] == outputs[0]["output"]
                    and bool(attempt.get("attempt_id"))
                    and attempt.get("operation_id") == entry.get("step_id"))
            else:
                lineage_valid = lineage_valid and candidate_body == value.get("body")
        except (KeyError, TypeError, ValueError, IndexError, ModelError):
            lineage_valid = False
        seen.add(identity)
        binding = native_binding(observer, summary_requests[index])
        lineage_valid = lineage_valid and binding["verified"]
        if entry.get("attempt_no") == 2:
            previous = prior_lanes.get(lane, {})
            observed = summary_context(observer, summary_requests[index])
            candidate = observed.get("lanes", {}).get(lane, {}).get("candidate_input", {})
            lineage_valid = lineage_valid and bool(previous) and (
                candidate.get("body") == previous.get("body")
                and candidate.get("summary_id") == previous.get("summary_id")
                and candidate.get("checkpoint_id")
                and candidate == value.get("candidate_input")
                and not binding["input_records"])
        prior_lanes[lane] = {**value, "body": candidate_body}
    ref = active.get("history_summary") or {}
    last_h = compaction.get("lanes", {}).get("H", {})
    proofs["summary_saved_and_activated"] = bool(
        lineage_valid and compaction.get("status") == "used" and compaction.get("confirmed_checkpoint_id")
        and ref.get("summary_id") == last_h.get("summary_id") and ref.get("checkpoint_id")
        and all(last_h.get(key) == prior_lanes.get("H", {}).get(key)
                for key in ("summary_id", "summary_request_sha256", "body", "candidate_input"))
        and active.get("version_id") != (before.get("active_context") or {}).get("version_id"))
    bindings = [native_binding(observer, request) for request in summary_requests if summary_lane(request) == "H"]
    source_items = [source for binding in bindings if binding["verified"] for source in binding["input_records"]]
    prior_runs = {state.get("run_id") for state in states[:pressure_turn - 1] if state.get("run_id")}
    proofs["all_prior_turns_selected"] = bool(prior_runs) and all(
        any(source.get("run_id") == run_id for source in source_items)
        for run_id in prior_runs)
    proofs["pressure_has_no_current_summary"] = bool(summary_requests) and all(
        summary_lane(request) == "H" for request in summary_requests)
    proofs["completed_registered_without_summary"] = len(states) >= pressure_turn - 1 and all(
        state.get("summary_calls") == 0
        and (state.get("active_context") or {}).get("current_run_id") is None
        and any(turn.get("run_id") == state.get("run_id") for turn in
                (state.get("active_context") or {}).get("turns", []))
        for state in states[:pressure_turn - 1])
    first_run = states[source_turn - 1].get("run_id") if len(states) >= source_turn else None
    selected = [source for source in source_items if source.get("run_id") == first_run]
    admitted = (old_record or {}).get("admitted_input", [])
    old_tools = [item for item in admitted if item.get("type") == "function_call_output"]
    def original_selected(item):
        return any(source.get("item") == item for source in selected)
    proofs["old_record_selected"] = bool(old_tools) and all(original_selected(item) for item in old_tools)
    old_user_items = [item for item in admitted if item.get("role") == "user"]
    proofs["old_user_requirements_selected"] = bool(old_user_items) and all(original_selected(item) for item in old_user_items)
    following = [r for r in observer.requests.get(follow_up_turn, []) if not is_summary_request(r)]
    h_spans = [span for span in active.get("summary_spans", [])
               if span.get("lane") == "H" and span.get("summary_id") == ref.get("summary_id")]
    message = compacted.get("input", [])[h_spans[0]["index"]] if len(h_spans) == 1 else None
    try:
        from lineage_agent.agent.compaction import _message
        proofs["summary_saved_and_activated"] &= message == _message("H", last_h, last_h["body"])
    except (KeyError, TypeError, ValueError):
        proofs["summary_saved_and_activated"] = False
    proofs["following_uses_saved_version"] = bool(
        following and message and following[0].get("input") == compacted.get("input", []) + [{"role": "user", "content": prompts[follow_up_turn - 1]}]
        and following[0]["input"][0] == message)
    proofs["old_tool_originals_removed"] = bool(following and old_tools and all(item not in following[0].get("input", []) for item in old_tools))
    proofs["old_user_originals_removed"] = bool(following and old_user_items and all(item not in following[0].get("input", []) for item in old_user_items))
    # Ordinary appends now change the active version, so compare the stable saved
    # summary identity rather than requiring an unchanged overall view version.
    proofs["following_keeps_selected_version"] = bool(
        len(states) >= follow_up_turn and states[follow_up_turn - 1].get("summary_calls") == 0
        and (states[follow_up_turn - 1].get("active_context") or {}).get("history_summary") == active.get("history_summary")
        and ref)
    proofs["no_recent_tool_reads"] = bool(len(states) >= follow_up_turn and all(state.get("tool_calls") == 0 for state in states[mechanism["no_tools_from_turn"] - 1:]))
    proofs["original_record_unchanged"] = unchanged_record is True
    proofs["first_turn_requested_files_read"] = any(event["turn"] == source_turn and event["name"] == "read_text_file"
        and event["outcome"] == "succeeded" and event.get("path") == mechanism["source_path"] for event in observer.tool_events)
    proofs["first_turn_background_read"] = any(event["turn"] == source_turn and event["name"] == "read_text_file"
        and event["outcome"] == "succeeded" and event.get("path") == mechanism["background_path"]
        and event.get("bytes_read") == mechanism["background_bytes"] for event in observer.tool_events)
    process = evidence(observer.output_dir / "mechanism.json", "mechanism", "in-memory actual SDK comparisons")
    atomic_write_json(observer.output_dir / "mechanism.json", {
        "checks": proofs, "first_run_id": first_run, "summary_source_ids": [item.get("source_ref") for item in selected],
        "old_tool_item_sha256": [digest(item) for item in old_tools],
        "summary_request_sha256": [digest(item) for item in summary_requests],
        "following_request_sha256": [digest(item) for item in following],
        "semantic_boundary": "Recent answers require separate human review for restated facts; no semantic pass is inferred."})
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
        with patch.object(runtime, "create_client", side_effect=observer.client_factory(runtime.create_client)), \
                patch.object(runtime, "send_prepared_response",
                             side_effect=capture_summary_context(observer, runtime.send_prepared_response)):
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
                compact_metrics, compact_path = turn_compact_evidence(observer, state, index, settings)
                current["metrics"].update(compact_metrics)
                current["process_evidence"].append(evidence(compact_path, "compact", "actual local summary/recall evidence"))
                answer = (state.get("result") or {}).get("text")
                if isinstance(answer, str):
                    answer_path = output / f"turn-{index:02d}-answer.md"
                    atomic_write_text(answer_path, answer)
                    current["answer_evidence"] = evidence(answer_path, "answer", f"Agent run {state['run_id']}")
                    current["answer_sha256"] = sha256_file(answer_path)
                process = {"run_id": state["run_id"], "session_id": sid, "checkpoint_id": current["checkpoint_id"],
                           "conversation_id": state.get("conversation_id"), "record_files": state.get("record_files", {}),
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
                for lane, selected in compaction.get("lanes", {}).items():
                    if lane in {"H", "R"} and isinstance(selected.get("body"), str):
                        summary_path = output / f"turn-{index:02d}-summary-{lane}.txt"
                        atomic_write_text(summary_path, selected["body"])
                        current["process_evidence"].append(evidence(
                            summary_path, "summary", "actual summary candidate body; activation is checked separately"))
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
        elif mechanism.get("kind") == "history_recall":
            result["coverage"] = history_coverage(observer, states, mechanism)
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
        elif observer and mechanism.get("kind") == "history_recall":
            result["coverage"] = history_coverage(observer, states, mechanism)
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
