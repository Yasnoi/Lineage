"""Reusable fixtures and evidence assertions for compaction."""


import json
from benchmarks.checks.support.events import observe_turn as observe_actual_turn
from benchmarks.checks.support.runtime import (
    LocalClient,
    SETTINGS as BASE_SETTINGS,
    message,
    outputs,
    read_events,
    response,
)
from benchmarks.checks.support.session_faults import resources, closed_resources
from benchmarks.checks.support.sessions import saved_state
from benchmarks.core.files import ROOT
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from lineage_agent.agent import compaction, context, graph, runtime
from lineage_agent.agent.checkpointing import CheckpointError
from lineage_agent.agent.sessions import SessionManager
from lineage_agent.agent.state import CompactionStatus
from lineage_agent.models.summarization import SUMMARY_INSTRUCTIONS
from typing import get_args
from unittest.mock import patch
from uuid import uuid4

CHECK_DIR = ROOT / "logs" / "checks" / str(uuid4())

SETTINGS = replace(
    BASE_SETTINGS, log_dir=CHECK_DIR, context_budget_tokens=24000,
    max_output_tokens=256, context_safety_margin_tokens=64,
    summary_max_output_tokens=512, summary_max_bytes=1200,
)

RUNS = []

PRIVATE = "COMPACTION_EXCEPTION_PRIVATE_01e3"

KEYS = ("requirements", "facts", "proposals", "corrections", "open_items")

def is_summary(request):
    return request.get("instructions") == SUMMARY_INSTRUCTIONS

def material_for(request):
    assert is_summary(request)
    assert "tools" not in request and request["tool_choice"] == "none"
    assert request["stream"] is False and request["store"] is False
    assert request["truncation"] == "disabled"
    assert len(request["input"]) == 1 and request["input"][0]["role"] == "user"
    return json.loads(request["input"][0]["content"])

def source_ids(material):
    result = []

    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get("source_id"), str):
                result.append(value["source_id"])
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(material)
    return list(dict.fromkeys(result))

def summary_document(request, text="Earlier confirmed facts remain available."):
    ids = source_ids(material_for(request))
    assert ids, "summary must carry explicit stable source identifiers"
    result = {key: [] for key in KEYS}
    result["facts"] = [{"text": text, "source_ids": ids[:1]}]
    return result

def summary_response(request, *, text="Earlier confirmed facts remain available.", usage=(20, 5, 25)):
    return response(message(json.dumps(summary_document(request, text), ensure_ascii=False)), usage=usage)

@contextmanager
def captured_measurements():
    observations = []

    def wrap(original, purpose):
        def measure(request, risks, settings, **kwargs):
            before, annotations = deepcopy(request), deepcopy(risks)
            result = original(request, risks, settings, **kwargs)
            assert before == request and annotations == risks
            observations.append({"purpose": purpose, "request": before, "budget": deepcopy(result)})
            return result
        return measure

    with patch.object(graph, "check_context_budget", side_effect=wrap(graph.check_context_budget, "business")), \
            patch.object(compaction, "check_context_budget", side_effect=wrap(compaction.check_context_budget, "compaction")), \
            patch.object(context, "check_context_budget", side_effect=wrap(context.check_context_budget, "selection")):
        yield observations

class Client(LocalClient):
    """Capture actual kwargs, then choose summary/normal responses independently."""

    def __init__(self, replies=(), *, summary=None):
        super().__init__(replies)
        self.summary = summary or summary_response

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if is_summary(kwargs):
            result = self.summary(kwargs)
        else:
            try:
                result = next(self.replies)
            except StopIteration:
                raise AssertionError("unexpected extra business model request") from None
        if isinstance(result, Exception):
            raise result
        return result

def audit(turn, client, scenario, observations=None):
    observe_actual_turn(turn)
    state = turn.state
    events = read_events(state)
    business = [item for item in client.requests if not is_summary(item)]
    summaries = [item for item in client.requests if is_summary(item)]
    assert state["model_calls"] == len(business)
    assert state["summary_calls"] == len(summaries)
    assert len(state["input"]) == len(state["context_risks"])
    assert 0 <= state["history_end"] <= len(state["input"])
    assert all(item["run_id"] == state["run_id"] and item["session_id"] == turn.session_id for item in events)
    starts = [item for item in events if item["event"] == "model.started"]
    summary_starts = [item for item in events if item["event"] == "summary.started"]
    if state["log_complete"]:
        assert len(starts) == len(business) and len(summary_starts) == len(summaries)
        assert events[-1]["event"] == "run.ended"
    for request, event in zip(business, starts):
        results = [item for item in request["input"] if item.get("type") == "function_call_output"]
        assert event["input_tool_result_call_ids"] == [item["call_id"] for item in results]
        assert len(results) == len({item["call_id"] for item in results})
        assert request["truncation"] == "disabled"
        if event["final_answer_call"]:
            assert request["instructions"] == graph.FINAL_ANSWER_INSTRUCTIONS
            assert request["tool_choice"] == "none"
        else:
            assert "instructions" not in request and "tool_choice" not in request
        assert not {"remaining_model_calls", "read_budget", "read_budget_error"} & request.keys()
    for request in summaries:
        material_for(request)
    assert PRIVATE not in json.dumps(events, ensure_ascii=False)
    assert PRIVATE not in (state.get("error") or "")
    compaction = state.get("compaction") or {}
    if compaction:
        assert compaction["status"] in get_args(CompactionStatus)
        assert {"request_no", "status", "reason"} <= compaction.keys()
        # Exercise the contract on actual graph states, not constructed examples.
        if compaction["status"] in {"preparing", "candidate", "pending_save", "confirmed", "used"}:
            assert {"summary_id", "request_options", "old_version_id"} <= compaction.keys()
        if compaction["status"] in {"candidate", "pending_save", "confirmed", "used"}:
            assert {"sources", "removed_items", "removed_turns", "document", "message",
                    "candidate_version_id", "summary_bytes", "summary_call"} <= compaction.keys()
        if compaction["status"] in {"pending_save", "confirmed", "used"}:
            assert {"summary_checkpoint_id", "candidate_budget"} <= compaction.keys()
        if compaction["status"] in {"confirmed", "used"}:
            assert compaction["confirmed_checkpoint_id"]
    active = state["active_context"]
    print(json.dumps({
        "scenario": scenario, "session_id": turn.session_id, "run_id": state["run_id"],
        "checkpoint_id": turn.checkpoint_id, "checkpoint_complete": turn.checkpoint_complete,
        "checkpoint_error": turn.checkpoint_error, "summary_id": compaction.get("summary_id"),
        "version_id": active["version_id"], "compaction_status": compaction.get("status"),
        "compaction_reason": compaction.get("reason"),
        "steps": list(dict.fromkeys(item["step_id"] for item in events if "step_id" in item)),
        "call_ids": [item["call_id"] for item in outputs(state)],
        "model_calls": state["model_calls"], "summary_calls": state["summary_calls"],
        "usage_totals": state["usage_totals"], "summary_usage_totals": state["summary_usage_totals"],
        "log_path": state["log_path"], "fault_injection_observations": observations or [],
    }, ensure_ascii=False))
    RUNS.append(state["run_id"])
    return events

async def run(manager, sid, prompt, replies=(), *, summary=None, scenario, observations=None):
    client = Client(replies, summary=summary)
    with resources() as observed, captured_measurements() as measured, patch.object(runtime, "create_client", return_value=client) as factory:
        turn = await manager.run_turn(sid, prompt)
        observe_actual_turn(turn)
    assert not factory.called or client.closed
    closed_resources(manager, sid, observed)
    if turn.checkpoint_complete:
        assert await saved_state(manager, turn) == turn.state
    audit(turn, client, scenario, observations)
    cursor = 0
    for request in client.requests:
        matching = [(index, item) for index, item in enumerate(measured[cursor:], cursor)
                    if item["request"] == request
                    and item["purpose"] == ("compaction" if is_summary(request) else "business")]
        assert matching and matching[0][1]["budget"]["decision"] == "allow", (
            "actual SDK kwargs were not the same request that passed budget measurement"
        )
        cursor, counted = matching[0]
        cursor += 1
        assert request["max_output_tokens"] == counted["budget"]["output_reserve_tokens"]
    return turn, client

async def seed(*, settings=SETTINGS, manager=None, sid=None, count=4, tools=None, executor=None,
               prefix="seed", prompt_size=2000, answer_size=2500):
    manager = manager or SessionManager()
    sid = sid or manager.create_session(settings, tools=tools, execute_tool=executor)
    turns = []
    for index in range(count):
        prompt = f"OLD_SOURCE_{index}: " + "a" * prompt_size
        turn, client = await run(manager, sid, prompt, [response(message("b" * answer_size))], scenario=f"{prefix}_{index + 1}")
        assert turn.state["stop_reason"] == "completed" and turn.state["summary_calls"] == 0
        assert turn.checkpoint_complete
        if turns:
            assert client.requests[0]["input"] == turns[-1].state["input"] + [{"role": "user", "content": prompt}]
        turns.append(turn)
    return manager, sid, turns

TRIGGER = "CURRENT_QUESTION_MUST_REMAIN: " + "c" * 5000

async def check_turn_slice(manager, turn):
    """Compare fixed-checkpoint selection with the unchanged public archive."""
    saver, sid = manager._checkpointer, turn.session_id
    ref = {"run_id": turn.state["run_id"], "checkpoint_id": turn.checkpoint_id}
    with patch.object(saver, "read_confirmed", wraps=saver.read_confirmed) as reads:
        original = await manager.get_record(sid, ref["run_id"])
    full_reads = reads.await_count
    assert full_reads >= 1
    if turn.state["model_calls"] or turn.state["summary_calls"]:
        assert full_reads > 1
    with patch.object(saver, "read_confirmed", wraps=saver.read_confirmed) as reads:
        selected = await context.read_turn_slice(saver, sid, ref)
    reads.assert_awaited_once_with(sid, turn.checkpoint_id)
    expected = {key: original[key] for key in ("admitted_input", "stop_reason", "error_code", "tool_outcomes")}
    assert selected == expected
    selected["admitted_input"].clear()
    selected["tool_outcomes"]["skipped"] = 999
    assert await context.read_turn_slice(saver, sid, ref) == expected
    assert await manager.get_record(sid, ref["run_id"]) == original

    class CopyForbidden:
        def __deepcopy__(self, memo):
            raise AssertionError("selection copied an unrelated response archive")

    read_confirmed = saver.read_confirmed

    async def observed_snapshot(thread_id, checkpoint_id):
        saved = await read_confirmed(thread_id, checkpoint_id)
        # Change only this newly deserialized observation; never write it back.
        values = saved.checkpoint["channel_values"]
        values["result"] = CopyForbidden()
        values["summary_result"] = CopyForbidden()
        values["compaction"] = CopyForbidden()
        return saved

    with patch.object(saver, "read_confirmed", side_effect=observed_snapshot):
        assert await context.read_turn_slice(saver, sid, ref) == expected
    assert await manager.get_record(sid, ref["run_id"]) == original

    for reader in (context.read_turn_slice, context.read_record):
        for corruption in ("run", "boundary"):
            async def invalid_snapshot(thread_id, checkpoint_id):
                saved = await read_confirmed(thread_id, checkpoint_id)
                values = saved.checkpoint["channel_values"]
                if corruption == "run":
                    values["run_id"] = "different-run"
                else:
                    values["history_end"] = len(values["input"]) + 1
                return saved

            with patch.object(saver, "read_confirmed", side_effect=invalid_snapshot):
                try:
                    await reader(saver, sid, ref)
                except CheckpointError as exc:
                    assert exc.code == "checkpoint_read_error"
                else:
                    raise AssertionError("a corrupt final-turn reference passed validation")
    print(json.dumps({
        "scenario": "turn_slice_matches_full_record", "session_id": sid,
        "run_id": ref["run_id"], "checkpoint_id": turn.checkpoint_id,
        "log_path": turn.state["log_path"], "full_record_reads": full_reads,
        "selection_reads": 1, "record_model_results": len(original["model_results"]),
        "record_summary_results": len(original["summary_results"]),
    }, ensure_ascii=False))
