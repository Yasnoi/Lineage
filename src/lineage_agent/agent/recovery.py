"""Validation of an exact, published conversation state (never reconstruction by inference)."""

import re
from contextlib import closing

from lineage_agent.agent.record_journal import RecordJournal
from lineage_agent.agent.record_codec import encode_record
from lineage_agent.agent.state import ActiveContext, AgentState
from lineage_agent.archive import ArchiveError, ArchiveStore


_SOURCE = re.compile(r"history://record/([A-Za-z0-9_-]{1,128})/(0|[1-9][0-9]*)\Z")


def validate_snapshot_state(state: dict, archive: ArchiveStore, view_id: str) -> None:
    _validate_state(state, archive, view_id, terminal=True)


def validate_execution_state(state: dict, archive: ArchiveStore, view_id: str) -> None:
    _validate_state(state, archive, view_id, terminal=False)
    execution = state.get("execution")
    if (type(execution) is not dict or not execution.get("point_id")
            or execution.get("phase") not in {"run_ready", "model", "request_ready", "response_saved",
                "tools_prepare", "tool_plan_ready", "tool_result_saved", "group_ready", "prepare_compaction",
                "summarize_history", "summarize_current", "validate_candidate", "activate_context", "finalizing"}):
        raise ArchiveError("archive_integrity_error")
    try:
        point = archive.execution(execution["point_id"])
        if (point["run_id"] != state["run_id"] or point["phase"] != execution["phase"]
                or encode_record(point["state"]) != encode_record(state)):
            raise ValueError("execution state mismatch")
        phase = execution["phase"]
        if phase in {"request_ready", "response_saved"}:
            operation = archive.operation(execution["operation_id"])
            kind, payload = operation["kind"], operation["payload"]
            if (operation["run_id"] != state["run_id"] or kind not in {"model", "summary"}
                    or payload["return_node"] != execution["return_node"]
                    or payload["return_node"] not in ({"model"} if kind == "model" else
                                                      {"summarize_history", "summarize_current"})):
                raise ValueError("request relationship mismatch")
            if kind == "model" and payload["request"]["input"] != state["input"]:
                raise ValueError("request context mismatch")
            if phase == "response_saved" and sum(a.get("result") is not None for a in
                    archive.attempts(operation["operation_id"])) != 1:
                raise ValueError("missing saved response")
        if phase in {"tool_plan_ready", "tool_result_saved", "group_ready"}:
            operation = archive.operation(execution["operation_id"])
            plan = operation["payload"]
            index = execution["tool_index"]
            if (operation["run_id"] != state["run_id"] or operation["kind"] != "tool_plan"
                    or plan["calls"] != state["result"]["tool_calls"]
                    or len(plan["operation_ids"]) != len(plan["calls"])
                    or type(index) is not int or not 0 <= index <= len(plan["calls"])):
                raise ValueError("tool plan mismatch")
            for position, (identity, call) in enumerate(zip(plan["operation_ids"], plan["calls"], strict=True)):
                child = archive.operation(identity)
                if (child["run_id"] != state["run_id"] or child["kind"] != "tool"
                        or child["payload"] != {"call": call, "index": position,
                            "plan_id": operation["operation_id"], "view_id": plan["view_id"]}):
                    raise ValueError("tool relationship mismatch")
                if position < index and archive.operation_result(identity) is None:
                    raise ValueError("missing processed result")
            if phase != "group_ready" and index >= len(plan["calls"]):
                raise ValueError("invalid next tool")
    except ArchiveError:
        raise
    except (KeyError, TypeError, ValueError, IndexError):
        raise ArchiveError("archive_integrity_error") from None


def _validate_state(state: dict, archive: ArchiveStore, view_id: str, *, terminal: bool) -> None:
    """Verify stored relationships without regenerating input, summaries or usage."""
    from lineage_agent.agent.runtime import _input_association_error, _input_shape_error

    def require(condition):
        if not condition:
            raise ArchiveError("archive_integrity_error")

    try:
        require(type(state) is dict and AgentState.__required_keys__.issubset(state))
        archive.verify_view(view_id)
        view = archive.view(view_id)
        require(state["conversation_id"] == archive.conversation_id and state["resource_view_id"] == view_id)
        require(state["record_files"] == view["record_files"] and state["history_scopes"] == view["history_scopes"])
        require(state["run_id"] in view["record_files"])
        require(state["record_sequence"] == view["record_files"][state["run_id"]]["record_count"])
        items, boundary, risks, active = (state[name] for name in ("input", "history_end", "context_risks", "active_context"))
        require(type(items) is list and all(type(item) is dict for item in items))
        require(type(boundary) is int and 0 <= boundary <= len(items))
        require(type(risks) is list and len(risks) == len(items))
        require(_input_shape_error(items[:boundary]) is None and _input_association_error(items[:boundary]) is None)
        require(type(active) is dict and ActiveContext.__required_keys__.issubset(active))
        require(type(active["version_id"]) is str and bool(active["version_id"]))
        require(type(active["turn_start"]) is int and 0 <= active["turn_start"] <= boundary)
        require(active["current_run_id"] in (None, state["run_id"]))
        require(type(active["item_refs"]) is list and len(active["item_refs"]) == len(items))
        require(type(active["turns"]) is list and type(active["groups"]) is list)
        require(type(active["summary_spans"]) is list)
        require(state["stop_reason"] in ("completed", "error", "max_model_calls", "context_budget_exceeded")
                or not terminal and state["stop_reason"] is None)
        for name in ("model_calls", "summary_calls", "tool_requests", "tool_calls", "record_revision",
                     "record_admitted_through"):
            require(type(state[name]) is int and state[name] >= 0)
        for name in ("usage_totals", "summary_usage_totals"):
            usage = state[name]
            require(type(usage) is dict and set(usage) == {"input_tokens", "output_tokens", "total_tokens", "missing_usage_calls"})
            require(all(value is None or type(value) is int and value >= 0 for value in usage.values()))
            require(type(usage["missing_usage_calls"]) is int)
        seen_turns = set()
        for turn in active["turns"]:
            require(type(turn) is dict and turn["run_id"] in view["record_files"] and turn["run_id"] not in seen_turns)
            seen_turns.add(turn["run_id"])
            require(all(type(turn[key]) is int and turn[key] >= 0 for key in ("item_count", "group_count", "record_end")))
            require(turn["record_end"] == view["record_files"][turn["run_id"]]["record_count"])
        require(active["turn_start"] == int(active["history_summary"] is not None) + sum(turn["item_count"] for turn in active["turns"]))
        if active["current_run_id"] is None:
            require(active["turn_start"] == boundary and not active["groups"])
        else:
            require(state["run_id"] not in seen_turns)
        for group in active["groups"]:
            require(type(group) is dict and type(group["start"]) is int and type(group["end"]) is int)
            require(active["turn_start"] <= group["start"] < group["end"] <= len(items) and type(group["closed"]) is bool)
        references = [active["history_summary"], active["run_summary"]]
        indices = set()
        for span in active["summary_spans"]:
            index, ref = span["index"], span["reference"]
            require(type(index) is int and 0 <= index < len(items) and index not in indices)
            require(span["summary_id"] == ref["summary_id"] and span["lane"] == ref["lane"])
            require(active["item_refs"][index] is None)
            summary = archive.summary(ref["summary_id"], view_id)
            require(span["run_id"] == summary["run_id"] and span["lane"] in ("H", "R"))
            content = items[index].get("content")
            require(type(content) is str and summary["body"] in content and ref["history_ref"] in content)
            references.append(ref)
            indices.add(index)
        if active["history_summary"] is not None:
            require(active["history_summary"].get("lane") == "H")
            require(any(span["index"] == 0 and span["reference"] == active["history_summary"]
                        for span in active["summary_spans"]))
        if active["run_summary"] is not None:
            require(active["current_run_id"] == state["run_id"] and active["run_summary"].get("lane") == "R")
            matches = [span for span in active["summary_spans"] if span["reference"] == active["run_summary"]]
            require(len(matches) == 1 and matches[0]["run_id"] == state["run_id"]
                    and matches[0]["index"] >= active["turn_start"])
        for ref in references:
            if ref is not None:
                summary = archive.summary(ref["summary_id"], view_id)
                require(all(ref.get(key) == summary.get(key) for key in ("lane", "history_ref", "ranges")))
        span_by_index = {span["index"]: span for span in active["summary_spans"]}
        start = int(active["history_summary"] is not None)
        ownership = [(turn["run_id"], turn["item_count"]) for turn in active["turns"]]
        if active["current_run_id"] is not None:
            ownership.append((state["run_id"], len(items) - active["turn_start"]))
        for run_id, count in ownership:
            for index in range(start, start + count):
                ref = active["item_refs"][index]
                if ref is None:
                    span = span_by_index.get(index)
                    require(span is not None and span["lane"] == "R" and span["run_id"] == run_id)
                else:
                    match = _SOURCE.fullmatch(ref) if type(ref) is str else None
                    require(match is not None and match.group(1) == run_id)
            start += count
        require(start == len(items))
        with closing(RecordJournal(archive.directory / "history", create=False)) as journal:
            records = {}
            for index, ref in enumerate(active["item_refs"]):
                if ref is None:
                    require(index in indices)
                    continue
                match = _SOURCE.fullmatch(ref) if type(ref) is str else None
                require(match is not None)
                run_id, sequence = match.group(1), int(match.group(2))
                require(run_id in view["record_files"] and sequence < view["record_files"][run_id]["record_count"])
                if run_id not in records:
                    records[run_id] = journal.records(run_id, view["record_files"][run_id])
                require(encode_record({"item": records[run_id][sequence].get("item")}) == encode_record({"item": items[index]}))
            for run_id, watermark in view["record_files"].items():
                receipts = [r for r in state["record_batch"] if r.get("run_id") == run_id]
                journal.verify(run_id, watermark, receipts=receipts)
    except ArchiveError:
        raise
    except Exception:
        raise ArchiveError("archive_integrity_error") from None
