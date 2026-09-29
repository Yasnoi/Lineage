"""Task/turn and review contract validation independent of presentation."""
from typing import Any

STATUSES = {"completed", "failed", "blocked", "not_run"}
VERDICTS = {"passed", "failed", "pending_review"}
COVERAGE_STATUSES = {"verified", "gap", "pending_review"}


def research_units(result: dict) -> list[dict]:
    return [unit for unit in result["units"] if unit.get("execution_mode") == "real_model"]


def _available(value: Any, label: str) -> None:
    if not isinstance(value, dict) or value.get("availability") not in {
        "known", "unknown", "not_applicable"
    } or "value" not in value:
        raise ValueError(f"invalid Research {label} availability")
    if value["availability"] != "known" and (value["value"] is not None or not value.get("reason")):
        raise ValueError(f"invalid missing Research {label}")


def _trace_turn(unit: dict, turn: dict) -> None:
    tracing = turn.get("tracing")
    if tracing is None:  # Historical v3 records have no transport observation.
        return
    if not isinstance(tracing, dict) or type(tracing.get("requested")) is not bool or not isinstance(tracing.get("traces"), list):
        raise ValueError("invalid tracing observation")
    identities = set()
    for trace in tracing["traces"]:
        identifier = trace.get("agent_run_id")
        if not isinstance(identifier, str) or not identifier or identifier in identities:
            raise ValueError("invalid trace Agent identity")
        identities.add(identifier)
        if turn["run_id"]["availability"] == "known" and identifier != turn["run_id"]["value"]:
            raise ValueError("trace belongs to another Agent run")
        binding = trace.get("binding", {})
        if (binding.get("case_id") != unit["case_id"] or binding.get("turn") != turn["turn"]
                or unit.get("sample_run_id") and binding.get("sample_run_id") != unit["sample_run_id"]):
            raise ValueError("trace belongs to another sample or turn")
        if type(trace.get("enabled")) is not bool or trace["enabled"] and not tracing["requested"]:
            raise ValueError("unrequested tracing")
        if trace.get("trace_id") != (identifier if trace["enabled"] else None):
            raise ValueError("trace root differs from Agent identity")
        if trace.get("upload_state") not in {"disabled", "configuration_error", "pending", "submission_completed", "incomplete"}:
            raise ValueError("invalid trace submission status")
        spans, counts = trace.get("spans"), trace.get("counts")
        if not isinstance(spans, list) or not isinstance(counts, dict):
            raise ValueError("invalid trace counts")
        flags = ("queued", "attempted", "submitted", "failed", "dropped")
        if any(type(counts.get(key)) is not int or counts[key] < 0 for key in (*flags, "generated", "unconfirmed")):
            raise ValueError("invalid trace count value")
        span_ids = set()
        for span in spans:
            if (not span.get("span_id") or span["span_id"] in span_ids or span.get("trace_id") != identifier
                    or any(type(span.get(key)) is not bool for key in flags)):
                raise ValueError("invalid trace span")
            span_ids.add(span["span_id"])
        if counts["generated"] != len(spans) or any(counts[key] != sum(span[key] for span in spans) for key in flags):
            raise ValueError("trace counts differ from observations")
        if counts["unconfirmed"] != sum(not (s["submitted"] or s["failed"] or s["dropped"]) for s in spans):
            raise ValueError("trace unconfirmed count differs")
        if trace["upload_state"] == "submission_completed" and (
                not trace.get("closed") or not trace.get("shutdown_confirmed") or trace.get("warning_codes")
                or counts["submitted"] != counts["generated"]):
            raise ValueError("false trace submission completion")


def validate_research_result(result: dict) -> None:
    """Validate only real-model extensions; old contract examples stay valid.

The common validator may call this function, so it deliberately does not call
the common validator itself. File existence belongs to evidence collection or
review binding, not historical result parsing.
"""
    for unit in research_units(result):
        if result.get("schema_version") != "3" or unit.get("category") != "research":
            raise ValueError("real Research requires schema 3 research category")
        if not unit.get("case_id") or not unit.get("definition_version"):
            raise ValueError("Research case identity is missing")
        if type(unit.get("repeat")) is not int or unit["repeat"] < 1:
            raise ValueError("invalid Research repetition")
        for name in ("model", "configuration", "materials", "rubric"):
            _available(unit.get("conditions", {}).get(name), name)
        turns = unit.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError("Research needs a complete planned turn list")
        for number, turn in enumerate(turns, 1):
            if turn.get("turn") != number or turn.get("execution_status") not in STATUSES:
                raise ValueError("invalid or reordered Research turn")
            for name in ("run_id", "session_id", "checkpoint_id", "agent_outcome"):
                _available(turn.get(name), name)
                if turn[name]["availability"] == "known" and not isinstance(turn[name]["value"], str):
                    raise ValueError("Research turn identity or outcome must be text")
            for field in ("answer_evidence", "process_evidence"):
                items = turn.get(field)
                if field == "answer_evidence" and isinstance(items, dict):
                    items = [items]
                if not isinstance(items, list):
                    raise ValueError("Research turn evidence must be an object or list")
                for item in items:
                    if not isinstance(item, dict) or item.get("availability") not in {
                        "available", "missing", "unknown"
                    } or not item.get("path"):
                        raise ValueError("invalid Research turn evidence")
            if turn["execution_status"] == "not_run" and turn["run_id"]["availability"] == "known":
                raise ValueError("unstarted Research turn cannot claim an Agent run")
            if turn["execution_status"] == "completed" and turn["run_id"]["availability"] != "known":
                raise ValueError("completed Research turn requires an observed Agent run")
            _trace_turn(unit, turn)
        coverage = unit.get("coverage", [])
        if not isinstance(coverage, list):
            raise ValueError("invalid Research coverage")
        names = set()
        for item in coverage:
            if not isinstance(item, dict) or not item.get("name") or item["name"] in names:
                raise ValueError("duplicate or missing Research mechanism identity")
            names.add(item["name"])
            if item.get("status") not in COVERAGE_STATUSES or not item.get("reason"):
                raise ValueError("invalid Research mechanism observation")
        if unit.get("verdict") != "pending_review" and not unit.get("review"):
            raise ValueError("Research quality verdict needs an independent human review")
        if unit.get("verdict") == "passed":
            review = unit.get("review")
            if not isinstance(review, dict) or review.get("complete") is not True or not review.get("reviewer"):
                raise ValueError("Research pass requires a completed named human review")
            if any(item["status"] != "verified" for item in coverage):
                raise ValueError("Research pass cannot hide a mechanism gap or pending review")
            if any(turn["execution_status"] != "completed" for turn in turns):
                raise ValueError("Research pass requires all planned turns")
            if any(dimension["verdict"] != "passed" for dimension in unit["dimensions"]):
                raise ValueError("Research pass requires all applicable dimensions")
