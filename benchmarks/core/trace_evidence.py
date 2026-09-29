"""Bounded read-only verification of existing production traces; no model calls."""
from copy import deepcopy
import logging
from queue import Empty, Queue
import threading
import time
from urllib.parse import urlsplit

from benchmarks.core.files import atomic_write_json, utc_now
from benchmarks.core.metadata import _evidence
from lineage_agent.recording.tracing import _TraceSettings, _create_client, _WorkerLogFilter


REMOTE_FIELDS = ["id", "name", "run_type", "trace_id", "parent_run_id", "start_time", "end_time",
                 "extra", "app_path", "prompt_tokens", "completion_tokens", "total_tokens"]


def compare_trace(expected, runs):
    """Project only locally expected identities; remote text is never retained."""
    by_id = {str(run.id): run for run in runs}
    spans = expected["spans"]
    mismatches, observations = [], []
    for span in spans:
        identifier = span["span_id"]
        remote = by_id.get(identifier)
        if remote is None:
            mismatches.append({"span_id": identifier, "reason": "missing_span"})
            continue
        metadata = (getattr(remote, "extra", None) or {}).get("metadata", {})
        differences = []
        for key, value in (("trace_id", span["trace_id"]), ("parent_run_id", span["parent_run_id"]),
                           ("run_type", span["run_type"])):
            actual = getattr(remote, key, None)
            if (str(actual) if actual is not None else None) != value:
                differences.append(key)
        for key, value in span["metadata"].items():
            if metadata.get(key) != value:
                differences.append("metadata." + key)
        if getattr(remote, "end_time", None) is None:
            differences.append("end_time")
        usage = {}
        for key, remote_key in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"),
                                ("total_tokens", "total_tokens")):
            value = getattr(remote, remote_key, None)
            usage[key] = value if type(value) is int and value >= 0 else None
            if key in span.get("usage", {}) and usage[key] != span["usage"][key]:
                differences.append("usage." + key)
        if differences:
            mismatches.append({"span_id": identifier, "reason": "fields_differ", "fields": differences})
        observations.append({"span_id": identifier, "identity_matches": not differences, "usage": usage})
    expected_ids = {span["span_id"] for span in spans}
    if set(by_id) - expected_ids:
        mismatches.append({"reason": "unexpected_spans", "count": len(set(by_id) - expected_ids)})
    root = by_id.get(expected["trace_id"])
    url = getattr(root, "url", None) if root is not None else None
    if isinstance(url, str):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            url = None
    else:
        url = None
    return {"trace_id": expected["trace_id"], "status": "verified" if spans and not mismatches else "incomplete",
            "expected_spans": len(spans), "observed_spans": len(runs), "mismatches": mismatches,
            "spans": observations, "url": url}


def query_traces(traces, settings, *, timeout_seconds=30, client_factory=_create_client):
    """At most three query rounds and one shared deadline, including stalled DNS."""
    timeout_seconds = max(0.0, min(30.0, timeout_seconds))
    started = time.monotonic()
    stop, updates = threading.Event(), Queue()
    initial = {trace["trace_id"]: {"trace_id": trace["trace_id"], "status": "not_checked",
                                  "expected_spans": len(trace["spans"]), "url": None}
               for trace in traces}
    deadline = started + timeout_seconds

    def query():
        client = None
        logger = logging.getLogger("langsmith.client")
        log_filter = _WorkerLogFilter()
        logger.addFilter(log_filter)
        current = deepcopy(initial)
        try:
            client = client_factory(settings)
            for round_no in range(1, 4):
                for expected in traces:
                    identifier = expected["trace_id"]
                    if stop.is_set() or time.monotonic() >= deadline:
                        return
                    if current[identifier]["status"] == "verified":
                        continue
                    current[identifier] = {**current[identifier], "status": "query_pending", "query_rounds": round_no}
                    updates.put(("progress", deepcopy(current)))
                    try:
                        runs = list(client.list_runs(trace_id=identifier, select=REMOTE_FIELDS,
                                                     limit=len(expected["spans"]) + 1))
                        current[identifier] = compare_trace(expected, runs)
                    except Exception:
                        # No endpoint exception text or remote metadata escapes this boundary.
                        current[identifier] = {**current[identifier], "status": "query_failed"}
                    current[identifier]["query_rounds"] = round_no
                    updates.put(("progress", deepcopy(current)))
                if all(item["status"] == "verified" for item in current.values()):
                    break
                if round_no < 3 and stop.wait(min(1, max(0, deadline - time.monotonic()))):
                    return
        except Exception:
            updates.put(("progress", {key: {**value, "status": "query_failed"} for key, value in current.items()}))
        finally:
            if client is not None:
                try:
                    client.close(timeout=0)
                except Exception:
                    pass
            logger.removeFilter(log_filter)
            updates.put(("done", None))

    if not traces or timeout_seconds <= 0:
        return list(initial.values()), 0.0
    threading.Thread(target=query, name="benchmark-trace-verification", daemon=True).start()
    latest = initial
    try:
        while time.monotonic() < deadline:
            try:
                kind, value = updates.get(timeout=min(0.05, max(0.001, deadline - time.monotonic())))
            except Empty:
                continue
            if kind == "done":
                break
            latest = value
    finally:
        stop.set()
    snapshots = [({**item, "status": "query_failed", "reason": "query_deadline"}
                  if item["status"] == "query_pending" else item) for item in latest.values()]
    return snapshots, time.monotonic() - started


def verify_sample_traces(unit, spec, directory, connection, *, timeout_seconds=30):
    """Attach durable diagnostics without changing Agent outcomes or verdicts."""
    traces = [trace for turn in unit.get("turns", []) for trace in turn.get("tracing", {}).get("traces", [])
              if trace.get("enabled") and trace.get("trace_id")]
    tracing = spec["tracing"]
    settings = _TraceSettings(tracing["endpoint"], connection["LANGSMITH_API_KEY"],
                              tracing["project"], connection.get("LANGSMITH_WORKSPACE_ID", ""))
    remote, elapsed = query_traces(traces, settings, timeout_seconds=timeout_seconds)
    payload = {"sample_run_id": spec["run_id"], "batch_id": spec["batch_id"], "checked_at": utc_now(),
               "duration_seconds": elapsed, "query_limit_rounds": 3, "timeout_seconds": min(30, timeout_seconds),
               "traces": remote, "basis": "Remote identity, completed spans and known usage; filtered traces are not full requests."}
    path = directory / "trace-remote.json"
    atomic_write_json(path, payload)
    unit["evidence"].append(_evidence("remote_trace_verification", path, "read-only LangSmith query"))
    unit["tracing"] = {"requested": True, "remote": remote, "verification_seconds": elapsed,
                       "submission_states": [trace["upload_state"] for trace in traces]}
