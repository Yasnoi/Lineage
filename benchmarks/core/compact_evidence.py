"""Local Compact evidence from actual requests; no semantic success inference."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from benchmarks.core.contracts import known, unknown
from benchmarks.core.evidence import digest
from benchmarks.core.files import atomic_write_json
from lineage_agent.models.budget import check_context_budget
from lineage_agent.models.summarization import is_summary_request, summary_budget_settings
from lineage_agent.agent.record_codec import decode_record
from lineage_agent.agent.records import record_projection


HISTORY_CHECKS = (
    "complete_chain_read_once", "all_chain_results_inline", "current_prefix_compacted",
    "target_record_covered", "target_original_removed", "history_search_used",
    "history_read_used", "history_result_matches_original", "no_original_file_reread",
)


def summary_context(observer, request):
    """Metadata captured at the application send boundary, never SDK input."""
    return deepcopy(getattr(observer, "summary_contexts", {}).get(digest(request), {}))


def capture_summary_context(observer, original):
    async def send(client, request, **kwargs):
        if kwargs.get("purpose") == "summary":
            identity = digest(request)
            contexts = getattr(observer, "summary_contexts", None)
            if contexts is None:
                contexts = observer.summary_contexts = {}
            if identity in contexts:
                raise ValueError("duplicate summary request identity")
            contexts[identity] = deepcopy(kwargs.get("summary_metadata") or {})
        return await original(client, request, **kwargs)
    return send


def _confirmed_records(metadata, selected):
    """Read only the checkpoint-confirmed byte prefix; never create or heal files."""
    records = {}
    paths = {}
    for entry in selected.get("source_files", []):
        run_id, path = entry["run_id"], Path(entry["path"])
        if run_id in paths and paths[run_id] != path:
            raise ValueError("conflicting original paths")
        paths[run_id] = path
    for run_id, path in paths.items():
        boundary = metadata["snapshot"]["record_files"][run_id]
        if (boundary.get("format_version") != 1 or type(boundary.get("byte_end")) is not int
                or boundary["byte_end"] < 0 or type(boundary.get("record_count")) is not int):
            raise ValueError("invalid original boundary")
        with path.open("rb") as handle:
            content = handle.read(boundary["byte_end"])
        if (len(content) != boundary["byte_end"]
                or hashlib.sha256(content).hexdigest() != boundary["prefix_sha256"]):
            raise ValueError("original prefix does not match confirmation")
        lines = content.splitlines(keepends=True)
        if len(lines) != boundary["record_count"]:
            raise ValueError("original count differs")
        for sequence, line in enumerate(lines):
            record = decode_record(line)
            ref = f"history://record/{run_id}/{sequence}"
            if (not line.endswith(b"\n") or record.get("source_ref") != ref
                    or record.get("run_id") != run_id or record.get("sequence") != sequence):
                raise ValueError("original identity differs")
            records[ref] = record
    return records


def native_binding(observer, request):
    """Join actual native input to trusted sidecars and immutable original files."""
    metadata = summary_context(observer, request)
    result = {"verified": False, "reason": "summary source binding unavailable", "records": [],
              "input_records": [], "metadata": metadata}
    try:
        lane = summary_lane(request)
        selected = metadata["lanes"][lane]
        if lane != metadata.get("lane") or selected["summary_request_sha256"] != digest(request):
            raise ValueError("request lineage differs")
        sidecar = metadata["summary_input_metadata"]
        items, refs = request["input"], sidecar["item_refs"]
        if (not isinstance(refs, list) or len(items) != len(refs)
                or len(sidecar["context_risks"]) != len(items)):
            raise ValueError("native sidecar alignment differs")
        records = _confirmed_records(metadata, selected)
        seen = set()
        for item, ref in zip(items, refs, strict=True):
            if ref is None:
                continue
            if ref in seen or ref not in records or records[ref].get("item") != item:
                raise ValueError("native original differs")
            seen.add(ref)
            result["input_records"].append(records[ref])
        selected_refs = selected.get("source_refs", [])
        if metadata.get("attempt_no") != 2 and not all(
                ref in records and (records[ref].get("item") is None or ref in seen) for ref in selected_refs):
            raise ValueError("selected native original missing")
        if metadata.get("attempt_no") != 2:
            expected = [ref for ref in selected_refs if records[ref].get("item") is not None]
            if [ref for ref in refs if ref in selected_refs] != expected:
                raise ValueError("selected native order differs")
        spans = sidecar["derived_spans"]
        positions = set()
        for span in spans:
            index = span["index"]
            if (type(index) is not int or not 0 <= index < len(items) or index in positions
                    or refs[index] is not None or type(items[index].get("content")) is not str
                    or not isinstance(span.get("reference"), dict)
                    or hashlib.sha256(items[index]["content"].encode("utf-8")).hexdigest() != span["body_sha256"]):
                raise ValueError("derived body binding differs")
            positions.add(index)
        if {index for index, ref in enumerate(refs) if ref is None} != {0, *positions}:
            raise ValueError("unbound native input")
        if metadata.get("attempt_no") == 2:
            candidate = selected["candidate_input"]
            if (seen or len(spans) != 1 or items[spans[0]["index"]]["content"] != candidate["body"]
                    or spans[0]["reference"].get("summary_id") != candidate["summary_id"]
                    or spans[0]["reference"].get("checkpoint_id") != candidate["checkpoint_id"]):
                raise ValueError("candidate-only retry differs")
        else:
            previous = selected.get("previous", [])
            if [span["reference"] for span in spans] != previous:
                raise ValueError("previous summary ownership differs")
        ranges = selected["scope"]["ranges"]
        for bounds in ranges:
            if not all(f"history://record/{bounds['run_id']}/{sequence}" in records
                       for sequence in range(bounds["start"], bounds["end"])):
                raise ValueError("covered original unavailable")
        result["records"] = [record for ref, record in records.items() if covered(ref, ranges)]
        result.update(verified=True, reason="native objects match confirmed files and derived identities")
    except (KeyError, IndexError, TypeError, ValueError, OSError, UnicodeError):
        result["records"] = []
        result["input_records"] = []
    return result


def summary_lane(request):
    if not is_summary_request(request):
        return None
    from lineage_agent.models.summarization import summary_request_lane
    return summary_request_lane(request)


def _json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None
    return value


def protocol_shape(item):
    """Keep endpoint shape diagnostics without message or reasoning contents."""
    if not isinstance(item, dict):
        return {"value_type": type(item).__name__}
    result = {"keys": sorted(item), "field_types": {key: type(value).__name__ for key, value in item.items()}}
    for key in ("type", "status", "phase"):
        if key in item:
            result[key] = item[key] if item[key] is None or isinstance(item[key], str) else "invalid_type"
    for key in ("content", "summary"):
        if isinstance(item.get(key), list):
            result[f"{key}_parts"] = [
                {"keys": sorted(part), "type": part.get("type")}
                if isinstance(part, dict) else {"value_type": type(part).__name__}
                for part in item[key]]
    return result


def source_has_text(source, text):
    return text in record_projection(source)[0]


def covered(source_ref, ranges):
    if not isinstance(source_ref, str):
        return False
    try:
        prefix, run_id, sequence = source_ref.rsplit("/", 2)
        return prefix == "history://record" and any(
            item["run_id"] == run_id and item["start"] <= int(sequence) < item["end"] for item in ranges)
    except (KeyError, TypeError, ValueError):
        return False


def _tool_pairs(requests, final_items=None):
    """Unique actual protocol results, including a terminal unsent tool batch."""
    calls, outputs = {}, {}
    inputs = []
    for request in requests:
        if is_summary_request(request):
            continue
        inputs.append(request.get("input", []))
    if final_items is not None:
        inputs.append(final_items)
    for items in inputs:
        for item in items:
            if item.get("type") == "function_call":
                calls.setdefault(item.get("call_id"), item)
            elif item.get("type") == "function_call_output":
                outputs.setdefault(item.get("call_id"), item)
    return [(call, outputs[call_id]) for call_id, call in calls.items() if call_id in outputs]


def _pair_sources(requests):
    return {call["call_id"]: "sdk_input" for call, _ in _tool_pairs(requests)}


def _matching_recall_pages(reads, targets, text):
    """Validate exact projection offsets, including evidence spanning pages."""
    originals = {source["source_ref"]: record_projection(source)[0] for source in targets}
    spans = {}
    for call, output in reads:
        ref = (_json(call.get("arguments")) or {}).get("source_ref")
        envelope = _json(output.get("output")) or {}
        page = _json(envelope.get("observation", {}).get("raw_result")) or {}
        start, end = page.get("start"), page.get("end")
        original = originals.get(ref)
        if (not isinstance(original, str) or page.get("source_ref") != ref
                or type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(original)
                or page.get("text") != original[start:end]):
            return False
        spans.setdefault(ref, []).append((start, end))
    for ref, intervals in spans.items():
        start = originals[ref].find(text)
        if start < 0:
            continue
        through = start
        for low, high in sorted(intervals):
            if low <= through:
                through = max(through, high)
        if through >= start + len(text):
            return True
    return False


def history_coverage(observer, states, mechanism):
    requests = [item for turn in sorted(observer.requests) for item in observer.requests[turn]]
    summaries = [request for request in requests if is_summary_request(request)]
    bindings = [native_binding(observer, request) for request in summaries if summary_lane(request) == "R"]
    r_sources = [source for binding in bindings if binding["verified"] for source in binding["records"]]
    targets = [source for source in r_sources if source_has_text(source, mechanism["target_path"])
               and source_has_text(source, mechanism["source_text"])]
    target_refs = {source.get("source_ref") for source in targets}
    events = [event for state in states for event in state.get("compaction_events", [])]
    # Candidate events alone cannot prove activation; only the confirmed use does.
    used = [event for event in events if event.get("status") == "used"]
    ranges = [part for event in used for scope in event.get("history_scopes", []) for part in scope.get("ranges", [])]
    target_covered = any(covered(ref, ranges) for ref in target_refs if isinstance(ref, str))
    pairs = _tool_pairs(requests, [item for state in states for item in state.get("input", [])])
    pair_sources = _pair_sources(requests)
    reads = [(call, output) for call, output in pairs if call.get("name") == "read_text_file"]
    paths = [(_json(call.get("arguments")) or {}).get("path") for call, _ in reads]
    all_inline = bool(reads) and all((_json(output.get("output")) or {}).get("observation", {}).get("mode") == "inline"
                                   for _, output in reads)
    recalls = [(call, output) for call, output in pairs if call.get("name") in {"history_read", "history_search"}]
    searches = [(call, output) for call, output in recalls if call.get("name") == "history_search"
                and (_json(output.get("output")) or {}).get("ok") is True]
    history_reads = [(call, output) for call, output in recalls if call.get("name") == "history_read"
                     and (_json(output.get("output")) or {}).get("ok") is True
                     and (_json(call.get("arguments")) or {}).get("source_ref") in target_refs]
    matched = _matching_recall_pages(history_reads, targets, mechanism["source_text"])
    # Find a business request after compaction with history tools; the target original
    # must be absent while the derived/recall form is permitted.
    old_call_ids = {call["call_id"] for call, _ in reads
                    if (_json(call.get("arguments")) or {}).get("path") == mechanism["target_path"]}
    active_inputs = [request.get("input", []) for request in requests if not is_summary_request(request)]
    active_inputs.extend(state.get("input", []) for state in states)
    removed = any(not any(item.get("type") == "function_call_output" and item.get("call_id") in old_call_ids
                          for item in items)
                  and any(item.get("name") in {"history_read", "history_search"} for item in items)
                  for items in active_inputs)
    proofs = {
        "complete_chain_read_once": paths == mechanism["chain_paths"],
        "all_chain_results_inline": len(reads) == len(mechanism["chain_paths"]) and all_inline,
        "current_prefix_compacted": any((event.get("lanes", {}).get("R") or {}).get("summary_id") for event in used),
        "target_record_covered": target_covered,
        "target_original_removed": bool(old_call_ids) and removed,
        "history_search_used": any(any(ref in json.dumps(_json(output.get("output")), ensure_ascii=False)
                                       for ref in target_refs) for _, output in searches),
        "history_read_used": bool(history_reads),
        "history_result_matches_original": matched,
        "no_original_file_reread": bool(paths) and len(paths) == len(set(paths)),
    }
    documents = [entry for entry in observer.response_outputs if entry["purpose"] == "summary"]
    key_retained = any(mechanism["record_key"] in json.dumps(entry["output"], ensure_ascii=False) for entry in documents)
    details = {"checks": proofs, "target_source_refs": sorted(ref for ref in target_refs if isinstance(ref, str)),
               "calls": [{"call": call, "output": output,
                          "evidence_source": pair_sources.get(call["call_id"], "final_state_input")}
                         for call, output in recalls],
               "target_original_call_ids": sorted(old_call_ids),
               "summary_record_key_retained": key_retained,
               "summary_omission_condition": "requires_semantic_review",
               "semantic_boundary": "Exact recall is mechanical evidence; answer use and omitted meaning require source review."}
    path = observer.output_dir / "mechanism.json"
    atomic_write_json(path, details)
    evidence = {"kind": "mechanism", "path": str(path.absolute()), "source": "actual SDK inputs, final Agent protocol and confirmed Compact events",
                "availability": "available"}
    return [{"name": name, "status": "verified" if passed else "gap",
             "reason": "observed original/source/recall binding satisfied" if passed else "required binding not observed",
             "evidence": [evidence]} for name, passed in proofs.items()]


def turn_compact_evidence(observer, state, turn, settings):
    requests = observer.requests.get(turn, [])
    summaries = [request for request in requests if is_summary_request(request)]
    business = [request for request in requests if not is_summary_request(request)]
    events = deepcopy(state.get("compaction_events", []))
    current_call_ids = {item.get("call_id") for response in observer.response_outputs
                        if response["turn"] == turn and response["purpose"] == "business"
                        for item in response["output"] if item.get("type") == "function_call"}
    pairs = [(call, output) for call, output in _tool_pairs(requests, state.get("input", []))
             if call.get("call_id") in current_call_ids]
    pair_sources = _pair_sources(requests)
    recalls = {name: [(call, output) for call, output in pairs if call.get("name") == name]
               for name in ("history_read", "history_search", "artifact_read", "artifact_search")}
    metrics = {"compact_events": known(events), "record_storage": known(state["record_metrics"])
               if state.get("record_metrics") else unknown("confirmed record metrics unavailable"),
               "summary_lanes": known({lane: sum(summary_lane(request) == lane for request in summaries) for lane in ("H", "R")}),
               "recall_calls": known({name: len(items) for name, items in recalls.items()}),
               "recall_return_bytes": known({name: sum(len(output["output"].encode("utf-8")) for _, output in items)
                                               for name, items in recalls.items()})}
    metrics["compact_attempts"] = known(len({(event.get("request_no"), event.get("attempt_no")) for event in events}))
    metrics["compact_successes"] = known(sum(event.get("status") == "used" for event in events))
    native = []
    for request in summaries:
        binding = native_binding(observer, request)
        metadata = binding["metadata"]
        sidecar = metadata.get("summary_input_metadata", {})
        lane = summary_lane(request)
        measured = check_context_budget(request, sidecar.get("context_risks", []),
                                        summary_budget_settings(settings, lane))
        saved_budget = metadata.get("lanes", {}).get(lane, {}).get("summary_budget", {})
        budget_fields = ("decision", "required_tokens", "transparent_tokens", "opaque_items",
                         "opaque_risk_tokens", "output_reserve_tokens", "safety_margin_tokens", "limit_tokens")
        native.append({"lane": lane, "request_sha256": digest(request), "input": deepcopy(request["input"]),
                       "instructions": request.get("instructions"),
                       "summary_input_metadata": sidecar,
                       "source_binding": {key: value for key, value in binding.items()
                                          if key not in {"records", "input_records", "metadata"}},
                       "source_metadata": {key: metadata.get(key) for key in
                                           ("snapshot", "lanes", "request_no", "attempt_no")},
                       "summary_budget": measured,
                       "budget_matches_preflight": all(measured.get(key) == saved_budget.get(key)
                                                       for key in budget_fields)})
    if business:
        # Isolate the actual schema overhead. The previous Run can contain opaque
        # reasoning whose admission risks are not part of this transport observer.
        request = {**business[0], "input": []}
        without = {**request, "tools": [tool for tool in request.get("tools", [])
                                        if tool.get("name") not in {"history_read", "history_search"}]}
        full = check_context_budget(request, [None] * len(request.get("input", [])), settings)
        old = check_context_budget(without, [None] * len(request.get("input", [])), settings)
        metrics["history_tool_fixed_budget_overhead"] = known(full["required_tokens"] - old["required_tokens"])
        metrics["history_tool_fixed_budget_overhead_basis"] = known("actual request fields with identical empty input")
    else:
        metrics["history_tool_fixed_budget_overhead"] = unknown("no actual business request")
    path = observer.output_dir / f"turn-{turn:02d}-compact-evidence.json"
    # This local review artifact is intentionally separate from production tracing.
    atomic_write_json(path, {"run_id": state["run_id"], "events": events,
        "business_protocol_shapes": [
            {"request_id": entry.get("request_id"), "output": [protocol_shape(item) for item in entry["output"]]}
            for entry in observer.response_outputs if entry["turn"] == turn and entry["purpose"] == "business"],
        "summary_requests": native,
        "summary_outputs": [entry for entry in observer.response_outputs if entry["turn"] == turn and entry["purpose"] == "summary"],
        "recalls": {name: [{"call": call, "output": output,
                            "evidence_source": pair_sources.get(call["call_id"], "final_state_input")}
                           for call, output in items] for name, items in recalls.items()},
        "record_metrics": state.get("record_metrics"), "record_files": state.get("record_files", {})})
    return metrics, path
