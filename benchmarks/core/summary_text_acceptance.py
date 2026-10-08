"""Freeze historical contexts for a committed, current-only text-input acceptance.

Old requests are read as data, never executed. All provenance is local evidence.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from benchmarks.core.compact_evidence import _confirmed_records
from benchmarks.core.files import ROOT, atomic_write_json, sha256_file, utc_now
from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.summarization import prepare_summary_request
from lineage_agent.models.summary_input import INPUT_FORMAT, record_details, serialize_conversation


DESIGN = "text-conversation-v1"
VARIANTS = ("high", "max")
LIMITS = {"sdk_requests": 40, "execution_seconds": 2400, "request_seconds": 120, "repeats": 2}
CANDIDATE_FILES = (
    "src/lineage_agent/models/summarization.py", "src/lineage_agent/models/summary_input.py",
    "src/lineage_agent/agent/context.py", "src/lineage_agent/agent/compaction.py",
    "benchmarks/core/summary_prompt_comparison.py", "benchmarks/core/summary_text_acceptance.py",
    "benchmarks/core/compact_evidence.py",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def context_settings(case):
    return {key: case[key] for key in ("context_budget_tokens", "safety_margin_tokens")}


def build_request(case, effort):
    if effort not in VARIANTS:
        raise ValueError("unknown text acceptance effort")
    settings = ModelSettings(model_name=case["model_name"], base_url="https://unused.invalid",
                             reasoning_effort=effort, summary_history_max_output_tokens=16384,
                             summary_run_max_output_tokens=16384)
    rendered = serialize_conversation(case["source_input"], item_details=case["item_details"])
    return prepare_summary_request(settings, rendered, lane=case["lane"], attempt_no=1)


def validate(manifest):
    if (manifest.get("design") != DESIGN or manifest.get("limits") != LIMITS
            or not 1 <= len(manifest["cases"]) <= 10 or manifest.get("input_format") != INPUT_FORMAT):
        raise ValueError("invalid text acceptance design")
    identities = set()
    for case in manifest["cases"]:
        if (not case["id"] or case["id"] in identities or case["lane"] not in ("H", "R")
                or case["model_name"] != manifest["settings"]["model_name"]
                or case.get("design") != DESIGN or case.get("previous_summary_is_fixed") is not True
                or digest(case["source_input"]) != case["source_input_sha256"]
                or digest(case["item_details"]) != case["item_details_sha256"]
                or type(case["context_budget_tokens"]) is not int or case["context_budget_tokens"] <= 0
                or type(case["safety_margin_tokens"]) is not int or case["safety_margin_tokens"] < 0
                or digest(context_settings(case)) != case["context_settings_sha256"]
                or set(case["requests"]) != set(VARIANTS)):
            raise ValueError("invalid frozen text source")
        identities.add(case["id"])
        for effort in VARIANTS:
            request = build_request(case, effort)
            if (request != case["requests"][effort]
                    or digest(request) != case["request_sha256"][effort]):
                raise ValueError("candidate or frozen text request changed")


def _remember(files, path):
    files[Path(path).resolve().relative_to(ROOT).as_posix()] = sha256_file(path)


def _source_details(case, files):
    """Recover only trusted labels, checking original items against confirmed files."""
    source = ROOT / case["source"]
    evidence = json.loads(source.read_text(encoding="utf-8"))
    items = case["request"]["input"]
    details = [{"kind": "scope"}]
    if "summary_requests" in evidence:
        matches = [row for row in evidence["summary_requests"] if row["input"] == items and row["lane"] == case["lane"]]
        if len(matches) != 1:
            raise ValueError("historical input not uniquely bound")
        row = matches[0]
        selected = row["source_metadata"]["lanes"][case["lane"]]
        records = _confirmed_records(row["source_metadata"], selected)
        sidecar = row["summary_input_metadata"]
        spans = {span["index"]: span for span in sidecar["derived_spans"]}
        if len(sidecar["item_refs"]) != len(items) or sidecar["item_refs"][0] is not None:
            raise ValueError("historical sidecar alignment differs")
        for index, item in enumerate(items[1:], 1):
            ref = sidecar["item_refs"][index]
            if ref is not None:
                record = records[ref]
                if record["item"] != item:
                    raise ValueError("historical original differs")
                details.append(record_details(record, background=case["lane"] == "R" and record["kind"] == "input"))
            else:
                span = spans[index]
                if hashlib.sha256(item["content"].encode("utf-8")).hexdigest() != span["body_sha256"]:
                    raise ValueError("fixed previous summary differs")
                details.append({"kind": "summary", "lane": span["lane"],
                                "run_id": span["run_id"] if span["lane"] == "R" else None})
        for entry in selected["source_files"]:
            _remember(files, Path(entry["path"]))
    else:
        # The four compatibility probes have explicitly documented inserted
        # summaries. Do not infer derived identity from ordinary message text.
        probe = case["id"].removeprefix("probe-")
        if case["id"] not in {"probe-H1", "probe-H2", "probe-R1", "probe-R2"}:
            raise ValueError("unknown compatibility probe")
        manifest_path = source.parent / "probe-manifest.json"
        provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
        _remember(files, manifest_path)
        origin = provenance["sources"]["R" if probe.startswith("R") else probe]
        journal = ROOT / origin["journal"]
        _remember(files, journal)
        records = _confirmed_records({"snapshot": {"record_files": {origin["run_id"]: origin["watermark"]}}},
                                     {"source_files": [{"run_id": origin["run_id"], "path": str(journal)}]})
        cursor = -1
        for index, item in enumerate(items[1:], 1):
            if index == 1 and probe in {"H2", "R2"}:
                details.append({"kind": "summary", "lane": case["lane"],
                                "run_id": origin["run_id"] if case["lane"] == "R" else None})
                continue
            matches = [record for record in records.values() if record["sequence"] > cursor and record.get("item") == item]
            if len(matches) != 1:
                raise ValueError("probe original not uniquely bound")
            record = matches[0]
            cursor = record["sequence"]
            details.append(record_details(record, background=case["lane"] == "R" and record["kind"] == "input"))
    return details


def freeze(sources: Path, destination: Path):
    from benchmarks.core.summary_prompt_comparison import verify_sources
    old = json.loads(sources.read_text(encoding="utf-8"))
    verify_sources(old)
    if len(old["cases"]) != 10 or sum(case["lane"] == "H" for case in old["cases"]) != 4:
        raise ValueError("expected the ten fixed historical contexts")
    files = deepcopy(old["source_files"])
    _remember(files, sources)
    result = {"design": DESIGN, "input_format": INPUT_FORMAT, "limits": dict(LIMITS),
              "frozen_at": utc_now(), "settings": {**old["settings"],
                  "summary_history_max_output_tokens": 16384, "summary_run_max_output_tokens": 16384},
              "source_files": files, "candidate_files": {name: sha256_file(ROOT / name) for name in CANDIDATE_FILES},
              "historical_settings": old["settings"], "cases": []}
    for source in old["cases"]:
        case = {key: deepcopy(source[key]) for key in ("id", "lane", "context_budget_tokens", "safety_margin_tokens", "source")}
        case.update(design=DESIGN, model_name=source["request"]["model"],
                    source_input=deepcopy(source["request"]["input"]),
                    item_details=_source_details(source, files), previous_summary_is_fixed=True,
                    historical_request_sha256=digest(source["request"]))
        case["source_input_sha256"] = digest(case["source_input"])
        case["item_details_sha256"] = digest(case["item_details"])
        case["context_settings_sha256"] = digest(context_settings(case))
        case["requests"] = {effort: build_request(case, effort) for effort in VARIANTS}
        case["request_sha256"] = {effort: digest(request) for effort, request in case["requests"].items()}
        result["cases"].append(case)
    validate(result)
    destination.mkdir(parents=True, exist_ok=False)
    atomic_write_json(destination / "manifest.json", result)
