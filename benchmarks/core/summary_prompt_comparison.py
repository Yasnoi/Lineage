"""Fixed-input summary Instructions comparison; never runs an Agent or its tools.

Explicit freeze/run entry points, independent durable accounting, no judge calls.
Historical prompts are identified by frozen hashes, not a relaxed production classifier.
"""

import argparse
import asyncio
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from benchmarks.core.budgets import RequestBudget, WorkerAbort, read_ledger
from benchmarks.core.files import ROOT, atomic_write_json, atomic_write_text, sha256_file, utc_now
from lineage_agent.config.settings import load_settings
from lineage_agent.models import summarization
from lineage_agent.models.budget import check_context_budget
from lineage_agent.models.responses import ModelError, create_client, send_prepared_response
from lineage_agent.models.summarization import (
    SummaryBudgetSettings, SummaryError, _summary_instructions, parse_summary,
)


LIMITS = {"sdk_requests": 40, "execution_seconds": 1200, "request_seconds": 120, "repeats": 2}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def without_instructions(request):
    return {key: value for key, value in request.items() if key != "instructions"}


def verify_sources(manifest):
    for relative, expected in manifest["source_files"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError("source evidence changed")


def freeze(sources: Path, destination: Path):
    """Freeze both prompts before any comparison requests; refuse overwriting evidence."""
    manifest = json.loads(sources.read_text(encoding="utf-8"))
    verify_sources(manifest)
    manifest["sources_sha256"] = sha256_file(sources)
    manifest["frozen_at"] = utc_now()
    manifest["candidate_source_sha256"] = sha256_file(ROOT / "src/lineage_agent/models/summarization.py")
    for case in manifest["cases"]:
        request = case["request"]
        case["instructions"] = {"A": request["instructions"], "B": _summary_instructions(
            request["max_output_tokens"], case["lane"], 1)}
        case["instruction_sha256"] = {key: digest(value) for key, value in case["instructions"].items()}
        case["risks_sha256"] = digest(case["risks"])
    validate(manifest)
    destination.mkdir(parents=True, exist_ok=False)
    atomic_write_json(destination / "manifest.json", manifest)


def validate(manifest):
    if manifest["limits"] != LIMITS or not 1 <= len(manifest["cases"]) <= 10:
        raise ValueError("invalid comparison limits")
    identities = set()
    for case in manifest["cases"]:
        if case["id"] in identities or not case["id"]:
            raise ValueError("duplicate or empty case identity")
        identities.add(case["id"])
        request = case["request"]
        if (case["lane"] not in ("H", "R") or request["reasoning"] != {"effort": "none"}
                or request["tool_choice"] != "none" or request.get("tools")
                or request["stream"] is not False or request["store"] is not False
                or request["truncation"] != "disabled"
                or request["max_output_tokens"] != (16384 if case["lane"] == "H" else 8192)
                or len(case["risks"]) != len(request["input"])
                or digest(case["risks"]) != case["risks_sha256"]
                or request["model"] != manifest["settings"]["model_name"]
                or digest(request) != case["baseline_request_sha256"]
                or digest(request["input"]) != case["input_sha256"]
                or digest(without_instructions(request)) != case["fixed_request_sha256"]
                or request["instructions"] != case["instructions"]["A"]
                or set(case["instructions"]) != {"A", "B"}
                or any(type(value) is not str or not value.strip()
                       or digest(value) != case["instruction_sha256"][key]
                       for key, value in case["instructions"].items())):
            raise ValueError("invalid frozen request")


def prepare(case, variant):
    request = deepcopy(case["request"])
    request["instructions"] = case["instructions"][variant]
    if digest(without_instructions(request)) != case["fixed_request_sha256"]:
        raise ValueError("comparison changed more than Instructions")
    settings = SummaryBudgetSettings(case["context_budget_tokens"], request["max_output_tokens"],
                                     case["safety_margin_tokens"])
    return request, check_context_budget(request, case["risks"], settings)


def schedule(manifest):
    rows = []
    for repeat in (1, 2):
        for index, case in enumerate(manifest["cases"]):
            # Reverse first/second exposure in the second repeat.
            for variant in (("A", "B") if repeat == 1 else ("B", "A")):
                rows.append({"case_index": index, "case": case["id"], "variant": variant,
                             "repeat": repeat, "status": "not_run", "review_id": uuid4().hex})
    return rows


async def compare(manifest, output: Path, client, *, provenance=None):
    """One serial batch. All failures occupy their planned slot; never retry/resume."""
    validate(manifest)
    ledger = output / "sdk-ledger.jsonl"
    with ledger.open("xb"):
        pass
    identity = str(uuid4())
    budget = RequestBudget(ledger, identity, "summary-instructions", max_requests=40,
                           remaining_seconds=1200, batch_id=identity)
    started = time.monotonic()
    state = {"started_at": utc_now(), "provenance": provenance, "limits": LIMITS,
             "results": schedule(manifest), "stop_reason": None, "requests": 0,
             "seconds": 0, "semantic_review": "pending", "human_review": "pending"}
    active = {}

    def save():
        state["seconds"] = time.monotonic() - started
        try:
            atomic_write_json(output / "results.json", state)
        except (OSError, ValueError) as exc:
            raise WorkerAbort("evidence_write_failed") from exc

    async def capture(sent):
        try:
            body = json.loads(sent.content)
            if body != active["request"] or active["row"]["http_requests"]:
                raise WorkerAbort("http_request_mismatch_or_retry")
            active["row"]["http_requests"] += 1
            atomic_write_json(output / f'{active["prefix"]}-http.json', {"body": body})
        except (OSError, ValueError, KeyError) as exc:
            raise WorkerAbort("http_evidence_failed") from exc

    hooks = client._client.event_hooks.setdefault("request", [])
    hooks.append(capture)
    try:
        save()
        for number, row in enumerate(state["results"], 1):
            case = manifest["cases"][row["case_index"]]
            request, preflight = prepare(case, row["variant"])
            prefix = f'{number:02d}'
            row.update(budget=preflight, request_sha256=digest(request), http_requests=0, file_prefix=prefix)
            atomic_write_json(output / f"{prefix}-request.json", request)
            if preflight["decision"] != "allow":
                row["status"] = "preflight_rejected"
                save()
                continue
            # Persist the complete slot before making an SDK call.
            row["status"] = "ready"
            save()
            event = budget.reserve(number, "summary")
            state["requests"] += 1
            row.update(status="started", request_id=event["request_id"], started_at=event["started_at"])
            save()
            active.update(request=deepcopy(request), row=row, prefix=prefix)
            before = time.monotonic()
            response = None
            try:
                async with asyncio.timeout(min(120, max(0, budget.deadline - before))):
                    response = await send_prepared_response(client, request, purpose="summary",
                        summary_metadata={"lane": case["lane"], "comparison_variant": row["variant"]})
                text = parse_summary(response)
                if row["http_requests"] != 1 or request != active["request"]:
                    raise WorkerAbort("http_request_mismatch_or_retry")
                row.update(status="accepted", text_chars=len(text),
                           response_model=response["raw_response"].get("model"))
                # Opaque IDs permit a first review without displaying prompt assignments.
                atomic_write_text(output / f'{row["review_id"]}.md', text)
            except (ModelError, SummaryError, TimeoutError) as exc:
                if isinstance(exc, ModelError):
                    response = exc.result
                status = getattr(exc.__cause__, "status_code", None)
                row.update(status="failed", error_type=type(exc).__name__,
                           error_code=getattr(exc, "code", "timeout"), http_status=status)
                if status in (401, 403):
                    state["stop_reason"] = "authentication_failed"
            finally:
                row["duration_seconds"] = time.monotonic() - before
                if response is not None:
                    row["usage"] = response.get("usage")
                    row["raw_usage"] = response["raw_response"].get("usage")
                    atomic_write_json(output / f"{prefix}-response.json", response)
                save()
            print(json.dumps({key: row[key] for key in ("case", "variant", "repeat", "status", "duration_seconds")}), flush=True)
            if state["stop_reason"]:
                break
    except WorkerAbort as exc:
        state["stop_reason"] = exc.code
        save()
    except (OSError, ValueError) as exc:
        state["stop_reason"] = "evidence_or_input_failure"
        save()
        raise WorkerAbort(state["stop_reason"]) from exc
    finally:
        hooks.remove(capture)
    events = read_ledger(ledger, 40, batch_id=identity)
    if len(events) != state["requests"]:
        raise WorkerAbort("accounting_mismatch")
    state.update(finished_at=utc_now(), unused_allowance_closed=True)
    save()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("sources", type=Path)
    freeze_parser.add_argument("destination", type=Path)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.sources, args.destination)
        return
    manifest_path = args.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate(manifest)
    verify_sources(manifest)
    if sha256_file(ROOT / "src/lineage_agent/models/summarization.py") != manifest["candidate_source_sha256"]:
        raise ValueError("candidate changed after freeze")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit the verified candidate before real comparison")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    settings = load_settings(ROOT / ".env")
    if any(getattr(settings, key) != manifest["settings"][key] for key in ("model_name", "base_url")):
        raise ValueError("comparison endpoint/model changed")
    settings = replace(settings, timeout_seconds=120)

    async def run():
        async with create_client(settings) as client:
            if client.max_retries != 0:
                raise ValueError("automatic retries must be disabled")
            return await compare(manifest, args.directory, client, provenance={
                "candidate_commit": commit, "manifest_sha256": sha256_file(manifest_path),
                "source_module": str(Path(summarization.__file__).resolve()),
            })
    result = asyncio.run(run())
    verify_sources(manifest)
    print(json.dumps({key: result[key] for key in ("requests", "seconds", "stop_reason")}), flush=True)


if __name__ == "__main__":
    main()
