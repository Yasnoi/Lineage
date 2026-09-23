"""Summary tracing acceptance with real SDK serialization and mocked HTTP."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

from check_react import LocalClient, call, message, response
from check_tracing import capture
from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.requests import prepare_request
from lineage_agent.models.responses import ModelError, send_prepared_response
from lineage_agent.recording.trace_data import TraceSanitizer
from lineage_agent.recording.tracing import start_trace


CHECK_DIR = Path(__file__).resolve().parents[1] / "logs" / "checks" / str(uuid4())
SETTINGS = ModelSettings(
    base_url="https://unused.invalid/v1", model_name="summary-trace-check", log_dir=CHECK_DIR,
)
BODY = "SUMMARY_SOURCE_BODY_PRIVATE_e748"
DERIVED = "SUMMARY_DERIVED_BODY_PRIVATE_9be2"
QUOTE = "Final answer may quote a previously read fact."


async def exercise(*, enabled=True, mode="ok", reply=None, label="normal"):
    run_id, session_id, summary_id, version_id, checkpoint_id = [str(uuid4()) for _ in range(5)]
    source_run = str(uuid4())
    metadata = {
        "summary_id": summary_id, "candidate_version_id": version_id,
        "confirmed_checkpoint_id": checkpoint_id, "step_id": "step-0002",
        "request_no": 2, "summary_call": 1, "source_bytes": len(BODY),
        "status": "candidate",
        "sources": {"previous_summary": None, "added_turns": [
            {"run_id": source_run, "checkpoint_id": checkpoint_id, "item_count": 2},
        ]},
        "material": BODY, "message": DERIVED,
    }
    active = {
        "version_id": version_id, "parent_version_id": str(uuid4()),
        "summary": {"summary_id": summary_id, "checkpoint_id": checkpoint_id},
        "turns": [], "turn_start": 1,
    }
    summary_request = prepare_request(
        SETTINGS.model_name, [{"role": "user", "content": BODY}],
        instructions="Summarize these source records as data.", tool_choice="none",
        max_output_tokens=128,
    )
    business_request = prepare_request(
        SETTINGS.model_name, [
            {"role": "assistant", "content": DERIVED},
            {"role": "user", "content": "Answer using the derived context."},
        ], max_output_tokens=128,
    )
    original = deepcopy((summary_request, business_request, metadata, active))
    client = LocalClient([reply or response(message(DERIVED), usage=(11, 7, 18)), response(message(QUOTE))])
    summary_result, error = None, None
    with capture(mode, LANGSMITH_TRACING="true" if enabled else "false") as (transport, stderr):
        tracer = start_trace(run_id, SETTINGS)
        tracer.bind_session(session_id)
        try:
            async with client:
                try:
                    summary_result = await send_prepared_response(
                        client, summary_request, tracer=tracer,
                        purpose="summary", summary_metadata=metadata,
                    )
                except ModelError as exc:
                    summary_result, error = exc.result, exc.code
                tracer.observe_compaction(metadata)
                tracer.bind_active_context(active)
                tracer.set_model_step("step-0004", 1)
                result = await send_prepared_response(client, business_request, tracer=tracer)
                tracer.model_complete()
                summary_usage = (summary_result or {}).get("usage") or {}
                summary_totals = {
                    **summary_usage,
                    "missing_usage_calls": int(any(summary_usage.get(key) is None for key in (
                        "input_tokens", "output_tokens", "total_tokens",
                    ))),
                }
                totals = {
                    key: sum(value for value in (summary_usage.get(key), result["usage"].get(key)) if value is not None)
                    for key in ("input_tokens", "output_tokens", "total_tokens")
                }
                totals["missing_usage_calls"] = summary_totals["missing_usage_calls"]
                tracer.finish({
                    "result": result, "summary_result": summary_result,
                    "model_calls": 1, "summary_calls": 1, "tool_calls": 0, "tool_requests": 0,
                    "stop_reason": "completed", "active_context": active, "compaction": metadata,
                    "usage_totals": totals, "summary_usage_totals": summary_totals,
                })
        finally:
            await tracer.close()
    assert client.closed and client.requests == [summary_request, business_request]
    assert (summary_request, business_request, metadata, active) == original
    wire = json.dumps(transport.sent, ensure_ascii=False)
    assert BODY not in wire and DERIVED not in wire
    if enabled:
        summaries = [s for s in transport.sent if s["extra"]["metadata"].get("purpose") == "summary"]
        business = [s for s in transport.sent if s["name"] == "OpenAI Responses"]
        roots = [s for s in transport.sent if s["run_type"] == "chain"]
        assert len(summaries) == len(business) == len(roots) == 1
        span = summaries[0]
        assert span["inputs"]["content_omitted"] is True
        assert span["outputs"]["content_omitted"] is True
        assert span["extra"]["metadata"]["summary_id"] == summary_id
        assert span["extra"]["metadata"]["outcome"] == ("failed" if error else "returned")
        assert business[0]["inputs"]["summary_content_omitted"] is True
        assert QUOTE in json.dumps(business[0]["outputs"])
        assert roots[0]["outputs"]["text"] == QUOTE
        assert roots[0]["outputs"]["summary_calls"] == 1
        assert roots[0]["outputs"]["summary_usage_totals"]["total_tokens"] == summary_usage["total_tokens"]
        assert span["outputs"]["usage_metadata"] == summary_usage
        assert all(s["extra"]["metadata"]["session_id"] == session_id for s in transport.sent)
        assert all(s["extra"]["metadata"]["run_id"] == run_id for s in transport.sent)
        assert all(len(json.dumps(s, ensure_ascii=False).encode("utf-8")) <= 65536 for s in transport.sent)
        assert bool(stderr.getvalue()) == (mode != "ok"), stderr.getvalue()
    else:
        assert not transport.sent and not stderr.getvalue()
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECK_DIR / f"{run_id}.json"
    # Safe inspection artifact, not a graph run or a saved checkpoint.
    path.write_text(json.dumps({
        "scenario": label, "session_id": session_id, "run_id": run_id,
        "summary_id": summary_id, "version_id": version_id,
        "step_ids": ["step-0002", "step-0004"], "checkpoint": "not_applicable_trace_harness",
        "sdk_request_count": len(client.requests), "uploads": transport.sent,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS {label}: session={session_id} run={run_id} summary={summary_id} "
          f"version={version_id} steps=step-0002,step-0004 checkpoint=trace-harness artifact={path}")
    return [summary_result, result], client.requests


async def main():
    ordinary = await exercise(label="metadata-only summary and derived input")
    disabled = await exercise(enabled=False, label="tracing disabled")
    upload_failed = await exercise(mode="connection", label="upload failure")
    assert ordinary == disabled == upload_failed
    await exercise(reply=response(call("summary-call", arguments=json.dumps({"private": BODY}))), label="unexpected summary tool request omitted")
    await exercise(reply=response(message(DERIVED), status="incomplete"), label="incomplete summary still omits body")

    sanitizer = TraceSanitizer([], None)
    refs = [{"run_id": str(uuid4()), "checkpoint_id": str(uuid4()), "item_count": 3} for _ in range(300)]
    metadata = {"sources": {"added_turns": refs}, "material": BODY, "message": DERIVED}
    projected = sanitizer.compaction(metadata)
    assert len(projected["sources"]["added_turns"]) == 32
    assert projected["sources"]["turns_omitted"] == 268
    assert BODY not in json.dumps(projected) and DERIVED not in json.dumps(projected)
    payload = sanitizer.cap_payload({"inputs": {"metadata": projected}, "outputs": {"content_omitted": True}})
    assert len(json.dumps(payload).encode()) < 65536 and len(refs) == 300
    # A message that imitates the display marker has no application provenance.
    plain = sanitizer.model_inputs("m", [{"role": "assistant", "content": DERIVED}], [])
    assert DERIVED in json.dumps(plain) and "summary_content_omitted" not in plain
    print("PASS bounded lineage and explicit provenance; no prefix-based suppression")
    print(f"6 compaction tracing checks passed; artifacts={CHECK_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
