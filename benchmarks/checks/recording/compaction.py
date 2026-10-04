"""recording / compaction behavior contracts."""


import asyncio
import json
from benchmarks.checks.support.events import check_group, observe_harness
from benchmarks.checks.support.runtime import call, message, response
from benchmarks.checks.support.summary_traces import BODY, CHECK_DIR, DERIVED, exercise
from lineage_agent.recording.trace_data import TraceSanitizer
from uuid import uuid4

async def main():
    with check_group("recording.compaction.summary_trace_equivalence"):
        for lane in ("H", "R"):
            ordinary = await exercise(label=f"metadata-only {lane} summary and derived input", lane=lane)
            disabled = await exercise(enabled=False, label=f"{lane} tracing disabled", lane=lane)
            upload_failed = await exercise(mode="connection", label=f"{lane} upload failure", lane=lane)
            assert ordinary == disabled == upload_failed
    with check_group("recording.compaction.summary_tool_request"):
        await exercise(reply=response(call("summary-call", arguments=json.dumps({"private": BODY}))), label="unexpected summary tool request omitted")
    with check_group("recording.compaction.summary_incomplete"):
        await exercise(reply=response(message(DERIVED), status="incomplete"), label="incomplete summary still omits body")

    with check_group("recording.compaction.bounded_lineage_provenance"):
        observe_harness("trace_projection")
        sanitizer = TraceSanitizer([], None)
        refs = [{"run_id": str(uuid4()), "checkpoint_id": str(uuid4()), "item_count": 3} for _ in range(300)]
        retry_ref = {"summary_id": str(uuid4()), "checkpoint_id": str(uuid4())}
        recovery = {"input_budget_tokens": 9000, "fixed_tokens": 500,
                    "preserved_tokens": 6500, "available_tokens": 2000, "target_tokens": 800}
        metadata = {"sources": {"added_turns": refs}, "material": BODY, "message": DERIVED,
                    "attempt_no": 2, "candidate_input": {**retry_ref, "document": DERIVED},
                    "recovery_budget": {**recovery, "private": BODY}}
        projected = sanitizer.compaction(metadata)
        assert len(projected["sources"]["added_turns"]) == 32
        assert projected["sources"]["turns_omitted"] == 268
        assert projected["attempt_no"] == 2 and projected["candidate_input"] == retry_ref
        assert projected["recovery_budget"] == recovery
        assert BODY not in json.dumps(projected) and DERIVED not in json.dumps(projected)
        payload = sanitizer.cap_payload({"inputs": {"metadata": projected}, "outputs": {"content_omitted": True}})
        assert len(json.dumps(payload).encode()) < 65536 and len(refs) == 300
        # A message that imitates the display marker has no application provenance.
        plain = sanitizer.model_inputs("m", [{"role": "assistant", "content": DERIVED}], [])
        assert DERIVED in json.dumps(plain) and "summary_content_omitted" not in plain
        spans = [{"index": index, "lane": "R", "summary_id": str(uuid4()), "run_id": str(uuid4()),
                  "document": BODY} for index in range(300)]
        active = {"summary_spans": spans}
        projected_active = sanitizer.active_context(active)
        assert len(projected_active["summary_spans"]) == 32
        assert projected_active["summary_span_count"] == 300
        assert projected_active["summary_spans_omitted"] == 268
        assert BODY not in json.dumps(projected_active)
        masked = sanitizer.model_inputs("m", [{"role": "assistant", "content": DERIVED} for _ in spans], [],
                                        active_context=active)
        assert DERIVED not in json.dumps(masked) and masked["summary_items_omitted"] == 300
    print("PASS bounded lineage and explicit provenance; no prefix-based suppression")
    print(f"4 compaction tracing groups passed (8 trace harness exercises); artifacts={CHECK_DIR}")

def execute():
    asyncio.run(main())
