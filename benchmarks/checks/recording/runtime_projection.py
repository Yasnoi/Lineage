"""recording / runtime_projection behavior contracts."""


import asyncio
import json
from benchmarks.checks.support.access import CloseFailureClient, file_call, run_case
from benchmarks.checks.support.events import check_group
from benchmarks.checks.support.runtime import message, response
from benchmarks.checks.support.tracing import capture, stable
from copy import deepcopy
from lineage_agent.models.responses import parse_response
from lineage_agent.recording.trace_data import TraceSanitizer
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL
from pathlib import Path
from tempfile import TemporaryDirectory

async def check_review_tracing(root):
    marker = "BODY_NOT_IN_TOOL_TRACE"
    (root / "trace.py").write_text(marker, encoding="utf-8")
    replies = [response(file_call("list", ".", name="list_directory")), response(file_call("read", "trace.py")), response(message(marker))]
    baseline = None
    for enabled, mode in ((False, "ok"), (True, "ok"), (True, "connection")):
        with capture(mode, LANGSMITH_TRACING="true" if enabled else "false") as (transport, stderr):
            state, client, events = await run_case(f"review_trace_{enabled}_{mode}", deepcopy(replies), tools=[DIRECTORY_TOOL, TEXT_TOOL], executor=create_filesystem_executor(root, max_entries=2), limit=3)
        if baseline is None:
            baseline = stable(state), client.requests
        assert (stable(state), client.requests) == baseline
        if not enabled:
            assert not transport.sent
            continue
        if mode != "ok":
            assert stderr.getvalue()
            continue
        assert not stderr.getvalue(), stderr.getvalue()
        models = sorted((span for span in transport.sent if span["run_type"] == "llm"), key=lambda s: s["extra"]["metadata"]["model_call"])
        tools = {span["extra"]["metadata"]["call_id"]: span for span in transport.sent if span["run_type"] == "tool"}
        assert tools["list"]["inputs"]["arguments"]["after"] is None
        assert "next_after" in tools["list"]["outputs"]["result"]
        assert marker not in json.dumps(tools) and marker in json.dumps(models[-1]["outputs"])
        for index, model in enumerate(models):
            assert model["extra"]["metadata"]["remaining_model_calls"] == 3 - index
            assert model["inputs"].get("instructions") == client.requests[index].get("instructions")
            assert model["extra"]["metadata"]["read_budget"] == next(
                event["read_budget"] for event in events
                if event["event"] == "model.started" and event["model_call"] == index + 1
            )
            for item in model["inputs"]["input"]:
                if item.get("type") == "function_call_output":
                    assert json.loads(item["output"]) == tools[item["call_id"]]["outputs"]
        assert models[-1]["inputs"]["tool_choice"] == "none"
    refusal = message()
    refusal["content"] = [message("API_KEY=")["content"][0], {"type": "refusal", "refusal": "synthetic-review-secret"}]
    with capture() as (transport, stderr):
        state, _, _ = await run_case("refusal_and_close_trace", [], client=CloseFailureClient([response(refusal)]))
    assert "synthetic-review-secret" not in json.dumps(transport.sent)
    root_span = next(span for span in transport.sent if span["run_type"] == "chain")
    assert root_span["error"] == "client_close_error" and root_span["outputs"]["has_refusal"]
    assert root_span["outputs"]["cleanup_error"]["previous_stop_reason"] == "completed"
    # A large refusal still undergoes the same text/payload limits, on a copy.
    raw = response(message()).model_dump(mode="json")
    raw["output"][0]["content"] = [{"type": "refusal", "refusal": "API_KEY=synthetic-review-secret " + "x" * 100000}]
    result = parse_response(raw)
    original = deepcopy(result)
    cleaner = TraceSanitizer((), None)
    projected = cleaner.cap_payload({"outputs": cleaner.model_outputs(result)})
    assert "synthetic-review-secret" not in json.dumps(projected) and len(json.dumps(projected).encode()) < 65536
    assert result == original

def execute():
    with TemporaryDirectory(prefix="lineage-access-contract-") as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        for name in ("graph.py", "pyproject.toml", ".gitignore", "Dockerfile", "Makefile", ".env", "无扩展名"):
            (root / name).write_text("synthetic text only: " + name, encoding="utf-8")
        with check_group("recording.runtime_projection.review_tracing"):
            asyncio.run(check_review_tracing(root))
