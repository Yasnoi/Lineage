"""recording / files behavior contracts."""


import asyncio
import json
from benchmarks.checks.support.events import check_group
from benchmarks.checks.support.file_traces import BODY, CHECK_DIR, TOOLS, run_case, spans_by_call, text_call
from benchmarks.checks.support.projection import readable
from benchmarks.checks.support.runtime import call, message, outputs, response
from benchmarks.checks.support.tracing import stable
from copy import deepcopy
from lineage_agent.recording import tracing
from lineage_agent.recording.trace_data import TraceSanitizer
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL, create_text_executor
from lineage_agent.tools.types import TOOL_ARGUMENT_REASONS, TOOL_EXECUTION_REASONS
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

async def check_real_read_and_equivalence(root):
    replies = [response(text_call()), response(message("The read finished."))]
    with patch.object(tracing, "Client", side_effect=AssertionError("disabled tracing created client")):
        baseline, actual, _ = await run_case(root, replies, scenario="text_trace_disabled", enabled=False)
    for mode in ("ok", "connection", "timeout", 401, 429, 500):
        state, client, spans = await run_case(root, replies, scenario=f"text_trace_{mode}", mode=mode)
        assert stable(state) == stable(baseline) and client.requests == actual.requests
        assert state["tool_outcomes"] == {"succeeded": 1, "failed": 0, "rejected": 0, "skipped": 0}
        result = json.loads(json.loads(outputs(state)[0]["output"])["result"])
        assert result["content"] == BODY and result["bytes_read"] == len(BODY.encode("utf-8"))
        assert "TEXT_TOOL_BODY_PRIVATE_59af" not in json.dumps(spans)
        if mode == "ok":
            safe = spans_by_call(spans)["read"]["outputs"]["result"]
            assert safe == {
                "path": "中文 文件.md", "encoding": "utf-8", "bytes_read": len(BODY.encode("utf-8")),
                "complete": True, "content_omitted": True,
            }

async def check_listing_and_model_quotation(root):
    replies = [
        response(call("list", name="list_directory", arguments='{"path":"."}')),
        response(text_call("read_bom", "BOM.TXT")),
        response(message("资料引用：" + BODY)),
    ]
    state, client, spans = await run_case(
        root, replies, scenario="text_trace_directory_read_model_quote",
        executor=create_filesystem_executor(root), tools=TOOLS,
    )
    tool_span = spans_by_call(spans)["read_bom"]
    assert "TEXT_TOOL_BODY_PRIVATE_59af" not in json.dumps(tool_span)
    assert tool_span["outputs"]["result"]["encoding"] == "utf-8-sig"
    assert tool_span["outputs"]["result"]["bytes_read"] == len(BODY.encode("utf-8-sig"))
    for span in spans:
        if span["run_type"] == "llm":
            assert "TEXT_TOOL_BODY_PRIVATE_59af" not in json.dumps(span["inputs"])
    final_llm = max((span for span in spans if span["run_type"] == "llm"), key=lambda span: span["extra"]["metadata"]["model_call"])
    assert BODY in readable(final_llm["outputs"]["output"])
    root_span = next(span for span in spans if span["run_type"] == "chain")
    assert BODY in root_span["outputs"]["text"]
    assert state["model_calls"] == 3 and state["tool_calls"] == 2
    assert BODY in json.loads(json.loads(client.requests[-1]["input"][-1]["output"])["result"])["content"]

async def check_failure_reasons(root):
    requests = [
        text_call("shape", arguments='{"path":false}'), text_call("directory", "."),
        text_call("limit", "long.txt"),
        text_call("decode", "invalid.txt"), text_call("missing", "missing.txt"),
        text_call("permission", "permission.txt"), text_call("io", "io.txt"),
        text_call("binary", "binary.txt"),
        text_call("empty", "empty.txt"),
    ]
    original_open = Path.open

    class ReadFailure:
        """A real open file yields a real first byte, then an injected I/O error."""

        def __init__(self, handle):
            self.handle = handle
            self.called = False

        def fileno(self):
            return self.handle.fileno()

        def read(self, size):
            if self.called:
                raise OSError("PRIVATE_READ_EXCEPTION")
            self.called = True
            return self.handle.read(min(size, 1))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

    def injected_open(path, *args, **kwargs):
        if args and args[0] == "rb" and path == root / "permission.txt":
            raise PermissionError("PRIVATE_PERMISSION_EXCEPTION")
        handle = original_open(path, *args, **kwargs)
        if args and args[0] == "rb" and path == root / "io.txt":
            return ReadFailure(handle)
        return handle

    with patch.object(Path, "open", new=injected_open):
        state, _, spans = await run_case(
            root, [response(*requests), response(message("Some files were rejected or failed."))],
            scenario="text_trace_safe_reasons_injected_permission_and_mid_read",
            executor=create_text_executor(root, max_bytes=8),
        )
    tools = spans_by_call(spans)
    events = [json.loads(line) for line in Path(state["log_path"]).read_text(encoding="utf-8").splitlines()]
    by_id = {item["call_id"]: json.loads(item["output"]) for item in outputs(state)}
    assert by_id["empty"]["ok"]
    expected = {
        "shape": "invalid_text_argument_shape", "directory": "target_not_regular_file",
        "limit": "file_too_large", "decode": "decode_error",
        "missing": "file_not_found", "permission": "permission_denied", "io": "io_error",
        "binary": "non_text_content",
    }
    for identifier, expected_reason in expected.items():
        output = by_id[identifier]
        assert not output["ok"] and output["error"]["reason_code"]
        code = output["error"]["reason_code"]
        assert code == expected_reason
        reasons = TOOL_ARGUMENT_REASONS if output["error"]["code"] == "invalid_arguments" else TOOL_EXECUTION_REASONS
        assert output["error"]["reason_message"] == reasons[code]
        span = tools[identifier]
        assert span["extra"]["metadata"]["reason_code"] == code
        assert span["outputs"]["error"]["reason_code"] == code
        matching = [event for event in events if event.get("call_id") == identifier and event["event"] in ("tool.rejected", "tool.failed", "tool.result_created")]
        assert len(matching) == 2 and all(event["reason_code"] == code for event in matching)
    assert state["tool_outcomes"] == {"succeeded": 1, "rejected": 4, "failed": 4, "skipped": 0}
    assert state["tool_calls"] == 9 and state["model_calls"] == 2
    assert "PRIVATE_READ_EXCEPTION" not in json.dumps((state, spans, events))
    assert "PRIVATE_PERMISSION_EXCEPTION" not in json.dumps((state, spans, events))

def check_projection_boundaries(root):
    sanitizer = TraceSanitizer(("KNOWN_FILE_SECRET",), root)
    metadata = {"path": "password=PATH_SECRET.txt", "encoding": "utf-8", "bytes_read": 300000, "complete": True}
    for body in ("BODY_PRIVATE " + "KNOWN_FILE_SECRET", "x" * 300000):
        output = {"ok": True, "result": json.dumps({**metadata, "content": body, "private": "EXTRA_PRIVATE"})}
        source = [text_call("projection"), {"type": "function_call_output", "call_id": "projection", "output": json.dumps(output)}]
        before = deepcopy(source)
        payload = sanitizer.cap_payload({"inputs": sanitizer.model_inputs("local", source, [TEXT_TOOL]), "outputs": sanitizer.tool_outputs(output, "read_text_file")})
        encoded = json.dumps(payload, ensure_ascii=False)
        assert all(marker not in encoded for marker in ("BODY_PRIVATE", "KNOWN_FILE_SECRET", "PATH_SECRET", "EXTRA_PRIVATE"))
        assert len(encoded.encode("utf-8")) <= 65536
        assert source == before
        assert sanitizer.cap_payload(payload) == payload
    for top_code, reasons in (("invalid_arguments", TOOL_ARGUMENT_REASONS), ("execution_error", TOOL_EXECUTION_REASONS)):
        for code, expected in reasons.items():
            raw = {"ok": False, "error": {"code": top_code, "message": "PRIVATE_EXCEPTION", "reason_code": code, "reason_message": "PRIVATE_EXCEPTION"}}
            projected = sanitizer.tool_outputs(raw, "read_text_file")
            assert projected["error"]["reason_code"] == code and projected["error"]["reason_message"] == expected
            assert "PRIVATE_EXCEPTION" not in json.dumps(projected)
        raw["error"]["reason_code"] = "UNTRUSTED_REASON"
        assert "UNTRUSTED_REASON" not in json.dumps(sanitizer.tool_outputs(raw, "read_text_file"))
    source = {"ok": True, "result": json.dumps({**metadata, "path": "a.txt", "content": "PRIVATE_BODY", "encoding": "PRIVATE_ENCODING", "bytes_read": True, "complete": "PRIVATE_FLAG"})}
    result = sanitizer.tool_outputs(source, "read_text_file")["result"]
    assert result == {"path": "a.txt", "content_omitted": True}
    assert "PRIVATE_BODY" not in json.dumps(sanitizer.tool_outputs(source, "unknown"))
    print(json.dumps({"scenario": "text_trace_projection_boundaries", "source": "direct projection checks; no Agent run", "result": "passed"}))

async def main():
    with TemporaryDirectory(prefix="lineage-text-tracing-") as temporary:
        root = Path(temporary)
        fixture = {
            "中文 文件.md": BODY.encode("utf-8"), "BOM.TXT": BODY.encode("utf-8-sig"),
            "empty.txt": b"", "long.txt": b"x" * 9, "invalid.txt": b"\xff",
            "unsupported.py": b"print('not read')", "permission.txt": b"safe",
            "io.txt": b"safe", "binary.txt": b"\x00",
        }
        for name, content in fixture.items():
            (root / name).write_bytes(content)
        before = {name: (root / name).read_bytes() for name in fixture}
        cwd = Path.cwd()
        with check_group("recording.files.real_read_and_equivalence"):
            await check_real_read_and_equivalence(root)
        print("PASS real reads, tracing toggle and five simulated upload failures")
        with check_group("recording.files.listing_and_model_quotation"):
            await check_listing_and_model_quotation(root)
        print("PASS directory/read metadata and model quotation are distinct")
        with check_group("recording.files.failure_reasons"):
            await check_failure_reasons(root)
        print("PASS safe reasons agree across model outputs, JSONL and tracing")
        with check_group("recording.files.projection_boundaries"):
            check_projection_boundaries(root)
        print("PASS trace capacity, source isolation and untrusted reason rejection")
        with check_group("recording.files.fixture_integrity"):
            assert {name: (root / name).read_bytes() for name in fixture} == before
            assert sorted(path.name for path in root.iterdir()) == sorted(fixture)
            assert Path.cwd() == cwd
    print("PASS 5 text tracing groups; real files and SDK serialization, no real endpoint calls")
    print(f"Check logs: {CHECK_DIR}")

def execute():
    asyncio.run(main())
