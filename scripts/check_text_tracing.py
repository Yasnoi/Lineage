"""Text-tool tracing checks: real files and SDK serialization, mocked HTTP only."""

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from check_react import LocalClient, audit_recording, call, message, outputs, response
from check_trace_data import readable
from check_tracing import capture, stable
from lineage_agent.agent import runtime
from lineage_agent.config.settings import ModelSettings
from lineage_agent.recording import tracing
from lineage_agent.recording.trace_data import TraceSanitizer
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL, create_text_executor
from lineage_agent.tools.types import TOOL_ARGUMENT_REASONS, TOOL_EXECUTION_REASONS


CHECK_DIR = Path(__file__).resolve().parents[1] / "logs" / "checks" / str(uuid4())
SETTINGS = ModelSettings(
    base_url="https://unused.invalid/v1", model_name="text-tracing-check", log_dir=CHECK_DIR,
)
BODY = "正文标记 TEXT_TOOL_BODY_PRIVATE_59af\r\n第二行\r第三行\n"
TOOLS = [DIRECTORY_TOOL, TEXT_TOOL]


def text_call(identifier="read", path="中文 文件.md", *, arguments=None):
    return call(
        identifier, name="read_text_file",
        arguments=json.dumps({"path": path}, ensure_ascii=False) if arguments is None else arguments,
    )


def spans_by_call(spans):
    return {span["extra"]["metadata"]["call_id"]: span for span in spans if span["run_type"] == "tool"}


def projected_results(span):
    return {
        item["call_id"]: json.loads(item["output"])
        for item in span["inputs"]["input"] if item.get("type") == "function_call_output"
    }


async def run_case(root, replies, *, scenario, mode="ok", enabled=True, executor=None, tools=None):
    """Keep actual executor returns, SDK inputs and HTTP payloads independent."""
    executor = executor or create_text_executor(root)
    observed = []

    async def observed_executor(name, arguments):
        result = await executor(name, arguments)
        observed.append((name, deepcopy(arguments), result))
        return result

    client = LocalClient(deepcopy(replies))
    with capture(mode, LANGSMITH_TRACING="true" if enabled else "false") as (transport, stderr):
        with patch.object(runtime, "create_client", return_value=client):
            state = await runtime.run_agent(
                "Use the requested local file as data.", replace(SETTINGS, project_root=root),
                tools=tools or [TEXT_TOOL], execute_tool=observed_executor,
            )
    assert client.closed and state["log_complete"]
    events = audit_recording(state, client, scenario=scenario)
    generated = outputs(state)
    for request in client.requests:
        actual = [item for item in request["input"] if item.get("type") == "function_call_output"]
        assert actual == generated[:len(actual)]
    assert [json.loads(item["output"])["result"] for item in generated if json.loads(item["output"])["ok"]] == [item[2] for item in observed]
    assert "TEXT_TOOL_BODY_PRIVATE_59af" not in Path(state["log_path"]).read_text(encoding="utf-8")
    assert str(root) not in json.dumps(events)
    if enabled and mode == "ok":
        assert not stderr.getvalue(), stderr.getvalue()
        llms = sorted(
            (span for span in transport.sent if span["run_type"] == "llm"),
            key=lambda span: span["extra"]["metadata"]["model_call"],
        )
        assert len(llms) == len(client.requests)
        tool_spans = spans_by_call(transport.sent)
        for llm, request in zip(llms, client.requests):
            actual = [item for item in request["input"] if item.get("type") == "function_call_output"]
            projected = projected_results(llm)
            assert list(projected) == [item["call_id"] for item in actual]
            for item in actual:
                assert projected[item["call_id"]] == tool_spans[item["call_id"]]["outputs"]
        assert all(span["extra"]["metadata"]["run_id"] == state["run_id"] for span in transport.sent)
    elif enabled:
        assert stderr.getvalue() and transport.sent
    else:
        assert not transport.sent
    return state, client, transport.sent


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
        text_call("format", "unsupported.py"), text_call("limit", "long.txt"),
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
        "format": "unsupported_extension", "limit": "file_too_large", "decode": "decode_error",
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
    assert state["tool_outcomes"] == {"succeeded": 1, "rejected": 5, "failed": 4, "skipped": 0}
    assert state["tool_calls"] == 10 and state["model_calls"] == 2
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
        await check_real_read_and_equivalence(root)
        print("PASS real reads, tracing toggle and five simulated upload failures")
        await check_listing_and_model_quotation(root)
        print("PASS directory/read metadata and model quotation are distinct")
        await check_failure_reasons(root)
        print("PASS safe reasons agree across model outputs, JSONL and tracing")
        check_projection_boundaries(root)
        print("PASS trace capacity, source isolation and untrusted reason rejection")
        assert {name: (root / name).read_bytes() for name in fixture} == before
        assert sorted(path.name for path in root.iterdir()) == sorted(fixture)
        assert Path.cwd() == cwd
    print("PASS 4 text tracing groups; real files and SDK serialization, no real endpoint calls")
    print(f"Check logs: {CHECK_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
