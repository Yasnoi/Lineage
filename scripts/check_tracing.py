"""Offline tracing acceptance: real SDK serialization, mocked HTTP, bounded exit."""

import asyncio
from contextlib import contextmanager, redirect_stderr
from copy import deepcopy
from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from unittest.mock import patch
from uuid import uuid4

import requests
from langgraph.errors import NodeCancelledError

from check_react import LocalClient, call, message, recording_failure, response
from lineage_agent.agent import runtime
from lineage_agent.config.settings import ModelSettings
from lineage_agent.recording import tracing
from lineage_agent.recording.trace_data import TraceSanitizer
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL, create_directory_executor


CHECK_DIR = Path(__file__).resolve().parents[1] / "logs" / "checks" / str(uuid4())
SETTINGS = ModelSettings(
    base_url="https://unused.invalid/v1", model_name="tracing-check", log_dir=CHECK_DIR,
)
MODEL_SECRET = "synthetic-model-secret-41f0"
TRACE_SECRET = "synthetic-trace-secret-2741"
ENV = {
    "LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": TRACE_SECRET,
    "LANGSMITH_ENDPOINT": "https://tracing.invalid", "LANGSMITH_PROJECT": "offline-check",
    "LANGSMITH_WORKSPACE_ID": "", "API_KEY": MODEL_SECRET,
}


class Transport:
    def __init__(self, mode="ok"):
        self.mode = mode
        self.sent = []
        self.timeouts = []

    def request(self, session, method, url, **kwargs):
        assert method.upper() == "POST" and url == "https://tracing.invalid/runs", url
        retry = session.get_adapter(url).max_retries
        assert retry.total == 0 and retry.respect_retry_after_header is False
        self.sent.append(json.loads(kwargs["data"]))
        self.timeouts.append(kwargs["timeout"])
        if self.mode == "hang":
            threading.Event().wait(120)
        if self.mode == "connection":
            raise requests.ConnectionError("PRIVATE_TRANSPORT_EXCEPTION " + TRACE_SECRET)
        if self.mode == "timeout":
            raise requests.Timeout("PRIVATE_TRANSPORT_EXCEPTION " + TRACE_SECRET)
        result = requests.Response()
        result.status_code = self.mode if isinstance(self.mode, int) else 200
        result._content = b'{"detail":"PRIVATE_SERVER_EXCEPTION"}'
        result.headers["Retry-After"] = "3600"
        result.url = url
        return result


@contextmanager
def capture(mode="ok", **env):
    transport = Transport(mode)
    stderr = StringIO()
    with patch.dict(os.environ, {**ENV, **env}, clear=True), patch.object(
        requests.Session, "request", autospec=True, side_effect=transport.request,
    ), redirect_stderr(stderr):
        yield transport, stderr


def directory_call(identifier, path=".", **kwargs):
    return call(identifier, name="list_directory", arguments=json.dumps({"path": path}), **kwargs)


async def run(replies, *, settings=SETTINGS, prompt="List the synthetic directory.", executor=None):
    client = LocalClient(deepcopy(replies))
    with patch.object(runtime, "create_client", return_value=client):
        state = await runtime.run_agent(
            prompt, settings, tools=[DIRECTORY_TOOL] if executor else None, execute_tool=executor,
        )
    assert client.closed and state["log_complete"]
    return state, client


def stable(state):
    return {key: value for key, value in state.items() if key not in {"run_id", "log_path"}}


async def check_disabled_and_equivalence(root):
    settings = replace(SETTINGS, project_root=root)
    executor = create_directory_executor(root)
    replies = [response(
        directory_call("ok"), directory_call("not_directory", "example.txt"),
        call("bad_json", name="list_directory", arguments="{PRIVATE_INVALID_JSON"),
        call("unknown", name="not_registered", arguments='{"private":"PRIVATE_UNKNOWN"}'),
        directory_call("missing", "missing"),
    ), response(message("The permitted files have been listed."))]
    with capture(LANGSMITH_TRACING="false", LANGSMITH_API_KEY="", LANGCHAIN_TRACING_V2="true") as (off, _):
        with patch.object(tracing, "Client", side_effect=AssertionError("disabled client constructed")):
            baseline, baseline_client = await run(replies, settings=settings, executor=executor)
        assert not off.sent
    with capture() as (on, stderr):
        state, client = await run(replies, settings=settings, executor=executor)
    assert not stderr.getvalue(), stderr.getvalue()
    assert stable(state) == stable(baseline)
    assert client.requests == baseline_client.requests
    assert state["tool_outcomes"] == {"succeeded": 1, "rejected": 3, "failed": 1, "skipped": 0}
    assert state["tool_calls"] == 3 and state["tool_requests"] == 5 and state["model_calls"] == 2
    spans = on.sent
    assert len(spans) == 8, [(span["name"], span["run_type"]) for span in spans]
    root_span = next(span for span in spans if span["run_type"] == "chain")
    assert root_span["id"] == state["run_id"]
    children = [span for span in spans if span is not root_span]
    assert all(span["parent_run_id"] == state["run_id"] for span in children)
    assert all(span["trace_id"] == state["run_id"] for span in spans)
    assert all(span["start_time"] <= span["end_time"] for span in spans)
    tools = {span["extra"]["metadata"]["call_id"]: span for span in spans if span["run_type"] == "tool"}
    assert set(tools) == {"ok", "not_directory", "bad_json", "unknown", "missing"}
    assert not tools["ok"].get("error")
    assert all(tools[key].get("error") for key in set(tools) - {"ok"})
    assert tools["not_directory"]["extra"]["metadata"]["reason_code"] == "target_not_directory"
    assert [tools[key]["extra"]["metadata"]["request_index"] for key in tools] == [1, 2, 3, 4, 5]
    for span in tools.values():
        metadata = span["extra"]["metadata"]
        assert metadata["run_id"] == state["run_id"] and metadata["step_id"] == "step-0002"
    llms = [span for span in spans if span["run_type"] == "llm"]
    assert len(llms) == 2
    assert all(span["outputs"]["usage_metadata"]["input_tokens"] == 10 for span in llms)
    serialized = json.dumps(spans)
    assert "example.txt" in serialized and "target_not_directory" in serialized
    for marker in ("PRIVATE_INVALID_JSON", "PRIVATE_UNKNOWN", MODEL_SECRET, TRACE_SECRET, str(root), "raw_response", "runtime_version", "log_path"):
        assert marker not in serialized, marker
    assert all(timeout == (1.0, 2.0) for timeout in on.timeouts)

    with capture(API_KEY="  " + MODEL_SECRET + "  ") as (transport, _):
        await run([response(message(MODEL_SECRET))], prompt=MODEL_SECRET)
    assert MODEL_SECRET not in json.dumps(transport.sent), "normalize credentials like the model client"


async def check_failures_and_shutdown(root):
    replies = [response(directory_call("a")), response(message("done"))]
    settings = replace(SETTINGS, project_root=root)
    executor = create_directory_executor(root)
    with capture(LANGSMITH_TRACING="false"):
        baseline, observed = await run(replies, settings=settings, executor=executor)
    for mode in ("connection", "timeout", 401, 429, 500):
        started = time.monotonic()
        with capture(mode) as (transport, stderr):
            state, client = await run(replies, settings=settings, executor=executor)
        assert stable(state) == stable(baseline) and client.requests == observed.requests
        assert len(transport.sent) <= 4, "an upload was retried"
        assert len({span["id"] for span in transport.sent}) == len(transport.sent)
        assert time.monotonic() - started < 6
        assert stderr.getvalue(), "upload failure must be reported safely"
        for marker in ("PRIVATE_TRANSPORT_EXCEPTION", "PRIVATE_SERVER_EXCEPTION", TRACE_SECRET):
            assert marker not in stderr.getvalue(), stderr.getvalue()
    for env in ({"LANGSMITH_API_KEY": ""}, {"LANGSMITH_ENDPOINT": "not-a-url"}, {"LANGSMITH_TRACING": "typo"}, {"LANGSMITH_USE_PYO3_CLIENT": "true"}):
        with capture(**env) as (transport, stderr):
            state, client = await run(replies, settings=settings, executor=executor)
        assert stable(state) == stable(baseline) and client.requests == observed.requests
        assert not transport.sent and stderr.getvalue()
    with capture() as (_, stderr), patch.object(tracing, "Client", side_effect=RuntimeError("PRIVATE_INIT_EXCEPTION")):
        state, client = await run(replies, settings=settings, executor=executor)
    assert stable(state) == stable(baseline) and client.requests == observed.requests
    assert "PRIVATE_INIT_EXCEPTION" not in stderr.getvalue() and stderr.getvalue()
    with capture() as (_, stderr), patch.object(TraceSanitizer, "model_inputs", side_effect=ValueError("PRIVATE_FILTER_EXCEPTION")):
        state, client = await run(replies, settings=settings, executor=executor)
    assert stable(state) == stable(baseline) and client.requests == observed.requests
    assert "PRIVATE_FILTER_EXCEPTION" not in stderr.getvalue() and stderr.getvalue()


async def check_missing_usage_and_model_error():
    for reply in (response(message(), usage=None), RuntimeError("PRIVATE_MODEL_EXCEPTION")):
        with capture() as (transport, _):
            state, _ = await run([reply])
        llm = next(span for span in transport.sent if span["run_type"] == "llm")
        assert "usage_metadata" not in llm.get("outputs", {}) or not llm["outputs"]["usage_metadata"]
        if isinstance(reply, Exception):
            assert llm.get("error") and state["stop_reason"] == "error"
        assert "PRIVATE_MODEL_EXCEPTION" not in json.dumps(transport.sent)


async def check_concurrency_and_skips(root):
    settings = replace(SETTINGS, project_root=root, max_model_calls=1)
    clients = [LocalClient([response(directory_call(f"call_{index}"))]) for index in range(2)]
    with capture() as (transport, _), patch.object(runtime, "create_client", side_effect=clients):
        states = await asyncio.gather(*[
            runtime.run_agent("List the directory.", settings, tools=[DIRECTORY_TOOL], execute_tool=create_directory_executor(root))
            for _ in range(2)
        ])
    assert len(transport.sent) == 6
    for index, state in enumerate(states):
        spans = [span for span in transport.sent if span["trace_id"] == state["run_id"]]
        assert len(spans) == 3 and state["tool_calls"] == 0
        tool = next(span for span in spans if span["run_type"] == "tool")
        assert tool["extra"]["metadata"]["call_id"] == f"call_{index}"
        assert tool["extra"]["metadata"]["outcome"] == "skipped" and tool.get("error")


async def check_cancellation_and_queue_overflow():
    class CancelledClient(LocalClient):
        async def create(self, **kwargs):
            raise asyncio.CancelledError()

    client = CancelledClient([])
    with capture() as (transport, _), patch.object(runtime, "create_client", return_value=client):
        try:
            await runtime.run_agent("cancel me", SETTINGS)
        except (asyncio.CancelledError, NodeCancelledError):
            pass
        else:
            raise AssertionError("cancellation was swallowed")
    assert client.closed and len(transport.sent) == 2
    assert all(span.get("error") for span in transport.sent)

    async def cancelled_executor(name, arguments):
        raise asyncio.CancelledError()

    client = LocalClient([response(directory_call("cancel_tool"))])
    with capture() as (transport, _), patch.object(runtime, "create_client", return_value=client):
        try:
            await runtime.run_agent("cancel tool", SETTINGS, tools=[DIRECTORY_TOOL], execute_tool=cancelled_executor)
        except (asyncio.CancelledError, NodeCancelledError):
            pass
        else:
            raise AssertionError("tool cancellation was swallowed")
    tool = next(span for span in transport.sent if span["run_type"] == "tool")
    assert tool["error"] == "tool_interrupted" and client.closed

    entered, release = threading.Event(), threading.Event()

    class BlockedClient:
        def create_run(self, **kwargs):
            entered.set()
            release.wait(3)

        def close(self, **kwargs):
            pass

    with capture() as (_, stderr), patch.object(tracing, "QUEUE_SIZE", 1), patch.object(tracing, "_create_client", return_value=BlockedClient()):
        trace = tracing.start_trace(str(uuid4()), SETTINGS)
        span = trace.model_start("check", "first", None)
        trace.model_end(span)
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        for index in range(4):
            trace.tool_skipped(directory_call(str(index)), step_id="step-0001", request_index=index + 1, reason="max_model_calls")
        trace.finish()
        release.set()
        await trace.close()
    assert "队列已满" in stderr.getvalue()


async def check_local_recording_failure(root):
    client = LocalClient([response(directory_call("first"), directory_call("second"))])
    with capture() as (transport, _), patch.object(runtime, "create_client", return_value=client), recording_failure("tool.started"):
        state = await runtime.run_agent(
            "List the directory.", replace(SETTINGS, project_root=root),
            tools=[DIRECTORY_TOOL], execute_tool=create_directory_executor(root),
        )
    assert state["stop_reason"] == "error" and state["log_complete"] is False
    assert state["tool_calls"] == 0 and state["tool_outcomes"]["skipped"] == 2
    spans = [span for span in transport.sent if span["run_type"] == "tool"]
    assert len(spans) == 2
    assert all(span["extra"]["metadata"]["outcome"] == "skipped" for span in spans)
    assert all(span["extra"]["metadata"]["error_code"] == "logging_error" for span in spans)


async def check_late_protocol_failure(root):
    settings = replace(SETTINGS, project_root=root)
    with capture() as (transport, _):
        state, _ = await run(
            [response(directory_call("duplicate")), response(directory_call("duplicate"))],
            settings=settings, executor=create_directory_executor(root),
        )
    assert state["stop_reason"] == "error" and state["tool_calls"] == 1
    models = sorted(
        (span for span in transport.sent if span["run_type"] == "llm"),
        key=lambda span: span["extra"]["metadata"]["model_call"],
    )
    assert not models[0].get("error")
    assert models[1]["error"] == "duplicate_call_id"
    assert models[1]["extra"]["metadata"]["outcome"] == "failed"


def check_projection(root):
    sanitizer = TraceSanitizer((MODEL_SECRET, TRACE_SECRET), root)
    response_body = response(message("visible " + MODEL_SECRET)).model_dump(mode="json")
    response_body["private_extension"] = "PRIVATE_RESPONSE_EXTENSION"
    result = {
        "text": "visible " + MODEL_SECRET, "output_items": response_body["output"],
        "raw_response": response_body, "usage": {"input_tokens": 10, "output_tokens": None, "total_tokens": None},
        "tool_calls": [],
    }
    inputs = [
        {"role": "user", "content": "hello " + MODEL_SECRET + " " + str(root)},
        {"type": "reasoning", "encrypted_content": "PRIVATE_REASONING"},
        {"type": "message", "role": "user", "content": [{"type": "input_file", "file_data": "PRIVATE_ATTACHMENT"}]},
        directory_call("directory", "example.txt"),
        {"type": "function_call_output", "call_id": "directory", "output": json.dumps({"ok": True, "result": json.dumps({
            "path": ".", "entries": [{"name": "example.txt", "path": "example.txt", "type": "file", "private": "PRIVATE_ENTRY"}],
            "limit": 100, "truncated": False, "private": "PRIVATE_LISTING",
        })})},
        call("other", name="unknown", arguments='{"path":"PRIVATE_OTHER_ARGUMENT"}'),
        {"type": "function_call_output", "call_id": "other", "output": '{"ok":true,"result":"PRIVATE_OTHER_RESULT"}'},
    ]
    before = deepcopy((inputs, result))
    payload = {"inputs": sanitizer.model_inputs("tracing-check", inputs, [DIRECTORY_TOOL]), "outputs": sanitizer.model_outputs(result)}
    serialized = json.dumps(payload)
    assert "example.txt" in serialized and "visible" in serialized
    for marker in (MODEL_SECRET, TRACE_SECRET, str(root), "PRIVATE_REASONING", "PRIVATE_ATTACHMENT", "PRIVATE_ENTRY", "PRIVATE_LISTING", "PRIVATE_OTHER_ARGUMENT", "PRIVATE_OTHER_RESULT", "PRIVATE_RESPONSE_EXTENSION"):
        assert marker not in serialized, marker
    assert (inputs, result) == before
    nested = sanitizer.cap_payload(sanitizer.model_inputs("check", [directory_call("quoted", "password=private-value")], None))
    arguments = json.loads(nested["input"][0]["arguments"])
    assert "private-value" not in arguments["path"]
    capped = sanitizer.cap_payload({"input": "x" * 200000, "call_id": "keep-me"})
    assert "truncat" in json.dumps(capped).lower()
    assert len(json.dumps(capped).encode()) < 100000

    split_parts = [{"type": "output_text", "text": MODEL_SECRET[:12]},
                   {"type": "output_text", "text": MODEL_SECRET[12:]}]
    original_parts = deepcopy(split_parts)
    split = sanitizer.model_inputs("check", [{"type": "message", "role": "assistant", "content": split_parts}], None)
    assert [part["text"] for part in split["input"][0]["content"]] == ["[redacted]", "[redacted]"]
    assert split_parts == original_parts

    windows = TraceSanitizer((), Path(r"C:\Synthetic\Tracing Root"))
    for path in (r"c:\synthetic\tracing root\child", "C:/SYNTHETIC/TRACING ROOT/child", r"c:/Synthetic\Tracing Root/child"):
        assert windows.text(path).startswith("[project_root]")

    nested_items = [directory_call("nested", "password=private-value"), {
        "type": "function_call_output", "call_id": "nested",
        "output": json.dumps({"ok": True, "result": json.dumps({
            "path": ".", "entries": [{"name": 'password="private-value"',
                                       "path": 'password="private-value"', "type": "file"}] * 100,
            "limit": 100, "truncated": False,
        })}),
    }]
    nested_original = deepcopy(nested_items)
    repeated = sanitizer.model_inputs("check", nested_items, [DIRECTORY_TOOL])
    for _ in range(2):
        repeated = sanitizer.cap_payload(repeated)
        assert json.loads(repeated["input"][0]["arguments"])["path"] == "password=[redacted]"
        assert json.loads(repeated["input"][1]["output"])["result"]["entries"]
        assert "private-value" not in json.dumps(repeated)
    assert nested_items == nested_original
    assert sanitizer.tool_outputs({"ok": False, "error": {"code": "tool_interrupted"}}, "list_directory")["error"]["code"] == "tool_interrupted"


def check_cli_exit_deadline():
    env = {key: value for key, value in os.environ.items() if not key.startswith(("LANGSMITH_", "LANGCHAIN_"))}
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--hanging-upload-child"],
        env=env, capture_output=True, text=True, encoding="utf-8", timeout=12,
    )
    assert result.returncode == 0, result.stderr
    assert 4.5 <= time.monotonic() - started < 12
    assert "PRIVATE_" not in result.stderr and "incomplete" in result.stdout


def hanging_upload_child():
    from lineage_agent.entrypoints import cli

    client = LocalClient([response(message("done"))])
    with capture("hang") as (_, stderr), patch.object(cli, "load_settings", return_value=SETTINGS), patch.object(
        runtime, "create_client", return_value=client,
    ), patch.object(sys, "argv", ["lineage-agent", "a harmless prompt"]):
        assert cli.main() == 0
    assert stderr.getvalue()
    print("incomplete upload; CLI returned")


async def main():
    with TemporaryDirectory(prefix="lineage-tracing-check-") as temporary:
        root = Path(temporary)
        (root / "example.txt").touch()
        check_projection(root)
        print("PASS projection and redaction")
        for check in (check_disabled_and_equivalence, check_failures_and_shutdown, check_concurrency_and_skips, check_local_recording_failure, check_late_protocol_failure):
            await check(root)
            print(f"PASS {check.__name__}")
        await check_missing_usage_and_model_error()
        print("PASS missing usage and model failure")
        await check_cancellation_and_queue_overflow()
        print("PASS cancellation and bounded queue")
    check_cli_exit_deadline()
    print("PASS CLI bounded exit with stuck upload")
    print(f"Check logs: {CHECK_DIR}")


if __name__ == "__main__":
    if "--hanging-upload-child" in sys.argv:
        hanging_upload_child()
    else:
        asyncio.run(main())
