"""recording / transport behavior contracts."""


import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from benchmarks.checks.recording.projection import check_capacity_and_rounds, check_redaction_boundaries
from benchmarks.checks.recording.usage import main as check_usage_coverage
from benchmarks.checks.support.events import group_id, check_group, evidence, observe_harness, observe_state
from benchmarks.checks.support.projection import directory_pair, readable, text_item
from benchmarks.checks.support.runtime import LocalClient, call, message, recording_failure, response
from benchmarks.checks.support.tracing import (
    CHECK_DIR,
    MODEL_SECRET,
    SETTINGS,
    TRACE_SECRET,
    capture,
    directory_call,
    run,
    stable,
)
from copy import deepcopy
from dataclasses import replace
from langgraph.errors import NodeCancelledError
from lineage_agent.agent import runtime
from lineage_agent.recording import tracing
from lineage_agent.recording.trace_data import TraceSanitizer
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL, create_directory_executor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

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
    for state in states:
        observe_state(state)
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
    for span in transport.sent:
        if span["run_type"] == "chain":
            evidence(CHECK_DIR / f"{span['id']}.jsonl", kind="interrupted_agent_log", run_id=span["id"])
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
    for span in transport.sent:
        if span["run_type"] == "chain":
            evidence(CHECK_DIR / f"{span['id']}.jsonl", kind="interrupted_agent_log", run_id=span["id"])
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
        trace_run_id = str(uuid4())
        trace = tracing.start_trace(trace_run_id, SETTINGS)
        observe_harness("trace_queue_harness", run_id=trace_run_id)
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
    observe_state(state)
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
    observe_harness("trace_projection")
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
    nested = sanitizer.cap_payload(sanitizer.model_inputs("check", [
        directory_call("quoted", "password=private-value"),
        {"type": "function_call_output", "call_id": "quoted", "output": '{"ok":false,"error":{"code":"invalid_arguments"}}'},
    ], None))
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
        [sys.executable, "-m", "benchmarks.checks.recording.upload_child"],
        env=env, capture_output=True, text=True, encoding="utf-8", timeout=12,
    )
    observe_harness("child_process", returncode=result.returncode)
    assert result.returncode == 0, result.stderr
    assert 4.5 <= time.monotonic() - started < 12
    assert "PRIVATE_" not in result.stderr and "incomplete" in result.stdout

async def check_serialized_redaction():
    """Inspect the SDK wire payload and compare unchanged business objects."""
    quoted = 'password="SYNTHETIC_HEAD,SYNTHETIC_TAIL"'
    split = [MODEL_SECRET[:12], MODEL_SECRET[12:]]
    prompt = [{"role": "user", "content": quoted}, message(split[0]), message(split[1])]
    replies = [response(directory_call("sensitive_path", quoted)), response(
        message(split[0]), message(split[1]), message(quoted),
    )]

    async def executor(name, arguments):
        return json.dumps({"path": ".", "entries": [{"name": quoted, "path": quoted, "type": "file"}],
                           "limit": 1, "truncated": False})

    before = deepcopy((prompt, replies))
    with capture(LANGSMITH_TRACING="false"):
        baseline, baseline_client = await run(replies, prompt=prompt, executor=executor)
    with capture() as (transport, stderr):
        state, client = await run(replies, prompt=prompt, executor=executor)
    assert not stderr.getvalue(), stderr.getvalue()
    assert stable(state) == stable(baseline) and client.requests == baseline_client.requests
    assert (prompt, replies) == before
    assert len(transport.sent) == 4
    encoded = json.dumps(transport.sent)
    assert all(marker not in encoded for marker in (MODEL_SECRET, TRACE_SECRET, "SYNTHETIC_HEAD", "SYNTHETIC_TAIL"))
    for span in transport.sent:
        if span["run_type"] == "llm":
            assert "text" not in span["outputs"]
            for owner, field in (("inputs", "input"), ("outputs", "output")):
                joined = readable(span[owner][field])
                assert MODEL_SECRET not in joined and all(part not in joined for part in split)
            for item in span["inputs"]["input"]:
                if item.get("type") == "function_call":
                    assert json.loads(item["arguments"])["path"] == 'password="[redacted]"'
                elif item.get("type") == "function_call_output":
                    assert json.loads(item["output"])["observation"]["raw_result"]["entries"]

async def check_serialized_capacity():
    history = [text_item("system instructions", "system"), text_item("initial user request", "user")]
    for index in range(20):
        history.extend(directory_pair(f"wire-{index}", entries=30, width=60))
    answer = "SHORT_SYNTHETIC_ANSWER " * 40
    replies = [response(message(answer))]
    # Exercise tracing's wire-size cap while explicitly admitting the large fixture.
    settings = replace(SETTINGS, context_budget_tokens=2_000_000)
    with capture(LANGSMITH_TRACING="false"):
        baseline, baseline_client = await run(replies, prompt=history, settings=settings)
    with capture() as (transport, stderr):
        state, client = await run(replies, prompt=history, settings=settings)
    assert not stderr.getvalue(), stderr.getvalue()
    assert stable(state) == stable(baseline) and client.requests == baseline_client.requests
    assert len(transport.sent) == 2 and state["model_calls"] == 1 and state["tool_requests"] == 0
    llm = next(span for span in transport.sent if span["run_type"] == "llm")
    assert readable(llm["outputs"]["output"]) == answer
    assert {key: llm["outputs"]["usage_metadata"][key] for key in ("input_tokens", "output_tokens", "total_tokens")} == {
        "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
    }
    assert llm["inputs"]["trace_omissions"]["input"]["omitted_count"] > 0
    assert llm["extra"]["metadata"]["run_id"] == state["run_id"]
    assert "duration_ms" in llm["extra"]["metadata"] and not llm.get("error")
    assert len(json.dumps(llm, ensure_ascii=False, separators=(",", ":")).encode()) <= 65536

async def main():
    with TemporaryDirectory(prefix="lineage-tracing-check-") as temporary:
        root = Path(temporary)
        (root / "example.txt").touch()
        with check_group("recording.transport.projection"):
            check_projection(root)
        print("PASS projection and redaction")
        check_redaction_boundaries()
        print("PASS credential boundaries and source immutability")
        check_capacity_and_rounds()
        print("PASS trace capacity, omissions and complete rounds")
        check_usage_coverage()
        with check_group("recording.transport.serialized_redaction"):
            await check_serialized_redaction()
        print("PASS SDK serialized and reassembled credential redaction")
        with check_group("recording.transport.serialized_capacity"):
            await check_serialized_capacity()
        print("PASS SDK oversized input preserves answer and usage")
        for check in (check_disabled_and_equivalence, check_failures_and_shutdown, check_concurrency_and_skips, check_local_recording_failure, check_late_protocol_failure):
            with check_group(group_id(check)):
                await check(root)
            print(f"PASS {check.__name__}")
        with check_group("recording.transport.missing_usage_and_model_error"):
            await check_missing_usage_and_model_error()
        print("PASS missing usage and model failure")
        with check_group("recording.transport.cancellation_and_queue_overflow"):
            await check_cancellation_and_queue_overflow()
        print("PASS cancellation and bounded queue")
    with check_group("recording.transport.cli_exit_deadline"):
        check_cli_exit_deadline()
    print("PASS CLI bounded exit with stuck upload")
    print("PASS 17 offline tracing check groups; no real endpoint calls")
    print(f"Check logs: {CHECK_DIR}")

def execute():
    asyncio.run(main())
