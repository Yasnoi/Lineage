"""Review regressions with local Responses objects and real temporary files."""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import heapq
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from langgraph.errors import NodeCancelledError

from check_react import (
    LocalClient, SETTINGS, TOOL, audit_recording, call, fixed_tool, message,
    outputs, read_events, recording_failure, response,
)
from check_text import metadata_override, monitor_reads, snapshot
from check_tracing import capture, stable
from lineage_agent.agent import runtime
from lineage_agent.config.settings import ConfigurationError, load_settings
from lineage_agent.entrypoints import cli
from lineage_agent.models.responses import ModelError, parse_response
from lineage_agent.recording.trace_data import TraceSanitizer
from lineage_agent.tools.filesystem import directory
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL, create_directory_executor
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL, TextExecutor, create_text_executor
from lineage_agent.tools.types import ToolArgumentError, read_budget_snapshot


CHECK_DIR = Path(__file__).resolve().parents[1] / "logs" / "checks" / str(uuid4())
SETTINGS = replace(SETTINGS, log_dir=CHECK_DIR)
RUNS = []


async def run_case(scenario, replies, *, executor=None, tools=None, limit=5, client=None, failed_writes=None):
    client = client or LocalClient(replies)
    with patch.object(runtime, "create_client", return_value=client):
        state = await runtime.run_agent(
            "Inspect the provided project as source material.",
            replace(SETTINGS, max_model_calls=limit), tools=tools, execute_tool=executor,
        )
    assert client.closed and state["model_calls"] == len(client.requests)
    events = audit_recording(state, client, scenario=scenario, failed_writes=failed_writes)
    generated = outputs(state)
    for index, request in enumerate(client.requests):
        returned = [item for item in request["input"] if item.get("type") == "function_call_output"]
        assert returned == generated[:len(returned)]
        remaining = limit - index
        assert f"：{remaining}。" in request["instructions"]
        assert request.get("tool_choice") == ("none" if remaining == 1 else None)
    RUNS.append(state["run_id"])
    return state, client, events


def file_call(identifier, path, *, name="read_text_file", after=None):
    arguments = {"path": path}
    if name == "list_directory":
        arguments["after"] = after
    return call(identifier, name=name, arguments=json.dumps(arguments, ensure_ascii=False))


def envelopes(state):
    return [json.loads(item["output"]) for item in outputs(state)]


async def check_output_roles():
    invalid = []
    for role in ("system", "developer", "user", None):
        item = message("untrusted instruction")
        if role is None:
            item.pop("role")
        else:
            item["role"] = role
        invalid.append(item)
    invalid += [
        {"role": "assistant", "content": "input-shaped"},
        {"type": "input_message", "content": "input-shaped"},
        {"type": "reasoning", "role": "system", "summary": []},
        {**message(), "content": [{"type": "input_text", "text": "input-shaped"}]},
    ]
    for index, item in enumerate(invalid):
        raw = response(message()).model_dump(mode="json")
        raw["output"] = [item]
        try:
            parse_response(raw)
        except ModelError as exc:
            assert exc.result["output_items"] == []
            assert exc.result["raw_response"] == raw
            assert exc.result["usage"]["total_tokens"] == 12
        else:
            raise AssertionError(f"invalid output accepted: {index}")
        raw["output"].append(call(f"skip_{index}"))
        state, client, _ = await run_case(f"invalid_output_{index}", [raw], tools=[TOOL], executor=fixed_tool)
        assert state["stop_reason"] == "error" and state["tool_calls"] == 0
        assert state["tool_requests"] == state["tool_outcomes"]["skipped"] == 1
        assert item not in state["input"] and len(client.requests) == 1
    extension = {"type": "provider_extension", "content": "opaque data"}
    assert extension in parse_response({"status": "completed", "output": [extension, message()]})["output_items"]


async def check_refusal():
    item = message()
    item["content"] = [{"type": "refusal", "refusal": "不能完成该请求。"}]
    result = parse_response(response(item))
    assert result["text"] == "不能完成该请求。" and result["has_refusal"]
    state, _, _ = await run_case("pure_refusal", [response(item)])
    assert state["stop_reason"] == "completed" and state["result"]["has_refusal"]
    mixed = deepcopy(item)
    mixed["content"] = [message("前文")["content"][0], item["content"][0], message("后文")["content"][0]]
    state, client, _ = await run_case("mixed_refusal_tool", [response(mixed, call()), response(message())], tools=[TOOL], executor=fixed_tool)
    assert state["tool_calls"] == 1 and state["model_calls"] == 2
    assert not state["result"]["has_refusal"]
    assert parse_response(response(mixed))["text"] == "前文不能完成该请求。后文"
    for value in (None, [], 1):
        raw = {"status": "completed", "output": [{**item, "content": [{"type": "refusal", "refusal": value}]}]}
        state, _, _ = await run_case("invalid_refusal", [raw])
        assert state["error_code"] == "invalid_response"


async def check_direct_config():
    for changes in (
        {"base_url": "file:///tmp"}, {"base_url": "https://host:99999"},
        {"model_name": " "}, {"timeout_seconds": True},
        {"timeout_seconds": float("nan")}, {"timeout_seconds": -1},
    ):
        try:
            replace(SETTINGS, **changes)
        except ConfigurationError:
            continue
        raise AssertionError(f"invalid direct config accepted: {list(changes)}")
    mapping = {
        "base_url": "BASE_URL", "model_name": "MODEL_NAME", "timeout_seconds": "TIMEOUT_SECONDS",
        "max_model_calls": "MAX_MODEL_CALLS", "text_max_bytes": "TEXT_MAX_BYTES",
        "text_total_max_bytes": "TEXT_TOTAL_MAX_BYTES", "directory_max_entries": "DIRECTORY_MAX_ENTRIES",
    }
    cases = {
        "base_url": ("", "http://", "ftp://host", "https://host:bad", "https://ho st", "https://["),
        "model_name": ("", "  "), "timeout_seconds": (True, 0, -1, float("inf"), float("nan")),
        "max_model_calls": (True, 0, -1, 1.5), "text_max_bytes": (False, 0, -1, 1.5),
        "text_total_max_bytes": (True, 0, -1, 1.5), "directory_max_entries": (True, 0, -1, 1.5),
    }
    with TemporaryDirectory() as temporary:
        env_file = Path(temporary) / ".env"
        for key, values in cases.items():
            for value in values:
                for source in ("direct", "dotenv"):
                    try:
                        if source == "direct":
                            replace(SETTINGS, **{key: value})
                        else:
                            data = {"BASE_URL": SETTINGS.base_url, "MODEL_NAME": SETTINGS.model_name, mapping[key]: value}
                            env_file.write_text("\n".join(f"{k}={v}" for k, v in data.items()), encoding="utf-8")
                            with patch.dict(os.environ, {}, clear=True):
                                load_settings(env_file)
                    except ConfigurationError:
                        pass
                    else:
                        raise AssertionError((source, key, value))
        env_file.write_text("BASE_URL=https://example.invalid/v1\nMODEL_NAME=valid\nTIMEOUT_SECONDS=1.5\nTEXT_TOTAL_MAX_BYTES=99\nPROJECT_ROOT=project", encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            loaded = load_settings(env_file)
        assert loaded.timeout_seconds == 1.5 and loaded.text_total_max_bytes == 99
        assert loaded.project_root == Path(temporary) / "project"
    assert str(ToolArgumentError(reason_code="read_budget_exceeded"))
    try:
        replace(SETTINGS, timeout_seconds=10**1000)
    except ConfigurationError:
        pass
    else:
        raise AssertionError("oversized numeric timeout accepted")
    assert str(ModelError("PRIVATE", code="unknown", stage="PRIVATE")) == "模型调用失败。"
    for code in ("model_timeout", "connection_error", "http_error"):
        state, _, events = await run_case(f"safe_{code}", [ModelError("PRIVATE", code=code, stage="untrusted")])
        assert state["error_code"] == code and "PRIVATE" not in state["error"]
        assert next(event for event in events if event["event"] == "model.failed")["stage"] == "model_request"


class CloseFailureClient(LocalClient):
    close_calls = 0

    async def __aexit__(self, *_):
        self.closed = True
        self.close_calls += 1
        raise OSError("PRIVATE_CLOSE_ERROR")


async def check_close_failure():
    client = CloseFailureClient([response(message("retained answer"))])
    with patch.object(runtime, "create_client", return_value=client):
        state = await runtime.run_agent("question", SETTINGS)
    assert state["stop_reason"] == "error" and state["error_code"] == "client_close_error"
    assert state["result"]["text"] == "retained answer"
    assert state["model_calls"] == 1 and state["usage_totals"]["total_tokens"] == 12
    assert state["cleanup_error"]["previous_stop_reason"] == "completed"
    assert client.close_calls == 1 and state["log_complete"]
    assert "PRIVATE_CLOSE_ERROR" not in Path(state["log_path"]).read_text(encoding="utf-8")
    audit_recording(state, client, scenario="answer_then_close_failure")
    RUNS.append(state["run_id"])
    states = [state]
    for name, reply, expected in (
        ("request_failed", ModelError("PRIVATE", code="model_timeout"), "model_timeout"),
        ("protocol_failed", response(message("unfinished"), status="incomplete"), "incomplete_response"),
        ("limit_reached", response(call()), "client_close_error"),
    ):
        client = CloseFailureClient([reply])
        state, _, _ = await run_case(f"close_{name}", [], client=client, limit=1, tools=[TOOL], executor=fixed_tool)
        states.append(state)
        assert state["error_code"] == expected and client.close_calls == 1
        assert state["cleanup_error"]["previous_stop_reason"] == ("max_model_calls" if name == "limit_reached" else "error")
    for state in states:
        def finish(coroutine):
            coroutine.close()
            return state
        out, err = StringIO(), StringIO()
        with patch("sys.argv", ["lineage-agent", "question"]), patch.object(cli, "load_settings", return_value=SETTINGS), patch.object(cli.asyncio, "run", side_effect=finish), redirect_stdout(out), redirect_stderr(err):
            assert cli.main() == 1
        assert bool(out.getvalue()) == (state["cleanup_error"]["previous_stop_reason"] == "completed")
        assert state["run_id"] in err.getvalue() and "PRIVATE" not in err.getvalue()
        assert "客户端关闭失败" in err.getvalue()
    for event in ("run.cleanup_failed", "run.ended"):
        with recording_failure(event) as observations:
            state, client, _ = await run_case(f"close_log_failure_{event}", [], client=CloseFailureClient([response(message())]), failed_writes=observations)
        assert not state["log_complete"] and state["result"] is not None and client.close_calls == 1
    for exception in (ConfigurationError("original configuration error"), asyncio.CancelledError()):
        class InterruptedClient(CloseFailureClient):
            async def create(self, **kwargs):
                raise exception

            async def __aenter__(self):
                if isinstance(exception, ConfigurationError):
                    raise exception
                return self

        client = InterruptedClient([])
        with patch.object(runtime, "create_client", return_value=client):
            try:
                await runtime.run_agent("question", SETTINGS)
            except (ConfigurationError, asyncio.CancelledError, NodeCancelledError) as exc:
                assert isinstance(exc, type(exception)) or isinstance(exc, NodeCancelledError)
            else:
                raise AssertionError("primary exception lost")
        assert client.close_calls == 1


async def check_file_scope(root):
    paths = ["graph.py", "pyproject.toml", ".gitignore", "Dockerfile", "Makefile", ".env", "无扩展名"]
    executor = create_filesystem_executor(root)
    before = snapshot(root)
    requests = [file_call(f"source_{index}", path) for index, path in enumerate(paths)]
    state, client, events = await run_case("source_and_config_scope", [response(*requests), response(message())], tools=[TEXT_TOOL], executor=executor)
    assert state["tool_outcomes"]["succeeded"] == len(paths)
    for path, envelope in zip(paths, envelopes(state)):
        assert json.loads(envelope["result"])["content"] == (root / path).read_text(encoding="utf-8")
    assert snapshot(root) == before and read_budget_snapshot(executor)["used_bytes"] == 0
    assert events[-1]["read_budget"]["used_bytes"] == sum((root / path).stat().st_size for path in paths)


async def check_pagination(root):
    pages = root / "pages"
    pages.mkdir()
    names = ["z.txt", "B.txt", "a.txt", "目录 空格", "中文.md"] + [f"entry_{i:03}" for i in range(75)]
    for name in reversed(names):
        (pages / name).write_bytes(b"")
    ordered = sorted(names, key=lambda name: (name.casefold(), name))
    executor = create_directory_executor(root, max_entries=3)
    after, seen, requests = None, [], []
    max_heap = []
    original_replace = heapq._heapreplace_max
    original_heapify = heapq._heapify_max

    def replace_heap(heap, item):
        max_heap.append(len(heap))
        return original_replace(heap, item)

    def heapify(heap):
        max_heap.append(len(heap))
        return original_heapify(heap)

    with patch.object(heapq, "_heapreplace_max", side_effect=replace_heap), patch.object(heapq, "_heapify_max", side_effect=heapify):
        while True:
            page = json.loads(await executor("list_directory", {"path": "pages", "after": after}))
            assert page["after"] == after and page["limit"] == 3
            batch = [item["name"] for item in page["entries"]]
            assert len(batch) <= 3
            seen += batch
            if not page["truncated"]:
                assert page["next_after"] is None
                break
            assert page["next_after"] == batch[-1]
            after = page["next_after"]
    assert seen == ordered and max(max_heap) == 4
    # Actual graph replay of the first two pages, compared to actual filesystem.
    requests = [file_call("page1", "pages", name="list_directory"), file_call("page2", "pages", name="list_directory", after=ordered[2])]
    state, _, _ = await run_case("pagination_actual_results", [response(requests[0]), response(requests[1]), response(message())], tools=[DIRECTORY_TOOL], executor=executor)
    results = [json.loads(item["result"]) for item in envelopes(state)]
    assert [entry["name"] for result in results for entry in result["entries"]] == ordered[:6]
    assert (await executor("list_directory", {"path": "pages"})) == (await executor("list_directory", {"path": "pages", "after": None}))
    for cursor in ("", 1, False, [], "a/b", "a\\b"):
        state, _, _ = await run_case("invalid_directory_cursor", [response(file_call("bad_cursor", "pages", name="list_directory", after=cursor)), response(message())], tools=[DIRECTORY_TOOL], executor=executor)
        assert envelopes(state)[0]["error"]["reason_code"] == "invalid_directory_cursor"
    last = ordered[-1]
    assert json.loads(await executor("list_directory", {"path": "pages", "after": last}))["entries"] == []
    (pages / "00_new").write_bytes(b"")
    (pages / ordered[1]).unlink()
    page = json.loads(await executor("list_directory", {"path": "pages", "after": ordered[2]}))
    assert [entry["name"] for entry in page["entries"]] == ordered[3:6]
    assert json.loads(await executor("list_directory", {"path": "pages", "after": None}))["entries"][0]["name"] == "00_new"
    print("PASS sorted pages cover stable directory; full scan with heap <= N+1; mutation is not a snapshot")


async def check_read_budgets(root):
    (root / "large.py").write_bytes(b"x" * 65536)
    (root / "small").write_bytes(b"12")
    (root / "four").write_bytes(b"1234")
    (root / "empty").write_bytes(b"")
    (root / "invalid").write_bytes(b"\xff")
    executor = create_text_executor(root)
    calls = [file_call(f"many_{index}", "large.py") for index in range(20)]
    with monitor_reads(root) as (opened, probes):
        state, _, events = await run_case("twenty_file_budget", [response(*calls), response(message())], tools=[TEXT_TOOL], executor=executor)
    assert state["tool_outcomes"] == {"succeeded": 4, "rejected": 16, "failed": 0, "skipped": 0}
    assert len(opened) == 4 and sum(sum(length for _, length in p.reads) for p in probes) == 262144
    assert events[-1]["read_budget"] == {"limit_bytes": 262144, "used_bytes": 262144, "remaining_bytes": 0}
    assert all(item["error"]["reason_code"] == "read_budget_exceeded" for item in envelopes(state)[4:])
    bounded = create_text_executor(root, max_bytes=8, max_total_bytes=6)
    state, _, events = await run_case(
        "cross_round_repeat_failure_credit_and_small_file",
        [response(file_call("bad_encoding", "invalid"), file_call("first", "four")),
         response(file_call("over", "four"), file_call("small", "small"), file_call("empty", "empty")), response(message())],
        tools=[TEXT_TOOL], executor=bounded,
    )
    assert state["tool_outcomes"] == {"succeeded": 3, "rejected": 1, "failed": 1, "skipped": 0}
    assert events[-1]["read_budget"]["used_bytes"] == 6
    started = [event for event in events if event["event"] == "model.started"]
    assert [event["read_budget"]["remaining_bytes"] for event in started] == [6, 2, 0]
    assert [json.loads(item["result"])["content"] for item in envelopes(state) if item["ok"]] == ["1234", "12", ""]
    for value in (True, 0, -1, 1.5):
        for factory in (create_text_executor, create_filesystem_executor):
            try:
                factory(root, max_total_bytes=value)
            except ConfigurationError:
                pass
            else:
                raise AssertionError("invalid total accepted")
    # Growth/short-read injection: both metadata checks understate real bytes.
    with metadata_override(root / "four", size=0), metadata_override(root / "four", size=0, before_open=False), monitor_reads(root, short=1) as (_, probes):
        state, _, _ = await run_case("budget_growth_short_reads_injection", [response(file_call("growth", "four")), response(message())], tools=[TEXT_TOOL], executor=create_text_executor(root, max_total_bytes=2))
    assert envelopes(state)[0]["error"]["reason_code"] == "read_budget_exceeded"
    assert sum(length for _, length in probes[0].reads) == 3
    assert [size for size, _ in probes[0].reads] == [3, 2, 1]
    bound_instances = []
    original_bind = TextExecutor.bind_run

    def observe_bind(executor):
        bound = original_bind(executor)
        bound_instances.append(bound)
        return bound

    with patch.object(TextExecutor, "bind_run", new=observe_bind), recording_failure("tool.succeeded") as observations:
        state, _, _ = await run_case("read_then_log_failure", [response(file_call("debit", "small"), file_call("skip", "four"))], tools=[TEXT_TOOL], executor=bounded, failed_writes=observations)
    assert state["tool_calls"] == 1 and state["tool_outcomes"]["skipped"] == 1
    assert envelopes(state)[0]["ok"] and not state["log_complete"]
    assert bound_instances[0].read_budget["used_bytes"] == 2
    # Debit only after successful serialization; BOM and repeated success count.
    (root / "bom").write_bytes(b"\xef\xbb\xbfx")
    state, _, events = await run_case("bom_repeat_budget", [response(file_call("bom1", "bom"), file_call("bom2", "bom")), response(message())], tools=[TEXT_TOOL], executor=create_text_executor(root, max_total_bytes=8))
    assert events[-1]["read_budget"]["used_bytes"] == 8


async def check_budget_isolation(root):
    executor = create_filesystem_executor(root, max_bytes=4, max_total_bytes=4)
    replies = [response(file_call("same", "four")), response(message())]
    states = []
    for _ in range(2):
        state, _, events = await run_case("reused_executor_sequential", deepcopy(replies), tools=[TEXT_TOOL], executor=executor)
        assert events[-1]["read_budget"]["used_bytes"] == 4
        states.append(state)

    arrived = 0
    ready = asyncio.Event()

    class Interleaved(LocalClient):
        async def create(self, **kwargs):
            nonlocal arrived
            if not self.requests:
                arrived += 1
                if arrived == 2:
                    ready.set()
                await ready.wait()
            return await super().create(**kwargs)

    clients = [Interleaved(deepcopy(replies)) for _ in range(2)]
    cwd = Path.cwd()
    with patch.object(runtime, "create_client", side_effect=clients):
        concurrent = await asyncio.gather(*[
            runtime.run_agent("read", SETTINGS, tools=[TEXT_TOOL], execute_tool=executor) for _ in range(2)
        ])
    for state, client in zip(concurrent, clients):
        events = audit_recording(state, client, scenario="reused_executor_interleaved")
        assert events[-1]["read_budget"]["used_bytes"] == 4 and client.closed
        assert envelopes(state)[0]["ok"]
        RUNS.append(state["run_id"])
    assert len({state["run_id"] for state in states + concurrent}) == 4 and Path.cwd() == cwd
    assert read_budget_snapshot(executor)["used_bytes"] == 0


async def check_final_answer_allowance():
    for limit in (1, 2, 5):
        for ignores in (False, True):
            replies = [response(call(f"turn_{index}")) for index in range(limit - 1)]
            replies.append(response(call("ignored_none")) if ignores else response(message("final")))
            state, client, events = await run_case(f"final_call_{limit}_ignores_{ignores}", replies, tools=[TOOL], executor=fixed_tool, limit=limit)
            assert state["tool_calls"] == limit - 1 and state["model_calls"] == limit
            assert state["stop_reason"] == ("max_model_calls" if ignores else "completed")
            started = [event for event in events if event["event"] == "model.started"]
            assert [event["remaining_model_calls"] for event in started] == list(range(limit, 0, -1))
            assert [event["final_answer_call"] for event in started] == [False] * (limit - 1) + [True]
    state, client, _ = await run_case("early_answer_allowance", [response(message())], limit=5)
    assert state["model_calls"] == 1 and "tool_choice" not in client.requests[0]


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
            assert model["inputs"]["instructions"] == client.requests[index]["instructions"]
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


async def main():
    failures = []
    for check in (check_output_roles, check_refusal, check_direct_config, check_close_failure):
        try:
            await check()
        except Exception as exc:
            failures.append(check.__name__)
            print(f"FAIL {check.__name__}: {type(exc).__name__}")
        else:
            print(f"PASS {check.__name__}")
    assert not failures, failures
    with TemporaryDirectory(prefix="lineage-review-check-") as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        for name in ("graph.py", "pyproject.toml", ".gitignore", "Dockerfile", "Makefile", ".env", "无扩展名"):
            (root / name).write_text("synthetic text only: " + name, encoding="utf-8")
        for check in (check_file_scope, check_pagination, check_read_budgets, check_budget_isolation, check_review_tracing):
            await check(root)
            print(f"PASS {check.__name__}")
    await check_final_answer_allowance()
    print("PASS check_final_answer_allowance")
    print(json.dumps({"groups_passed": 10, "graph_runs": len(RUNS), "logs": str(CHECK_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
