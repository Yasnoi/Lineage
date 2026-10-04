"""filesystem / access_budget behavior contracts."""


import asyncio
import heapq
import json
from benchmarks.checks.support.observations import inline_raw, assert_tools
from benchmarks.checks.support.access import RUNS, envelopes, file_call, run_case
from benchmarks.checks.support.events import check_group, observe_state
from benchmarks.checks.support.runtime import (
    LocalClient,
    SETTINGS,
    audit_recording,
    message,
    recording_failure,
    response,
)
from benchmarks.checks.support.text import metadata_override, monitor_reads, snapshot
from copy import deepcopy
from lineage_agent.agent import runtime
from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL, create_directory_executor
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL, TextExecutor, create_text_executor
from lineage_agent.tools.types import read_budget_snapshot
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

async def check_file_scope(root):
    paths = ["graph.py", "pyproject.toml", ".gitignore", "Dockerfile", "Makefile", ".env", "无扩展名"]
    executor = create_filesystem_executor(root)
    before = snapshot(root)
    requests = [file_call(f"source_{index}", path) for index, path in enumerate(paths)]
    state, client, events = await run_case("source_and_config_scope", [response(*requests), response(message())], tools=[TEXT_TOOL], executor=executor)
    assert state["tool_outcomes"]["succeeded"] == len(paths)
    for path, envelope in zip(paths, envelopes(state)):
        assert json.loads(inline_raw(envelope))["content"] == (root / path).read_text(encoding="utf-8")
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
    results = [json.loads(inline_raw(item)) for item in envelopes(state)]
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
    executor = create_text_executor(root, max_bytes=65536, max_total_bytes=262144)
    calls = [file_call(f"many_{index}", "large.py") for index in range(20)]
    with monitor_reads(root) as (opened, probes):
        state, _, events = await run_case("twenty_file_budget", [response(*calls), response(message())], tools=[TEXT_TOOL], executor=executor)
    assert state["tool_outcomes"] == {"succeeded": 4, "rejected": 16, "failed": 0, "skipped": 0}
    assert len(opened) == 4 and sum(sum(length for _, length in p.reads) for p in probes) == 262144
    assert events[-1]["read_budget"] == {"limit_bytes": 262144, "used_bytes": 262144, "remaining_bytes": 0}
    assert all(item["error"]["reason_code"] == "read_budget_exceeded" for item in envelopes(state)[4:])
    # A larger tool allowance is now disclosed through bounded artifacts. The
    # real disk-read quota remains unchanged and independent of context size.
    expanded = create_text_executor(root)
    production = ModelSettings(base_url=SETTINGS.base_url, model_name=SETTINGS.model_name,
                               log_dir=SETTINGS.log_dir, application_root=SETTINGS.application_root)
    with monitor_reads(root) as (opened, probes), patch("benchmarks.checks.support.access.SETTINGS", production):
        state, client, events = await run_case(
            "production_total_read_budget_archives_large_results",
            [response(*calls), response(message())], tools=[TEXT_TOOL], executor=expanded,
        )
    assert state["tool_outcomes"] == {"succeeded": 16, "rejected": 4, "failed": 0, "skipped": 0}
    assert len(opened) == 16 and sum(sum(length for _, length in p.reads) for p in probes) == 1048576
    assert events[-1]["read_budget"] == {"limit_bytes": 1048576, "used_bytes": 1048576, "remaining_bytes": 0}
    assert all(item["error"]["reason_code"] == "read_budget_exceeded" for item in envelopes(state)[16:])
    for item in envelopes(state)[:16]:
        observation = item["observation"]
        assert observation["mode"] == "artifact" and observation["total_chars"] == 65536
        assert all(segment["text"] == "x" * (segment["end"] - segment["start"]) for segment in observation["segments"])
        assert len(json.dumps(item, ensure_ascii=False).encode("utf-8")) <= 4096
    assert state["stop_reason"] == "completed" and state["model_calls"] == len(client.requests) == 2
    assert state["summary_calls"] == 0 and state["context_budget"]["request_sent"] is True
    assert state["context_budget"]["limit_tokens"] == 250000
    bounded = create_text_executor(root, max_bytes=8, max_total_bytes=6)
    state, client, events = await run_case(
        "cross_round_repeat_failure_credit_and_small_file",
        [response(file_call("bad_encoding", "invalid"), file_call("first", "four")),
         response(file_call("over", "four"), file_call("small", "small"), file_call("empty", "empty")), response(message())],
        tools=[TEXT_TOOL], executor=bounded,
    )
    assert state["tool_outcomes"] == {"succeeded": 3, "rejected": 1, "failed": 1, "skipped": 0}
    assert events[-1]["read_budget"]["used_bytes"] == 6
    started = [event for event in events if event["event"] == "model.started"]
    assert [event["read_budget"]["remaining_bytes"] for event in started] == [6, 2, 0]
    assert all("instructions" not in request for request in client.requests)
    assert all(assert_tools(request["tools"], [TEXT_TOOL]) for request in client.requests)
    assert [json.loads(inline_raw(item))["content"] for item in envelopes(state) if item["ok"]] == ["1234", "12", ""]
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
        for state in concurrent:
            observe_state(state)
    for state, client in zip(concurrent, clients):
        events = audit_recording(state, client, scenario="reused_executor_interleaved")
        assert events[-1]["read_budget"]["used_bytes"] == 4 and client.closed
        assert envelopes(state)[0]["ok"]
        RUNS.append(state["run_id"])
    assert len({state["run_id"] for state in states + concurrent}) == 4 and Path.cwd() == cwd
    assert read_budget_snapshot(executor)["used_bytes"] == 0

def execute():
    with TemporaryDirectory(prefix="lineage-access-contract-") as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        for name in ("graph.py", "pyproject.toml", ".gitignore", "Dockerfile", "Makefile", ".env", "无扩展名"):
            (root / name).write_text("synthetic text only: " + name, encoding="utf-8")
        with check_group("filesystem.access_budget.file_scope"):
            asyncio.run(check_file_scope(root))
        with check_group("filesystem.access_budget.pagination"):
            asyncio.run(check_pagination(root))
        with check_group("filesystem.access_budget.read_budgets"):
            asyncio.run(check_read_budgets(root))
        with check_group("filesystem.access_budget.budget_isolation"):
            asyncio.run(check_budget_isolation(root))
