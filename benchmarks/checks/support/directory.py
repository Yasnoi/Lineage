"""Reusable fixtures and evidence assertions for directory."""


import builtins
import io
import json
from benchmarks.checks.support.observations import assert_observed
import os
from benchmarks.checks.support.events import coverage_gap, observe_state
from benchmarks.checks.support.runtime import (
    LocalClient,
    audit_recording,
    call,
    counts,
    message,
    outputs,
    read_events,
    response,
)
from benchmarks.core.files import ROOT
from contextlib import contextmanager
from copy import deepcopy
from lineage_agent.agent import runtime
from lineage_agent.config.settings import ModelSettings
from lineage_agent.tools.filesystem import directory
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL, create_directory_executor
from lineage_agent.tools.types import TOOL_ARGUMENT_REASONS
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

CHECK_ID = str(uuid4())

CHECK_DIR = ROOT / "logs" / "checks" / CHECK_ID

SETTINGS = ModelSettings(
    base_url="https://unused.invalid/v1", model_name="filesystem-check", log_dir=CHECK_DIR, application_root=CHECK_DIR,
)

SKIPPED = []

def directory_call(call_id, path=".", *, arguments=None):
    return call(
        call_id, name="list_directory",
        arguments=json.dumps({"path": path}, ensure_ascii=False) if arguments is None else arguments,
    )

def fixture(base):
    root = base / "BOUND_ROOT_PRIVATE"
    root.mkdir()
    (root / "文件 private.txt").write_text("BODY_SECRET_DO_NOT_READ", encoding="utf-8")
    (root / "child private").mkdir()
    (root / "child private" / "nested.txt").write_text("NESTED_BODY_SECRET", encoding="utf-8")
    (root / "空目录 private").mkdir()
    (root / ".hidden private").write_text("hidden", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".gitignore").write_text("*.txt", encoding="utf-8")
    sibling = base / "BOUND_ROOT_PRIVATE-neighbor"
    sibling.mkdir()
    (sibling / "OUTSIDE_PRIVATE.txt").write_text("OUTSIDE_BODY_SECRET", encoding="utf-8")
    (sibling / "external child").mkdir()
    return root, sibling

@contextmanager
def forbid_content_reads(root):
    """Trip on Python file-content reads within the fixture, without changing tool output."""
    original_builtin, original_io = builtins.open, io.open

    def guard(opener):
        def checked(file, mode="r", *args, **kwargs):
            if not isinstance(file, int):
                candidate = Path(file).absolute()
                if candidate.is_relative_to(root) and ("r" in mode or "+" in mode):
                    raise AssertionError("directory tool attempted a file-content read")
            return opener(file, mode, *args, **kwargs)
        return checked

    with patch.object(builtins, "open", guard(original_builtin)), patch.object(io, "open", guard(original_io)):
        yield

def snapshot(root):
    return {
        str(path.relative_to(root)): ("directory" if path.is_dir() else path.read_bytes())
        for path in root.rglob("*")
    }

async def run_case(scenario, root, replies, *, max_entries=100, settings=None, executor=None):
    actual = executor or create_directory_executor(root, max_entries=max_entries)
    observed = []

    async def tracked(name, arguments):
        record = {"name": name, "arguments": deepcopy(arguments)}
        observed.append(record)
        try:
            value = await actual(name, arguments)
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        record["result"] = value
        return value

    client = LocalClient(replies)
    before_cwd = Path.cwd()
    with patch.object(runtime, "create_client", return_value=client), forbid_content_reads(root):
        state = await runtime.run_agent(
            "List the requested directory metadata.", settings or SETTINGS,
            tools=[DIRECTORY_TOOL], execute_tool=tracked,
        )
        observe_state(state)
    assert client.closed and state["log_complete"] and Path.cwd() == before_cwd
    assert state["model_calls"] == len(client.requests)
    audit_recording(state, client, scenario=scenario)
    generated = outputs(state)
    by_call_id = {item["call_id"]: item for item in generated}
    assert len(by_call_id) == len(generated)
    for request in client.requests:
        requested_ids = [
            item["call_id"] for item in request["input"]
            if item.get("type") == "function_call"
        ]
        submitted = [
            item for item in request["input"]
            if item.get("type") == "function_call_output"
        ]
        # Every completed batch appears exactly once, in order, with the entire
        # generated payload intact in every later actual SDK request.
        assert submitted == [by_call_id[call_id] for call_id in requested_ids]
    if len(client.requests) > 1:
        # Compare full protocol items, not only call IDs or log self-consistency.
        final_items = state["result"]["output_items"]
        assert client.requests[-1]["input"] == state["input"][:-len(final_items)]
        submitted = [
            item for item in client.requests[-1]["input"]
            if item.get("type") == "function_call_output"
        ]
        assert submitted == generated
        successful_values = [json.loads(item["output"]) for item in generated if json.loads(item["output"])["ok"]]
        originals = [item["result"] for item in observed if "result" in item]
        assert len(successful_values) == len(originals)
        for envelope, raw in zip(successful_values, originals, strict=True):
            assert_observed(envelope, raw, "list_directory")
    log_text = Path(state["log_path"]).read_text(encoding="utf-8")
    for marker in (
        str(root), root.name, "文件 private.txt", "child private", "nested.txt",
        "空目录 private", ".hidden private", "BODY_SECRET", "OUTSIDE_PRIVATE",
    ):
        assert marker not in log_text, marker
    for record in observed:
        if "result" in record:
            for entry in json.loads(record["result"])["entries"]:
                assert entry["name"] not in log_text
    return state, client, observed

def decoded_results(state):
    return [json.loads(item["output"]) for item in outputs(state)]

class ScanProbe:
    """Delegates to actual scandir; measures bounded consumption or injects an I/O fault."""

    def __init__(self, scan, *, fail_after=None, fail_stat_at=None):
        self.scan = scan
        self.fail_after = fail_after
        self.fail_stat_at = fail_stat_at
        self.yielded = []
        self.closed = False

    def __enter__(self):
        self.scan.__enter__()
        return self

    def __exit__(self, *args):
        self.closed = True
        return self.scan.__exit__(*args)

    def __iter__(self):
        return self

    def __next__(self):
        if self.fail_after is not None and len(self.yielded) == self.fail_after:
            raise OSError("INJECTED_PRIVATE_IO_FAILURE")
        item = next(self.scan)
        self.yielded.append(item.name)
        if self.fail_stat_at == len(self.yielded):
            return EntryStatFailure(item)
        return item

class EntryStatFailure:
    """Keep a real enumerated entry but inject failure while classifying it."""

    def __init__(self, entry):
        self.entry = entry

    def __getattr__(self, name):
        return getattr(self.entry, name)

    def stat(self, **kwargs):
        assert kwargs == {"follow_symlinks": False}
        raise OSError("INJECTED_PRIVATE_CLASSIFICATION_FAILURE")

async def rejected_then_success(root, scenario, *, path=None, arguments=None, attempts=2, expected_code="invalid_arguments", outcome="rejected", expected_reason=None):
    requested = [directory_call("fs_bad", path, arguments=arguments), directory_call("fs_good", ".")]
    real_scandir = os.scandir
    scanned = []

    def tracked_scan(value):
        if not isinstance(value, (str, os.PathLike)) or not Path(value).is_relative_to(root.parent):
            return real_scandir(value)
        scanned.append(Path(value))
        return real_scandir(value)

    with patch.object(directory.os, "scandir", tracked_scan):
        state, client, observed = await run_case(scenario, root, [response(*requested), response(message())])
    counts(state, 2, 2, attempts)
    assert state["tool_outcomes"] == {
        "succeeded": 1, "rejected": int(outcome == "rejected"),
        "failed": int(outcome == "failed"), "skipped": 0,
    }
    first, second = decoded_results(state)
    assert first["ok"] is False and first["error"]["code"] == expected_code and second["ok"] is True
    assert [item["call_id"] for item in outputs(state)] == ["fs_bad", "fs_good"]
    assert scanned == [root], scanned
    assert len(observed) == attempts
    if expected_reason is not None:
        diagnostics = [
            event for event in read_events(state)
            if event.get("call_id") == "fs_bad"
            and event["event"] in ("tool.rejected", "tool.result_created")
        ]
        assert len(diagnostics) == 2
        for event in diagnostics:
            assert event["reason_code"] == expected_reason
            assert event["reason_message"] == TOOL_ARGUMENT_REASONS[expected_reason]
        assert first == {
            "ok": False,
            "error": {
                "code": "invalid_arguments", "message": "工具参数未通过执行器校验。",
                "reason_code": expected_reason,
                "reason_message": TOOL_ARGUMENT_REASONS[expected_reason],
            },
        }
    return state, client, observed

def skip_link(kind, exc):
    value = {"scenario": kind, "status": "SKIP", "reason": type(exc).__name__, "actual_link_created": False}
    SKIPPED.append(value)
    coverage_gap(kind, value["reason"])
    print(json.dumps(value))
