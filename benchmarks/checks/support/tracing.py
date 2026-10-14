"""Reusable fixtures and evidence assertions for tracing."""


import json
import os
import re
import requests
import sys
import threading
from benchmarks.checks.support.events import observe_state
from benchmarks.checks.support.runtime import LocalClient, call, message, response
from benchmarks.core.files import ROOT
from contextlib import contextmanager, redirect_stderr
from copy import deepcopy
from io import StringIO
from lineage_agent.agent import runtime
from lineage_agent.config.settings import ModelSettings
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL
from unittest.mock import patch
from uuid import UUID, uuid4

CHECK_DIR = ROOT / "logs" / "checks" / str(uuid4())

SETTINGS = ModelSettings(
    base_url="https://unused.invalid/v1", model_name="tracing-check", log_dir=CHECK_DIR, application_root=CHECK_DIR,
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
    observe_state(state)
    assert client.closed and state["log_complete"]
    return state, client

def stable(state):
    value = deepcopy({key: item for key, item in state.items() if key not in {"run_id", "session_id", "log_path"}})
    # Generated identities and hashes include per-run UUIDs. Normalize their
    # correspondence, retaining turn order, receipts, byte counts and all bodies.
    active = value.get("active_context")
    identities = {state["run_id"]: "<current-run>"}
    if active:
        active["version_id"] = "<active-version>"
        if active["parent_version_id"] is not None:
            active["parent_version_id"] = "<parent-version>"
        for index, turn in enumerate(active["turns"]):
            identities.setdefault(turn["run_id"], f"<turn-{index}>")
            turn["run_id"] = identities[turn["run_id"]]
            if "checkpoint_id" in turn:
                turn["checkpoint_id"] = f"<checkpoint-{index}>"
    watermarks = value.get("record_files", {})
    for index, run_id in enumerate(watermarks):
        identities.setdefault(run_id, f"<record-run-{index}>")

    def identity(text):
        if isinstance(text, str):
            for identifier, replacement in identities.items():
                text = text.replace(identifier, replacement)
        return text

    if active:
        active["current_run_id"] = identity(active.get("current_run_id"))
        for group in active.get("groups", []):
            group["group_id"] = identity(group["group_id"])
        active["item_refs"] = [identity(reference) for reference in active.get("item_refs", [])]
    conversation_id = value.get("conversation_id")
    if conversation_id is not None:
        assert isinstance(conversation_id, str) and str(UUID(conversation_id)) == conversation_id
        value["conversation_id"] = (identity(conversation_id) if conversation_id == state["run_id"]
                                    else "<session-conversation>")
    resource_view_id = value.get("resource_view_id")
    if resource_view_id is not None:
        assert isinstance(resource_view_id, str) and str(UUID(resource_view_id)) == resource_view_id
        value["resource_view_id"] = "<confirmed-resource-view>"

    digests = {}
    def digest(value):
        assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
        return digests.setdefault(value, f"<digest-{len(digests)}>")

    for watermark in watermarks.values():
        assert watermark["format_version"] == 1
        assert type(watermark["record_count"]) is int and watermark["record_count"] >= 0
        assert type(watermark["byte_end"]) is int and watermark["byte_end"] >= watermark["record_count"]
    for record in value.get("record_batch", []):
        # Receipts identify immutable originals; they no longer contain a copy
        # of the raw item. Their coordinates stay exact across equivalent runs.
        if "sha256" in record:
            mark = watermarks[record["run_id"]]
            assert type(record["sequence"]) is int and 0 <= record["sequence"] < mark["record_count"]
            assert record["source_ref"] == f"history://record/{record['run_id']}/{record['sequence']}"
            assert type(record["offset"]) is int and record["offset"] >= 0
            assert type(record["length"]) is int and record["length"] > 0
            assert record["offset"] + record["length"] <= mark["byte_end"]
            record["sha256"] = digest(record["sha256"])
        for key in ("run_id", "group_id", "source_ref"):
            if key in record:
                record[key] = identity(record[key])
    for watermark in watermarks.values():
        watermark["prefix_sha256"] = digest(watermark["prefix_sha256"])
    if "record_files" in value:
        value["record_files"] = {identity(run_id): watermark for run_id, watermark in watermarks.items()}
    metrics = value.get("record_metrics", {})
    if "checkpoint_serialized_bytes" in metrics:
        measured = metrics["checkpoint_serialized_bytes"]
        assert type(measured) is int and measured >= 0
        # Native checkpoint serialization can compress distinct UUIDs to
        # different lengths. Keep availability/positive storage use comparable.
        if measured:
            metrics["checkpoint_serialized_bytes"] = "<positive-byte-measurement>"
    return value


def hanging_upload_child():
    from lineage_agent.entrypoints import cli

    client = LocalClient([response(message("done"))])
    with capture("hang") as (_, stderr), patch.object(cli, "load_settings", return_value=SETTINGS), patch.object(
        runtime, "create_client", return_value=client,
    ), patch.object(sys, "argv", ["lineage-agent", "a harmless prompt"]):
        assert cli.main() == 0
    assert stderr.getvalue()
    print("incomplete upload; CLI returned")
