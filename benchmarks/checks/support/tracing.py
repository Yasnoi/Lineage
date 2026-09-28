"""Reusable fixtures and evidence assertions for tracing."""


import json
import os
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
from uuid import uuid4

CHECK_DIR = ROOT / "logs" / "checks" / str(uuid4())

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
    observe_state(state)
    assert client.closed and state["log_complete"]
    return state, client

def stable(state):
    value = deepcopy({key: value for key, value in state.items() if key not in {"run_id", "session_id", "log_path"}})
    # Independent runs allocate independent version and checkpoint identities.
    # Keep the descriptor shape, turn order and all behavioral data comparable.
    active = value.get("active_context")
    if active:
        active["version_id"] = "<active-version>"
        if active["parent_version_id"] is not None:
            active["parent_version_id"] = "<parent-version>"
        for index, turn in enumerate(active["turns"]):
            turn["run_id"] = f"<turn-{index}>"
            turn["checkpoint_id"] = f"<checkpoint-{index}>"
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
