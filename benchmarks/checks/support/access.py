"""Reusable fixtures and evidence assertions for access."""


import json
from benchmarks.checks.support.observations import assert_tools
from benchmarks.checks.support.events import observe_state
from benchmarks.checks.support.runtime import LocalClient, SETTINGS, audit_recording, call, outputs
from benchmarks.core.files import ROOT
from dataclasses import replace
from lineage_agent.agent import runtime
from lineage_agent.agent.graph import FINAL_ANSWER_INSTRUCTIONS
from unittest.mock import patch
from uuid import uuid4

CHECK_DIR = ROOT / "logs" / "checks" / str(uuid4())

SETTINGS = replace(SETTINGS, log_dir=CHECK_DIR)

RUNS = []

async def run_case(scenario, replies, *, executor=None, tools=None, limit=5, client=None, failed_writes=None,
                   prompt="Inspect the provided project as source material."):
    client = client or LocalClient(replies)
    with patch.object(runtime, "create_client", return_value=client):
        state = await runtime.run_agent(
            prompt,
            replace(SETTINGS, max_model_calls=limit), tools=tools, execute_tool=executor,
        )
        observe_state(state)
    assert client.closed and state["model_calls"] == len(client.requests)
    events = audit_recording(state, client, scenario=scenario, failed_writes=failed_writes)
    generated = outputs(state)
    for index, request in enumerate(client.requests):
        returned = [item for item in request["input"] if item.get("type") == "function_call_output"]
        assert returned == generated[:len(returned)]
        remaining = limit - index
        if remaining == 1:
            assert request["instructions"] == FINAL_ANSWER_INSTRUCTIONS
            assert request["tool_choice"] == "none"
        else:
            assert "instructions" not in request and "tool_choice" not in request
        assert request["input"] == state["input"][:len(request["input"])], "management metadata changed actual input"
        assert_tools(request["tools"], tools or [])
        assert not {"remaining_model_calls", "read_budget", "read_budget_error"} & request.keys()
        if index:
            previous = client.requests[index - 1]["input"]
            assert request["input"][:len(previous)] == previous, "existing context was rewritten"
    RUNS.append(state["run_id"])
    return state, client, events

def file_call(identifier, path, *, name="read_text_file", after=None):
    arguments = {"path": path}
    if name == "list_directory":
        arguments["after"] = after
    return call(identifier, name=name, arguments=json.dumps(arguments, ensure_ascii=False))

def envelopes(state):
    return [json.loads(item["output"]) for item in outputs(state)]

class CloseFailureClient(LocalClient):
    close_calls = 0

    async def __aexit__(self, *_):
        self.closed = True
        self.close_calls += 1
        raise OSError("PRIVATE_CLOSE_ERROR")
