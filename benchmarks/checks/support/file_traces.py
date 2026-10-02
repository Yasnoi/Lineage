"""Reusable fixtures and evidence assertions for file traces."""


import json
from benchmarks.checks.support.observations import assert_observed
from benchmarks.checks.support.events import observe_state
from benchmarks.checks.support.runtime import LocalClient, audit_recording, call, outputs
from benchmarks.checks.support.tracing import capture
from benchmarks.core.files import ROOT
from copy import deepcopy
from dataclasses import replace
from lineage_agent.agent import runtime
from lineage_agent.config.settings import ModelSettings
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL
from lineage_agent.tools.filesystem.text import TEXT_TOOL, create_text_executor
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

CHECK_DIR = ROOT / "logs" / "checks" / str(uuid4())

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
    observe_state(state)
    assert client.closed and state["log_complete"]
    events = audit_recording(state, client, scenario=scenario)
    generated = outputs(state)
    for request in client.requests:
        actual = [item for item in request["input"] if item.get("type") == "function_call_output"]
        assert actual == generated[:len(actual)]
    successful = [json.loads(item["output"]) for item in generated if json.loads(item["output"])["ok"]]
    assert len(successful) == len(observed)
    for envelope, (name, _, raw) in zip(successful, observed, strict=True):
        assert_observed(envelope, raw, name)
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
