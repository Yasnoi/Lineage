"""Deterministic ReAct checks; uses local responses and makes no network calls."""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from io import StringIO
import json
from types import SimpleNamespace
from unittest.mock import patch

from openai.types.responses import Response, ResponseUsage

from lineage_agent.agent import runtime
from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.entrypoints import cli
from lineage_agent.models.responses import ModelError, generate_response
from lineage_agent.tools.types import ToolArgumentError


SETTINGS = ModelSettings(base_url="https://unused.invalid/v1", model_name="local-check")
TOOL = {
    "type": "function",
    "name": "lookup_sample",
    "description": "Return a fixed value for the requested sample.",
    "parameters": {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    },
    "strict": True,
}


def message(text="done"):
    return {
        "type": "message", "id": "msg_local", "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def call(call_id="call_1", *, name="lookup_sample", arguments='{"key":"sample"}'):
    return {
        "type": "function_call", "id": f"fc_{call_id}",
        "call_id": call_id, "name": name, "arguments": arguments,
        "status": "completed",
    }


def response(*items, usage=(10, 2, 12), status="completed"):
    value = Response.model_validate({
        "id": "resp_local", "created_at": 0, "object": "response",
        "model": "local-check", "status": status,
        "output": list(items), "parallel_tool_calls": False,
        "tool_choice": "auto", "tools": [], "usage": None,
    })
    if usage is not None:
        # Construct partial usage deliberately: compatible endpoints can omit fields.
        value.usage = ResponseUsage.model_construct(
            input_tokens=usage[0], output_tokens=usage[1], total_tokens=usage[2],
            input_tokens_details={"cached_tokens": 0},
            output_tokens_details={"reasoning_tokens": 0},
        )
    return value


class LocalClient:
    """Records SDK request arguments and yields predetermined SDK responses."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.closed = False
        self.responses = SimpleNamespace(create=self.create)

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        try:
            item = next(self.replies)
        except StopIteration as exc:
            raise AssertionError("unexpected extra model request") from exc
        if isinstance(item, Exception):
            raise item
        return item

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True


async def fixed_tool(name, arguments):
    assert name == "lookup_sample"
    if set(arguments) != {"key"} or not isinstance(arguments["key"], str):
        raise ToolArgumentError("key must be a string and the only parameter")
    return f"fixed:{arguments['key']}"


async def run(replies, *, input="question", tools=None, execute_tool=None, limit=5):
    client = LocalClient(replies)
    with patch.object(runtime, "create_client", return_value=client):
        state = await runtime.run_agent(
            input, replace(SETTINGS, max_model_calls=limit),
            tools=tools, execute_tool=execute_tool,
        )
    assert client.closed, "client must close after every run"
    assert state["model_calls"] == len(client.requests)
    return state, client


def counts(state, model, requested, executed, reason="completed"):
    assert (state["model_calls"], state["tool_requests"], state["tool_calls"]) == (
        model, requested, executed
    ), state
    assert state["stop_reason"] == reason, state


def outputs(state):
    return [item for item in state["input"] if item.get("type") == "function_call_output"]


async def check_direct_answer():
    state, client = await run([response(message("direct"))])
    counts(state, 1, 0, 0)
    assert state["result"]["text"] == "direct"
    assert not client.requests[0].get("tools")
    assert len(state["input"]) == 2
    assert state["usage_totals"] == {
        "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
        "missing_usage_calls": 0,
    }


async def check_unified_model_input():
    items = [{"role": "user", "content": "structured"}]
    client = LocalClient([response(message()), response(call())])
    await generate_response(client, "local-check", "string input")
    result = await generate_response(client, "local-check", items, tools=[TOOL])
    assert client.requests[0]["input"] == "string input"
    assert client.requests[1]["input"] == items
    assert client.requests[1]["tools"] == [TOOL]
    assert result["text"] == "" and result["tool_calls"][0]["call_id"] == "call_1"
    for request in client.requests:
        assert request["stream"] is False and request["store"] is False
        assert "previous_response_id" not in request


async def check_tool_loop():
    state, client = await run(
        [response(call()), response(message("fixed:sample"), usage=(20, 3, 23))],
        tools=[TOOL], execute_tool=fixed_tool,
    )
    counts(state, 2, 1, 1)
    output = outputs(state)[0]
    assert output["call_id"] == "call_1"
    assert json.loads(output["output"]) == {"ok": True, "result": "fixed:sample"}
    assert client.requests[1]["input"][-1] == output
    assert state["result"]["usage"] == {
        "input_tokens": 20, "output_tokens": 3, "total_tokens": 23,
    }
    assert state["usage_totals"] == {
        "input_tokens": 30, "output_tokens": 5, "total_tokens": 35,
        "missing_usage_calls": 0,
    }


async def check_multiple_requests_and_context():
    reasoning = {
        "type": "reasoning", "id": "rs_local", "summary": [],
        "encrypted_content": "opaque-encrypted-content", "provider_extension": "keep-me",
    }
    initial = [{"role": "user", "content": "two samples"}]
    original = deepcopy(initial)
    events = []

    async def sequential_tool(name, arguments):
        events.append(("start", arguments["key"]))
        await asyncio.sleep(0)
        events.append(("end", arguments["key"]))
        return await fixed_tool(name, arguments)

    first = response(
        reasoning, message("checking"), call("a", arguments='{"key":"a"}'),
        call("b", arguments='{"key":"b"}'),
    )
    state, client = await run(
        [first, response(message())], input=initial,
        tools=[TOOL], execute_tool=sequential_tool,
    )
    counts(state, 2, 2, 2)
    assert initial == original, "caller-owned context must remain unchanged"
    assert events == [("start", "a"), ("end", "a"), ("start", "b"), ("end", "b")]
    history = client.requests[1]["input"]
    assert len(history) == 7 and len(state["input"]) == 8
    assert history[1:5] == first.model_dump(mode="json", by_alias=True, exclude_unset=True)["output"]
    assert history[1]["provider_extension"] == "keep-me"
    assert history[1]["encrypted_content"] == "opaque-encrypted-content"
    assert [item["call_id"] for item in outputs(state)] == ["a", "b"]


async def check_limits():
    for limit in (1, 5, 15):
        state, _ = await run(
            [response(call(f"call_{i}")) for i in range(limit)],
            tools=[TOOL], execute_tool=fixed_tool, limit=limit,
        )
        counts(state, limit, limit, limit - 1, "max_model_calls")
        assert len(outputs(state)) == limit - 1
    state, _ = await run([response(message())], limit=1)
    counts(state, 1, 0, 0)


async def check_tool_errors():
    async def failing_tool(name, arguments):
        raise RuntimeError("synthetic executor failure")

    scenarios = [
        (call(name="unknown"), fixed_tool, "unknown_tool", 0),
        (call(arguments="not-json"), fixed_tool, "invalid_arguments", 0),
        (call(arguments="[]"), fixed_tool, "invalid_arguments", 0),
        (call(arguments='{"key":42}'), fixed_tool, "invalid_arguments", 1),
        (call(), failing_tool, "execution_error", 1),
    ]
    for request, executor, code, attempts in scenarios:
        state, client = await run(
            [response(request), response(message("reported error"))],
            tools=[TOOL], execute_tool=executor,
        )
        counts(state, 2, 1, attempts)
        result = outputs(state)[0]
        assert result["call_id"] == request["call_id"]
        envelope = json.loads(result["output"])
        assert envelope["ok"] is False and envelope["error"]["code"] == code
        assert client.requests[1]["input"][-1] == result


async def check_model_failures():
    state, _ = await run(
        [response(call()), ModelError("synthetic model failure")],
        tools=[TOOL], execute_tool=fixed_tool,
    )
    counts(state, 2, 1, 1, "error")
    assert state["result"] is None and state["error"]
    assert state["usage_totals"]["total_tokens"] == 12
    assert state["usage_totals"]["missing_usage_calls"] == 1
    state, _ = await run([response(message("partial"), status="incomplete")])
    counts(state, 1, 0, 0, "error")
    assert state["result"]["raw_response"]["status"] == "incomplete"
    assert state["usage_totals"]["total_tokens"] == 12
    for malformed in (response(), response(call(call_id="")), response(call(), call())):
        state, _ = await run([malformed], tools=[TOOL], execute_tool=fixed_tool)
        assert state["stop_reason"] == "error" and state["error"]
        assert state["model_calls"] == 1 and state["tool_calls"] == 0


async def check_usage_gaps_and_run_isolation():
    state, _ = await run(
        [response(call(), usage=(10, None, None)), response(message(), usage=None)],
        tools=[TOOL], execute_tool=fixed_tool,
    )
    assert state["result"]["usage"] is None
    assert state["usage_totals"] == {
        "input_tokens": 10, "output_tokens": None, "total_tokens": None,
        "missing_usage_calls": 2,
    }
    state, _ = await run([response(message(), usage=(0, 0, 0))])
    counts(state, 1, 0, 0)
    assert state["usage_totals"] == {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "missing_usage_calls": 0,
    }
    state, _ = await run([response(message(), usage=None)])
    assert state["usage_totals"] == {
        "input_tokens": None, "output_tokens": None, "total_tokens": None,
        "missing_usage_calls": 1,
    }


def check_cli_exit_codes():
    for reason, expected in (("completed", 0), ("error", 1), ("max_model_calls", 3)):
        async def invoke(*args, **kwargs):
            return {
                "stop_reason": reason, "error": "synthetic error",
                "result": {"text": "done"},
            }

        with (
            patch("sys.argv", ["lineage-agent", "question"]),
            patch.object(cli, "load_settings", return_value=SETTINGS),
            patch.object(cli, "run_agent", side_effect=invoke),
            redirect_stdout(StringIO()), redirect_stderr(StringIO()),
        ):
            assert cli.main() == expected
    with (
        patch("sys.argv", ["lineage-agent", "question"]),
        patch.object(cli, "load_settings", side_effect=ConfigurationError("missing config")),
        redirect_stderr(StringIO()),
    ):
        assert cli.main() == 2
    with (
        patch("sys.argv", ["lineage-agent", "--verify-tools", "question"]),
        redirect_stderr(StringIO()),
    ):
        try:
            cli.main()
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("CLI unexpectedly accepts a verification flag")


def main():
    checks = (
        check_direct_answer, check_unified_model_input, check_tool_loop,
        check_multiple_requests_and_context, check_limits, check_tool_errors,
        check_model_failures, check_usage_gaps_and_run_isolation,
    )
    for check in checks:
        asyncio.run(check())
        print(f"PASS {check.__name__}")
    check_cli_exit_codes()
    print("PASS check_cli_exit_codes")
    print(f"PASS {len(checks) + 1} deterministic check groups; no real endpoint calls")


if __name__ == "__main__":
    main()
