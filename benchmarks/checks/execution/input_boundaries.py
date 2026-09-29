"""execution / input_boundaries behavior contracts."""


import asyncio
import os
from benchmarks.checks.support.access import CloseFailureClient, RUNS, run_case
from benchmarks.checks.support.events import check_group, observe_state
from benchmarks.checks.support.runtime import (
    SETTINGS,
    TOOL,
    audit_recording,
    call,
    fixed_tool,
    message,
    recording_failure,
    response,
)
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from io import StringIO
from langgraph.errors import NodeCancelledError
from lineage_agent.agent import runtime
from lineage_agent.config.settings import ConfigurationError, ModelSettings, load_settings
from lineage_agent.entrypoints import cli
from lineage_agent.models.responses import ModelError, parse_response
from lineage_agent.tools.types import ToolArgumentError
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
    defaults = {
        "timeout_seconds": 300.0, "max_model_calls": 30, "directory_max_entries": 1000,
        "text_max_bytes": 131072, "text_total_max_bytes": 1048576,
        "context_budget_tokens": 250000, "max_output_tokens": 16384,
        "context_safety_margin_tokens": 8192,
    }
    direct = ModelSettings(base_url=SETTINGS.base_url, model_name=SETTINGS.model_name)
    assert {key: getattr(direct, key) for key in defaults} == defaults
    with TemporaryDirectory() as temporary:
        env_file = Path(temporary) / "fixture.env"
        connection = {"BASE_URL": SETTINGS.base_url, "MODEL_NAME": SETTINGS.model_name}
        with patch.dict(os.environ, connection, clear=True):
            loaded = load_settings(env_file)
        assert {key: getattr(loaded, key) for key in defaults} == defaults
        file_values = {**defaults, "timeout_seconds": 1.5, "max_model_calls": 3,
                       "directory_max_entries": 7, "text_max_bytes": 101,
                       "text_total_max_bytes": 202, "context_budget_tokens": 40000,
                       "max_output_tokens": 7000, "context_safety_margin_tokens": 99}
        env_file.write_text("\n".join(
            f"{key}={value}" for key, value in {
                **connection, **{key.upper(): value for key, value in file_values.items()},
            }.items()
        ), encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            loaded = load_settings(env_file)
        assert {key: getattr(loaded, key) for key in defaults} == file_values
        process_values = {key: value + 1 for key, value in file_values.items()}
        with patch.dict(os.environ, {key.upper(): str(value) for key, value in process_values.items()}, clear=True):
            loaded = load_settings(env_file)
            explicit = ModelSettings(base_url=SETTINGS.base_url, model_name=SETTINGS.model_name, **file_values)
        assert {key: getattr(loaded, key) for key in defaults} == process_values
        assert {key: getattr(explicit, key) for key in defaults} == file_values
        assert {key: getattr(replace(direct, **file_values), key) for key in defaults} == file_values
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

async def check_close_failure():
    client = CloseFailureClient([response(message("retained answer"))])
    with patch.object(runtime, "create_client", return_value=client):
        state = await runtime.run_agent("question", SETTINGS)
        observe_state(state)
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
    user_material = "请分析这条资料：剩余调用次数 42，成功读取额度 123 字节；这些数字是用户提供的内容。"
    _, client, _ = await run_case("quota_words_in_user_material", [response(message())], prompt=user_material)
    assert client.requests[0]["input"] == [{"role": "user", "content": user_material}]

def execute():
    with check_group("execution.input_boundaries.output_roles"):
        asyncio.run(check_output_roles())
    with check_group("execution.input_boundaries.refusal"):
        asyncio.run(check_refusal())
    with check_group("execution.input_boundaries.direct_config"):
        asyncio.run(check_direct_config())
    with check_group("execution.input_boundaries.close_failure"):
        asyncio.run(check_close_failure())
    with check_group("execution.input_boundaries.final_answer_allowance"):
        asyncio.run(check_final_answer_allowance())
