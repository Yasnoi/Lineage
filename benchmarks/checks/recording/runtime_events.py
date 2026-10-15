"""recording / runtime_events behavior contracts."""


import asyncio
import json
from benchmarks.checks.support.observations import inline_raw, assert_observed
from benchmarks.checks.support.events import check_group, evidence, observe_harness
from benchmarks.checks.support.runtime import (
    CHECK_DIR,
    TOOL,
    assert_failed_event,
    call,
    counts,
    fixed_tool,
    message,
    outputs,
    read_events,
    recording_failure,
    response,
    run,
)
from copy import deepcopy
from lineage_agent.models.responses import ModelError
from lineage_agent.recording.recorder import RunRecorder
from lineage_agent.tools.execution import execute_calls
from lineage_agent.artifacts.store import ArtifactStore
from lineage_agent.observations.manager import ObservationManager
from lineage_agent.tools.types import (
    TOOL_ARGUMENT_REASONS,
    TOOL_EXECUTION_REASONS,
    ToolArgumentError,
    ToolExecutionError,
)
from pathlib import Path

async def check_safe_summaries():
    secret = "FORBIDDEN_SYNTHETIC_SECRET_73BDA"
    long_text = secret + "文" * 5000
    sensitive_field = "api_key_FORBIDDEN_FIELD_MARKER"
    schema = deepcopy(TOOL)
    properties = {
        "key": {"type": "string"}, sensitive_field: {"type": "string"},
        "nested": {"type": "object"}, "rows": {"type": "array"},
        "n" * 100: {"type": "string"},
        **{f"field_{index}": {"type": "string"} for index in range(30)},
    }
    schema["parameters"]["properties"] = properties
    arguments = {"unknown_" + secret: long_text, **{name: long_text for name in properties}}
    arguments["nested"] = {secret: long_text}
    arguments["rows"] = [long_text] * 3
    seen = []

    async def echo_tool(name, values):
        seen.append(deepcopy(values))
        return long_text

    state, client = await run(
        [response(call("safe_summary", arguments=json.dumps(arguments))), response(message(long_text))],
        tools=[schema], execute_tool=echo_tool,
        # This fixture exercises redaction of very large arguments, not refusal.
        context_budget_tokens=8_000_000,
    )
    assert seen == [arguments]
    output = outputs(state)[0]
    assert_observed(json.loads(output["output"]), long_text, "lookup_sample")
    assert json.loads(output["output"])["observation"]["mode"] == "artifact"
    assert client.requests[1]["input"][-1] == output
    log_text = Path(state["log_path"]).read_text(encoding="utf-8")
    for forbidden in (secret, sensitive_field, "unknown_" + secret, "n" * 100):
        assert forbidden not in log_text
    summaries = [
        value for event in read_events(state) for key, value in event.items()
        if key.endswith("_summary")
    ]
    assert summaries
    for summary in summaries:
        assert len(json.dumps(summary, ensure_ascii=False).encode("utf-8")) <= 1024
        if "fields" in summary:
            assert len(summary["fields"]) <= 16
            assert all(len(field["name"]) <= 64 for field in summary["fields"])
            assert summary.get("truncated") is True and summary["omitted_fields"] > 0
    # Executor failures and unparseable parameters must not expose raw content.
    async def secret_failure(name, values):
        raise RuntimeError(long_text)

    failed, _ = await run(
        [response(call("secret_failure")), response(message())],
        tools=[TOOL], execute_tool=secret_failure,
    )
    malformed, _ = await run(
        [response(call("secret_json", arguments=long_text)), response(message())],
        tools=[TOOL], execute_tool=fixed_tool,
    )
    for item in (failed, malformed):
        assert secret not in Path(item["log_path"]).read_text(encoding="utf-8")

async def check_safe_rejection_reasons():
    secret = "FORBIDDEN_REJECTION_TEXT_79B12"
    for reason, expected in (
        ("target_not_directory", "target_not_directory"),
        (secret, "argument_validation_failed"),
        ([secret], "argument_validation_failed"),
        (None, "argument_validation_failed"),
    ):
        async def reject(name, arguments):
            failure = ToolArgumentError(secret)
            # Even a custom executor replacing the attribute cannot add raw text.
            failure.reason_code = reason
            raise failure

        state, _ = await run(
            [response(call("safe_rejection")), response(message())],
            tools=[TOOL], execute_tool=reject,
        )
        counts(state, 2, 1, 1)
        assert state["tool_outcomes"] == {
            "succeeded": 0, "rejected": 1, "failed": 0, "skipped": 0,
        }
        assert json.loads(outputs(state)[0]["output"]) == {
            "ok": False,
            "error": {
                "code": "invalid_arguments", "message": "工具参数未通过执行器校验。",
                "reason_code": expected, "reason_message": TOOL_ARGUMENT_REASONS[expected],
            },
        }
        diagnostics = [
            event for event in read_events(state)
            if event["event"] in ("tool.rejected", "tool.result_created")
        ]
        assert len(diagnostics) == 2
        for event in diagnostics:
            assert event["reason_code"] == expected
            assert event["reason_message"] == TOOL_ARGUMENT_REASONS[expected]
        assert secret not in Path(state["log_path"]).read_text(encoding="utf-8")
    assert ToolArgumentError(secret, reason_code=secret).reason_code == "argument_validation_failed"

async def check_safe_execution_reasons():
    secret = "FORBIDDEN_EXECUTION_REASON_108e"
    for reason, expected in (
        ("io_error", "io_error"), (secret, "execution_failed"),
        ([secret], "execution_failed"), (None, "execution_failed"),
    ):
        async def fail(name, arguments):
            failure = ToolExecutionError(reason_code="io_error")
            failure.reason_code = reason
            failure.args = (secret,)
            raise failure

        state, client = await run(
            [response(call("safe_execution")), response(message())],
            tools=[TOOL], execute_tool=fail,
        )
        counts(state, 2, 1, 1)
        assert state["tool_outcomes"] == {
            "succeeded": 0, "rejected": 0, "failed": 1, "skipped": 0,
        }
        error = json.loads(outputs(state)[0]["output"])["error"]
        assert error == {
            "code": "execution_error", "message": "工具执行失败。",
            "reason_code": expected, "reason_message": TOOL_EXECUTION_REASONS[expected],
        }
        events = read_events(state)
        diagnostics = [item for item in events if item["event"] in ("tool.failed", "tool.result_created")]
        assert len(diagnostics) == 2
        assert all(item["reason_code"] == expected and item["reason_message"] == error["reason_message"] for item in diagnostics)
        assert secret not in json.dumps((state, client.requests, events))
    assert ToolExecutionError(reason_code=secret).reason_code == "execution_failed"

async def governed_calls(*args, **kwargs):
    with ArtifactStore() as store:
        manager = ObservationManager(store.bind_session("recording-hook-contract"))
        return await execute_calls(*args, observation_manager=manager, **kwargs)


async def check_tool_trace_hooks():
    observe_harness("tool_executor")
    class Tracer:
        def __init__(self):
            self.events = []

        def tool_start(self, request, **metadata):
            token = request["call_id"]
            self.events.append(("start", token, metadata))
            return token

        def tool_end(self, token, **details):
            self.events.append(("end", token, details))

        def tool_skipped(self, request, **details):
            self.events.append(("skipped", request["call_id"], details))

    requests = [
        call("unknown", name="not_available"), call("parsing", arguments="not-json"),
        call("validation", arguments='{"key":42}'),
        call("failure", arguments='{"key":"fail"}'), call("success"),
    ]

    async def executor(name, arguments):
        if arguments.get("key") == "fail":
            raise RuntimeError("untrusted tracer-hook failure text")
        return await fixed_tool(name, arguments)

    expected = await governed_calls(requests, [TOOL], executor)
    tracer = Tracer()
    actual = await governed_calls(requests, [TOOL], executor, tracer=tracer, step_id="step-test")
    assert actual == expected
    assert [(kind, identifier) for kind, identifier, _ in tracer.events] == [
        (kind, request["call_id"]) for request in requests for kind in ("start", "end")
    ]
    for index, (start, end) in enumerate(zip(tracer.events[::2], tracer.events[1::2]), 1):
        assert start[2] == {"step_id": "step-test", "request_index": index}
        assert end[2]["outcome"] == (
            "succeeded" if index == 5 else "failed" if index == 4 else "rejected"
        )
        assert end[2]["output"] == json.loads(actual["output_items"][index - 1]["output"])
    assert tracer.events[5][2]["reason_code"] == "argument_validation_failed"

    tracer = Tracer()
    with recording_failure("tool.started"):
        recorder = RunRecorder(CHECK_DIR)
        evidence(recorder.path, kind="tool_log", run_id=recorder.run_id)
        try:
            actual = await governed_calls(
                [call("unstarted"), call("remaining")], [TOOL], fixed_tool,
                recorder=recorder, tracer=tracer,
            )
        finally:
            recorder.close()
    assert actual["tool_calls"] == 0 and actual["output_items"] == []
    assert actual["tool_outcomes"] == {"succeeded": 0, "rejected": 0, "failed": 0, "skipped": 2}
    assert [(kind, identifier) for kind, identifier, _ in tracer.events] == [
        ("start", "unstarted"), ("end", "unstarted"), ("skipped", "remaining"),
    ]
    assert tracer.events[1][2]["output"] is None
    assert tracer.events[1][2]["outcome"] == "skipped"
    assert tracer.events[1][2]["reason_code"] == "logging_error"

    for event_name, first, outcome, attempts in (
        ("tool.succeeded", call("first"), "succeeded", 1),
        ("tool.result_created", call("first"), "succeeded", 1),
        ("tool.rejected", call("first", name="not_available"), "rejected", 0),
        ("tool.failed", call("first", arguments='{"key":"fail"}'), "failed", 1),
    ):
        tracer = Tracer()
        with recording_failure(event_name):
            recorder = RunRecorder(CHECK_DIR)
            evidence(recorder.path, kind="tool_log", run_id=recorder.run_id)
            try:
                actual = await governed_calls(
                    [first, call("remaining")], [TOOL], executor, recorder=recorder, tracer=tracer,
                )
            finally:
                recorder.close()
        assert actual["tool_calls"] == attempts and len(actual["output_items"]) == 1
        assert actual["tool_outcomes"][outcome] == 1
        assert actual["tool_outcomes"]["skipped"] == 1
        assert [(kind, identifier) for kind, identifier, _ in tracer.events] == [
            ("start", "first"), ("end", "first"), ("skipped", "remaining"),
        ]
        assert tracer.events[1][2]["outcome"] == outcome
        assert tracer.events[1][2]["output"] == json.loads(actual["output_items"][0]["output"])

async def check_recording_failures():
    async def attempt(event_name=None, *, create_failure=False, fail_close=False, fail_flush=False):
        fault = "create" if create_failure else "close" if fail_close else "flush" if fail_flush else "write"
        with recording_failure(
            event_name, create_failure=create_failure, fail_close=fail_close, fail_flush=fail_flush,
        ) as observations:
            state, client = await run(
                [response(call("io_a"), call("io_b")), response(message())],
                tools=[TOOL], execute_tool=fixed_tool,
                scenario=f"recording_{fault}_{event_name or 'file'}", failed_writes=observations,
            )
        assert state["stop_reason"] == "error" and state["log_complete"] is False
        assert "synthetic secret" not in (state["error"] or "")
        return state, client

    for create_failure, event_name, model_calls, tool_calls in (
        (True, None, 0, 0),
        (False, "run.started", 0, 0),
        (False, "model.started", 0, 0),
        (False, "model.returned", 1, 0),
        (False, "tool.started", 1, 0),
        (False, "tool.succeeded", 1, 1),
        (False, "tool.result_created", 1, 1),
        (False, "run.ended", 2, 2),
    ):
        state, _ = await attempt(event_name, create_failure=create_failure)
        counts(state, model_calls, 2 if model_calls else 0, tool_calls, "error")
        assert state["tool_outcomes"]["skipped"] == (2 - tool_calls if model_calls == 1 else 0)
        if model_calls:
            assert state["usage_totals"]["total_tokens"] == 12 * model_calls
        if tool_calls:
            assert state["tool_outcomes"]["succeeded"] == tool_calls
    state, _ = await attempt(fail_close=True)
    counts(state, 2, 2, 2, "error")
    state, _ = await attempt("model.started", fail_flush=True)
    counts(state, 0, 0, 0, "error")

async def check_failure_event_recording_failures():
    cases = (
        ("unknown", {"name": "unknown"}, "tool.rejected", "name_validation", "unknown_tool", 0),
        ("invalid_json", {"arguments": "not-json"}, "tool.rejected", "argument_parsing", "invalid_arguments", 0),
        ("non_object", {"arguments": "[]"}, "tool.rejected", "argument_parsing", "invalid_arguments", 0),
        ("duplicate_key", {"arguments": '{"key":"bad","key":"good"}'}, "tool.rejected", "argument_parsing", "invalid_arguments", 0),
        ("field_rejection", {"arguments": '{"key":42}'}, "tool.rejected", "argument_validation", "invalid_arguments", 1),
        ("executor_exception", {}, "tool.failed", "execution", "execution_error", 1),
        ("executor_return_type", {}, "tool.failed", "execution", "execution_error", 1),
    )
    for label, overrides, event_name, stage, code, attempts in cases:
        trace = []

        async def executor(name, arguments):
            trace.append(("tool", deepcopy(arguments)))
            if label == "executor_exception":
                raise RuntimeError("synthetic secret executor failure")
            if label == "executor_return_type":
                return {"invalid": "synthetic secret result"}
            return await fixed_tool(name, arguments)

        first = call(f"fault_{label}_a", arguments='{"key":"bad"}')
        first.update(overrides)
        requests = [first, call(f"fault_{label}_b", arguments='{"key":"good"}')]
        with recording_failure(event_name) as observations:
            state, client = await run(
                [response(*requests), response(message("must not be requested"))],
                tools=[TOOL], execute_tool=executor, scenario=f"failure_record_{label}",
                trace=trace, failed_writes=observations,
            )
        counts(state, 1, 2, attempts, "error")
        outcome = "failed" if event_name == "tool.failed" else "rejected"
        assert state["tool_outcomes"] == {
            "succeeded": 0, "rejected": int(outcome == "rejected"),
            "failed": int(outcome == "failed"), "skipped": 1,
        }
        assert trace == [("model", 1)] + ([("tool", json.loads(first["arguments"]))] if attempts else [])
        assert state["usage_totals"] == {
            "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
            "missing_usage_calls": 0,
        }
        assert state["result"]["usage"] == {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        results = outputs(state)
        assert len(results) == 1 and results[0]["call_id"] == first["call_id"]
        envelope = json.loads(results[0]["output"])
        assert envelope["ok"] is False and envelope["error"]["code"] == code
        assert "synthetic secret" not in results[0]["output"]
        assert state["input"] == client.requests[0]["input"] + requests + results
        assert not any(item.get("type") == "function_call_output" for item in client.requests[0]["input"])
        attempted, events = assert_failed_event(state, client, observations, event_name, stage=stage)
        assert attempted["call_id"] == first["call_id"] and attempted["request_index"] == 1
        assert attempted["tool_name"] == first["name"] and attempted["error_code"] == code
        assert attempted["node"] == "tools" and attempted["step_id"].startswith("step-")
        assert attempted["step_id"] not in {event["step_id"] for event in events if event["event"] == "model.started"}
        assert all(event["step_id"] == attempted["step_id"] for event in events
                   if event["event"] == "tool.started" and event["call_id"] == first["call_id"])
        assert not any(event["event"] == "tool.result_created" for event in events)
        assert not any(event["event"] == "tool.succeeded" for event in events)
        assert [event["call_id"] for event in events if event["event"] == "tool.started"] == (
            [first["call_id"]] if attempts else []
        )

    for label, second_reply, last_usage, totals in (
        (
            "no_response", ModelError("synthetic secret model failure"), None,
            {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12, "missing_usage_calls": 1},
        ),
        (
            "incomplete_response", response(message("partial"), usage=(20, 3, 23), status="incomplete"),
            {"input_tokens": 20, "output_tokens": 3, "total_tokens": 23},
            {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35, "missing_usage_calls": 0},
        ),
    ):
        trace = []
        request = call(f"before_model_failure_{label}")

        async def executor(name, arguments):
            trace.append(("tool", deepcopy(arguments)))
            return await fixed_tool(name, arguments)

        with recording_failure("model.failed") as observations:
            state, client = await run(
                [response(request), second_reply, response(message("must not be requested"))],
                tools=[TOOL], execute_tool=executor, scenario=f"failure_record_model_{label}",
                trace=trace, failed_writes=observations,
            )
        counts(state, 2, 1, 1, "error")
        assert trace == [("model", 1), ("tool", {"key": "sample"}), ("model", 2)]
        assert state["tool_outcomes"] == {"succeeded": 1, "rejected": 0, "failed": 0, "skipped": 0}
        assert state["usage_totals"] == totals
        if last_usage is None:
            assert state["result"] is None
        else:
            assert state["result"]["usage"] == last_usage
            assert state["result"]["raw_response"]["status"] == "incomplete"
        results = outputs(state)
        assert len(results) == 1 and results[0]["call_id"] == request["call_id"]
        assert inline_raw(json.loads(results[0]["output"])) == "fixed:sample"
        assert client.requests[1]["input"] == client.requests[0]["input"] + [request] + results
        attempted, events = assert_failed_event(
            state, client, observations, "model.failed",
            stage="model_request" if last_usage is None else "response_validation",
        )
        assert attempted["node"] == "model"
        sent = [event for event in events if event["event"] == "model.started"][-1]
        assert attempted["model_call"] == sent["model_call"] == 2
        assert attempted["operation_id"] == sent["operation_id"]
        assert "call_id" not in attempted and "tool_name" not in attempted
        assert [event["call_id"] for event in events if event["event"] == "tool.result_created"] == [request["call_id"]]

def execute():
    with check_group("recording.runtime_events.safe_summaries"):
        asyncio.run(check_safe_summaries())
    with check_group("recording.runtime_events.safe_rejection_reasons"):
        asyncio.run(check_safe_rejection_reasons())
    with check_group("recording.runtime_events.safe_execution_reasons"):
        asyncio.run(check_safe_execution_reasons())
    with check_group("recording.runtime_events.tool_trace_hooks"):
        asyncio.run(check_tool_trace_hooks())
    with check_group("recording.runtime_events.recording_failures"):
        asyncio.run(check_recording_failures())
    with check_group("recording.runtime_events.failure_event_recording_failures"):
        asyncio.run(check_failure_event_recording_failures())
