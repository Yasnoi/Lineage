"""Deterministic acceptance for the four reviewed session/runtime boundaries.

The real graph, checkpoint saver and JSONL writer run throughout. Model HTTP is
replaced at the SDK boundary; tracing uses its real SDK with a fake transport.
"""

import asyncio
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import gc
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4
import weakref

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from openai import APIStatusError

from check_react import (
    LocalClient, SETTINGS as BASE_SETTINGS, TOOL, call, fixed_tool, message,
    outputs, read_events, recording_failure, response,
)
from check_sessions import audit_turn, checkpoint_fault, config, saved_state
from check_text import snapshot
from check_tracing import capture
from lineage_agent.agent import runtime, sessions
from lineage_agent.agent.graph import FINAL_ANSWER_INSTRUCTIONS
from lineage_agent.agent.checkpointing import CheckpointError, SessionCheckpointer
from lineage_agent.agent.sessions import SessionManager
from lineage_agent.config.settings import ConfigurationError
from lineage_agent.entrypoints import cli
from lineage_agent.recording.tracing import RunTrace
from lineage_agent.tools.filesystem.text import TEXT_TOOL, create_text_executor
from lineage_agent.tools.types import read_budget_snapshot


CHECK_DIR = Path(__file__).resolve().parents[1] / "logs" / "checks" / str(uuid4())
SETTINGS = replace(BASE_SETTINGS, log_dir=CHECK_DIR)
PRIVATE = "REVIEW_PRIVATE_EXCEPTION_NEVER_RECORD_8340"
RUNS = []


class CountingClient(LocalClient):
    def __init__(self, replies):
        super().__init__(replies)
        self.close_calls = 0

    async def __aexit__(self, *args):
        self.close_calls += 1
        await super().__aexit__(*args)


@contextmanager
def resources():
    """Observe resource ownership without replacing cleanup or state results."""
    recorders, tracers = [], []
    recorder_factory, trace_factory = runtime.RunRecorder, runtime.start_trace

    def recorder(*args, **kwargs):
        value = recorder_factory(*args, **kwargs)
        recorders.append(value)
        return value

    def tracer(*args, **kwargs):
        value = trace_factory(*args, **kwargs)
        tracers.append(value)
        return value

    with patch.object(runtime, "RunRecorder", side_effect=recorder), patch.object(
        runtime, "start_trace", side_effect=tracer,
    ):
        yield recorders, tracers


def closed_resources(manager, sid, observed):
    recorders, tracers = observed
    assert sid not in manager._checkpointer._recorders
    for recorder in recorders:
        assert recorder._closed
        stream = getattr(recorder._stream, "stream", recorder._stream)
        assert stream is None or stream.closed
    for tracer in tracers:
        assert tracer._closed
        assert tracer._worker is None or not tracer._worker.is_alive()


async def run_turn(manager, sid, replies, *, scenario, prompt="question", observations=None, client=None):
    client = client or CountingClient(replies)
    with resources() as observed, patch.object(runtime, "create_client", return_value=client) as factory:
        result = await manager.run_turn(sid, prompt)
    closed_resources(manager, sid, observed)
    assert client.close_calls == (1 if factory.called else 0)
    audit_turn(result, client, scenario, observations)
    starts = [event for event in read_events(result.state) if event["event"] == "model.started"]
    for request, event in zip(client.requests, starts, strict=True):
        if event["final_answer_call"]:
            assert request["instructions"] == FINAL_ANSWER_INSTRUCTIONS and request["tool_choice"] == "none"
        else:
            assert "instructions" not in request and "tool_choice" not in request
        assert not {"remaining_model_calls", "read_budget", "read_budget_error"} & request.keys()
    assert PRIVATE not in json.dumps(read_events(result.state), ensure_ascii=False)
    assert PRIVATE not in (result.state["error"] or "")
    if result.checkpoint_complete:
        assert await saved_state(manager, result) == result.state
    RUNS.append(result.state["run_id"])
    return result, client


async def blocked(manager, sid):
    with patch.object(runtime, "create_client", side_effect=AssertionError("blocked session started")):
        try:
            await manager.run_turn(sid, "must not execute")
        except sessions.SessionError as exc:
            assert exc.code == "session_blocked"
        else:
            raise AssertionError("session did not remain blocked")


async def check_cache_does_not_keep_frames():
    manager = SessionManager()
    sid = manager.create_session(SETTINGS)
    did_close = []

    class WatchedClient(CountingClient):
        async def __aexit__(self, *args):
            await super().__aexit__(*args)
            did_close.append(True)

    async def fail_one_run():
        client = WatchedClient([response(message("already happened"))])
        reference = weakref.ref(client)
        with checkpoint_fault(
            manager, "put", lambda cp, meta, *_: meta["source"] == "loop" and cp["channel_values"].get("model_calls") == 1,
        ) as observations:
            result, _ = await run_turn(manager, sid, [], scenario="cache_failed_node", client=client, observations=observations)
        assert result.state["model_calls"] == 1 and result.state["usage_totals"]["total_tokens"] == 12
        assert result.checkpoint_error == "checkpoint_write_error"
        return result, reference

    result, reference = await fail_one_run()
    await asyncio.sleep(0)
    gc.collect()
    assert did_close == [True]
    assert reference() is None, "failure cache retained the closed client through runtime frames"
    assert manager._checkpointer._failures == {sid: "checkpoint_write_error"}
    errors = [manager._checkpointer.failure(sid), manager._checkpointer.failure(sid)]
    assert errors[0] is not errors[1]
    assert all(error.code == "checkpoint_write_error" and error.__traceback__ is None for error in errors)
    for _ in range(2):
        try:
            manager._checkpointer._healthy(sid)
        except CheckpointError as error:
            errors.append(error)
    assert len({id(error) for error in errors}) == 4
    assert manager._checkpointer._failed(sid, "checkpoint_read_error", "probe").code == "checkpoint_write_error"
    assert PRIVATE not in str(errors)
    await blocked(manager, sid)
    other = manager.create_session(SETTINGS)
    await run_turn(manager, other, [response(message("unaffected"))], scenario="cache_other_session_healthy")

    # A local sentinel directly proves that no failed storage frame survives.
    saver = SessionCheckpointer()
    local_sid = str(uuid4())

    class Sentinel:
        pass

    async def storage_failure():
        sentinel = Sentinel()
        reference = weakref.ref(sentinel)
        with patch.object(InMemorySaver, "get_tuple", side_effect=OSError(PRIVATE)):
            try:
                await saver.aget_tuple(config(local_sid))
            except CheckpointError as error:
                assert error.code == "checkpoint_read_error"
        return reference

    sentinel_ref = await storage_failure()
    gc.collect()
    assert sentinel_ref() is None
    assert saver._failures == {local_sid: "checkpoint_read_error"}
    print(json.dumps({"scenario": "cache_no_traceback_retention", "session_id": sid, "run_id": result.state["run_id"], "log_path": result.state["log_path"], "closed_client_collected": True, "fresh_exception_count": len(errors)}))


class InvalidMapping(dict):
    def __getitem__(self, key):
        raise RuntimeError(PRIVATE)


class MetadataExecutor:
    def __init__(self, *, fail_at=(), invalid=False, inner=None):
        self.fail_at = set(fail_at)
        self.invalid = invalid
        self.inner = inner
        self.reads = 0

    @property
    def read_budget(self):
        self.reads += 1
        if self.reads in self.fail_at or -1 in self.fail_at:
            if self.invalid:
                return InvalidMapping()
            raise RuntimeError(PRIVATE)
        return {"limit_bytes": 9, "used_bytes": 0, "remaining_bytes": 9}

    async def __call__(self, name, arguments):
        return await (self.inner or fixed_tool)(name, arguments)


async def check_budget_helper_boundary():
    class Missing:
        pass

    class DeclaredMissing:
        @property
        def read_budget(self):
            raise AttributeError(PRIVATE)

    class Interrupted:
        @property
        def read_budget(self):
            raise asyncio.CancelledError()

    diagnostics = []
    for executor in (None, Missing(), fixed_tool):
        assert read_budget_snapshot(executor, on_unavailable=lambda: diagnostics.append(True)) is None
    assert not diagnostics
    for executor in (DeclaredMissing(), MetadataExecutor(fail_at={-1}), MetadataExecutor(fail_at={-1}, invalid=True)):
        assert read_budget_snapshot(executor, on_unavailable=lambda: diagnostics.append(True)) is None
    assert len(diagnostics) == 3
    try:
        read_budget_snapshot(Interrupted(), on_unavailable=lambda: diagnostics.append(True))
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("cancellation was swallowed by metadata observation")
    assert len(diagnostics) == 3
    print(json.dumps({"scenario": "metadata_helper_boundary", "ordinary_faults_observed": 3, "cancellation_propagates": True}))


async def check_metadata_stages_and_tracing():
    for mode, enabled, index, invalid in (
        ("ok", True, 1, False), ("ok", True, 2, False),
        ("ok", True, 3, False), ("ok", True, 3, True),
        ("ok", False, 3, False), ("connection", True, 3, False),
    ):
        executor = MetadataExecutor(fail_at={index}, invalid=invalid)
        manager = SessionManager()
        sid = manager.create_session(SETTINGS, tools=[TOOL], execute_tool=executor)
        with capture(mode, LANGSMITH_TRACING="true" if enabled else "false") as (transport, stderr):
            result, client = await run_turn(manager, sid, [response(message("retained answer"))], scenario=f"metadata_stage_{index}_{invalid}_{enabled}_{mode}")
        assert executor.reads == 3, "final JSONL and tracing must share one observation"
        assert result.checkpoint_complete and result.state["stop_reason"] == "completed"
        assert result.state["result"]["text"] == "retained answer"
        assert result.state["usage_totals"]["total_tokens"] == 12
        relevant = [event for event in read_events(result.state) if event["event"] in {"run.started", "model.started", "run.ended"}]
        assert len(relevant) == 3
        for position, event in enumerate(relevant, 1):
            assert event.get("read_budget_error") == ("read_budget_unavailable" if position == index else None)
            assert (event["read_budget"] is None) == (position == index)
        assert "instructions" not in client.requests[0], "metadata availability must not alter model instructions"
        assert PRIVATE not in json.dumps(transport.sent, ensure_ascii=False)
        assert PRIVATE not in stderr.getvalue()
        if enabled:
            root = next(span for span in transport.sent if span["run_type"] == "chain")
            metadata = root["extra"]["metadata"]
            assert metadata["read_budget"] == relevant[-1]["read_budget"]
            assert metadata.get("read_budget_error") == relevant[-1].get("read_budget_error")
            assert "read_budget_unavailable" in json.dumps(transport.sent)
            model = next(span for span in transport.sent if span["run_type"] == "llm")
            assert model["extra"]["metadata"].get("read_budget_error") == relevant[1].get("read_budget_error")
            assert model["extra"]["metadata"]["read_budget"] == relevant[1]["read_budget"]
            assert "instructions" not in model["inputs"]
            assert metadata["checkpoint_saved"] is True
        else:
            assert not transport.sent
        assert bool(stderr.getvalue()) == (enabled and mode == "connection")


async def check_metadata_keeps_real_budget():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "body.txt").write_bytes(b"abcd")
        before = snapshot(root)
        real_executor = create_text_executor(root, max_bytes=4, max_total_bytes=4)
        executor = MetadataExecutor(fail_at={-1}, inner=real_executor)
        manager = SessionManager()
        sid = manager.create_session(SETTINGS, tools=[TEXT_TOOL], execute_tool=executor)
        with capture() as (transport, stderr):
            result, client = await run_turn(manager, sid, [
                response(call("read_ok", name="read_text_file", arguments='{"path":"body.txt"}'), call("read_refused", name="read_text_file", arguments='{"path":"body.txt"}')),
                response(message("The second read was refused.")),
            ], scenario="metadata_failure_real_read_budget")
        assert not stderr.getvalue()
        assert result.state["tool_calls"] == 2
        assert result.state["tool_outcomes"] == {"succeeded": 1, "rejected": 1, "failed": 0, "skipped": 0}
        results = outputs(result.state)
        assert json.loads(json.loads(results[0]["output"])["result"])["content"] == "abcd"
        assert json.loads(results[1]["output"])["error"]["reason_code"] == "read_budget_exceeded"
        assert client.requests[1]["input"][-2:] == results
        assert real_executor.read_budget == {"limit_bytes": 4, "used_bytes": 4, "remaining_bytes": 0}
        assert snapshot(root) == before
        assert all("instructions" not in request and request["tools"] == [TEXT_TOOL] for request in client.requests)
        starts = [event for event in read_events(result.state) if event["event"] == "model.started"]
        assert all(event["read_budget"] is None and event["read_budget_error"] == "read_budget_unavailable" for event in starts)
        assert PRIVATE not in json.dumps(transport.sent)


async def check_metadata_failure_combinations():
    for fault in ("log", "checkpoint", "both"):
        manager = SessionManager()
        executor = MetadataExecutor(fail_at={3})
        sid = manager.create_session(SETTINGS, tools=[TOOL], execute_tool=executor)
        observations = []
        with capture() as (transport, _):
            if fault == "log":
                with recording_failure("run.ended") as observations:
                    result, _ = await run_turn(manager, sid, [response(message("still available"))], scenario="metadata_and_log_failure", observations=observations)
            else:
                with checkpoint_fault(manager, "put", lambda cp, meta, *_: meta["source"] == "update") as observations:
                    if fault == "both":
                        with recording_failure("run.ended"):
                            result, _ = await run_turn(manager, sid, [response(message("still available"))], scenario="metadata_log_checkpoint_failure", observations=observations)
                    else:
                        result, _ = await run_turn(manager, sid, [response(message("still available"))], scenario="metadata_and_final_save_failure", observations=observations)
        assert observations and executor.reads == 3
        assert result.state["result"]["text"] == "still available"
        assert result.state["model_calls"] == 1 and result.state["usage_totals"]["total_tokens"] == 12
        assert result.checkpoint_complete is (fault == "log")
        assert result.state["log_complete"] is (fault == "checkpoint")
        assert result.state["error_code"] == ("logging_error" if fault != "checkpoint" else None)
        assert PRIVATE not in json.dumps(transport.sent, ensure_ascii=False)
        if fault != "log":
            await blocked(manager, sid)


async def check_invalid_input_history():
    bad_inputs = {
        "numeric_content": [{"role": "user", "content": 123}],
        "unknown_role": [{"role": "PRIVATE_ROLE", "content": "bad"}],
        "non_dictionary_part": [{"role": "user", "content": ["bad"]}],
        "part_missing_type": [{"role": "user", "content": [{"text": "bad"}]}],
        "empty_part_type": [{"role": "user", "content": [{"type": "", "text": "bad"}]}],
        "numeric_input_text": [{"role": "user", "content": [{"type": "input_text", "text": 123}]}],
        "numeric_refusal": [{"role": "assistant", "content": [{"type": "refusal", "refusal": 123}]}],
        "call_empty_name": [call("bad_name", name=""), {"type": "function_call_output", "call_id": "bad_name", "output": "refused"}],
        "call_object_arguments": [call("bad_args", arguments={"key": "sample"}), {"type": "function_call_output", "call_id": "bad_args", "output": "refused"}],
        "numeric_tool_output": [call("bad_output"), {"type": "function_call_output", "call_id": "bad_output", "output": 123}],
    }
    for scenario, bad in bad_inputs.items():
        manager = SessionManager()
        sid = manager.create_session(SETTINGS)
        first, _ = await run_turn(manager, sid, [response(message("saved fact"))], prompt="Original fact", scenario="input_baseline_" + scenario)
        original = deepcopy(bad)
        rejected, client = await run_turn(manager, sid, [], prompt=bad, scenario="invalid_input_" + scenario)
        assert not client.requests and bad == original
        assert rejected.state["error_code"] == "invalid_input"
        assert rejected.state["history_end"] == len(first.state["input"])
        assert rejected.state["input"] == first.state["input"] + bad
        event = next(event for event in read_events(rejected.state) if event["event"] == "run.input_rejected")
        assert event["error_code"] == "invalid_input" and event["stage"] == "input_validation"
        assert type(event["input_index"]) is int
        following, captured = await run_turn(manager, sid, [response(message("saved fact"))], prompt="Recall", scenario="after_invalid_input_" + scenario)
        assert captured.requests[0]["input"] == first.state["input"] + [{"role": "user", "content": "Recall"}]
        assert following.state["stop_reason"] == "completed"


async def check_valid_protocol_and_http_history():
    manager = SessionManager()
    sid = manager.create_session(SETTINGS)
    source = [
        {"role": "system", "content": "trusted application instruction", "extension": 17},
        {"role": "developer", "content": [{"type": "input_text", "text": "application facts"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "question"}, {"type": "provider_input_extension", "payload": {"nested": True}}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "prior answer"}, {"type": "refusal", "refusal": "prior refusal"}]},
        {"type": "reasoning", "summary": [], "encrypted_content": "local opaque reasoning"},
        {"type": "provider_extension", "payload": {"nested": [1]}},
        call("historical_rejected_arguments", arguments="not valid JSON"),
        {"type": "function_call_output", "call_id": "historical_rejected_arguments", "output": '{"ok":false,"error":{"code":"invalid_arguments"}}'},
        call("historical_list_output"),
        {"type": "function_call_output", "call_id": "historical_list_output", "output": [{"type": "input_text", "text": "part result"}]},
    ]
    original = deepcopy(source)
    result, client = await run_turn(manager, sid, [response(message("accepted"))], prompt=source, scenario="supported_messages_tools_and_extensions")
    assert client.requests[0]["input"] == original and source == original
    assert result.state["history_end"] == len(result.state["input"])
    http_error = APIStatusError(
        PRIVATE, response=httpx.Response(400, request=httpx.Request("POST", "https://unused.invalid/v1/responses")),
        body={"error": {"message": PRIVATE}},
    )
    extension = [{"type": "provider_private_extension", "payload": "still not a locally known error"}]
    failed, client = await run_turn(manager, sid, [http_error], prompt=extension, scenario="http_400_not_input_rejection")
    assert failed.state["error_code"] == "http_error" and failed.state["model_calls"] == 1
    assert failed.state["history_end"] == len(result.state["input"]) + 1
    assert not any(event["event"] == "run.input_rejected" for event in read_events(failed.state))
    assert failed.state["usage_totals"]["missing_usage_calls"] == 1
    later, captured = await run_turn(manager, sid, [response(message("accepted again"))], prompt="Retry service", scenario="after_http_400_keeps_accepted_input")
    assert captured.requests[0]["input"] == failed.state["input"] + [{"role": "user", "content": "Retry service"}]
    assert later.checkpoint_complete


def report_aborted(sid, checkpoint_id, observed, scenario):
    if not observed[0]:
        print(json.dumps({"scenario": scenario, "session_id": sid, "run_id": None, "last_confirmed_checkpoint_id": checkpoint_id, "log_path": None, "execution_started": False, "checkpoint_unchanged": True}, ensure_ascii=False))
    for recorder in observed[0]:
        events = [json.loads(line) for line in recorder.path.read_text(encoding="utf-8").splitlines()]
        assert all(event.get("session_id") == sid for event in events)
        assert not any(event["event"] == "model.started" for event in events)
        assert PRIVATE not in json.dumps(events, ensure_ascii=False)
        print(json.dumps({"scenario": scenario, "session_id": sid, "run_id": recorder.run_id, "last_confirmed_checkpoint_id": checkpoint_id, "log_path": str(recorder.path), "model_calls": 0, "checkpoint_unchanged": True}, ensure_ascii=False))


async def check_preexecution_config_retry():
    for phase in ("missing_key", "binding", "tool_configuration"):
        manager = SessionManager()
        sid = manager.create_session(SETTINGS)
        first, _ = await run_turn(manager, sid, [response(message("retained history"))], scenario="before_preconfig_" + phase)
        committed = deepcopy(manager._checkpointer.storage[sid])
        with resources() as observed, capture() as (transport, stderr):
            try:
                if phase == "missing_key":
                    with patch.dict(os.environ, {"API_KEY": ""}):
                        await manager.run_turn(sid, "must not become history")
                elif phase == "binding":
                    class BadBinding:
                        async def __call__(self, name, arguments):
                            raise AssertionError("never executed")

                        def bind_run(self):
                            raise ConfigurationError("工具运行配置无效。")
                    manager._sessions[sid].execute_tool = BadBinding()
                    await manager.run_turn(sid, "must not become history")
                else:
                    with patch.object(runtime, "validate_tools", side_effect=ConfigurationError("工具配置无效。")):
                        await manager.run_turn(sid, "must not become history")
            except runtime.PreExecutionConfigurationError as exc:
                assert isinstance(exc, ConfigurationError)
            else:
                raise AssertionError("pre-execution configuration rejection was hidden")
        closed_resources(manager, sid, observed)
        assert not manager._sessions[sid].blocked
        assert manager._sessions[sid].last_checkpoint_id == first.checkpoint_id
        assert manager._checkpointer.storage[sid] == committed
        assert PRIVATE not in json.dumps(transport.sent, ensure_ascii=False) + stderr.getvalue()
        report_aborted(sid, first.checkpoint_id, observed, "preconfig_" + phase)
        manager._sessions[sid].execute_tool = None
        later, client = await run_turn(manager, sid, [response(message("retained history"))], prompt="Recall", scenario="after_preconfig_" + phase)
        assert client.requests[0]["input"] == first.state["input"] + [{"role": "user", "content": "Recall"}]
        assert later.checkpoint_complete


async def check_execution_started_still_blocks():
    class EnterFailure(CountingClient):
        async def __aenter__(self):
            raise ConfigurationError("safe test configuration error after construction")

    manager = SessionManager()
    sid = manager.create_session(SETTINGS)
    first, _ = await run_turn(manager, sid, [response(message("original"))], scenario="before_client_enter_failure")
    client = EnterFailure([])
    with resources() as observed, capture(), patch.object(runtime, "create_client", return_value=client):
        try:
            await manager.run_turn(sid, "client entry failure")
        except ConfigurationError as exc:
            assert not isinstance(exc, runtime.PreExecutionConfigurationError)
        else:
            raise AssertionError("client entry failure was accepted")
    assert client.closed and client.close_calls == 1 and not client.requests
    closed_resources(manager, sid, observed)
    report_aborted(sid, first.checkpoint_id, observed, "client_enter_configuration_error_blocks")
    await blocked(manager, sid)


async def check_finally_and_primary_error():
    class InterruptedEndBudget(MetadataExecutor):
        @property
        def read_budget(self):
            self.reads += 1
            if self.reads == 3:
                raise asyncio.CancelledError()
            return {"limit_bytes": 9, "used_bytes": 0, "remaining_bytes": 9}

    manager = SessionManager()
    sid = manager.create_session(SETTINGS, tools=[TOOL], execute_tool=InterruptedEndBudget())
    client = CountingClient([response(message("answer before interrupted finalization"))])
    with resources() as observed, capture(), patch.object(runtime, "create_client", return_value=client):
        try:
            await manager.run_turn(sid, "finish is interrupted")
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("finalization cancellation did not propagate")
    closed_resources(manager, sid, observed)
    assert client.closed and client.close_calls == 1 and len(client.requests) == 1
    assert manager._sessions[sid].last_checkpoint_id is None
    last_confirmed = manager._checkpointer.last_confirmed(sid)
    saved = await manager._checkpointer.aget_tuple(config(sid, last_confirmed))
    assert saved.checkpoint["channel_values"]["model_calls"] == 1
    assert saved.checkpoint["channel_values"]["result"]["text"] == "answer before interrupted finalization"
    await blocked(manager, sid)
    print(json.dumps({"scenario": "finalization_cancellation_closes_resources", "session_id": sid, "run_id": observed[0][0].run_id, "last_confirmed_checkpoint_id": last_confirmed, "final_checkpoint_complete": False, "step_id": "step-0001", "log_path": str(observed[0][0].path), "model_calls": 1}, ensure_ascii=False))

    primary = ConfigurationError("safe primary client-entry rejection")

    class EnterFailure(CountingClient):
        async def __aenter__(self):
            raise primary

    manager = SessionManager()
    sid = manager.create_session(SETTINGS)
    client = EnterFailure([])
    real_close = RunTrace.close
    close_calls = []

    async def fail_close_once(tracer):
        close_calls.append(tracer.run_id)
        await real_close(tracer)
        if len(close_calls) == 1:
            raise RuntimeError(PRIVATE)

    with resources() as observed, capture(), patch.object(runtime, "create_client", return_value=client), patch.object(RunTrace, "close", fail_close_once):
        try:
            await manager.run_turn(sid, "primary failure plus cleanup fault")
        except ConfigurationError as error:
            assert error is primary
        else:
            raise AssertionError("cleanup replaced the primary error")
    closed_resources(manager, sid, observed)
    assert client.close_calls == 1 and close_calls
    report_aborted(sid, None, observed, "primary_error_survives_secondary_close_failure")
    await blocked(manager, sid)


async def check_prepare_and_final_save_secondary_errors():
    real_close = RunTrace.close
    close_calls = []

    async def fail_after_close(tracer):
        close_calls.append(tracer.run_id)
        await real_close(tracer)
        raise RuntimeError(PRIVATE)

    manager = SessionManager()
    sid = manager.create_session(SETTINGS)
    first, _ = await run_turn(manager, sid, [response(message("committed history"))], scenario="before_preconfig_and_trace_close_failure")
    committed = deepcopy(manager._checkpointer.storage[sid])
    with resources() as observed, capture() as (transport, stderr), patch.object(
        runtime, "create_client", side_effect=ConfigurationError(PRIVATE),
    ) as factory, patch.object(RunTrace, "close", fail_after_close):
        try:
            await manager.run_turn(sid, "configuration rejected before execution")
        except runtime.PreExecutionConfigurationError as error:
            assert isinstance(error, ConfigurationError) and PRIVATE not in str(error)
        else:
            raise AssertionError("secondary tracing close error replaced the preparation rejection")
    closed_resources(manager, sid, observed)
    assert factory.call_count == 1 and len(close_calls) == 1
    assert manager._checkpointer.storage[sid] == committed
    assert manager._sessions[sid].last_checkpoint_id == first.checkpoint_id
    assert not manager._sessions[sid].blocked
    assert PRIVATE not in json.dumps(transport.sent, ensure_ascii=False) + stderr.getvalue()
    report_aborted(sid, first.checkpoint_id, observed, "preconfig_survives_secondary_trace_close_failure")
    later, client = await run_turn(manager, sid, [response(message("committed history"))], prompt="Recall original history", scenario="after_preconfig_and_trace_close_failure")
    assert client.requests[0]["input"] == first.state["input"] + [{"role": "user", "content": "Recall original history"}]
    assert later.checkpoint_complete

    manager = SessionManager()
    sid = manager.create_session(SETTINGS)
    client = CountingClient([response(message("answer retained before cancelled save"))])
    cancellation = asyncio.CancelledError()
    real_put, real_finish = InMemorySaver.put, RunTrace.finish
    writes, finishes = [], []
    close_calls.clear()

    def cancel_final_put(saver, cfg, checkpoint, metadata, versions):
        if saver is manager._checkpointer and metadata["source"] == "update":
            writes.append({"source": "fault_injection", "operation": "put", "session_id": sid, "checkpoint_id": cfg["configurable"].get("checkpoint_id"), "persistence": "operation_cancelled"})
            raise cancellation
        return real_put(saver, cfg, checkpoint, metadata, versions)

    def fail_after_finish(tracer, *args, **kwargs):
        finishes.append(tracer.run_id)
        real_finish(tracer, *args, **kwargs)
        raise RuntimeError(PRIVATE)

    with resources() as observed, capture() as (transport, stderr), patch.object(
        runtime, "create_client", return_value=client,
    ), patch.object(InMemorySaver, "put", cancel_final_put), patch.object(
        RunTrace, "finish", fail_after_finish,
    ), patch.object(RunTrace, "close", fail_after_close):
        try:
            await manager.run_turn(sid, "cancel final save after the real answer")
        except asyncio.CancelledError as error:
            assert error is cancellation
        else:
            raise AssertionError("secondary tracing errors replaced final-save cancellation")
    closed_resources(manager, sid, observed)
    assert client.close_calls == 1 and len(client.requests) == 1
    assert len(writes) == len(close_calls) == 1 and finishes
    assert manager._sessions[sid].last_checkpoint_id is None
    last_confirmed = manager._checkpointer.last_confirmed(sid)
    saved = await manager._checkpointer.aget_tuple(config(sid, last_confirmed))
    facts = saved.checkpoint["channel_values"]
    assert facts["model_calls"] == 1 and facts["usage_totals"]["total_tokens"] == 12
    assert facts["result"]["text"] == "answer retained before cancelled save"
    assert PRIVATE not in json.dumps(transport.sent, ensure_ascii=False) + stderr.getvalue()
    events = [json.loads(line) for line in observed[0][0].path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event"] == "run.ended" and events[-1]["checkpoint_finalization"] == "pending"
    assert all(event.get("session_id") == sid for event in events)
    assert PRIVATE not in json.dumps(events, ensure_ascii=False)
    root_span = next(span for span in transport.sent if span["run_type"] == "chain")
    assert root_span["extra"]["metadata"]["checkpoint_saved"] is False
    await blocked(manager, sid)
    print(json.dumps({"scenario": "cancelled_final_save_survives_finish_and_close_errors", "session_id": sid, "run_id": observed[0][0].run_id, "last_confirmed_checkpoint_id": last_confirmed, "final_checkpoint_complete": False, "step_id": "step-0001", "log_path": str(observed[0][0].path), "model_calls": 1, "fault_injection_observations": writes}, ensure_ascii=False))


def check_cli_preconfiguration_retry():
    manager = SessionManager()
    first = CountingClient([response(message("CLI retained fact"))])
    last = CountingClient([response(message("CLI retained fact"))])
    actual_create = runtime.create_client
    calls = 0

    def factory(settings):
        nonlocal calls
        calls += 1
        if calls == 2:
            with patch.dict(os.environ, {"API_KEY": ""}):
                return actual_create(settings)
        return first if calls == 1 else last

    turns = []
    original_display = cli.display_turn

    def display(result, settings):
        turns.append(result)
        return original_display(result, settings)

    out, err = StringIO(), StringIO()
    with patch("sys.argv", ["lineage-agent"]), patch("builtins.input", side_effect=["remember fact", "rejected setup", "recall fact", "/exit"]), patch.object(
        cli, "load_settings", return_value=SETTINGS,
    ), patch.object(cli, "SessionManager", return_value=manager), patch.object(
        runtime, "create_client", side_effect=factory,
    ), patch.object(cli, "display_turn", side_effect=display), redirect_stdout(out), redirect_stderr(err):
        assert cli.main() == 0
    assert calls == 3 and len(turns) == 2
    assert turns[0].session_id == turns[1].session_id
    assert last.requests[0]["input"] == turns[0].state["input"] + [{"role": "user", "content": "recall fact"}]
    assert first.close_calls == last.close_calls == 1
    assert "API_KEY" in err.getvalue() and PRIVATE not in err.getvalue()
    for index, (result, client) in enumerate(zip(turns, (first, last), strict=True)):
        audit_turn(result, client, f"cli_preconfiguration_retry_{index}")
        RUNS.append(result.state["run_id"])
    with patch("sys.argv", ["lineage-agent", "single"]), patch.object(cli, "load_settings", return_value=SETTINGS), patch.dict(
        os.environ, {"API_KEY": ""},
    ), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        assert cli.main() == 2
    with patch("sys.argv", ["lineage-agent"]), patch.object(cli, "load_settings", side_effect=ConfigurationError("配置无效。")), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        assert cli.main() == 2


async def main():
    checks = (
        check_cache_does_not_keep_frames, check_budget_helper_boundary,
        check_metadata_stages_and_tracing, check_metadata_keeps_real_budget,
        check_metadata_failure_combinations, check_invalid_input_history,
        check_valid_protocol_and_http_history, check_preexecution_config_retry,
        check_execution_started_still_blocks, check_finally_and_primary_error,
        check_prepare_and_final_save_secondary_errors,
    )
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
        for check in checks:
            await check()
            print(f"PASS {check.__name__}")
    return len(checks)


if __name__ == "__main__":
    groups = asyncio.run(main())
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
        check_cli_preconfiguration_retry()
    print("PASS check_cli_preconfiguration_retry")
    print(json.dumps({"groups": groups + 1, "session_runs": len(RUNS), "check_log_dir": str(CHECK_DIR)}, ensure_ascii=False))
