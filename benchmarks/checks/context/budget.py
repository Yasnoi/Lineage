"""context / budget behavior contracts."""


import asyncio
import json
from benchmarks.checks.support.observations import inline_raw, assert_tools
import os
from benchmarks.checks.support.budget import (
    CHECK_DIR,
    PRIVATE,
    RUNS,
    SETTINGS,
    audit,
    capture_measurements,
    compact_bytes,
    opaque_item,
    run,
)
from benchmarks.checks.support.events import group_id, check_group, observe_turn
from benchmarks.checks.support.runtime import (
    FaultyFile,
    LocalClient,
    TOOL,
    call,
    fixed_tool,
    message,
    outputs,
    read_events,
    recording_failure,
    response,
)
from benchmarks.checks.support.sessions import checkpoint_fault, saved_state
from benchmarks.checks.support.text import snapshot
from benchmarks.checks.support.tracing import capture
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from io import StringIO
from lineage_agent.tools.history import HISTORY_TOOLS
from lineage_agent.artifacts.tools import ARTIFACT_TOOLS
from lineage_agent.agent import graph, runtime
from lineage_agent.agent.sessions import SessionError, SessionManager
from lineage_agent.config.settings import ConfigurationError, ModelSettings, load_settings
from lineage_agent.entrypoints import cli
from lineage_agent.models.budget import annotate_context_risks, check_context_budget
from lineage_agent.models.requests import prepare_request
from lineage_agent.recording.recorder import RunRecorder
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

async def check_configuration():
    defaults = {
        "context_budget_tokens": 250000, "max_output_tokens": 16384,
        "context_safety_margin_tokens": 8192, "opaque_fallback_tokens": 4096,
    }
    configured_defaults = ModelSettings(base_url=SETTINGS.base_url, model_name=SETTINGS.model_name)
    assert all(getattr(configured_defaults, key) == value for key, value in defaults.items())
    assert (SETTINGS.context_budget_tokens, SETTINGS.max_output_tokens,
            SETTINGS.context_safety_margin_tokens) == (32768, 4096, 1024)
    for field in defaults:
        invalid = [True, False, -1, 1.5, "2"]
        if field != "context_safety_margin_tokens":
            invalid.append(0)
        for value in invalid:
            try:
                replace(SETTINGS, **{field: value})
            except ConfigurationError:
                pass
            else:
                raise AssertionError((field, value))
    replace(SETTINGS, context_safety_margin_tokens=0)
    try:
        replace(SETTINGS, context_budget_tokens=5120)
    except ConfigurationError:
        pass
    else:
        raise AssertionError("threshold must leave room beyond fixed reserves")
    with TemporaryDirectory() as temporary:
        env = {
            "BASE_URL": SETTINGS.base_url, "MODEL_NAME": SETTINGS.model_name,
            "CONTEXT_BUDGET_TOKENS": "123456", "MAX_OUTPUT_TOKENS": "5000",
            "CONTEXT_SAFETY_MARGIN_TOKENS": "0", "OPAQUE_FALLBACK_TOKENS": "9000",
            "SUMMARY_TARGET_PERCENT": "35",
        }
        with patch.dict(os.environ, env, clear=True):
            loaded = load_settings(Path(temporary) / "missing.env")
        assert loaded.context_budget_tokens == 123456 and loaded.max_output_tokens == 5000
        assert loaded.context_safety_margin_tokens == 0 and loaded.opaque_fallback_tokens == 9000
        assert loaded.summary_target_percent == 35
    print(json.dumps({"scenario": "configuration_validation", "defaults": defaults}))

async def check_transparent_request_and_boundary():
    prompt = [{"role": "user", "content": "中文 \"quoted\" \\ slash\nline"}]
    original = deepcopy(prompt)
    baseline, client, measured = await run([response(message())], prompt=prompt, scenario="transparent_reference")
    request = client.requests[0]
    diagnostic = measured[0]["diagnostic"]
    assert diagnostic["transparent_tokens"] == compact_bytes(request)
    assert diagnostic["opaque_items"] == diagnostic["opaque_risk_tokens"] == 0
    required = diagnostic["required_tokens"]
    assert required == compact_bytes(request) + SETTINGS.max_output_tokens + SETTINGS.context_safety_margin_tokens
    assert prompt == original
    for delta, expected in ((1, "completed"), (0, "completed"), (-1, "context_budget_exceeded")):
        # This fixture isolates the actual complete request's inclusive W boundary.
        configured = replace(SETTINGS, context_budget_tokens=required + delta)
        state, actual, _ = await run([response(message())], prompt=prompt, settings=configured, scenario=f"boundary_{delta:+d}")
        assert state["stop_reason"] == expected
        assert state["model_calls"] == (1 if delta >= 0 else 0)
        assert state["context_budget"]["request_sent"] is (delta >= 0)
        assert state["context_budget"]["trigger_percent"] == 100
        assert state["context_budget"]["trigger_tokens"] == required + delta
        if delta < 0:
            assert not actual.requests and state["history_end"] == 0
            assert state["usage_totals"] == {"input_tokens": None, "output_tokens": None, "total_tokens": None, "missing_usage_calls": 0}
    state, actual, checked = await run(
        [response(message())], tools=[TOOL], executor=fixed_tool,
        settings=replace(SETTINGS, max_model_calls=1), scenario="schema_and_final_instructions_in_budget",
    )
    assert actual.requests[0]["instructions"] == graph.FINAL_ANSWER_INSTRUCTIONS
    assert checked[0]["request"] == actual.requests[0]
    prepared = prepare_request(SETTINGS.model_name, original, tools=[TOOL], instructions="预算指令", tool_choice="none", max_output_tokens=4096)
    diag = check_context_budget(prepared, [None], SETTINGS, step_id="unit")
    assert diag["transparent_tokens"] == compact_bytes(prepared)
    stripped = {key: value for key, value in prepared.items() if key not in {"tools", "instructions", "tool_choice"}}
    assert compact_bytes(prepared) > compact_bytes(stripped)
    assert state["model_calls"] == 1

    # Exercise the new default window at its inclusive boundary using actual
    # requests; retain the original small-window cases above independently.
    production = ModelSettings(base_url=SETTINGS.base_url, model_name=SETTINGS.model_name, log_dir=SETTINGS.log_dir)
    padded = [{"role": "user", "content": ""}]
    prepared = prepare_request(production.model_name, padded, tools=ARTIFACT_TOOLS + HISTORY_TOOLS)
    assert prepared["max_output_tokens"] == production.max_output_tokens == 16384
    padded[0]["content"] = "p" * (225424 - compact_bytes(prepared))
    for extra in (0, 1):
        candidate = [{**padded[0], "content": padded[0]["content"] + "x" * extra}]
        state, actual, checked = await run(
            [response(message())], prompt=candidate, settings=production,
            scenario=f"production_input_boundary_{extra}",
        )
        assert state["context_budget"]["required_tokens"] == 250000 + extra
        assert state["stop_reason"] == ("completed" if extra == 0 else "context_budget_exceeded")
        assert len(actual.requests) == state["model_calls"] == (1 if extra == 0 else 0)
        if extra == 0:
            assert checked[0]["request"] == actual.requests[0]

async def check_stable_history_and_rejected_input():
    settings = replace(SETTINGS, context_budget_tokens=11000)
    manager = SessionManager()
    sid = manager.create_session(settings)
    source = [{"role": "user", "content": [{"type": "input_text", "text": "fact BLUE_19"}]}]
    first, _, _ = await run([response(message("BLUE_19 saved"))], manager=manager, sid=sid, prompt=source, scenario="history_first")
    saved = deepcopy(first.state)
    source[0]["content"][0]["text"] = "CALLER_MUTATION"
    first.state["input"][0]["content"][0]["text"] = "RETURN_MUTATION"
    large = "UNSENT_HUGE_" + "x" * 12000
    refused, client, _ = await run([], manager=manager, sid=sid, prompt=large, scenario="history_first_call_rejected")
    assert not client.requests and refused.checkpoint_complete
    assert refused.state["input"][-1]["content"] == large
    assert refused.state["history_end"] == len(saved["input"])
    assert refused.state["input"][:len(saved["input"])] == saved["input"]
    third, actual, _ = await run([response(message("BLUE_19"))], manager=manager, sid=sid, prompt="Recall?", scenario="history_after_rejection")
    assert actual.requests[0]["input"] == saved["input"] + [{"role": "user", "content": "Recall?"}]
    assert large not in json.dumps(actual.requests)
    assert third.state["model_calls"] == 1 and third.state["usage_totals"]["total_tokens"] == 12
    assert third.state["history_end"] == len(third.state["input"])

async def check_tool_result_limit_and_real_file():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        body = "REAL_BUDGET_BODY_" + "资料" * 2500
        (root / "budget source.txt").write_text(body, encoding="utf-8")
        original, cwd = snapshot(root), Path.cwd()
        settings = replace(SETTINGS, context_budget_tokens=11000)
        executor = create_filesystem_executor(root)
        manager = SessionManager()
        sid = manager.create_session(settings, tools=[TEXT_TOOL], execute_tool=executor)
        request = call("real_read", name="read_text_file", arguments='{"path":"budget source.txt"}')
        first, client, measured = await run([response(request, usage=(25, 4, 29))], manager=manager, sid=sid, scenario="file_result_exceeds_next_request")
        state = first.state
        assert state["stop_reason"] == "context_budget_exceeded" and state["model_calls"] == state["tool_calls"] == 1
        assert state["tool_outcomes"]["succeeded"] == 1 and len(measured) == 2
        result = outputs(state)[0]
        observation = json.loads(result["output"])["observation"]
        assert observation["mode"] == "artifact" and observation["total_chars"] == len(body)
        assert all(part["text"] == body[part["start"]:part["end"]] for part in observation["segments"])
        assert len(result["output"].encode("utf-8")) <= 4096
        assert result not in client.requests[0]["input"] and measured[1]["request"]["input"][-1] == result
        assert state["history_end"] == len(state["input"]) and first.checkpoint_complete
        assert state["usage_totals"] == {"input_tokens": 25, "output_tokens": 4, "total_tokens": 29, "missing_usage_calls": 0}
        again, second_client, _ = await run([], manager=manager, sid=sid, prompt="Recall the source", scenario="file_result_stays_in_history")
        assert not second_client.requests and again.state["tool_calls"] == again.state["model_calls"] == 0
        assert again.state["input"][:len(state["input"])] == state["input"]
        assert outputs(again.state) == [result]
        assert again.state["history_end"] == state["history_end"]
        assert snapshot(root) == original and Path.cwd() == cwd
        # A generous budget admits the bounded observation unchanged; it does
        # not restore the full archived body into the actual SDK request.
        roomy = replace(SETTINGS, context_budget_tokens=100000)
        ok, actual, checked = await run([response(request), response(message("used body"))], settings=roomy, tools=[TEXT_TOOL], executor=executor, scenario="real_file_result_dispatched")
        assert actual.requests[1]["input"][-1] == outputs(ok)[0] and len(checked) == 2
        assert json.loads(outputs(ok)[0]["output"])["observation"]["mode"] == "artifact"
        assert body not in json.dumps(actual.requests, ensure_ascii=False)
        assert body not in Path(ok["log_path"]).read_text(encoding="utf-8")

async def check_opaque_counting():
    items = [
        opaque_item(),
        {"type": "provider_unknown", "nested": {"encrypted_content": "do not double count"}},
        {"role": "user", "content": [{"type": "input_text", "text": "visible sibling"}, {"type": "input_image", "image_url": "PRIVATE_IMAGE"}]},
    ]
    original = deepcopy(items)
    risks = annotate_context_risks(items, fallback_tokens=4096, run_id="caller")
    assert [item["opaque_count"] for item in risks] == [1, 1, 1]
    assert all(item["risk_tokens"] == 4096 and item["source_model_call"] is None for item in risks)
    request = prepare_request(SETTINGS.model_name, items, max_output_tokens=SETTINGS.max_output_tokens)
    diag = check_context_budget(request, risks, SETTINGS, step_id="opaque-unit")
    assert diag["opaque_items"] == 3 and diag["opaque_risk_tokens"] == 3 * 4096
    larger_cipher = deepcopy(items)
    larger_cipher[0]["encrypted_content"] *= 5000
    grown = check_context_budget(prepare_request(
        SETTINGS.model_name, larger_cipher, max_output_tokens=SETTINGS.max_output_tokens,
    ), risks, SETTINGS)
    assert grown["transparent_tokens"] == diag["transparent_tokens"], "encrypted bytes were charged as transparent text"
    visible_growth = deepcopy(items)
    visible_growth[0]["summary"][0]["text"] += "visible growth" * 10
    assert check_context_budget(prepare_request(
        SETTINGS.model_name, visible_growth, max_output_tokens=SETTINGS.max_output_tokens,
    ), risks, SETTINGS)["transparent_tokens"] > diag["transparent_tokens"]
    assert items == original
    for output_tokens in (None, 0, -1, True, "99", 1.5):
        usage = {"output_tokens": output_tokens}
        annotated = annotate_context_risks(items, fallback_tokens=777, run_id="model-source", model_call=2, usage=usage)
        assert all(item["risk_tokens"] == 777 for item in annotated)
    genuine = annotate_context_risks(items, fallback_tokens=777, run_id="model-source", model_call=2, usage={"output_tokens": 91})
    assert all(item["risk_tokens"] == 91 and item["source_model_call"] == 2 for item in genuine)
    assert check_context_budget(request, genuine, SETTINGS)["opaque_risk_tokens"] == 273
    for invalid_risks in ([], [None, None, None], [{**genuine[0], "opaque_count": 2}, *genuine[1:]]):
        assert check_context_budget(request, invalid_risks, SETTINGS)["decision"] == "error"
    # Caller-supplied usage in an opaque protocol item is not source evidence.
    forged = [{"type": "provider_unknown", "usage": {"output_tokens": 1}}]
    assert annotate_context_risks(forged, fallback_tokens=777, run_id="caller")[0]["risk_tokens"] == 777
    string_result = [call("opaque_string"), {"type": "function_call_output", "call_id": "opaque_string", "output": '{"encrypted_content":"ordinary tool text"}'}]
    assert annotate_context_risks(string_result, fallback_tokens=777, run_id="caller") == [None, None]
    print(json.dumps({"scenario": "opaque_counting_units", "diagnostic": diag, "model_risk_tokens": 273}))

async def check_opaque_provenance_and_isolation():
    manager = SessionManager()
    settings = replace(SETTINGS, context_budget_tokens=100000)
    a, b = manager.create_session(settings), manager.create_session(settings)
    second_opaque = {"type": "provider_extension", "hidden": "PRIVATE_SECOND_OPAQUE"}
    raw = {"status": "completed", "output": [opaque_item("PRIVATE_CIPHER_A"), second_opaque, message("A")],
           "usage": {"input_tokens": 12, "output_tokens": 73, "total_tokens": 85}}
    first, _, _ = await run([raw], manager=manager, sid=a, scenario="opaque_source_a")
    source = deepcopy(first.state)
    opaque = next(value for value in source["context_risks"] if value is not None)
    assert opaque["risk_tokens"] == 73 and opaque["source_run_id"] == source["run_id"] and opaque["source_model_call"] == 1
    first.state["context_risks"][1]["risk_tokens"] = 999999
    other, _, _ = await run([response(opaque_item("PRIVATE_CIPHER_B"), message("B"), usage=None)], manager=manager, sid=b, scenario="opaque_source_missing_usage_b")
    assert next(value for value in other.state["context_risks"] if value)["risk_tokens"] == settings.opaque_fallback_tokens
    again, client, checks = await run([response(message("follow up"), usage=(10, 999, 1009))], manager=manager, sid=a, prompt="A follow-up", scenario="opaque_source_inherited_unchanged")
    assert client.requests[0]["input"][:len(source["input"])] == source["input"]
    assert checks[0]["diagnostic"]["opaque_risk_tokens"] == 146, "same-response opaque units must not share the output allowance"
    assert again.state["context_risks"][1] == source["context_risks"][1]
    assert "PRIVATE_CIPHER_B" not in json.dumps(client.requests)
    assert again.state["usage_totals"]["output_tokens"] == 999 and again.state["model_calls"] == 1
    assert await saved_state(manager, first) == source

async def check_file_sessions_isolation():
    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        manager, cases = SessionManager(), []
        for name, limit in (("A", 20000), ("B", 30000)):
            root = base / name
            root.mkdir()
            (root / "source.txt").write_text(f"FACT_{name}_BODY", encoding="utf-8")
            settings = replace(SETTINGS, context_budget_tokens=limit)
            sid = manager.create_session(settings, tools=[TEXT_TOOL], execute_tool=create_filesystem_executor(root, max_total_bytes=11))
            cases.append((name, sid, root, settings))
        before, cwd = snapshot(base), Path.cwd()
        historical = {}
        for name, sid, _, settings in cases:
            request = call(f"root_{name}", name="read_text_file", arguments='{"path":"source.txt"}')
            turn, actual, _ = await run([response(request), response(message(f"FACT_{name}_BODY"))], manager=manager, sid=sid, scenario=f"root_{name}_first")
            assert turn.state["context_budget"]["limit_tokens"] == settings.context_budget_tokens
            historical[name] = deepcopy(turn.state)
            assert json.loads(inline_raw(json.loads(outputs(turn.state)[0]["output"])))["content"] == f"FACT_{name}_BODY"
            assert actual.requests[1]["input"][-1] == outputs(turn.state)[0]
            starts = [event for event in read_events(turn.state) if event["event"] == "model.started"]
            assert [event["read_budget"]["remaining_bytes"] for event in starts] == [11, 0]
            assert all("instructions" not in request and assert_tools(request["tools"], [TEXT_TOOL]) for request in actual.requests)
        for name, sid, _, settings in reversed(cases):
            turn, actual, _ = await run([response(message(f"FACT_{name}_BODY"))], manager=manager, sid=sid, prompt="Recall the source", scenario=f"root_{name}_followup")
            assert actual.requests[0]["input"] == historical[name]["input"] + [{"role": "user", "content": "Recall the source"}]
            assert ("FACT_B_BODY" if name == "A" else "FACT_A_BODY") not in json.dumps(actual.requests)
            assert turn.state["model_calls"] == 1 and turn.state["tool_calls"] == 0
            assert turn.state["context_budget"]["limit_tokens"] == settings.context_budget_tokens
            started = next(event for event in read_events(turn.state) if event["event"] == "model.started")
            assert started["read_budget"] == {"limit_bytes": 11, "used_bytes": 0, "remaining_bytes": 11}
            assert started["remaining_model_calls"] == settings.max_model_calls
            assert "instructions" not in actual.requests[0] and assert_tools(actual.requests[0]["tools"], [TEXT_TOOL])
        assert snapshot(base) == before and Path.cwd() == cwd

async def check_every_call_and_usage():
    state, client, checked = await run([
        response(call("one"), usage=(10, None, None)),
        response(call("two"), usage=None),
        response(message("complete"), usage=(30, 5, 35)),
    ], tools=[TOOL], executor=fixed_tool, scenario="every_model_call_checked")
    assert len(client.requests) == len(checked) == 3
    assert state["usage_totals"] == {"input_tokens": 40, "output_tokens": 5, "total_tokens": 35, "missing_usage_calls": 2}
    assert [request["input"][-1].get("call_id") for request in client.requests[1:]] == ["one", "two"]
    assert [item["diagnostic"]["step_id"] for item in checked] == ["step-0001", "step-0003", "step-0005"]

async def check_measurement_failure():
    invalid = [float("nan"), float("inf"), object()]
    cycle = {"type": "provider_extension"}
    cycle["cycle"] = cycle
    inputs = [[{"type": "provider_extension", "value": value}] for value in invalid] + [[cycle]]
    for index, items in enumerate(inputs):
        # Independent Runs now use the same confirmed saver as sessions.
        # Its serializer rejects object/cycle before the business budget gate.
        state, client, _ = await run([], prompt=items, scenario=f"measurement_unserializable_{index}")
        assert not client.requests and state["model_calls"] == 0
        if index < 2:
            assert state["error_code"] == "context_measurement_error"
            assert state["context_budget"]["decision"] == "error" and state["context_budget"]["request_sent"] is False
        else:
            assert state["error_code"] == "checkpoint_write_error" and state["context_budget"] is None
        assert state["usage_totals"]["missing_usage_calls"] == 0
    with patch.object(runtime, "prepare_request", side_effect=ValueError(PRIVATE)):
        state, _, _ = await run([], scenario="request_preparation_failure")
    assert state["error_code"] == "context_measurement_error"

async def check_session_invalid_storage_boundaries():
    """A real saver failure precedes measurement; JSON rejection is separate.

    These cases do not inject storage faults: InMemorySaver's actual serializer
    decides which input facts can be checkpointed. Invalid JSON that it accepts
    still must fail the model-request measurement and remain non-inheritable.
    """
    cycle = {"type": "provider_extension"}
    cycle["cycle"] = cycle
    cases = (
        ("object", {"type": "provider_extension", "value": object()}, False),
        ("cycle", cycle, False),
        ("set", {"type": "provider_extension", "value": {"one", "two"}}, True),
        ("nan", {"type": "provider_extension", "value": float("nan")}, True),
    )
    for label, invalid, serializable in cases:
        manager = SessionManager()
        sid = manager.create_session(SETTINGS)
        first, _, _ = await run([response(message("SAVED_VALID_HISTORY"))], manager=manager, sid=sid, prompt="remember the fact", scenario=f"session_invalid_{label}_prior")
        prior = deepcopy(first.state)
        prior_storage = deepcopy(manager._checkpointer.storage[sid])
        refused, client, measured = await run([], manager=manager, sid=sid, prompt=[invalid], scenario=f"session_invalid_{label}_submission")
        state = refused.state
        assert not client.requests and state["model_calls"] == state["tool_calls"] == 0
        assert state["usage_totals"] == {"input_tokens": None, "output_tokens": None, "total_tokens": None, "missing_usage_calls": 0}
        assert state["history_end"] == len(prior["input"])
        assert state["input"][:len(prior["input"])] == prior["input"]
        assert len(state["input"]) == len(prior["input"]) + 1
        for namespace, checkpoints in prior_storage.items():
            for checkpoint_id, stored in checkpoints.items():
                assert manager._checkpointer.storage[sid][namespace][checkpoint_id] == stored, "failed submission modified a previous checkpoint"
        if serializable:
            assert await saved_state(manager, first) == prior
            assert refused.checkpoint_complete and len(measured) == 1
            assert state["error_code"] == "context_measurement_error"
            assert state["context_budget"]["decision"] == "error" and not state["context_budget"]["request_sent"]
            third, actual, _ = await run([response(message("SAVED_VALID_HISTORY"))], manager=manager, sid=sid, prompt="Recall the fact", scenario=f"session_invalid_{label}_followup")
            assert third.checkpoint_complete and third.state["stop_reason"] == "completed"
            assert actual.requests[0]["input"] == prior["input"] + [{"role": "user", "content": "Recall the fact"}]
        else:
            assert refused.checkpoint_error == "checkpoint_write_error" and not refused.checkpoint_complete
            assert refused.checkpoint_id is None and state["context_budget"] is None and not measured
            assert state["error_code"] == "checkpoint_write_error"
            assert not any(event["event"] == "context_budget.checked" for event in read_events(state))
            if label == "cycle":
                assert state["input"][-1]["cycle"] is state["input"][-1], "returned facts lost the original cyclic relation"
            else:
                assert type(state["input"][-1]["value"]) is object
            with patch.object(runtime, "create_client", side_effect=AssertionError("blocked session tried to start")):
                try:
                    await manager.run_turn(sid, "must remain blocked")
                except SessionError as exc:
                    assert exc.code == "session_blocked"
                else:
                    raise AssertionError("uncertain save must block this session")
            print(json.dumps({"scenario": f"session_invalid_{label}_blocked", "session_id": sid,
                              "run_id": state["run_id"], "prior_checkpoint_id": first.checkpoint_id,
                              "checkpoint_error": refused.checkpoint_error, "log_path": state["log_path"],
                              "failure_source": "actual_checkpoint_serialization"}))

async def check_budget_recording_failures():
    for event in ("context_budget.checked", "model.started"):
        with recording_failure(event) as faults:
            state, client, measured = await run([], scenario=f"logging_failure_{event}", faults=faults)
        assert faults and not client.requests and state["model_calls"] == 0
        assert state["context_budget"]["decision"] == "allow" and state["context_budget"]["request_sent"] is False
        assert state["error_code"] == "logging_error" and not state["log_complete"]
    manager = SessionManager()
    sid = manager.create_session(SETTINGS)
    with recording_failure("context_budget.checked") as faults:
        turn, _, _ = await run([], manager=manager, sid=sid, prompt="x" * 40000, scenario="budget_exceeded_and_log_failure", faults=faults)
    assert turn.checkpoint_complete and turn.state["context_budget"]["decision"] == "exceeded"
    assert turn.state["error_code"] == "logging_error" and turn.state["history_end"] == 0
    with recording_failure("context_budget.checked") as faults:
        state, client, _ = await run([], prompt=[{"type": "provider_extension", "value": float("nan")}], scenario="measurement_error_and_log_failure", faults=faults)
    assert faults and not client.requests and state["model_calls"] == 0
    assert state["context_budget"]["decision"] == "error"
    assert state["context_budget"]["error_code"] == "context_measurement_error"
    assert state["error_code"] == "logging_error" and not state["log_complete"]
    assert state["usage_totals"]["missing_usage_calls"] == 0

    # Fail the actual write only at the second model step, after a real tool
    # completed. Capture safe attempted-event metadata rather than fake a State.
    original_open, faults, streams = RunRecorder._open, [], []
    class FailSecondBudget(FaultyFile):
        def write(self, value):
            event = json.loads(value)
            if event["event"] == "context_budget.checked" and event.get("step_id") == "step-0003":
                self.event_name = "context_budget.checked"
            return super().write(value)
    def open_file(recorder):
        stream = FailSecondBudget(original_open(recorder), faults)
        streams.append(stream)
        return stream
    # The tool result is archived; the model's own current-turn output still
    # exercises the complete-request budget gate after tool execution.
    async def huge_result(name, arguments):
        return "TOOL_FACT_PRESERVED_" + "x" * 40000
    with patch.object(RunRecorder, "_open", open_file):
        state, client, _ = await run([response(call("before_budget_log_failure"), message("CURRENT_MODEL_CONTEXT_" + "z" * 40000), usage=(22, 3, 25))], tools=[TOOL], executor=huge_result, scenario="tool_result_budget_recording_failure", faults=faults)
    assert faults and all(stream.stream.closed for stream in streams)
    assert state["context_budget"]["decision"] == "exceeded" and state["error_code"] == "logging_error"
    assert state["model_calls"] == state["tool_calls"] == 1 and len(client.requests) == 1
    assert state["usage_totals"]["total_tokens"] == 25 and state["usage_totals"]["missing_usage_calls"] == 0
    assert json.loads(outputs(state)[0]["output"])["ok"] is True and state["history_end"] == len(state["input"])

async def check_budget_checkpoint_failures():
    for stage in ("node", "final"):
        manager = SessionManager()
        sid = manager.create_session(SETTINGS)
        def predicate(checkpoint, metadata, *_):
            if stage == "final":
                return metadata["source"] == "update"
            # Recovery now has its own preparation checkpoint. Fail the settled
            # budget-stop node, preserving this check's original terminal boundary.
            values = checkpoint["channel_values"]
            return (metadata["source"] == "loop"
                    and values.get("stop_reason") == "context_budget_exceeded"
                    and (values.get("context_budget") or {}).get("decision") == "exceeded")
        with checkpoint_fault(manager, "put", predicate) as faults:
            turn, client, _ = await run([], manager=manager, sid=sid, prompt="x" * 40000, scenario=f"budget_{stage}_checkpoint_failure", faults=faults)
        assert faults and not client.requests and not turn.checkpoint_complete
        assert turn.checkpoint_error == "checkpoint_write_error" and turn.checkpoint_id is None
        assert turn.state["context_budget"]["decision"] == "exceeded"
        assert turn.state["error_code"] == "context_budget_exceeded"
        assert turn.state["usage_totals"]["missing_usage_calls"] == 0
    manager = SessionManager()
    # The tool result is archived; the model's own current-turn output still
    # exercises the complete-request budget gate after tool execution.
    async def huge_result(name, arguments):
        return "TOOL_FACT_PRESERVED_" + "x" * 40000
    sid = manager.create_session(SETTINGS, tools=[TOOL], execute_tool=huge_result)
    with checkpoint_fault(manager, "put", lambda cp, meta, *_: meta["source"] == "update") as faults:
        turn, client, _ = await run([response(call("before_budget_save_failure"), message("CURRENT_MODEL_CONTEXT_" + "z" * 40000), usage=(22, 3, 25))], manager=manager, sid=sid, scenario="tool_result_budget_final_save_failure", faults=faults)
    assert faults and turn.state["context_budget"]["decision"] == "exceeded"
    assert turn.state["model_calls"] == turn.state["tool_calls"] == 1 and len(client.requests) == 1
    assert turn.state["usage_totals"]["total_tokens"] == 25 and len(outputs(turn.state)) == 1
    assert not turn.checkpoint_complete and turn.checkpoint_error == "checkpoint_write_error"

async def check_tracing_diagnostics():
    results = []
    for mode in ("disabled", "ok", "connection"):
        with capture("connection" if mode == "connection" else "ok", LANGSMITH_TRACING="false" if mode == "disabled" else "true") as (transport, stderr):
            state, client, _ = await run([], prompt="x" * 40000, scenario=f"tracing_budget_{mode}")
        assert not client.requests and state["context_budget"]["decision"] == "exceeded"
        if mode == "disabled":
            assert not transport.sent
        else:
            serialized = json.dumps(transport.sent)
            assert "context_budget_exceeded" in serialized and "trigger_tokens" in serialized
            assert "x" * 40000 not in serialized
            assert not any(span["run_type"] == "llm" for span in transport.sent)
            if mode == "connection":
                assert stderr.getvalue() and "PRIVATE_TRANSPORT_EXCEPTION" not in stderr.getvalue()
        results.append(state)
    fields = ("stop_reason", "error_code", "model_calls", "tool_calls", "usage_totals", "context_budget")
    assert all({key: result[key] for key in fields} == {key: results[0][key] for key in fields} for result in results[1:])
    with capture() as (transport, _):
        state, client, measured = await run([response(message("normal"))], scenario="tracing_allowed_budget")
    assert state["model_calls"] == 1 and len(client.requests) == 1
    assert "trigger_tokens" in json.dumps(transport.sent)

async def check_cli():
    settings = replace(SETTINGS, context_budget_tokens=10000)
    state, _, _ = await run([], prompt="x" * 10000, settings=settings, scenario="cli_budget_state")
    with redirect_stderr(StringIO()) as error, redirect_stdout(StringIO()) as output:
        assert cli.display_run(state, settings) == 4
    assert not output.getvalue() and state["run_id"] in error.getvalue()
    assert "预算" in error.getvalue()
    clients = [LocalClient([]), LocalClient([response(message("accepted follow-up"))])]
    displayed = []
    original_display = cli.display_turn
    def display(turn, configured):
        observe_turn(turn)
        displayed.append(turn)
        return original_display(turn, configured)
    with patch("builtins.input", side_effect=["x" * 10000, "small", "/exit"]), patch.object(runtime, "create_client", side_effect=clients), patch.object(cli, "display_turn", side_effect=display), capture_measurements() as measured, redirect_stderr(StringIO()) as error, redirect_stdout(StringIO()) as output:
        assert await cli.interactive(settings, {}) == 0
    assert "accepted follow-up" in output.getvalue()
    assert not clients[0].requests and len(clients[1].requests) == 1 and all(client.closed for client in clients)
    assert clients[1].requests[0]["input"] == [{"role": "user", "content": "small"}]
    assert len(displayed) == len(measured) == 2
    for index, (turn, client, checked) in enumerate(zip(displayed, clients, measured, strict=True)):
        assert turn.checkpoint_complete
        audit(turn.state, client, [checked], scenario=f"interactive_budget_turn_{index + 1}", turn=turn)
    print(json.dumps({"scenario": "interactive_continues_after_budget_refusal", "model_calls": [len(client.requests) for client in clients]}))

async def main():
    checks = (
        check_configuration, check_transparent_request_and_boundary,
        check_stable_history_and_rejected_input, check_tool_result_limit_and_real_file,
        check_opaque_counting, check_opaque_provenance_and_isolation,
        check_file_sessions_isolation,
        check_every_call_and_usage, check_measurement_failure,
        check_session_invalid_storage_boundaries,
        check_budget_recording_failures, check_budget_checkpoint_failures,
        check_tracing_diagnostics, check_cli,
    )
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
        for check in checks:
            with check_group(group_id(check)):
                await check()
            print(f"PASS {check.__name__}")
    print(f"Passed {len(checks)} context budget groups; {len(RUNS)} captured runs. Logs: {CHECK_DIR}")

def execute():
    asyncio.run(main())
