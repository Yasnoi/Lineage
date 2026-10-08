"""End-to-end deterministic contracts for current-Run and two-lane recovery."""

from copy import deepcopy
import asyncio
from dataclasses import replace
import json
import re
import xml.etree.ElementTree as ET
from lineage_agent.models.summary_input import serialize_conversation
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from langgraph.errors import NodeCancelledError

from benchmarks.checks.support.runtime import LocalClient, TOOL, call, message, read_events, response
from lineage_agent.agent import runtime
from lineage_agent.agent.checkpointing import SessionCheckpointer
from lineage_agent.agent.context import ContextRecords, json_bytes
from lineage_agent.agent.sessions import SessionManager
from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.budget import annotate_context_risks, check_context_budget
from lineage_agent.models.requests import prepare_request
from lineage_agent.models.summarization import is_summary_request, summary_request_lane
from lineage_agent.tools.types import ToolExecutionError


def _summary(request, text="Earlier tool observations are available through history recall."):
    return response(message(text))



class _Client(LocalClient):
    def __init__(self, replies, summary=None):
        super().__init__(replies)
        self.summary = summary or _summary
        self.summary_requests = []
        self.business_requests = []

    async def create(self, **request):
        self.requests.append(deepcopy(request))
        if is_summary_request(request):
            self.summary_requests.append(deepcopy(request))
            value = self.summary(request)
        else:
            self.business_requests.append(deepcopy(request))
            try:
                value = next(self.replies)
            except StopIteration:
                raise AssertionError("unexpected extra business request") from None
            if callable(value):
                value = value(request)
        if isinstance(value, BaseException):
            raise value
        return value


class PrefixCompactionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(prefix="lineage-prefix-test-")
        self.addCleanup(self.tmp.cleanup)
        self.settings = ModelSettings(
            base_url="https://unused.invalid/v1", model_name="local-check",
            log_dir=Path(self.tmp.name), application_root=Path(self.tmp.name), context_budget_tokens=24000,
            summary_history_max_output_tokens=512, summary_run_max_output_tokens=256,
            max_output_tokens=256, context_safety_margin_tokens=64, max_model_calls=25,
        )
        self.executed = []

    async def tool(self, name, arguments):
        self.executed.append(arguments["key"])
        return "omitted-detail=37; not 73; " + arguments["key"] + "; " + "x" * 3200

    def replies(self, count, *, prefix="c"):
        return [response(call(f"{prefix}{index}", arguments=json.dumps({"key": f"{prefix}{index}"})))
                for index in range(count)] + [response(message("completed"))]

    async def standalone(self, count=10, *, replies=None, summary=None, prompt="必须保留 17 个字段；不得写文件；只输出 JSON。"):
        client = _Client(self.replies(count) if replies is None else replies, summary)
        with patch.object(runtime, "create_client", return_value=client):
            state = await runtime.run_agent(prompt, self.settings, tools=[TOOL], execute_tool=self.tool)
        return state, client

    def assert_complete(self, state):
        self.assertEqual(state["stop_reason"], "completed", (state.get("error_code"), state.get("error"), state.get("compaction")))

    async def test_independent_long_run_recovers_current_prefix_and_protects_three_groups(self):
        from benchmarks.core.compact_evidence import native_binding
        from benchmarks.core.evidence import digest
        original_material = ContextRecords.material
        original_send = runtime.send_prepared_response
        materials = []
        bindings = []
        async def audit_send(client, request, **kwargs):
            if kwargs.get("purpose") == "summary":
                observer = SimpleNamespace(summary_contexts={digest(request): kwargs["summary_metadata"]})
                binding = native_binding(observer, request)
                self.assertTrue(binding["verified"], binding["reason"])
                bindings.append(binding)
            return await original_send(client, request, **kwargs)
        async def audit_material(records, compaction, lane, state):
            material = await original_material(records, compaction, lane, state)
            self.assertNotIn("allowed_source_ids", material)
            self.assertEqual(len(material["input"]), len(material["item_refs"]))
            for item, ref in zip(material["input"], material["item_refs"], strict=True):
                if ref is not None:
                    record = await records.store.get(ref, checkpoint_id=compaction["snapshot"]["source_checkpoint_id"])
                    self.assertEqual(item, record["item"])
            self.assertNotIn("sources", material)
            materials.append(deepcopy(material))
            return material
        with patch.object(ContextRecords, "material", new=audit_material), \
                patch.object(runtime, "send_prepared_response", new=audit_send):
            state, client = await self.standalone()
        self.assertEqual([request["input"] for request in client.summary_requests], [serialize_conversation(material["input"], item_details=material["item_details"]).input for material in materials])
        self.assert_complete(state)
        self.assertGreater(state["summary_calls"], 0)
        self.assertEqual(len(bindings), state["summary_calls"])
        self.assertEqual(state["model_calls"], 11)
        self.assertEqual(self.executed, [f"c{index}" for index in range(10)])
        self.assertTrue(all(summary_request_lane(request) == "R" for request in client.summary_requests))
        self.assertIsNone(state["active_context"]["history_summary"])
        self.assertTrue(any(span["lane"] == "R" for span in state["active_context"]["summary_spans"]))
        self.assertIsNone(state["active_context"]["current_run_id"])
        original = client.business_requests[0]["input"][0]
        for request in client.business_requests:
            self.assertIn(original, request["input"])
        business_count = 0
        for request in client.requests:
            if not is_summary_request(request):
                business_count += 1
                continue
            transcript = ET.fromstring("<root>" + request["input"][0]["content"] + "</root>")
            source_calls = [item.attrib["call_id"] for item in transcript.iter("tool_call")]
            self.assertTrue(source_calls)
            for index in range(business_count - 3, business_count):
                self.assertNotIn(f"c{index}", source_calls)
            self.assertIn(original["content"], transcript.findtext(".//current_request_background"))
        self.assertTrue(all(event.get("session_id") is None for event in read_events(state)))
        self.assertGreaterEqual(len(client.summary_requests), 2)
        self.assertEqual(len([span for span in state["active_context"]["summary_spans"] if span["lane"] == "R"]), 1)

    async def test_session_first_long_run_originals_and_r_inherit_with_old_turn(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=self.tool)
            first_client = _Client(self.replies(10))
            with patch.object(runtime, "create_client", return_value=first_client):
                first = await manager.run_turn(sid, "原始要求必须保留数字 17。")
            self.assert_complete(first.state)
            record = await manager.get_record(sid, first.state["run_id"])
            outputs = [item for item in record["input"] if item.get("type") == "function_call_output"]
            self.assertEqual(len(outputs), 10)
            self.assertNotIn("derived_context_summary", json.dumps(record["input"]))
            second_client = _Client([response(message("second done"))])
            with patch.object(runtime, "create_client", return_value=second_client):
                second = await manager.run_turn(sid, "新的问题。")
            self.assert_complete(second.state)
            self.assertIsNone(second.state["active_context"]["run_summary"])
            self.assertEqual(record, await manager.get_record(sid, first.state["run_id"]))
            self.assertTrue(any(span["run_id"] == first.state["run_id"] for span in second.state["active_context"]["summary_spans"]))

    async def test_original_call_ids_remain_reserved_after_prefix_compaction(self):
        replies = self.replies(8)[:-1] + [response(call("c0", arguments='{"key":"must-not-repeat"}'))]
        state, client = await self.standalone(replies=replies)
        self.assertGreater(state["summary_calls"], 0)
        self.assertEqual(state["error_code"], "duplicate_call_id")
        self.assertNotIn("must-not-repeat", self.executed)
        self.assertEqual(self.executed.count("c0"), 1)
        self.assertEqual(state["tool_calls"], 8)

    async def test_summary_failure_keeps_active_originals_and_does_not_replay_tools(self):
        state, client = await self.standalone(summary=lambda request: RuntimeError("failed summary"))
        self.assertEqual(state["summary_calls"], 1)
        self.assertEqual(len(self.executed), len(set(self.executed)))
        self.assertIsNone(state["active_context"]["run_summary"])
        actual = [item["call_id"] for item in state["input"] if item.get("type") == "function_call_output"]
        self.assertEqual(actual, self.executed)
        self.assertNotEqual(state["stop_reason"], "completed")

    async def test_history_and_current_prefix_generate_h_then_r_and_commit_together(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=self.tool)
            seed = _Client([response(message("historical observations " + "h" * 4200))])
            with patch.object(runtime, "create_client", return_value=seed):
                old = await manager.run_turn(sid, "history-only task")
            client = _Client(self.replies(6))
            with patch.object(runtime, "create_client", return_value=client):
                turn = await manager.run_turn(sid, "current task; preserve 17 and do not write")
            self.assert_complete(turn.state)
            lanes = [summary_request_lane(request) for request in client.summary_requests]
            self.assertEqual(lanes[:2], ["H", "R"])
            self.assertIsNotNone(turn.state["active_context"]["history_summary"])
            self.assertTrue(any(span["lane"] == "R" for span in turn.state["active_context"]["summary_spans"]))
            history = client.summary_requests[0]["input"][0]["content"]
            current = client.summary_requests[1]["input"][0]["content"]
            self.assertIn(old.state["input"][0]["content"], history)
            self.assertNotIn(old.state["input"][0]["content"], current)
            self.assertIn("current task; preserve 17 and do not write", current)
            self.assertNotIn("<tool_call", history)
            self.assertEqual(turn.state["summary_calls"], len(client.summary_requests))

    async def test_h_success_r_failure_keeps_both_original_views(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=self.tool)
            seed = _Client([response(message("historical source " + "h" * 4200))])
            with patch.object(runtime, "create_client", return_value=seed):
                old = await manager.run_turn(sid, "previous task")
            def fail_current(request):
                return RuntimeError("R failed") if summary_request_lane(request) == "R" else _summary(request)
            client = _Client(self.replies(6), fail_current)
            with patch.object(runtime, "create_client", return_value=client):
                turn = await manager.run_turn(sid, "current task")
            state = turn.state
            self.assertEqual([summary_request_lane(request) for request in client.summary_requests], ["H", "R"])
            self.assertEqual(state["summary_calls"], 2)
            self.assertIsNone(state["active_context"]["history_summary"])
            self.assertIsNone(state["active_context"]["run_summary"])
            self.assertIn(old.state["input"][0], state["input"])
            actual = [item["call_id"] for item in state["input"] if item.get("type") == "function_call_output"]
            self.assertEqual(actual, self.executed)
            self.assertEqual(len(actual), len(set(actual)))

    async def test_failed_first_model_call_history_keeps_error_outcome_when_compacted(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=self.tool)
            failed_client = _Client([RuntimeError("first request failed")])
            with patch.object(runtime, "create_client", return_value=failed_client):
                old = await manager.run_turn(sid, "previous task failed before obtaining any model output")
            self.assertEqual(old.state["stop_reason"], "error")
            self.assertTrue(old.checkpoint_complete)
            client = _Client(self.replies(6))
            with patch.object(runtime, "create_client", return_value=client):
                turn = await manager.run_turn(sid, "current task")
            self.assert_complete(turn.state)
            historical = next(request for request in client.summary_requests if summary_request_lane(request) == "H")
            note = historical["input"][0]["content"]
            self.assertIn(old.state["run_id"], note)
            self.assertIn("error", note)
            self.assertIn("model_error", note)
            self.assertFalse(any(item.get("type") == "function_call_output" for item in historical["input"]))

    async def test_current_protection_never_preserves_historical_turns(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=self.tool)
            turns = []
            for prompt, replies in (("old removable", [response(message("h" * 6000))]),
                                    ("boundary entire user input", self.replies(2, prefix="old")),
                                    ("last user", [response(message("z" * 1000))])):
                client = _Client(replies)
                with patch.object(runtime, "create_client", return_value=client):
                    turns.append(await manager.run_turn(sid, prompt))
                self.assert_complete(turns[-1].state)
            old_input = turns[-1].state["input"]
            tool_defs = client.business_requests[0]["tools"]
            request = prepare_request(self.settings.model_name, old_input + [{"role": "user", "content": ""}],
                                      tools=tool_defs, max_output_tokens=self.settings.max_output_tokens)
            base = check_context_budget(request, annotate_context_risks(request["input"], fallback_tokens=4096, run_id="test"), self.settings)["required_tokens"]
            prompt = "n" * (self.settings.context_budget_tokens - base + 128)
            client = _Client([response(message("current done"))])
            with patch.object(runtime, "create_client", return_value=client):
                turn = await manager.run_turn(sid, prompt)
            self.assert_complete(turn.state)
            self.assertEqual([summary_request_lane(request) for request in client.summary_requests], ["H"])
            active_turns = turn.state["active_context"]["turns"]
            self.assertEqual([ref["run_id"] for ref in active_turns], [turn.state["run_id"]])
            self.assertNotIn({"role": "user", "content": "boundary entire user input"}, client.business_requests[0]["input"])
            preserved = [item["call_id"] for item in client.business_requests[0]["input"] if item.get("type") == "function_call_output"]
            self.assertEqual(preserved, [])
            ranges = turn.state["active_context"]["history_summary"]["ranges"]
            self.assertEqual({bounds["run_id"] for bounds in ranges}, {item.state["run_id"] for item in turns})

    async def test_model_searches_and_reads_detail_missing_from_r_without_tool_replay(self):
        observed = {}
        def ask_search(request):
            active = request["input"]
            self.assertFalse(any(item.get("type") == "function_call_output" and item["call_id"] == "c0" for item in active))
            summaries = [item["content"] for item in active
                         if item.get("role") == "assistant" and isinstance(item.get("content"), str)
                         and item["content"].startswith("当前进展摘要 R")]
            self.assertEqual(len(summaries), 1)
            self.assertNotIn("omitted-detail=37", summaries[0])
            observed["history_ref"] = re.search(r"history://scope/[0-9a-f-]+", summaries[0])[0]
            return response(call("hs0", name="history_search", arguments=json.dumps({
                "history_ref": observed["history_ref"], "keywords": ["omitted-detail=37; not 73; c0"], "cursor": None,
            })))
        def ask_read(request):
            output = next(item for item in request["input"] if item.get("call_id") == "hs0" and item.get("type") == "function_call_output")
            search = json.loads(json.loads(output["output"])["observation"]["raw_result"])
            self.assertEqual(len(search["hits"]), 1)
            observed["source_ref"] = search["hits"][0]["source_ref"]
            return response(call("hr0", name="history_read", arguments=json.dumps({
                "source_ref": observed["source_ref"], "start": 0, "end": None,
            })))
        def answer(request):
            output = next(item for item in request["input"] if item.get("call_id") == "hr0" and item.get("type") == "function_call_output")
            recalled = json.loads(json.loads(output["output"])["observation"]["raw_result"])
            self.assertIn("omitted-detail=37; not 73; c0", recalled["text"])
            return response(message("37; not 73; supported by " + observed["source_ref"]))
        replies = self.replies(6)[:-1] + [ask_search, ask_read, answer]
        state, client = await self.standalone(replies=replies)
        self.assert_complete(state)
        self.assertGreater(state["summary_calls"], 0)
        self.assertEqual(state["tool_calls"], 8)
        self.assertEqual(self.executed, [f"c{index}" for index in range(6)])
        self.assertIn("37; not 73", state["result"]["text"])
        self.assertTrue(observed["source_ref"].startswith("history://record/"))

    async def test_oversized_valid_candidate_is_shortened_once_without_original_bodies(self):
        requests = []
        def summary(request):
            requests.append(deepcopy(request))
            return _summary(request, "L" * 11000 if len(requests) == 1 else "shortened candidate facts")
        state, client = await self.standalone(count=6, summary=summary)
        self.assert_complete(state)
        self.assertEqual(len(requests), 2)
        self.assertEqual(state["compaction"]["attempt_no"], 2)
        first, second = requests[0]["input"], requests[1]["input"]
        self.assertIn("<tool_result", first[0]["content"])
        self.assertEqual(len(second), 1)
        self.assertIn("<candidate_summary", second[0]["content"])
        self.assertIn("L" * 11000, second[0]["content"])
        self.assertFalse(any(item.get("type") == "function_call_output" for item in second))
        self.assertNotIn("omitted-detail=37", json.dumps(second))
        self.assertEqual(state["summary_calls"], 2)
        self.assertEqual(self.executed, [f"c{index}" for index in range(6)])

    async def test_empty_summary_stops_after_one_call_and_keeps_original_view(self):
        def invalid(request):
            return response(message("   "))
        state, client = await self.standalone(count=6, summary=invalid)
        self.assertEqual(state["summary_calls"], 1)
        self.assertEqual(state["error_code"], "summary_model_error")
        self.assertIsNone(state["active_context"]["run_summary"])
        self.assertEqual(len(self.executed), 6)
        self.assertEqual(len(client.business_requests), 6)

    async def test_checkpoint_failures_keep_old_activity_and_actual_tool_and_summary_facts(self):
        for target in ("current_ready", "validated", "pending_save"):
            with self.subTest(target=target):
                self.executed.clear()
                native_put = SessionCheckpointer.put
                observed = []
                def fail(saver, config, checkpoint, metadata, new_versions):
                    values = checkpoint["channel_values"]
                    if (values.get("compaction") or {}).get("status") == target:
                        observed.append(deepcopy(values))
                        raise OSError("injected persistence failure")
                    return native_put(saver, config, checkpoint, metadata, new_versions)
                with patch.object(SessionCheckpointer, "put", new=fail):
                    state, client = await self.standalone(count=6)
                self.assertTrue(observed)
                self.assertEqual(state["error_code"], "checkpoint_write_error")
                self.assertEqual(state["summary_calls"], 1)
                self.assertIsNone(state["active_context"]["run_summary"])
                self.assertEqual(self.executed, [f"c{index}" for index in range(6)])
                self.assertEqual([item["call_id"] for item in state["input"] if item.get("type") == "function_call_output"], self.executed)
                self.assertEqual(state["tool_outcomes"]["succeeded"], 6)
                self.assertEqual(len(client.business_requests), 6)

    async def test_two_complete_attempts_can_call_both_lanes_but_never_a_fifth_summary(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=self.tool)
            seed = _Client([response(message("historical source " + "h" * 4200))])
            with patch.object(runtime, "create_client", return_value=seed):
                await manager.run_turn(sid, "previous task")
            requests = []
            def summary(request):
                requests.append(deepcopy(request))
                return _summary(request, "L" * 11000 if len(requests) <= 2 else "shortened valid facts")
            client = _Client(self.replies(6), summary)
            with patch.object(runtime, "create_client", return_value=client):
                turn = await manager.run_turn(sid, "current task")
            self.assert_complete(turn.state)
            self.assertEqual([summary_request_lane(request) for request in requests], ["H", "R", "H", "R"])
            self.assertEqual(turn.state["summary_calls"], 4)
            self.assertEqual(turn.state["compaction"]["attempt_no"], 2)
            for request in requests[2:]:
                self.assertEqual(len(request["input"]), 1)
                self.assertIn("<candidate_summary", request["input"][0]["content"])
                self.assertIn("L" * 11000, request["input"][0]["content"])
                self.assertNotIn("omitted-detail=37", json.dumps(request["input"]))

    async def test_second_oversized_candidate_stops_without_reinitializing_attempts(self):
        state, client = await self.standalone(count=6, summary=lambda request: _summary(request, "L" * 11000))
        self.assertEqual(state["summary_calls"], 2)
        self.assertEqual(state["compaction"]["attempt_no"], 2)
        self.assertEqual(state["stop_reason"], "context_budget_exceeded")
        self.assertIsNone(state["active_context"]["run_summary"])
        self.assertEqual(len(client.business_requests), 6)
        self.assertEqual(self.executed, [f"c{index}" for index in range(6)])

    async def test_source_version_change_preserves_latest_confirmed_view(self):
        native_put = SessionCheckpointer.put
        changed = []
        def mutate(saver, config, checkpoint, metadata, new_versions):
            values = checkpoint["channel_values"]
            if not changed and (values.get("compaction") or {}).get("status") == "current_ready":
                checkpoint = deepcopy(checkpoint)
                checkpoint["channel_values"]["active_context"]["version_id"] = "confirmed-newer-version"
                new_versions = dict(new_versions)
                version = saver.get_next_version(checkpoint["channel_versions"]["active_context"], None)
                checkpoint["channel_versions"]["active_context"] = version
                new_versions["active_context"] = version
                changed.append(True)
            return native_put(saver, config, checkpoint, metadata, new_versions)
        with patch.object(SessionCheckpointer, "put", new=mutate):
            state, client = await self.standalone(count=6)
        self.assertTrue(changed)
        self.assertEqual(state["compaction"]["reason"], "source_changed")
        self.assertEqual(state["active_context"]["version_id"], "confirmed-newer-version")
        self.assertIsNone(state["active_context"]["run_summary"])
        self.assertEqual(state["summary_calls"], 1)
        self.assertEqual(len(client.business_requests), 6)

    async def test_protected_context_too_large_stops_without_reducing_three_groups(self):
        self.settings = replace(self.settings, context_budget_tokens=14000)
        state, client = await self.standalone(count=6)
        self.assertEqual(state["summary_calls"], 0)
        self.assertEqual(state["stop_reason"], "context_budget_exceeded")
        self.assertLessEqual(len(self.executed), 3)
        self.assertIsNone(state["active_context"]["run_summary"])
        self.assertFalse(client.summary_requests)

    async def test_summary_request_preflight_refuses_without_sending_or_recursive_recovery(self):
        # Inject the budgeter's rejection boundary and count actual SDK calls;
        # no larger model or recursive summary recovery may bypass this gate.
        from lineage_agent.agent import compaction
        original = compaction.check_context_budget
        observed = []
        def reject_summary(request, risks, settings, **kwargs):
            budget = original(request, risks, settings, **kwargs)
            if is_summary_request(request):
                observed.append(deepcopy(request))
                budget.update(decision="exceeded", required_tokens=settings.context_budget_tokens + 1,
                              error_code="context_budget_exceeded")
            return budget
        with patch.object(compaction, "check_context_budget", side_effect=reject_summary):
            state, client = await self.standalone(count=6)
        self.assertEqual(len(observed), 1)
        self.assertEqual(state["summary_calls"], 0)
        self.assertFalse(client.summary_requests)
        self.assertEqual(state["compaction"]["reason"], "summary_budget_exceeded")
        self.assertEqual(len(self.executed), 6)

    async def test_multiple_tool_calls_and_legal_failures_remain_in_whole_groups(self):
        self.settings = replace(self.settings, context_budget_tokens=34000)
        replies = [response(call(f"g{index}a", arguments=json.dumps({"key": f"g{index}a"})),
                            call(f"g{index}b", arguments=json.dumps({"key": f"g{index}b"})))
                   for index in range(6)] + [response(message("completed"))]
        original = self.tool
        async def tool(name, arguments):
            if arguments["key"] == "g0b":
                self.executed.append("g0b")
                raise ToolExecutionError()
            return await original(name, arguments)
        self.tool = tool
        state, client = await self.standalone(replies=replies)
        self.assert_complete(state)
        self.assertGreater(state["summary_calls"], 0)
        self.assertEqual(state["tool_outcomes"]["failed"], 1)
        self.assertEqual(state["tool_outcomes"]["succeeded"], 11)
        for request in client.business_requests:
            outputs = {item["call_id"] for item in request["input"] if item.get("type") == "function_call_output"}
            for index in range(6):
                self.assertEqual(f"g{index}a" in outputs, f"g{index}b" in outputs)
        originals = client.summary_requests[0]["input"]
        transcript = ET.fromstring("<root>" + originals[0]["content"] + "</root>")
        outputs = {item.attrib["call_id"] for item in transcript.iter("tool_result")}
        self.assertIn("g0a", outputs)
        self.assertIn("g0b", outputs)
        self.assertIn("failed", originals[0]["content"])
        self.assertIn("g0b", originals[0]["content"])

    async def test_unclosed_multi_tool_group_is_excluded_from_confirmed_prefix_selection(self):
        self.settings = replace(self.settings, context_budget_tokens=100000)
        original = self.tool
        async def tool(name, arguments):
            if arguments["key"] == "interrupt":
                raise asyncio.CancelledError()
            return await original(name, arguments)
        with SessionManager() as manager:
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=tool)
            replies = self.replies(6)[:-1] + [response(
                call("unfinished_a", arguments='{"key":"interrupt"}'),
                call("unfinished_b", arguments='{"key":"must-not-run"}'))]
            client = _Client(replies)
            with patch.object(runtime, "create_client", return_value=client), self.assertRaises((asyncio.CancelledError, NodeCancelledError)):
                await manager.run_turn(sid, "current task interrupted within a tool group")
            records = ContextRecords(manager._checkpointer, sid)
            saved = await records.snapshot(records.confirmed())
            self.assertFalse(saved["active_context"]["groups"][-1]["closed"])
            self.assertEqual(len(saved["active_context"]["groups"]), 7)
            selected, reason = await records.select(saved, self.settings)
            self.assertIsNone(reason)
            self.assertEqual(len(selected["removed_groups"]), 3)
            incomplete = saved["active_context"]["groups"][-1]
            self.assertNotIn(incomplete["group_id"], selected["removed_groups"])
            self.assertFalse(set(range(incomplete["start"], incomplete["end"])) & set(selected["removed_indices"]))
            self.assertNotIn("must-not-run", self.executed)
            self.assertFalse(client.summary_requests)

    async def test_previous_h_is_reused_without_call_or_target_allocation(self):
        from lineage_agent.agent.compaction import _message
        with SessionManager() as manager:
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=self.tool)
            seed = _Client([response(message("historical source " + "h" * 4200))])
            with patch.object(runtime, "create_client", return_value=seed):
                await manager.run_turn(sid, "previous task")
            client = _Client(self.replies(10))
            with patch.object(runtime, "create_client", return_value=client):
                turn = await manager.run_turn(sid, "current task")
            self.assert_complete(turn.state)
            lanes = [summary_request_lane(request) for request in client.summary_requests]
            self.assertEqual(lanes.count("H"), 1)
            self.assertGreaterEqual(lanes.count("R"), 2)
            candidate = turn.state["compaction"]
            history, current = candidate["lanes"]["H"], candidate["lanes"]["R"]
            self.assertEqual(history["status"], "reused")
            self.assertFalse(history["model_called"])
            self.assertFalse(history["source_refs"])
            self.assertGreater(json_bytes(_message("H", history, history["body"])), 0)
            self.assertNotIn("target_bytes", current)
            self.assertNotIn("recovery_budget", candidate)
            for request in client.summary_requests:
                if summary_request_lane(request) == "R":
                    self.assertNotIn("historical source", json.dumps(request["input"]))

    async def test_actual_summary_request_larger_than_window_is_not_sent(self):
        # A fixed lane output reserve alone can make the native request exceed W;
        # business requests keep their smaller output cap.
        self.settings = replace(self.settings, summary_history_max_output_tokens=24000)
        with SessionManager() as manager:
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=self.tool)
            for index in range(4):
                # Keep genuine quoted source text; no old wrapping overhead is
                # needed to exercise the real summary preflight boundary.
                client = _Client([response(message('\\"' * 3000 if index == 0 else "short historical response"))])
                with patch.object(runtime, "create_client", return_value=client):
                    turn = await manager.run_turn(sid, f"old task {index}")
                self.assert_complete(turn.state)
            tool_defs = client.business_requests[0]["tools"]
            request = prepare_request(self.settings.model_name, turn.state["input"] + [{"role": "user", "content": ""}],
                                      tools=tool_defs, max_output_tokens=self.settings.max_output_tokens)
            base = check_context_budget(request, annotate_context_risks(request["input"], fallback_tokens=4096, run_id="test"), self.settings)["required_tokens"]
            prompt = "n" * (self.settings.context_budget_tokens - base + 128)
            client = _Client([response(message("must not send"))])
            with patch.object(runtime, "create_client", return_value=client):
                stopped = await manager.run_turn(sid, prompt)
            self.assertEqual(stopped.state["compaction"]["reason"], "summary_budget_exceeded")
            self.assertEqual(stopped.state["summary_calls"], 0)
            self.assertFalse(client.requests)
            self.assertGreater(stopped.state["compaction"]["summary_budget"]["required_tokens"], self.settings.context_budget_tokens)


if __name__ == "__main__":
    unittest.main()
