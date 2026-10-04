"""End-to-end deterministic contracts for current-Run and two-lane recovery."""

from copy import deepcopy
import asyncio
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
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


def _ids(value):
    result = []
    if isinstance(value, dict):
        if isinstance(value.get("source_id"), str):
            result.append(value["source_id"])
        for child in value.values():
            result.extend(_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_ids(child))
    return list(dict.fromkeys(result))


def _summary(request, text="Earlier tool observations are available through history recall."):
    material = json.loads(request["input"][0]["content"])
    refs = _ids(material)
    if not refs:
        refs = [ref for entries in material.get("candidate_input", {}).get("document", {}).values()
                for entry in entries for ref in entry["source_ids"]]
    if not refs:
        raise AssertionError("summary request has no immutable sources")
    document = {name: [] for name in ("requirements", "facts", "proposals", "corrections", "open_items")}
    document["facts"] = [{"text": text, "source_ids": refs[:1]}]
    return response(message(json.dumps(document)))


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
            log_dir=Path(self.tmp.name), context_budget_tokens=24000,
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
        state, client = await self.standalone()
        self.assert_complete(state)
        self.assertGreater(state["summary_calls"], 0)
        self.assertEqual(state["model_calls"], 11)
        self.assertEqual(self.executed, [f"c{index}" for index in range(10)])
        self.assertTrue(all(summary_request_lane(request) == "R" for request in client.summary_requests))
        self.assertIsNone(state["active_context"]["history_summary"])
        self.assertIsNotNone(state["active_context"]["run_summary"])
        original = client.business_requests[0]["input"][0]
        for request in client.business_requests:
            self.assertIn(original, request["input"])
        business_count = 0
        for request in client.requests:
            if not is_summary_request(request):
                business_count += 1
                continue
            material = json.loads(request["input"][0]["content"])
            source_calls = [(source["payload"].get("item") or {}).get("call_id")
                            for source in material.get("sources", []) if isinstance(source.get("payload"), dict)]
            self.assertTrue(source_calls)
            for index in range(business_count - 3, business_count):
                self.assertNotIn(f"c{index}", source_calls)
            self.assertEqual(material["current_request_readonly"], [original])
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
            self.assertIsNotNone(turn.state["active_context"]["run_summary"])
            history = json.loads(client.summary_requests[0]["input"][0]["content"])
            current = json.loads(client.summary_requests[1]["input"][0]["content"])
            self.assertTrue(all(old.state["run_id"] in ref for ref in _ids(history)))
            self.assertTrue(all(turn.state["run_id"] in ref for ref in _ids(current)))
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
            material = json.loads(historical["input"][0]["content"])
            outcomes = [source["payload"]["outcome"] for source in material["sources"] if source["kind"] == "outcome"]
            self.assertEqual(len(outcomes), 1)
            self.assertEqual(outcomes[0]["stop_reason"], "error")
            self.assertEqual(outcomes[0]["tool_outcomes"]["succeeded"], 0)

    async def test_three_group_protection_keeps_entire_boundary_old_turn(self):
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
            self.assertEqual([ref["run_id"] for ref in active_turns], [item.state["run_id"] for item in turns[1:]])
            self.assertIn({"role": "user", "content": "boundary entire user input"}, client.business_requests[0]["input"])
            preserved = [item["call_id"] for item in client.business_requests[0]["input"] if item.get("type") == "function_call_output"]
            self.assertEqual(preserved, ["old0", "old1"])

    async def test_model_searches_and_reads_detail_missing_from_r_without_tool_replay(self):
        observed = {}
        def ask_search(request):
            active = request["input"]
            self.assertFalse(any(item.get("type") == "function_call_output" and item["call_id"] == "c0" for item in active))
            summaries = [json.loads(item["content"]) for item in active
                         if item.get("role") == "assistant" and isinstance(item.get("content"), str)
                         and "derived_context_summary" in item["content"]]
            summary = next(item for item in summaries if item["lane"] == "R")
            self.assertNotIn("omitted-detail=37", json.dumps(summary["document"]))
            observed["history_ref"] = summary["history"]["history_ref"]
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
        first = json.loads(requests[0]["input"][0]["content"])
        second = json.loads(requests[1]["input"][0]["content"])
        self.assertTrue(any("payload" in item for item in first["sources"]))
        self.assertIn("candidate_input", second)
        self.assertFalse(any("payload" in item for item in second["sources"]))
        self.assertNotIn("omitted-detail=37", json.dumps(second))
        self.assertEqual(state["summary_calls"], 2)
        self.assertEqual(self.executed, [f"c{index}" for index in range(6)])

    async def test_invalid_summary_source_stops_after_one_call_and_keeps_original_view(self):
        def invalid(request):
            document = {name: [] for name in ("requirements", "facts", "proposals", "corrections", "open_items")}
            document["facts"] = [{"text": "unsupported claim", "source_ids": ["history://record/missing/0"]}]
            return response(message(json.dumps(document)))
        state, client = await self.standalone(count=6, summary=invalid)
        self.assertEqual(state["summary_calls"], 1)
        self.assertEqual(state["error_code"], "invalid_summary")
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
                material = json.loads(request["input"][0]["content"])
                self.assertIn("candidate_input", material)
                self.assertTrue(all("payload" not in source for source in material["sources"]))

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
        material = json.loads(client.summary_requests[0]["input"][0]["content"])
        originals = [source["payload"] for source in material["sources"]]
        outputs = {record["item"]["call_id"] for record in originals if record["item"].get("type") == "function_call_output"}
        self.assertIn("g0a", outputs)
        self.assertIn("g0b", outputs)
        self.assertTrue(any((record.get("execution") or {}).get("outcome") == "failed" for record in originals))

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

    async def test_previous_h_is_reused_without_call_and_deducted_before_r_allocation(self):
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
            empty = {name: [] for name in ("requirements", "facts", "proposals", "corrections", "open_items")}
            reused_cost = json_bytes(_message("H", history, history["document"])) - json_bytes(_message("H", history, empty))
            current_capacity = current["target_bytes"] - json_bytes(_message("R", current, empty))
            self.assertGreater(reused_cost, 0)
            self.assertEqual(reused_cost + current_capacity, candidate["recovery_budget"]["target_bytes"])

    async def test_actual_summary_request_larger_than_window_is_not_sent(self):
        with SessionManager() as manager:
            sid = manager.create_session(self.settings, tools=[TOOL], execute_tool=self.tool)
            for index in range(4):
                # Transparent text fits business requests but its nested JSON
                # projection needs substantially more escaping in a summary.
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
