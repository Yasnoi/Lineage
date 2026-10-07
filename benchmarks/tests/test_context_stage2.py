"""Independent integration checks for current protection and final ownership."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from benchmarks.checks.support.runtime import message, response
from benchmarks.tests import test_prefix_compaction as fixtures
from lineage_agent.agent import compaction, graph, runtime
from lineage_agent.agent.context import ContextRecords
from lineage_agent.agent.sessions import SessionManager
from lineage_agent.models.summarization import summary_request_lane


class ContextStage2Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = fixtures.PrefixCompactionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tmp.cleanup)

    async def test_zero_through_three_current_groups_are_kept_and_all_old_turns_enter_h(self):
        fixture = self.fixture
        native_budget, native_select = graph.check_context_budget, ContextRecords.select
        for protected_count in range(4):
            with self.subTest(groups=protected_count), SessionManager() as manager:
                sid = manager.create_session(fixture.settings, tools=[fixtures.TOOL], execute_tool=fixture.tool)
                old = []
                for index in range(2):
                    seed = fixtures._Client([response(message(f"OLD_STAGE2_{index} " + "h" * 1000))])
                    with patch.object(runtime, "create_client", return_value=seed):
                        old.append(await manager.run_turn(sid, f"old task {index}"))
                    fixture.assert_complete(old[-1].state)
                selections = []

                async def selected(records, state, settings):
                    result = await native_select(records, state, settings)
                    selections.append((deepcopy(state), deepcopy(result)))
                    return result

                def pressure(request, risks, settings, **kwargs):
                    budget = native_budget(request, risks, settings, **kwargs)
                    items = request["input"]
                    current = any(item.get("content") == "CURRENT_STAGE2" for item in items)
                    old_visible = "OLD_STAGE2_0" in json.dumps(items)
                    groups = sum(item.get("type") == "function_call_output"
                                 and item.get("call_id", "").startswith("stage") for item in items)
                    if current and old_visible and groups == protected_count:
                        budget.update(decision="exceeded", required_tokens=settings.context_budget_tokens + 1)
                    return budget

                client = fixtures._Client(fixture.replies(protected_count, prefix="stage"))
                with patch.object(runtime, "create_client", return_value=client), \
                        patch.object(graph, "check_context_budget", side_effect=pressure), \
                        patch.object(ContextRecords, "select", new=selected):
                    current = await manager.run_turn(sid, "CURRENT_STAGE2")
                fixture.assert_complete(current.state)
                self.assertEqual(len(selections), 1)
                before, (selection, reason) = selections[0]
                self.assertIsNone(reason)
                self.assertEqual(len(before["active_context"]["groups"]), protected_count)
                self.assertEqual(selection["removed_groups"], [])
                self.assertEqual(selection["removed_turns"], 2)
                self.assertEqual(selection["lanes"]["R"]["status"], "empty")
                self.assertEqual({bound["run_id"] for bound in selection["lanes"]["H"]["scope"]["ranges"]},
                                 {turn.state["run_id"] for turn in old})
                self.assertEqual([summary_request_lane(request) for request in client.summary_requests], ["H"])
                resumed = client.business_requests[-1]["input"]
                self.assertEqual([item["call_id"] for item in resumed if item.get("type") == "function_call_output"],
                                 [f"stage{index}" for index in range(protected_count)])
                self.assertIn({"role": "user", "content": "CURRENT_STAGE2"}, resumed)

    async def test_completed_run_is_historical_before_next_turn_and_inherits_once(self):
        fixture = self.fixture
        with SessionManager() as manager:
            sid = manager.create_session(fixture.settings, tools=[fixtures.TOOL], execute_tool=fixture.tool)
            first_client = fixtures._Client(fixture.replies(10))
            with patch.object(runtime, "create_client", return_value=first_client):
                first = await manager.run_turn(sid, "current original 17; do not write files")
            fixture.assert_complete(first.state)
            active = first.state["active_context"]
            self.assertIsNone(active["current_run_id"])
            self.assertIsNone(active["run_summary"])
            self.assertEqual(active["groups"], [])
            self.assertEqual(active["initial_input_count"], 0)
            self.assertEqual(active["turn_start"], first.state["history_end"])
            self.assertEqual([turn["run_id"] for turn in active["turns"]], [first.state["run_id"]])
            self.assertTrue(any(span["lane"] == "R" for span in active["summary_spans"]))
            record = await manager.get_record(sid, first.state["run_id"])
            self.assertEqual(len([item for item in record["input"] if item.get("type") == "function_call_output"]), 10)
            second_client = fixtures._Client([response(message("next completed"))])
            with patch.object(runtime, "create_client", return_value=second_client):
                second = await manager.run_turn(sid, "next request")
            fixture.assert_complete(second.state)
            self.assertEqual(second_client.business_requests[0]["input"],
                             first.state["input"] + [{"role": "user", "content": "next request"}])
            self.assertEqual([turn["run_id"] for turn in second.state["active_context"]["turns"]],
                             [first.state["run_id"], second.state["run_id"]])
            self.assertEqual(second.state["active_context"]["summary_spans"], active["summary_spans"])

    async def test_allowed_candidate_without_positive_gain_stops_without_retry(self):
        native = compaction.check_context_budget

        def no_gain(request, risks, settings, **kwargs):
            budget = native(request, risks, settings, **kwargs)
            if "NO_GAIN_STAGE2" in json.dumps(request["input"]):
                # Exercise the defensive inconsistency branch; real allowed
                # candidates already use less than the exceeded original.
                budget.update(decision="allow", required_tokens=10**9)
            return budget

        with patch.object(compaction, "check_context_budget", side_effect=no_gain):
            state, client = await self.fixture.standalone(
                count=6, summary=lambda request: fixtures._summary(request, "NO_GAIN_STAGE2"))
        self.assertEqual(state["compaction"]["reason"], "summary_not_effective")
        self.assertEqual(state["compaction"]["attempt_no"], 1)
        self.assertEqual(len(client.summary_requests), 1)
        self.assertEqual(state["summary_calls"], 1)
        self.assertIsNone(state["active_context"]["run_summary"])
        self.assertEqual(len(self.fixture.executed), 6)

    async def test_r_has_only_current_sources_previous_r_and_current_request(self):
        fixture = self.fixture
        with SessionManager() as manager:
            sid = manager.create_session(fixture.settings, tools=[fixtures.TOOL], execute_tool=fixture.tool)
            seed = fixtures._Client([response(message("H_RAW_STAGE2 " + "h" * 4200))])
            with patch.object(runtime, "create_client", return_value=seed):
                old = await manager.run_turn(sid, "old task")
            fixture.assert_complete(old.state)

            def summarized(request):
                return fixtures._summary(request, "H_DERIVED_STAGE2" if summary_request_lane(request) == "H"
                                         else "R_STAGE2截至覆盖范围")

            client = fixtures._Client(fixture.replies(10), summarized)
            prompt = "CURRENT_REQUIRED_17; never modify files"
            with patch.object(runtime, "create_client", return_value=client):
                current = await manager.run_turn(sid, prompt)
            fixture.assert_complete(current.state)
            self.assertTrue(any(summary_request_lane(request) == "H" for request in client.summary_requests))
            requests = [request for request in client.summary_requests if summary_request_lane(request) == "R"]
            self.assertGreaterEqual(len(requests), 2)
            for request in requests:
                items = request["input"]
                self.assertNotIn("H_RAW_STAGE2", json.dumps(items))
                self.assertNotIn("H_DERIVED_STAGE2", json.dumps(items))
                self.assertIn({"role": "user", "content": prompt}, items)
                self.assertNotIn(old.state["run_id"], items[0]["content"])
                self.assertTrue(all(item.get("content") == "R_STAGE2截至覆盖范围"
                                    for item in items if item.get("role") == "assistant"))



if __name__ == "__main__":
    unittest.main()
