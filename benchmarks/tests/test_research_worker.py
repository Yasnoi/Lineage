"""Deterministic transport and real lifecycle checks; never real Research scoring."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import APIStatusError

from benchmarks.research.budget import BudgetRecordError, RequestBudget, WorkerAbort, read_ledger
from benchmarks.research.observe import ObservedTools, Observer, digest
from benchmarks.research import worker
from lineage_agent.models.summarization import SUMMARY_INSTRUCTIONS
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.types import ToolArgumentError


def message(text="ack", usage=None):
    return {"status": "completed", "output": [{"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": text}]}], "usage": usage or {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}


def call(path, call_id="test-call"):
    return {"status": "completed", "output": [{"type": "function_call", "call_id": call_id,
             "name": "read_text_file", "arguments": json.dumps({"path": path}), "status": "completed"}],
             "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}


class Client:
    def __init__(self, respond, observed):
        self.respond, self.observed, self.closed = respond, observed, False
        self.responses = SimpleNamespace(create=self.create)

    async def create(self, **kwargs):
        self.observed.append(deepcopy(kwargs))
        return self.respond(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True


class ResearchWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="research-worker-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.material = self.root / "material"
        self.material.mkdir()
        self.output = self.root / "output"
        self.output.mkdir()
        (self.root / "requests.jsonl").touch()
        self.spec = {"batch_id": "test-batch", "run_id": "sample-id", "case_id": "wechat-capabilities", "definition_version": "test",
                     "prompts": ["test input"], "profile": {"model_name": "deepseek-flash", "base_url": "https://api.deepseek.com", "max_model_calls": 8},
                     "material_root": str(self.material), "output_dir": str(self.output), "ledger_path": str(self.root / "requests.jsonl"),
                     "remaining_seconds": 300, "max_requests": 400}

    def execute(self, respond, spec=None):
        captured, clients = [], []

        def factory(settings):
            client = Client(respond, captured)
            clients.append(client)
            return client

        with patch.dict("os.environ", {"API_KEY": "test-secret-not-for-artifacts", "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}), \
                patch.object(worker.runtime, "create_client", side_effect=factory):
            result, code = asyncio.run(worker.execute_sample(spec or self.spec, "spec-hash"))
        self.assertTrue(all(client.closed for client in clients))
        return result, code, captured

    def test_durable_budget_before_forward_and_no_retry(self):
        def respond(kwargs):
            ledger = read_ledger(Path(self.spec["ledger_path"]))
            self.assertEqual(len(ledger), 1)
            self.assertEqual(ledger[0]["purpose"], "business")
            return message("actual answer")
        result, code, captured = self.execute(respond)
        self.assertEqual(code, 0)
        self.assertEqual(result["verdict"], "pending_review")
        self.assertEqual(result["agent_outcome"]["value"], "completed")
        self.assertEqual(result["turns"][0]["checkpoint_id"]["availability"], "not_applicable")
        self.assertEqual(result["metrics"]["total_tokens"]["value"], 12)
        self.assertEqual(result["metrics"]["sdk_requests"]["value"], 1)
        self.assertEqual(len(captured), 1)
        self.assertEqual((self.output / "turn-01-answer.md").read_text(encoding="utf-8"), "actual answer")
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertNotIn("test-secret-not-for-artifacts", path.read_text(encoding="utf-8"))

    def test_transport_kwargs_exact_and_only_hashes_persisted(self):
        (self.root / "ledger").touch()
        budget = RequestBudget(self.root / "ledger", "run", "case")
        observer = Observer(budget, self.output, self.material)
        observer.turn = 1
        captured = []
        client = Client(lambda _: message(), captured)
        request = {"model": "test", "input": [{"role": "user", "content": "private request marker"}],
                   "instructions": "fixed", "max_output_tokens": 123, "tools": []}

        async def run():
            async with observer.client_factory(lambda _: client)(None) as observed:
                await observed.responses.create(**request)
        asyncio.run(run())
        self.assertEqual(captured, [request])
        self.assertEqual(observer.requests[1], [request])
        journal = (self.output / "observations.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("private request marker", journal)
        self.assertIn(digest(request), journal)

    def test_budget_refusal_never_calls_client(self):
        self.spec["max_requests"] = 1
        budget = RequestBudget(Path(self.spec["ledger_path"]), "previous", "other", max_requests=1, batch_id="test-batch")
        budget.reserve(1, "summary")
        result, code, captured = self.execute(lambda _: self.fail("must not call endpoint"))
        self.assertEqual(code, 2)
        self.assertTrue(result["global_stop"])
        self.assertEqual(result["global_stop_reason"], "request_budget_exhausted")
        self.assertEqual(captured, [])

    def test_invalid_ledger_never_reuses_allowance(self):
        path = Path(self.spec["ledger_path"])
        path.write_text('{"seq": 1}', encoding="utf-8")
        with self.assertRaises(BudgetRecordError):
            read_ledger(path)
        result, code, captured = self.execute(lambda _: self.fail("must not call endpoint"))
        self.assertEqual(code, 2)
        self.assertEqual(result["global_stop_reason"], "request_ledger_invalid")
        self.assertFalse(captured)

    def test_missing_and_foreign_ledgers_rejected(self):
        with self.assertRaises(BudgetRecordError):
            read_ledger(self.root / "does-not-exist")
        path = Path(self.spec["ledger_path"])
        budget = RequestBudget(path, "run", "case", batch_id="foreign-batch")
        budget.reserve(1, "business")
        with self.assertRaises(BudgetRecordError):
            read_ledger(path, batch_id="test-batch")
        with self.assertRaises(BudgetRecordError):
            read_ledger(path, run_ids={"unplanned-run"})
        with self.assertRaises(BudgetRecordError):
            read_ledger(self.root / "missing-ledger")
        result, code, captured = self.execute(lambda _: self.fail("must not call endpoint"))
        self.assertEqual(code, 2)
        self.assertEqual(result["global_stop_reason"], "request_ledger_invalid")
        self.assertFalse(captured)

    def test_profile_mismatch_is_blocked_without_endpoint(self):
        self.spec["profile"]["model_name"] = "not-approved"
        result, code, captured = self.execute(lambda _: self.fail("must not call endpoint"))
        self.assertEqual(code, 2)
        self.assertEqual(result["execution_status"], "blocked")
        self.assertTrue(result["global_stop"])
        self.assertFalse(captured)

    def test_missing_credentials_preserves_plan_and_mechanism_gap(self):
        self.spec["case_id"] = "wechat-read-limit"
        self.spec["prompts"] = ["first", "followup"]
        with patch.dict("os.environ", {"API_KEY": ""}):
            result, code = asyncio.run(worker.execute_sample(self.spec, "spec-hash"))
        self.assertEqual(code, 2)
        self.assertEqual(result["execution_status"], "blocked")
        self.assertEqual(len(result["turns"]), 2)
        self.assertEqual(result["coverage"][0]["status"], "gap")
        self.assertTrue(all(turn["answer_evidence"] == [] for turn in result["turns"]))

    def test_existing_artifacts_are_not_overwritten(self):
        path = self.output / "worker-result.json"
        path.write_text("existing", encoding="utf-8")
        result, code, captured = self.execute(lambda _: self.fail("must not call endpoint"))
        self.assertEqual(code, 2)
        self.assertEqual(path.read_text(encoding="utf-8"), "existing")
        self.assertFalse(captured)

    def test_deadline_prevents_next_reservation(self):
        with patch("benchmarks.research.budget.time.monotonic", return_value=100):
            budget = RequestBudget(self.root / "ledger", "run", "case", remaining_seconds=2)
        with patch("benchmarks.research.budget.time.monotonic", return_value=102):
            with self.assertRaises(WorkerAbort) as raised:
                budget.reserve(1, "business")
        self.assertEqual(raised.exception.code, "execution_time_exhausted")
        self.assertFalse((self.root / "ledger").exists())

    def test_truncated_previously_observed_ledger_cannot_reset_budget(self):
        path = Path(self.spec["ledger_path"])
        budget = RequestBudget(path, "run", "case")
        budget.reserve(1, "business")
        path.write_bytes(b"")
        with self.assertRaises(WorkerAbort) as raised:
            budget.reserve(1, "business")
        self.assertEqual(raised.exception.code, "request_ledger_changed")

    def test_global_endpoint_error_is_safe_and_stops(self):
        def respond(_):
            response = httpx.Response(401, request=httpx.Request("POST", "https://unused.invalid"))
            raise APIStatusError("credential-sensitive-server-message", response=response, body=None)
        result, code, captured = self.execute(respond)
        self.assertEqual(code, 2)
        self.assertEqual(result["global_stop_reason"], "credentials_rejected")
        self.assertEqual(result["turns"][0]["agent_outcome"]["value"], "error")
        self.assertEqual(len(captured), 1)
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertNotIn("credential-sensitive-server-message", path.read_text(encoding="utf-8"))

    def test_incomplete_model_response_keeps_agent_fact_and_stops_only_sample(self):
        self.spec["prompts"] = ["first", "unexecuted"]
        response = message("partial")
        response["status"] = "incomplete"
        result, code, captured = self.execute(lambda _: response)
        self.assertEqual(code, 0)
        self.assertFalse(result["global_stop"])
        self.assertEqual(result["agent_outcome"]["value"], "error")
        self.assertEqual(result["turns"][1]["execution_status"], "not_run")
        self.assertEqual(len(captured), 1)

    def test_journal_failure_before_transport_stops(self):
        with patch("benchmarks.research.observe.append_durable", side_effect=WorkerAbort("record_write_failed")):
            result, code, captured = self.execute(lambda _: self.fail("must not call endpoint"))
        self.assertEqual(code, 2)
        self.assertTrue(result["global_stop"])
        self.assertFalse(captured)
        self.assertEqual(len(read_ledger(Path(self.spec["ledger_path"]))), 1)

    def test_real_session_history_and_per_sample_isolation(self):
        self.spec["prompts"] = ["first-round", "follow-up"]
        result, code, captured = self.execute(lambda _: message("saved-response"))
        self.assertEqual(code, 0)
        first, second = result["turns"]
        self.assertTrue(first["checkpoint_complete"] and second["checkpoint_complete"])
        self.assertEqual(first["session_id"], second["session_id"])
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertIn("first-round", json.dumps(captured[1]))
        self.assertIn("saved-response", json.dumps(captured[1]))
        separate = self.root / "separate"
        separate.mkdir()
        spec = {**self.spec, "output_dir": str(separate), "run_id": "separate-sample"}
        other, _, other_captured = self.execute(lambda _: message(), spec)
        self.assertNotEqual(other["turns"][0]["session_id"], first["session_id"])
        self.assertNotIn("saved-response", json.dumps(other_captured[0]))

    def test_tools_binding_preserves_budget_and_actual_read_limit(self):
        target = self.material / "code/tests/collection-service.test.ts"
        target.parent.mkdir(parents=True)
        target.write_text("x" * 10000, encoding="utf-8")
        self.spec["case_id"] = "wechat-read-limit"
        self.spec["profile"]["text_max_bytes"] = 4096
        self.spec["prompts"] = ["read target", "limitations"]
        count = 0
        def respond(_):
            nonlocal count
            count += 1
            return call("code/tests/collection-service.test.ts") if count == 1 else message("could not read")
        result, code, captured = self.execute(respond)
        self.assertEqual(code, 0)
        self.assertEqual(result["coverage"][0]["status"], "verified")
        self.assertIn("file_too_large", json.dumps(captured[1]))
        observer = Observer(RequestBudget(self.root / "other-ledger", "run", "case"), self.output, self.material)
        wrapped = ObservedTools(create_filesystem_executor(self.material, max_total_bytes=12000), observer)
        bound = wrapped.bind_run()
        observer.turn = 1
        asyncio.run(bound("read_text_file", {"path": "code/tests/collection-service.test.ts"}))
        self.assertEqual(bound.read_budget["used_bytes"], 10000)
        self.assertEqual(wrapped.bind_run().read_budget["used_bytes"], 0)

    def test_agent_quota_terminal_is_not_framework_failure(self):
        target = self.material / "a.txt"
        target.write_text("tiny", encoding="utf-8")
        self.spec["profile"]["max_model_calls"] = 1
        self.spec["prompts"] = ["read", "must not run"]
        result, code, _ = self.execute(lambda _: call("a.txt"))
        self.assertEqual(code, 0)
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(result["agent_outcome"]["value"], "max_model_calls")
        self.assertEqual(result["turns"][1]["execution_status"], "not_run")

    def test_missing_usage_stays_unknown(self):
        response = message()
        response["usage"] = {"input_tokens": 10}
        result, code, _ = self.execute(lambda _: response)
        self.assertEqual(code, 0)
        self.assertEqual(result["metrics"]["input_tokens"]["value"], 10)
        self.assertEqual(result["metrics"]["output_tokens"]["availability"], "unknown")
        self.assertEqual(result["metrics"]["output_tokens_missing_requests"]["value"], 1)

    def test_save_failure_retains_previous_partial(self):
        original_write = worker.atomic_write_json
        def write(path, value):
            if path.name == "worker-result.json" and value.get("final"):
                raise OSError("injected save failure")
            return original_write(path, value)
        with patch.object(worker, "atomic_write_json", side_effect=write):
            result, code, _ = self.execute(lambda _: message())
        self.assertEqual(code, 2)
        saved = json.loads((self.output / "worker-result.json").read_text(encoding="utf-8"))
        self.assertFalse(saved["final"])
        self.assertEqual(saved["turns"][0]["execution_status"], "completed")

    def test_compaction_actual_requests_and_versions_without_semantic_score(self):
        form = self.material / "code/miniprogram/domain/form.ts"
        form.parent.mkdir(parents=True)
        form.write_text("signature fact\n", encoding="utf-8")
        background = self.material / "evaluation-fixtures/background.txt"
        background.parent.mkdir()
        background.write_text("b" * 12288, encoding="utf-8")
        self.spec["case_id"] = "wechat-compaction"
        self.spec["prompts"] = ["read initial files", "ack only", "ack only", "padding no facts: ", "report old facts without tools"]
        self.spec["profile"].update(summary_max_output_tokens=8192, summary_max_bytes=4096)
        business_count = 0
        def respond(kwargs):
            nonlocal business_count
            if kwargs.get("instructions") == SUMMARY_INSTRUCTIONS:
                material = json.loads(kwargs["input"][0]["content"])
                document = {key: [] for key in ("requirements", "facts", "proposals", "corrections", "open_items")}
                document["facts"] = [{"text": "signature fact", "source_ids": [material["sources"][0]["source_id"]]}]
                return message(json.dumps(document))
            business_count += 1
            if business_count == 1:
                response = call("code/miniprogram/domain/form.ts", "form-call")
                response["output"] += call("evaluation-fixtures/background.txt", "bg-call")["output"]
                return response
            return message()
        result, code, captured = self.execute(respond)
        self.assertEqual(code, 0, result["issues"])
        self.assertEqual(len(result["turns"]), 5)
        self.assertTrue(all(c["status"] == "verified" for c in result["coverage"]), result["coverage"])
        self.assertEqual(result["verdict"], "pending_review")
        self.assertEqual(result["metrics"]["sdk_requests"]["value"], len(captured))
        self.assertEqual(result["metrics"]["summary_requests"]["value"], 1)
        self.assertEqual(result["metrics"]["total_tokens"]["value"], len(captured) * 12)

    def test_compaction_missing_mechanism_and_reread_cannot_pass(self):
        observer = Observer(RequestBudget(self.root / "ledger", "run", "case"), self.output, self.material)
        result = worker.compaction_coverage(observer, [], None, None, ["a"] * 5)
        self.assertTrue(all(check["status"] == "gap" for check in result))
        states = [{"summary_calls": 0, "tool_calls": 0} for _ in range(5)]
        states[4]["tool_calls"] = 1
        result = worker.compaction_coverage(observer, states, None, None, ["a"] * 5)
        self.assertEqual(next(check for check in result if check["name"] == "no_recent_tool_reads")["status"], "gap")


if __name__ == "__main__":
    unittest.main()
