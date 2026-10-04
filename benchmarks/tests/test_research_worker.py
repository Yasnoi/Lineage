"""Deterministic transport and real lifecycle checks; never real Research scoring."""

import asyncio
from copy import deepcopy
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import APIStatusError

from benchmarks.core.budgets import BudgetRecordError, RequestBudget, WorkerAbort, read_ledger
from benchmarks.core.evidence import ObservedTools, Observer, digest
from benchmarks.core.executors import agent_worker as worker
from lineage_agent.models.summarization import is_summary_request, unpack_summary_material
from lineage_agent.recording import tracing
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

    def test_invalid_profile_is_blocked_without_endpoint(self):
        self.spec["profile"]["model_name"] = ""
        result, code, captured = self.execute(lambda _: self.fail("must not call endpoint"))
        self.assertEqual(code, 2)
        self.assertEqual(result["execution_status"], "blocked")
        self.assertTrue(result["global_stop"])
        self.assertFalse(captured)

    def test_missing_credentials_preserves_plan_and_mechanism_gap(self):
        self.spec["case_id"] = "read-limit-fixture"
        self.spec["mechanism"] = {"kind": "read_limit", "path": "code/tests/collection-service.test.ts"}
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
        with patch("benchmarks.core.budgets.time.monotonic", return_value=100):
            budget = RequestBudget(self.root / "ledger", "run", "case", remaining_seconds=2)
        with patch("benchmarks.core.budgets.time.monotonic", return_value=102):
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
        with patch("benchmarks.core.evidence.append_durable", side_effect=WorkerAbort("record_write_failed")):
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
        for turn in result["turns"]:
            snapshot = turn["tracing"]["traces"][0]
            self.assertEqual(snapshot["agent_run_id"], turn["run_id"]["value"])
            self.assertEqual(snapshot["upload_state"], "disabled")
            self.assertEqual(snapshot["counts"]["generated"], 0)

    def test_inherited_opaque_protocol_keeps_schema_metrics_and_worker_running(self):
        self.spec["prompts"] = ["first-round", "follow-up"]
        def respond(_):
            result = message("saved-response")
            result["output"].insert(0, {"type": "reasoning", "id": "reasoning",
                "encrypted_content": "opaque-synthetic", "summary": [], "content": None})
            result["output"][-1]["phase"] = "final_answer"
            return result
        result, code, captured = self.execute(respond)
        self.assertEqual(code, 0)
        self.assertEqual(len(captured), 2)
        self.assertEqual([turn["execution_status"] for turn in result["turns"]], ["completed", "completed"])
        second = result["turns"][1]["metrics"]
        self.assertGreater(second["history_tool_fixed_budget_overhead"]["value"], 0)

    def test_terminal_recall_is_recorded_when_next_sdk_request_is_budget_refused(self):
        (self.material / "large.txt").write_text("evidence-" * 2000, encoding="utf-8")
        self.spec["profile"]["context_budget_tokens"] = 35000
        sent = 0
        def respond(kwargs):
            nonlocal sent
            sent += 1
            if sent == 1:
                return call("large.txt")
            self.assertEqual(sent, 2, "The next business request must fail local budget precheck.")
            envelope = json.loads(next(item["output"] for item in kwargs["input"]
                if item.get("type") == "function_call_output"))
            reference = envelope["observation"]["artifact_ref"]
            return {"status": "completed", "output": [{"type": "function_call", "status": "completed",
                "call_id": "recall", "name": "artifact_read", "arguments": json.dumps(
                    {"artifact_ref": reference, "start": 0, "end": None})}]}
        result, code, captured = self.execute(respond)
        self.assertEqual(code, 0)
        self.assertEqual(len(captured), 2)
        self.assertEqual(result["agent_outcome"]["value"], "context_budget_exceeded")
        self.assertEqual(result["turns"][0]["metrics"]["recall_calls"]["value"]["artifact_read"], 1)
        saved = json.loads((self.output / "turn-01-compact-evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["recalls"]["artifact_read"][0]["evidence_source"], "final_state_input")

    def test_production_trace_binding_and_disabled_override(self):
        uploaded = []
        class Upload:
            def create_run(self, **payload):
                uploaded.append(deepcopy(payload))
            def close(self, **kwargs):
                pass
        self.spec["tracing"] = {"enabled": True}
        self.spec["prompts"] = ["first", "follow up"]
        with patch.dict("os.environ", {"LANGSMITH_API_KEY": "trace-secret-marker",
                                      "LANGSMITH_PROJECT": "offline-benchmark",
                                      "LANGSMITH_ENDPOINT": "https://trace.invalid"}), \
                patch.object(tracing, "_create_client", return_value=Upload()):
            result, code, captured = self.execute(lambda _: message("answer"))
        self.assertEqual(code, 0, result["issues"])
        self.assertEqual(len(uploaded), 4)
        observations = [json.loads(line) for line in (self.output / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
        for index, turn in enumerate(result["turns"], 1):
            snapshot = turn["tracing"]["traces"][0]
            self.assertEqual(snapshot["upload_state"], "submission_completed")
            self.assertEqual(snapshot["binding"]["turn"], index)
            admitted = next(event for event in observations if event["turn"] == index and event["event"] == "request_started")
            self.assertEqual(admitted["correlation"]["agent_run_id"], turn["run_id"]["value"])
            model = next(p for p in uploaded if p["id"] == admitted["correlation"]["span_id"])
            self.assertEqual(model["extra"]["metadata"]["request_id"], admitted["request_id"])
            self.assertEqual(model["extra"]["metadata"]["sample_run_id"], self.spec["run_id"])
        self.assertNotIn("benchmark_batch_id", json.dumps(captured))
        self.assertNotIn("trace-secret-marker", json.dumps(uploaded))
        other = self.root / "disabled"
        other.mkdir()
        spec = {**self.spec, "output_dir": str(other), "tracing": {"enabled": False}}
        with patch.dict("os.environ", {"API_KEY": "model-secret", "LANGSMITH_TRACING": "true",
                                      "LANGSMITH_API_KEY": "trace-secret-marker"}), \
                patch.object(tracing, "_create_client") as trace_client, \
                patch.object(worker.runtime, "create_client", return_value=Client(lambda _: message(), [])):
            disabled, code = asyncio.run(worker.execute_sample(spec, "disabled-spec"))
        trace_client.assert_not_called()
        self.assertEqual(code, 0)
        self.assertEqual(disabled["turns"][0]["tracing"]["traces"][0]["upload_state"], "disabled")

    def test_trace_snapshot_write_failure_stops_sample(self):
        original = worker.atomic_write_json
        def write(path, value):
            if path.name.endswith("-tracing.json"):
                raise OSError("injected local trace recording failure")
            return original(path, value)
        with patch.object(worker, "atomic_write_json", side_effect=write):
            result, code, captured = self.execute(lambda _: message())
        self.assertEqual(code, 2)
        self.assertTrue(result["global_stop"])
        self.assertEqual(len(captured), 1)
        self.assertFalse((self.output / "turn-01-tracing.json").exists())

    def test_tools_binding_preserves_budget_and_actual_read_limit(self):
        target = self.material / "code/tests/collection-service.test.ts"
        target.parent.mkdir(parents=True)
        target.write_text("x" * 10000, encoding="utf-8")
        self.spec["case_id"] = "read-limit-fixture"
        self.spec["mechanism"] = {"kind": "read_limit", "path": "code/tests/collection-service.test.ts"}
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
        self._check_compaction_sample(False)

    def test_second_compaction_pass_uses_confirmed_candidate_chain(self):
        self._check_compaction_sample(True)

    def test_current_compaction_mechanism_budget_and_pressure(self):
        self._check_compaction_sample(False, current_profile=True)

    def _check_compaction_sample(self, second_pass, current_profile=False):
        form = self.material / "code/miniprogram/domain/form.ts"
        form.parent.mkdir(parents=True)
        form.write_text("signature fact\n" + (" " * 2741 if current_profile else ""), encoding="utf-8")
        background = self.material / "evaluation-fixtures/background.txt"
        background.parent.mkdir()
        background_bytes = 6144 if current_profile else 12288
        background.write_text("b" * background_bytes, encoding="utf-8")
        self.spec["case_id"] = "compaction-fixture"
        self.spec["mechanism"] = {"kind": "compaction", "source_turn": 1, "pressure_turn": 5, "follow_up_turn": 6, "no_tools_from_turn": 2, "source_path": "code/miniprogram/domain/form.ts", "background_path": "evaluation-fixtures/background.txt", "background_bytes": 12288, "pressure_max_bytes": 32768}
        self.spec["prompts"] = ["read initial files", "ack only", "ack only", "ack only", "padding no facts: ", "report old facts without tools"]
        # Raw background is archived before entering context. Allow bounded
        # pressure to compensate for the shorter observation while retaining
        # the real compaction/source/version checks below.
        self.spec["profile"].update(
            context_budget_tokens=32768, max_output_tokens=4096,
            context_safety_margin_tokens=1024,
        )
        if second_pass:
            # Keep this small-window fixture's candidate-only retry request
            # sendable under its independently configured H output reservation.
            self.spec["profile"]["summary_history_max_output_tokens"] = 4096
        if current_profile:
            self.spec["profile"].update(context_budget_tokens=65536, max_output_tokens=16384,
                                        context_safety_margin_tokens=8192)
            self.spec["mechanism"].update(background_bytes=background_bytes, pressure_max_bytes=32768)
        business_count = 0
        summaries = []
        def respond(kwargs):
            nonlocal business_count
            if is_summary_request(kwargs):
                material = unpack_summary_material(kwargs)
                summaries.append(material)
                text = "x" * 20000 if second_pass and len(summaries) == 1 else "signature fact"
                return message(text)
            business_count += 1
            if business_count == 1:
                response = call("code/miniprogram/domain/form.ts", "form-call")
                response["output"] += call("evaluation-fixtures/background.txt", "bg-call")["output"]
                return response
            return message()
        original_coverage = worker.compaction_coverage
        with patch.object(worker, "compaction_coverage", wraps=original_coverage) as coverage_spy:
            result, code, captured = self.execute(respond)
        self.assertEqual(code, 0, result["issues"])
        self.assertEqual(len(result["turns"]), 6)
        self.assertTrue(all(c["status"] == "verified" for c in result["coverage"]), result["coverage"])
        self.assertEqual(result["verdict"], "pending_review")
        self.assertEqual(result["metrics"]["sdk_requests"]["value"], len(captured))
        self.assertEqual(result["metrics"]["summary_requests"]["value"], 2 if second_pass else 1)
        self.assertEqual(result["metrics"]["total_tokens"]["value"], len(captured) * 12)
        self.assertEqual((self.output / "turn-05-summary-H.txt").read_text(encoding="utf-8"), "signature fact")
        process = json.loads((self.output / "turn-05-process.json").read_text(encoding="utf-8"))
        self.assertTrue(process["conversation_id"])
        self.assertTrue(process["record_files"])
        pressure = result["turns"][4]["pressure"]
        self.assertEqual(pressure["formula_version"], "actual-inherited-request-v4")
        self.assertEqual(pressure["decision"], "exceeded")
        self.assertGreater(pressure["required_tokens"], pressure["limit_tokens"])
        self.assertLessEqual(pressure["preceding_required_tokens"], pressure["limit_tokens"])
        self.assertEqual(result["turns"][4]["metrics"]["summary_lanes"]["value"]["R"], 0)
        if current_profile:
            self.assertEqual(pressure["max_padding_bytes"], 32768)
            self.assertFalse(pressure["capped"])
        if second_pass:
            self.assertEqual(summaries[1]["candidate_input"]["body"], "x" * 20000)
            self.assertNotIn("previous_summary", summaries[1])
            self.assertTrue(all("payload" not in item for item in summaries[1]["sources"]))
            args = list(coverage_spy.call_args.args)
            original_record = deepcopy(args[-1])
            for field, value in (("summary_request_sha256", "foreign"), ("candidate_input", {"summary_id": "wrong", "checkpoint_id": "wrong"}), ("body", "foreign")):
                with self.subTest(corrupted_binding=field):
                    args[-1] = deepcopy(original_record)
                    args[-1]["summary_results"][1]["compaction"]["lanes"]["H"][field] = value
                    coverage = original_coverage(*args)
                    self.assertEqual(next(item for item in coverage if item["name"] == "summary_saved_and_activated")["status"], "gap")

    def test_retired_profile_fields_are_not_silently_filtered(self):
        for name in ("context_trigger_percent", "context_target_percent", "summary_max_output_tokens", "summary_max_bytes",
                     "summary_target_percent", "context_recovery_margin_percent", "context_keep_recent_turns"):
            with self.subTest(name=name):
                spec = deepcopy(self.spec)
                spec["profile"][name] = 1
                with self.assertRaisesRegex(ValueError, "SUMMARY_HISTORY_MAX_OUTPUT_TOKENS"):
                    worker.make_settings(spec)

    def test_pressure_cap_reports_mechanism_gap_without_faking_rejection(self):
        previous = {
            "run_id": "old", "input": [{"role": "user", "content": "old question"}],
            "context_risks": [None], "history_end": 1, "record_sequence": 1,
            "active_context": {"turn_start": 1, "current_run_id": None, "groups": [],
                "turns": [{"run_id": "old", "item_count": 1, "group_count": 0, "record_end": 1}],
                "version_id": "old-version", "item_refs": ["history://record/old/0"], "summary_spans": []},
        }
        original = deepcopy(previous)
        settings = worker.make_settings(self.spec)
        from lineage_agent.models.requests import prepare_request
        request = prepare_request(settings.model_name, previous["input"], max_output_tokens=settings.max_output_tokens)
        prompt, details = worker.pressure_prompt("new", previous, [request], settings, 32)
        self.assertEqual(prompt, "new" + "p" * 32)
        self.assertTrue(details["capped"])
        self.assertEqual(details["decision"], "allow")
        self.assertEqual(previous, original)

    def test_settings_application_root_is_trusted_program_path(self):
        self.spec["application_root"] = str(self.root / "application")
        settings = worker.make_settings(self.spec)
        self.assertEqual(settings.application_root, (self.root / "application").resolve())
        self.assertEqual(settings.project_root, self.material)
        self.assertEqual(settings.log_dir, self.output / "agent-logs")

    def test_single_run_recalls_small_original_omitted_from_summary(self):
        from benchmarks.core.files import ROOT
        from benchmarks.suites.wechat_reporting.suite import definition
        task = next(task for task in definition()["units"] if task["case_id"] == "history-recall")
        for fixture in task["fixtures"]:
            path = self.material / fixture["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / fixture["text_path"]).read_bytes())
        self.spec.update(case_id=task["case_id"], prompts=task["turns"], mechanism=task["mechanism"])
        self.spec["profile"].update(task["settings_overrides"], max_model_calls=30)
        count = 0
        def native_call(name, arguments):
            return {"status": "completed", "output": [{"type": "function_call", "call_id": f"native-{count}",
                    "name": name, "arguments": json.dumps(arguments), "status": "completed"}],
                    "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}
        def respond(request):
            nonlocal count
            if is_summary_request(request):
                # Deliberately omit the target number/negation to prove original recall,
                # independently from the real model's semantic quality checks.
                return message("Early chain was read; use original history for details.")
            count += 1
            if count <= 16:
                return call(task["mechanism"]["chain_paths"][count - 1], f"chain-{count}")
            if count == 17:
                for item in request["input"]:
                    if item.get("role") == "assistant" and isinstance(item.get("content"), str):
                        scope = re.search(r"history://scope/[a-zA-Z0-9-]+", item["content"])
                        if scope:
                            return native_call("history_search", {"history_ref": scope.group(),
                                               "keywords": ["LARCH-17"], "cursor": None})
                self.fail("expected current prefix summary and discoverable scope")
            if count == 18:
                output = next(item for item in reversed(request["input"]) if item.get("type") == "function_call_output")
                result = json.loads(json.loads(output["output"])["observation"]["raw_result"])
                return native_call("history_read", {"source_ref": result["hits"][0]["source_ref"], "start": 0, "end": None})
            return message("Original recalled; semantic answer is not scored by this fixture.")
        result, code, captured = self.execute(respond)
        self.assertEqual(code, 0, result["issues"])
        self.assertEqual(result["agent_outcome"]["value"], "completed")
        self.assertTrue(all(check["status"] == "verified" for check in result["coverage"]), result["coverage"])
        self.assertGreater(result["metrics"]["summary_requests"]["value"], 0)
        self.assertEqual(result["turns"][0]["metrics"]["recall_calls"]["value"]["history_read"], 1)

    def test_compaction_missing_mechanism_and_reread_cannot_pass(self):
        observer = Observer(RequestBudget(self.root / "ledger", "run", "case"), self.output, self.material)
        result = worker.compaction_coverage(observer, [], None, None, ["a"] * 6, {"kind": "compaction", "source_turn": 1, "pressure_turn": 5, "follow_up_turn": 6, "no_tools_from_turn": 2, "source_path": "code/miniprogram/domain/form.ts", "background_path": "evaluation-fixtures/background.txt", "background_bytes": 12288})
        self.assertTrue(all(check["status"] == "gap" for check in result))
        states = [{"summary_calls": 0, "tool_calls": 0} for _ in range(6)]
        states[5]["tool_calls"] = 1
        result = worker.compaction_coverage(observer, states, None, None, ["a"] * 6, {"kind": "compaction", "source_turn": 1, "pressure_turn": 5, "follow_up_turn": 6, "no_tools_from_turn": 2, "source_path": "code/miniprogram/domain/form.ts", "background_path": "evaluation-fixtures/background.txt", "background_bytes": 12288})
        self.assertEqual(next(check for check in result if check["name"] == "no_recent_tool_reads")["status"], "gap")


if __name__ == "__main__":
    unittest.main()
