"""Comparison isolation, transport identity and durable failure boundaries."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI

from benchmarks.core.summary_prompt_comparison import (
    LIMITS, compare, digest, prepare, schedule, validate, without_instructions,
)
from benchmarks.core.budgets import WorkerAbort
from lineage_agent.models.summarization import summary_request_lane


def manifest():
    request = {"model": "offline", "input": [
        {"role": "user", "content": "已确认旧资料；以下均为来源。"},
        {"type": "reasoning", "id": "original", "summary": [], "encrypted_content": "opaque"},
        {"role": "assistant", "content": "旧摘要正文：137 已撤销，37 有效。"}],
        "instructions": "Frozen historical Instructions", "max_output_tokens": 8192,
        "reasoning": {"effort": "none"}, "tool_choice": "none", "stream": False,
        "store": False, "truncation": "disabled"}
    prompts = {"A": request["instructions"], "B": "Candidate handoff Instructions"}
    case = {"id": "fixed-R", "lane": "R", "request": request,
        "risks": [None, {"opaque_count": 1, "risk_tokens": 123, "basis": "response_output_usage",
                          "source_run_id": "old", "source_model_call": 2}, None],
        "input_sha256": digest(request["input"]), "fixed_request_sha256": digest(without_instructions(request)),
        "baseline_request_sha256": digest(request), "instructions": prompts,
        "instruction_sha256": {key: digest(value) for key, value in prompts.items()},
        "context_budget_tokens": 65536, "safety_margin_tokens": 8192}
    case["risks_sha256"] = digest(case["risks"])
    return {"limits": dict(LIMITS), "cases": [case], "settings": {"model_name": "offline"}}


def completed(text="# Handoff\nEffective limit 37; former 137 revoked."):
    return {"id": "test", "object": "response", "created_at": 0, "model": "offline",
        "status": "completed", "error": None, "output": [{"id": "msg", "type": "message",
        "role": "assistant", "status": "completed", "content": [{"type": "output_text",
        "text": text, "annotations": []}]}],
        "usage": {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70,
                  "input_tokens_details": {"cached_tokens": 10}}}


class ComparisonTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    def client(self, handler):
        return AsyncOpenAI(api_key="offline", base_url="https://test.invalid", max_retries=0,
                           http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def test_only_instructions_change_with_fixed_old_summary_and_opaque_risks(self):
        frozen = manifest()
        original = deepcopy(frozen)
        requests = []
        def capture(sent):
            requests.append(json.loads(sent.content))
            return httpx.Response(200, json=completed(f"New output {len(requests)}"))
        async with self.client(capture) as client:
            result = await compare(frozen, self.path, client)
        self.assertEqual([row["variant"] for row in result["results"]], ["A", "B", "B", "A"])
        self.assertEqual(result["requests"], 4)
        self.assertEqual(frozen, original)
        for request, row in zip(requests, result["results"], strict=True):
            self.assertEqual(without_instructions(request), without_instructions(frozen["cases"][0]["request"]))
            self.assertEqual(row["status"], "accepted")
            self.assertEqual(row["http_requests"], 1)
            self.assertEqual(row["budget"]["opaque_risk_tokens"], 123)
            self.assertEqual(row["raw_usage"]["input_tokens_details"]["cached_tokens"], 10)
            self.assertEqual(json.loads((self.path / f'{row["file_prefix"]}-http.json').read_text())["body"], request)
            # Explicit comparison purpose does not broaden production recognition.
            self.assertIsNone(summary_request_lane(request))

    async def test_budget_rejection_makes_no_sdk_call_or_ledger_entry(self):
        frozen = manifest()
        frozen["cases"][0]["context_budget_tokens"] = 100
        async with self.client(lambda _: self.fail("preflight rejection sent a request")) as client:
            result = await compare(frozen, self.path, client)
        self.assertEqual(result["requests"], 0)
        self.assertTrue(all(row["status"] == "preflight_rejected" for row in result["results"]))
        self.assertEqual((self.path / "sdk-ledger.jsonl").read_bytes(), b"")

    async def test_failed_calls_count_without_retry_and_authentication_stops(self):
        calls = []
        def fail(sent):
            calls.append(sent)
            return httpx.Response(500 if len(calls) == 1 else 401, json={"error": {"message": "offline"}})
        async with self.client(fail) as client:
            result = await compare(manifest(), self.path, client)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["stop_reason"], "authentication_failed")
        self.assertEqual([row["status"] for row in result["results"]], ["failed", "failed", "not_run", "not_run"])

    async def test_invalid_output_preserves_usage_and_is_not_replaced(self):
        async with self.client(lambda _: httpx.Response(200, json=completed(" "))) as client:
            result = await compare(manifest(), self.path, client)
        self.assertEqual(result["requests"], 4)
        self.assertTrue(all(row["status"] == "failed" and row["usage"]["total_tokens"] == 70
                            for row in result["results"]))
        self.assertEqual(len(list(self.path.glob("*-response.json"))), 4)

    async def test_evidence_write_failure_stops_before_call(self):
        async with self.client(lambda _: self.fail("write failure allowed transport")) as client:
            with patch("benchmarks.core.summary_prompt_comparison.atomic_write_json", side_effect=OSError):
                with self.assertRaises(WorkerAbort):
                    await compare(manifest(), self.path, client)
        self.assertEqual((self.path / "sdk-ledger.jsonl").read_bytes(), b"")

    async def test_http_evidence_failure_stops_before_network(self):
        from benchmarks.core.files import atomic_write_json
        def save(path, value):
            if path.name.endswith("-http.json"):
                raise OSError("injected")
            atomic_write_json(path, value)
        async with self.client(lambda _: self.fail("unrecorded request reached network")) as client:
            with patch("benchmarks.core.summary_prompt_comparison.atomic_write_json", side_effect=save):
                result = await compare(manifest(), self.path, client)
        self.assertEqual(result["stop_reason"], "http_evidence_failed")
        self.assertEqual(result["requests"], 1)  # reserved/started attempt, never refunded
        self.assertTrue(all(row["status"] == "not_run" for row in result["results"][1:]))

    async def test_existing_ledger_cannot_restart_allowance(self):
        (self.path / "sdk-ledger.jsonl").write_text("existing")
        async with self.client(lambda _: self.fail("existing batch replayed")) as client:
            with self.assertRaises(FileExistsError):
                await compare(manifest(), self.path, client)

    def test_tampering_native_input_risks_cap_or_frozen_prompt_is_rejected(self):
        for change in (
            lambda case: case["request"]["input"].reverse(),
            lambda case: case["risks"].pop(),
            lambda case: case["risks"][1].update(risk_tokens=1),
            lambda case: case["request"].update(max_output_tokens=256),
            lambda case: case["request"].update(tools=[{"type": "function"}]),
            lambda case: case["instructions"].update(A="marker-only forgery"),
        ):
            frozen = manifest()
            change(frozen["cases"][0])
            with self.assertRaises(ValueError):
                validate(frozen)

    def test_prompt_overhead_is_measured_without_changing_output_reserve(self):
        case = manifest()["cases"][0]
        request, before = prepare(case, "B")
        case["instructions"]["B"] += "繁體資料" * 100
        changed, after = prepare(case, "B")
        self.assertEqual(without_instructions(request), without_instructions(changed))
        self.assertEqual(after["output_reserve_tokens"], 8192)
        self.assertEqual(after["required_tokens"] - before["required_tokens"], len(("繁體資料" * 100).encode()))

    def test_full_schedule_is_bounded_and_balanced(self):
        frozen = manifest()
        frozen["cases"] *= 10
        rows = schedule(frozen)
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(row["variant"] == "A" for row in rows), 20)


if __name__ == "__main__":
    unittest.main()
