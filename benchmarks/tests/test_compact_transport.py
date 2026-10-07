"""Offline SDK serialization checks for dynamic compact requests."""

from copy import deepcopy
import json
import os
import unittest
from unittest.mock import patch

import httpx
from openai import APITimeoutError, AsyncOpenAI

from lineage_agent.config.settings import ModelSettings
from lineage_agent.models import responses
from lineage_agent.models.budget import annotate_context_risks, check_context_budget
from lineage_agent.models.requests import prepare_request
from lineage_agent.models.summarization import (
    SUMMARY_INSTRUCTIONS, is_summary_request, prepare_summary_request, summary_budget_settings,
    summary_request_lane, SummaryError,
)


SETTINGS = ModelSettings(base_url="https://compact.invalid/v1", model_name="offline-compact")
NATIVE_INPUT = [
    {"role": "user", "content": "整理历史 H；以下为已确认结束的原生历史资料。"},
    {"role": "user", "content": '保留路径 src/form.ts 与 "引用"。'},
]



def completed_response() -> dict:
    return {
        "id": "response-offline", "object": "response", "created_at": 0,
        "model": SETTINGS.model_name, "status": "completed", "error": None,
        "output": [{"id": "message-offline", "type": "message", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text", "text": "ack", "annotations": []}]}],
        "usage": {"input_tokens": 13, "output_tokens": 2, "total_tokens": 15},
    }


class CompactTransportTests(unittest.IsolatedAsyncioTestCase):
    def native_history(self, *, opaque=False):
        output = json.dumps({"observation": {"mode": "inline", "raw_result": json.dumps({
            "path": "source.txt", "content": ('原文 "quote"\n\\path\\ ' * 80)
            + 'compact_source_outputs_v1 {"source_index":999}'}, ensure_ascii=False)}}, ensure_ascii=False)
        items = deepcopy(NATIVE_INPUT)
        if opaque:
            items.append({"type": "reasoning", "id": "reasoning-original", "status": "completed",
                          "summary": [], "content": [{"type": "reasoning_text", "text": "observed thought"}],
                          "encrypted_content": "opaque-do-not-decode-密文"})
        for index in range(3):
            items.extend([
                {"type": "function_call", "id": f"native-call-{index}", "call_id": f"call-{index}",
                 "name": "read_text_file", "arguments": '{"path":"source.txt"}', "status": "completed"},
                {"type": "function_call_output", "id": f"native-result-{index}",
                 "call_id": f"call-{index}", "output": output},
            ])
        return items

    async def test_native_source_items_reach_http_without_wrapping_or_output_splitting(self):
        items = self.native_history()
        original = deepcopy(items)
        request = prepare_summary_request(SETTINGS, items, lane="R")
        self.assertEqual(request["input"], original)
        self.assertEqual(items, original)
        self.assertIsNot(request["input"], items)
        for actual, source in zip(request["input"], items, strict=True):
            self.assertIsNot(actual, source)
        self.assertEqual([item.get("type") for item in request["input"]],
                         [None, None, "function_call", "function_call_output"]
                         + ["function_call", "function_call_output"] * 2)
        observed = []
        def capture(sent):
            observed.append(json.loads(sent.content))
            return httpx.Response(200, json=completed_response())
        risks = annotate_context_risks(request["input"], fallback_tokens=SETTINGS.opaque_fallback_tokens, run_id="offline")
        budget = check_context_budget(request, risks, summary_budget_settings(SETTINGS, "R"))
        self.assertEqual(budget["decision"], "allow")
        client, _ = self.make_client(capture)
        async with client:
            await responses.send_prepared_response(client, request, purpose="summary")
        self.assertEqual(observed, [request])
        self.assertEqual(observed[0]["input"], original)
        self.assertEqual(budget["transparent_tokens"], len(json.dumps(
            observed[0], ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")))
        self.assertEqual(items, original)
        # Tool text may itself name the removed transport; it remains ordinary data.
        self.assertIn("compact_source_outputs_v1", observed[0]["input"][3]["output"])
        self.assertTrue(all("payload" not in item and "source_id" not in item for item in request["input"]))
        self.assertNotIn("compact_source_outputs_v1", request["instructions"])
        items[3]["output"] = "caller edit after preparation"
        self.assertEqual(request["input"], original)

    async def test_native_opaque_fields_preserve_provenance_and_exact_sent_body(self):
        items = self.native_history(opaque=True)
        request = prepare_summary_request(SETTINGS, items, lane="H")
        risks = annotate_context_risks(items, fallback_tokens=SETTINGS.opaque_fallback_tokens,
                                      run_id="original-run", model_call=7,
                                      usage={"input_tokens": 800, "output_tokens": 257, "total_tokens": 1057})
        original_risks = deepcopy(risks)
        budget = check_context_budget(request, risks, summary_budget_settings(SETTINGS, "H"))
        self.assertEqual(budget["decision"], "allow")
        self.assertEqual(budget["opaque_items"], 1)
        self.assertEqual(budget["opaque_risk_tokens"], 257)
        self.assertEqual(risks[2], {"opaque_count": 1, "risk_tokens": 257,
                         "basis": "response_output_usage", "source_run_id": "original-run", "source_model_call": 7})
        measured_projection = deepcopy(request)
        measured_projection["input"][2]["encrypted_content"] = None
        self.assertEqual(budget["transparent_tokens"], len(json.dumps(
            measured_projection, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")))
        observed = []
        def capture(sent):
            observed.append(json.loads(sent.content))
            return httpx.Response(200, json=completed_response())
        client, _ = self.make_client(capture)
        async with client:
            await responses.send_prepared_response(client, request, purpose="summary")
        self.assertEqual(observed, [request])
        self.assertEqual(observed[0]["input"], items)
        self.assertEqual(observed[0]["input"][2]["encrypted_content"], "opaque-do-not-decode-密文")
        self.assertEqual(risks, original_risks)
        missing_risk = [None] * len(items)
        self.assertEqual(check_context_budget(request, missing_risk, summary_budget_settings(SETTINGS, "H"))["decision"], "error")
        fallback = annotate_context_risks(items, fallback_tokens=SETTINGS.opaque_fallback_tokens,
                                          run_id="original-run", model_call=7)
        self.assertEqual(check_context_budget(request, fallback, summary_budget_settings(SETTINGS, "H"))["opaque_risk_tokens"],
                         SETTINGS.opaque_fallback_tokens)

    def test_native_builder_rejects_incomplete_or_unsupported_protocol_without_repair(self):
        complete = self.native_history()
        invalid = [complete[:-1], complete + [deepcopy(complete[3])], complete[3:],
                   [complete[0], complete[2], complete[2], complete[3]],
                   [{"role": "system", "content": "fixed instructions are not summary material"}],
                   [{"type": "vendor_private", "payload": "opaque"}],
                   [{"role": "user", "content": [{"type": "input_image", "image_url": "https://invalid.test"}]}]]
        for items in invalid:
            original = deepcopy(items)
            with self.subTest(items=items), self.assertRaises(SummaryError):
                prepare_summary_request(SETTINGS, items, lane="H")
            self.assertEqual(items, original)

    def make_client(self, handler):
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        def construct(**kwargs):
            self.assertEqual(kwargs["max_retries"], 0)
            self.assertEqual(SETTINGS.timeout_seconds, 300.0)
            self.assertEqual(kwargs["timeout"], 300.0)
            return AsyncOpenAI(http_client=http, **kwargs)

        # Exercise the production no-retry client configuration. Only a local
        # marker is supplied; every request terminates in MockTransport.
        with patch.dict(os.environ, {"API_KEY": "offline-compact-test-key"}), \
                patch.object(responses, "AsyncOpenAI", side_effect=construct):
            client = responses.create_client(SETTINGS)
        return client, http

    async def test_measured_summary_is_exact_http_body_and_does_not_pollute_business(self):
        bodies = []

        def capture(request):
            self.assertEqual(request.url.host, "compact.invalid")
            self.assertEqual(request.url.path, "/v1/responses")
            self.assertEqual(request.extensions["timeout"], {
                "connect": 300.0, "read": 300.0, "write": 300.0, "pool": 300.0,
            })
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=completed_response())

        client, http = self.make_client(capture)
        async with client:
            for lane, cap in (("H", 16384), ("R", 8192)):
                with self.subTest(lane=lane, cap=cap):
                    request = prepare_summary_request(SETTINGS, NATIVE_INPUT, lane=lane)
                    measured = deepcopy(request)
                    risks = annotate_context_risks(
                        request["input"], fallback_tokens=SETTINGS.opaque_fallback_tokens, run_id="offline-run",
                    )
                    budget = check_context_budget(request, risks, summary_budget_settings(SETTINGS, lane))
                    self.assertEqual(budget["decision"], "allow")
                    self.assertEqual(budget["output_reserve_tokens"], cap)
                    self.assertEqual(budget["transparent_tokens"], len(json.dumps(
                        measured, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
                    ).encode("utf-8")))
                    result = await responses.send_prepared_response(client, request, purpose="summary")
                    self.assertEqual(request, measured)
                    self.assertEqual(bodies[-1], measured)
                    self.assertEqual(bodies[-1]["max_output_tokens"], cap)
                    self.assertNotIn("UTF-8 bytes", bodies[-1]["instructions"])
                    self.assertEqual(summary_request_lane(bodies[-1]), lane)
                    self.assertEqual(bodies[-1]["reasoning"], {"effort": "none"})
                    self.assertEqual(bodies[-1]["tool_choice"], "none")
                    self.assertNotIn("tools", bodies[-1])
                    self.assertTrue(is_summary_request(bodies[-1]))
                    self.assertEqual(bodies[-1]["input"], NATIVE_INPUT)
                    self.assertEqual(result["usage"]["total_tokens"], 15)

            business = prepare_request(
                SETTINGS.model_name, [{"role": "user", "content": "Use the retained facts."}],
                instructions="Only answer from available evidence.", tool_choice="none",
                max_output_tokens=SETTINGS.max_output_tokens,
            )
            budget = check_context_budget(business, [None], SETTINGS)
            self.assertEqual(budget["decision"], "allow")
            await responses.send_prepared_response(client, business)
            self.assertEqual(bodies[-1], business)
            self.assertNotIn("reasoning", bodies[-1])
            self.assertFalse(is_summary_request(bodies[-1]))
        self.assertEqual(len(bodies), 3)
        self.assertTrue(http.is_closed)

    async def test_http_400_does_not_retry_or_fall_back_to_business(self):
        bodies = []

        def reject(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(400, json={"error": {
                "message": "unsupported summary option", "type": "invalid_request_error",
            }})

        client, http = self.make_client(reject)
        request = prepare_summary_request(SETTINGS, NATIVE_INPUT, lane="H")
        async with client:
            with self.assertRaises(responses.ModelError) as raised:
                await responses.send_prepared_response(client, request, purpose="summary")
        self.assertEqual(raised.exception.code, "http_error")
        self.assertEqual(bodies, [request])
        self.assertTrue(is_summary_request(bodies[0]))
        self.assertTrue(http.is_closed)

        # Exercise the SDK timeout path immediately, without a real 300-second
        # wait or network access, while preserving the configured HTTP deadline.
        bodies = []

        def timed_out(request):
            bodies.append(json.loads(request.content))
            self.assertEqual(request.extensions["timeout"]["read"], 300.0)
            raise httpx.ReadTimeout("offline injected timeout", request=request)

        client, http = self.make_client(timed_out)
        async with client:
            with self.assertRaises(responses.ModelError) as raised:
                await responses.send_prepared_response(client, request, purpose="summary")
        self.assertEqual(raised.exception.code, "model_timeout")
        self.assertIsInstance(raised.exception.__cause__, APITimeoutError)
        self.assertEqual(bodies, [request])
        self.assertTrue(http.is_closed)

    def test_summary_recognition_requires_canonical_dynamic_protocol(self):
        for lane, cap in (("H", 16384), ("R", 8192)):
            for attempt in (1, 2):
                request = prepare_summary_request(SETTINGS, NATIVE_INPUT, lane=lane, attempt_no=attempt)
                self.assertTrue(is_summary_request(request))
                self.assertIn('"format":"compact_native_v1"', request["instructions"])
                for change in (
                    {"max_output_tokens": cap + 1}, {"reasoning": {"effort": "low"}},
                    {"max_output_tokens": float(cap)}, {"instructions": SUMMARY_INSTRUCTIONS},
                    {"instructions": request["instructions"].replace("compact_native_v1", "unknown")},
                    {"tools": [{"type": "function", "name": "unexpected", "parameters": {}}]},
                ):
                    with self.subTest(lane=lane, attempt=attempt, change=change):
                        self.assertFalse(is_summary_request({**request, **change}))
                ordinary = prepare_request(
                    SETTINGS.model_name, [{"role": "user", "content": request["instructions"]}],
                    tool_choice="none", max_output_tokens=cap,
                )
                ordinary["reasoning"] = {"effort": "none"}
                self.assertFalse(is_summary_request(ordinary))


if __name__ == "__main__":
    unittest.main()
