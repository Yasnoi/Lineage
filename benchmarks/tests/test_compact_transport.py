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
)


SETTINGS = ModelSettings(base_url="https://compact.invalid/v1", model_name="offline-compact")
MATERIAL = {
    "previous_summary": None,
    "sources": [{"source_id": "source-1", "role": "user", "kind": "message",
                 "payload": {"role": "user", "content": "保留路径 src/form.ts 与 \\\"引用\\\"。"}}],
    "turn_outcomes": [], "opaque_omissions": [],
}


def completed_response() -> dict:
    return {
        "id": "response-offline", "object": "response", "created_at": 0,
        "model": SETTINGS.model_name, "status": "completed", "error": None,
        "output": [{"id": "message-offline", "type": "message", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text", "text": "ack", "annotations": []}]}],
        "usage": {"input_tokens": 13, "output_tokens": 2, "total_tokens": 15},
    }


class CompactTransportTests(unittest.IsolatedAsyncioTestCase):
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
            for target in (640, 1379):
                with self.subTest(target=target):
                    request = prepare_summary_request(SETTINGS, MATERIAL, target_tokens=target)
                    measured = deepcopy(request)
                    risks = annotate_context_risks(
                        request["input"], fallback_tokens=SETTINGS.opaque_fallback_tokens, run_id="offline-run",
                    )
                    budget = check_context_budget(request, risks, summary_budget_settings(SETTINGS, target))
                    self.assertEqual(budget["decision"], "allow")
                    self.assertEqual(budget["output_reserve_tokens"], target)
                    self.assertEqual(budget["transparent_tokens"], len(json.dumps(
                        measured, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
                    ).encode("utf-8")))
                    result = await responses.send_prepared_response(client, request, purpose="summary")
                    self.assertEqual(request, measured)
                    self.assertEqual(bodies[-1], measured)
                    self.assertEqual(bodies[-1]["max_output_tokens"], target)
                    self.assertEqual(bodies[-1]["reasoning"], {"effort": "none"})
                    self.assertEqual(bodies[-1]["tool_choice"], "none")
                    self.assertNotIn("tools", bodies[-1])
                    self.assertTrue(is_summary_request(bodies[-1]))
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
        request = prepare_summary_request(SETTINGS, MATERIAL, target_tokens=900)
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
        for target in (1, 900, 4000):
            request = prepare_summary_request(SETTINGS, MATERIAL, target_tokens=target)
            self.assertTrue(is_summary_request(request))
            for change in (
                {"max_output_tokens": target + 1}, {"reasoning": {"effort": "low"}},
                {"instructions": SUMMARY_INSTRUCTIONS},
                {"tools": [{"type": "function", "name": "unexpected", "parameters": {}}]},
            ):
                with self.subTest(target=target, change=change):
                    self.assertFalse(is_summary_request({**request, **change}))
            ordinary = prepare_request(
                SETTINGS.model_name, [{"role": "user", "content": request["instructions"]}],
                tool_choice="none", max_output_tokens=target,
            )
            ordinary["reasoning"] = {"effort": "none"}
            self.assertFalse(is_summary_request(ordinary))


if __name__ == "__main__":
    unittest.main()
