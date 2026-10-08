"""One explicit effort contract across configuration, preflight and actual transport."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI

from benchmarks.tests.test_compact_transport import completed_response
from benchmarks.tests import test_prefix_compaction as prefix
from lineage_agent.config.settings import ConfigurationError, ModelSettings, load_settings
from lineage_agent.models.budget import check_context_budget
from lineage_agent.models.requests import prepare_request
from lineage_agent.models.responses import generate_response, send_prepared_response, ModelError
from lineage_agent.models.summarization import prepare_summary_request, summary_request_lane, summary_budget_settings, parse_summary


class EffortTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_effort_is_measured_and_sent_once_for_business_and_both_lanes(self):
        for effort in (None, "none", "low", "high", "max"):
            settings = ModelSettings(base_url="https://offline.invalid", model_name="offline", reasoning_effort=effort)
            sent = []
            def capture(request):
                sent.append(json.loads(request.content))
                return httpx.Response(200, json=completed_response())
            async with AsyncOpenAI(api_key="offline", base_url=settings.base_url, max_retries=0,
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(capture))) as client:
                items = [{"role": "user", "content": "source"}]
                for lane in ("H", "R"):
                    for attempt in (1, 2):
                        request = prepare_summary_request(settings, items, lane=lane, attempt_no=attempt)
                        before = deepcopy(request)
                        budget = check_context_budget(request, [None], summary_budget_settings(settings, lane))
                        self.assertEqual(budget["decision"], "allow")
                        self.assertEqual(budget["output_reserve_tokens"], request["max_output_tokens"])
                        await send_prepared_response(client, request, purpose="summary")
                        self.assertEqual(sent[-1], before)
                        self.assertEqual(summary_request_lane(sent[-1]), lane)
                        altered = {**request, "reasoning": {"effort": "low" if effort != "low" else "high"}}
                        self.assertIsNone(summary_request_lane(altered))
                await generate_response(client, settings.model_name, items, reasoning_effort=effort)
            for request in sent:
                if effort is None:
                    self.assertNotIn("reasoning", request)
                else:
                    self.assertEqual(request["reasoning"], {"effort": effort})

    async def test_reasoning_never_becomes_summary_text_and_empty_final_fails(self):
        for text in ("Visible summary.", ""):
            raw = completed_response()
            raw["output"][0]["content"][0]["text"] = text
            raw["output"].insert(0, {"type": "reasoning", "id": "r", "summary": [],
                                      "content": [{"type": "reasoning_text", "text": "PRIVATE THOUGHT"}]})
            async with AsyncOpenAI(api_key="offline", base_url="https://offline.invalid", max_retries=0,
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=raw)))) as client:
                request = prepare_request("offline", [{"role": "user", "content": "source"}], reasoning_effort="high")
                if text:
                    self.assertEqual(parse_summary(await send_prepared_response(client, request, purpose="summary")), text)
                else:
                    with self.assertRaises(ModelError):
                        await send_prepared_response(client, request, purpose="summary")

    def test_environment_and_direct_validation_distinguish_omitted_from_none(self):
        with TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            for effort in (None, "none", "low", "high", "max"):
                env.write_text("BASE_URL=https://offline.invalid\nMODEL_NAME=offline\n" +
                               ("" if effort is None else f"REASONING_EFFORT={effort}\n"), encoding="utf-8")
                with patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(load_settings(env).reasoning_effort, effort)
            for bad in ("", "HIGH", "medium", "xhigh", 1, True, []):
                with self.assertRaises(ConfigurationError):
                    ModelSettings(base_url="https://offline.invalid", model_name="offline", reasoning_effort=bad)
                with self.assertRaises(ConfigurationError):
                    prepare_request("offline", [], reasoning_effort=bad)

    async def test_runtime_uses_same_effort_after_prefix_compaction(self):
        fixture = prefix.PrefixCompactionTests()
        fixture.setUp()
        try:
            fixture.settings = replace(fixture.settings, reasoning_effort="high")
            state, client = await fixture.standalone()
            fixture.assert_complete(state)
            self.assertTrue(client.summary_requests)
            self.assertTrue(client.business_requests)
            self.assertTrue(all(request["reasoning"] == {"effort": "high"} for request in client.requests))
        finally:
            fixture.doCleanups()
