"""Four-arm effort/outline experiment with frozen native input and one allowance."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI

from benchmarks.core import summary_prompt_comparison as study
from benchmarks.tests.test_summary_prompt_comparison import manifest, completed


def frozen():
    value = manifest()
    value.update(design=study.EFFORT_DESIGN, limits=dict(study.EFFORT_LIMITS))
    case = value["cases"][0]
    case["design"] = study.EFFORT_DESIGN
    case["instructions"] = {key: case["instructions"]["A" if key.startswith("P0-") else "B"]
                            for key in study.EFFORT_VARIANTS}
    case["instruction_sha256"] = {key: study.digest(text) for key, text in case["instructions"].items()}
    return value


class EffortComparisonTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_authorized_fields_change_and_actual_http_matches_preflight(self):
        value = frozen()
        original = deepcopy(value)
        sent = []
        def capture(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json=completed())
        with TemporaryDirectory() as tmp:
            async with AsyncOpenAI(api_key="offline", base_url="https://offline.invalid", max_retries=0,
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(capture))) as client:
                result = await study.compare(value, Path(tmp), client)
            self.assertEqual(result["requests"], 8)
            for row, request in zip(result["results"], sent, strict=True):
                expected, budget = study.prepare(value["cases"][0], row["variant"])
                self.assertEqual(expected, request)
                self.assertEqual(row["budget"], budget)
                self.assertEqual(request["input"][1:], original["cases"][0]["request"]["input"][1:])
                self.assertEqual(row["budget"]["opaque_risk_tokens"], 123)
                restored = deepcopy(request)
                if row["variant"].startswith("P1-"):
                    note = restored["input"][0]["content"]
                    self.assertTrue(note.startswith("<summary_scope>\n") and note.endswith("\n</summary_scope>"))
                    restored["input"][0]["content"] = note[len("<summary_scope>\n"):-len("\n</summary_scope>")]
                restored["instructions"] = original["cases"][0]["request"]["instructions"]
                restored["reasoning"] = {"effort": "none"}
                self.assertEqual(restored, original["cases"][0]["request"])
            self.assertEqual(value, original)

    def test_schedule_has_twenty_requests_per_arm_and_alternates_exposure(self):
        value = frozen()
        value["cases"] = [deepcopy(value["cases"][0]) for _ in range(10)]
        for index, case in enumerate(value["cases"]):
            case["id"] = str(index)
        study.validate(value)
        rows = study.schedule(value)
        self.assertEqual(len(rows), 80)
        for variant in study.EFFORT_VARIANTS:
            self.assertEqual(sum(row["variant"] == variant for row in rows), 20)
        for index in range(0, 80, 4):
            self.assertEqual({row["variant"] for row in rows[index:index+4]}, set(study.EFFORT_VARIANTS))
        self.assertNotEqual(rows[0]["variant"], rows[40]["variant"])

    def test_tampered_sources_risks_effort_or_limits_are_rejected(self):
        for change in (
            lambda m: m["cases"][0]["request"]["input"][0].update(content="changed"),
            lambda m: m["cases"][0]["request"].update(reasoning={"effort": "max"}),
            lambda m: m["cases"][0]["risks"].pop(),
            lambda m: m["cases"][0]["instructions"].update({"P1-high": "changed"}),
            lambda m: m["limits"].update(sdk_requests=81),
            lambda m: m.update(design="unknown"),
        ):
            value = frozen()
            change(value)
            with self.assertRaises(ValueError):
                study.validate(value)

    async def test_cumulative_deadline_stops_without_resetting_the_allowance(self):
        # The batch deadline remains fixed even though each slot has a new request timeout.
        with TemporaryDirectory() as tmp:
            async with AsyncOpenAI(api_key="offline", base_url="https://offline.invalid", max_retries=0,
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: self.fail("deadline exceeded")))) as client:
                with patch.object(study.time, "monotonic", side_effect=[0, 0, 5000, 5000, 5000, 5000, 5000]):
                    result = await study.compare(frozen(), Path(tmp), client)
            self.assertEqual(result["requests"], 0)
            self.assertEqual(result["stop_reason"], "execution_time_exhausted")
