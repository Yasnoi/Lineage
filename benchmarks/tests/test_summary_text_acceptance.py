"""The acceptance sends only current rendered requests and never chains outputs."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import httpx
from openai import AsyncOpenAI

from benchmarks.core import summary_text_acceptance as text
from benchmarks.core.summary_prompt_comparison import compare, prepare, schedule, validate
from benchmarks.tests.test_summary_prompt_comparison import completed
from lineage_agent.models.summarization import summary_request_lane
from lineage_agent.models.summary_input import INPUT_FORMAT


def manifest():
    case = {"id": "fixed-R", "lane": "R", "design": text.DESIGN, "model_name": "offline",
            "source_input": [{"role": "user", "content": "Existing scope note"},
                             {"role": "assistant", "content": "Fixed previous summary: 37, not 137."},
                             {"type": "reasoning", "summary": [], "encrypted_content": "NEVER_SEND",
                              "content": [{"type": "reasoning_text", "text": "Historical plan"}]},
                             {"role": "assistant", "content": "A historical claim"}],
            "item_details": [{"kind": "scope"}, {"kind": "summary", "lane": "R", "run_id": "B"},
                             {"run_id": "B", "group_id": "B:4"}, {"run_id": "B", "group_id": "B:4"}],
            "context_budget_tokens": 65536, "safety_margin_tokens": 8192, "previous_summary_is_fixed": True}
    case["source_input_sha256"] = text.digest(case["source_input"])
    case["item_details_sha256"] = text.digest(case["item_details"])
    case["requests"] = {effort: text.build_request(case, effort) for effort in text.VARIANTS}
    case["request_sha256"] = {effort: text.digest(request) for effort, request in case["requests"].items()}
    return {"design": text.DESIGN, "input_format": INPUT_FORMAT, "limits": dict(text.LIMITS), "cases": [case],
            "settings": {"model_name": "offline", "context_budget_tokens": 65536, "context_safety_margin_tokens": 8192}}


class TextAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_version_only_two_efforts_and_fixed_inherited_summary(self):
        frozen = manifest()
        before = deepcopy(frozen)
        sent = []
        def capture(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json=completed(f"new summary {len(sent)}"))
        with TemporaryDirectory() as directory:
            async with AsyncOpenAI(api_key="offline", base_url="https://unused.invalid", max_retries=0,
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(capture))) as client:
                result = await compare(frozen, Path(directory), client)
            self.assertEqual(result["requests"], 4)
            self.assertTrue(result["unused_allowance_closed"])
            self.assertEqual(len((Path(directory) / "sdk-ledger.jsonl").read_text().splitlines()), 4)
            for request, row in zip(sent, result["results"], strict=True):
                self.assertEqual(summary_request_lane(request), "R")
                self.assertEqual(request["reasoning"], {"effort": row["variant"]})
                self.assertEqual(request["max_output_tokens"], 16384)
                body = request["input"][0]["content"]
                self.assertIn("Fixed previous summary", body)
                self.assertIn("Historical plan", body)
                self.assertNotIn("NEVER_SEND", body)
                self.assertNotIn("new summary", body)
                self.assertEqual(row["budget"]["opaque_items"], 0)
                self.assertEqual(row["http_requests"], 1)
                self.assertEqual(row["status"], "accepted")
        self.assertEqual(frozen, before)
        self.assertEqual([row["variant"] for row in schedule(frozen)], ["high", "max", "max", "high"])

    def test_modified_sources_labels_request_caps_or_limits_are_rejected(self):
        for defect in ("source", "details", "cap", "instructions", "limits"):
            value = manifest()
            if defect == "source":
                value["cases"][0]["source_input"][1]["content"] = "changed"
            elif defect == "details":
                value["cases"][0]["item_details"][1]["kind"] = "record"
            elif defect in ("cap", "instructions"):
                request = value["cases"][0]["requests"]["high"]
                request["max_output_tokens" if defect == "cap" else "instructions"] = 8192 if defect == "cap" else "old"
            else:
                value["limits"]["sdk_requests"] += 1
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                validate(value)
        with self.assertRaises(ValueError):
            prepare(manifest()["cases"][0], "none")


if __name__ == "__main__":
    unittest.main()
