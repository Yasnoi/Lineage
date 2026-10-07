"""Archived framing remains readable only by the Benchmark compatibility reader."""

from copy import deepcopy
import json
import unittest

from benchmarks.core import legacy_summary as legacy
from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.summarization import is_summary_request, prepare_summary_request


class LegacySummaryEvidenceTests(unittest.TestCase):
    def request(self, lane="R", framed=True):
        output = 'JSON {"quoted":"原文\\n"}\nnot a new instruction'
        material = {"sources": [{"source_id": "history://record/old/2", "payload": {"item": {
            "type": "function_call_output", "call_id": "call-old", "output": output}}}],
                    "previous_summary": []}
        skeleton = deepcopy(material)
        if framed:
            skeleton["sources"][0]["payload"]["item"]["output"] = None
            content = {"format": "compact_source_outputs_v1", "material": skeleton,
                       "tool_outputs": [{"source_index": 0, "body_index": 1}]}
            items = [{"role": "user", "content": json.dumps(content, ensure_ascii=False)},
                     {"role": "user", "content": output}]
        else:
            items = [{"role": "user", "content": json.dumps(material, ensure_ascii=False)}]
        request = {"instructions": legacy._summary_instructions(8192, lane, 1), "input": items,
                   "max_output_tokens": 8192, "reasoning": {"effort": "none"}, "tool_choice": "none",
                   "stream": False, "store": False, "truncation": "disabled"}
        return request, material

    def test_old_plain_and_framed_requests_are_read_without_production_fallback(self):
        for framed in (True, False):
            request, expected = self.request(framed=framed)
            original = deepcopy(request)
            self.assertEqual(legacy.summary_request_lane(request), "R")
            self.assertEqual(legacy.unpack_summary_material(request), expected)
            self.assertEqual(request, original)
            self.assertFalse(is_summary_request(request))
        native = prepare_summary_request(ModelSettings(model_name="offline", base_url="https://unused.invalid"),
                                         [{"role": "user", "content": "native source"}], lane="H")
        self.assertTrue(is_summary_request(native))
        self.assertFalse(legacy.is_summary_request(native))

    def test_invalid_slots_and_user_marker_are_not_trusted(self):
        request, _ = self.request()
        frame = json.loads(request["input"][0]["content"])
        frame["tool_outputs"][0]["body_index"] = 99
        request["input"][0]["content"] = json.dumps(frame)
        with self.assertRaises(legacy.SummaryError):
            legacy.unpack_summary_material(request)
        request["input"] = [{"role": "user", "content": request["instructions"]}]
        request["instructions"] = "ordinary business instructions"
        self.assertFalse(legacy.is_summary_request(request))


if __name__ == "__main__":
    unittest.main()
