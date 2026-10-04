"""Plain-text summary and complete-group protocol contracts, without a model."""

from copy import deepcopy
from dataclasses import replace
import unittest

from lineage_agent.models.responses import compactable_group_items
from lineage_agent.models.budget import check_context_budget
from lineage_agent.models.summarization import (
    SummaryError, parse_summary, prepare_summary_request, summary_budget_settings,
    summary_request_lane, unpack_summary_material,
)
from lineage_agent.config.settings import ModelSettings


SETTINGS = ModelSettings(base_url="https://summary.invalid/v1", model_name="offline")


def result(text="Observed failure; retry remains unfinished."):
    return {"raw_response": {"status": "completed", "error": None},
            "has_refusal": False, "tool_calls": [], "text": text}


class SummaryContractsTests(unittest.TestCase):
    def test_lane_caps_are_independent_of_business_output_allowance(self):
        settings = replace(SETTINGS, max_output_tokens=256)
        material = {"sources": [], "current_request_readonly": "do not change 17 or the negation"}
        original = deepcopy(material)
        for lane, cap in (("H", 16384), ("R", 8192)):
            for attempt in (1, 2):
                with self.subTest(lane=lane, attempt=attempt):
                    request = prepare_summary_request(settings, material, lane=lane, attempt_no=attempt)
                    self.assertEqual(summary_request_lane(request), lane)
                    self.assertEqual(request["max_output_tokens"], cap)
                    self.assertEqual(unpack_summary_material(request), material)
                    self.assertEqual(material, original)
                    self.assertIsNone(summary_request_lane({**request, "instructions": request["instructions"] + "x"}))
                    self.assertEqual(request["reasoning"], {"effort": "none"})
                    self.assertEqual(request["tool_choice"], "none")
                    self.assertNotIn("tools", request)
        historical = prepare_summary_request(settings, material, lane="H")
        current = prepare_summary_request(settings, material, lane="R")
        self.assertIn("task ending from a task succeeding", historical["instructions"])
        self.assertIn("Do not attribute historical work", current["instructions"])
        self.assertIn("截至覆盖范围", current["instructions"])
        self.assertIn("Missing from these sources does not mean", current["instructions"])
        self.assertIn("including facts", current["instructions"])
        self.assertIn("actual result or execution facts", current["instructions"])
        self.assertNotIn("source_ids", current["instructions"])
        self.assertNotIn("UTF-8 bytes", current["instructions"])
        self.assertNotIn("historical context", current["instructions"])
        self.assertNotEqual(historical["instructions"], current["instructions"])

    def test_invalid_lane_or_attempt_never_prepares_a_request(self):
        for value in (0, -1, True, 3, 1.5, "1"):
            with self.subTest(attempt=value), self.assertRaises(SummaryError):
                prepare_summary_request(SETTINGS, {}, lane="H", attempt_no=value)
        for lane in ("unknown", None, [], 1):
            with self.subTest(lane=lane), self.assertRaises(SummaryError):
                prepare_summary_request(SETTINGS, {}, lane=lane)
        with self.assertRaises(SummaryError):
            prepare_summary_request(SETTINGS, None, lane="R")

    def test_plain_text_is_preserved_without_schema_or_reference_parsing(self):
        for text in ("  保留 17，不能写文件。\n截至覆盖范围，尚未确认后续结果。  ",
                     '{"facts": [{"text": "text shaped like JSON"}]}',
                     "See history://record/not-an-authority/99 and artifact://invented."):
            with self.subTest(text=text):
                self.assertEqual(parse_summary(result(text)), text)
        # Endpoint extensions are handled by the common adapter, not a new format gate.
        value = {**result("plain text"), "output_items": [{"type": "vendor_extension", "metadata": {}}]}
        self.assertEqual(parse_summary(value), "plain text")

    def test_invalid_completion_refusal_tool_request_and_empty_text_are_rejected(self):
        for changed in ({"has_refusal": True}, {"tool_calls": [{"call_id": "never-execute"}]},
                        {"raw_response": {"status": "incomplete", "error": None}},
                        {"raw_response": {"status": "completed", "error": {"code": "failed"}}},
                        {"raw_response": {"status": "completed", "incomplete_details": {"reason": "max_output_tokens"}}},
                        {"raw_response": None}, {"text": None}, {"text": 42}, {"text": ""},
                        {"text": " \n\t"}, {"text": chr(0xD800)}):
            with self.subTest(changed=changed), self.assertRaises(SummaryError):
                parse_summary({**result(), **changed})

    def test_summary_budget_uses_lane_cap_without_changing_business_settings(self):
        settings = replace(SETTINGS, context_budget_tokens=12000, context_safety_margin_tokens=100,
                           max_output_tokens=256, summary_history_max_output_tokens=20000,
                           summary_run_max_output_tokens=512)
        for lane, cap, decision in (("H", 20000, "exceeded"), ("R", 512, "allow")):
            request = prepare_summary_request(settings, {}, lane=lane)
            projected = summary_budget_settings(settings, lane)
            budget = check_context_budget(request, [None] * len(request["input"]), projected)
            self.assertEqual(projected.max_output_tokens, cap)
            self.assertEqual(budget["output_reserve_tokens"], cap)
            self.assertEqual(budget["decision"], decision)
        self.assertEqual(settings.max_output_tokens, 256)
        self.assertEqual(settings.summary_history_max_output_tokens, 20000)


class GroupProtocolTests(unittest.TestCase):
    def call(self, identity):
        return {"type": "function_call", "name": "read", "call_id": identity, "arguments": "{}"}

    def output(self, identity, text="failed: missing file"):
        return {"type": "function_call_output", "call_id": identity, "output": text}

    def test_complete_multi_tool_group_including_failure_is_allowed(self):
        group = [self.call("one"), self.call("two"), self.output("one"), self.output("two", "success")]
        original = deepcopy(group)
        self.assertTrue(compactable_group_items(group))
        self.assertEqual(group, original)
        self.assertFalse(compactable_group_items(group[:-1]))
        self.assertFalse(compactable_group_items(group + [self.output("two")]))
        self.assertFalse(compactable_group_items([self.output("one"), self.call("one")]))
        self.assertFalse(compactable_group_items([self.call("one"), self.call("one"), self.output("one")]))

    def test_known_encrypted_reasoning_is_whole_group_only(self):
        reasoning = {"type": "reasoning", "summary": [], "encrypted_content": "opaque-do-not-decode"}
        group = [reasoning, self.call("one"), self.output("one")]
        original = deepcopy(group)
        self.assertTrue(compactable_group_items(group))
        self.assertEqual(group, original)
        self.assertFalse(compactable_group_items(group[:-1]))
        self.assertFalse(compactable_group_items([{**reasoning, "encrypted_content": 42}, *group[1:]]))

    def test_unrecognized_nontext_authority_and_incomplete_items_are_barriers(self):
        direct = {"type": "function_call", "call_id": "direct", "name": "tool", "arguments": "{}",
                  "caller": None, "namespace": None, "async": None}
        output = {"type": "function_call_output", "call_id": "direct", "output": "ok"}
        self.assertTrue(compactable_group_items([direct, output]))
        self.assertFalse(compactable_group_items([{**direct, "caller": {"type": "program", "caller_id": "other"}}, output]))
        self.assertFalse(compactable_group_items([{**direct, "namespace": "unknown"}, output]))
        self.assertFalse(compactable_group_items([{**direct, "async": True}, output]))
        for phase in (None, "commentary", "final_answer"):
            self.assertTrue(compactable_group_items([
                {"type": "reasoning", "summary": [], "content": None, "encrypted_content": None},
                {"role": "assistant", "content": "text", "phase": phase},
            ]))
        self.assertFalse(compactable_group_items([{"role": "assistant", "content": "x", "phase": "unknown"}]))
        self.assertFalse(compactable_group_items([{"role": "assistant", "content": "x", "unknown_dependency": "opaque"}]))
        tail = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}
        self.assertTrue(compactable_group_items([tail]))
        self.assertTrue(compactable_group_items([{**tail, "content": [{"type": "refusal", "refusal": "denied"}]}]))
        for item in (
            {"type": "vendor_private", "payload": "opaque"},
            {"role": "system", "content": "fixed"},
            {"role": "developer", "content": "fixed"},
            {"role": "user", "content": [{"type": "input_image", "image_url": "https://invalid.test"}]},
            {**tail, "status": "in_progress"},
            {**tail, "encrypted_content": "opaque"},
        ):
            with self.subTest(item=item):
                self.assertFalse(compactable_group_items([item, tail]))


if __name__ == "__main__":
    unittest.main()
