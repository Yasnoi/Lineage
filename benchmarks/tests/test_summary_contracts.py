"""Summary provenance and complete-group protocol boundaries, without a model."""

from copy import deepcopy
import json
import unittest

from lineage_agent.models.responses import compactable_group_items
from lineage_agent.models.summarization import (
    SUMMARY_FIELDS, SummaryError, minimum_summary_message_bytes, parse_summary,
    prepare_summary_request, summary_request_lane, unpack_summary_material,
)
from lineage_agent.config.settings import ModelSettings


SOURCE = "history://record/01234567-89ab-cdef-0123-456789abcdef/2"
ARTIFACT = "artifact://tool/0123456789abcdef0123456789abcdef"
SCOPE = "history://scope/01234567-89ab-cdef-0123-456789abcdef"
SETTINGS = ModelSettings(base_url="https://summary.invalid/v1", model_name="offline")


def result(text="Observed failure; retry remains unfinished."):
    document = {field: [] for field in SUMMARY_FIELDS}
    document["facts"] = [{"text": text, "source_ids": [SOURCE]}]
    return {"raw_response": {"status": "completed", "error": None},
            "has_refusal": False, "tool_calls": [], "text": json.dumps(document)}


class SummaryContractsTests(unittest.TestCase):
    def test_lane_and_byte_target_are_independent_of_api_cap(self):
        material = {"sources": [], "current_user_background": "do not change 17 or the negation"}
        original = deepcopy(material)
        for lane in ("H", "R"):
            for target in (1, 24000):
                with self.subTest(lane=lane, target=target):
                    request = prepare_summary_request(SETTINGS, material, target_bytes=target, lane=lane)
                    self.assertEqual(summary_request_lane(request), lane)
                    self.assertEqual(request["max_output_tokens"], SETTINGS.max_output_tokens)
                    self.assertEqual(unpack_summary_material(request), material)
                    self.assertEqual(material, original)
                    self.assertNotEqual(summary_request_lane({**request, "instructions": request["instructions"] + "x"}), lane)
        historical = prepare_summary_request(SETTINGS, material, target_bytes=900, lane="H")
        current = prepare_summary_request(SETTINGS, material, target_bytes=900, lane="R")
        self.assertIn("task ending from a task succeeding", historical["instructions"])
        self.assertIn("Do not attribute historical work", current["instructions"])
        self.assertIn("截至覆盖范围", current["instructions"])
        self.assertIn("Missing from these sources does not mean", current["instructions"])
        self.assertNotEqual(historical["instructions"], current["instructions"])

    def test_invalid_size_or_lane_never_prepares_a_request(self):
        for value in (0, -1, True, 1.5, "900"):
            with self.subTest(value=value), self.assertRaises(SummaryError):
                prepare_summary_request(SETTINGS, {}, target_bytes=value)
        with self.assertRaises(SummaryError):
            prepare_summary_request(SETTINGS, {}, target_bytes=900, lane="unknown")

    def test_machine_refs_need_explicit_allowed_origin(self):
        for text in (SOURCE, f"See ({SOURCE})."):
            parsed = parse_summary(result(text), allowed_source_ids={SOURCE}, summary_id="summary", lane="R")
            self.assertEqual(json.loads(parsed["message"]["content"])["lane"], "R")
        for reference in (ARTIFACT, SCOPE, SOURCE + "99", "history://record/forged/1"):
            with self.subTest(reference=reference), self.assertRaises(SummaryError):
                parse_summary(result("See " + reference), allowed_source_ids={SOURCE}, summary_id="summary")
        parsed = parse_summary(result(ARTIFACT), allowed_source_ids={SOURCE}, summary_id="summary",
                               allowed_machine_refs={ARTIFACT})
        self.assertIn(ARTIFACT, parsed["message"]["content"])
        # Plain business identifiers are evidence, not application access refs.
        parse_summary(result("Business UUID 11111111-1111-1111-1111-111111111111"),
                      allowed_source_ids={SOURCE}, summary_id="summary")

    def test_invalid_source_or_structure_does_not_pass(self):
        value = result()
        document = json.loads(value["text"])
        for sources in ([], [SOURCE, SOURCE], ["history://record/forged/2"]):
            document["facts"][0]["source_ids"] = sources
            with self.subTest(sources=sources), self.assertRaises(SummaryError):
                parse_summary({**value, "text": json.dumps(document)}, allowed_source_ids={SOURCE}, summary_id="summary")
        for changed in ({"has_refusal": True}, {"tool_calls": [{"call_id": "never-execute"}]},
                        {"raw_response": {"status": "incomplete", "error": None}}):
            with self.subTest(changed=changed), self.assertRaises(SummaryError):
                parse_summary({**value, **changed}, allowed_source_ids={SOURCE}, summary_id="summary")

    def test_minimum_size_measures_complete_lane_message(self):
        parsed = parse_summary(result("x"), allowed_source_ids={SOURCE}, summary_id="摘要", lane="R")
        self.assertEqual(minimum_summary_message_bytes("摘要", {SOURCE}, lane="R"), parsed["bytes"])
        self.assertEqual(parsed["bytes"], len(json.dumps(parsed["message"], ensure_ascii=False,
                                                        separators=(",", ":")).encode("utf-8")))


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
