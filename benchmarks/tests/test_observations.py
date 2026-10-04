"""Deterministic contracts for bounded observations and optional schema hints."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from lineage_agent.artifacts.store import ArtifactStore
from lineage_agent.artifacts.tools import ARTIFACT_TOOLS
from lineage_agent.config.settings import ConfigurationError
from lineage_agent.observations.hints import enhance_tools, split_observation
from lineage_agent.observations.manager import ObservationManager, safe_observation_error
from lineage_agent.tools.filesystem import paths
from lineage_agent.tools.filesystem.text import TEXT_TOOL
from lineage_agent.tools.types import (
    MAX_OBSERVATION_BYTES, MAX_RECALL_BYTES, inline_tool_output, serialize_tool_output,
    tool_output_bytes,
)


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="lineage-observation-unit-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root_patch = patch("lineage_agent.artifacts.store.Path.cwd", return_value=self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.store = ArtifactStore()
        self.addCleanup(self.store.close)
        self.view = self.store.bind_session("test-session")
        self.manager = ObservationManager(self.view)

    def check_preview(self, output, content):
        self.assertTrue(output["ok"])
        self.assertLessEqual(tool_output_bytes(output), MAX_OBSERVATION_BYTES)
        observation = output["observation"]
        self.assertEqual(observation["status"], "available")
        self.assertEqual(observation["mode"], "artifact")
        self.assertEqual(observation["total_chars"], len(content))
        previous = -1
        for segment in observation["segments"]:
            self.assertGreater(segment["start"], previous)
            self.assertGreater(segment["end"], segment["start"])
            self.assertEqual(segment["text"], content[segment["start"]:segment["end"]])
            previous = segment["end"]
        if content:
            self.assertTrue(observation["segments"])
        covered = sum(segment["end"] - segment["start"] for segment in observation["segments"])
        self.assertEqual(observation["complete"], covered == len(content))
        return observation

    def test_inline_preserves_raw_and_exact_threshold(self):
        allowance = MAX_OBSERVATION_BYTES - tool_output_bytes(inline_tool_output(""))
        for raw in ("", "raw\r\ntext", '{"content":"正文"}', "x" * allowance):
            with self.subTest(raw_size=len(raw)), patch.object(self.view, "put", side_effect=AssertionError("small result archived")):
                result = self.manager.observe(raw, tool_name="custom")
            self.assertEqual(result, inline_tool_output(raw))
            self.assertLessEqual(tool_output_bytes(result), MAX_OBSERVATION_BYTES)
        self.assertEqual(tool_output_bytes(inline_tool_output("x" * allowance)), MAX_OBSERVATION_BYTES)
        result = self.manager.observe("x" * (allowance + 1), tool_name="custom")
        self.check_preview(result, "x" * (allowance + 1))

    def test_escaping_and_multibyte_are_measured_on_final_output(self):
        for content in ('中😀"\\\t' * 9000, "a" * 30000, "prefix。\n" + "尾" * 15000):
            raw = serialize_tool_output({"content": content})
            result = self.manager.observe(raw, tool_name="read_text_file")
            observation = self.check_preview(result, content)
            self.assertEqual(observation["view_kind"], "content")
            self.assertEqual(observation["strategy"], "head_tail")
            self.assertEqual(observation["keyword_status"], "not_requested")
            self.assertIsNone(observation["matches_truncated"])
            self.assertEqual(observation["segments"][0]["start"], 0)
            self.assertEqual(observation["segments"][-1]["end"], len(content))
            file = self.store._directory / (observation["artifact_ref"].rsplit("/", 1)[1] + ".raw")
            self.assertEqual(file.read_bytes(), raw.encode("utf-8"))

    def test_only_known_content_is_decoded_and_fallback_is_explicit(self):
        raw = json.dumps({"content": "x" * 6000})
        observation = self.check_preview(self.manager.observe(raw, tool_name="unknown"), raw)
        self.assertEqual(observation["view_kind"], "raw_result")
        broken = '{"content":42,"padding":"' + "x" * 6000 + '"}'
        observation = self.check_preview(self.manager.observe(broken, tool_name="read_text_file"), broken)
        self.assertEqual(observation["degradation_reason"], "invalid_text_result")

    def test_head_tail_budget_is_balanced_by_serialized_bytes(self):
        content = "a" * 10000 + "😀" * 10000
        observation = self.check_preview(self.manager.observe(content, tool_name="custom"), content)
        self.assertEqual(observation["raw_bytes"], len(content.encode("utf-8")))
        head, tail = observation["segments"]
        sizes = [len(serialize_tool_output(item["text"]).encode("utf-8")) - 2 for item in (head, tail)]
        self.assertLessEqual(abs(sizes[0] - sizes[1]), 4)
        self.assertGreater(len(head["text"]), len(tail["text"]) * 3)

    def test_full_short_or_empty_body_may_fit_after_archive(self):
        for content in ("", "small decoded body"):
            raw = json.dumps({"content": content, "large_metadata": "x" * 8000})
            observation = self.check_preview(self.manager.observe(raw, tool_name="read_text_file"), content)
            self.assertTrue(observation["complete"])

    def test_matching_preview_records_whole_groups_and_merges_overlaps(self):
        content = "x" * 5000 + "ERROR failed ERROR" + "尾" * 5000
        raw = json.dumps({"content": content})
        observation = self.check_preview(self.manager.observe(raw, tool_name="read_text_file", keywords=("ERROR", "ERR", "failed")), content)
        self.assertEqual(observation["keyword_status"], "matched")
        self.assertEqual(observation["strategy"], "keywords")
        self.assertFalse(observation["matches_truncated"])
        self.assertEqual(len(observation["segments"]), 1)
        self.assertEqual(observation["matches"], [
            {"keyword_index": 0, "start": 5000, "end": 5005},
            {"keyword_index": 1, "start": 5000, "end": 5003},
            {"keyword_index": 2, "start": 5006, "end": 5012},
            {"keyword_index": 0, "start": 5013, "end": 5018},
            {"keyword_index": 1, "start": 5013, "end": 5016},
        ])

    def test_nonmatching_keywords_fall_back_without_claiming_no_errors(self):
        content = "x" * 6000 + "error" + "y" * 1000
        observation = self.check_preview(self.manager.observe(content, tool_name="custom", keywords=("ERROR",)), content)
        self.assertEqual(observation["strategy"], "head_tail")
        self.assertEqual(observation["keyword_status"], "no_match")
        self.assertIsNone(observation["matches_truncated"])
        self.assertNotIn("matches", observation)

    def test_full_segment_coverage_does_not_claim_all_matches_were_returned(self):
        content = "KEY" * 30
        raw = json.dumps({"content": content, "large_metadata": "x" * 8000})
        observation = self.check_preview(self.manager.observe(raw, tool_name="read_text_file", keywords=("KEY",)), content)
        self.assertTrue(observation["complete"])
        self.assertTrue(observation["matches_truncated"])
        self.assertEqual(len(observation["matches"]), 10)

    def test_context_is_reduced_before_dropping_matching_groups(self):
        content = "\x00" * 300
        for _ in range(5):
            content += "KEY" + "\x00" * 300
        observation = self.check_preview(self.manager.observe(content, tool_name="custom", keywords=("KEY",)), content)
        self.assertEqual(len(observation["matches"]), 5)
        self.assertFalse(observation["matches_truncated"])
        for match in observation["matches"]:
            self.assertEqual(content[match["start"]:match["end"]], "KEY")

    def test_dense_same_start_groups_are_not_partially_disclosed(self):
        terms = ["x" * length for length in range(121, 129)]
        content = "x" * 20000
        observation = self.check_preview(self.manager.observe(content, tool_name="custom", keywords=terms), content)
        matches = observation["matches"]
        starts = sorted({match["start"] for match in matches})
        self.assertTrue(observation["matches_truncated"])
        self.assertEqual(starts, list(range(len(starts))))
        for start in starts:
            self.assertEqual([match["keyword_index"] for match in matches if match["start"] == start], list(range(8)))
        self.assertLessEqual(len(starts), 10)

    def test_native_recall_uses_eight_kib_without_recursive_archive(self):
        allowance = MAX_RECALL_BYTES - tool_output_bytes(inline_tool_output(""))
        raw = "x" * allowance
        with patch.object(self.view, "put", side_effect=AssertionError("native recall archived")):
            result = self.manager.observe(raw, tool_name="artifact_read", native_recall=True)
        self.assertEqual(result, inline_tool_output(raw))
        self.assertEqual(tool_output_bytes(result), MAX_RECALL_BYTES)
        failed = self.manager.observe(raw + "x", tool_name="artifact_read", native_recall=True)
        self.assertEqual(failed["observation"]["stage"], "serialize")
        self.assertEqual(list(self.store._directory.iterdir()), [])
        fake = self.manager.observe(raw, tool_name="artifact_read")
        self.assertEqual(fake["observation"]["mode"], "artifact")

    def test_native_store_read_and_search_fit_the_same_envelope(self):
        ref = self.view.put(('中\\"\x00KEY' * 5000), tool_name="custom")["artifact_ref"]
        for page in (self.view.read(ref, 0), self.view.search(ref, ["KEY"])):
            raw = serialize_tool_output(page)
            result = self.manager.observe(raw, tool_name="artifact_read", native_recall=True)
            self.assertEqual(result, inline_tool_output(raw))
            self.assertLessEqual(tool_output_bytes(result), MAX_RECALL_BYTES)
        self.assertEqual(len(list(self.store._directory.glob("*.raw"))), 1)

    def test_serialization_failure_does_not_publish_or_disclose_raw(self):
        for raw in ("PRIVATE\ud800", None, b"PRIVATE"):
            result = self.manager.observe(raw, tool_name="custom")
            self.assertTrue(result["ok"])
            self.assertEqual(result["observation"]["stage"], "serialize")
            self.assertNotIn("PRIVATE", serialize_tool_output(result))
            self.assertLessEqual(tool_output_bytes(result), MAX_OBSERVATION_BYTES)
        self.assertEqual(list(self.store._directory.iterdir()), [])

    def test_save_failure_preserves_execution_success_without_reference(self):
        with patch.object(self.view, "put", side_effect=OSError("PRIVATE")):
            result = self.manager.observe("x" * 8000, tool_name="custom")
        self.assertTrue(result["ok"])
        self.assertEqual(result["observation"]["status"], "unavailable")
        self.assertEqual(result["observation"]["stage"], "save")
        self.assertNotIn("artifact_ref", result["observation"])
        self.assertNotIn("PRIVATE", serialize_tool_output(result))
        self.store.close()
        closed = self.manager.observe("x" * 8000, tool_name="custom")
        self.assertEqual(closed["observation"]["error"]["code"], "artifact_store_closed")

    def test_preview_failure_preserves_saved_reference_and_snapshot(self):
        raw = "KEY" + "x" * 8000
        with patch.object(self.view, "search", side_effect=OSError("PRIVATE")):
            result = self.manager.observe(raw, tool_name="custom", keywords=("KEY",))
        observation = result["observation"]
        self.assertTrue(result["ok"])
        self.assertEqual(observation["stage"], "preview")
        self.assertNotIn("PRIVATE", serialize_tool_output(result))
        self.assertEqual(self.view.read(observation["artifact_ref"], 0, 3)["text"], "KEY")
        self.assertEqual(len(list(self.store._directory.glob("*.raw"))), 1)

    def test_cancellation_is_not_converted_to_a_tool_observation(self):
        with patch.object(self.view, "put", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.manager.observe("x" * 8000, tool_name="custom")

    def test_unknown_error_text_is_not_trusted(self):
        code, message = safe_observation_error("PRIVATE")
        self.assertEqual(code, "observation_preview_error")
        self.assertNotIn("PRIVATE", message)


class HintTests(unittest.TestCase):
    def test_only_canonical_reader_is_enhanced_without_mutating_inputs(self):
        reader = deepcopy(TEXT_TOOL)
        reader["description"] = "description supplied by application"
        tools = [reader, *deepcopy(ARTIFACT_TOOLS)]
        original = deepcopy(tools)
        enhanced, names = enhance_tools(tools)
        self.assertEqual(tools, original)
        self.assertEqual(names, frozenset({"read_text_file"}))
        self.assertEqual(enhanced[1:], tools[1:])
        parameters = enhanced[0]["parameters"]
        self.assertEqual(parameters["required"], ["path", "observation"])
        self.assertFalse(parameters["additionalProperties"])
        self.assertEqual(parameters["properties"]["observation"]["required"], ["intent", "keywords"])

    def test_canonical_name_collision_rejects_but_custom_business_fields_remain(self):
        reader = deepcopy(TEXT_TOOL)
        reader["parameters"]["properties"]["observation"] = {"type": "string"}
        with self.assertRaises(ConfigurationError):
            enhance_tools([reader])
        reader["name"] = "custom_reader"
        enhanced, names = enhance_tools([reader])
        self.assertEqual(enhanced, [reader])
        self.assertFalse(names)

    def test_valid_hints_are_removed_from_business_arguments_only(self):
        arguments = {"path": "file.txt", "observation": {"intent": "定位失败", "keywords": ["ERROR", "失败"]}}
        original = deepcopy(arguments)
        business, words = split_observation(arguments)
        self.assertEqual(business, {"path": "file.txt"})
        self.assertEqual(words, ("ERROR", "失败"))
        self.assertEqual(arguments, original)
        self.assertEqual(split_observation({"path": "x", "observation": {"intent": None, "keywords": []}}), ({"path": "x"}, ()))

    def test_invalid_hints_never_turn_into_business_arguments_or_block_execution(self):
        invalid = [None, 42, "hint", {}, {"keywords": ["ERROR"]},
                   {"intent": "x" * 513, "keywords": ["ERROR"]},
                   {"intent": False, "keywords": ["ERROR"]},
                   {"intent": None, "keywords": "ERROR"},
                   {"intent": None, "keywords": [""]},
                   {"intent": None, "keywords": ["x" * 129]},
                   {"intent": None, "keywords": ["x"] * 9},
                   {"intent": None, "keywords": ["\ud800"]},
                   {"intent": None, "keywords": ["ERROR"], "extra": True}]
        for hint in invalid:
            with self.subTest(hint=repr(hint)):
                business, words = split_observation({"path": "x", "observation": hint, "business_extra": True})
                self.assertEqual(business, {"path": "x", "business_extra": True})
                self.assertEqual(words, ())
        self.assertEqual(split_observation({"path": "x"}), ({"path": "x"}, ()))


if __name__ == "__main__":
    unittest.main()
