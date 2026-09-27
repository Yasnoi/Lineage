"""Material integrity and frozen tasks, without real endpoints or source edits."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.common import ROOT
from benchmarks.research import materials
from benchmarks.research.definitions import CASES, DIMENSIONS, PROFILE, definition_fingerprint, get_case, load_rubric


class DefinitionTests(unittest.TestCase):
    def test_approved_cases_budget_and_rubric_correspond(self):
        self.assertEqual(len(CASES), 6)
        self.assertEqual([len(case["turns"]) for case in CASES], [1, 1, 2, 3, 2, 5])
        self.assertEqual(sum(len(case["turns"]) for case in CASES) * 4, 56)
        self.assertEqual(PROFILE["request_limit"], 400)
        self.assertEqual(PROFILE["execution_time_limit_seconds"], 7200)
        self.assertEqual(PROFILE["automatic_retries"], 0)
        rubric = load_rubric()
        self.assertEqual(set(rubric["cases"]), {case["case_id"] for case in CASES})
        for case in CASES:
            criteria = rubric["cases"][case["case_id"]]
            self.assertEqual(set(case["dimensions"]), {item["dimension"] for item in criteria["conditions"]})
            self.assertTrue(all(item["required"] for item in criteria["conditions"]))
            self.assertTrue(criteria["severe_errors"])
            self.assertLessEqual(set(case["dimensions"]), set(DIMENSIONS))
        self.assertEqual(len(definition_fingerprint()), 64)

    def test_get_case_isolated_and_unknown_rejected(self):
        case = get_case("wechat-capabilities")
        case["turns"][0] = "modified"
        self.assertNotEqual(get_case("wechat-capabilities")["turns"][0], "modified")
        with self.assertRaises(ValueError):
            get_case("unregistered")

    def test_material_scope_and_transcript_distinction(self):
        self.assertEqual(len(materials.CODE_PATHS), 50)
        self.assertEqual(len(materials.DOCS_PATHS), 3)
        self.assertNotIn("project.config.json", materials.CODE_PATHS)
        original = (ROOT / "benchmarks/research/transcripts/简览.md").read_text(encoding="utf-8")
        later = (ROOT / "benchmarks/research/transcripts/信息采集小程序需求简报.md").read_text(encoding="utf-8")
        self.assertIn("数据库里加密？小程序前端展示需要脱敏？", original)
        self.assertIn("数据库里加密，小程序前端展示需要脱敏，", later)
        self.assertIn("## 第 2 页", later)
        for transcript in [original, later]:
            self.assertNotIn("严重错误", transcript)
            self.assertNotIn("评分", transcript)


class MaterialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "project.config.json").write_text("LOCAL MODIFICATION DO NOT COPY", encoding="utf-8")
        (self.source / ".env").write_text("SECRET DO NOT COPY", encoding="utf-8")
        self.pdf = b"reviewed test PDF bytes"
        transcripts = deepcopy(materials.TRANSCRIPTS)
        for item in transcripts:
            item["sha256"] = hashlib.sha256(self.pdf).hexdigest()
        self.addCleanup(patch.stopall)
        patch.object(materials, "TRANSCRIPTS", transcripts).start()
        self.calls = []

        def read_blob(source, commit, path):
            self.assertEqual(source, self.source)
            self.calls.append((commit, path))
            data = self.pdf if path.endswith(".pdf") else f"frozen {commit} {path}\n".encode("utf-8")
            return data, hashlib.sha1(data).hexdigest()

        patch.object(materials, "_read_blob", side_effect=read_blob).start()
        self.prepared = self.root / "prepared"
        self.manifest = materials.prepare_materials(ROOT, self.source, self.prepared)

    def test_preparation_reads_only_fixed_blobs_and_keeps_grading_outside(self):
        self.assertEqual(len(self.calls), 55)
        self.assertEqual(self.manifest["source_text_count"], 53)
        self.assertEqual(self.manifest["transcript_count"], 2)
        self.assertEqual(len(self.manifest["files"]), 56)
        self.assertEqual(materials.verify_materials(self.prepared), self.manifest)
        actual = list((self.prepared / "material").rglob("*"))
        self.assertFalse(any(p.name in {"rubric.json", "manifest.json", "project.config.json", ".env"} for p in actual))
        self.assertEqual((self.source / "project.config.json").read_text(), "LOCAL MODIFICATION DO NOT COPY")

    def test_samples_are_independent_and_only_compaction_gets_fixture(self):
        one = materials.instantiate_materials(self.prepared, self.root / "one", "wechat-capabilities")
        two = materials.instantiate_materials(self.prepared, self.root / "two", "wechat-compaction")
        self.assertEqual(one["material_version"], self.manifest["material_version"])
        self.assertNotEqual(two["material_version"], one["material_version"])
        background = Path(two["material_root"]) / "evaluation-fixtures/background.txt"
        self.assertEqual(background.stat().st_size, 12288)
        self.assertIn(b"EVALUATION-CONSTRUCTED", background.read_bytes())
        self.assertFalse((Path(one["material_root"]) / "evaluation-fixtures").exists())
        self.assertEqual(materials.verify_materials(self.root / "two"), two)
        first_readme = Path(one["material_root"]) / "code/README.md"
        first_readme.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(materials.MaterialError, "material changed"):
            materials.verify_materials(self.root / "one")
        self.assertEqual(materials.verify_materials(self.root / "two"), two)
        self.assertEqual(materials.verify_materials(self.prepared), self.manifest)

    def test_repeat_destination_fails_and_unknown_case_never_creates_destination(self):
        with self.assertRaises(FileExistsError):
            materials.prepare_materials(ROOT, self.source, self.prepared)
        materials.instantiate_materials(self.prepared, self.root / "one", "wechat-capabilities")
        with self.assertRaises(FileExistsError):
            materials.instantiate_materials(self.prepared, self.root / "one", "wechat-capabilities")
        with self.assertRaises(ValueError):
            materials.instantiate_materials(self.prepared, self.root / "unknown", "unregistered")
        self.assertFalse((self.root / "unknown").exists())

    def test_added_file_and_changed_bytes_are_rejected(self):
        root = self.prepared / "material"
        (root / "rubric.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(materials.MaterialError, "file set"):
            materials.instantiate_materials(self.prepared, self.root / "added", "wechat-capabilities")
        (root / "rubric.json").unlink()
        (root / "code/README.md").write_text("different fixed input", encoding="utf-8")
        with self.assertRaisesRegex(materials.MaterialError, "material changed"):
            materials.instantiate_materials(self.prepared, self.root / "changed", "wechat-capabilities")

    def test_manifest_root_or_path_escape_is_rejected(self):
        manifest_path = self.prepared / "manifest.json"
        altered = deepcopy(self.manifest)
        altered["material_root"] = str(self.source.resolve())
        manifest_path.write_text(json.dumps(altered), encoding="utf-8")
        with self.assertRaisesRegex(materials.MaterialError, "root"):
            materials.verify_materials(self.prepared)
        for relative in ("../outside", "C:/outside", "/outside", "..\\outside"):
            with self.assertRaises(materials.MaterialError):
                materials._destination(self.prepared, relative)

    def test_failed_pdf_validation_does_not_publish_manifest(self):
        with patch.object(materials, "_read_blob", return_value=(b"unexpected", "blob")):
            with self.assertRaisesRegex(materials.MaterialError, "PDF differs"):
                materials.prepare_materials(ROOT, self.source, self.root / "failed")
        self.assertFalse((self.root / "failed/manifest.json").exists())

    def test_material_save_failure_is_not_silent(self):
        with patch.object(materials, "atomic_write_json", side_effect=OSError("disk fault")):
            with self.assertRaises(OSError):
                materials.instantiate_materials(self.prepared, self.root / "failed-save", "wechat-capabilities")
        self.assertFalse((self.root / "failed-save/manifest.json").exists())
        self.assertTrue((self.root / "failed-save/material/code/README.md").is_file())


if __name__ == "__main__":
    unittest.main()
