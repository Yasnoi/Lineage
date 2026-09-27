"""Offline faults for the real-model dispatcher; no endpoint is called."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from benchmarks import __main__ as cli
from benchmarks.common import sha256_file
from benchmarks.research import batch


class ResearchDispatchTests(unittest.TestCase):
    def test_explicit_profile_and_separate_selection_required(self):
        for args in (["--suite", "research-wechat"],
                     ["--suite", "research-wechat", "--suite", "control-react", "--research-profile", "wechat-v1"],
                     ["--research-profile", "wechat-v1"]):
            with self.subTest(args=args), patch.object(cli, "run_benchmark") as run, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    cli.main(args)
                self.assertEqual(raised.exception.code, 2)
                run.assert_not_called()

    def test_default_keeps_control_path(self):
        result = {"run_id": "fixture", "units": [], "framework_errors": []}
        with patch.object(cli, "run_benchmark", return_value=(result, [])) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main([]), 2)
        run.assert_called_once_with(["control-react"])

    def test_filtered_connection_and_no_mutating_environment(self):
        profile = {"base_url": "https://api.deepseek.com", "model_name": "deepseek-flash"}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env").write_text("API_KEY=credential-marker\nBASE_URL=https://api.deepseek.com\n"
                                       "MODEL_NAME=deepseek-flash\nLANGSMITH_API_KEY=trace-marker\n"
                                       "MAX_MODEL_CALLS=999\n", encoding="utf-8")
            env = {}
            connection = batch.load_connection(root, profile, env)
            self.assertEqual(env, {})
            self.assertEqual(set(connection), {"API_KEY", "BASE_URL", "MODEL_NAME"})
            with patch.dict("os.environ", {"LANGSMITH_API_KEY": "trace-marker", "PYTHONOPTIMIZE": "2"}):
                child = batch.research_environment(connection)
            self.assertNotIn("trace-marker", json.dumps(child))
            self.assertNotIn("PYTHONOPTIMIZE", child)
            self.assertEqual(child["LANGSMITH_TRACING"], "false")
            with self.assertRaises(ValueError):
                batch.load_connection(root, profile, {"MODEL_NAME": "unapproved"})
            with self.assertRaises(ValueError):
                batch.load_connection(root, profile, {"API_KEY": ""})

    def test_receipt_identity_and_finality_not_success_text(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            spec = {"run_id": "sample-a", "case_id": "case", "prompts": ["one"]}
            spec_path = directory / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            path = directory / "receipt.json"
            data = {"schema_version": "1", "run_id": "sample-a", "case_id": "case",
                    "spec_sha256": sha256_file(spec_path), "final": True,
                    "verdict": "pending_review", "global_stop": False,
                    "execution_status": "completed", "ended_at": "now", "turns": [{}]}
            path.write_text(json.dumps(data), encoding="utf-8")
            batch._receipt(path, spec, spec_path, final=True, returncode=0)
            for changes in ({"run_id": "old"}, {"spec_sha256": "old"}, {"final": False}, {"turns": []}):
                path.write_text(json.dumps({**data, **changes}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    batch._receipt(path, spec, spec_path, final=True, returncode=0)

    def test_complete_batch_keeps_all_samples_and_review_packets(self):
        from benchmarks.research import materials
        from benchmarks.results import known
        from benchmarks.research.definitions import CASES
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"material_version": "fixture", "files": []}

            def instantiate(prepared, destination, case):
                material_root = destination / "material"
                material_root.mkdir(parents=True)
                return {**manifest, "material_root": str(material_root)}

            def execute(unit, spec, directory, connection, **kwargs):
                # Explicit adapter fixture: no Agent/model runs, only real saved artifacts.
                unit["execution_status"] = "completed"
                unit["agent_outcome"] = known("completed")
                unit["coverage"] = []
                unit["metrics"]["duration_seconds"] = known(0.1)
                for turn in unit["turns"]:
                    answer = directory / f"turn-{turn['turn']}-answer.md"
                    answer.write_text("确定性适配 fixture；不是 Research 结果。", encoding="utf-8")
                    turn.update(execution_status="completed", run_id=known(f"fixture-{unit['unit_id']}-{turn['turn']}"),
                                agent_outcome=known("completed"), answer_sha256=sha256_file(answer),
                                answer_evidence={"kind": "answer", "path": str(answer), "source": "fixture", "availability": "available"})
                return False

            with (patch.object(batch, "collect_baseline", return_value={}),
                  patch.object(batch, "load_connection", return_value={}),
                  patch.object(materials, "prepare_materials", return_value=manifest),
                  patch.object(materials, "instantiate_materials", side_effect=instantiate),
                  patch.object(materials, "verify_materials", side_effect=lambda path: {**manifest, "material_root": str(path / "material")}),
                  patch.object(batch, "execute_sample", side_effect=execute), redirect_stdout(io.StringIO())):
                result, saved = batch.run_research("wechat-v1", root=root)
            self.assertEqual(len(result["units"]), 24)
            self.assertEqual(sum(len(unit["turns"]) for unit in result["units"]), 56)
            self.assertEqual(len({unit["sample_run_id"] for unit in result["units"]}), 24)
            self.assertTrue(all(unit["verdict"] == "pending_review" for unit in result["units"]))
            self.assertEqual(result["framework_errors"], [])
            self.assertEqual({path.name for path in saved}, {"result.json", "report.md", "review-template.json", "review.md"})
            self.assertTrue((saved[0].parent / "pilot-gate.json").is_file())
            for sample in result["batch"]["samples"]:
                self.assertTrue((Path(sample["directory"]) / "result.json").is_file())

    def test_missing_configuration_blocks_without_worker(self):
        with TemporaryDirectory() as temporary:
            with (patch.object(batch, "collect_baseline", return_value={}),
                  patch.object(batch, "load_connection", side_effect=ValueError("missing_credentials")),
                  patch.object(batch, "execute_sample") as execute, redirect_stderr(io.StringIO())):
                result, saved = batch.run_research("wechat-v1", root=Path(temporary))
            execute.assert_not_called()
            self.assertTrue(all(unit["execution_status"] == "blocked" for unit in result["units"]))
            self.assertEqual(len(result["units"]), 24)
            self.assertTrue(any(path.name == "report.md" for path in saved))


if __name__ == "__main__":
    unittest.main()
