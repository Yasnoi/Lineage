"""Public CLI diagnostics and recovery commands never expose input contents."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmarks.core import runner
from benchmarks.core.executors import agent
from benchmarks.core.registry import load_suite
from benchmarks.entrypoints import cli
from benchmarks.evaluators.review import ReviewSaveError
from benchmarks.tests.test_research_batch import CONFIG, SUITE


class CliTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="benchmark-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def invoke(self, arguments):
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = cli.main(arguments)
        return code, output.getvalue(), errors.getvalue()

    def test_list_suite_shows_every_actual_id_without_loading_environment_or_executing(self):
        with patch.object(cli, "run_benchmark") as run, \
                patch("benchmarks.core.configuration.load_settings", side_effect=AssertionError("no environment")):
            code, output, errors = self.invoke(["list", "wechat-reporting"])
        self.assertEqual(code, 0)
        self.assertFalse(errors)
        for unit in load_suite("wechat-reporting")["units"]:
            self.assertIn(f"  {unit['unit_id']}：{unit['title']}（{len(unit['turns'])} 轮）\n", output)
        run.assert_not_called()

    def test_list_engineering_suite_shows_each_chain_and_its_actual_group_count(self):
        with patch.object(cli, "run_benchmark") as run, \
                patch("benchmarks.core.configuration.load_settings", side_effect=AssertionError("no environment")):
            code, output, errors = self.invoke(["list", "agent-contracts"])
        self.assertEqual(code, 0)
        self.assertFalse(errors)
        for unit in load_suite("agent-contracts")["units"]:
            self.assertIn(f"  {unit['unit_id']}：行为链（{len(unit['groups'])} 个行为组）\n", output)
        run.assert_not_called()

    def test_unreadable_and_invalid_configuration_are_safe_and_actionable(self):
        marker = "private-marker-do-not-print"
        path = self.root / "input.json"
        with patch.object(cli, "run_benchmark") as run:
            code, _, errors = self.invoke(["run", "wechat-reporting", "--config", str(path)])
            self.assertEqual(code, 2)
            self.assertIn("--config [file_unreadable]", errors)
            path.write_text('{"secret": "' + marker + '",\n trailing}', encoding="utf-8")
            code, _, errors = self.invoke(["run", "wechat-reporting", "--config", str(path)])
            self.assertEqual(code, 2)
            self.assertIn("--config [invalid_json]", errors)
            self.assertIn("第 2 行", errors)
            self.assertNotIn(marker, errors)
            path.write_bytes(b"\xff\xfe")
            code, _, errors = self.invoke(["review", str(self.root), "--config", str(path)])
            self.assertEqual(code, 2)
            self.assertIn("file_unreadable", errors)
        run.assert_not_called()

    def test_unknown_task_and_blocked_config_explain_the_field(self):
        marker = "private-unit-do-not-print"
        path = self.root / "run.json"
        path.write_text(json.dumps({**CONFIG, "units": [marker]}), encoding="utf-8")
        code, _, errors = self.invoke(["run", "wechat-reporting", "--config", str(path)])
        self.assertEqual(code, 2)
        self.assertIn("units[0] [unknown_unit_id]", errors)
        self.assertIn("benchmarks list", errors)
        self.assertNotIn(marker, errors)
        for config, field in (({**CONFIG, "enabled": False}, "enabled"),
                              ({**CONFIG, "limits": {**CONFIG["limits"], "requests": 0}}, "limits.requests")):
            path.write_text(json.dumps(config), encoding="utf-8")
            def execute(suite, *, config):
                return runner.run_benchmark("second-project", config=config, root=self.root,
                                            registry={"second-project": lambda: SUITE})
            with patch.object(cli, "run_benchmark", side_effect=execute), \
                    patch.object(runner, "collect_baseline", return_value={}), \
                    patch.object(agent, "execute_sample") as worker:
                code, output, errors = self.invoke(["run", "wechat-reporting", "--config", str(path)])
            worker.assert_not_called()
            self.assertEqual(code, 2)
            self.assertIn(field, errors)
            self.assertIn("Saved:", output)
            self.assertEqual(errors.count("配置错误："), 1)

    def test_review_prepare_and_partial_save_publish_only_confirmed_paths(self):
        saved = self.root / "review-packets" / "fresh" / "review-template.json"
        with patch("benchmarks.evaluators.review.prepare_review_packet",
                   return_value={"saved": {"template": saved}, "exit_code": 0}) as prepare:
            code, output, errors = self.invoke(["review", str(self.root), "--prepare"])
        self.assertEqual(code, 0)
        self.assertFalse(errors)
        self.assertIn(f"Saved: {saved}", output)
        prepare.assert_called_once_with(self.root)
        with patch("benchmarks.evaluators.review.prepare_review_packet",
                   side_effect=ReviewSaveError("private-save-marker", saved={"template": saved})):
            code, _, errors = self.invoke(["review", str(self.root), "--prepare"])
        self.assertEqual(code, 2)
        self.assertIn(f"Partial artifact: {saved}", errors)
        self.assertIn("评审产物未完整保存", errors)
        self.assertIn("已保存的执行结果和评审产物保留", errors)
        self.assertIn("benchmarks review <运行目录> --prepare", errors)
        self.assertNotIn("private-save-marker", errors)
        with patch("benchmarks.evaluators.review.import_review",
                   side_effect=ReviewSaveError("private-import-marker", saved={"template": saved})):
            code, _, errors = self.invoke(["review", str(self.root), "--file", str(self.root / "filled.json")])
        self.assertEqual(code, 2)
        self.assertIn("--file <填写文件>", errors)
        self.assertNotIn("--prepare", errors)
        self.assertNotIn("private-import-marker", errors)

    def test_unexpected_exception_text_is_not_disclosed(self):
        with patch.object(cli, "run_benchmark", side_effect=ValueError("private-exception-marker")):
            code, _, errors = self.invoke(["run", "wechat-reporting"])
        self.assertEqual(code, 2)
        self.assertIn("ValueError", errors)
        self.assertNotIn("private-exception-marker", errors)


if __name__ == "__main__":
    unittest.main()
