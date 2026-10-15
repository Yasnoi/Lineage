"""Keep concise CLI output separate from execution and persistence facts."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from lineage_agent.agent.sessions import SessionTurn
from lineage_agent.config.settings import ModelSettings
from lineage_agent.entrypoints import cli


class CliPresentationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="cli-presentation-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.log = self.root / "run.jsonl"
        self.log.write_text('{"event":"finished"}\n', encoding="utf-8")
        self.settings = ModelSettings(base_url="https://model.invalid", model_name="controlled")
        self.state = {
            "run_id": str(uuid4()), "stop_reason": "completed", "result": {"text": "模型正文"},
            "error": None, "error_code": None, "log_path": str(self.log), "log_complete": True,
            "tool_outcomes": {"succeeded": 1, "rejected": 2, "failed": 3, "skipped": 4},
            "model_calls": 5, "summary_calls": 6, "active_context": {"version_id": str(uuid4())},
            "context_budget": {
                "method": "utf8_json_opaque_v1", "step_id": "step-0002", "decision": "allow",
                "request_sent": True, "transparent_tokens": 100, "opaque_risk_tokens": 20,
                "opaque_items": 2, "output_reserve_tokens": 30, "safety_margin_tokens": 40,
                "required_tokens": 190, "limit_tokens": 200, "trigger_tokens": 200, "trigger_percent": 100,
            },
        }

    def display_run(self, state=None):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.display_run(self.state if state is None else state, self.settings)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_completed_summary_keeps_counts_and_budget_without_redundant_ids(self):
        before = deepcopy(self.state)
        code, out, err = self.display_run()
        self.assertEqual(code, 0)
        self.assertEqual(out, "模型正文\n")
        self.assertIn("工具处理：成功 1，拒绝 2，失败 3，未执行 4", err)
        self.assertIn("模型调用：业务 5，摘要 6", err)
        self.assertIn("上下文预算（本地估算；非端点精确 Token）：放行，请求已发起，步骤 step-0002", err)
        self.assertIn("透明内容 100 + opaque 风险预留 20（2 项） + 输出预留 30 + 安全余量 40", err)
        self.assertIn("所需 190／上限 200", err)
        for omitted in (self.state["run_id"], self.state["active_context"]["version_id"],
                        str(self.log), "运行日志", "触发阈值", "100%", "模型正文"):
            self.assertNotIn(omitted, err)
        self.assertEqual(self.state, before)

    def test_failures_and_budget_termination_keep_readable_log_diagnostics(self):
        for reason, expected in (("error", 1), ("max_model_calls", 3), ("context_budget_exceeded", 4)):
            with self.subTest(reason=reason):
                state = {**self.state, "stop_reason": reason, "error": "受控失败"}
                code, out, err = self.display_run(state)
                self.assertEqual(code, expected)
                self.assertEqual(out, "")
                self.assertIn("运行日志（记录完整；文件可读取）", err)
                self.assertIn(str(self.log), err)

    def test_incomplete_log_and_missing_file_are_reported_independently(self):
        state = {**self.state, "stop_reason": "error", "log_complete": False, "error": "记录失败"}
        _, _, err = self.display_run(state)
        self.assertIn("记录不完整；文件可读取", err)
        self.log.unlink()
        _, _, err = self.display_run(state)
        self.assertIn("记录不完整；文件不可读取", err)
        _, _, err = self.display_run({**state, "log_path": None})
        self.assertIn("未记录可用路径", err)

    def test_published_turn_omits_repeated_session_identity_and_log(self):
        turn = SessionTurn("session-1", self.state, "checkpoint-1", True, None, "snapshot-1", "published")
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            cli.display_turn(turn, self.settings)
        self.assertIn("对话快照已发布：snapshot-1", err.getvalue())
        self.assertNotIn("会话标识", err.getvalue())
        self.assertNotIn(str(self.log), err.getvalue())
        self.assertEqual(out.getvalue(), "模型正文\n")

    def test_unconfirmed_snapshot_shows_log_without_claiming_publication(self):
        for status in ("not_published", "unknown"):
            with self.subTest(status=status):
                turn = SessionTurn("session-1", self.state, None, False, "checkpoint_error", None, status)
                out, err = StringIO(), StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    cli.display_turn(turn, self.settings)
                diagnostic = err.getvalue()
                self.assertEqual(diagnostic.count("运行日志"), 1)
                self.assertIn("记录完整；文件可读取", diagnostic)
                self.assertNotIn("对话快照已发布", diagnostic)
                self.assertIn("发布结果不确定" if status == "unknown" else "状态保存未确认", diagnostic)
                self.assertEqual(out.getvalue(), "模型正文\n")

    def test_run_failure_and_save_failure_do_not_duplicate_log_path(self):
        state = {**self.state, "stop_reason": "error", "error": "运行失败"}
        turn = SessionTurn("session-1", state, None, False, "checkpoint_error")
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            cli.display_turn(turn, self.settings)
        self.assertEqual(err.getvalue().count(str(self.log)), 1)


if __name__ == "__main__":
    unittest.main()
