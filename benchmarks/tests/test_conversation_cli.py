"""Conversation discovery/open commands do not require a current model setup."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from lineage_agent.agent.sessions import SessionError, SessionOpened
from lineage_agent.archive import ArchiveError
from lineage_agent.config.settings import ConfigurationError, ModelSettings
from lineage_agent.entrypoints import cli
from lineage_agent.storage import StorageError


class _Manager:
    cleanup_error = None

    def __init__(self, settings, *, recovery=False, missing=False):
        self.saved_settings = settings
        self.recovery, self.missing = recovery, missing
        self.opens, self.runs, self.closed, self.listed = [], [], [], []
        self.created = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def create_session(self, settings, **dependencies):
        self.created += 1
        return "active"

    def list_sessions(self, root):
        self.listed.append(root)
        return [{"conversation_id": "saved", "current_snapshot_id": "snapshot-1",
                 "unfinished_runs": [{"run_id": "unfinished-1"}] if self.recovery else [],
                 "status": "recovery_required" if self.recovery else "ready"}]

    async def open_session(self, identity, **options):
        self.opens.append((identity, options))
        if self.missing:
            raise SessionError("no_snapshot")
        if self.recovery and options.get("recovery") is None:
            raise SessionError("recovery_required", details={"snapshot_id": "snapshot-1",
                                                             "unfinished_run_ids": ["unfinished-1"]})
        return SessionOpened(identity, "snapshot-1", self.saved_settings, ("沿用保存模型和额度。",))

    def close_session(self, identity):
        self.closed.append(identity)

    def settings(self, identity):
        return self.saved_settings

    async def run_turn(self, identity, prompt):
        self.runs.append((identity, prompt))
        return SimpleNamespace(state={"stop_reason": "completed"}, checkpoint_complete=True)


class ConversationCliTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="conversation-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = ModelSettings(base_url="https://saved.invalid", model_name="saved-model",
                                      application_root=self.root, log_dir=self.root / "logs")
        self.paths = {"application_root": self.root, "project_root": None, "log_dir": self.root / "logs"}

    def invoke(self, args, manager, inputs=()):
        out, err = StringIO(), StringIO()
        with patch("sys.argv", ["lineage-agent", *args]), patch("builtins.input", side_effect=inputs), \
                patch.object(cli, "SessionManager", return_value=manager), \
                patch.object(cli, "load_local_paths", return_value=self.paths) as paths, \
                patch.object(cli, "load_current_limits", return_value={}), \
                patch.object(cli, "load_settings", side_effect=AssertionError("must not load current model")), \
                patch.object(cli, "display_turn") as display, \
                redirect_stdout(out), redirect_stderr(err):
            code = cli.main()
        return code, out.getvalue(), err.getvalue(), paths, display

    def test_list_needs_no_model_and_does_not_create_or_open(self):
        manager = _Manager(self.settings)
        code, _, err, paths, _ = self.invoke(["--list", "--application-root", str(self.root)], manager)
        self.assertEqual(code, 0)
        self.assertEqual(manager.listed, [self.root])
        self.assertEqual(manager.created, 0)
        self.assertFalse(manager.opens)
        self.assertIn("saved", err)
        paths.assert_called_once_with(application_root=self.root)

    def test_empty_interactive_lists_without_new_session_or_model_config(self):
        manager = _Manager(self.settings)
        code, _, err, _, _ = self.invoke([], manager, ["/list", "/exit"])
        self.assertEqual(code, 0)
        self.assertEqual(manager.created, 0)
        self.assertIn("当前会话：未选择", err)

    def test_open_saved_model_can_submit_a_new_prompt_without_current_model_config(self):
        manager = _Manager(self.settings)
        code, _, err, _, display = self.invoke(["--open", "saved", "new question"], manager)
        self.assertEqual(code, 0)
        self.assertEqual(manager.created, 0)
        self.assertEqual(manager.runs, [("saved", "new question")])
        self.assertIsNone(manager.opens[0][1]["current_project_root"])
        self.assertIn("沿用保存模型和额度", err)
        self.assertEqual(display.call_args.args[1], self.settings)

    def test_recovery_requires_literal_yes_and_keeps_default_cancel(self):
        for answer in ("", "no", "y", "YES", EOFError()):
            manager = _Manager(self.settings, recovery=True)
            code, _, err, _, _ = self.invoke(["--open", "saved"], manager, [answer])
            self.assertEqual(code, 1)
            self.assertEqual(len(manager.opens), 1)
            self.assertFalse(manager.runs)
            self.assertIn("unfinished-1", err)
            self.assertIn("snapshot-1", err)
            self.assertIn("已取消", err)
        manager = _Manager(self.settings, recovery=True)
        code, _, _, _, _ = self.invoke(["--open", "saved"], manager, ["yes", "/exit"])
        self.assertEqual(code, 0)
        choice = manager.opens[1][1]["recovery"]
        self.assertEqual(choice.snapshot_id, "snapshot-1")
        self.assertEqual(choice.unfinished_run_ids, ("unfinished-1",))
        self.assertFalse(manager.runs)

    def test_no_snapshot_is_an_error_without_confirmation_or_empty_fallback(self):
        manager = _Manager(self.settings, missing=True)
        code, _, err, _, _ = self.invoke(["--open", "saved"], manager)
        self.assertEqual(code, 1)
        self.assertIn("no_snapshot", err)
        self.assertEqual(manager.created, 0)
        self.assertFalse(manager.runs)

    def test_cancelled_open_preserves_current_and_close_revokes_switch_target(self):
        manager = _Manager(self.settings, recovery=True)
        out, err = StringIO(), StringIO()
        commands = ["first", "/open saved", "no", "second", "/close", "/switch active",
                    "/open saved", "yes", "third", "/exit"]
        with patch("builtins.input", side_effect=commands), \
                patch.object(cli, "load_current_limits", return_value={}), \
                patch.object(cli, "display_turn"), redirect_stdout(out), redirect_stderr(err):
            result = cli.asyncio.run(cli._interactive_loop(self.settings, {}, manager, local_paths=self.paths))
        self.assertEqual(result, 0)
        self.assertEqual(manager.runs, [("active", "first"), ("active", "second"), ("saved", "third")])
        self.assertEqual(manager.closed, ["active"])
        self.assertIn("请先使用 /open", err.getvalue())

    def test_invalid_new_model_configuration_does_not_prevent_interactive_exit(self):
        manager = _Manager(self.settings)
        out, err = StringIO(), StringIO()
        with patch("sys.argv", ["lineage-agent"]), patch("builtins.input", side_effect=["/new", "/exit"]), \
                patch.object(cli, "SessionManager", return_value=manager), \
                patch.object(cli, "load_local_paths", return_value=self.paths), \
                patch.object(cli, "load_settings", side_effect=ConfigurationError("新模型配置缺失。")), \
                redirect_stdout(out), redirect_stderr(err):
            result = cli.main()
        self.assertEqual(result, 0)
        self.assertEqual(manager.created, 0)
        self.assertIn("新模型配置缺失", err.getvalue())

    def test_real_creation_failures_preserve_the_current_conversation(self):
        cases = (
            (cli.session_api.ArchiveStore, "register_conversation", "archive_write_error", ArchiveError),
            (cli.session_api.ConversationLease, "acquire", "storage_lock_error", StorageError),
            (cli.session_api, "conversation_paths", "storage_create_error", StorageError),
        )
        for owner, attribute, code, error_type in cases:
            with self.subTest(code=code), cli.SessionManager() as manager:
                original = getattr(owner, attribute)
                calls = 0

                def fail_second(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise error_type(code) from OSError("private-storage-path")
                    return original(*args, **kwargs)

                out, err = StringIO(), StringIO()
                turn = SimpleNamespace(state={"stop_reason": "completed"}, checkpoint_complete=True)
                with patch.object(owner, attribute, new=fail_second), \
                        patch.object(manager, "run_turn", new=AsyncMock(return_value=turn)) as run, \
                        patch("builtins.input", side_effect=["first", "/new", "second", "/exit"]), \
                        patch.object(cli, "display_turn"), redirect_stdout(out), redirect_stderr(err):
                    result = cli.asyncio.run(cli._interactive_loop(self.settings, {}, manager, local_paths=self.paths))
                self.assertEqual(result, 0)
                self.assertEqual(run.await_count, 2)
                first, second = run.await_args_list
                self.assertEqual(first.args[0], second.args[0])
                self.assertEqual((first.args[1], second.args[1]), ("first", "second"))
                self.assertEqual(manager.settings(first.args[0]), self.settings)
                self.assertIn(code, err.getvalue())
                self.assertNotIn("private-storage-path", err.getvalue())

    def test_initial_creation_failure_allows_retry_without_running_the_rejected_prompt(self):
        for error in (ArchiveError("archive_create_error"), StorageError("storage_create_error")):
            with self.subTest(code=error.code):
                manager = _Manager(self.settings)
                out, err = StringIO(), StringIO()
                with patch.object(manager, "create_session", side_effect=[error, "active"]), \
                        patch("builtins.input", side_effect=["rejected", "retry", "/exit"]), \
                        patch.object(cli, "display_turn"), redirect_stdout(out), redirect_stderr(err):
                    result = cli.asyncio.run(cli._interactive_loop(self.settings, {}, manager, local_paths=self.paths))
                self.assertEqual(result, 0)
                self.assertEqual(manager.runs, [("active", "retry")])
                self.assertIn(error.code, err.getvalue())


if __name__ == "__main__":
    unittest.main()
