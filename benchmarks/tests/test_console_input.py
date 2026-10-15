"""Exercise actual prompt buffers with controlled keys, not terminal rendering."""

import asyncio
from contextlib import contextmanager, redirect_stderr
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, Mock, patch

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lineage_agent.entrypoints.console_input import ConsoleInput, ConsoleInputError, _terminal_available


class ConsoleInputTests(unittest.IsolatedAsyncioTestCase):
    async def edit(self, text):
        with create_pipe_input() as source, redirect_stderr(StringIO()):
            with ConsoleInput(input=source, output=DummyOutput(), terminal=True) as reader:
                source.send_text(text)
                return await asyncio.wait_for(reader.read("[会话] > "), 2)

    async def test_enter_submits_without_stripping_and_tab_inserts_indent(self):
        self.assertEqual(await self.edit("  中文 English\t尾部  \r"), "  中文 English\t尾部  ")

    async def test_escape_enter_inserts_real_newline_then_enter_submits(self):
        self.assertEqual(await self.edit("第一行\x1b\r  第二行\r"), "第一行\n  第二行")

    async def test_lf_remains_submission(self):
        self.assertEqual(await self.edit("text\n"), "text")

    async def test_multiline_paste_waits_for_explicit_submission(self):
        text = "  中文\r\n    /exit\n/open old\n" + "边界" * 1300
        expected = text.replace("\r\n", "\n")
        with create_pipe_input() as source, redirect_stderr(StringIO()), \
                ConsoleInput(input=source, output=DummyOutput(), terminal=True) as reader:
            pending = asyncio.create_task(reader.read("[会话] > "))
            source.send_text("\x1b[200~" + text[:1000])
            await asyncio.sleep(0.02)
            self.assertFalse(pending.done())
            source.send_text(text[1000:] + "\x1b[201~")
            await asyncio.sleep(0.02)
            self.assertFalse(pending.done())
            self.assertEqual(reader._session.default_buffer.text, expected)
            source.send_text("\r")
            self.assertEqual(await asyncio.wait_for(pending, 2), expected)

    async def test_cursor_and_delete_across_logical_lines_preserve_chinese(self):
        self.assertEqual(await self.edit("中文\x1b\rabc\x1b[D\x7fZ\r"), "中文\naZc")

    async def test_escape_prefix_survives_pause_before_enter(self):
        with create_pipe_input() as source, redirect_stderr(StringIO()), \
                ConsoleInput(input=source, output=DummyOutput(), terminal=True) as reader:
            pending = asyncio.create_task(reader.read("> "))
            source.send_text("甲\x1b")
            # Let the real prompt flush an isolated Escape after its VT timeout.
            await asyncio.sleep(0.6)
            self.assertFalse(pending.done())
            source.send_text("\r乙\r")
            self.assertEqual(await asyncio.wait_for(pending, 2), "甲\n乙")

    @unittest.skipUnless(sys.platform == "win32", "Windows console EOF chord")
    async def test_ctrl_z_enter_is_eof_without_submission(self):
        with self.assertRaises(EOFError):
            await self.edit("\x1a\r")

    async def test_pasted_ctrl_z_is_text_not_eof(self):
        self.assertEqual(await self.edit("\x1b[200~\x1a\x1b[201~\r"), "\x1a")

    async def test_confirm_rejects_multiline_even_if_stripped_copy_is_yes(self):
        for value in ("yes\n", "\nyes", "yes\r\n", "YES", "y", ""):
            with ConsoleInput(terminal=False) as reader, patch("builtins.input", return_value=value), \
                    redirect_stderr(StringIO()):
                self.assertFalse(await reader.confirm("confirm: "))
        with ConsoleInput(terminal=False) as reader, patch("builtins.input", return_value=" yes "), \
                redirect_stderr(StringIO()):
            self.assertTrue(await reader.confirm("confirm: "))

    async def test_confirm_eof_is_cancel_but_interrupt_propagates(self):
        with ConsoleInput(terminal=False) as reader, redirect_stderr(StringIO()):
            with patch("builtins.input", side_effect=EOFError):
                self.assertFalse(await reader.confirm("confirm: "))
            with patch("builtins.input", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                await reader.confirm("confirm: ")

    async def test_nonterminal_read_uses_original_line_reader(self):
        with ConsoleInput(terminal=False) as reader, patch("builtins.input", return_value="  a  ") as read, \
                patch.object(reader, "_editor", side_effect=AssertionError("no editor")), \
                redirect_stderr(StringIO()) as output:
            self.assertEqual(await reader.read("[saved] > "), "  a  ")
            self.assertEqual(output.getvalue(), "[saved] > ")
            read.assert_called_once_with()

    async def test_cancelled_prompt_restores_input_modes(self):
        with create_pipe_input() as source, redirect_stderr(StringIO()), \
                ConsoleInput(input=source, output=DummyOutput(), terminal=True) as reader:
            raw_mode = source.raw_mode
            entered = asyncio.Event()
            exited = False

            @contextmanager
            def tracked_raw_mode():
                nonlocal exited
                with raw_mode():
                    entered.set()
                    try:
                        yield
                    finally:
                        exited = True

            with patch.object(source, "raw_mode", tracked_raw_mode):
                pending = asyncio.create_task(reader.read("> "))
                await asyncio.wait_for(entered.wait(), 2)
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
            self.assertTrue(exited)

    async def test_owned_handle_cleanup_preserves_interruption(self):
        with create_pipe_input() as source, redirect_stderr(StringIO()) as errors:
            reader = ConsoleInput(input=source, output=DummyOutput(), terminal=True)
            reader._owns_input = True
            with patch.object(source, "close", side_effect=OSError("close failed")), \
                    self.assertRaises(KeyboardInterrupt):
                with reader:
                    raise KeyboardInterrupt
            self.assertIn("终端输入清理失败", errors.getvalue())

    async def test_input_io_error_is_a_cli_error_not_a_submitted_buffer(self):
        with create_pipe_input() as source, redirect_stderr(StringIO()), \
                ConsoleInput(input=source, output=DummyOutput(), terminal=True) as reader:
            with patch.object(reader._editor(), "prompt_async",
                              AsyncMock(side_effect=OSError("read failed"))), \
                    self.assertRaises(ConsoleInputError):
                await reader.read("> ")

    async def test_closed_reader_cannot_reopen_and_borrowed_input_is_not_closed(self):
        with create_pipe_input() as source:
            with ConsoleInput(input=source, output=DummyOutput(), terminal=True) as reader:
                pass
            self.assertFalse(source.closed)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                await reader.read("> ")

    async def test_edited_multiline_creates_one_run_and_survives_session_reopen(self):
        from benchmarks.checks.support.runtime import LocalClient, SETTINGS, message, response
        from lineage_agent.agent import runtime
        from lineage_agent.agent.sessions import SessionManager
        from lineage_agent.entrypoints import cli

        text = "  保存中文上下文\n    /exit\n/open remains text  "
        with TemporaryDirectory(prefix="edited-conversation-") as directory:
            root = Path(directory)
            settings = replace(SETTINGS, application_root=root, log_dir=root / "logs")
            first = LocalClient([response(message("saved"))])
            with create_pipe_input() as source, redirect_stderr(StringIO()), redirect_stdout(StringIO()), \
                    ConsoleInput(input=source, output=DummyOutput(), terminal=True) as reader, \
                    SessionManager() as manager:
                source.send_text("\x1b[200~" + text + "\x1b[201~\r/exit\r")
                with patch.object(runtime, "create_client", return_value=first), \
                        patch.object(cli, "display_turn", wraps=cli.display_turn) as displayed:
                    code = await cli._interactive_loop(settings, {}, manager, input_reader=reader)
                self.assertEqual(code, 0)
                self.assertEqual(displayed.call_count, 1)
                turn = displayed.call_args.args[0]
                self.assertEqual(turn.snapshot_status, "published")
                self.assertEqual(len(first.requests), 1)
                self.assertIn({"role": "user", "content": text}, first.requests[0]["input"])
            with SessionManager() as restored:
                with patch.object(runtime, "create_client", side_effect=AssertionError("open called model")):
                    opened = await restored.open_session(turn.session_id, application_root=root)
                self.assertEqual(opened.snapshot_id, turn.snapshot_id)
                second = LocalClient([response(message("continued"))])
                with patch.object(runtime, "create_client", return_value=second):
                    continued = await restored.run_turn(turn.session_id, "继续")
                self.assertEqual(continued.snapshot_status, "published")
                self.assertNotEqual(continued.state["run_id"], turn.state["run_id"])
                self.assertIn({"role": "user", "content": text}, second.requests[0]["input"])


class TerminalDetectionTests(unittest.TestCase):
    def test_all_three_streams_must_be_ttys(self):
        for redirected in (None, "stdin", "stdout", "stderr"):
            streams = {name: Mock(isatty=Mock(return_value=name != redirected))
                       for name in ("stdin", "stdout", "stderr")}
            with patch.multiple(sys, **streams):
                self.assertEqual(_terminal_available(), redirected is None)

    def test_real_ctrl_c_key_propagates_interrupt(self):
        async def run():
            with create_pipe_input() as source, redirect_stderr(StringIO()), \
                    ConsoleInput(input=source, output=DummyOutput(), terminal=True) as reader:
                source.send_text("unsubmitted\x03")
                await reader.read("> ")
        with self.assertRaises(KeyboardInterrupt):
            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
