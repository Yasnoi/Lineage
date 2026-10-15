"""Synthetic protocol tests; these do not certify Windows Terminal rendering."""

import asyncio
from collections import deque
from contextlib import contextmanager
import sys
import unittest
from unittest.mock import patch
from unittest.mock import Mock

from prompt_toolkit.keys import Keys

if sys.platform == "win32":
    from prompt_toolkit.input.win32 import Win32Input
    from lineage_agent.entrypoints._windows_input import EnhancedWin32Input, WindowsKeyParser


def frame(vk, char=0, *, modifiers=0, down=1, repeat=1):
    return f"\x1b[{vk};0;{char};{down};{modifiers};{repeat}_"


@unittest.skipUnless(sys.platform == "win32", "Windows console input adapter")
class WindowsParserTests(unittest.TestCase):
    def parse(self, source, chunk=1):
        keys = []
        parser = WindowsKeyParser(keys.append)
        for start in range(0, len(source), chunk):
            parser.feed(source[start:start + chunk])
        parser.flush()
        return parser, keys

    def test_split_enter_shift_enter_and_repeat(self):
        _, keys = self.parse(frame(13, 13) + frame(13, 13, modifiers=16, repeat=2))
        self.assertEqual([key.key for key in keys], [Keys.ControlM, Keys.Escape, Keys.ControlM,
                                                    Keys.Escape, Keys.ControlM])

    def test_keyup_ignored_and_ctrl_j_retains_normal_enter_semantics(self):
        _, keys = self.parse(frame(13, 13, down=0) + frame(74, 10, modifiers=8))
        self.assertEqual([key.key for key in keys], [Keys.ControlM])

    def test_navigation_modifiers_and_altgr(self):
        _, keys = self.parse(frame(37, modifiers=8 | 16) + frame(9, 9, modifiers=16)
                             + frame(65, ord("@"), modifiers=1 | 8))
        self.assertEqual([key.key for key in keys], [Keys.ControlShiftLeft, Keys.BackTab, "@"])

    def test_escape_enter_fallback_and_normal_vt_navigation(self):
        _, keys = self.parse(frame(27, 27) + frame(13, 13) + "\x1b[D")
        self.assertEqual([key.key for key in keys], [Keys.Escape, Keys.ControlM, Keys.Left])

    def test_mixed_vt_escape_and_enhanced_frame(self):
        _, keys = self.parse("\x1b" + frame(13, 13))
        self.assertEqual([key.key for key in keys], [Keys.Escape, Keys.ControlM])

    def test_utf16_across_frames_and_reads(self):
        _, keys = self.parse(frame(0, 0xD83D) + frame(0, 0xDE00) + "中" + "\ud83d\ude80")
        self.assertEqual("".join(key.data for key in keys), "😀中🚀")

    def test_multiline_paste_is_one_literal_event_even_with_protocol_text(self):
        content = "中文" * 1200 + "\n  /exit\n\t/open old\n" + frame(13, 13, modifiers=16) + "\x1b[?9001;1$y\ud83d\ude80"
        parser, keys = self.parse("\x1b[200~" + content + "\x1b[201~" + frame(13, 13), chunk=17)
        self.assertEqual([key.key for key in keys], [Keys.BracketedPaste, Keys.ControlM])
        self.assertEqual(keys[0].data, content[:-2] + "🚀")
        self.assertFalse(parser.reports)

    def test_late_report_does_not_become_user_text(self):
        parser, keys = self.parse("前\x1b[?9001;2$y后")
        self.assertEqual("".join(key.data for key in keys), "前后")
        self.assertEqual(list(parser.reports), [2])

    def test_partial_reports_and_frames_survive_flush(self):
        keys = []
        parser = WindowsKeyParser(keys.append)
        parser.feed("\x1b[?9001;1$")
        parser.flush()
        parser.feed("y\x1b[13;0;13;")
        parser.flush()
        self.assertFalse(keys)
        parser.feed("1;16;1_")
        parser.flush()
        self.assertEqual([key.key for key in keys], [Keys.Escape, Keys.ControlM])
        self.assertEqual(list(parser.reports), [1])

    def test_csi_frame_and_report_survive_timeout_at_every_split(self):
        for source in (frame(13, 13, modifiers=16), "\x1b[?9001;1$y"):
            for split in range(2, len(source)):
                with self.subTest(source=source, split=split):
                    keys = []
                    parser = WindowsKeyParser(keys.append)
                    parser.feed(source[:split])
                    parser.flush()
                    parser.feed(source[split:])
                    parser.flush()
                    if source.endswith("_"):
                        self.assertEqual([key.key for key in keys], [Keys.Escape, Keys.ControlM])
                    else:
                        self.assertFalse(keys)
                        self.assertEqual(list(parser.reports), [1])

    def test_malformed_enhanced_frames_fail_instead_of_submitting_partial_text(self):
        for source in (frame(13, 70000), "\x1b[1;2;3;4;5;6;7_", "\x1b[" + "1" * 100):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.parse(source)


class _Reader:
    def __init__(self):
        self.keys = []

    def read(self):
        keys, self.keys = self.keys, []
        return keys

    def flush_keys(self):
        return []


class _Output:
    def __init__(self, source, replies):
        self.source = source
        self.replies = deque(replies)
        self.writes = []
        self.callback = None
        self.fail_enable = False

    def write_raw(self, text):
        self.writes.append(text)
        if text == "\x1b[?9001h" and self.fail_enable:
            raise OSError("write failed after enable")
        if text == "\x1b[?9001$p" and self.replies:
            reply = self.replies.popleft()
            if reply is not None:
                self.send(reply)

    def send(self, text):
        self.source._parser.feed(text)
        if self.callback is not None:
            self.callback()

    def flush(self):
        pass

    def enable_bracketed_paste(self):
        self.writes.append("paste-on")

    def disable_bracketed_paste(self):
        self.writes.append("paste-off")


@unittest.skipUnless(sys.platform == "win32", "Windows console input adapter")
class WindowsNegotiationTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, replies):
        source = object.__new__(EnhancedWin32Input)
        source.console_input_reader = _Reader()
        source._pending = []
        source._changed = source._paste_enabled = False
        source._probe_disabled = False
        source._failed = False
        source._parser = WindowsKeyParser(lambda key: source.console_input_reader.keys.append(key))
        output = _Output(source, replies)

        @contextmanager
        def attached(_, callback):
            old, output.callback = output.callback, callback
            try:
                yield
            finally:
                output.callback = old

        self.enterContext(patch.object(Win32Input, "attach", attached))
        return source, output

    async def test_enable_confirm_restore_and_keep_probe_input(self):
        source, output = self.fixture(["中文\x1b[?9001;2$y", "\x1b[?9001;1$y" + frame(13, 13)])
        self.assertTrue(await source.prepare(output))
        received = []
        with source.attach(lambda: received.extend(source.read_keys())):
            await asyncio.sleep(0)
        source.restore(output)
        self.assertEqual("".join(key.data for key in received), "中文\r")
        self.assertEqual(output.writes, ["paste-on", "\x1b[?9001$p", "\x1b[?9001h", "\x1b[?9001$p",
                                         "\x1b[?9001l", "paste-off"])
        self.assertIsNone(output.callback)

    async def test_already_enabled_is_not_disabled(self):
        source, output = self.fixture(["\x1b[?9001;1$y"])
        self.assertTrue(await source.prepare(output))
        source.restore(output)
        self.assertNotIn("\x1b[?9001h", output.writes)
        self.assertNotIn("\x1b[?9001l", output.writes)

    async def test_unsupported_and_initial_timeout_use_fallback_without_mode_change(self):
        for reply in ("\x1b[?9001;0$y", "\x1b[?9001;4$y", None):
            with self.subTest(reply=reply):
                source, output = self.fixture([reply])
                self.assertFalse(await source.prepare(output, timeout=0.02))
                source.restore(output)
                self.assertNotIn("\x1b[?9001h", output.writes)
                self.assertNotIn("\x1b[?9001l", output.writes)

    async def test_failed_enable_confirmation_restores_mode_before_fallback(self):
        source, output = self.fixture(["\x1b[?9001;2$y", None])
        self.assertFalse(await source.prepare(output, timeout=0.02))
        self.assertIn("\x1b[?9001l", output.writes)
        source.restore(output)
        self.assertEqual(output.writes.count("\x1b[?9001l"), 1)
        output.send("\x1b[?9001;1$y迟到")
        self.assertEqual("".join(key.data for key in source.read_keys()), "迟到")

    async def test_write_failure_still_restores_and_disables_paste(self):
        source, output = self.fixture(["\x1b[?9001;2$y"])
        output.fail_enable = True
        with self.assertRaises(OSError):
            try:
                await source.prepare(output)
            finally:
                source.restore(output)
        self.assertIn("\x1b[?9001l", output.writes)
        self.assertEqual(output.writes[-1], "paste-off")

    async def test_timeout_disables_later_negotiation_so_late_set_cannot_enable(self):
        source, output = self.fixture(["\x1b[?9001;2$y", None])
        self.assertFalse(await source.prepare(output, timeout=0.02))
        source.restore(output)
        output.send("\x1b[?9001;1$y")
        before = output.writes.count("\x1b[?9001$p")
        self.assertFalse(await source.prepare(output))
        source.restore(output)
        self.assertEqual(output.writes.count("\x1b[?9001$p"), before)
        self.assertFalse(source.read_keys())

    async def test_shift_enter_during_failed_confirmation_remains_newline(self):
        source, output = self.fixture(["\x1b[?9001;2$y", frame(13, 13, modifiers=16)])
        self.assertFalse(await source.prepare(output, timeout=0.02))
        source.restore(output)
        self.assertEqual([key.key for key in source.read_keys()], [Keys.Escape, Keys.ControlM])

    async def test_paste_during_probe_retains_newline_and_indentation(self):
        text = "\n  中\n/exit\n"
        source, output = self.fixture(["\x1b[200~" + text + "\x1b[201~\x1b[?9001;0$y"])
        self.assertFalse(await source.prepare(output))
        source.restore(output)
        keys = source.read_keys()
        self.assertEqual([(key.key, key.data) for key in keys], [(Keys.BracketedPaste, text)])

    async def test_input_failure_exits_prompt_instead_of_stranding_callback(self):
        source, _ = self.fixture([])
        failure = ValueError("invalid input frame")
        app = Mock(is_running=True)
        with patch.object(source.console_input_reader, "read", side_effect=failure), \
                patch("lineage_agent.entrypoints._windows_input.get_app_or_none", return_value=app):
            self.assertEqual(source.read_keys(), [])
        app.exit.assert_called_once_with(exception=failure)
        self.assertFalse(source.read_keys())
        self.assertFalse(source.flush_keys())

    async def test_flush_failure_exits_prompt_and_freezes_further_reads(self):
        source, _ = self.fixture([])
        failure = OSError("flush failed")
        app = Mock(is_running=True)
        with patch.object(source.console_input_reader, "flush_keys", side_effect=failure), \
                patch("lineage_agent.entrypoints._windows_input.get_app_or_none", return_value=app):
            self.assertFalse(source.flush_keys())
        app.exit.assert_called_once_with(exception=failure)
        self.assertFalse(source.read_keys())

    async def test_read_failure_during_negotiation_propagates(self):
        source, output = self.fixture(["\x1b[?9001;2$y"])
        with patch.object(source.console_input_reader, "read", side_effect=OSError("read failed")):
            with self.assertRaises(OSError):
                try:
                    await source.prepare(output)
                finally:
                    source.restore(output)
        self.assertEqual(output.writes[-1], "paste-off")


if __name__ == "__main__":
    unittest.main()
