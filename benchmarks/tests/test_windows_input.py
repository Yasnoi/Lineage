"""Controlled Windows input checks; these do not certify terminal rendering."""

import sys
import unittest
from unittest.mock import Mock, patch

from prompt_toolkit.keys import Keys

if sys.platform == "win32":
    from prompt_toolkit.input.win32 import Vt100ConsoleInputReader, Win32Input
    from lineage_agent.entrypoints._windows_input import WindowsInput


@unittest.skipUnless(sys.platform == "win32", "Windows console input adapter")
class WindowsInputTests(unittest.TestCase):
    def make_input(self):
        reader = object.__new__(Vt100ConsoleInputReader)
        reader._buffer = []

        def initialize(source, stdin=None):
            source.console_input_reader = reader

        with patch.object(Win32Input, "__init__", initialize):
            return WindowsInput()

    def parse(self, text, chunk=1):
        source = self.make_input()
        keys = []
        parser = source.console_input_reader._vt100_parser
        for start in range(0, len(text), chunk):
            parser.feed(text[start:start + chunk])
            # Drain emitted keys without flushing an incomplete VT sequence.
            reader = source.console_input_reader
            keys.extend(reader._buffer)
            reader._buffer = []
        keys.extend(source.flush_keys())
        return keys

    def test_escape_enter_and_normal_vt_navigation(self):
        keys = self.parse("\x1b\r\x1b[D")
        self.assertEqual([key.key for key in keys], [Keys.Escape, Keys.ControlM, Keys.Left])

    def test_utf16_across_input_chunks(self):
        keys = self.parse("\ud83d\ude00中\ud83d\ude80")
        self.assertEqual("".join(key.data for key in keys), "😀中🚀")

    def test_multiline_paste_is_one_literal_event(self):
        content = "中文" * 1200 + "\n  /exit\n\t/open old\n\x1b[D\ud83d\ude80"
        keys = self.parse("\x1b[200~" + content + "\x1b[201~\r", chunk=17)
        self.assertEqual([key.key for key in keys], [Keys.BracketedPaste, Keys.ControlM])
        self.assertEqual(keys[0].data, content[:-2] + "🚀")

    def test_input_failure_exits_prompt_instead_of_stranding_callback(self):
        source = self.make_input()
        failure = OSError("read failed")
        app = Mock(is_running=True)
        with patch.object(source.console_input_reader, "read", side_effect=failure), \
                patch("lineage_agent.entrypoints._windows_input.get_app_or_none", return_value=app):
            self.assertEqual(source.read_keys(), [])
        app.exit.assert_called_once_with(exception=failure)
        self.assertFalse(source.read_keys())
        self.assertFalse(source.flush_keys())

    def test_flush_failure_exits_prompt_and_freezes_further_reads(self):
        source = self.make_input()
        failure = OSError("flush failed")
        app = Mock(is_running=True)
        with patch.object(source.console_input_reader, "flush_keys", side_effect=failure), \
                patch("lineage_agent.entrypoints._windows_input.get_app_or_none", return_value=app):
            self.assertFalse(source.flush_keys())
        app.exit.assert_called_once_with(exception=failure)
        self.assertFalse(source.read_keys())


if __name__ == "__main__":
    unittest.main()
