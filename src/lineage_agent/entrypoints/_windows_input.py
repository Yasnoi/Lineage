"""Windows input compatibility for prompt_toolkit 3.0.53.

The standard Win32 input and VT parser own keyboard decoding and console modes.
This adapter only joins UTF-16 characters and forwards input failures to the
active prompt. Import this module only on Windows.
"""

from __future__ import annotations

from typing import TextIO

from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.input.win32 import Vt100ConsoleInputReader, Win32Input
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys


class WindowsInput(Win32Input):
    """Keep Unicode input intact and stop the prompt on reader failures."""

    def __init__(self, stdin: TextIO | None = None) -> None:
        super().__init__(stdin)
        self._failed = False
        self._high_surrogate: str | None = None
        if isinstance(self.console_input_reader, Vt100ConsoleInputReader):
            self.console_input_reader._vt100_parser = Vt100Parser(self._emit_key)

    def _emit_key(self, key: KeyPress) -> None:
        # The reader replaces its buffer after each read or flush.
        callback = self.console_input_reader._buffer.append
        if key.key == Keys.BracketedPaste:
            text = key.data.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
            key = KeyPress(Keys.BracketedPaste, text)
        text_key = not isinstance(key.key, Keys)
        if self._high_surrogate is not None:
            high, self._high_surrogate = self._high_surrogate, None
            if text_key and "\udc00" <= key.key <= "\udfff":
                text = (high + key.key).encode("utf-16-le", "surrogatepass").decode("utf-16-le")
                callback(KeyPress(text, text))
                return
            callback(KeyPress("\ufffd", "\ufffd"))
        if text_key and "\ud800" <= key.key <= "\udbff":
            self._high_surrogate = key.key
        elif text_key and "\udc00" <= key.key <= "\udfff":
            callback(KeyPress("\ufffd", "\ufffd"))
        else:
            callback(key)

    def read_keys(self) -> list[KeyPress]:
        if self._failed:
            return []
        try:
            return super().read_keys()
        except Exception as exc:
            return self._read_failed(exc)

    def flush_keys(self) -> list[KeyPress]:
        if self._failed:
            return []
        try:
            return super().flush_keys()
        except Exception as exc:
            return self._read_failed(exc)

    def _read_failed(self, exc: Exception) -> list[KeyPress]:
        self._failed = True
        # PTK invokes this from an event-loop callback. Merely raising there
        # logs the exception but leaves its prompt future pending forever.
        app = get_app_or_none()
        if app is None or not app.is_running:
            raise exc
        app.exit(exception=exc)
        return []
