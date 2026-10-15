"""Windows Terminal keyboard negotiation, isolated from the CLI and agent.

This adapter deliberately targets prompt_toolkit 3.0.53's Win32 VT reader.
Only the parser is replaced; the component still owns console handles, raw
mode, bracketed paste, cursor movement and rendering. Import this module only
on Windows.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import re
from typing import TextIO

from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.input.ansi_escape_sequences import REVERSE_ANSI_SEQUENCES
from prompt_toolkit.input.win32 import ConsoleInputReader, Vt100ConsoleInputReader, Win32Input
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import Output
from prompt_toolkit.win32_types import KEY_EVENT_RECORD


_QUERY = "\x1b[?9001$p"
_ENABLE = "\x1b[?9001h"
_DISABLE = "\x1b[?9001l"
_REPORT = re.compile(r"\x1b\[\?9001;([0-4])\$y\Z")
_KEY = re.compile(r"\x1b\[([0-9;]*)_\Z")
_PASTE_START = "\x1b[200~"
_PASTE_END = "\x1b[201~"


class WindowsKeyParser:
    """Decode enhanced key frames while retaining the regular VT parser.

    Microsoft win32-input-mode sends KEY_EVENT_RECORD fields as
    CSI Vk;Sc;Uc;Kd;Cs;Rc _. Paste is an ordinary bracketed VT stream and must
    bypass this decoder, even if its contents resemble reports or key frames.
    """

    def __init__(self, callback: Callable[[KeyPress], None]) -> None:
        self._callback = callback
        self._vt = Vt100Parser(self._emit)
        # The conversion method does not access an OS handle. Reuse the pinned
        # library's modifier and navigation mapping without opening CONIN$.
        self._converter = object.__new__(ConsoleInputReader)
        self._prefix = ""
        self._paste_tail: str | None = None
        self._high_surrogate: str | None = None
        self.reports: deque[int] = deque(maxlen=16)
        self.on_report: Callable[[], None] | None = None

    def _emit(self, key: KeyPress) -> None:
        if key.key == Keys.BracketedPaste:
            # Console UTF-16 input may split non-BMP characters into two events.
            text = key.data.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
            key = KeyPress(Keys.BracketedPaste, text)
        text_key = not isinstance(key.key, Keys)
        if self._high_surrogate is not None:
            high, self._high_surrogate = self._high_surrogate, None
            if text_key and "\udc00" <= key.key <= "\udfff":
                text = (high + key.key).encode("utf-16-le", "surrogatepass").decode("utf-16-le")
                self._callback(KeyPress(text, text))
                return
            self._callback(KeyPress("\ufffd", "\ufffd"))
        if text_key and "\ud800" <= key.key <= "\udbff":
            self._high_surrogate = key.key
        elif text_key and "\udc00" <= key.key <= "\udfff":
            self._callback(KeyPress("\ufffd", "\ufffd"))
        else:
            self._callback(key)

    def feed(self, data: str) -> None:
        for char in data:
            if self._paste_tail is not None:
                self._vt.feed(char)
                self._paste_tail = (self._paste_tail + char)[-len(_PASTE_END):]
                if self._paste_tail == _PASTE_END:
                    self._paste_tail = None
                continue
            if not self._prefix:
                if char == "\x1b":
                    self._prefix = char
                else:
                    self._vt.feed(char)
                continue
            if char == "\x1b":
                self._vt.feed(self._prefix)
                self._prefix = char
                continue
            self._prefix += char
            prefix = self._prefix
            if len(prefix) > 96:
                raise ValueError("Windows 输入协议帧过长。")
            if prefix == "\x1b[" or prefix == "\x1b":
                continue
            if prefix == _PASTE_START:
                self._prefix = ""
                self._paste_tail = ""
                self._vt.feed(prefix)
                continue
            report = _REPORT.fullmatch(prefix)
            if report:
                self._prefix = ""
                self.reports.append(int(report[1]))
                if self.on_report is not None:
                    self.on_report()
                continue
            frame = _KEY.fullmatch(prefix)
            if frame:
                self._prefix = ""
                self._key_frame(frame[1])
                continue
            # Hold only potential enhanced frames/reports. Every other escape
            # sequence (including ordinary navigation and CPR) belongs to PTK.
            if re.fullmatch(r"\x1b\[[0-9;]*", prefix):
                continue
            if "\x1b[?9001;".startswith(prefix) or re.fullmatch(r"\x1b\[\?9001;[0-4]\$?", prefix):
                continue
            self._prefix = ""
            self._vt.feed(prefix)

    def _key_frame(self, fields: str) -> None:
        values = fields.split(";")
        if len(values) > 6 or any(len(value) > 10 for value in values):
            raise ValueError("Windows 输入协议字段无效。")
        defaults = [0, 0, 0, 0, 0, 1]
        for index, value in enumerate(values):
            if value:
                defaults[index] = int(value)
        vk, scan, char, down, modifiers, repeat = defaults
        if (max(vk, scan, char, repeat) > 65535 or modifiers > 0xFFFFFFFF or down not in (0, 1)):
            raise ValueError("Windows 输入协议字段超出范围。")
        if not down:
            return
        event = KEY_EVENT_RECORD()
        event.KeyDown, event.RepeatCount = True, repeat
        event.VirtualKeyCode, event.VirtualScanCode = vk, scan
        event.uChar.UnicodeChar, event.ControlKeyState = chr(char), modifiers
        if vk == 13 and modifiers & ConsoleInputReader.SHIFT_PRESSED:
            # Use the unconditional newline binding, including keys buffered
            # while an enable-confirmation query later times out.
            keys = [KeyPress(Keys.Escape, ""), KeyPress(Keys.ControlM, "\r")]
        elif char == 10 and vk != 13:
            keys = [KeyPress(Keys.ControlM, "\r")]
        else:
            keys = self._converter._event_to_key_presses(event)
            # PTK normally treats physical Ctrl+J as Enter.
            keys = [KeyPress(Keys.ControlM, "\r") if key.key == Keys.ControlJ else key for key in keys]
        for _ in range(repeat):
            for key in keys:
                # Feeding the VT parser also joins separate Escape + Enter
                # events into the normal PTK key sequence used by the fallback.
                # Use the mapped key, not its original character (Shift+Tab,
                # for example, still carries a plain tab in native input).
                data = REVERSE_ANSI_SEQUENCES.get(key.key, key.data)
                if data:
                    self._vt.feed(data)
                else:
                    self._emit(key)

    def flush(self) -> None:
        # A split report/frame must not leak into the user buffer on a timeout.
        # A bare Escape still needs PTK's usual escape-key timeout handling.
        if self._prefix.startswith("\x1b["):
            return
        if self._prefix:
            prefix, self._prefix = self._prefix, ""
            self._vt.feed(prefix)
        self._vt.flush()


class EnhancedWin32Input(Win32Input):
    """Win32Input with scoped, verified enhanced-key mode.

    The caller holds ``raw_mode()`` over prepare, prompt_async and restore.
    ``restore`` belongs in finally, including when preparation itself fails.
    """

    def __init__(self, stdin: TextIO | None = None) -> None:
        super().__init__(stdin)
        self._pending: list[KeyPress] = []
        self._parser: WindowsKeyParser | None = None
        self._changed = False
        self._paste_enabled = False
        self._probe_disabled = False
        self._failed = False
        if isinstance(self.console_input_reader, Vt100ConsoleInputReader):
            reader = self.console_input_reader
            self._parser = WindowsKeyParser(lambda key: reader._buffer.append(key))
            reader._vt100_parser = self._parser

    @contextmanager
    def attach(self, input_ready_callback: Callable[[], None]) -> Iterator[None]:
        with super().attach(input_ready_callback):
            # Preparation consumed OS events. Without a wakeup, already typed
            # text/Enter would wait indefinitely for another physical key.
            pending = None
            if self._pending:
                pending = asyncio.get_running_loop().call_soon(input_ready_callback)
            try:
                yield
            finally:
                if pending is not None:
                    pending.cancel()

    def read_keys(self) -> list[KeyPress]:
        if self._failed:
            return []
        pending, self._pending = self._pending, []
        try:
            return pending + super().read_keys()
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
        self._pending.clear()
        # PTK invokes this from an event-loop callback. Merely raising there
        # logs the exception but leaves its prompt future pending forever.
        app = get_app_or_none()
        if app is None or not app.is_running:
            raise exc
        app.exit(exception=exc)
        return []

    async def prepare(self, output: Output, timeout: float = 0.2) -> bool:
        parser = self._parser
        if parser is None:
            return False
        # Cover the negotiation interval too, before Application enables paste.
        self._paste_enabled = True
        output.enable_bracketed_paste()
        output.flush()
        # DECRPM has no request identity. After a timeout, a delayed "set"
        # report must never be mistaken for a future prompt's initial mode.
        if self._probe_disabled:
            return False
        arrived = asyncio.Event()
        parser.on_report = arrived.set
        probe_error: Exception | None = None

        def collect() -> None:
            nonlocal probe_error
            try:
                self._pending.extend(self.console_input_reader.read())
            except Exception as exc:
                probe_error = exc
                arrived.set()

        async def query() -> int | None:
            parser.reports.clear()
            arrived.clear()
            output.write_raw(_QUERY)
            output.flush()
            try:
                await asyncio.wait_for(arrived.wait(), timeout)
            except TimeoutError:
                self._probe_disabled = True
                return None
            if probe_error is not None:
                raise probe_error
            return parser.reports.popleft()

        try:
            with super().attach(collect):
                initial = await query()
                if initial in (1, 3):  # Already set (or permanently set).
                    return True
                if initial != 2:
                    return False
                # Set this before writing: even a failing flush may have sent it.
                self._changed = True
                output.write_raw(_ENABLE)
                output.flush()
                if await query() in (1, 3):
                    return True
                self._restore_keyboard(output)
                return False
        finally:
            parser.on_report = None

    def _restore_keyboard(self, output: Output) -> None:
        if self._changed:
            output.write_raw(_DISABLE)
            output.flush()
            self._changed = False

    def restore(self, output: Output) -> None:
        try:
            self._restore_keyboard(output)
        finally:
            if self._paste_enabled:
                output.disable_bracketed_paste()
                output.flush()
                self._paste_enabled = False
