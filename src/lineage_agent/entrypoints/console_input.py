"""CLI-owned editing buffers; submitted text and sessions belong to the caller."""

from __future__ import annotations

import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.history import DummyHistory
from prompt_toolkit.input import Input, create_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import Output, create_output


def _terminal_available() -> bool:
    # Windows output sizing uses STD_OUTPUT_HANDLE even when rendering to stderr.
    # Keep all redirected modes on the existing line-oriented input path.
    return all(stream is not None and stream.isatty()
               for stream in (sys.stdin, sys.stdout, sys.stderr))


class ConsoleInputError(RuntimeError):
    """An input failure that must not submit the current editing buffer."""


def _report_cleanup_error() -> None:
    try:
        print("终端输入清理失败；请检查终端模式。", file=sys.stderr)
    except (OSError, ValueError):
        # A broken diagnostic stream must not replace Ctrl+C/EOF/cancellation.
        pass


class ConsoleInput:
    """One CLI's editor, created lazily so one-shot calls need no terminal."""

    def __init__(self, *, input: Input | None = None, output: Output | None = None,
                 terminal: bool | None = None) -> None:
        self._input = input
        self._output = output
        self._terminal = terminal
        self._owns_input = input is None
        self._session: PromptSession[str] | None = None
        self._closed = False
        self._enhanced = False
        self._hint: bool | None = None
        self._eof_pending = False

    def __enter__(self) -> ConsoleInput:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.close()
        except ConsoleInputError:
            if exc is None:
                raise
            _report_cleanup_error()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._owns_input and self._input is not None:
                try:
                    self._input.close()
                except Exception as exc:
                    raise ConsoleInputError("终端输入句柄未能关闭。") from exc

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("enter")
        def submit(event):
            if self._eof_pending and not event.current_buffer.text:
                event.app.exit(exception=EOFError())
            else:
                event.current_buffer.validate_and_handle()

        @bindings.add("escape", "enter")
        def newline(event):
            event.current_buffer.insert_text("\n")

        @bindings.add("c-j")
        def line_feed(event):
            # Decoded Shift+Enter uses the same sequence as the fallback binding;
            # its meaning must not change if negotiation subsequently times out.
            submit(event)

        @bindings.add("tab")
        def indent(event):
            event.current_buffer.insert_text("\t")

        if sys.platform == "win32":
            @bindings.add("c-z")
            def windows_eof(event):
                # EOF is a keyboard action, not a special value in pasted text.
                if not event.current_buffer.text:
                    self._eof_pending = True

        return bindings

    def _editor(self) -> PromptSession[str]:
        if self._session is None:
            if self._input is None:
                if sys.platform == "win32":
                    from ._windows_input import EnhancedWin32Input
                    self._input = EnhancedWin32Input(sys.stdin)
                else:
                    self._input = create_input(stdin=sys.stdin)
            if self._output is None:
                self._output = create_output(stdout=sys.stderr)
            self._session = PromptSession(
                input=self._input, output=self._output, multiline=True, wrap_lines=True,
                history=DummyHistory(), key_bindings=self._bindings(),
                prompt_continuation="... ", complete_while_typing=False,
                enable_suspend=False, enable_system_prompt=False, enable_open_in_editor=False,
            )
            # Esc is a prefix, not a standalone action here. Allow a deliberate
            # pause between Esc and Enter instead of silently submitting text.
            self._session.app.timeoutlen = None
            self._session.default_buffer.on_text_changed += self._text_changed
        return self._session

    def _text_changed(self, _: object) -> None:
        self._eof_pending = False

    async def read(self, prompt: str) -> str:
        """Read one whole message; EOF and interruption retain their exceptions."""
        if self._closed:
            raise RuntimeError("Input reader is closed")
        terminal = _terminal_available() if self._terminal is None else self._terminal
        if not terminal:
            print(prompt, end="", file=sys.stderr, flush=True)
            return input()

        session = self._editor()
        source, output = self._input, self._output
        self._eof_pending = False
        self._enhanced = False
        prepare = getattr(source, "prepare", None)
        restore = getattr(source, "restore", None)
        # The outer raw scope also covers negotiation; PromptSession's nested
        # raw scope restores to it before we restore the original console mode.
        with source.raw_mode():
            primary_error = False
            try:
                if prepare is not None:
                    self._enhanced = await prepare(output)
                if self._hint is None or self._hint != self._enhanced:
                    self._hint = self._enhanced
                    newline = ("Shift+Enter 换行；也可先按 Esc 再按 Enter"
                               if self._enhanced else
                               "未启用增强按键；换行请先按 Esc 再按 Enter")
                    print(f"输入：Enter 提交；{newline}。", file=sys.stderr)
                return await session.prompt_async(prompt)
            except (OSError, ValueError) as exc:
                primary_error = True
                raise ConsoleInputError("终端输入读取失败；本条输入未提交。") from exc
            except BaseException:
                primary_error = True
                raise
            finally:
                try:
                    if restore is not None:
                        restore(output)
                except Exception as exc:
                    if not primary_error:
                        raise ConsoleInputError("终端输入模式未能恢复；本条输入未提交。") from exc
                    _report_cleanup_error()
                finally:
                    self._enhanced = False

    async def confirm(self, prompt: str) -> bool:
        """Recovery confirmation never accepts a multiline paste as consent."""
        try:
            text = await self.read(prompt)
        except EOFError:
            return False
        return "\n" not in text and "\r" not in text and text.strip() == "yes"
