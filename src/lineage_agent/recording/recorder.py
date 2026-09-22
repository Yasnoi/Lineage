"""Write independent JSONL files without sharing mutable run context."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4


LOG_FAILURE_MESSAGE = "运行日志无法完整写入，已停止后续调用。"


class RunRecorder:
    """One run's writer; callers stop execution when an operation returns False."""

    def __init__(self, log_dir: Path) -> None:
        self.run_id = str(uuid4())
        self.session_id: str | None = None
        self.path = (Path(log_dir) / f"{self.run_id}.jsonl").absolute()
        self.failed = False
        self.failure_stage: str | None = None
        self._event_seq = 0
        self._step_seq = 0
        self._stream: TextIO | None = None
        self._closed = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self._open()
        except (OSError, ValueError):
            self._fail("create")

    def _open(self) -> TextIO:
        """Keep the file boundary small enough to exercise I/O failures."""
        return self.path.open("x", encoding="utf-8", newline="\n")

    def _fail(self, stage: str) -> None:
        self.failed = True
        if self.failure_stage is None:
            self.failure_stage = stage

    def next_step(self) -> str:
        self._step_seq += 1
        return f"step-{self._step_seq:04d}"

    def bind_session(self, session_id: str) -> None:
        """Attach the application's session identity without recording history."""
        self.session_id = session_id

    def emit(
        self,
        event: str,
        *,
        step_id: str | None = None,
        node: str | None = None,
        **details: Any,
    ) -> bool:
        if self.failed:
            return False
        if self._closed or self._stream is None:
            self._fail("write")
            return False
        self._event_seq += 1
        item = {
            **details,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event_seq": self._event_seq,
            "event": event,
            "run_id": self.run_id,
        }
        if step_id is not None:
            item["step_id"] = step_id
        if self.session_id is not None:
            item["session_id"] = self.session_id
        if node is not None:
            item["node"] = node
        try:
            self._stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            self._stream.flush()
        except (OSError, ValueError, TypeError, OverflowError, RecursionError):
            self._fail("write")
            return False
        return True

    def close(self) -> bool:
        if not self._closed:
            try:
                if self._stream is not None:
                    self._stream.close()
            except (OSError, ValueError):
                self._fail("close")
            finally:
                self._closed = True
        return not self.failed
