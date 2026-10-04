"""Run-bound diagnostics and fail-closed writes for native in-memory checkpoints."""

from typing import Any
from secrets import token_bytes

from langgraph.checkpoint.memory import InMemorySaver

from lineage_agent.recording.recorder import RunRecorder


CHECKPOINT_ERRORS = {
    "checkpoint_create_error": "会话检查点无法创建，已停止本次请求。",
    "checkpoint_read_error": "会话检查点无法读取，已停止本次请求。",
    "checkpoint_write_error": "会话检查点未能确认保存，已停止后续调用。",
    "record_create_error": "原始记录文件无法创建，未发起后续调用。",
    "record_write_error": "原始记录文件未能确认写入，已停止后续调用。",
    "record_read_error": "原始记录文件无法读取。",
    "record_integrity_error": "原始记录文件或确认范围校验失败。",
    "record_encoding_error": "原始记录包含不支持保存的值。",
}


class CheckpointError(Exception):
    """Expose only a fixed reason, never storage exceptions or state contents."""

    def __init__(self, code: str) -> None:
        self.code = code if code in CHECKPOINT_ERRORS else "checkpoint_write_error"
        super().__init__(CHECKPOINT_ERRORS[self.code])


class SessionCheckpointer(InMemorySaver):
    """In-process native saver; a failed thread cannot silently continue.

    InMemorySaver's async methods already call synchronous storage methods. Keep
    that no-await boundary explicitly: LangGraph schedules pending writes before
    its next checkpoint, whose sync durability barrier observes our fault flag.
    This is not a barrier implementation for an asynchronous disk backend.
    """

    def __init__(self) -> None:
        try:
            super().__init__()
        except Exception:
            raise CheckpointError("checkpoint_create_error") from None
        # Raised exceptions retain frames through their traceback. Keep only
        # the fixed reason across turns, and create a fresh exception per use.
        self._failures: dict[str, str] = {}
        self._confirmed: dict[str, str] = {}
        self._confirmed_ids: dict[str, set[str]] = {}
        self._recorders: dict[str, RunRecorder] = {}
        self._history_cursor_keys: dict[str, bytes] = {}
        self._history_journals: dict[str, Any] = {}

    def bind_history(self, thread_id: str, journal) -> None:
        previous = self._history_journals.get(thread_id)
        if previous is not None and previous.directory != journal.directory:
            raise CheckpointError("record_integrity_error")
        self._history_journals[thread_id] = journal

    def history_journal(self, thread_id: str):
        if thread_id not in self._history_journals:
            raise CheckpointError("record_read_error")
        return self._history_journals[thread_id]

    def bind_run(self, thread_id: str, recorder: RunRecorder) -> None:
        self._recorders[thread_id] = recorder

    def history_cursor_key(self, thread_id: str) -> bytes:
        """Keep opaque recall cursors valid for the storage scope's lifetime."""
        if thread_id not in self._history_cursor_keys:
            self._history_cursor_keys[thread_id] = token_bytes(32)
        return self._history_cursor_keys[thread_id]

    def release(self, thread_id: str) -> None:
        """Release private Run storage without affecting another session."""
        self.delete_thread(thread_id)
        self._failures.pop(thread_id, None)
        self._confirmed.pop(thread_id, None)
        self._confirmed_ids.pop(thread_id, None)
        self._recorders.pop(thread_id, None)
        self._history_cursor_keys.pop(thread_id, None)
        journal = self._history_journals.pop(thread_id, None)
        if journal is not None:
            journal.close()

    def unbind_run(self, thread_id: str) -> None:
        self._recorders.pop(thread_id, None)

    def last_confirmed(self, thread_id: str) -> str | None:
        return self._confirmed.get(thread_id)

    async def read_confirmed(self, thread_id: str, checkpoint_id: str):
        """Read an explicit immutable record, even when execution is blocked.

        This never clears the fault flag or selects a newer/unconfirmed state.
        References are issued only after successful native checkpoint writes.
        """
        try:
            if checkpoint_id not in self._confirmed_ids.get(thread_id, set()):
                raise ValueError("unconfirmed record")
            saved = self.get_tuple({"configurable": {
                "thread_id": thread_id, "checkpoint_id": checkpoint_id,
            }})
            if saved is None:
                raise ValueError("missing record")
            return saved
        except Exception:
            raise CheckpointError("checkpoint_read_error") from None

    def failure(self, thread_id: str) -> CheckpointError | None:
        code = self._failures.get(thread_id)
        return CheckpointError(code) if code is not None else None

    @staticmethod
    def _thread(config: dict[str, Any]) -> str:
        return config["configurable"]["thread_id"]

    def _healthy(self, thread_id: str) -> None:
        if failure := self.failure(thread_id):
            raise failure

    def _emit(self, thread_id: str, event: str, **details: Any) -> None:
        recorder = self._recorders.get(thread_id)
        if recorder is not None:
            recorder.emit(event, **details)

    def _failed(self, thread_id: str, code: str, stage: str) -> CheckpointError:
        safe_code = code if code in CHECKPOINT_ERRORS else "checkpoint_write_error"
        failure = CheckpointError(self._failures.setdefault(thread_id, safe_code))
        self._emit(
            thread_id, "checkpoint.failed", stage=stage,
            checkpoint_id=self.last_confirmed(thread_id),
            error_code=failure.code, message=str(failure),
        )
        return failure

    async def aget_tuple(self, config):
        thread_id = self._thread(config)
        self._healthy(thread_id)
        try:
            saved = self.get_tuple(config)
        except Exception:
            raise self._failed(thread_id, "checkpoint_read_error", "checkpoint_read") from None
        self._emit(
            thread_id, "checkpoint.read", found=saved is not None,
            checkpoint_id=saved.config["configurable"]["checkpoint_id"] if saved else None,
        )
        return saved

    async def alist(self, config, *, filter=None, before=None, limit=None):
        # The current session API reads exact checkpoints. Keep the inherited
        # native history reader safe as well, without exposing a replay API.
        thread_id = self._thread(config) if config is not None else None
        if thread_id is not None:
            self._healthy(thread_id)
        try:
            for saved in self.list(config, filter=filter, before=before, limit=limit):
                yield saved
        except Exception:
            if thread_id is None:
                raise CheckpointError("checkpoint_read_error") from None
            raise self._failed(thread_id, "checkpoint_read_error", "checkpoint_read") from None

    async def aput(self, config, checkpoint, metadata, new_versions):
        thread_id = self._thread(config)
        self._healthy(thread_id)
        try:
            self._verify_records(thread_id, checkpoint["channel_values"])
            saved_config = self.put(config, checkpoint, metadata, new_versions)
        except Exception as exc:
            code = getattr(exc, "code", "checkpoint_write_error")
            raise self._failed(thread_id, code, "checkpoint_write") from None
        checkpoint_id = saved_config["configurable"]["checkpoint_id"]
        self._confirmed[thread_id] = checkpoint_id
        self._confirmed_ids.setdefault(thread_id, set()).add(checkpoint_id)
        details: dict[str, Any] = {"checkpoint_id": checkpoint_id}
        if type(metadata.get("step")) is int:
            details["checkpoint_step"] = metadata["step"]
        if metadata.get("source") in {"input", "loop", "update", "fork"}:
            details["source"] = metadata["source"]
        self._emit(thread_id, "checkpoint.saved", **details)
        return saved_config

    def _verify_records(self, thread_id: str, values: dict) -> None:
        """A confirmed checkpoint may reference only durable immutable file prefixes."""
        files = values.get("record_files", {})
        previous = {}
        if head := self.last_confirmed(thread_id):
            saved = self.get_tuple({"configurable": {"thread_id": thread_id, "checkpoint_id": head}})
            if saved is None:
                raise CheckpointError("checkpoint_read_error")
            previous = saved.checkpoint["channel_values"]
        if not files and not values.get("record_batch") and not previous.get("record_files"):
            return
        journal = self.history_journal(thread_id)
        if any(not isinstance(receipt, dict) or receipt.get("run_id") not in files
               for receipt in values.get("record_batch", [])):
            raise CheckpointError("record_integrity_error")
        for run_id, watermark in previous.get("record_files", {}).items():
            if run_id not in files or files[run_id]["record_count"] < watermark["record_count"]:
                raise CheckpointError("record_integrity_error")
            journal.verify(run_id, watermark)
        for ref, scope in previous.get("history_scopes", {}).items():
            if values.get("history_scopes", {}).get(ref) != scope:
                raise CheckpointError("record_integrity_error")
        for run_id, watermark in files.items():
            receipts = [receipt for receipt in values.get("record_batch", []) if receipt.get("run_id") == run_id]
            journal.verify(run_id, watermark, receipts=receipts)
        run_id = values.get("run_id")
        if run_id is not None and (run_id not in files or values.get("record_sequence") != files[run_id]["record_count"]):
            raise CheckpointError("record_integrity_error")

    async def aput_writes(self, config, writes, task_id, task_path=""):
        thread_id = self._thread(config)
        self._healthy(thread_id)
        try:
            self.put_writes(config, writes, task_id, task_path)
        except Exception:
            raise self._failed(thread_id, "checkpoint_write_error", "checkpoint_pending_writes") from None
