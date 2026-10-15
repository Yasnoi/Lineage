"""Run-bound diagnostics and fail-closed writes for native in-memory checkpoints."""

from copy import deepcopy
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from lineage_agent.recording.recorder import RunRecorder
from lineage_agent.archive import ArchiveError, ArchiveStore


CHECKPOINT_ERRORS = {
    "checkpoint_create_error": "会话检查点无法创建，已停止本次请求。",
    "checkpoint_read_error": "会话检查点无法读取，已停止本次请求。",
    "checkpoint_write_error": "会话检查点未能确认保存，已停止后续调用。",
    "record_create_error": "原始记录文件无法创建，未发起后续调用。",
    "record_write_error": "原始记录文件未能确认写入，已停止后续调用。",
    "record_read_error": "原始记录文件无法读取。",
    "record_integrity_error": "原始记录文件或确认范围校验失败。",
    "record_encoding_error": "原始记录包含不支持保存的值。",
    "archive_create_error": "业务归档无法创建，已停止本次请求。",
    "archive_write_error": "业务归档未能确认保存，已停止后续调用。",
    "archive_read_error": "业务归档无法读取。",
    "archive_integrity_error": "业务归档或资源关联校验失败。",
    "archive_version_error": "业务归档格式版本不兼容。",
    "archive_publication_unknown": "轮次发布结果无法确认，当前会话已阻断。",
    "artifact_write_error": "工具结果正文或登记未能保存，已停止后续调用。",
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
        self._history_journals: dict[str, Any] = {}
        self._archives: dict[str, ArchiveStore] = {}
        self._views: dict[str, str] = {}
        self._artifact_bindings: dict[str, Any] = {}
        self._restoring: dict[str, str] = {}

    def bind_archive(self, thread_id: str, archive: ArchiveStore) -> None:
        """Transfer an already opened archive to this scope's lifecycle."""
        if thread_id in self._archives and self._archives[thread_id] is not archive:
            raise CheckpointError("archive_integrity_error")
        journal = self._history_journals.get(thread_id)
        if journal is not None and journal.directory.parent != archive.directory:
            raise CheckpointError("archive_integrity_error")
        self._archives[thread_id] = archive

    async def restore_state(self, thread_id: str, state: dict, *,
                            archive: ArchiveStore, artifacts) -> str:
        """Import a validated snapshot into a fresh, idle in-memory graph.

        Only public graph state-update/read APIs build and inspect the native
        checkpoint. No node is invoked and no existing business view is
        published again. Ownership of ``archive`` transfers only on success.
        """
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Overwrite
        from lineage_agent.agent.recovery import validate_snapshot_state
        from lineage_agent.agent.record_codec import encode_record
        from lineage_agent.agent.record_journal import RecordJournal
        from lineage_agent.agent.state import AgentState

        if (thread_id in self._archives or thread_id in self._history_journals
                or thread_id in self._confirmed or thread_id in self._failures
                or self.get_tuple({"configurable": {"thread_id": thread_id}}) is not None):
            raise CheckpointError("checkpoint_create_error")
        journal = None
        try:
            frozen = deepcopy(state)
            view_id = frozen["resource_view_id"]
            validate_snapshot_state(frozen, archive, view_id)
            journal = RecordJournal(archive.directory / "history", create=False)
            self.bind_archive(thread_id, archive)
            self.bind_history(thread_id, journal)
            self.bind_artifacts(thread_id, artifacts)
            self._restoring[thread_id] = view_id
            builder = StateGraph(AgentState)

            def imported(_state):
                raise AssertionError("snapshot import must not execute a graph node")

            builder.add_node("import_snapshot", imported)
            builder.add_edge(START, "import_snapshot")
            builder.add_edge("import_snapshot", END)
            graph = builder.compile(checkpointer=self)
            config = {"configurable": {"thread_id": thread_id}}
            saved = await graph.aupdate_state(
                config, {**frozen, "input": Overwrite(deepcopy(frozen["input"]))},
                as_node="import_snapshot",
            )
            snapshot = await graph.aget_state(saved)
            checkpoint_id = saved["configurable"]["checkpoint_id"]
            if (snapshot.next or snapshot.tasks or encode_record(snapshot.values) != encode_record(frozen)
                    or self.last_confirmed(thread_id) != checkpoint_id
                    or self.confirmed_view(thread_id) != view_id):
                raise CheckpointError("checkpoint_read_error")
            return checkpoint_id
        except BaseException as exc:
            # The caller still owns the borrowed archive and Artifact binding
            # on a failed import. Remove every temporary native handle first.
            self._archives.pop(thread_id, None)
            self._history_journals.pop(thread_id, None)
            try:
                self.release(thread_id)
            finally:
                if journal is not None:
                    journal.close()
            if not isinstance(exc, Exception):
                raise
            raise CheckpointError(getattr(exc, "code", "checkpoint_create_error")) from None
        finally:
            self._restoring.pop(thread_id, None)

    def bind_history(self, thread_id: str, journal) -> None:
        previous = self._history_journals.get(thread_id)
        if previous is not None and previous.directory != journal.directory:
            raise CheckpointError("record_integrity_error")
        if thread_id not in self._archives:
            try:
                self._archives[thread_id] = ArchiveStore(journal.directory.parent)
            except ArchiveError as exc:
                raise CheckpointError(exc.code) from None
        elif self._archives[thread_id].directory != journal.directory.parent:
            raise CheckpointError("record_integrity_error")
        self._history_journals[thread_id] = journal

    def archive(self, thread_id: str) -> ArchiveStore:
        if thread_id not in self._archives:
            raise CheckpointError("archive_read_error")
        return self._archives[thread_id]

    def confirmed_view(self, thread_id: str) -> str:
        if thread_id not in self._views:
            raise CheckpointError("archive_read_error")
        return self._views[thread_id]

    def bind_artifacts(self, thread_id: str, artifacts) -> None:
        self._artifact_bindings[thread_id] = artifacts

    def unbind_artifacts(self, thread_id: str) -> None:
        self._artifact_bindings.pop(thread_id, None)

    def verify_artifact_writes(self, thread_id: str) -> None:
        binding = self._artifact_bindings.get(thread_id)
        if binding is not None and binding.persistence_error() is not None:
            raise self._failed(thread_id, "artifact_write_error", "artifact_save")

    def block(self, thread_id: str, code: str) -> None:
        self._failed(thread_id, code, "archive_finalization")

    def prepare_state(self, thread_id: str, state: dict) -> str:
        """Stage a business view; only a later successful aput grants access."""
        self._healthy(thread_id)
        self.verify_artifact_writes(thread_id)
        try:
            archive = self.archive(thread_id)
            parent_id = self._views.get(thread_id)
            parent = archive.view(parent_id) if parent_id else {}
            summaries = set(parent.get("summaries", []))
            for scope in state.get("history_scopes", {}).values():
                archive.save_scope(scope)
            compaction = state.get("compaction") or {}
            for lane, selected in compaction.get("lanes", {}).items():
                if selected.get("status") not in {"generated", "reused"} or not selected.get("body"):
                    continue
                scope = selected["scope"]
                prior = selected.get("previous", [])
                candidate = selected.get("candidate_input")
                if candidate is not None:
                    prior = [{"summary_id": candidate["summary_id"]}]
                archive.save_summary({
                    "summary_id": selected["summary_id"], "lane": lane,
                    "body": selected["body"], "run_id": state["run_id"],
                    "history_ref": scope["history_ref"], "ranges": deepcopy(scope["ranges"]),
                    "previous": [ref["summary_id"] for ref in prior],
                    "source_refs": [] if candidate is not None else deepcopy(selected.get("source_refs", [])),
                })
                summaries.add(selected["summary_id"])
            artifacts = set(parent.get("artifacts", []))
            binding = self._artifact_bindings.get(thread_id)
            if binding is not None:
                artifacts.update(binding.registered_refs())
            fact_ends = dict(parent.get("fact_ends", {}))
            if run_id := state.get("run_id"):
                fact_ends[run_id] = archive.event_end(run_id)
            return archive.prepare_view(
                record_files=state.get("record_files", {}),
                history_scopes=state.get("history_scopes", {}),
                summaries=sorted(summaries), artifacts=sorted(artifacts),
                parent_id=parent_id, fact_ends=fact_ends,
            )
        except ArchiveError as exc:
            raise self._failed(thread_id, exc.code, "archive_prepare") from None

    def archive_facts(self, thread_id: str, state: dict, update: dict, node: str) -> None:
        """Record actual results once, before any later History/checkpoint failure."""
        archive = self.archive(thread_id)
        run_id = state["run_id"]
        try:
            if update.get("model_calls", 0) > state.get("model_calls", 0):
                count = update["model_calls"]
                archive.append_event(run_id, "model_result", {
                    "model_call": count, "result": deepcopy(update.get("result")),
                    "checkpoint_id": None, "error_code": update.get("error_code"),
                }, key=f"model:{count}")
            if update.get("summary_calls", 0) > state.get("summary_calls", 0):
                count = update["summary_calls"]
                archive.append_event(run_id, "summary_result", {
                    "summary_call": count, "result": deepcopy(update.get("summary_result")),
                    "compaction": deepcopy(update.get("compaction", state.get("compaction"))),
                    "checkpoint_id": None,
                }, key=f"summary:{count}")
            before = len(state.get("execution_facts", []))
            facts = update.get("execution_facts", [])[before:]
            if facts:
                archive.append_event(run_id, "execution_facts", {"facts": deepcopy(facts)})
            if "compaction" in update and update["compaction"] != state.get("compaction"):
                archive.append_event(run_id, "compaction", {"node": node, "compaction": deepcopy(update["compaction"])})
        except ArchiveError as exc:
            raise self._failed(thread_id, exc.code, "archive_facts") from None

    def history_journal(self, thread_id: str):
        if thread_id not in self._history_journals:
            raise CheckpointError("record_read_error")
        return self._history_journals[thread_id]

    def bind_run(self, thread_id: str, recorder: RunRecorder) -> None:
        self._recorders[thread_id] = recorder

    def history_cursor_key(self, thread_id: str) -> bytes:
        """Keep opaque recall cursors valid for the storage scope's lifetime."""
        return self.archive(thread_id).cursor_key()

    def release(self, thread_id: str) -> None:
        """Release private Run storage without affecting another session."""
        self.delete_thread(thread_id)
        self._failures.pop(thread_id, None)
        self._confirmed.pop(thread_id, None)
        self._confirmed_ids.pop(thread_id, None)
        self._recorders.pop(thread_id, None)
        self._views.pop(thread_id, None)
        self._artifact_bindings.pop(thread_id, None)
        self._restoring.pop(thread_id, None)
        journal = self._history_journals.pop(thread_id, None)
        if journal is not None:
            journal.close()
        archive = self._archives.pop(thread_id, None)
        if archive is not None:
            archive.close()

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
            view_id = checkpoint["channel_values"].get("resource_view_id")
            restoring = self._restoring.get(thread_id)
            if restoring is not None and view_id != restoring:
                raise CheckpointError("archive_integrity_error")
            if view_id:
                archive = self.archive(thread_id)
                archive.verify_view(view_id, confirmed=restoring is not None)
                view = archive.view(view_id, confirmed=restoring is not None)
                values = checkpoint["channel_values"]
                if (view["record_files"] != values.get("record_files", {})
                        or view["record_order"] != list(values.get("record_files", {}))
                        or view["history_scopes"] != values.get("history_scopes", {})
                        or values.get("conversation_id", archive.conversation_id) != archive.conversation_id):
                    raise CheckpointError("archive_integrity_error")
                active = values.get("active_context") or {}
                references = [active.get("history_summary"), active.get("run_summary")]
                references += [span["reference"] for span in active.get("summary_spans", [])]
                if any(ref is not None and ref["summary_id"] not in view["summaries"] for ref in references):
                    raise CheckpointError("archive_integrity_error")
                for ref in references:
                    if ref is None:
                        continue
                    summary = archive.summary(ref["summary_id"], view_id, confirmed=False)
                    if any(ref.get(key) != summary.get(key) for key in ("lane", "history_ref", "ranges")):
                        raise CheckpointError("archive_integrity_error")
                if any(span.get("summary_id") != span["reference"].get("summary_id")
                       or span.get("lane") != span["reference"].get("lane")
                       for span in active.get("summary_spans", [])):
                    raise CheckpointError("archive_integrity_error")
            saved_config = self.put(config, checkpoint, metadata, new_versions)
            checkpoint_id = saved_config["configurable"]["checkpoint_id"]
            if view_id and restoring is None:
                self.archive(thread_id).confirm_view(view_id, checkpoint_id)
        except Exception as exc:
            code = getattr(exc, "code", "checkpoint_write_error")
            raise self._failed(thread_id, code, "checkpoint_write") from None
        checkpoint_id = saved_config["configurable"]["checkpoint_id"]
        self._confirmed[thread_id] = checkpoint_id
        self._confirmed_ids.setdefault(thread_id, set()).add(checkpoint_id)
        if view_id:
            self._views[thread_id] = view_id
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
