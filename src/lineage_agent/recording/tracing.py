"""Optional, bounded LangSmith uploads of explicitly selected run data."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from queue import Empty, Full, Queue
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from langsmith import Client
from urllib3.util import Retry

from lineage_agent.agent.checkpointing import CHECKPOINT_ERRORS, CheckpointError
from lineage_agent.config.settings import ModelSettings
from lineage_agent.models.responses import MODEL_ERRORS
from lineage_agent.recording.recorder import LOG_FAILURE_MESSAGE
from lineage_agent.recording.trace_data import TraceSanitizer
from lineage_agent.tools.types import safe_tool_argument_reason, safe_tool_execution_reason


QUEUE_SIZE = 128
CLOSE_TIMEOUT_SECONDS = 5.0
_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"
_DEFAULT_PROJECT = "lineage-agent-dev"
_WARNINGS = {
    "configuration": "LangSmith 配置无效，已关闭本次追踪；Agent 继续运行。",
    "recording": "LangSmith 追踪记录未能完整生成；Agent 继续运行。",
    "upload": "LangSmith 上传失败，远端追踪可能不完整；Agent 继续运行。",
    "overflow": "LangSmith 上传队列已满，部分追踪已丢弃；Agent 继续运行。",
    "timeout": "LangSmith 收尾等待超时，远端追踪可能不完整；已结束等待。",
}
_MODEL_ERRORS = set(MODEL_ERRORS) | {"model_interrupted", "logging_error"}
_TOOL_ERRORS = {"unknown_tool", "invalid_arguments", "execution_error", "logging_error", "tool_interrupted"}
_STAGES = {
    "model_request", "response_validation", "name_validation", "argument_parsing",
    "executor_configuration", "execution", "argument_validation", "result_generation",
}
_OUTCOMES = {"succeeded", "rejected", "failed", "skipped", "unknown"}
_BENCHMARK_FIELDS = {"benchmark_batch_id", "sample_run_id", "case_id", "turn"}
_IDENTITY_FIELDS = {
    "step_id", "model_call", "call_id", "request_index", "summary_id", "summary_call",
}
_scope: ContextVar["TraceScope | None"] = ContextVar("lineage_trace_scope", default=None)
_current_trace: ContextVar["RunTrace | None"] = ContextVar("lineage_current_trace", default=None)


class TraceScope:
    """An opt-in observer exposing only safe identities and submission facts."""

    def __init__(self, binding: dict[str, Any]) -> None:
        if not isinstance(binding, dict) or set(binding) != _BENCHMARK_FIELDS:
            raise ValueError("invalid trace scope binding")
        if type(binding["turn"]) is not int or not 1 <= binding["turn"] <= 2**31 - 1:
            raise ValueError("invalid trace scope turn")
        for key in _BENCHMARK_FIELDS - {"turn"}:
            if type(binding[key]) is not str or re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", binding[key]) is None:
                raise ValueError("invalid trace scope identifier")
        self._binding = deepcopy(binding)
        self._traces: list[RunTrace] = []

    def _register(self, trace: "RunTrace", settings: ModelSettings) -> None:
        # A disabled trace still needs a safe local Agent identity. Reuse the
        # production text filter without constructing an uploader or client.
        secrets = [os.environ.get(name, "") for name in ("API_KEY", "LANGSMITH_API_KEY")]
        sanitizer = TraceSanitizer(secrets=secrets + [value.strip() for value in secrets],
                                   project_root=settings.project_root)
        trace._binding = {key: sanitizer.text(value) if isinstance(value, str) else value
                          for key, value in self._binding.items()}
        self._traces.append(trace)
        _current_trace.set(trace)

    def statuses(self) -> list[dict[str, Any]]:
        """Return independent snapshots, including disabled and failed starts."""
        return [trace.status() for trace in self._traces]


@contextmanager
def trace_scope(binding: dict[str, Any]):
    """Associate one Benchmark turn without changing Agent State or requests."""
    observer = TraceScope(binding)
    scope_token = _scope.set(observer)
    trace_token = _current_trace.set(None)
    try:
        yield observer
    finally:
        _current_trace.reset(trace_token)
        _scope.reset(scope_token)


def bind_request(request_id: str, purpose: str) -> dict[str, Any]:
    """Bind an actually admitted SDK request to its current safe span identity."""
    trace = _current_trace.get()
    if trace is None or trace._closed:
        return {}
    if type(purpose) is not str or purpose not in {"business", "summary"} or type(request_id) is not str:
        raise ValueError("invalid trace request binding")
    try:
        UUID(request_id)
    except ValueError:
        raise ValueError("invalid trace request identifier") from None
    try:
        return trace._bind_request(request_id, purpose)
    except Exception:
        # A caller's valid request must survive failures in optional observation.
        # Keep only the already filtered scope identity, without claiming a span
        # association that could have failed midway through recording.
        trace._warn("recording")
        return {"agent_run_id": trace.run_id, "request_id": request_id, "purpose": purpose,
                **deepcopy(trace._binding)}


def _warning(kind: str) -> None:
    try:
        print(_WARNINGS[kind], file=sys.stderr)
    except Exception:
        pass


def _safe(method):
    """Only suppress tracing failures, never exceptions from the business call."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            self._warn("recording")
            return None
    return wrapped


@dataclass(frozen=True)
class _TraceSettings:
    endpoint: str
    api_key: str = field(repr=False)
    project: str
    workspace_id: str


@dataclass
class _Span:
    payload: dict[str, Any]
    started: float
    tool_name: str | None = None
    ended: bool = False


class _WorkerLogFilter(logging.Filter):
    """SDK exception details must not escape the uploader's safe error boundary."""
    def __init__(self) -> None:
        super().__init__()
        self.thread_id = threading.get_ident()

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread != self.thread_id


def _read_settings() -> _TraceSettings | None:
    enabled = os.environ.get("LANGSMITH_TRACING", "false").strip().casefold()
    if enabled in {"", "false"}:
        return None
    if enabled != "true":
        raise ValueError("invalid tracing switch")
    # These SDK paths introduce their own destinations, threads, or disk dumps.
    unsupported = ("USE_PYO3_CLIENT", "RUNS_ENDPOINTS", "FAILED_TRACES_DIR")
    if any(f"{prefix}_{name}" in os.environ for prefix in ("LANGSMITH", "LANGCHAIN") for name in unsupported):
        raise ValueError("unsupported SDK override")
    api_key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    endpoint = os.environ.get("LANGSMITH_ENDPOINT", _DEFAULT_ENDPOINT).strip().rstrip("/")
    project = os.environ.get("LANGSMITH_PROJECT", _DEFAULT_PROJECT).strip()
    workspace = os.environ.get("LANGSMITH_WORKSPACE_ID", "").strip()
    parsed = urlsplit(endpoint)
    parsed.port  # Reject malformed port values without retaining the input.
    if (
        not api_key or not project or len(project) > 256
        or any(ord(char) < 32 for char in project)
        or parsed.scheme not in {"http", "https"} or not parsed.hostname
        or parsed.username is not None or parsed.password is not None
        or parsed.query or parsed.fragment
        or any(char.isspace() for char in endpoint)
    ):
        raise ValueError("invalid tracing configuration")
    if workspace:
        UUID(workspace)
    return _TraceSettings(endpoint, api_key, project, workspace)


def _create_client(settings: _TraceSettings) -> Client:
    return Client(
        api_url=settings.endpoint,
        api_key=settings.api_key,
        workspace_id=settings.workspace_id,
        auto_batch_tracing=False,
        timeout_ms=(1000, 2000),
        retry_config=Retry(
            total=0, connect=0, read=0, redirect=0, status=0,
            respect_retry_after_header=False,
        ),
        omit_traced_runtime_info=True,
        disable_prompt_cache=True,
        tracing_mode="langsmith",
        tracing_sampling_rate=1,
        hide_inputs=False,
        hide_outputs=False,
        hide_metadata=False,
    )


class RunTrace:
    """One run's explicit spans and its best-effort upload worker."""

    def __init__(self, run_id: str, *, trace_id: str | None = None) -> None:
        self.run_id = run_id
        self.trace_id = trace_id or run_id
        self._session_id: str | None = None
        self.enabled = False
        self._warned: set[str] = set()
        self._warning_lock = threading.Lock()
        self._queue: Queue[dict[str, Any]] = Queue(maxsize=QUEUE_SIZE)
        self._stop = threading.Event()
        self._abort = threading.Event()
        self._worker: threading.Thread | None = None
        self._settings: _TraceSettings | None = None
        self._sanitizer: TraceSanitizer | None = None
        self._root: _Span | None = None
        self._step_id: str | None = None
        self._model_call: int | None = None
        self._pending_model: _Span | None = None
        self._active_context: dict[str, Any] = {}
        self._finished = False
        self._closed = False
        self._binding: dict[str, Any] = {}
        self._request_span: _Span | None = None
        self._status_lock = threading.Lock()
        self._span_statuses: dict[str, dict[str, Any]] = {}
        self._final_status: dict[str, Any] | None = None

    def _status_snapshot(self) -> dict[str, Any]:
        spans = deepcopy(list(self._span_statuses.values()))
        counts = {key: sum(span[key] for span in spans)
                  for key in ("queued", "attempted", "submitted", "failed", "dropped")}
        counts["generated"] = len(spans)
        counts["unconfirmed"] = sum(not (span["submitted"] or span["failed"] or span["dropped"])
                                    for span in spans)
        with self._warning_lock:
            warnings = sorted(self._warned)
        stopped = self._worker is None or not self._worker.is_alive()
        if "configuration" in warnings:
            state = "configuration_error"
        elif not self.enabled:
            state = "disabled"
        elif not self._closed:
            state = "pending"
        elif stopped and not warnings and counts["submitted"] == counts["generated"]:
            state = "submission_completed"
        else:
            state = "incomplete"
        return {
            "agent_run_id": self.run_id, "trace_id": self.trace_id if self.enabled else None,
            "binding": deepcopy(self._binding), "session_id": self._session_id,
            "enabled": self.enabled, "closed": self._closed,
            "shutdown_confirmed": self._closed and stopped, "upload_state": state,
            "warning_codes": warnings, "counts": counts, "spans": spans,
        }

    def status(self) -> dict[str, Any]:
        """Safe submission facts, not proof that remote data can be retrieved."""
        with self._status_lock:
            return deepcopy(self._final_status) if self._final_status is not None else self._status_snapshot()

    def _span_status(self, span_id: str, **changes: Any) -> None:
        with self._status_lock:
            if self._final_status is None:
                self._span_statuses[span_id].update(changes)

    def _bind_request(self, request_id: str, purpose: str) -> dict[str, Any]:
        identity = {"agent_run_id": self.run_id, "trace_id": self.trace_id if self.enabled else None,
                    "request_id": request_id, "purpose": purpose, **deepcopy(self._binding)}
        if self._session_id is not None:
            identity["session_id"] = self._session_id
        token = self._request_span
        if self.enabled and token is not None and not token.ended:
            metadata = token.payload["extra"]["metadata"]
            actual_purpose = "summary" if metadata.get("purpose") == "summary" else "business"
            if purpose != actual_purpose or "request_id" in metadata:
                raise ValueError("trace request does not match the current span")
            metadata["request_id"] = request_id
            identity["span_id"] = token.payload["id"]
            safe_metadata = self._identity_metadata(metadata)
            identity.update({key: safe_metadata[key] for key in _IDENTITY_FIELDS if key in safe_metadata})
            self._span_status(token.payload["id"], metadata=safe_metadata)
        return identity

    def _identity_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        allowed = _BENCHMARK_FIELDS | _IDENTITY_FIELDS | {
            "run_id", "session_id", "purpose", "request_id", "outcome", "reason_code",
            "skip_reason", "error_code", "stage", "checkpoint_id", "checkpoint_error", "checkpoint_saved",
            "reused_result", "executor_entered",
        }
        return {key: self._sanitizer.text(value) if isinstance(value, str) else value
                for key, value in metadata.items() if key in allowed
                and (value is None or type(value) in {str, bool, int})}

    def _warn(self, kind: str) -> None:
        try:
            with self._warning_lock:
                if kind in self._warned:
                    return
                self._warned.add(kind)
            _warning(kind)
        except Exception:
            pass

    def _start(self, config: _TraceSettings, settings: ModelSettings) -> None:
        UUID(self.run_id)
        UUID(self.trace_id)
        self._settings = config
        model_key = os.environ.get("API_KEY", "")
        self._sanitizer = TraceSanitizer(
            secrets=[model_key, model_key.strip(), config.api_key,
                     os.environ.get("LANGSMITH_API_KEY", "")],
            project_root=settings.project_root,
        )
        self._root = self._new_span("Lineage Agent", "chain", {}, {}, run_id=self.trace_id)
        self.enabled = True
        self._worker = threading.Thread(
            target=self._upload, name="lineage-langsmith-upload", daemon=True,
        )
        self._worker.start()

    def _new_span(self, name, run_type, inputs, metadata, *, run_id=None) -> _Span:
        now = datetime.now(timezone.utc)
        span_id = run_id or str(uuid4())
        order = now.strftime("%Y%m%dT%H%M%S%fZ") + span_id
        payload = {
            "id": span_id, "trace_id": self.trace_id,
            "name": name, "run_type": run_type, "inputs": inputs,
            "start_time": now.isoformat(),
            "project_name": self._settings.project,
            "extra": {"metadata": {"run_id": self.run_id, **self._binding, **metadata}},
        }
        if self._session_id is not None:
            payload["extra"]["metadata"]["session_id"] = self._session_id
        if self._root is not None:
            payload["parent_run_id"] = self.trace_id
            order = self._root.payload["dotted_order"] + "." + order
        payload["dotted_order"] = order
        with self._status_lock:
            self._span_statuses[span_id] = {
                "span_id": span_id, "trace_id": self.trace_id,
                "parent_run_id": payload.get("parent_run_id"), "run_type": run_type,
                "metadata": self._identity_metadata(payload["extra"]["metadata"]), "usage": {},
                **{key: False for key in ("queued", "attempted", "submitted", "failed", "dropped")},
            }
        return _Span(payload, time.monotonic())

    @_safe
    def bind_session(self, session_id: str) -> None:
        """Associate filtered spans without sending the saved graph State."""
        if not self.enabled or self._finished or self._closed:
            return
        self._session_id = self._sanitizer.text(session_id)
        self._root.payload["extra"]["metadata"]["session_id"] = self._session_id
        self._span_status(self.trace_id, metadata=self._identity_metadata(self._root.payload["extra"]["metadata"]))

    @_safe
    def finish_checkpoint(self, checkpoint_id: str | None, checkpoint_error=None) -> None:
        """Set confirmation metadata before finish submits the root span."""
        if not self.enabled or self._finished or self._closed:
            return
        code = checkpoint_error.code if isinstance(checkpoint_error, CheckpointError) else checkpoint_error
        if not isinstance(code, str) or code not in CHECKPOINT_ERRORS:
            code = "checkpoint_write_error" if checkpoint_error is not None else None
        metadata = self._root.payload["extra"]["metadata"]
        metadata["checkpoint_id"] = self._sanitizer.text(checkpoint_id) if isinstance(checkpoint_id, str) else None
        metadata["checkpoint_error"] = code
        metadata["checkpoint_saved"] = checkpoint_id is not None and code is None

    @_safe
    def observe_read_budget(self, stage: str, error: str | None) -> None:
        """Retain bounded, fixed diagnostics without executor exception data."""
        if not self.enabled or self._finished or self._closed:
            return
        if (
            type(stage) is not str or stage not in {"run_start", "model", "run_end"}
            or type(error) is not str or error != "read_budget_unavailable"
        ):
            return
        metadata = self._root.payload["extra"]["metadata"]
        diagnostics = metadata.setdefault("read_budget_diagnostics", [])
        diagnostic = {"stage": stage, "error_code": "read_budget_unavailable"}
        if diagnostic not in diagnostics:
            diagnostics.append(diagnostic)

    @_safe
    def observe_context_budget(self, diagnostic) -> None:
        """Keep the latest safe check without inventing a model-call span."""
        if not self.enabled or self._finished or self._closed:
            return
        projected = self._sanitizer.context_budget(diagnostic)
        if projected:
            self._root.payload["extra"]["metadata"]["context_budget"] = projected

    @_safe
    def bind_active_context(self, active_context) -> None:
        """Bind app-owned summary provenance without retaining its body."""
        if not self.enabled or self._finished or self._closed:
            return
        self._active_context = self._sanitizer.active_context(active_context)
        self._root.payload["extra"]["metadata"]["active_context"] = self._active_context
        # Uploads have bounded lineage, but internal filtering must remember
        # every app-owned summary position, including old Runs' prefix summaries.
        self._active_context = {**self._active_context,
                                "summary_spans": self._sanitizer.summary_spans(active_context)}

    @_safe
    def observe_compaction(self, metadata) -> None:
        if not self.enabled or self._finished or self._closed:
            return
        projected = self._sanitizer.compaction(metadata)
        if projected:
            self._root.payload["extra"]["metadata"]["compaction"] = projected

    @_safe
    def summary_start(self, model_name, metadata=None) -> _Span | None:
        """Summary sources are intentionally absent from this interface."""
        if not self.enabled or self._closed or self._finished:
            return None
        projected = self._sanitizer.compaction(metadata)
        token = self._new_span(
            "Context summary", "llm",
            {"model": self._sanitizer.text(model_name), "content_omitted": True},
            {"purpose": "summary", **projected,
             "ls_provider": "openai", "ls_model_type": "chat",
             "ls_model_name": self._sanitizer.text(model_name),
             "ls_invocation_params": {"use_responses_api": True}},
        )
        self._request_span = token
        return token

    @_safe
    def summary_end(self, token, result=None, error_code=None, stage=None) -> None:
        if not self.enabled or not isinstance(token, _Span):
            return
        if self._request_span is token:
            self._request_span = None
        token.payload["outputs"] = self._sanitizer.summary_outputs(result)
        metadata = token.payload["extra"]["metadata"]
        # A model return is not evidence that the candidate was valid or saved.
        metadata["outcome"] = "failed" if error_code else "returned"
        if stage in _STAGES:
            metadata["stage"] = stage
        if error_code:
            code = error_code if error_code in _MODEL_ERRORS else "model_error"
            metadata["error_code"] = code
            token.payload["error"] = code
        self._submit(token)

    @staticmethod
    def _stamp_end(span: _Span) -> None:
        if "end_time" not in span.payload:
            span.payload["end_time"] = datetime.now(timezone.utc).isoformat()
            span.payload["extra"]["metadata"]["duration_ms"] = max(0, round((time.monotonic() - span.started) * 1000))

    def _submit(self, span: _Span) -> None:
        if span.ended or self._closed:
            return
        span.ended = True
        self._stamp_end(span)
        payload = self._sanitizer.cap_payload(span.payload)
        output = payload.get("outputs", {})
        usage = output.get("usage_metadata", {}) if isinstance(output, dict) else {}
        self._span_status(span.payload["id"], metadata=self._identity_metadata(payload["extra"]["metadata"]),
                          usage={key: value for key, value in usage.items()
                                 if key in {"input_tokens", "output_tokens", "total_tokens"}
                                 and type(value) is int and value >= 0} if isinstance(usage, dict) else {})
        try:
            self._queue.put_nowait(payload)
            self._span_status(span.payload["id"], queued=True)
        except Full:
            self._span_status(span.payload["id"], dropped=True)
            self._warn("overflow")

    def _upload(self) -> None:
        client = None
        sdk_logger = logging.getLogger("langsmith.client")
        log_filter = _WorkerLogFilter()
        sdk_logger.addFilter(log_filter)
        try:
            client = _create_client(self._settings)
            while not self._abort.is_set():
                try:
                    payload = self._queue.get(timeout=0.05)
                except Empty:
                    if self._stop.is_set():
                        break
                    continue
                try:
                    self._span_status(payload["id"], attempted=True)
                    client.create_run(**payload)
                except Exception:
                    self._span_status(payload["id"], failed=True)
                    self._warn("upload")
                else:
                    self._span_status(payload["id"], submitted=True)
                finally:
                    self._queue.task_done()
        except Exception:
            self._warn("upload")
        finally:
            if client is not None:
                try:
                    client.close(timeout=0)
                except Exception:
                    self._warn("upload")
            sdk_logger.removeFilter(log_filter)

    @_safe
    def set_model_step(self, step_id: str | None, model_call: int, *, remaining_model_calls=None, final_answer_call=False, read_budget=None, read_budget_error=None) -> None:
        if self.enabled:
            if self._pending_model is not None:
                self.model_complete(error_code="model_interrupted", stage="model_request")
            self._step_id = step_id
            self._model_call = model_call
            self._allowances = {
                "remaining_model_calls": remaining_model_calls,
                "final_answer_call": final_answer_call, "read_budget": read_budget,
                "read_budget_error": (
                    "read_budget_unavailable" if type(read_budget_error) is str
                    and read_budget_error == "read_budget_unavailable" else None
                ),
            }

    @_safe
    def model_start(self, model_name, input, tools, *, instructions=None, tool_choice=None) -> _Span | None:
        if not self.enabled or self._closed or self._finished:
            return None
        token = self._new_span(
            "OpenAI Responses", "llm",
            self._sanitizer.model_inputs(
                model_name, input, tools, instructions=instructions,
                tool_choice=tool_choice, active_context=self._active_context,
            ),
            {"purpose": "business", "step_id": self._step_id, "model_call": self._model_call,
             "ls_provider": "openai", "ls_model_type": "chat",
             "ls_model_name": self._sanitizer.text(model_name),
             **getattr(self, "_allowances", {}),
             "context_budget": self._sanitizer.context_budget(
                 self._root.payload["extra"]["metadata"].get("context_budget")
             ),
             "ls_invocation_params": {"use_responses_api": True}},
        )
        self._request_span = token
        return token

    @_safe
    def model_end(self, token, result=None, error_code=None, stage=None) -> None:
        if not self.enabled or not isinstance(token, _Span):
            return
        if self._request_span is token:
            self._request_span = None
        self._stamp_end(token)
        token.payload["outputs"] = self._sanitizer.model_outputs(result)
        metadata = token.payload["extra"]["metadata"]
        metadata["outcome"] = "failed" if error_code else "succeeded"
        if stage in _STAGES:
            metadata["stage"] = stage
        if error_code:
            code = error_code if error_code in _MODEL_ERRORS else "model_error"
            metadata["error_code"] = code
            token.payload["error"] = code
        if error_code or self._step_id is None:
            self._submit(token)
        else:
            # Graph validation can still reject an otherwise valid SDK response.
            self._pending_model = token

    @_safe
    def model_complete(self, error_code=None, stage=None) -> None:
        token = self._pending_model
        if not self.enabled or token is None:
            return
        self._pending_model = None
        metadata = token.payload["extra"]["metadata"]
        if error_code:
            code = error_code if error_code in _MODEL_ERRORS else "model_error"
            token.payload["error"] = code
            metadata.update(outcome="failed", error_code=code)
        if stage in _STAGES:
            metadata["stage"] = stage
        self._submit(token)

    @_safe
    def tool_start(self, call, *, step_id, request_index, reused_result=False) -> _Span | None:
        if not self.enabled or self._closed or self._finished:
            return None
        name = call.get("name")
        metadata = {"step_id": step_id, "model_call": self._model_call,
                    "request_index": request_index}
        if reused_result:
            metadata.update(reused_result=True, executor_entered=False)
        if isinstance(call.get("call_id"), str):
            metadata["call_id"] = self._sanitizer.text(call["call_id"])
        if isinstance(name, str):
            metadata["tool_name"] = self._sanitizer.text(name)
        token = self._new_span(
            self._sanitizer.text(name) if isinstance(name, str) else "Tool request",
            "tool", self._sanitizer.tool_inputs(call), metadata,
        )
        token.tool_name = name
        return token

    @_safe
    def tool_end(self, token, *, output, outcome, stage, reason_code=None, reason_message=None) -> None:
        if not self.enabled or not isinstance(token, _Span):
            return
        self._stamp_end(token)
        token.payload["outputs"] = self._sanitizer.tool_outputs(output, name=token.tool_name)
        metadata = token.payload["extra"]["metadata"]
        metadata["outcome"] = outcome if outcome in _OUTCOMES else "failed"
        if stage in _STAGES:
            metadata["stage"] = stage
        if reason_code == "logging_error":
            metadata.update(reason_code="logging_error", reason_message=LOG_FAILURE_MESSAGE,
                            skip_reason="logging_error")
        elif reason_code is not None:
            code, message = (
                safe_tool_execution_reason(reason_code) if outcome == "failed"
                else safe_tool_argument_reason(reason_code)
            )
            metadata.update(reason_code=code, reason_message=message)
        if outcome == "unknown":
            # The local wait was interrupted; the external outcome is not a
            # known tool failure and must remain distinguishable in diagnostics.
            metadata["error_code"] = "tool_interrupted"
            token.payload["error"] = "tool_interrupted"
        elif outcome != "succeeded":
            raw_error = output.get("error", {}) if isinstance(output, dict) else {}
            code = raw_error.get("code") if isinstance(raw_error, dict) else None
            code = code if code in _TOOL_ERRORS else "execution_error"
            if metadata.get("skip_reason") == "logging_error":
                code = "logging_error"
            metadata["error_code"] = code
            token.payload["error"] = metadata.get("reason_code", metadata.get("skip_reason", code))
        self._submit(token)

    @_safe
    def tool_skipped(self, call, *, step_id, request_index, reason) -> None:
        if not self.enabled:
            return
        token = self.tool_start(call, step_id=step_id, request_index=request_index)
        if token is None:
            return
        token.payload["extra"]["metadata"]["skip_reason"] = (
            reason if reason in {"max_model_calls", "logging_error", "protocol_error", "checkpoint_error"}
            else "not_executed"
        )
        code = "logging_error" if reason == "logging_error" else "execution_error"
        self.tool_end(
            token, output={"ok": False, "error": {"code": code}},
            outcome="skipped", stage="execution",
        )

    @_safe
    def finish(self, state=None, error_code=None, *, read_budget=None, read_budget_error=None) -> None:
        if not self.enabled or self._finished or self._closed:
            return
        self.observe_read_budget("run_end", read_budget_error)
        if isinstance(state, dict):
            self.observe_context_budget(state.get("context_budget"))
            self.observe_compaction(state.get("compaction"))
            self.bind_active_context(state.get("active_context"))
        if self._pending_model is not None:
            self.model_complete(error_code="model_interrupted", stage="model_request")
        self._finished = True
        self._root.payload["outputs"] = self._sanitizer.root_outputs(state)
        self._root.payload["extra"]["metadata"]["read_budget"] = read_budget
        self._root.payload["extra"]["metadata"]["read_budget_error"] = (
            "read_budget_unavailable" if type(read_budget_error) is str
            and read_budget_error == "read_budget_unavailable" else None
        )
        checkpoint_error = self._root.payload["extra"]["metadata"].get("checkpoint_error")
        failed = bool(error_code or checkpoint_error) or (isinstance(state, dict) and state.get("stop_reason") == "error")
        stopped = isinstance(state, dict) and state.get("stop_reason") in {"max_model_calls", "context_budget_exceeded"}
        if failed:
            code = error_code or (state.get("error_code") if isinstance(state, dict) else None) or checkpoint_error
            self._root.payload["error"] = code if code in _MODEL_ERRORS | set(CHECKPOINT_ERRORS) | {
                "run_interrupted", "client_close_error", "invalid_input", "pre_execution_configuration_error",
                "artifact_store_init_error", "artifact_cleanup_error",
                "context_budget_exceeded", "context_measurement_error",
                "summary_model_error", "invalid_summary", "summary_refusal", "summary_tool_request",
                "source_changed", "history_record_missing", "history_reference_invalid", "invalid_source_reference",
            } else "agent_error"
        elif stopped:
            self._root.payload["error"] = state["stop_reason"]
        self._root.payload["extra"]["metadata"]["outcome"] = (
            "failed" if failed else "stopped" if stopped else "succeeded"
        )
        self._submit(self._root)

    async def close(self) -> None:
        """Wait at most five seconds; a stuck upload cannot pin interpreter exit."""
        try:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            if self._worker is None:
                return
            deadline = time.monotonic() + CLOSE_TIMEOUT_SECONDS
            while self._worker.is_alive():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._warn("timeout")
                    break
                await asyncio.sleep(min(0.05, remaining))
        except Exception:
            self._warn("upload")
        finally:
            self._abort.set()
            with self._status_lock:
                if self._final_status is None:
                    self._final_status = self._status_snapshot()


def start_trace(run_id: str, settings: ModelSettings, *, trace_id: str | None = None) -> RunTrace:
    """Read only this application's dotenv variables; no credential fallback."""
    trace = RunTrace(run_id, trace_id=trace_id)
    try:
        observer = _scope.get()
        if observer is not None:
            observer._register(trace, settings)
        config = _read_settings()
        if config is not None:
            trace._start(config, settings)
    except Exception:
        trace.enabled = False
        trace._abort.set()
        trace._warn("configuration")
    return trace
