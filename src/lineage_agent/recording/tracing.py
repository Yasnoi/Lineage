"""Optional, bounded LangSmith uploads of explicitly selected run data."""

from __future__ import annotations

import asyncio
import logging
import os
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

from lineage_agent.config.settings import ModelSettings
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
_MODEL_ERRORS = {
    "model_error", "model_timeout", "connection_error", "http_error",
    "invalid_response", "unexpected_tool_result", "invalid_call_id",
    "incomplete_response", "empty_response", "model_interrupted",
    "duplicate_call_id", "logging_error",
}
_TOOL_ERRORS = {"unknown_tool", "invalid_arguments", "execution_error", "logging_error", "tool_interrupted"}
_STAGES = {
    "model_request", "response_validation", "name_validation", "argument_parsing",
    "executor_configuration", "execution", "argument_validation", "result_generation",
}
_OUTCOMES = {"succeeded", "rejected", "failed", "skipped"}


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

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
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
        self._finished = False
        self._closed = False

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
        self._settings = config
        model_key = os.environ.get("API_KEY", "")
        self._sanitizer = TraceSanitizer(
            secrets=[model_key, model_key.strip(), config.api_key,
                     os.environ.get("LANGSMITH_API_KEY", "")],
            project_root=settings.project_root,
        )
        self._root = self._new_span("Lineage Agent", "chain", {}, {}, run_id=self.run_id)
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
            "id": span_id, "trace_id": self.run_id,
            "name": name, "run_type": run_type, "inputs": inputs,
            "start_time": now.isoformat(),
            "project_name": self._settings.project,
            "extra": {"metadata": {"run_id": self.run_id, **metadata}},
        }
        if self._root is not None:
            payload["parent_run_id"] = self.run_id
            order = self._root.payload["dotted_order"] + "." + order
        payload["dotted_order"] = order
        return _Span(payload, time.monotonic())

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
        try:
            self._queue.put_nowait(payload)
        except Full:
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
                    client.create_run(**payload)
                except Exception:
                    self._warn("upload")
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
    def set_model_step(self, step_id: str | None, model_call: int) -> None:
        if self.enabled:
            if self._pending_model is not None:
                self.model_complete(error_code="model_interrupted", stage="model_request")
            self._step_id = step_id
            self._model_call = model_call

    @_safe
    def model_start(self, model_name, input, tools) -> _Span | None:
        if not self.enabled or self._closed or self._finished:
            return None
        return self._new_span(
            "OpenAI Responses", "llm",
            self._sanitizer.model_inputs(model_name, input, tools),
            {"step_id": self._step_id, "model_call": self._model_call,
             "ls_provider": "openai", "ls_model_type": "chat",
             "ls_model_name": self._sanitizer.text(model_name),
             "ls_invocation_params": {"use_responses_api": True}},
        )

    @_safe
    def model_end(self, token, result=None, error_code=None, stage=None) -> None:
        if not self.enabled or not isinstance(token, _Span):
            return
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
    def tool_start(self, call, *, step_id, request_index) -> _Span | None:
        if not self.enabled or self._closed or self._finished:
            return None
        name = call.get("name")
        metadata = {"step_id": step_id, "model_call": self._model_call,
                    "request_index": request_index}
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
        if outcome != "succeeded":
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
            reason if reason in {"max_model_calls", "logging_error", "protocol_error"}
            else "not_executed"
        )
        code = "logging_error" if reason == "logging_error" else "execution_error"
        self.tool_end(
            token, output={"ok": False, "error": {"code": code}},
            outcome="skipped", stage="execution",
        )

    @_safe
    def finish(self, state=None, error_code=None) -> None:
        if not self.enabled or self._finished or self._closed:
            return
        if self._pending_model is not None:
            self.model_complete(error_code="model_interrupted", stage="model_request")
        self._finished = True
        self._root.payload["outputs"] = self._sanitizer.root_outputs(state)
        failed = bool(error_code) or (isinstance(state, dict) and state.get("stop_reason") == "error")
        stopped = isinstance(state, dict) and state.get("stop_reason") == "max_model_calls"
        if failed:
            self._root.payload["error"] = error_code if error_code in {"logging_error", "run_interrupted"} else "agent_error"
        elif stopped:
            self._root.payload["error"] = "max_model_calls"
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


def start_trace(run_id: str, settings: ModelSettings) -> RunTrace:
    """Read only this application's dotenv variables; no credential fallback."""
    trace = RunTrace(run_id)
    try:
        config = _read_settings()
        if config is not None:
            trace._start(config, settings)
    except Exception:
        trace.enabled = False
        trace._abort.set()
        trace._warn("configuration")
    return trace
