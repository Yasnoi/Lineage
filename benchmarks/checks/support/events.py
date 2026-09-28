"""Optional observation hooks for checks; standalone commands keep their behavior.

Only an explicit in-process activation enables these hooks. No environment flag
is inherited by a check's own subprocesses, and production modules never import
this module. Observations contain selected identities, never request bodies.
"""


from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_observer = ContextVar("check_observer", default=None)
_group = ContextVar("check_group", default=None)
_FIELDS = {
    "run_id", "session_id", "checkpoint_id", "last_confirmed_checkpoint_id",
    "checkpoint_complete", "checkpoint_error", "summary_id", "version_id",
    "active_version_id", "step_id", "call_id", "steps", "call_ids", "step_ids",
    "log_path", "log_complete", "stop_reason", "error_code", "model_calls",
    "tool_calls", "tool_requests", "summary_calls", "returncode", "scenario",
}
_FLAGS = {"checkpoint_complete", "log_complete"}
_COUNTS = {"model_calls", "tool_calls", "tool_requests", "summary_calls", "returncode"}
_LISTS = {"steps", "call_ids", "step_ids"}

class RecordingFailure(BaseException):
    """Fatal adapter failure, deliberately not caught by legacy check aggregators."""

class CheckStopped(BaseException):
    """Stop Benchmark's script after a group error, preserving its original cause."""

    def __init__(self, cause):
        self.cause = cause
        super().__init__(type(cause).__name__)

@contextmanager
def activate(observer):
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)

def _emit(event, payload):
    observer = _observer.get()
    if observer is None:
        return
    try:
        observer.emit(event, _group.get(), payload)
    except RecordingFailure:
        raise
    except Exception as exc:
        raise RecordingFailure(type(exc).__name__) from exc

def _offsets():
    import os
    import sys
    values = {}
    for name, stream, descriptor in (("stdout", sys.stdout, 1), ("stderr", sys.stderr, 2)):
        try:
            stream.flush()
            values[name] = os.lseek(descriptor, 0, os.SEEK_CUR)
        except (OSError, ValueError, AttributeError):
            values[name] = None
    return values

@contextmanager
def check_group(group_id):
    if _observer.get() is None:
        yield
        return
    if _group.get() is not None:
        raise RecordingFailure("nested counted groups are not supported")
    token = _group.set(group_id)
    try:
        _emit("group_started", {"output_offsets": _offsets()})
        try:
            yield
        except (RecordingFailure, CheckStopped):
            raise
        except BaseException as exc:
            _emit("group_ended", {
                "outcome": "assertion_failed" if isinstance(exc, AssertionError) else "error",
                "error": {"stage": "check_execution", "code": "check_error",
                          "exception_type": type(exc).__name__}, "output_offsets": _offsets(),
            })
            raise CheckStopped(exc) from exc
        else:
            _emit("group_ended", {"outcome": "passed", "error": None, "output_offsets": _offsets()})
    finally:
        _group.reset(token)

def _selected(values):
    result = {}
    for key in _FIELDS & values.keys():
        value = values[key]
        if value is None:
            result[key] = value
        elif key in _FLAGS and type(value) is bool:
            result[key] = value
        elif key in _COUNTS and type(value) is int:
            result[key] = value
        elif key in _FIELDS - _FLAGS - _COUNTS - _LISTS and isinstance(value, str):
            result[key] = value[:2048 if key == "log_path" else 256]
        elif key in _LISTS and isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            result[key] = [item[:256] for item in value[:256]]
    return result

def valid_identifiers(values):
    """The reader rejects corrupt or unfiltered data instead of repairing it."""
    return isinstance(values, dict) and _selected(values) == values

def observe_state(state, *, kind="agent", **identifiers):
    if _observer.get() is None:
        return
    values = _selected(state)
    values.update(_selected(identifiers))
    active = state.get("active_context") or {}
    if isinstance(active, dict) and isinstance(active.get("version_id"), str):
        values["active_version_id"] = active["version_id"][:256]
    _emit("observation", {"kind": kind, "identifiers": values})

def observe_turn(turn):
    observe_state(turn.state, session_id=turn.session_id, checkpoint_id=turn.checkpoint_id,
                  checkpoint_complete=turn.checkpoint_complete, checkpoint_error=turn.checkpoint_error)

def observe_harness(kind, **identifiers):
    if kind == "agent":
        raise ValueError("use observe_state for actual Agent runs")
    _emit("observation", {"kind": kind, "identifiers": _selected(identifiers)})

def evidence(path, kind="artifact", **identifiers):
    if _observer.get() is not None:
        _emit("evidence", {"kind": kind, "path": str(Path(path).absolute()),
                           "identifiers": _selected(identifiers)})

def coverage_gap(scenario, reason, *, applicable=True):
    _emit("coverage_gap", {"scenario": str(scenario)[:256], "reason": str(reason)[:256],
                           "applicable": bool(applicable)})

def group_id(check, identifier=None):
    from benchmarks.suites.agent_contracts.suite import load_catalog
    symbol = identifier or check.__name__
    module = check.__module__
    matches = [g["group_id"] for unit in load_catalog()[0].values() for g in unit["groups"]
               if g["source"]["symbol"] == symbol and g["entry_module"] == module]
    if len(matches) != 1:
        raise RecordingFailure("ambiguous or unregistered behavior")
    return matches[0]
