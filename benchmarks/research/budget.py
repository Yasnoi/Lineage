"""Durable request accounting for the serial, fixed research batch."""

import json
import math
import os
from pathlib import Path
import time
from uuid import uuid4

from benchmarks.common import utc_now


class WorkerAbort(BaseException):
    """Stop transport without being mistaken for an ordinary Agent failure."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class BudgetRecordError(ValueError):
    pass


def read_ledger(path: Path, max_requests: int = 400, *, batch_id: str | None = None,
                run_ids: set[str] | None = None) -> list[dict]:
    """Reject damaged accounting rather than guessing a remaining allowance."""
    path = Path(path)
    try:
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            raise ValueError
        events = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        identities = set()
        for index, event in enumerate(events, 1):
            if (type(event.get("seq")) is not int or event["seq"] != index or event.get("event") != "request_started"
                    or event.get("purpose") not in {"business", "summary"}
                    or type(event.get("turn")) is not int or event["turn"] < 1
                    or any(not isinstance(event.get(key), str) or not event[key]
                           for key in ("request_id", "run_id", "case_id", "batch_id", "started_at"))
                    or batch_id is not None and event["batch_id"] != batch_id
                    or run_ids is not None and event["run_id"] not in run_ids
                    or event["request_id"] in identities):
                raise ValueError
            identities.add(event["request_id"])
        if len(events) > max_requests:
            raise ValueError
        return events
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError) as exc:
        raise BudgetRecordError("request ledger unavailable or invalid") from exc


def append_durable(path: Path, event: dict) -> None:
    """One serial writer; a failed or partial write aborts further requests."""
    try:
        with Path(path).open("ab", buffering=0) as handle:
            data = (json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            if handle.write(data) != len(data):
                raise OSError("short record write")
            os.fsync(handle.fileno())
    except (OSError, TypeError, ValueError) as exc:
        raise WorkerAbort("record_write_failed") from exc


class RequestBudget:
    def __init__(self, path: Path, run_id: str, case_id: str, *, max_requests: int = 400,
                 remaining_seconds: float = 300, batch_id: str | None = None):
        if (type(max_requests) is not int or not 1 <= max_requests <= 400
                or type(remaining_seconds) not in {int, float}
                or not math.isfinite(remaining_seconds) or not 0 < remaining_seconds <= 7200):
            raise ValueError("invalid research budget")
        self.path, self.run_id, self.case_id = Path(path), run_id, case_id
        self.batch_id = batch_id or run_id
        self.max_requests = max_requests
        self.deadline = time.monotonic() + min(300, remaining_seconds)
        self._seen: list[dict] | None = None

    def reserve(self, turn: int, purpose: str) -> dict:
        if time.monotonic() >= self.deadline:
            raise WorkerAbort("execution_time_exhausted")
        try:
            events = read_ledger(self.path, self.max_requests, batch_id=self.batch_id)
        except BudgetRecordError as exc:
            raise WorkerAbort("request_ledger_invalid") from exc
        if self._seen is not None and events != self._seen:
            # Serial scheduling means nobody else may append, truncate or rewrite
            # the ledger while this sample owns the transport.
            raise WorkerAbort("request_ledger_changed")
        if len(events) >= self.max_requests:
            raise WorkerAbort("request_budget_exhausted")
        event = {"seq": len(events) + 1, "event": "request_started", "request_id": str(uuid4()),
                 "run_id": self.run_id, "case_id": self.case_id, "turn": turn,
                 "batch_id": self.batch_id,
                 "purpose": purpose, "started_at": utc_now()}
        append_durable(self.path, event)
        self._seen = events + [event]
        return event
