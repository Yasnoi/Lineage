"""Observe the real transport and filesystem executor without changing inputs."""

from contextlib import asynccontextmanager
from copy import deepcopy
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

from benchmarks.core.files import utc_now
from benchmarks.core.budgets import RequestBudget, append_durable
from lineage_agent.models.summarization import SUMMARY_INSTRUCTIONS
from lineage_agent.tools.types import ToolArgumentError, ToolExecutionError


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def usage_fields(response) -> dict:
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    values = {key: value if type(value := (usage.get(key) if isinstance(usage, dict)
                     else getattr(usage, key, None))) is int and value >= 0 else None
            for key in ("input_tokens", "output_tokens", "total_tokens")}
    details = usage.get("input_tokens_details") if isinstance(usage, dict) else getattr(usage, "input_tokens_details", None)
    cached = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    values["cached_input_tokens"] = cached if type(cached) is int and cached >= 0 else None
    return values



class Observer:
    def __init__(self, budget: RequestBudget, output_dir: Path, material_root: Path):
        self.budget, self.output_dir, self.material_root = budget, Path(output_dir), Path(material_root)
        self.turn = 0
        self.requests: dict[int, list[dict]] = {}
        self.request_events: list[dict] = []
        self.request_usage: list[dict] = []
        self.tool_events: list[dict] = []
        self.global_stop = False
        self.global_stop_reason = None
        self.sequence = 0
        self.request_durations: list[float] = []

    def record(self, event: dict) -> dict:
        self.sequence += 1
        enriched = {"seq": self.sequence, "run_id": self.budget.run_id,
                    "case_id": self.budget.case_id, "turn": self.turn, "at": utc_now(), **event}
        append_durable(self.output_dir / "observations.jsonl", enriched)
        return enriched

    def client_factory(self, original_factory):
        @asynccontextmanager
        async def factory(settings):
            async with original_factory(settings) as client:
                original_create = client.responses.create

                async def create(**kwargs):
                    purpose = "summary" if kwargs.get("instructions") == SUMMARY_INSTRUCTIONS else "business"
                    reserved = self.budget.reserve(self.turn, purpose)
                    # Retain full request only in this process for mechanism comparisons.
                    self.requests.setdefault(self.turn, []).append(deepcopy(kwargs))
                    started = self.record({"event": "request_started", "request_id": reserved["request_id"],
                                           "ledger_seq": reserved["seq"], "purpose": purpose,
                                           "request_sha256": digest(kwargs),
                                           "input_items": len(kwargs.get("input", []))})
                    self.request_events.append(started)
                    sent_at = time.monotonic()
                    try:
                        response = await original_create(**kwargs)
                    except BaseException as exc:
                        status = getattr(exc, "status_code", None)
                        if status in {401, 402, 403, 404}:
                            self.global_stop = True
                            self.global_stop_reason = "credentials_rejected" if status in {401, 403} else "endpoint_prerequisite_failed"
                        self.record({"event": "request_failed", "request_id": reserved["request_id"],
                                     "purpose": purpose, "error_type": type(exc).__name__,
                                     "duration_seconds": time.monotonic() - sent_at,
                                     "http_status": status if type(status) is int else None})
                        raise
                    finally:
                        self.request_durations.append(time.monotonic() - sent_at)
                    reported = usage_fields(response)
                    self.request_usage.append(reported)
                    self.record({"event": "request_finished", "request_id": reserved["request_id"],
                                 "purpose": purpose, "usage": reported,
                                 "duration_seconds": self.request_durations[-1]})
                    return response

                with patch.object(client.responses, "create", side_effect=create):
                    yield client
        return factory

    def safe_path(self, value) -> dict:
        # Only existing approved material paths are persisted verbatim. A model's
        # arbitrary argument can contain unrelated data and is represented by a hash.
        if isinstance(value, str):
            try:
                target = (self.material_root / value).resolve()
                relative = target.relative_to(self.material_root.resolve())
                if target.exists():
                    return {"path": relative.as_posix()}
            except (OSError, ValueError):
                pass
        return {"path": None, "path_sha256": digest(value)}


class ObservedTools:
    def __init__(self, wrapped, observer: Observer):
        self.wrapped, self.observer = wrapped, observer

    def bind_run(self):
        return ObservedTools(self.wrapped.bind_run(), self.observer)

    @property
    def read_budget(self):
        return self.wrapped.read_budget

    async def __call__(self, name, arguments):
        observer = self.observer
        reference = observer.safe_path(arguments.get("path") if isinstance(arguments, dict) else None)
        started = observer.record({"event": "tool_started", "name": name, **reference})
        try:
            result = await self.wrapped(name, arguments)
        except BaseException as exc:
            reason = exc.reason_code if isinstance(exc, (ToolArgumentError, ToolExecutionError)) else "execution_failed"
            event = observer.record({"event": "tool_finished", "start_seq": started["seq"],
                                     "name": name, **reference, "outcome": "failed", "reason_code": reason})
            observer.tool_events.append(event)
            raise
        projection = {"event": "tool_finished", "start_seq": started["seq"], "name": name,
                      **reference, "outcome": "succeeded", "result_sha256": digest(result)}
        if name == "read_text_file":
            data = json.loads(result)
            projection.update(bytes_read=data["bytes_read"], content_sha256=digest(data["content"]),
                              complete=data["complete"])
        event = observer.record(projection)
        observer.tool_events.append(event)
        return result
