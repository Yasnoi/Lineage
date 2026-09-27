"""Durable group events and recovery of a validated prefix after interruption."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time

from benchmarks.common import utc_now
from scripts.check_support import valid_identifiers


def plan_digest(groups):
    return hashlib.sha256(json.dumps(groups, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def planned_groups(definition):
    return [{**deepcopy(group), "planned_index": index, "started_at": None, "ended_at": None,
             "execution_status": "not_run", "verdict": "pending_review",
             "assessment_basis": "已列入计划，尚未执行。", "issues": [], "evidence": [],
             "coverage_gaps": [], "observations": [], "output_ranges": {}, "duration_seconds": None}
            for index, group in enumerate(definition["groups"], 1)]


class GroupJournal:
    def __init__(self, path, run_id, definition):
        self.run_id = run_id
        self.definition = definition
        self.sequence = 0
        self.index = 0
        self.active = None
        self.failed = False
        self.coverage_gaps = False
        self.started = time.perf_counter()
        self.stream = Path(path).open("xb")
        try:
            self.emit("unit_started", None, {"plan_digest": plan_digest(definition["groups"]),
                                            "planned_groups": [g["group_id"] for g in definition["groups"]]})
        except BaseException:
            self.stream.close()
            raise

    def emit(self, event, group_id, payload):
        if self.failed:
            raise OSError("group journal already failed")
        try:
            if event == "group_started":
                if self.active is not None or self.index >= len(self.definition["groups"]):
                    raise ValueError("unexpected group start")
                if group_id != self.definition["groups"][self.index]["group_id"]:
                    raise ValueError("group differs from declared order")
                self.active = group_id
            elif event != "unit_started" and (self.active is None or group_id != self.active):
                raise ValueError("observation outside its active group")
            record = {"schema_version": "2", "run_id": self.run_id, "unit_id": self.definition["unit_id"],
                      "sequence": self.sequence + 1, "event": event, "group_id": group_id,
                      "timestamp": utc_now(), "elapsed_seconds": time.perf_counter() - self.started,
                      "payload": payload}
            encoded = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            self.stream.write(encoded)
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.sequence += 1
            if event == "coverage_gap" and payload["applicable"]:
                self.coverage_gaps = True
            if event == "group_ended":
                self.active = None
                self.index += 1
        except BaseException:
            self.failed = True
            raise

    def close(self):
        self.stream.close()


def read_group_events(path, run_id, definition):
    """Never repair a corrupt suffix using console output or later event records."""
    groups = planned_groups(definition)
    errors = []
    active = None
    index = 0
    count = 0
    starts = {}
    previous_elapsed = -1

    def offsets(payload, start=None):
        value = payload.get("output_offsets")
        if (not isinstance(value, dict) or set(value) != {"stdout", "stderr"}
                or any(item is not None and (type(item) is not int or item < 0) for item in value.values())):
            raise ValueError("invalid output offsets")
        if start and any(value[key] is not None and start[key] is not None and value[key] < start[key] for key in value):
            raise ValueError("output offsets moved backwards")
        return value
    try:
        with Path(path).open("rb") as stream:
            for raw in stream:
                record = json.loads(raw)
                if not raw.endswith(b"\n") or not isinstance(record, dict):
                    raise ValueError("incomplete event record")
                if type(record.get("sequence")) is not int or any(record.get(key) != value for key, value in (
                    ("schema_version", "2"), ("run_id", run_id),
                    ("unit_id", definition["unit_id"]), ("sequence", count + 1),
                )):
                    raise ValueError("event identity or sequence mismatch")
                payload, event = record["payload"], record["event"]
                elapsed = record["elapsed_seconds"]
                if (not isinstance(payload, dict) or not isinstance(record["timestamp"], str)
                        or not record["timestamp"] or type(elapsed) not in (int, float)
                        or not math.isfinite(elapsed) or elapsed < 0 or elapsed < previous_elapsed):
                    raise ValueError("invalid event metadata")
                if count == 0:
                    if (event != "unit_started" or record["group_id"] is not None
                            or payload != {"plan_digest": plan_digest(definition["groups"]),
                                           "planned_groups": [g["group_id"] for g in definition["groups"]]}):
                        raise ValueError("event plan mismatch")
                elif event == "group_started":
                    if active is not None or index >= len(groups) or record["group_id"] != groups[index]["group_id"]:
                        raise ValueError("unexpected group start")
                    output_offsets = offsets(payload)
                    active = groups[index]
                    starts[index] = elapsed
                    active["started_at"] = record["timestamp"]
                    active["output_ranges"]["start"] = output_offsets
                else:
                    if active is None or active["group_id"] != record["group_id"]:
                        raise ValueError("event does not belong to active group")
                    if event == "observation":
                        if (set(payload) != {"kind", "identifiers"} or not isinstance(payload.get("kind"), str)
                                or not payload["kind"] or not valid_identifiers(payload.get("identifiers"))):
                            raise ValueError("invalid observation")
                        active["observations"].append(payload)
                        log_path = payload["identifiers"].get("log_path")
                        if log_path:
                            active["evidence"].append({"kind": "original_log", "path": log_path,
                                                       "source": payload["kind"], "availability": "unknown"})
                    elif event == "evidence":
                        if (set(payload) != {"kind", "path", "identifiers"} or not isinstance(payload.get("path"), str)
                                or not payload["path"] or not isinstance(payload.get("kind"), str)
                                or not valid_identifiers(payload.get("identifiers"))):
                            raise ValueError("invalid evidence")
                        active["evidence"].append({**payload, "source": active["group_id"], "availability": "unknown"})
                    elif event == "coverage_gap":
                        if (type(payload.get("applicable")) is not bool or not isinstance(payload.get("scenario"), str)
                                or not isinstance(payload.get("reason"), str)):
                            raise ValueError("invalid coverage gap")
                        active["coverage_gaps"].append(payload)
                    elif event == "group_ended":
                        outcome = payload.get("outcome")
                        if outcome not in {"passed", "assertion_failed", "error"}:
                            raise ValueError("invalid group outcome")
                        error = payload.get("error")
                        if ((outcome == "passed" and error is not None)
                                or (outcome != "passed" and not isinstance(error, dict))):
                            raise ValueError("invalid group error")
                        if outcome == "assertion_failed" and error.get("exception_type") != "AssertionError":
                            raise ValueError("unconfirmed assertion failure")
                        if error and (error.get("stage") != "check_execution" or error.get("code") != "check_error"
                                      or not isinstance(error.get("exception_type"), str) or not error["exception_type"]):
                            raise ValueError("invalid failure location")
                        output_offsets = offsets(payload, active["output_ranges"]["start"])
                        active.update(ended_at=record["timestamp"], duration_seconds=elapsed - starts[index],
                                      execution_status="failed" if outcome == "error" else "completed",
                                      verdict={"passed": "passed", "assertion_failed": "failed", "error": "pending_review"}[outcome])
                        active["assessment_basis"] = "组及关联后置检查已完成。" if outcome == "passed" else "检查组未通过，见失败环节及原始输出。"
                        if outcome == "passed" and any(g["applicable"] for g in active["coverage_gaps"]):
                            active.update(verdict="pending_review", assessment_basis="部分适用样例受环境阻塞，覆盖不完整。")
                        if error:
                            active["issues"].append(error)
                        active["output_ranges"]["end"] = output_offsets
                        active = None
                        index += 1
                    else:
                        raise ValueError("unknown group event")
                count += 1
                previous_elapsed = elapsed
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append({"stage": "group_recording", "code": "invalid_event_stream", "exception_type": type(exc).__name__,
                       "message": "组记录缺失或后缀损坏，仅保留已校验前缀。", "evidence": [str(path)]})
    if not count and not errors:
        errors.append({"stage": "group_recording", "code": "empty_event_stream", "message": "没有有效开始记录。"})
    if active is not None:
        active.update(execution_status="failed", verdict="pending_review", assessment_basis="组已开始但没有有效结束记录。")
        active["issues"].append({"stage": "execution", "code": "unfinished_group"})
    stop_reason = "前序组断言失败，当前脚本按约定停止。" if any(g["verdict"] == "failed" for g in groups) else "入口或前序执行未完成。"
    for group in groups:
        if group["execution_status"] == "not_run":
            group["assessment_basis"] = stop_reason
        seen = {}
        unique = []
        for item in group["evidence"]:
            key = (item["kind"], item["path"])
            if key in seen:
                previous = seen[key]
                old_ids, new_ids = previous.get("identifiers", {}), item.get("identifiers", {})
                if not any(name in old_ids and old_ids[name] != value for name, value in new_ids.items()):
                    if old_ids or new_ids:
                        previous["identifiers"] = {**old_ids, **new_ids}
                    continue
                # Different observed identities for the same path are separate references.
            seen[key] = item
            try:
                item["availability"] = "available" if Path(item["path"]).exists() else "missing"
            except OSError:
                item["availability"] = "unknown"
            unique.append(item)
        group["evidence"] = unique
    return groups, errors, count
