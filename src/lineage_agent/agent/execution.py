"""Durable, application-owned operations; no framework task queue is serialized."""

from copy import deepcopy
from uuid import uuid4, uuid5, NAMESPACE_URL
import json

from lineage_agent.archive import ArchiveError
from lineage_agent.agent.checkpointing import CheckpointError
from lineage_agent.agent.records import RecordStore
from lineage_agent.models.responses import ModelError, capture_complete_response, parse_response
from lineage_agent.tools.execution import execute_calls
from lineage_agent.tools.types import ToolArgumentError, ToolExecutionError


class ExecutionPaused(BaseException):
    """A recoverable policy stop, never an invented model/tool error result."""

    def __init__(self, reason: str):
        self.code = reason
        super().__init__(reason)


def archived_usage(archive, run_id, *, summary=False):
    """Known totals plus unknown attempts, including results not yet adopted."""
    from lineage_agent.models.responses import _parse_usage
    total = {"input_tokens": None, "output_tokens": None, "total_tokens": None, "missing_usage_calls": 0}
    for operation in archive.operations(run_id):
        if operation["kind"] not in ({"summary"} if summary else {"model", "summary"}):
            continue
        for attempt in archive.attempts(operation["operation_id"]):
            result = attempt.get("result") or {}
            usage = (_parse_usage(result["raw_response"].get("usage")) if "raw_response" in result else
                     (result.get("model_result") or {}).get("usage"))
            missing = False
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = (usage or {}).get(key)
                if value is None:
                    missing = True
                else:
                    total[key] = (total[key] or 0) + value
            total["missing_usage_calls"] += int(missing)
    return total


class DurableExecution:
    def __init__(self, archive, run_id, settings, *, artifacts, executor, binding,
                 resuming=False):
        self.archive, self.run_id, self.settings = archive, run_id, settings
        self.artifacts, self.executor, self.binding = artifacts, executor, binding
        self.resuming = resuming
        self.budget_executor = executor
        while hasattr(self.budget_executor, "_delegate"):
            self.budget_executor = self.budget_executor._delegate
        self.restore_budget()

    def restore_budget(self):
        budget = self.budget_executor
        if not callable(getattr(budget, "restore_read_budget", None)):
            return
        saved = budget.export_read_budget()
        saved.update(used_bytes=0, reservations={}, settled={})
        for operation in self.archive.operations(self.run_id):
            for attempt in self.archive.attempts(operation["operation_id"]):
                if "read_bytes" not in attempt["reservation"]:
                    continue
                identity = attempt["attempt_id"]
                result = attempt.get("result")
                if result is None:
                    saved["reservations"][identity] = attempt["reservation"]["read_bytes"]
                else:
                    used = result.get("reservation_used", {}).get("read_bytes", 0)
                    saved["settled"][identity] = used
                    saved["used_bytes"] += used
        budget.restore_read_budget(saved)

    def prepare_request(self, state, request, *, kind="model", return_node="model", metadata=None):
        operation_id = str(uuid5(NAMESPACE_URL, f"{self.run_id}:{state['execution']['point_id']}:{kind}:{return_node}"))
        metadata = deepcopy(metadata)
        if metadata is not None:
            def stable_diagnostics(value):
                if isinstance(value, dict):
                    return {key: operation_id if key == "step_id" else stable_diagnostics(item)
                            for key, item in value.items()}
                if isinstance(value, list):
                    return [stable_diagnostics(item) for item in value]
                return value
            metadata = stable_diagnostics(metadata)
        self.archive.save_operation(self.run_id, operation_id, kind, {
            "request": deepcopy(request), "view_id": state["resource_view_id"],
            "return_node": return_node, "metadata": metadata,
        })
        return {"phase": "request_ready", "operation_id": operation_id, "return_node": return_node}

    def request(self, state):
        return self.archive.operation(state["execution"]["operation_id"])

    async def send_request(self, state, call_model, call_summary, *, before_send=None):
        operation = self.request(state)
        attempts = self.archive.attempts(operation["operation_id"])
        completed = next((a for a in attempts if a.get("result") is not None), None)
        if completed is None:
            if len(attempts) >= 2:
                raise ExecutionPaused("recovery_retry_exhausted")
            kind = operation["kind"]
            if kind == "model":
                started = sum(len(self.archive.attempts(o["operation_id"]))
                              for o in self.archive.operations(self.run_id) if o["kind"] == "model")
                if started >= self.settings.max_model_calls:
                    return {"stop_reason": "max_model_calls", "error": None, "error_code": None,
                            "model_calls": started, "usage_totals": self.usage(),
                            "attempt_counts": self.counts(),
                            "execution": {"phase": "finalizing"}}
            if before_send is not None:
                rejected = before_send(operation)
                if rejected is not None:
                    return rejected
            attempt = self.archive.start_attempt(operation["operation_id"],
                         {"model_calls" if kind == "model" else "summary_calls": 1})
            payload = operation["payload"]

            def save_raw(raw):
                self.archive.finish_attempt(attempt["attempt_id"], {"raw_response": deepcopy(raw)})

            try:
                with capture_complete_response(save_raw):
                    result = (await call_model(payload["request"]) if kind == "model" else
                              await call_summary(payload["request"], payload["metadata"]))
            except (CheckpointError, ArchiveError):
                raise
            except ModelError as exc:
                if not any(a.get("result") is not None for a in self.archive.attempts(operation["operation_id"])):
                    self.archive.finish_attempt(attempt["attempt_id"], {
                        "model_result": deepcopy(exc.result), "error_code": exc.code,
                    })
            except Exception:
                if not any(a.get("result") is not None for a in self.archive.attempts(operation["operation_id"])):
                    self.archive.finish_attempt(attempt["attempt_id"], {"model_result": None, "error_code": "model_error"})
            else:
                if not any(a.get("result") is not None for a in self.archive.attempts(operation["operation_id"])):
                    self.archive.finish_attempt(attempt["attempt_id"], {"model_result": deepcopy(result)})
        return {"execution": {**state["execution"], "phase": "response_saved"},
                "attempt_counts": self.counts()}

    def model_result(self, state):
        operation = self.request(state)
        attempts = self.archive.attempts(operation["operation_id"])
        done = [a for a in attempts if a.get("result") is not None]
        if len(done) != 1:
            raise CheckpointError("archive_integrity_error")
        result = done[0]["result"]
        if "raw_response" in result:
            return parse_response(result["raw_response"])
        if result.get("error_code"):
            raise ModelError(code=result["error_code"], result=result.get("model_result"))
        return deepcopy(result["model_result"])

    def attempt_count(self, state):
        return len(self.archive.attempts(state["execution"]["operation_id"]))

    def counts(self):
        result = {"model": 0, "summary": 0, "tool": 0, "unknown": 0}
        for operation in self.archive.operations(self.run_id):
            attempts = self.archive.attempts(operation["operation_id"])
            if operation["kind"] in result:
                result[operation["kind"]] += len(attempts)
            result["unknown"] += sum(a.get("result") is None for a in attempts)
        return result

    def usage(self, *, summary=False):
        return archived_usage(self.archive, self.run_id, summary=summary)

    def plan_tools(self, state):
        plan_id = str(uuid5(NAMESPACE_URL, f"{self.run_id}:{state['execution']['point_id']}:tool_plan"))
        calls = deepcopy(state["result"]["tool_calls"])
        operation_ids = [str(uuid5(NAMESPACE_URL, f"{plan_id}:{index}")) for index in range(len(calls))]
        self.archive.save_operation(self.run_id, plan_id, "tool_plan", {
            "calls": calls, "operation_ids": operation_ids, "view_id": state["resource_view_id"],
        })
        for index, (call, identity) in enumerate(zip(calls, operation_ids)):
            self.archive.save_operation(self.run_id, identity, "tool", {
                "call": call, "index": index, "plan_id": plan_id, "view_id": state["resource_view_id"],
            })
        return {"phase": "tool_plan_ready", "operation_id": plan_id, "tool_index": 0}

    def policy(self, name):
        if name in {"artifact_read", "artifact_search", "history_read", "history_search"}:
            return "safe_repeat"
        if self.binding.get("provider") in {"builtin.filesystem", "builtin.text", "builtin.directory"}:
            return self.binding.get("recovery", {}).get("tools", {}).get(name, "pause")
        recovery = self.binding.get("recovery") or {}
        if recovery.get("stateless") is True:
            return recovery.get("tools", {}).get(name, "pause")
        return "pause"

    async def tool_step(self, state, *, available_tools, recorder, step_id, tracer,
                        observation_manager, observation_hint_tools, native_recall_tools):
        progress = state["execution"]
        plan = self.archive.operation(progress["operation_id"])["payload"]
        index = progress["tool_index"]
        operation_id = plan["operation_ids"][index]
        call = plan["calls"][index]
        processed = self.archive.operation_result(operation_id)
        if processed is None:
            saved_raw = {}

            async def invoke(name, arguments):
                attempts = self.archive.attempts(operation_id)
                complete = next((a for a in attempts if a.get("result") is not None), None)
                if complete is None:
                    if (attempts or self.resuming) and self.policy(name) != "safe_repeat":
                        raise ExecutionPaused("tool_recovery_unsupported")
                    if len(attempts) >= 2:
                        raise ExecutionPaused("recovery_retry_exhausted")
                    attempt_id = str(uuid4())
                    reservation = {"tool_calls": 1}
                    budget = self.budget_executor
                    reading = name == "read_text_file" and callable(getattr(budget, "reserve_read", None))
                    if reading:
                        reservation["read_bytes"] = budget.reserve_read(attempt_id)
                    attempt = self.archive.start_attempt(operation_id, reservation, attempt_id=attempt_id)
                    artifact_id = attempt_id.replace("-", "")
                    result = None
                    error = None
                    try:
                        if reading:
                            raw = await budget.execute_reserved(attempt_id, name, arguments)
                        elif name in {"history_read", "history_search"}:
                            from lineage_agent.tools.history import HistoryExecutor
                            records = RecordStore(archive=self.archive, view_id=plan["view_id"])
                            try:
                                raw = await HistoryExecutor(records)(name, arguments)
                            finally:
                                records.close()
                        elif name in {"artifact_read", "artifact_search"}:
                            from lineage_agent.artifacts.tools import ArtifactExecutor
                            raw = await ArtifactExecutor(self.artifacts, view_provider=lambda: plan["view_id"])(name, arguments)
                        else:
                            raw = await self.executor(name, arguments)
                        if type(raw) is not str:
                            raise TypeError("tool result")
                    except ToolArgumentError as exc:
                        error = {"type": "argument", "code": exc.reason_code}
                    except ToolExecutionError as exc:
                        error = {"type": "execution", "code": exc.reason_code}
                    except Exception:
                        error = {"type": "execution", "code": "execution_failed"}
                    if error is not None:
                        result = {"error": error, "reservation_used": {"read_bytes": 0} if reading else {}}
                    else:
                        try:
                            metadata = self.artifacts.put_candidate(raw, tool_name=name, artifact_id=artifact_id,
                                                                   operation_id=operation_id, attempt_id=attempt_id)
                        except Exception as exc:
                            from lineage_agent.artifacts.store import ArtifactError
                            failure = exc if isinstance(exc, ArtifactError) else ArtifactError("artifact_write_error")
                            failure.execution_fact = {"call_id": call["call_id"], "name": name,
                                "request_index": index + 1, "outcome": "succeeded", "stage": "observation",
                                "reason_code": "artifact_write_error", "executor_entered": True,
                                "protocol_output_created": False}
                            failure.attempt_id = attempt_id
                            raise failure from None
                        used = json.loads(raw)["bytes_read"] if reading else 0
                        result = {"artifact": metadata, "reservation_used": {"read_bytes": used} if reading else {},
                                  "dependencies": {"artifacts": [metadata["artifact_ref"]]}}
                    try:
                        self.archive.finish_attempt(attempt_id, result)
                    except Exception as exc:
                        exc.execution_fact = {"call_id": call["call_id"], "name": name,
                            "request_index": index + 1,
                            "outcome": "succeeded" if error is None else "rejected" if error["type"] == "argument" else "failed",
                            "stage": "argument_validation" if error is not None and error["type"] == "argument" else "execution",
                            "reason_code": getattr(exc, "code", "archive_write_error"),
                            **({"tool_reason_code": error["code"]} if error is not None else {}),
                            "executor_entered": True, "protocol_output_created": False}
                        exc.attempt_id = attempt_id
                        raise
                    if reading:
                        budget.settle_read(attempt_id, result["reservation_used"]["read_bytes"])
                    complete = {**attempt, "result": result}
                result, attempt_id = complete["result"], complete["attempt_id"]
                if "error" in result:
                    error = result["error"]
                    cls = ToolArgumentError if error["type"] == "argument" else ToolExecutionError
                    raise cls(reason_code=error["code"])
                metadata = result["artifact"]
                ref = metadata["artifact_ref"]
                raw = self.artifacts.read_candidate_raw(ref, operation_id=operation_id, attempt_id=attempt_id,
                                                        sha256=metadata.get("sha256"))
                saved_raw.update(ref=ref, attempt_id=attempt_id, metadata=metadata)
                return raw

            class SavedObservation:
                def observe(_, raw, *, tool_name, keywords=(), native_recall=False):
                    return observation_manager.observe_saved(saved_raw["ref"], operation_id=operation_id,
                        attempt_id=saved_raw["attempt_id"], sha256=saved_raw["metadata"].get("sha256"),
                        keywords=keywords, native_recall=native_recall)

            processed = await execute_calls([call], available_tools, invoke, recorder=recorder,
                step_id=step_id, tracer=tracer, observation_manager=SavedObservation(),
                observation_hint_tools=observation_hint_tools, native_recall_tools=native_recall_tools,
                recoverable=True, request_index_offset=index,
                reused_result=any(attempt.get("result") is not None for attempt in self.archive.attempts(operation_id)))
            processed["tool_calls"] = len(self.archive.attempts(operation_id))
            self.archive.save_operation_result(operation_id, processed)
        # Only a reference actually disclosed by this observation is eligible
        # for business recall. Inline/native results still have private durable
        # candidates for replay, without recursively granting new Artifacts.
        disclosed = set()
        for item in processed["output_items"]:
            try:
                output = json.loads(item["output"])
                observation = output.get("observation", {})
                if type(observation.get("artifact_ref")) is str:
                    disclosed.add(observation["artifact_ref"])
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        for attempt in self.archive.attempts(operation_id):
            raw = attempt.get("result") or {}
            if "artifact" in raw and raw["artifact"]["artifact_ref"] in disclosed:
                self.artifacts.promote_candidate(raw["artifact"]["artifact_ref"], operation_id=operation_id,
                                                 attempt_id=attempt["attempt_id"])
        return {"execution": {**progress, "tool_index": index + 1,
                "phase": "group_ready" if index + 1 == len(plan["calls"]) or processed["error"] else "tool_result_saved"},
                "attempt_counts": self.counts()}

    def tool_batch(self, state, *, recorder=None, tracer=None, step_id=None):
        plan = self.archive.operation(state["execution"]["operation_id"])["payload"]
        batch = {"output_items": [], "tool_calls": 0, "call_facts": [],
                 "tool_outcomes": {"succeeded": 0, "failed": 0, "rejected": 0, "skipped": 0}, "error": None}
        for index, identity in enumerate(plan["operation_ids"]):
            item = self.archive.operation_result(identity)
            if item is None:
                if batch["error"]:
                    call = plan["calls"][index]
                    batch["tool_outcomes"]["skipped"] += 1
                    batch["call_facts"].append({"call_id": call["call_id"], "name": call["name"],
                        "request_index": index + 1, "outcome": "skipped", "stage": "recording",
                        "reason_code": "logging_error", "executor_entered": False,
                        "protocol_output_created": False, "output": None})
                    if recorder is not None:
                        recorder.emit("tool.skipped", step_id=step_id, node="tools", call_id=call["call_id"],
                                      tool_name=call["name"], request_index=index + 1, reason="logging_error")
                    if tracer is not None:
                        tracer.tool_skipped(call, step_id=step_id, request_index=index + 1, reason="logging_error")
                    continue
                raise CheckpointError("archive_integrity_error")
            for key in ("output_items", "call_facts"):
                batch[key].extend(item[key])
            batch["tool_calls"] += item["tool_calls"]
            for key, value in item["tool_outcomes"].items():
                batch["tool_outcomes"][key] += value
            batch["error"] = batch["error"] or item["error"]
        return batch
