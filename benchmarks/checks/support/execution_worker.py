"""Parent-killed workers at exact durable execution boundaries; no real model."""

import asyncio
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from threading import Event
from unittest.mock import patch
from uuid import UUID, uuid4

from benchmarks.checks.support.runtime import LocalClient, SETTINGS, TOOL, call, message, response
from lineage_agent.agent import runtime
from lineage_agent.agent.record_codec import encode_record
from lineage_agent.agent.sessions import SessionError, SessionManager
from lineage_agent.archive import ArchiveStore
from lineage_agent.models.summarization import is_summary_request, summary_request_lane
from lineage_agent.recording import recorder as recording
from lineage_agent.tools.filesystem.text import TEXT_TOOL, TextExecutor, create_text_executor
from lineage_agent.tools.types import ToolRecoveryBinding


def emit(value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def digest(value):
    return hashlib.sha256(encode_record({"value": value})).hexdigest()


def configuration(root):
    return replace(SETTINGS, application_root=root, log_dir=root / "logs",
        context_budget_tokens=24000, max_output_tokens=256, context_safety_margin_tokens=64,
        max_model_calls=25, summary_history_max_output_tokens=512, summary_run_max_output_tokens=256)


def history_files(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in (directory / "history").glob("*.jsonl")}


class Scenario:
    def __init__(self, root, name, *, resumed=False, interrupt_retry=False):
        self.root, self.name, self.resumed = root, name, resumed
        self.armed = resumed
        self.interrupt_retry = interrupt_retry
        self.manager = self.sid = None
        self.witness = root / "execution-witness.jsonl"
        self.observation = root / "external-observation.txt"

    def events(self):
        return [json.loads(line) for line in self.witness.read_text(encoding="utf-8").splitlines()] if self.witness.exists() else []

    def event(self, kind, **details):
        if not self.armed:
            return
        with self.witness.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": kind, **details}, ensure_ascii=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def binding(self):
        return None if self.name == "unsafe_tool" else ToolRecoveryBinding(
            "test.execution-observer", 1, {TOOL["name"]: "safe_repeat"})

    def options(self):
        if self.name == "read_budget":
            return ({"current_project_root": self.root} if self.resumed else
                    {"tools": [TEXT_TOOL], "execute_tool": create_text_executor(
                        self.root, max_bytes=8, max_total_bytes=20)})
        if self.name == "unknown_summary":
            return ({"current_project_root": self.root} if self.resumed else
                    {"tools": [TEXT_TOOL], "execute_tool": create_text_executor(
                        self.root, max_bytes=4096, max_total_bytes=40000)})
        return {"tools": [TOOL], "execute_tool": self.tool, "recovery_binding": self.binding()}

    def gate(self, archive, run_id, boundary):
        lifecycle = archive.run_status(run_id)
        point = (archive.execution(lifecycle["current_execution_id"])
                 if lifecycle["status"] == "published" else archive.read_execution(run_id))
        operations = self.operations(archive, run_id)
        files = history_files(archive.directory)
        compaction = point["state"].get("compaction") or {}
        generated = {lane["summary_id"] for lane in compaction.get("lanes", {}).values()
                     if lane.get("status") == "generated"}
        active = point["state"]["active_context"]
        active_summaries = {span["summary_id"] for span in active["summary_spans"]}
        accessible_summaries = set(archive.view(point["view_id"])["summaries"])
        emit({"status": "at_boundary", "scenario": self.name, "boundary": boundary,
              "conversation_id": self.sid, "run_id": run_id,
              "point_id": point["point_id"], "phase": point["phase"],
              "source_snapshot_id": point["source_snapshot_id"], "operations": operations,
              "events": self.events(), "history": {path: hashlib.sha256(data).hexdigest()
                                                     for path, data in files.items()},
              "target_history": [path for path in files if run_id in path],
              "compact_source": digest(compaction.get("snapshot")),
              **({"quota": archive.budget_projection(run_id)} if self.name == "unknown_summary" else {}),
              "candidates_active": sorted(generated & active_summaries),
              "candidates_accessible": sorted(generated & accessible_summaries),
              "candidate_ids": [lane.get("summary_id") for lane in compaction.get("lanes", {}).values()
                                if lane.get("status") in {"generated", "reused"}]})
        # The test parent terminates the OS process. This never unwinds a graph,
        # flushes a recorder, runs finally, or simulates a normal cancellation.
        Event().wait()
        raise AssertionError("unreachable")

    def operations(self, archive, run_id):
        result = []
        for operation in archive.operations(run_id):
            payload = operation["payload"]
            attempts = archive.attempts(operation["operation_id"])
            metadata = payload.get("metadata") or {}
            result.append({"id": operation["operation_id"], "kind": operation["kind"],
                "attempt_ids": [attempt["attempt_id"] for attempt in attempts],
                "completed": sum(attempt.get("result") is not None for attempt in attempts),
                "request_digest": digest(payload.get("request")),
                "call_id": (payload.get("call") or {}).get("call_id"),
                "lane": metadata.get("lane"), "source_digest": digest(metadata.get("snapshot")),
                "artifacts": [attempt["result"]["artifact"]["artifact_ref"] for attempt in attempts
                              if "artifact" in (attempt.get("result") or {})]})
        return result

    async def tool(self, name, arguments):
        assert name == TOOL["name"]
        key = arguments["key"]
        self.event("tool", key=key)
        if self.armed and not self.resumed and self.name in {"tool_partial", "unsafe_tool", "history_tail"} and key == "b":
            archive = self.manager._checkpointer.archive(self.sid)
            self.gate(archive, archive.unfinished_runs()[0]["run_id"], "tool-b-unknown")
        if self.name in {"compact_h", "compact_candidates"}:
            return "original-detail=37; " + key + "x" * 3200
        return self.observation.read_text(encoding="utf-8") + ":" + key + "x" * 5000

    def client(self):
        scenario = self

        class Client(LocalClient):
            async def create(self, **request):
                self.requests.append(deepcopy(request))
                if is_summary_request(request):
                    lane = summary_request_lane(request)
                    scenario.event("summary", lane=lane, request_digest=digest(request))
                    if scenario.name == "unknown_summary" and lane == "R" \
                            and (not scenario.resumed or scenario.interrupt_retry):
                        archive = scenario.manager._checkpointer.archive(scenario.sid)
                        scenario.gate(archive, archive.unfinished_runs()[0]["run_id"], "summary-R-in-flight")
                    return response(message("Earlier tool observations are available through history recall."))
                count = sum(event["kind"] == "model" for event in scenario.events())
                scenario.event("model", request_digest=digest(request))
                if scenario.name == "unknown_model" and (not scenario.resumed or scenario.interrupt_retry):
                    archive = scenario.manager._checkpointer.archive(scenario.sid)
                    scenario.gate(archive, archive.unfinished_runs()[0]["run_id"], "model-in-flight")
                if scenario.name in {"response_saved", "first_response", "unknown_model",
                                      "final_operation", "final_point", "final_published"}:
                    return response(message("completed saved response"))
                if scenario.name in {"compact_h", "compact_candidates"}:
                    return (response(call(f"r{count}", arguments=json.dumps({"key": f"r{count}"})))
                            if count < 6 else response(message("completed compact Run")))
                if scenario.name == "unknown_summary":
                    return (response(call(f"r{count}", name="read_text_file", arguments=json.dumps({"path": "large.txt"})))
                            if count < 6 else response(message("completed compact file Run")))
                if scenario.name == "read_budget":
                    return (response(*(call(key, name="read_text_file", arguments=json.dumps({"path": key + ".txt"}))
                                       for key in ("a", "b", "c")))
                            if count == 0 else response(message("completed bounded file batch")))
                return (response(*(call(key, arguments=json.dumps({"key": key})) for key in ("a", "b", "c")))
                        if count == 0 else response(message("completed partial tool batch")))

        return Client([])


async def start(root, scenario):
    root.mkdir(parents=True, exist_ok=True)
    scenario.observation.write_text("old-observation", encoding="utf-8")
    if scenario.name == "read_budget":
        for name in ("a", "b", "c"):
            (root / (name + ".txt")).write_bytes(b"1234")
    if scenario.name == "unknown_summary":
        (root / "large.txt").write_bytes(b"x" * 3200)
    manager = SessionManager()
    scenario.manager = manager
    scenario.sid = manager.create_session(configuration(root), **scenario.options())
    if scenario.name != "first_response":
        if scenario.name in {"compact_h", "compact_candidates", "unknown_summary"}:
            seed_client = LocalClient([response(message("historical observations " + "h" * 4200))])
        else:
            seed_client = LocalClient([response(message("seed completed"))])
        # Force chronological History order to differ from canonical JSON keys.
        with patch.object(runtime, "create_client", return_value=seed_client), \
                patch.object(recording, "uuid4", side_effect=[UUID("ffffffff-ffff-4fff-afff-ffffffffffff"), uuid4()]):
            seed = await manager.run_turn(scenario.sid, "saved seed instruction 17")
        assert seed.snapshot_status == "published", (seed.checkpoint_error, seed.state)
    scenario.armed = True
    original_finish = ArchiveStore.finish_attempt
    original_confirm = ArchiveStore.confirm_execution
    original_read = TextExecutor.execute_reserved
    original_operation = ArchiveStore.save_operation
    original_publish = ArchiveStore.publish_snapshot

    async def read(executor, attempt_id, name, arguments):
        result = await original_read(executor, attempt_id, name, arguments)
        scenario.event("read", path=arguments["path"], attempt_id=attempt_id,
                       bytes_read=json.loads(result)["bytes_read"])
        if scenario.name == "read_budget" and arguments["path"] == "b.txt":
            archive = manager._checkpointer.archive(scenario.sid)
            scenario.gate(archive, archive.unfinished_runs()[0]["run_id"], "file-read-unknown")
        return result

    def finish(archive, attempt_id, result):
        value = original_finish(archive, attempt_id, result)
        for run in archive.unfinished_runs():
            for operation in archive.operations(run["run_id"]):
                if any(attempt["attempt_id"] == attempt_id for attempt in archive.attempts(operation["operation_id"])):
                    if scenario.name in {"response_saved", "first_response"} and operation["kind"] == "model":
                        scenario.gate(archive, run["run_id"], "complete-model-result")
                    if scenario.name == "tool_raw_saved" and operation["kind"] == "tool":
                        scenario.gate(archive, run["run_id"], "complete-tool-raw")
        return value

    def confirm(archive, point_id, checkpoint_id, expected_parent):
        value = original_confirm(archive, point_id, checkpoint_id, expected_parent)
        point = archive.execution(point_id)
        if scenario.name == "final_point" and point["phase"] == "finalizing":
            scenario.gate(archive, point["run_id"], "final-point-before-publication")
        compaction = point["state"].get("compaction") or {}
        lanes = compaction.get("lanes", {})
        both = bool(lanes.get("H")) and bool(lanes.get("R"))
        if both and scenario.name == "compact_h" and compaction.get("status") == "history_ready" \
                and lanes["H"].get("status") == "generated" and lanes["R"].get("status") == "prepared":
            scenario.gate(archive, point["run_id"], "compact-H-complete")
        if both and scenario.name == "compact_candidates" and compaction.get("status") == "current_ready" \
                and all(lanes[lane].get("status") in {"generated", "reused"} for lane in ("H", "R")):
            scenario.gate(archive, point["run_id"], "compact-candidates-complete")
        return value

    def save_operation(archive, run_id, operation_id, kind, payload):
        result = original_operation(archive, run_id, operation_id, kind, payload)
        if scenario.name == "final_operation" and kind == "finalization":
            scenario.gate(archive, run_id, "final-operation-before-point")
        return result

    def publish(archive, record_id, view_id, checkpoint_id, expected_snapshot_id):
        result = original_publish(archive, record_id, view_id, checkpoint_id, expected_snapshot_id)
        if scenario.name == "final_published":
            scenario.gate(archive, result["run_id"], "published-before-display")
        return result

    with patch.object(runtime, "create_client", return_value=scenario.client()), \
            patch.object(recording, "uuid4", side_effect=[UUID("00000000-0000-4000-a000-000000000001"), uuid4()]), \
            patch.object(ArchiveStore, "finish_attempt", new=finish), \
            patch.object(ArchiveStore, "confirm_execution", new=confirm), \
            patch.object(ArchiveStore, "save_operation", new=save_operation), \
            patch.object(ArchiveStore, "publish_snapshot", new=publish), \
            patch.object(TextExecutor, "execute_reserved", new=read):
        result = await manager.run_turn(scenario.sid, "continue original task, retain condition 17")
    raise AssertionError("worker finished without reaching boundary: " + str(result.state))


async def resume(root, scenario, payload):
    sid, run_id = payload["conversation_id"], payload["run_id"]
    scenario.sid = sid
    directory = root / ".lineage" / sid
    before = history_files(directory)
    before_events = scenario.events()
    with SessionManager() as manager:
        scenario.manager = manager
        with patch.object(runtime, "create_client", side_effect=AssertionError("open executed model")):
            inspection = manager.inspect_run(sid, run_id, application_root=root)
            assert inspection.choice is not None, inspection
            opened = await manager.open_run(sid, run_id, application_root=root,
                                           choice=inspection.choice, **scenario.options())
        assert history_files(directory) == before and scenario.events() == before_events
        original_read = TextExecutor.execute_reserved

        async def read(executor, attempt_id, name, arguments):
            raw = await original_read(executor, attempt_id, name, arguments)
            scenario.event("read", path=arguments["path"], attempt_id=attempt_id,
                           bytes_read=json.loads(raw)["bytes_read"])
            return raw

        with patch.object(runtime, "create_client", return_value=scenario.client()), \
                patch.object(TextExecutor, "execute_reserved", new=read):
            result = await manager.resume_run(sid, choice=opened.choice)
            witness = scenario.events()
            repeated = await manager.resume_run(sid, choice=opened.choice)
        assert result == repeated and scenario.events() == witness
        archive = manager._checkpointer.archive(sid)
        output = {"status": result.status, "reason": result.reason,
                  "conversation_id": sid, "run_id": result.run_id,
                  "events": witness, "operations": scenario.operations(archive, run_id),
                  "inspected_quota": inspection.quota, "final_budget": archive.budget_projection(run_id),
                  "original_history_unchanged": all(path in history_files(directory)
                       and history_files(directory)[path] == data for path, data in before.items()),
                  "history_paths": list(history_files(directory)),
                  "no_open_effects": True, "duplicate_resume_no_dispatch": True}
        if scenario.name == "unknown_summary":
            point = archive.read_execution(run_id)
            compaction = point["state"].get("compaction") or {}
            candidates = {lane["summary_id"] for lane in compaction.get("lanes", {}).values()
                          if lane.get("status") in {"generated", "reused"}}
            active = {span["summary_id"] for span in point["state"]["active_context"]["summary_spans"]}
            accessible = set(archive.view(point["view_id"])["summaries"])
            output.update(compact_source=digest(compaction.get("snapshot")), candidate_ids=sorted(candidates),
                          candidates_active=sorted(candidates & active),
                          candidates_accessible=sorted(candidates & accessible))
        if result.turn is not None:
            state = result.turn.state
            assert state["run_id"] == run_id and result.turn.snapshot_status == "published", (
                result.status, result.reason, result.turn.checkpoint_error, result.turn.snapshot_status,
                state.get("error_code"), state.get("error"), state.get("execution"), state.get("compaction"))
            output.update(snapshot_status=result.turn.snapshot_status, snapshot_id=result.turn.snapshot_id,
                model_calls=state["model_calls"], summary_calls=state["summary_calls"], tool_calls=state["tool_calls"],
                usage_totals=state["usage_totals"], tool_outcomes=state["tool_outcomes"],
                final_input=state["input"], history_end=state["history_end"],
                record_sequence=state["record_sequence"], final_record=(await manager.get_record(sid, run_id)),
                published_runs=[item["run_id"] for item in archive.published_runs(result.turn.snapshot_id)])
        emit(output)


async def probe(root, scenario, payload):
    with SessionManager() as manager:
        try:
            manager.inspect_run(payload["conversation_id"], payload["run_id"], application_root=root)
        except SessionError as error:
            emit({"status": "busy", "code": error.code})
        else:
            raise AssertionError("live conversation inspection acquired writer lock")


async def published(root, scenario, payload):
    sid, run_id = payload["conversation_id"], payload["run_id"]
    scenario.sid = sid
    directory = root / ".lineage" / sid
    before, events = history_files(directory), scenario.events()
    with SessionManager() as manager:
        with patch.object(runtime, "create_client", side_effect=AssertionError("open executed model")):
            inspection = manager.inspect_run(sid, run_id, application_root=root)
            assert inspection.status == "completed" and inspection.choice is None
            opened = await manager.open_session(sid, application_root=root, **scenario.options())
            record = await manager.get_record(sid, run_id)
        archive = manager._checkpointer.archive(sid)
        from lineage_agent.archive import ArchiveError
        try:
            archive.read_execution(run_id)
        except ArchiveError as error:
            rejection = error.code
        else:
            raise AssertionError("sealed run execution head accepted")
        assert history_files(directory) == before and scenario.events() == events
        emit({"status": inspection.status, "run_id": run_id, "record_run_id": record["run_id"],
              "no_open_effects": True, "snapshot_id": opened.snapshot_id,
              "published_runs": [item["run_id"] for item in archive.published_runs(opened.snapshot_id)],
              "events": events, "old_execution_rejected": rejection})


async def main():
    mode, root, name = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    scenario = Scenario(root, name, resumed=mode != "start", interrupt_retry=mode == "retry")
    if mode == "start":
        await start(root, scenario)
    else:
        payload = json.loads(sys.stdin.readline())
        await {"resume": resume, "retry": resume, "probe": probe, "published": published}[mode](root, scenario, payload)


if __name__ == "__main__":
    asyncio.run(main())
