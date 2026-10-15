"""Real competing recovery processes and consumed-choice replay rejection."""

import asyncio
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from unittest.mock import patch

from benchmarks.checks.support.runtime import LocalClient, SETTINGS, TOOL, call, message, response
from lineage_agent.agent import runtime
from lineage_agent.agent.sessions import RunRecoveryChoice, SessionError, SessionManager
from lineage_agent.archive import ArchiveStore
from lineage_agent.storage import conversation_paths
from lineage_agent.tools.types import ToolRecoveryBinding


def _emit(value):
    print(json.dumps(value), flush=True)


async def _worker(mode, root):
    payload = json.loads(sys.stdin.readline())
    actor = payload.get("actor", mode)
    witness = root / "dispatch.jsonl"
    tool_calls = []

    def dispatched(kind):
        with witness.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"actor": actor, "kind": kind, "pid": os.getpid()}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    async def tool(name, arguments):
        assert name == TOOL["name"]
        dispatched("tool")
        tool_calls.append(arguments)
        return "trusted observation"

    class Client(LocalClient):
        async def create(self, **request):
            dispatched("model")
            return await super().create(**request)

    client = Client([response(call()), response(message("completed"))])
    options = {"tools": [TOOL], "execute_tool": tool,
               "recovery_binding": ToolRecoveryBinding("test.recovery-choices", 1, {TOOL["name"]: "safe_repeat"})}

    if mode == "seed":
        settings = replace(SETTINGS, application_root=root, log_dir=root / "logs", max_model_calls=6)
        with SessionManager() as manager:
            sid = manager.create_session(settings, **options)
            archive = manager._checkpointer.archive(sid)
            confirm = archive.confirm_execution

            def boundary(point_id, checkpoint_id, expected_parent):
                value = confirm(point_id, checkpoint_id, expected_parent)
                if value["phase"] == "request_ready":
                    assert not client.requests and not tool_calls
                    _emit({"status": "request_ready", "conversation_id": sid, "run_id": value["run_id"]})
                    # The parent terminates the process at this confirmed point.
                    sys.stdin.readline()
                    raise AssertionError("seed must be terminated externally")
                return value

            with patch.object(archive, "confirm_execution", side_effect=boundary), \
                    patch.object(runtime, "create_client", return_value=client):
                await manager.run_turn(sid, "continue this same read-only Run")
        raise AssertionError("seed unexpectedly finished")

    sid, run_id = payload["conversation_id"], payload["run_id"]
    if mode == "inspect":
        with SessionManager() as manager:
            inspected = manager.inspect_run(sid, run_id, application_root=root)
            assert inspected.status == "ready"
            _emit({"choice": asdict(inspected.choice)})
        return

    if mode == "report":
        with ArchiveStore(conversation_paths(root, sid, create=False).directory, create=False, read_only=True) as archive:
            head = archive.conversation()["current_snapshot_id"]
            _emit({"lifecycle": archive.run_status(run_id), "attempts": archive.budget_projection(run_id)["attempt_count"],
                   "published_runs": [item["run_id"] for item in archive.published_runs(head)] if head else [],
                   "receipt": archive.resume_choice(payload["choice"]["choice_id"])})
        return

    choice = RunRecoveryChoice(**payload["choice"])
    with SessionManager() as manager, patch.object(runtime, "create_client", return_value=client):
        if mode == "race":
            _emit({"status": "prepared", "actor": actor})
            assert sys.stdin.readline().strip() == "open"
        try:
            opened = await manager.open_run(sid, run_id, choice=choice, application_root=root, **options)
        except SessionError as error:
            assert not manager._sessions and not client.requests and not tool_calls
            _emit({"status": "rejected", "code": error.code, "actor": actor,
                   "model_calls": len(client.requests), "tool_calls": len(tool_calls)})
            return
        if mode == "race":
            _emit({"status": "locked", "actor": actor})
            # Hold the winner's OS lease until the other process has rejected.
            assert sys.stdin.readline().strip() == "resume"
        if mode == "consume":
            async def after_acceptance(*args, **kwargs):
                archive = manager._checkpointer.archive(sid)
                receipt = archive.resume_choice(choice.choice_id)
                assert receipt is not None and not client.requests and not tool_calls
                _emit({"status": "choice_consumed", "choice_id": choice.choice_id,
                       "revision": archive.run_status(run_id)["revision"]})
                await asyncio.to_thread(sys.stdin.readline)
                raise AssertionError("consumer must be terminated externally")

            with patch.object(runtime, "_resume_agent", side_effect=after_acceptance):
                await manager.resume_run(sid, choice=opened.choice)
            raise AssertionError("consumer unexpectedly finished")
        # A replay that erroneously opens continues here, so the test observes
        # any actual dispatch instead of hiding the regression behind a stub.
        resumed = await manager.resume_run(sid, choice=opened.choice)
        assert resumed.status == "completed" and resumed.turn.snapshot_status == "published"
        archive = manager._checkpointer.archive(sid)
        _emit({"status": resumed.status, "actor": actor, "run_id": resumed.run_id,
               "model_calls": len(client.requests), "tool_calls": len(tool_calls),
               "published_runs": [item["run_id"] for item in archive.published_runs(resumed.turn.snapshot_id)]})


class ExecutionChoiceProcessTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="execution-choices-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = {**os.environ, "PYTHONUTF8": "1", "LANGSMITH_TRACING": "false",
                            "LANGCHAIN_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}

    def command(self, mode):
        return [sys.executable, "-m", "benchmarks.tests.test_execution_choices_process", "--worker", mode, str(self.root)]

    def worker(self, mode, payload):
        result = subprocess.run(self.command(mode), input=json.dumps(payload) + "\n", capture_output=True,
                                text=True, encoding="utf-8", env=self.environment, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(len(result.stdout.splitlines()), 1, result.stdout)
        return json.loads(result.stdout)

    def start(self, mode, payload):
        child = subprocess.Popen(self.command(mode), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, encoding="utf-8", env=self.environment)
        self.addCleanup(self.close_child, child)
        self.send(child, json.dumps(payload))
        return child

    def close_child(self, child):
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=15)
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()

    def send(self, child, value):
        child.stdin.write(value + "\n")
        child.stdin.flush()

    def receive(self, child):
        ready = Queue()
        Thread(target=lambda: ready.put(child.stdout.readline()), daemon=True).start()
        try:
            line = ready.get(timeout=60)
        except Empty:
            child.kill()
            output, error = child.communicate(timeout=15)
            self.fail("worker did not reach handshake: " + output + error)
        if not line:
            output, error = child.communicate(timeout=15)
            self.fail("worker exited before handshake: " + output + error)
        return json.loads(line)

    def killed_seed(self):
        child = self.start("seed", {})
        payload = self.receive(child)
        self.assertEqual(payload["status"], "request_ready")
        child.kill()
        output, error = child.communicate(timeout=15)
        self.assertNotEqual(child.returncode, 0)
        self.assertEqual((output, error), ("", ""))
        self.assertEqual(self.dispatches(), [])
        payload.update(self.worker("inspect", payload))
        return payload

    def dispatches(self):
        path = self.root / "dispatch.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def test_two_processes_compete_to_resume_only_lease_holder_executes_and_seals_once(self):
        payload = self.killed_seed()
        children = [self.start("race", {**payload, "actor": actor}) for actor in ("first", "second")]
        for child in children:
            self.assertEqual(self.receive(child)["status"], "prepared")
        for child in children:
            self.send(child, "open")
        decisions = [self.receive(child) for child in children]
        self.assertEqual(sorted(item["status"] for item in decisions), ["locked", "rejected"])
        loser = next(item for item in decisions if item["status"] == "rejected")
        self.assertEqual(loser["code"], "conversation_busy")
        self.assertEqual((loser["model_calls"], loser["tool_calls"]), (0, 0))
        self.assertEqual(self.dispatches(), [])
        winner_index = next(index for index, item in enumerate(decisions) if item["status"] == "locked")
        self.send(children[winner_index], "resume")
        result = self.receive(children[winner_index])
        for child in children:
            output, error = child.communicate(timeout=60)
            self.assertEqual(child.returncode, 0, error)
            self.assertEqual((output, error), ("", ""))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["run_id"], payload["run_id"])
        self.assertEqual(result["published_runs"], [payload["run_id"]])
        self.assertEqual([item["kind"] for item in self.dispatches()], ["model", "tool", "model"])
        self.assertEqual({item["actor"] for item in self.dispatches()}, {result["actor"]})
        report = self.worker("report", payload)
        self.assertEqual(report["published_runs"], [payload["run_id"]])
        self.assertEqual(report["lifecycle"]["status"], "published")
        self.assertEqual(report["attempts"], 3)

    def test_consumed_choice_replayed_in_new_process_never_dispatches(self):
        payload = self.killed_seed()
        consumer = self.start("consume", payload)
        accepted = self.receive(consumer)
        self.assertEqual(accepted["status"], "choice_consumed")
        self.assertEqual(accepted["choice_id"], payload["choice"]["choice_id"])
        consumer.kill()
        output, error = consumer.communicate(timeout=15)
        self.assertNotEqual(consumer.returncode, 0)
        self.assertEqual((output, error), ("", ""))
        before = self.worker("report", payload)
        self.assertIsNotNone(before["receipt"])
        self.assertEqual(before["lifecycle"]["status"], "started")
        self.assertEqual(before["attempts"], 0)
        replayed = self.worker("replay", payload)
        self.assertEqual(replayed["status"], "rejected")
        self.assertIn(replayed["code"], {"resume_choice_used", "recovery_changed"})
        self.assertEqual((replayed["model_calls"], replayed["tool_calls"]), (0, 0))
        self.assertEqual(self.dispatches(), [])
        self.assertEqual(self.worker("report", payload), before)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        asyncio.run(_worker(sys.argv[2], Path(sys.argv[3])))
    else:
        unittest.main()
