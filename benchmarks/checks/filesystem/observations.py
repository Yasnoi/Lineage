"""Offline evidence for pre-context governance and default artifact recall."""

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

from benchmarks.checks.filesystem.artifacts import fixture, raw_path
from benchmarks.checks.support.events import check_group, observe_turn
from benchmarks.checks.support.runtime import (
    LocalClient, SETTINGS, TOOL, audit_recording, call, message, outputs, response,
)
from benchmarks.checks.support.tracing import capture
from lineage_agent.agent import runtime
from lineage_agent.agent.sessions import SessionError, SessionManager
from lineage_agent.artifacts.store import ArtifactError, ArtifactStore, SessionArtifacts
from lineage_agent.tools.filesystem.text import TEXT_TOOL, create_text_executor
from lineage_agent.tools.types import ToolArgumentError


SETTINGS = replace(SETTINGS, context_budget_tokens=250000, max_model_calls=8)
PRIVATE = "OBSERVATION_INTERNAL_EXCEPTION_PRIVATE"
BODY = "OBSERVATION_BODY_ONLY_MODEL"
INTENT = "OBSERVATION_INTENT_NOT_TRACE"
KEYWORD = "OBSERVATION_KEYWORD_NOT_TRACE"


def envelope(item):
    value = json.loads(item["output"])
    assert len(item["output"].encode("utf-8")) <= 8192
    return value


def observation(item):
    value = envelope(item)
    assert value["ok"] is True
    return value["observation"]


def tool_call(identifier, name, **arguments):
    return call(identifier, name=name, arguments=json.dumps(arguments, ensure_ascii=False))


class GuidedClient(LocalClient):
    def __init__(self, guide):
        super().__init__([])
        self.guide = guide

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return self.guide(len(self.requests), kwargs)


async def run(client, **kwargs):
    with patch.object(runtime, "create_client", return_value=client):
        state = await runtime.run_agent("Use the supplied tool results.", SETTINGS, **kwargs)
    assert client.closed
    audit_recording(state, client, scenario="observations")
    return state


async def turn(manager, sid, client):
    with patch.object(runtime, "create_client", return_value=client):
        value = await manager.run_turn(sid, "Continue with the available artifacts.")
    observe_turn(value)
    audit_recording(value.state, client, scenario="observations_session")
    assert value.checkpoint_complete
    return value.state


async def check_pipeline():
    raw = '首"\\\r\n' * 6000 + "MIDDLE_MATERIAL" + "尾😀" * 6000
    executed, published = [], []
    original_put = SessionArtifacts.put

    def save(view, value, **kwargs):
        descriptor = original_put(view, value, **kwargs)
        assert raw_path(view._store, descriptor).read_bytes() == value.encode("utf-8")
        published.append((view, descriptor, view._store._directory))
        return descriptor

    async def executor(name, arguments):
        executed.append((name, arguments))
        return raw + arguments["key"]

    def guide(number, request):
        items = [item for item in request["input"] if item.get("type") == "function_call_output"]
        assert all(raw not in item["output"] for item in items)
        if number == 1:
            names = {tool["name"] for tool in request["tools"]}
            assert names == {"lookup_sample", "artifact_read", "artifact_search"}
            return response(*(tool_call(f"large_{i}", "lookup_sample", key=str(i)) for i in range(3)))
        if number == 2:
            assert len(items) == len(published) == 3
            for item, (_, descriptor, _) in zip(items, published):
                obs = observation(item)
                assert obs["mode"] == "artifact" and obs["strategy"] == "head_tail"
                assert obs["artifact_ref"] == descriptor["artifact_ref"]
                assert obs["view_kind"] == "raw_result" and not obs["complete"]
                assert len(item["output"].encode("utf-8")) <= 4096
                assert "MIDDLE_MATERIAL" not in "".join(s["text"] for s in obs["segments"])
            ref = observation(items[0])["artifact_ref"]
            return response(
                tool_call("recall", "artifact_read", artifact_ref=ref, start=raw.index("MIDDLE_MATERIAL"), end=None),
                tool_call("search", "artifact_search", artifact_ref=ref, keywords=["MIDDLE_MATERIAL"], start=0),
            )
        assert number == 3 and len(published) == 3, "recall must not create another snapshot"
        recalled, searched = [json.loads(observation(item)["raw_result"]) for item in items[-2:]]
        assert recalled["text"].startswith("MIDDLE_MATERIAL") and recalled["next_start"] > recalled["start"]
        assert searched["hits"][0]["matches"][0]["start"] == raw.index("MIDDLE_MATERIAL")
        assert not searched["has_more"]
        assert all(observation(item)["mode"] == "inline" for item in items[-2:])
        return response(message())

    with fixture(), patch.object(SessionArtifacts, "put", save):
        client = GuidedClient(guide)
        state = await run(client, tools=[TOOL], execute_tool=executor)
        assert state["stop_reason"] == "completed" and len(executed) == 3
        assert state["tool_outcomes"]["succeeded"] == 5 and len(client.requests) == 3
        assert all(not directory.exists() for _, _, directory in published)
        for view, descriptor, _ in published:
            try:
                view.read(descriptor["artifact_ref"], 0)
            except ArtifactError as exc:
                assert exc.code == "artifact_store_closed"
            else:
                raise AssertionError("standalone artifacts outlived their run")


async def check_hints():
    content = "A" * 6000 + "ERROR cause\nTraceback details" + "Z" * 6000
    raw = json.dumps({"content": content})
    seen = []

    async def executor(name, arguments):
        seen.append(deepcopy(arguments))
        if set(arguments) != {"path"} or not isinstance(arguments["path"], str):
            raise ToolArgumentError()
        return raw

    requests = [
        tool_call("hint", "read_text_file", path="a.txt", observation={"intent": INTENT, "keywords": ["ERROR", "Traceback"]}),
        tool_call("missing", "read_text_file", path="a.txt"),
        tool_call("invalid", "read_text_file", path="a.txt", observation={"intent": 5, "keywords": ["ERROR"]}),
        tool_call("no_hit", "read_text_file", path="a.txt", observation={"intent": None, "keywords": ["ABSENT"]}),
        tool_call("bad_business", "read_text_file", path=3, observation={"intent": None, "keywords": []}),
        call("duplicate", name="read_text_file", arguments='{"path":"a.txt","observation":{"intent":null,"intent":null,"keywords":[]}}'),
    ]
    with fixture():
        client = LocalClient([response(*requests), response(message())])
        state = await run(client, tools=[TEXT_TOOL], execute_tool=executor)
        assert seen == [{"path": "a.txt"}] * 4 + [{"path": 3}]
        actual = [item for item in client.requests[1]["input"] if item.get("type") == "function_call_output"]
        observations = [observation(item) for item in actual[:4]]
        matched = observations[0]
        assert matched["strategy"] == "keywords" and matched["keyword_status"] == "matched"
        assert len(matched["segments"]) == 1 and len(matched["matches"]) == 2
        assert matched["matches_truncated"] is False and not matched["complete"]
        for segment in matched["segments"]:
            assert segment["text"] == content[segment["start"]:segment["end"]]
        assert [value["keyword_status"] for value in observations[1:]] == ["not_requested", "not_requested", "no_match"]
        assert all(value["strategy"] == "head_tail" for value in observations[1:])
        assert all(envelope(item)["ok"] is False for item in actual[4:])
        assert state["tool_outcomes"] == {"succeeded": 4, "rejected": 2, "failed": 0, "skipped": 0}


async def check_failures():
    async def executor(name, arguments):
        return "huge" * 5000 if arguments["key"] == "large" else "small"

    with fixture():
        for failure in (ArtifactError("artifact_write_error"), OSError(PRIVATE)):
            client = LocalClient([response(tool_call("save_fail", "lookup_sample", key="large"),
                                           tool_call("still_run", "lookup_sample", key="small")), response(message())])
            with capture() as (transport, stderr), patch.object(SessionArtifacts, "put", side_effect=failure) as save:
                state = await run(client, tools=[TOOL], execute_tool=executor)
            actual = [item for item in client.requests[1]["input"] if item.get("type") == "function_call_output"]
            failed = observation(actual[0])
            assert failed["status"] == "unavailable" and failed["stage"] == "save"
            assert "artifact_ref" not in failed and PRIVATE not in json.dumps(actual)
            assert observation(actual[1])["raw_result"] == "small"
            assert save.call_count == 1 and state["tool_outcomes"]["succeeded"] == 2
            assert state["stop_reason"] == "completed"
            events = [json.loads(line) for line in Path(state["log_path"]).read_text(encoding="utf-8").splitlines()]
            recorded = next(event for event in events if event["event"] == "tool.succeeded" and event["call_id"] == "save_fail")
            assert recorded["observation_status"] == "unavailable" and recorded["observation_stage"] == "save"
            spans = [span for span in transport.sent if span["run_type"] == "tool"]
            assert len(spans) == 2
            assert spans[0]["extra"]["metadata"]["outcome"] == "succeeded"
            assert spans[0]["outputs"]["observation"]["status"] == "unavailable"
            assert PRIVATE not in json.dumps(transport.sent) and PRIVATE not in stderr.getvalue()

        async def unencodable_executor(name, arguments):
            return "PRIVATE\ud800" if arguments["key"] == "large" else "small"

        client = LocalClient([response(tool_call("serialize_fail", "lookup_sample", key="large"),
                                       tool_call("after_serialize", "lookup_sample", key="small")), response(message())])
        state = await run(client, tools=[TOOL], execute_tool=unencodable_executor)
        delivered = [item for item in client.requests[1]["input"] if item.get("type") == "function_call_output"]
        assert observation(delivered[0])["stage"] == "serialize"
        assert "PRIVATE" not in json.dumps(delivered)
        assert observation(delivered[1])["raw_result"] == "small"
        assert state["tool_outcomes"]["succeeded"] == 2 and state["stop_reason"] == "completed"

        async def text_executor(name, arguments):
            return json.dumps({"content": "ERROR\n" * 5000})

        with SessionManager() as manager:
            sid = manager.create_session(SETTINGS, tools=[TEXT_TOOL], execute_tool=text_executor)
            request = tool_call("preview_fail", "read_text_file", path="a.txt", observation={"intent": None, "keywords": ["ERROR"]})
            client = LocalClient([response(request), response(message())])
            with patch.object(SessionArtifacts, "search", side_effect=OSError(PRIVATE)) as search:
                state = await turn(manager, sid, client)
            failed = observation(outputs(state)[0])
            assert failed["status"] == "unavailable" and failed["stage"] == "preview"
            assert PRIVATE not in json.dumps(failed) and search.call_count == 1
            assert manager.artifacts(sid).read(failed["artifact_ref"], 0, 5)["text"] == "ERROR"
            assert state["tool_outcomes"]["succeeded"] == 1 and state["stop_reason"] == "completed"


async def check_sessions():
    with fixture() as root, SessionManager() as manager:
        project = root / "project"
        project.mkdir()
        content = "跨轮正文\n" * 3000
        (project / "large.txt").write_bytes(content.encode("utf-8"))
        sid = manager.create_session(SETTINGS, tools=[TEXT_TOOL], execute_tool=create_text_executor(
            project, max_bytes=len(content.encode("utf-8")), max_total_bytes=len(content.encode("utf-8")),
        ))
        client = LocalClient([response(tool_call("first_read", "read_text_file", path="large.txt"),
                                       tool_call("over_quota", "read_text_file", path="large.txt")), response(message())])
        first = await turn(manager, sid, client)
        ref = observation(outputs(first)[0])["artifact_ref"]
        assert envelope(outputs(first)[1])["error"]["reason_code"] == "read_budget_exceeded"
        second_client = LocalClient([response(tool_call("old", "artifact_read", artifact_ref=ref, start=0, end=5),
                                              tool_call("fresh_quota", "read_text_file", path="large.txt")), response(message())])
        second = await turn(manager, sid, second_client)
        assert second["tool_outcomes"]["succeeded"] == 2
        recalled = json.loads(observation(outputs(second)[-2])["raw_result"])
        assert recalled["text"] == content[:5]
        other = manager.create_session(SETTINGS)
        outsider = LocalClient([response(tool_call("cross_session", "artifact_read", artifact_ref=ref, start=0, end=5)), response(message())])
        rejected = await turn(manager, other, outsider)
        assert envelope(outputs(rejected)[0])["error"]["reason_code"] == "artifact_not_found"


async def check_privacy():
    async def executor(name, arguments):
        return json.dumps({"content": (BODY + " " + KEYWORD + "\n") * 2000})

    def guide(number, request):
        if number == 1:
            return response(tool_call("private", "read_text_file", path="a.txt", observation={"intent": INTENT, "keywords": [KEYWORD]}))
        items = [item for item in request["input"] if item.get("type") == "function_call_output"]
        if number == 2:
            obs = observation(items[0])
            assert BODY in items[0]["output"] and obs["strategy"] == "keywords"
            return response(tool_call("private_recall", "artifact_read", artifact_ref=obs["artifact_ref"], start=0, end=100))
        assert number == 3 and BODY in items[-1]["output"]
        return response(message())

    with fixture(), capture() as (transport, stderr):
        client = GuidedClient(guide)
        state = await run(client, tools=[TEXT_TOOL], execute_tool=executor)
        trace = json.dumps(transport.sent, ensure_ascii=False)
        log = Path(state["log_path"]).read_text(encoding="utf-8")
        assert transport.sent and any(span["run_type"] == "tool" for span in transport.sent)
        assert sum(span["run_type"] == "llm" for span in transport.sent) == 3
        for hidden in (BODY, INTENT, KEYWORD, PRIVATE):
            assert hidden not in trace and hidden not in log and hidden not in stderr.getvalue()


async def check_lifecycle():
    with fixture():
        client = LocalClient([response(message())])
        with patch.object(runtime, "ArtifactStore", side_effect=ArtifactError("artifact_write_error")):
            with patch.object(runtime, "create_client", return_value=client):
                state = await runtime.run_agent("init failure", SETTINGS)
        assert state["stop_reason"] == "error" and state["error_code"] == "artifact_store_init_error"
        assert not client.requests and state["model_calls"] == state["tool_calls"] == 0

        original_close = ArtifactStore.close

        def failing_close(store):
            original_close(store)
            raise ArtifactError("artifact_cleanup_error")

        client = LocalClient([response(message("answer survives cleanup"))])
        with patch.object(ArtifactStore, "close", failing_close):
            state = await run(client)
        assert state["result"]["text"] == "answer survives cleanup"
        assert state["cleanup_error"]["code"] == "artifact_cleanup_error" and state["model_calls"] == 1

        with ArtifactStore() as borrowed:
            manager = SessionManager()
            sid = manager.create_session(SETTINGS, artifact_store=borrowed)
            view = manager.artifacts(sid)
            saved = view.put("borrowed survives manager", tool_name="custom")
            manager.close()
            assert view.read(saved["artifact_ref"], 0)["text"] == "borrowed survives manager"
            try:
                manager.create_session(SETTINGS)
            except SessionError:
                pass
            else:
                raise AssertionError("closed manager accepted a new session")

        manager = SessionManager()
        sid = manager.create_session(SETTINGS)
        view = manager.artifacts(sid)
        saved = view.put("owned", tool_name="custom")
        started, release = asyncio.Event(), asyncio.Event()

        class WaitingClient(LocalClient):
            async def create(self, **kwargs):
                started.set()
                await release.wait()
                return await super().create(**kwargs)

        client = WaitingClient([response(message())])
        with patch.object(runtime, "create_client", return_value=client):
            task = asyncio.create_task(manager.run_turn(sid, "wait for close attempt"))
            await started.wait()
            try:
                manager.close()
            except SessionError:
                pass
            else:
                raise AssertionError("manager closed an active turn")
            assert view.read(saved["artifact_ref"], 0)["text"] == "owned"
            release.set()
            await task
        manager.close()
        assert not view._store._directory.exists()


def execute():
    for name, check in (
        ("pipeline", check_pipeline), ("hints", check_hints),
        ("failures", check_failures), ("sessions", check_sessions),
        ("privacy", check_privacy), ("lifecycle", check_lifecycle),
    ):
        with check_group(f"filesystem.observations.{name}"):
            asyncio.run(check())
