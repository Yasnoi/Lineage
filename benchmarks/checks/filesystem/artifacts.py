"""Offline contracts for immutable, session-bound tool-result artifacts."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from benchmarks.checks.support.events import check_group, observe_turn
from benchmarks.checks.support.runtime import (
    LocalClient, SETTINGS, audit_recording, call, message, outputs, response,
)
from benchmarks.checks.support.tracing import capture
from lineage_agent.agent import runtime
from lineage_agent.agent.sessions import SessionManager
from lineage_agent.artifacts.store import ArtifactError, ArtifactStore
from lineage_agent.config.settings import ConfigurationError
from lineage_agent.tools.filesystem import paths
from lineage_agent.tools.filesystem.directory import create_directory_executor
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL, create_text_executor
from lineage_agent.tools.types import ToolArgumentError


MAX_BYTES = 8192
PRIVATE = "ARTIFACT_PRIVATE_EXCEPTION_NEVER_DISCLOSE"
BODY_MARKER = "ARTIFACT_BODY_PRIVATE_7df293"
SETTINGS = replace(SETTINGS, context_budget_tokens=250000)


@contextmanager
def fixture():
    with TemporaryDirectory(prefix="lineage-artifact-contract-") as temporary:
        root = Path(temporary)
        with patch.object(paths, "ARTIFACT_CACHE_ROOT", root / "artifact-cache"):
            yield root


@contextmanager
def error(code):
    try:
        yield
    except ArtifactError as exc:
        assert exc.code == code, (exc.code, code)
        assert PRIVATE not in str(exc)
    else:
        raise AssertionError(f"expected ArtifactError: {code}")


def wrapped(value):
    result = json.dumps(value, ensure_ascii=False, allow_nan=False)
    return json.dumps({"ok": True, "result": result}, ensure_ascii=False, allow_nan=False)


def bounded(value):
    assert len(wrapped(value).encode("utf-8")) <= MAX_BYTES


def raw_path(store, descriptor):
    return store._directory / (descriptor["artifact_ref"].rsplit("/", 1)[1] + ".raw")


def read_all(view, descriptor, expected):
    start, pieces = 0, []
    while True:
        page = view.read(descriptor["artifact_ref"], start, None)
        bounded(page)
        assert page["artifact_ref"] == descriptor["artifact_ref"]
        assert page["view_kind"] == descriptor["view_kind"]
        assert page["total_chars"] == len(expected)
        assert page["start"] == start and start <= page["end"] <= len(expected)
        assert page["text"] == expected[start:page["end"]]
        pieces.append(page["text"])
        if page["next_start"] is None:
            assert page["end"] == len(expected)
            break
        assert page["next_start"] == page["end"] > start
        start = page["next_start"]
    assert "".join(pieces) == expected
    return pieces


def check_saving():
    with fixture() as root, ArtifactStore() as store:
        view = store.bind_session(str(uuid4()))
        content = '中文 "引号" \\ 路径\r\n第二行\n末行\r😀'
        source = root / "source.txt"
        source.write_bytes(content.encode("utf-8"))
        raw = asyncio.run(create_text_executor(root)("read_text_file", {"path": source.name}))
        descriptor = view.put(raw, tool_name="read_text_file")
        assert descriptor["tool_name"] == "read_text_file"
        assert descriptor["raw_bytes"] == len(raw.encode("utf-8"))
        assert descriptor["view_kind"] == "content" and descriptor["total_chars"] == len(content)
        assert raw_path(store, descriptor).read_bytes() == raw.encode("utf-8")
        source.write_text("changed after snapshot", encoding="utf-8")
        read_all(view, descriptor, content)
        # Unknown tools are never decoded by guessing a coincidental content key.
        unknown = view.put(raw, tool_name="custom_tool")
        assert unknown["view_kind"] == "raw_result"
        read_all(view, unknown, raw)
        for malformed in ('not JSON', '{}', '{"content":42}', '{"content":null}', '[]'):
            fallback = view.put(malformed, tool_name="read_text_file")
            assert fallback["view_kind"] == "raw_result" and fallback["degradation_reason"]
            assert raw_path(store, fallback).read_bytes() == malformed.encode("utf-8")
            read_all(view, fallback, malformed)
        # A valid content field does not make non-standard JSON constants valid,
        # including constants nested in unrelated objects or arrays.
        for constant in ("NaN", "Infinity", "-Infinity"):
            for template in ('{"content":"正文","other":CONSTANT}',
                             '{"content":"正文","other":[{"value":CONSTANT}]}'):
                malformed = template.replace("CONSTANT", constant)
                fallback = view.put(malformed, tool_name="read_text_file")
                assert fallback["view_kind"] == "raw_result", (constant, malformed)
                assert fallback["degradation_reason"] == "invalid_text_result"
                assert fallback["total_chars"] == len(malformed)
                assert fallback["raw_bytes"] == len(malformed.encode("utf-8"))
                assert raw_path(store, fallback).read_bytes() == malformed.encode("utf-8")
                read_all(view, fallback, malformed)
                page = view.read(fallback["artifact_ref"], 0)
                assert page["degradation_reason"] == "invalid_text_result"
                found = view.search(fallback["artifact_ref"], ["other"])
                bounded(found)
                assert found["view_kind"] == "raw_result"
                assert found["total_chars"] == len(malformed)
                assert found["degradation_reason"] == "invalid_text_result"
                assert not found["has_more"] and found["next_start"] is None
                assert len(found["hits"]) == 1
                hit = found["hits"][0]
                position = malformed.index("other")
                assert hit["matches"] == [{"keyword_index": 0, "start": position, "end": position + 5}]
                assert hit["text"] == malformed[hit["start"]:hit["end"]]
            # The same words are ordinary text when quoted as JSON strings.
            valid = json.dumps({"content": constant, "other": {"value": constant}}, allow_nan=False)
            quoted = view.put(valid, tool_name="read_text_file")
            assert quoted["view_kind"] == "content" and quoted["total_chars"] == len(constant)
            assert "degradation_reason" not in quoted
            read_all(view, quoted, constant)
            page = view.read(quoted["artifact_ref"], 0)
            assert "degradation_reason" not in page
            found = view.search(quoted["artifact_ref"], [constant])
            assert found["view_kind"] == "content" and "degradation_reason" not in found
            assert found["hits"][0]["matches"] == [{"keyword_index": 0, "start": 0, "end": len(constant)}]
        empty = view.put('{"content":""}', tool_name="read_text_file")
        assert empty["view_kind"] == "content"
        read_all(view, empty, "")
        repeated = view.put(raw, tool_name="read_text_file")
        assert repeated["artifact_ref"] != descriptor["artifact_ref"]
        for invalid in (None, 42, b"bytes", "unpaired\ud800"):
            with error("invalid_artifact_result"):
                view.put(invalid, tool_name="custom_tool")


def check_reading():
    with fixture(), ArtifactStore() as store:
        view = store.bind_session(str(uuid4()))
        # One line, four-byte characters and JSON double-escaping all matter.
        for content in ("a" * 30000, '中😀"\\\t' * 9000, "first。\n" + "尾" * 9000):
            descriptor = view.put(json.dumps({"content": content}), tool_name="read_text_file")
            assert len(read_all(view, descriptor, content)) > 1
            part = view.read(descriptor["artifact_ref"], 1, 9)
            assert (part["start"], part["end"], part["text"]) == (1, 9, content[1:9])
            assert part["next_start"] is None
            empty = view.read(descriptor["artifact_ref"], 4, 4)
            assert empty["text"] == "" and empty["start"] == empty["end"] == 4
            assert empty["next_start"] is None
            for start, end in ((-1, None), (True, None), (0, False), (3, 2),
                               (len(content) + 1, None), (0, len(content) + 1), ("0", None)):
                with error("invalid_artifact_range"):
                    view.read(descriptor["artifact_ref"], start, end)


def expected_matches(text, keywords):
    return [(position, index, position + len(word))
            for position in range(len(text))
            for index, word in enumerate(keywords) if text.startswith(word, position)]


def check_searching():
    with fixture(), ArtifactStore() as store:
        view = store.bind_session(str(uuid4()))
        content = ('头\\"中😀 ERROR banana failed Traceback\r\n' * 70) + "Exception"
        descriptor = view.put(json.dumps({"content": content}), tool_name="read_text_file")
        keywords = ["ERROR", "ana", "an", "failed", "Exception"]
        start, actual, pages = 0, [], 0
        while True:
            page = view.search(descriptor["artifact_ref"], keywords, start=start)
            pages += 1
            bounded(page)
            assert page["artifact_ref"] == descriptor["artifact_ref"]
            assert page["view_kind"] == "content" and page["total_chars"] == len(content)
            assert len(page["hits"]) <= 10
            for hit in page["hits"]:
                assert hit["text"] == content[hit["start"]:hit["end"]]
                assert hit["matches"]
                for match in hit["matches"]:
                    position, index, end = match["start"], match["keyword_index"], match["end"]
                    assert position >= start and content[position:end] == keywords[index]
                    assert hit["start"] <= position < end <= hit["end"]
                    actual.append((position, index, end))
            if not page["has_more"]:
                assert page["next_start"] is None
                break
            assert page["hits"] and start < page["next_start"] <= len(content)
            start = page["next_start"]
        assert pages > 1
        assert actual == expected_matches(content, keywords)
        for terms in (["error"], ["NOT_PRESENT"]):
            absent = view.search(descriptor["artifact_ref"], terms)
            assert absent["hits"] == [] and not absent["has_more"] and absent["next_start"] is None
        eof = view.search(descriptor["artifact_ref"], keywords, start=len(content))
        assert eof["hits"] == [] and not eof["has_more"]
        for terms in ([], [""], "ERROR", [42], ["x"] * 9, ["x" * 129]):
            with error("invalid_artifact_keywords"):
                view.search(descriptor["artifact_ref"], terms)
        for offset in (-1, True, len(content) + 1):
            with error("invalid_artifact_range"):
                view.search(descriptor["artifact_ref"], keywords, start=offset)
        # Search itself must stay bounded even when all keyword matches align.
        long_terms = ["x" * length for length in range(121, 129)]
        dense = view.put("x" * 20000, tool_name="custom_tool")
        page = view.search(dense["artifact_ref"], long_terms)
        bounded(page)
        assert page["has_more"] and page["next_start"] > 0
        escape_terms = ['"\\\t😀' * 32] * 8
        escaped = view.put(escape_terms[0] * 30, tool_name="custom_tool")
        page = view.search(escaped["artifact_ref"], escape_terms)
        bounded(page)
        assert len(page["hits"][0]["matches"]) == 8
        assert page["has_more"] and page["next_start"] > 0
        # Regression: has_more=false,next_start=null can be larger than the
        # corresponding continuation envelope; both serializations must fit.
        material = ('\x00' * 75 + 'x') * 7 + 'a' * 17
        tricky = view.put(material, tool_name="custom_tool")
        terms, start, seen, pages = ["x"], 0, [], 0
        while True:
            page = view.search(tricky["artifact_ref"], terms, start=start)
            pages += 1
            bounded(page)
            seen.extend((match["start"], match["keyword_index"], match["end"])
                        for hit in page["hits"] for match in hit["matches"])
            if not page["has_more"]:
                assert page["next_start"] is None
                break
            assert page["next_start"] > start
            start = page["next_start"]
        assert seen == expected_matches(material, terms) and len(seen) == 7 and pages == 2


async def check_isolation():
    with fixture() as root:
        # Bind before creating storage: old executors must enforce the boundary too.
        old_executors = [create_text_executor(root), create_filesystem_executor(root)]
        directory = create_directory_executor(root)
        with ArtifactStore() as store, ArtifactStore() as other:
            sid = str(uuid4())
            owner = store.bind_session(sid)
            descriptor = owner.put(BODY_MARKER, tool_name="custom_tool")
            for view in (store.bind_session(str(uuid4())), other.bind_session(sid)):
                with error("artifact_not_found"):
                    view.read(descriptor["artifact_ref"], 0)
                with error("artifact_not_found"):
                    view.search(descriptor["artifact_ref"], [BODY_MARKER])
            for ref in ("../outside.raw", str(root / "outside.raw"), "artifact://tool/../outside",
                        "artifact://tool/" + "a" * 31, "artifact://tool/" + "g" * 32,
                        "artifact://tool/" + "a" * 32 + "/tail"):
                with error("invalid_artifact_ref"):
                    owner.read(ref, 0)
            with error("artifact_not_found"):
                owner.read("artifact://tool/" + "f" * 32, 0)
            saved = raw_path(store, descriptor)
            relative = saved.relative_to(root).as_posix()
            for executor in old_executors:
                try:
                    await executor("read_text_file", {"path": relative})
                except ToolArgumentError as exc:
                    assert exc.reason_code == "artifact_path_forbidden"
                else:
                    raise AssertionError("ordinary text tool bypassed session isolation")
            try:
                await directory("list_directory", {"path": store._directory.relative_to(root).as_posix(), "after": None})
            except ToolArgumentError as exc:
                assert exc.reason_code == "artifact_path_forbidden"
            else:
                raise AssertionError("directory tool entered artifact storage")
            for factory in (create_text_executor, create_directory_executor, create_filesystem_executor):
                try:
                    factory(store._directory)
                except (ConfigurationError, ToolArgumentError):
                    pass
                else:
                    raise AssertionError("artifact storage accepted as project root")


def check_lifecycle():
    with fixture():
        first = ArtifactStore()
        view = first.bind_session(str(uuid4()))
        descriptor = view.put("snapshot", tool_name="custom_tool")
        directory = first._directory
        with ArtifactStore() as second:
            retained = second.bind_session(str(uuid4())).put("other", tool_name="custom_tool")
            second_path = raw_path(second, retained)
            first.close()
            assert not directory.exists() and second_path.read_bytes() == b"other"
            first.close()  # Explicit shutdown is idempotent.
            for action in (lambda: view.read(descriptor["artifact_ref"], 0),
                           lambda: view.search(descriptor["artifact_ref"], ["snapshot"]),
                           lambda: view.put("new", tool_name="custom_tool")):
                with error("artifact_store_closed"):
                    action()
        assert not second_path.exists()


def check_faults():
    with fixture(), ArtifactStore() as store:
        view = store.bind_session(str(uuid4()))
        before = set(store._directory.iterdir())
        original_open = Path.open

        def failing_write(path, mode="r", *args, **kwargs):
            if path.parent == store._directory and any(flag in mode for flag in "wx"):
                raise OSError(PRIVATE)
            return original_open(path, mode, *args, **kwargs)

        with patch.object(Path, "open", failing_write), error("artifact_write_error"):
            view.put("must not publish", tool_name="custom_tool")
        assert set(store._directory.iterdir()) == before
        # A failure after physical write must also remove the unpublished temp file.
        with patch("lineage_agent.artifacts.store.os.fsync", side_effect=OSError(PRIVATE)), error("artifact_write_error"):
            view.put("written but not durable", tool_name="custom_tool")
        assert set(store._directory.iterdir()) == before
        with patch.object(Path, "rename", side_effect=OSError(PRIVATE)), error("artifact_write_error"):
            view.put("written but not published", tool_name="custom_tool")
        assert set(store._directory.iterdir()) == before
        good = view.put("retained", tool_name="custom_tool")
        saved = raw_path(store, good)

        def failing_read(path, mode="r", *args, **kwargs):
            if path == saved and "r" in mode:
                raise OSError(PRIVATE)
            return original_open(path, mode, *args, **kwargs)

        for action in (lambda: view.read(good["artifact_ref"], 0),
                       lambda: view.search(good["artifact_ref"], ["retained"])):
            with patch.object(Path, "open", failing_read), error("artifact_read_error"):
                action()
        saved.write_bytes(b"modified")
        with error("artifact_read_error"):
            view.read(good["artifact_ref"], 0)
        saved.unlink()
        with error("artifact_read_error"):
            view.read(good["artifact_ref"], 0)
        with patch("lineage_agent.artifacts.store.shutil.rmtree", side_effect=OSError(PRIVATE)), error("artifact_cleanup_error"):
            store.close()
        with error("artifact_store_closed"):
            view.put("closed after cleanup failure", tool_name="custom_tool")
        store.close()


def artifact_call(identifier, name, ref, **arguments):
    return call(identifier, name=name, arguments=json.dumps({"artifact_ref": ref, **arguments}))


async def run_turn(manager, sid, replies, scenario):
    client = LocalClient(replies)
    with patch.object(runtime, "create_client", return_value=client):
        turn = await manager.run_turn(sid, "Use only the explicitly supplied source material.")
    observe_turn(turn)
    events = audit_recording(turn.state, client, scenario=scenario)
    assert turn.checkpoint_complete and client.closed and turn.state["stop_reason"] == "completed"
    return turn, client, events


async def check_integration():
    with fixture() as root, ArtifactStore() as store:
        project = root / "project"
        project.mkdir()
        (project / "small.txt").write_bytes(b"ordinary text")
        manager = SessionManager()
        sid = manager.create_session(
            SETTINGS, tools=[TEXT_TOOL], execute_tool=create_text_executor(project, max_total_bytes=13), artifact_store=store,
        )
        view = manager.artifacts(sid)
        content = ('ERROR "中" \\ \t' * 3000) + "tail"
        descriptor = view.put(json.dumps({"content": content}), tool_name="read_text_file")
        ref = descriptor["artifact_ref"]
        first_calls = [artifact_call("artifact_first", "artifact_read", ref, start=0, end=None),
                       call("ordinary", name="read_text_file", arguments='{"path":"small.txt"}')]
        first, client, _ = await run_turn(manager, sid, [response(*first_calls), response(message())], "artifact_first_run")
        assert {tool["name"] for tool in client.requests[0]["tools"]} == {"read_text_file", "artifact_read", "artifact_search"}
        assert first.state["tool_outcomes"]["succeeded"] == 2
        first_outputs = outputs(first.state)
        assert [item["call_id"] for item in first_outputs] == ["artifact_first", "ordinary"]
        recalled = json.loads(json.loads(first_outputs[0]["output"])["result"])
        assert content.startswith(recalled["text"]) and recalled["next_start"] > 0
        assert len(first_outputs[0]["output"].encode("utf-8")) <= MAX_BYTES
        ordinary = json.loads(json.loads(first_outputs[1]["output"])["result"])
        assert ordinary["content"] == "ordinary text" and ordinary["complete"] is True
        second_calls = [artifact_call("artifact_second", "artifact_read", ref, start=recalled["next_start"], end=None),
                        artifact_call("artifact_search", "artifact_search", ref, keywords=["tail"], start=0),
                        call("ordinary_again", name="read_text_file", arguments='{"path":"small.txt"}')]
        second, next_client, _ = await run_turn(manager, sid, [response(*second_calls), response(message())], "artifact_second_run")
        assert second.state["run_id"] != first.state["run_id"]
        assert second.state["tool_outcomes"]["succeeded"] == 3
        for item in outputs(second.state)[-3:-1]:
            assert len(item["output"].encode("utf-8")) <= MAX_BYTES
            assert json.loads(json.loads(item["output"])["result"])["artifact_ref"] == ref
        assert json.loads(json.loads(outputs(second.state)[-1]["output"])["result"])["content"] == "ordinary text"
        assert outputs(first.state) == [item for item in next_client.requests[0]["input"] if item.get("type") == "function_call_output"]
        assert len(list(store._directory.glob("*.raw"))) == 1, "recall recursively archived new artifacts"
        outsider = manager.create_session(SETTINGS, artifact_store=store)
        attempts = [artifact_call("foreign", "artifact_read", ref, start=0, end=None),
                    artifact_call("invalid", "artifact_read", ref, start=-1, end=None),
                    artifact_call("extra", "artifact_read", ref, start=0, end=None, session_id=sid)]
        rejected, _, _ = await run_turn(manager, outsider, [response(*attempts), response(message())], "artifact_foreign_run")
        envelopes = [json.loads(item["output"]) for item in outputs(rejected.state)]
        assert all(not item["ok"] and "result" not in item for item in envelopes)
        assert envelopes[0]["error"]["reason_code"] == "artifact_not_found"
        plain = manager.create_session(SETTINGS)
        untouched, plain_client, _ = await run_turn(manager, plain, [response(message())], "artifact_opt_in_only")
        assert plain_client.requests[0].get("tools") is None


async def check_recording():
    with fixture(), ArtifactStore() as store:
        manager = SessionManager()
        sid = manager.create_session(SETTINGS, artifact_store=store)
        raw = (BODY_MARKER + "\n") * 1000
        descriptor = manager.artifacts(sid).put(raw, tool_name="custom_tool")
        requests = [artifact_call("private_artifact", "artifact_read", descriptor["artifact_ref"], start=0, end=None),
                    artifact_call("private_search", "artifact_search", descriptor["artifact_ref"], keywords=[BODY_MARKER], start=0)]
        with capture() as (transport, stderr):
            turn, client, events = await run_turn(manager, sid, [response(*requests), response(message())], "artifact_recording")
        assert turn.state["tool_outcomes"]["succeeded"] == 2
        for item in outputs(turn.state):
            actual = item["output"]
            assert BODY_MARKER in actual and len(actual.encode("utf-8")) <= MAX_BYTES
        assert any(BODY_MARKER in item.get("output", "") for item in client.requests[1]["input"])
        assert transport.sent, "tracing upload was not exercised"
        for serialized in (json.dumps(events, ensure_ascii=False), json.dumps(transport.sent, ensure_ascii=False), stderr.getvalue()):
            assert BODY_MARKER not in serialized and PRIVATE not in serialized
            assert str(store._directory) not in serialized


def execute():
    for name, check, asynchronous in (
        ("saving", check_saving, False), ("reading", check_reading, False),
        ("searching", check_searching, False), ("isolation", check_isolation, True),
        ("lifecycle", check_lifecycle, False), ("faults", check_faults, False),
        ("integration", check_integration, True), ("recording", check_recording, True),
    ):
        with check_group(f"filesystem.artifacts.{name}"):
            asyncio.run(check()) if asynchronous else check()
