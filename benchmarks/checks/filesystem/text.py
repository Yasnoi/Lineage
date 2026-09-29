"""filesystem / text behavior contracts."""


import asyncio
import io
import json
import os
import stat
import subprocess
import sys
from benchmarks.checks.support.events import group_id, check_group, coverage_gap, observe_state
from benchmarks.checks.support.runtime import (
    LocalClient,
    audit_recording,
    call,
    counts,
    message,
    outputs,
    recording_failure,
    response,
)
from benchmarks.checks.support.text import (
    BODY_MARKER,
    CHECK_DIR,
    RUNS,
    SETTINGS,
    SKIPPED,
    TOOLS,
    directory_call,
    fixture,
    metadata_override,
    monitor_reads,
    rejected_then_success,
    result_envelopes,
    run_case,
    snapshot,
    text_call,
)
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from lineage_agent.agent import runtime
from lineage_agent.config.settings import ConfigurationError, load_settings
from lineage_agent.entrypoints import cli
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL, create_text_executor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

async def check_reading(root):
    with monitor_reads(root) as (opened, probes):
        state, client, observed = await run_case(
            "discover_then_read_utf8_and_bom", root,
            [response(directory_call("text_discover")),
             response(text_call("text_md", "说明 PRIVATE.md"), text_call("text_bom", "docs private\\mixed space.TXT")),
             response(message("sources received"))],
        )
    counts(state, 3, 3, 3)
    listed, md, bom = [json.loads(item["result"]) for item in result_envelopes(state)]
    assert "说明 PRIVATE.md" in {item["name"] for item in listed["entries"]}
    for value, relative, encoding in (
        (md, "说明 PRIVATE.md", "utf-8"), (bom, "docs private/mixed space.TXT", "utf-8-sig"),
    ):
        raw = (root / relative).read_bytes()
        assert value == {
            "path": relative, "content": raw.decode(encoding), "encoding": encoding,
            "bytes_read": len(raw), "complete": True,
        }
    assert "\r\n" in md["content"] and "\n" in md["content"] and "\r" in md["content"]
    assert len(opened) == len(probes) == 2
    for scenario, path in (("empty_text", "empty.txt"), ("bom_only", "bom_only.txt"), ("uppercase_extension", "case.MD")):
        with monitor_reads(root):
            state, _, _ = await run_case(scenario, root, [response(text_call("text_one", path)), response(message())])
        value = json.loads(result_envelopes(state)[0]["result"])
        assert value["content"] == (root / path).read_bytes().decode("utf-8-sig")
        assert value["complete"] is True
        counts(state, 2, 1, 1)
    standalone = create_text_executor(root)
    state, client, _ = await run_case(
        "explicit_text_only", root, [response(text_call("text_only")), response(message())],
        executor=standalone, tools=[TEXT_TOOL],
    )
    assert client.requests[0]["tools"] == [TEXT_TOOL]

async def check_limits(root):
    with monitor_reads(root) as (_, probes):
        state, _, _ = await run_case("byte_limit_exact", root, [response(text_call("text_limit", "equal.txt")), response(message())])
    result = json.loads(result_envelopes(state)[0]["result"])
    assert result["content"] == "a" * 65536 and result["bytes_read"] == 65536
    assert sum(count for _, count in probes[0].reads) == 65536
    assert probes[0].reads[0][0] <= 65537
    for scenario, path in (("byte_limit_exceeded", "over.txt"), ("long_single_line_exceeded", "longline.txt")):
        await rejected_then_success(scenario, root, path=path, reason="file_too_large")
    for scenario, short in (("short_reads", 3), ("ordinary_bounded_reads", None)):
        with monitor_reads(root, short=short) as (_, probes):
            state, _, _ = await run_case(
                scenario, root, [response(text_call("text_short", "short.txt")), response(message())], max_bytes=64,
            )
        value = json.loads(result_envelopes(state)[0]["result"])
        assert value["content"].encode() == (root / "short.txt").read_bytes()
        total = 0
        for requested, returned in probes[0].reads:
            assert requested <= 65 - total
            total += returned
        assert total == len(value["content"].encode()) and total <= 65
        if short:
            assert len(probes[0].reads) > 2
    with metadata_override(root / "growing.txt", size=0), monitor_reads(root, short=3) as (_, probes):
        state, _, _ = await run_case(
            "growth_size_metadata_injection", root,
            [response(text_call("text_growth", "growing.txt")), response(message())], max_bytes=64,
        )
    counts(state, 2, 1, 1)
    assert state["tool_outcomes"]["rejected"] == 1
    assert result_envelopes(state)[0]["error"]["reason_code"] == "file_too_large"
    total = 0
    for requested, returned in probes[0].reads:
        assert requested <= 65 - total
        total += returned
    assert total == 65
    assert "result" not in result_envelopes(state)[0]
    print(json.dumps({"scenario": "growth_size_metadata_injection", "fault_source": "lstat/fstat size injection; real FileIO reads", "bytes_read": total, "run_id": state["run_id"]}))
    with metadata_override(root / "short.txt", size=65537, before_open=False), monitor_reads(root) as (opened, probes):
        state, _, _ = await run_case(
            "opened_descriptor_size_injection", root,
            [response(text_call("text_fstat_size", "short.txt"), text_call("text_fstat_ok")), response(message())],
        )
    counts(state, 2, 2, 2)
    assert state["tool_outcomes"] == {"succeeded": 1, "rejected": 1, "failed": 0, "skipped": 0}
    assert result_envelopes(state)[0]["error"]["reason_code"] == "file_too_large"
    assert opened == [root / "short.txt", root / "plain private.txt"] and probes[0].reads == []
    print(json.dumps({"scenario": "opened_descriptor_size_injection", "fault_source": "fstat size injection after real open", "run_id": state["run_id"]}))
    with TemporaryDirectory(prefix="lineage-default-text-") as temporary:
        expanded = Path(temporary)
        (expanded / "exact.txt").write_bytes(b"x" * 131072)
        (expanded / "over.txt").write_bytes(b"y" * 131073)
        original = snapshot(expanded)
        for factory in (create_text_executor, create_filesystem_executor):
            with monitor_reads(expanded) as (opened, probes):
                state, client, _ = await run_case(
                    f"production_text_budget_{factory.__name__}", expanded,
                    [response(text_call("default_exact", "exact.txt")),
                     response(text_call("default_over", "over.txt")), response(message())],
                    executor=factory(expanded), tools=[TEXT_TOOL],
                )
            exact, rejected = result_envelopes(state)
            value = json.loads(exact["result"])
            assert value["content"] == "x" * 131072 and value["bytes_read"] == 131072
            assert value["complete"] is True and rejected["error"]["reason_code"] == "file_too_large"
            assert opened == [expanded / "exact.txt"] and len(probes) == 1
            assert sum(count for _, count in probes[0].reads) == 131072
            counts(state, 3, 2, 2)
            assert client.requests[1]["input"][-1] == outputs(state)[0]
            assert snapshot(expanded) == original
    for value in (0, -1, True, False, 1.5, "64", None):
        for factory in (create_text_executor, create_filesystem_executor):
            try:
                factory(root, max_bytes=value)
            except ConfigurationError:
                pass
            else:
                raise AssertionError("invalid trusted byte limit accepted")

async def check_arguments(root):
    for scenario, arguments, attempts in (
        ("text_missing_field", "{}", 2),
        ("text_wrong_type", '{"path":4}', 2),
        ("text_null", '{"path":null}', 2),
        ("text_boolean", '{"path":false}', 2),
        ("text_extra_field", '{"path":"plain private.txt","extra":"SECRET"}', 2),
        ("text_duplicate_key", '{"path":"plain private.txt","path":"empty.txt"}', 1),
        ("text_escaped_duplicate_key", r'{"path":"empty.txt","p\u0061th":"plain private.txt"}', 1),
        ("text_malformed_json", '{"path":', 1),
        ("text_non_object", '["plain private.txt"]', 1),
    ):
        await rejected_then_success(
            scenario, root, arguments=arguments, attempts=attempts,
            reason="invalid_text_argument_shape" if attempts == 2 else None,
        )

async def check_paths_and_formats(root, sibling):
    for scenario, path, reason in (
        ("text_empty_path", "", "empty_path"),
        ("text_parent", "../" + sibling.name + "/outside.txt", "disallowed_path_component"),
        ("text_parent_inside", "docs private/../plain private.txt", "disallowed_path_component"),
        ("text_neighbor_absolute", str(sibling / "outside.txt"), "non_relative_path"),
        ("text_drive_relative", "C:outside.txt", "non_relative_path"),
        ("text_unc", "\\\\server\\share\\outside.txt", "non_relative_path"),
        ("text_device_path", "\\\\?\\C:\\outside.txt", "non_relative_path"),
        ("text_ads", "plain private.txt:stream", "disallowed_path_component"),
        ("text_device_name", "NUL.txt", "disallowed_path_component"),
        ("text_directory_target", "docs private", "target_not_regular_file"),
    ):
        await rejected_then_success(scenario, root, path=path, reason=reason)
    await rejected_then_success("text_missing", root, path="missing.txt", outcome="failed", reason="file_not_found")
    await rejected_then_success("text_invalid_utf8", root, path="invalid_utf8.txt", outcome="failed", reason="decode_error")
    for number in (*range(0, 9), 11, 12, *range(14, 32), 127):
        await rejected_then_success(f"text_control_{number}", root, path=f"control_{number}.txt", reason="non_text_content")
    with metadata_override(root / "special.txt", mode=stat.S_IFCHR | 0o600):
        state, _, _ = await rejected_then_success("special_type_metadata_injection", root, path="special.txt", reason="target_not_regular_file")
    print(json.dumps({"scenario": "special_type_metadata_injection", "fault_source": "pre-open lstat file-type injection", "run_id": state["run_id"]}))
    with metadata_override(root / "special.txt", mode=stat.S_IFCHR | 0o600, before_open=False), monitor_reads(root) as (opened, probes):
        state, _, _ = await run_case(
            "opened_descriptor_type_injection", root,
            [response(text_call("text_fstat_type", "special.txt"), text_call("text_fstat_ok")), response(message())],
        )
    counts(state, 2, 2, 2)
    assert state["tool_outcomes"] == {"succeeded": 1, "rejected": 1, "failed": 0, "skipped": 0}
    assert result_envelopes(state)[0]["error"]["reason_code"] == "target_not_regular_file"
    assert opened == [root / "special.txt", root / "plain private.txt"] and probes[0].reads == []
    print(json.dumps({"scenario": "opened_descriptor_type_injection", "fault_source": "fstat type injection after real open", "run_id": state["run_id"]}))

async def check_io_and_logging_failures(root):
    for scenario, fail_open, fail_after, reason in (
        ("permission_injection", True, None, "permission_denied"),
        ("mid_read_io_injection", False, 1, "io_error"),
    ):
        with monitor_reads(root, short=3, fail_path=root / "short.txt", fail_open=fail_open, fail_after=fail_after) as (opened, probes):
            state, client, _ = await run_case(
                scenario, root,
                [response(text_call("text_io_bad", "short.txt"), text_call("text_io_good")), response(message())],
            )
        counts(state, 2, 2, 2)
        assert state["tool_outcomes"] == {"succeeded": 1, "rejected": 0, "failed": 1, "skipped": 0}
        first, second = result_envelopes(state)
        assert first["error"]["reason_code"] == reason and first["error"]["code"] == "execution_error"
        assert "result" not in first and second["ok"] is True
        assert opened == [root / "short.txt", root / "plain private.txt"]
        print(json.dumps({"scenario": scenario, "fault_source": "file open/read injection", "run_id": state["run_id"]}))
    for event, first_path, outcome, reason in (
        ("tool.failed", "invalid_utf8.txt", "failed", "decode_error"),
        ("tool.rejected", "over.txt", "rejected", "file_too_large"),
    ):
        with recording_failure(event) as failed_writes, monitor_reads(root) as (opened, _):
            state, client, observed = await run_case(
                f"text_{event}_write_failure", root,
                [response(text_call("text_record_bad", first_path), text_call("text_record_skipped")), response(message())],
                failed_writes=failed_writes,
            )
        counts(state, 1, 2, 1, "error")
        assert state["log_complete"] is False and state["tool_outcomes"][outcome] == 1
        assert state["tool_outcomes"]["skipped"] == 1 and state["tool_outcomes"]["succeeded"] == 0
        assert len(observed) == 1 and len(client.requests) == 1
        assert len(outputs(state)) == 1 and outputs(state)[0]["call_id"] == "text_record_bad"
        assert result_envelopes(state)[0]["error"]["reason_code"] == reason
        assert not any(item.get("type") == "function_call_output" for item in client.requests[0]["input"])
        assert state["usage_totals"] == {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12, "missing_usage_calls": 0}
        assert failed_writes[0]["call_id"] == "text_record_bad" and failed_writes[0]["persistence"] == "not_written"
        assert root / "plain private.txt" not in opened

async def check_links(root, sibling):
    created = []
    for name, target, is_dir, kind in (
        ("outside link.txt", sibling / "outside.txt", False, "symlink"),
        ("inside link.txt", root / "plain private.txt", False, "symlink"),
        ("outside directory link", sibling, True, "symlink"),
        ("inside directory link", root / "docs private", True, "symlink"),
    ):
        link = root / name
        try:
            os.symlink(target, link, target_is_directory=is_dir)
        except (OSError, NotImplementedError) as exc:
            skipped = {"scenario": name, "status": "SKIP", "reason": type(exc).__name__, "actual_link_created": False}
            SKIPPED.append(skipped)
            coverage_gap(name, skipped["reason"])
            print(json.dumps(skipped))
        else:
            created.append((link, is_dir, kind))
    if os.name == "nt":
        link = root / "outside junction"
        environment = os.environ.copy()
        environment.update(LINEAGE_TEXT_LINK=str(link), LINEAGE_TEXT_TARGET=str(sibling))
        completed = subprocess.run(
            ["C:\\Program Files\\PowerShell\\7\\pwsh.exe", "-NoProfile", "-Command",
             "$ErrorActionPreference = 'Stop'; New-Item -ItemType Junction -Path $env:LINEAGE_TEXT_LINK -Value $env:LINEAGE_TEXT_TARGET | Out-Null"],
            env=environment, capture_output=True, text=True, check=False,
        )
        if completed.returncode == 0:
            created.append((link, True, "junction"))
        else:
            skipped = {"scenario": "junction", "status": "SKIP", "reason": "creation failed", "actual_link_created": False}
            SKIPPED.append(skipped)
            coverage_gap("junction", skipped["reason"])
            print(json.dumps(skipped))
    else:
        coverage_gap("junction", "Windows-only fixture", applicable=False)
    try:
        for index, (link, is_dir, kind) in enumerate(created):
            await rejected_then_success(f"text_{kind}_{index}_direct", root, path=link.name, reason="link_or_reparse_forbidden")
            if is_dir:
                name = "mixed space.TXT" if "inside" in link.name else "outside.txt"
                await rejected_then_success(f"text_{kind}_{index}_intermediate", root, path=link.name + "/" + name, reason="link_or_reparse_forbidden")
    finally:
        for link, is_dir, _ in reversed(created):
            # Remove only the created link itself, never recursively its target.
            if is_dir and os.name == "nt":
                os.rmdir(link)
            else:
                link.unlink()

async def check_untrusted_content(root, sibling):
    with monitor_reads(root):
        state, client, _ = await run_case(
            "text_pseudo_instructions_keep_tool_identity", root,
            [response(text_call("text_pseudo", "pseudo.md")),
             response(text_call("text_forbidden", "../" + sibling.name + "/outside.txt"), call("text_unknown", name="run_shell", arguments='{"command":"private-command"}')),
             response(message("source text cannot change tool permissions"))],
        )
    counts(state, 3, 3, 2)
    assert state["tool_outcomes"] == {"succeeded": 1, "rejected": 2, "failed": 0, "skipped": 0}
    first = result_envelopes(state)[0]
    assert json.loads(first["result"])["content"] == (root / "pseudo.md").read_text(encoding="utf-8")
    assert BODY_MARKER in client.requests[1]["input"][-1]["output"]
    for request in client.requests:
        assert not any(item.get("role") in {"system", "developer"} for item in request["input"])
        assert request["tools"] == TOOLS
    assert result_envelopes(state)[2]["error"]["code"] == "unknown_tool"

async def check_isolation(root, sibling):
    gate = asyncio.Event()
    started = 0
    shared = [{"role": "user", "content": "read the project file", "metadata": {"nested": ["original"]}}]
    original = deepcopy(shared)
    cwd = Path.cwd()
    roots = [root, sibling]
    paths = ["plain private.txt", "outside.txt"]
    clients = [LocalClient([response(text_call(f"isolation_{index}", paths[index])), response(message())]) for index in range(2)]
    executors = [create_filesystem_executor(bound) for bound in roots]

    def client_factory(settings):
        return clients[int(settings.model_name)]

    async def one(index):
        async def execute(name, arguments):
            nonlocal started
            started += 1
            if started == 2:
                gate.set()
            await gate.wait()
            return await executors[index](name, arguments)

        state = await runtime.run_agent(shared, replace(SETTINGS, model_name=str(index)), tools=TOOLS, execute_tool=execute)
        observe_state(state)
        return state

    with patch.object(runtime, "create_client", side_effect=client_factory):
        states = await asyncio.gather(one(0), one(1))
    assert Path.cwd() == cwd and shared == original and len({state["run_id"] for state in states}) == 2
    for index, state in enumerate(states):
        counts(state, 2, 1, 1)
        assert clients[index].closed and state["tool_outcomes"]["succeeded"] == 1
        assert state["usage_totals"]["total_tokens"] == 24
        item = outputs(state)[0]
        assert clients[index].requests[-1]["input"][-1] == item
        assert json.loads(json.loads(item["output"])["result"])["content"] == (roots[index] / paths[index]).read_text(encoding="utf-8")
        audit_recording(state, clients[index], scenario=f"text_interleaved_root_{index}")
        RUNS.append(state["run_id"])
    states[0]["input"][0]["metadata"]["nested"].append("mutated")
    assert states[1]["input"][0] == original[0] and shared == original

async def check_configuration_and_cli(root, base):
    env = base / "text.env"
    env.write_text("BASE_URL=https://unused.invalid/v1\nMODEL_NAME=local\nTEXT_MAX_BYTES=1234\n", encoding="utf-8")
    with patch.dict(os.environ, {}, clear=True):
        assert load_settings(env).text_max_bytes == 1234
    with patch.dict(os.environ, {"TEXT_MAX_BYTES": "4321"}, clear=True):
        assert load_settings(env).text_max_bytes == 4321
    assert SETTINGS.text_max_bytes == 131072
    for value in ("0", "-1", "false", "1.5", ""):
        with patch.dict(os.environ, {"TEXT_MAX_BYTES": value}, clear=True):
            try:
                load_settings(env)
            except ConfigurationError:
                pass
            else:
                raise AssertionError("invalid configured text limit accepted")
    for value in (0, -1, True, False, 1.5, "64", None):
        try:
            replace(SETTINGS, text_max_bytes=value)
        except ConfigurationError:
            pass
        else:
            raise AssertionError("invalid direct text setting accepted")
    for enabled in (False, True):
        scenario = "cli_directory_and_text" if enabled else "cli_no_project_no_tools"
        client = LocalClient(
            [response(directory_call("cli_discover")), response(text_call("cli_read")), response(message())]
            if enabled else [response(message())]
        )
        captured = []

        async def capture(*args, **kwargs):
            state = await runtime.run_agent(*args, **kwargs)
            observe_state(state)
            captured.append(state)
            return state

        def invoke():
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(cli, "load_settings", return_value=replace(SETTINGS, project_root=root if enabled else None)), \
                    patch.object(cli, "run_agent", capture), patch.object(runtime, "create_client", return_value=client), \
                    patch.object(sys, "argv", ["lineage-agent", "Use source material"]), redirect_stdout(stdout), redirect_stderr(stderr):
                code = cli.main()
            return code, stdout.getvalue(), stderr.getvalue()

        code, stdout, stderr = await asyncio.to_thread(invoke)
        assert code == 0 and client.closed and stdout.strip() == "done"
        assert captured[0]["run_id"] in stderr and captured[0]["log_path"] in stderr
        if enabled:
            assert client.requests[0]["tools"] == TOOLS
            assert json.loads(json.loads(outputs(captured[0])[-1]["output"])["result"])["content"] == (root / "plain private.txt").read_text(encoding="utf-8")
        else:
            assert not client.requests[0].get("tools")
        audit_recording(captured[0], client, scenario=scenario)
        RUNS.append(captured[0]["run_id"])

async def main():
    with TemporaryDirectory(prefix="lineage-text-check-") as temporary:
        base = Path(temporary)
        root, sibling = fixture(base)
        original_root, original_sibling = snapshot(root), snapshot(sibling)
        checks = [
            ("reading_content_encoding_and_empty", "check_reading", lambda: check_reading(root)),
            ("bounded_actual_reads_and_growth", "check_limits", lambda: check_limits(root)),
            ("text_argument_validation", "check_arguments", lambda: check_arguments(root)),
            ("text_paths_formats_and_controls", "check_paths_and_formats", lambda: check_paths_and_formats(root, sibling)),
            ("text_io_and_recording_failures", "check_io_and_logging_failures", lambda: check_io_and_logging_failures(root)),
            ("text_real_links_and_junctions", "check_links", lambda: check_links(root, sibling)),
            ("untrusted_content_cannot_expand_access", "check_untrusted_content", lambda: check_untrusted_content(root, sibling)),
            ("text_concurrent_run_isolation", "check_isolation", lambda: check_isolation(root, sibling)),
            ("text_configuration_and_cli", "check_configuration_and_cli", lambda: check_configuration_and_cli(root, base)),
        ]
        for name, identifier, check in checks:
            with check_group(group_id(check, identifier)):
                await check()
                assert snapshot(root) == original_root and snapshot(sibling) == original_sibling, "tools must not modify the input fixtures"
            print(f"PASS {name}")
    print(json.dumps({"groups_passed": len(checks), "graph_runs": len(RUNS), "skipped": SKIPPED, "logs": str(CHECK_DIR)}, ensure_ascii=False))

def execute():
    asyncio.run(main())
