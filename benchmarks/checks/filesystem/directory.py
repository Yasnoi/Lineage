"""filesystem / directory behavior contracts."""


import asyncio
import io
import json
from benchmarks.checks.support.observations import inline_raw, assert_tools
import os
import subprocess
import sys
from benchmarks.checks.support.directory import (
    CHECK_DIR,
    SETTINGS,
    SKIPPED,
    ScanProbe,
    decoded_results,
    directory_call,
    fixture,
    rejected_then_success,
    run_case,
    skip_link,
    snapshot,
)
from benchmarks.checks.support.events import group_id, check_group, coverage_gap, observe_state
from benchmarks.checks.support.runtime import LocalClient, audit_recording, counts, message, outputs, response
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from lineage_agent.agent import runtime
from lineage_agent.config.settings import ConfigurationError, ModelSettings, load_settings
from lineage_agent.entrypoints import cli
from lineage_agent.tools.filesystem import directory
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL, create_directory_executor
from lineage_agent.tools.filesystem.paths import bind_project_root, resolve_project_path
from lineage_agent.tools.filesystem.text import TEXT_TOOL
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

async def check_listing(root):
    before = snapshot(root)
    state, client, observed = await run_case(
        "root_then_child", root,
        [response(directory_call("fs_root")), response(directory_call("fs_child", "child private")), response(message())],
    )
    counts(state, 3, 2, 2)
    first, second = [json.loads(inline_raw(item)) for item in decoded_results(state)]
    expected = {
        "文件 private.txt": "file", "child private": "directory", "空目录 private": "directory",
        ".hidden private": "file", ".git": "directory", ".gitignore": "file",
    }
    assert first["path"] == "." and first["limit"] == 100 and first["truncated"] is False
    assert {item["name"]: item["type"] for item in first["entries"]} == expected
    assert [item["name"] for item in first["entries"]] == sorted(expected, key=lambda name: (name.casefold(), name))
    assert all(item["path"] == item["name"] for item in first["entries"])
    assert second == {
        "path": "child private", "entries": [{"name": "nested.txt", "path": "child private/nested.txt", "type": "file"}],
        "limit": 100, "truncated": False, "after": None, "next_after": None,
    }
    assert_tools(client.requests[0]["tools"], [DIRECTORY_TOOL])
    assert len(observed) == 2 and snapshot(root) == before
    for scenario, path in (
        ("empty_directory", "空目录 private"),
        ("forward_slash_separator", "child private/."),
        ("backslash_separator", "child private\\."),
    ):
        state, _, _ = await run_case(scenario, root, [response(directory_call("fs_path", path)), response(message())])
        result = json.loads(inline_raw(decoded_results(state)[0]))
        assert result["truncated"] is False
        assert result["entries"] == ([] if path == "空目录 private" else second["entries"])

async def check_limits(root):
    real_scandir = os.scandir
    for limit, expected_truncation in ((6, False), (2, True)):
        probes = []

        def scanned(path):
            if not isinstance(path, (str, os.PathLike)) or Path(path) != root:
                return real_scandir(path)
            result = ScanProbe(real_scandir(path))
            probes.append(result)
            return result

        with patch.object(directory.os, "scandir", scanned):
            state, _, _ = await run_case(
                f"entry_limit_{limit}", root, [response(directory_call("fs_limit")), response(message())], max_entries=limit,
            )
        result = json.loads(inline_raw(decoded_results(state)[0]))
        assert len(probes) == 1 and probes[0].closed
        assert len(probes[0].yielded) == 6
        assert result["limit"] == limit and result["truncated"] is expected_truncation
        assert [item["name"] for item in result["entries"]] == sorted(probes[0].yielded, key=lambda name: (name.casefold(), name))[:limit]
    with TemporaryDirectory(prefix="lineage-default-directory-") as temporary:
        expanded = Path(temporary)
        names = [f"entry_{index:03}.txt" for index in range(137)]
        for name in reversed(names):
            (expanded / name).write_bytes(b"")
        original = snapshot(expanded)
        state, client, observed = await run_case(
            "production_directory_budget_beyond_old_limit", expanded,
            [response(directory_call("default_directory")), response(message())],
            executor=create_directory_executor(expanded),
        )
        assert decoded_results(state)[0]["observation"]["mode"] == "artifact"
        result = json.loads(observed[0]["result"])
        assert result["limit"] == SETTINGS.directory_max_entries == 1000
        assert [entry["name"] for entry in result["entries"]] == names
        assert result["truncated"] is False and result["next_after"] is None
        counts(state, 2, 1, 1)
        assert client.requests[1]["input"][-1] == outputs(state)[0]
        assert snapshot(expanded) == original
    for invalid in (True, False, 0, -1, 1.5, "2", None):
        try:
            create_directory_executor(root, max_entries=invalid)
        except ConfigurationError:
            pass
        else:
            raise AssertionError("invalid directory limit accepted")

async def check_arguments(root):
    for scenario, arguments, attempts in (
        ("missing_field", "{}", 2),
        ("wrong_field_type", '{"path":4}', 2),
        ("null_field", '{"path":null}', 2),
        ("boolean_field", '{"path":false}', 2),
        ("extra_field", '{"path":".","extra":"SECRET"}', 2),
        ("duplicate_field", '{"path":".","path":"child private"}', 1),
        ("malformed_json", '{"path":', 1),
        ("non_object", '["."]', 1),
    ):
        await rejected_then_success(
            root, scenario, arguments=arguments, attempts=attempts,
            expected_reason="invalid_argument_shape" if attempts == 2 else None,
        )

async def check_paths(root, sibling):
    rejected = [
        ("empty_path", ""), ("parent", ".."), ("parent_internal", "child private/.."),
        ("prefix_neighbor", "../" + sibling.name), ("absolute_neighbor", str(sibling)),
        ("posix_absolute", "/outside"), ("drive_absolute", "C:\\outside"),
        ("drive_relative", "C:outside"), ("unc", "\\\\server\\share"),
        ("device", "\\\\?\\C:\\outside"), ("device_namespace", "\\\\.\\C:\\outside"),
        ("rooted_windows", "\\outside"), ("alternate_stream", "file:stream"),
        ("control_nul", "file\x00"), ("control_newline", "file\n"),
        ("reserved_star", "*"), ("reserved_question", "?"), ("reserved_quote", '"'),
        ("reserved_less", "<"), ("reserved_greater", ">"), ("reserved_pipe", "|"),
        ("reserved_device", "CON"), ("reserved_device_suffix", "nul.txt"),
        ("reserved_device_nested", "child private/COM1.txt"),
        ("trailing_dot", "file."), ("trailing_space", "file "),
    ]
    non_relative = {
        "absolute_neighbor", "posix_absolute", "drive_absolute", "drive_relative",
        "unc", "device", "device_namespace", "rooted_windows",
    }
    for scenario, value in rejected:
        reason = (
            "empty_path" if scenario == "empty_path"
            else "non_relative_path" if scenario in non_relative
            else "disallowed_path_component"
        )
        await rejected_then_success(root, scenario, path=value, expected_reason=reason)
    for scenario, path in (
        ("missing_target", "missing_private"), ("literal_tilde", "~"),
        ("literal_environment_percent", "%USERPROFILE%"), ("literal_environment_dollar", "$HOME"),
    ):
        # Literals are not expanded. These names do not exist inside the fixture.
        await rejected_then_success(root, scenario, path=path, outcome="failed", expected_code="execution_error")
    await rejected_then_success(
        root, "file_target", path="文件 private.txt", expected_reason="target_not_directory"
    )
    assert bind_project_root(root) == root
    assert resolve_project_path(root, "文件 private.txt") == root / "文件 private.txt", "shared path rules must also accept files"

async def check_io_failures(root):
    real_scandir = os.scandir
    for scenario in ("permission_injection", "enumeration_injection", "classification_injection"):
        probes = []
        calls = []

        def faulting_scan(path):
            if not isinstance(path, (str, os.PathLike)) or Path(path) != root:
                return real_scandir(path)
            calls.append(Path(path))
            if len(calls) == 1:
                if scenario == "permission_injection":
                    raise PermissionError("INJECTED_PRIVATE_PERMISSION")
                probe = ScanProbe(
                    real_scandir(path),
                    fail_after=1 if scenario == "enumeration_injection" else None,
                    fail_stat_at=2 if scenario == "classification_injection" else None,
                )
                probes.append(probe)
                return probe
            return real_scandir(path)

        with patch.object(directory.os, "scandir", faulting_scan):
            state, _, observed = await run_case(
                scenario, root,
                [response(directory_call("fs_io_fail"), directory_call("fs_io_ok")), response(message())],
            )
        counts(state, 2, 2, 2)
        assert state["tool_outcomes"] == {"succeeded": 1, "rejected": 0, "failed": 1, "skipped": 0}
        first, second = decoded_results(state)
        assert first["ok"] is False and first["error"]["code"] == "execution_error"
        assert first["error"]["reason_code"] == ("permission_denied" if scenario == "permission_injection" else "io_error")
        assert second["ok"] is True and "result" not in observed[0]
        assert calls == [root, root]
        assert all(probe.closed for probe in probes)
        assert "INJECTED_PRIVATE" not in Path(state["log_path"]).read_text(encoding="utf-8")
        print(json.dumps({"scenario": scenario, "fault_source": "scandir fault injection", "run_id": state["run_id"]}))

async def check_links(root, sibling):
    created = []
    for name, target, is_dir, kind in (
        ("link outside private", sibling, True, "symlink"),
        ("link inside private", root / "child private", True, "symlink"),
        ("link file private", sibling / "OUTSIDE_PRIVATE.txt", False, "symlink"),
    ):
        link = root / name
        try:
            os.symlink(target, link, target_is_directory=is_dir)
        except (OSError, NotImplementedError) as exc:
            skip_link(name, exc)
        else:
            created.append((link, kind, is_dir))
    if os.name == "nt":
        link = root / "junction outside private"
        environment = os.environ.copy()
        environment.update(LINEAGE_CHECK_LINK=str(link), LINEAGE_CHECK_TARGET=str(sibling))
        result = subprocess.run(
            ["C:\\Program Files\\PowerShell\\7\\pwsh.exe", "-NoProfile", "-Command",
             "$ErrorActionPreference = 'Stop'; New-Item -ItemType Junction -Path $env:LINEAGE_CHECK_LINK -Value $env:LINEAGE_CHECK_TARGET | Out-Null"],
            env=environment, capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            created.append((link, "junction", True))
        else:
            skip_link("junction", OSError("junction fixture creation failed"))
    else:
        coverage_gap("junction", "Windows-only fixture", applicable=False)
    try:
        if created:
            state, _, _ = await run_case("link_entries_without_targets", root, [response(directory_call("fs_links")), response(message())])
            result = json.loads(inline_raw(decoded_results(state)[0]))
            entries = {entry["name"]: entry for entry in result["entries"]}
            for index, (link, kind, is_dir) in enumerate(created):
                assert entries[link.name] == {"name": link.name, "path": link.name, "type": kind}
                assert str(sibling) not in json.dumps(result)
                await rejected_then_success(
                    root, f"{kind}_{index}_direct", path=link.name,
                    expected_reason="link_or_reparse_forbidden",
                )
                if is_dir:
                    await rejected_then_success(
                        root, f"{kind}_{index}_intermediate", path=link.name + "/OUTSIDE_PRIVATE.txt",
                        expected_reason="link_or_reparse_forbidden",
                    )
                candidates = [link]
                if is_dir and "outside" in link.name:
                    # This target exists and is a directory: missing/file rejection
                    # cannot accidentally satisfy the linked-ancestor assertion.
                    assert (sibling / "external child").is_dir()
                    candidates.append(link / "external child")
                for candidate in candidates:
                    try:
                        bind_project_root(candidate)
                    except ConfigurationError:
                        pass
                    else:
                        raise AssertionError("linked project root accepted")
                    configured = replace(SETTINGS, project_root=candidate)
                    assert configured.project_root == candidate, "configuration must not resolve away links"
                if len(candidates) == 2:
                    env_file = root.parent / "linked-root.env"
                    env_file.write_text(
                        "BASE_URL=https://unused.invalid/v1\nMODEL_NAME=local\n"
                        f"PROJECT_ROOT={candidates[1].relative_to(root.parent).as_posix()}\n",
                        encoding="utf-8",
                    )
                    with patch.dict(os.environ, {}, clear=True):
                        assert load_settings(env_file).project_root == candidates[1]
    finally:
        for link, kind, _ in reversed(created):
            # Remove only the link itself; never recursively delete its resolved target.
            assert link.parent == root and root.parent == sibling.parent
            if kind == "junction":
                os.rmdir(link)
            else:
                link.unlink()
    assert (sibling / "OUTSIDE_PRIVATE.txt").read_text(encoding="utf-8") == "OUTSIDE_BODY_SECRET"

async def check_isolation(root, sibling):
    original_cwd = Path.cwd()
    entered = 0
    barrier = asyncio.Event()
    clients = [LocalClient([response(directory_call("fs_isolated")), response(message())]) for _ in range(2)]
    roots = (root, sibling)
    executors = [create_directory_executor(value) for value in roots]
    settings = [replace(SETTINGS, model_name=f"isolated-{index}") for index in range(2)]
    seen = [[], []]

    def executor(index):
        async def execute(name, arguments):
            nonlocal entered
            entered += 1
            if entered == 2:
                barrier.set()
            await asyncio.wait_for(barrier.wait(), timeout=5)
            value = await executors[index](name, arguments)
            seen[index].append(value)
            return value
        return execute

    shared = [{"role": "user", "content": "list directories"}]
    untouched = deepcopy(shared)
    with patch.object(runtime, "create_client", side_effect=lambda config: clients[settings.index(config)]):
        states = await asyncio.gather(*(
            runtime.run_agent(shared, settings[index], tools=[DIRECTORY_TOOL], execute_tool=executor(index))
            for index in range(2)
        ))
        for state in states:
            observe_state(state)
    assert shared == untouched and Path.cwd() == original_cwd
    assert states[0]["run_id"] != states[1]["run_id"]
    for index, state in enumerate(states):
        counts(state, 2, 1, 1)
        assert clients[index].closed
        audit_recording(state, clients[index], scenario=f"interleaved_root_{index}")
        actual = [inline_raw(json.loads(item["output"])) for item in outputs(state)]
        assert actual == seen[index]
        assert outputs(state) == [item for item in clients[index].requests[1]["input"] if item.get("type") == "function_call_output"]
        expected = {path.name for path in roots[index].iterdir()}
        assert {entry["name"] for entry in json.loads(actual[0])["entries"]} == expected
        assert state["usage_totals"] == {"input_tokens": 20, "output_tokens": 4, "total_tokens": 24, "missing_usage_calls": 0}
    states[0]["input"][0]["content"] = "changed"
    assert shared == untouched and states[1]["input"][0] == untouched[0]

async def check_configuration_and_cli(root, base):
    env_file = base / "fixture.env"
    env_file.write_text(
        "BASE_URL=https://unused.invalid/v1\nMODEL_NAME=local\nPROJECT_ROOT=BOUND_ROOT_PRIVATE\nDIRECTORY_MAX_ENTRIES=2\n",
        encoding="utf-8",
    )
    with patch.dict(os.environ, {}, clear=True):
        settings = load_settings(env_file)
        assert settings.project_root == root and settings.directory_max_entries == 2
    with patch.dict(os.environ, {"PROJECT_ROOT": "", "DIRECTORY_MAX_ENTRIES": "3"}, clear=True):
        settings = load_settings(env_file)
        assert settings.project_root is None and settings.directory_max_entries == 3
    for value in ("0", "-1", "false", "1.5", ""):
        with patch.dict(os.environ, {"DIRECTORY_MAX_ENTRIES": value}, clear=True):
            try:
                load_settings(env_file)
            except ConfigurationError:
                pass
            else:
                raise AssertionError("invalid configured directory limit accepted")
    relative = ModelSettings(base_url="https://unused.invalid", model_name="local", project_root=Path("relative-project"))
    assert relative.project_root == Path.cwd() / "relative-project"
    assert ModelSettings(base_url="https://unused.invalid", model_name="local").project_root is None
    for target in (base / "not-present", root / "文件 private.txt"):
        try:
            bind_project_root(target)
        except ConfigurationError:
            pass
        else:
            raise AssertionError("invalid project root accepted")
    for scenario, project_root, replies in (
        ("cli_without_tools", None, [response(message())]),
        ("cli_directory_tool", root, [response(directory_call("fs_cli")), response(message())]),
        ("cli_invalid_root", base / "missing", []),
    ):
        client = LocalClient(replies)
        captured = []

        async def captured_run(*args, **kwargs):
            state = await runtime.run_agent(*args, **kwargs)
            observe_state(state)
            captured.append(state)
            return state

        def invoke():
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(cli, "load_settings", return_value=replace(SETTINGS, project_root=project_root)), \
                    patch.object(cli, "run_agent", captured_run), \
                    patch.object(runtime, "create_client", return_value=client) as factory, \
                    patch.object(sys, "argv", ["lineage-agent", "list directory"]), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                code = cli.main()
            return code, stdout.getvalue(), stderr.getvalue(), factory.called

        code, stdout, stderr, created = await asyncio.to_thread(invoke)
        if scenario == "cli_invalid_root":
            assert code == 2 and not created and not client.requests and not captured
            print(json.dumps({"scenario": scenario, "status": "PASS", "model_calls": 0, "run_id": None, "log_path": None}))
        else:
            assert code == 0 and client.closed and stdout.strip() == "done"
            assert captured[0]["run_id"] in stderr and captured[0]["log_path"] in stderr
            audit_recording(captured[0], client, scenario=scenario)
            if project_root is None:
                assert assert_tools(client.requests[0]["tools"]) and len(client.requests) == 1
            else:
                assert assert_tools(client.requests[0]["tools"], [DIRECTORY_TOOL, TEXT_TOOL]) and len(client.requests) == 2

async def main():
    with TemporaryDirectory(prefix="lineage-filesystem-check-") as temporary:
        base = Path(temporary)
        root, sibling = fixture(base)
        original_root, original_sibling = snapshot(root), snapshot(sibling)
        checks = [
            ("listing_and_content_isolation", "check_listing", lambda: check_listing(root)),
            ("bounded_enumeration", "check_limits", lambda: check_limits(root)),
            ("parameter_validation", "check_arguments", lambda: check_arguments(root)),
            ("path_access_boundaries", "check_paths", lambda: check_paths(root, sibling)),
            ("io_failure_boundaries", "check_io_failures", lambda: check_io_failures(root)),
            ("real_links_and_junctions", "check_links", lambda: check_links(root, sibling)),
            ("concurrent_run_isolation", "check_isolation", lambda: check_isolation(root, sibling)),
            ("configuration_and_cli", "check_configuration_and_cli", lambda: check_configuration_and_cli(root, base)),
        ]
        for name, identifier, check in checks:
            with check_group(group_id(check, identifier)):
                await check()
            print(f"PASS {name}")
        with check_group("filesystem.directory.fixture_integrity"):
            assert snapshot(root) == original_root and snapshot(sibling) == original_sibling
    print(json.dumps({"groups_passed": len(checks), "skipped": SKIPPED, "logs": str(CHECK_DIR)}, ensure_ascii=False))

def execute():
    asyncio.run(main())
