"""Durable attempts retain byte reservations and explicit tool authority."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lineage_agent.config.recovery import describe_tools, restore_tools
from lineage_agent.config.settings import ConfigurationError
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL, create_directory_executor
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL, create_text_executor
from lineage_agent.tools.types import ToolExecutionError, ToolRecoveryBinding, read_budget_snapshot


class ExecutionToolSupportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="execution-tool-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "small.txt").write_bytes(b"abcd")
        (self.root / "invalid.txt").write_bytes(b"\xff")

    async def test_unknown_reservation_survives_reconstruction_and_settles_once(self):
        executor = create_text_executor(self.root, max_bytes=8, max_total_bytes=20)
        self.assertEqual(executor.reserve_read("unknown"), 8)
        self.assertEqual(executor.reserve_read("unknown"), 8)
        self.assertEqual(executor.reserve_read("complete"), 8)
        raw = await executor.execute_reserved("complete", "read_text_file", {"path": "small.txt"})
        self.assertEqual(json.loads(raw)["bytes_read"], 4)
        self.assertEqual(executor.read_budget["used_bytes"], 0)
        executor.settle_read("complete", 4)
        executor.settle_read("complete", 4)
        saved = executor.export_read_budget()
        rebuilt = create_text_executor(self.root, max_bytes=8, max_total_bytes=20)
        rebuilt.restore_read_budget(saved)
        self.assertEqual(read_budget_snapshot(rebuilt), {
            "limit_bytes": 20, "used_bytes": 4, "reserved_bytes": 8, "remaining_bytes": 8})
        self.assertEqual(rebuilt.reserve_read("unknown"), 8)
        self.assertEqual(rebuilt.reserve_read("retry"), 8)
        (self.root / "small.txt").write_bytes(b"new")
        retry = json.loads(await rebuilt.execute_reserved("retry", "read_text_file", {"path": "small.txt"}))
        rebuilt.settle_read("retry", retry["bytes_read"])
        self.assertEqual(retry["content"], "new")
        self.assertEqual(rebuilt.read_budget, {
            "limit_bytes": 20, "used_bytes": 7, "reserved_bytes": 8, "remaining_bytes": 5})
        with self.assertRaises(ConfigurationError):
            rebuilt.settle_read("complete", 3)
        with self.assertRaises(ConfigurationError):
            rebuilt.reserve_read("complete")
        with self.assertRaises(ConfigurationError):
            await rebuilt.execute_reserved("complete", "read_text_file", {"path": "small.txt"})

    async def test_known_failure_releases_reservation_and_normal_calls_keep_old_semantics(self):
        executor = create_filesystem_executor(self.root, max_bytes=8, max_total_bytes=8)
        executor.reserve_read("bad")
        with self.assertRaises(ToolExecutionError):
            await executor.execute_reserved("bad", "read_text_file", {"path": "invalid.txt"})
        executor.settle_read("bad", 0)
        self.assertEqual(executor.read_budget["remaining_bytes"], 8)
        await executor("read_text_file", {"path": "small.txt"})
        self.assertEqual(executor.read_budget["used_bytes"], 4)
        rebuilt = create_filesystem_executor(self.root, max_bytes=8, max_total_bytes=8)
        rebuilt.restore_read_budget(executor.export_read_budget())
        self.assertEqual(rebuilt.read_budget, executor.read_budget)
        self.assertEqual(executor.bind_run().read_budget, {
            "limit_bytes": 8, "used_bytes": 0, "remaining_bytes": 8})

    def test_invalid_budget_restore_cannot_mutate_live_allowance(self):
        executor = create_text_executor(self.root, max_bytes=8, max_total_bytes=12)
        executor.reserve_read("pending")
        saved = executor.export_read_budget()
        invalid = [
            {**saved, "version": True}, {**saved, "max_bytes": 9},
            {**saved, "used_bytes": 5}, {**saved, "settled": {"pending": 0}},
            {**saved, "reservations": {"pending": True}},
            {**saved, "settled": {"completed": 4}},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                executor.restore_read_budget(value)
            self.assertEqual(executor.export_read_budget(), saved)

    async def test_external_recovery_declaration_is_explicit_exact_and_not_name_inferred(self):
        async def external(name, arguments):
            return "application result"
        tools = [TEXT_TOOL]
        plain = describe_tools(tools, external)
        self.assertIsNone(plain["recovery"])
        binding = ToolRecoveryBinding("application.reader", 3, {"read_text_file": "safe_repeat"})
        saved = describe_tools(tools, external, recovery_binding=binding)
        self.assertEqual(saved["provider"], "external")
        self.assertTrue(saved["recovery"]["stateless"])
        rebuilt = restore_tools(saved, current_project_root=None, tools=tools,
                                execute_tool=external, recovery_binding=binding)
        self.assertIs(rebuilt["execute_tool"], external)
        for different in (None, ToolRecoveryBinding("application.reader", 4, binding.tools),
                          ToolRecoveryBinding("application.reader", 3, {"read_text_file": "pause"})):
            with self.assertRaises(ConfigurationError):
                restore_tools(saved, current_project_root=None, tools=tools,
                              execute_tool=external, recovery_binding=different)
        with self.assertRaises(ConfigurationError):
            describe_tools(tools, external, recovery_binding=ToolRecoveryBinding("x", 1, {}))

    def test_builtin_policies_follow_factories_and_cannot_be_overridden(self):
        cases = [([TEXT_TOOL], create_text_executor(self.root), False),
                 ([DIRECTORY_TOOL], create_directory_executor(self.root), True)]
        for tools, executor, stateless in cases:
            saved = describe_tools(tools, executor)
            self.assertEqual(saved["recovery"]["stateless"], stateless)
            self.assertEqual(set(saved["recovery"]["tools"].values()), {"safe_repeat"})
            restore_tools(saved, current_project_root=self.root)
            changed = deepcopy(saved)
            changed["recovery"]["tools"][tools[0]["name"]] = "pause"
            with self.assertRaises(ConfigurationError):
                restore_tools(changed, current_project_root=self.root)
            with self.assertRaises(ConfigurationError):
                describe_tools(tools, executor,
                    recovery_binding=ToolRecoveryBinding("forged", 1, {tools[0]["name"]: "safe_repeat"}))


if __name__ == "__main__":
    unittest.main()
