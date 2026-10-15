"""Saved settings and explicit tool rebinding never restore directory authority."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from lineage_agent.config.recovery import (
    configuration_warnings, describe_tools, restore_settings, restore_tools, serialize_settings,
)
from lineage_agent.config.settings import ConfigurationError, ModelSettings, load_current_limits, load_local_paths
from lineage_agent.recording.trace_data import TraceSanitizer
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL, create_directory_executor
from lineage_agent.tools.filesystem.executor import create_filesystem_executor
from lineage_agent.tools.filesystem.text import TEXT_TOOL, create_text_executor


class RecoveryConfigurationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="recovery-config-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "fact.txt").write_text("saved source", encoding="utf-8")
        self.settings = ModelSettings(base_url="https://saved.invalid/v1", model_name="saved-model",
                                      application_root=self.root, log_dir=self.root / "old-logs",
                                      project_root=self.project, max_model_calls=41)

    def test_settings_are_whitelisted_and_do_not_read_current_model_or_credential(self):
        with patch.dict(os.environ, {"API_KEY": "secret-marker", "MODEL_NAME": "other-model",
                                    "BASE_URL": "https://different.invalid"}, clear=True):
            saved = serialize_settings(self.settings)
            restored = restore_settings(saved, application_root=self.root / "new-storage",
                                        log_dir=self.root / "new-logs")
        self.assertNotIn("secret-marker", json.dumps(saved))
        self.assertNotIn("application_root", saved)
        self.assertNotIn("log_dir", saved)
        self.assertEqual(saved["project_root"], str(self.project))
        self.assertEqual(restored.model_name, "saved-model")
        self.assertEqual(restored.base_url, "https://saved.invalid/v1")
        self.assertEqual(restored.max_model_calls, 41)
        self.assertEqual(restored.project_root, self.project)
        self.assertNotIn(str(self.project), TraceSanitizer([], restored.project_root).text(str(self.project / "fact.txt")))
        self.assertEqual(restored.application_root, self.root / "new-storage")
        self.assertEqual(restored.log_dir, self.root / "new-logs")
        for malformed in ({**saved, "api_key": "secret"}, {**saved, "version": 2},
                          {**saved, "version": True}, {key: value for key, value in saved.items() if key != "model_name"}):
            with self.assertRaises(ConfigurationError):
                restore_settings(malformed, application_root=self.root)

    def test_saved_endpoint_rejects_credentials_query_and_fragment(self):
        for endpoint in ("https://user:secret@host.invalid/v1", "https://user@host.invalid/v1",
                         "https://host.invalid/v1?key=secret", "https://host.invalid/v1#secret",
                         "https://host.invalid/v1?", "https://host.invalid/v1#"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ConfigurationError) as error:
                serialize_settings(replace(self.settings, base_url=endpoint))
            self.assertNotIn("secret", str(error.exception))

    def test_local_paths_work_without_any_model_configuration(self):
        env_file = self.root / "local.env"
        env_file.write_text("PROJECT_ROOT=project\nLOG_DIR=fresh-logs\nTIMEOUT_SECONDS=0.5\n", encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            paths = load_local_paths(env_file, application_root=self.root / "storage")
            limits = load_current_limits(env_file)
        self.assertEqual(paths, {"application_root": self.root / "storage", "project_root": self.project,
                                 "log_dir": self.root / "fresh-logs"})
        self.assertEqual(limits["max_model_calls"], 30)
        self.assertEqual(limits["timeout_seconds"], 0.5)
        self.assertNotIn("base_url", limits)

    async def test_actual_builtin_descriptions_restore_under_current_authorized_parent(self):
        cases = (
            ([DIRECTORY_TOOL], create_directory_executor(self.project, max_entries=2), "list_directory", {"path": "."}),
            ([TEXT_TOOL], create_text_executor(self.project, max_bytes=19, max_total_bytes=38), "read_text_file", {"path": "fact.txt"}),
            ([DIRECTORY_TOOL, TEXT_TOOL], create_filesystem_executor(
                self.project, max_entries=3, max_bytes=20, max_total_bytes=40), "read_text_file", {"path": "fact.txt"}),
        )
        for tools, executor, name, arguments in cases:
            with self.subTest(name=name, count=len(tools)):
                descriptor = describe_tools(tools, executor)
                rebuilt = restore_tools(descriptor, current_project_root=self.root)
                self.assertEqual(rebuilt["tools"], tools)
                self.assertEqual(await rebuilt["execute_tool"](name, arguments), await executor(name, arguments))
                self.assertEqual(describe_tools(tools, rebuilt["execute_tool"]), descriptor)
                if hasattr(rebuilt["execute_tool"], "bind_run"):
                    self.assertEqual(describe_tools(tools, rebuilt["execute_tool"].bind_run()), descriptor)
                descriptor["options"]["root"] = "mutated description"
                self.assertEqual(describe_tools(tools, executor)["options"]["root"], str(self.project))

    def test_current_directory_grant_cannot_be_replaced_by_saved_root_or_retarget_paths(self):
        descriptor = describe_tools([TEXT_TOOL], create_text_executor(self.project))
        child = self.project / "child"
        child.mkdir()
        other = self.root / "other"
        other.mkdir()
        for granted in (None, child, other):
            with self.subTest(granted=granted), self.assertRaises(ConfigurationError):
                restore_tools(descriptor, current_project_root=granted)
        self.assertEqual(restore_tools(descriptor, current_project_root=self.project)["tools"], [TEXT_TOOL])
        self.assertEqual(restore_tools(describe_tools([], None), current_project_root=self.project),
                         {"tools": [], "execute_tool": None})

    async def test_external_executor_requires_explicit_exact_binding_even_for_builtin_names(self):
        async def custom(name, arguments):
            return "application-defined result"
        saved = describe_tools([TEXT_TOOL], custom)
        self.assertEqual(saved["provider"], "external")
        with self.assertRaises(ConfigurationError):
            restore_tools(saved, current_project_root=self.project)
        with self.assertRaises(ConfigurationError):
            restore_tools(saved, current_project_root=None, tools=[DIRECTORY_TOOL], execute_tool=custom)
        rebuilt = restore_tools(saved, current_project_root=None, tools=[TEXT_TOOL], execute_tool=custom)
        self.assertIs(rebuilt["execute_tool"], custom)
        self.assertEqual(await rebuilt["execute_tool"]("read_text_file", {}), "application-defined result")
        altered = deepcopy(saved)
        altered["provider"] = "some.module:factory"
        with self.assertRaises(ConfigurationError):
            restore_tools(altered, current_project_root=self.project, tools=[TEXT_TOOL], execute_tool=custom)

    def test_numeric_differences_warn_without_lowering_saved_actual_limits(self):
        binding = describe_tools([TEXT_TOOL], create_text_executor(self.project, max_bytes=50, max_total_bytes=100))
        warnings = configuration_warnings(self.settings, binding,
            current_limits={"max_model_calls": 2, "text_max_bytes": 5, "text_total_max_bytes": 10})
        self.assertEqual(len(warnings), 3)
        self.assertIn("保存值 50", warnings[1])
        self.assertEqual(self.settings.max_model_calls, 41)
        self.assertEqual(binding["options"]["max_total_bytes"], 100)
        self.assertEqual(configuration_warnings(self.settings, binding), [])


if __name__ == "__main__":
    unittest.main()
