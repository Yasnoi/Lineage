"""Explicit migration, fixed summary caps and application-root settings."""

from dataclasses import asdict, replace
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from lineage_agent.config.settings import ConfigurationError, ModelSettings, load_settings


class CompactSettingsTests(unittest.TestCase):
    def setUp(self):
        self.settings = ModelSettings(base_url="https://settings.invalid", model_name="offline")

    def test_defaults_and_limits(self):
        self.assertEqual(self.settings.context_keep_recent_groups, 3)
        self.assertEqual(self.settings.summary_history_max_output_tokens, 16384)
        self.assertEqual(self.settings.summary_run_max_output_tokens, 8192)
        for name in ("context_keep_recent_turns", "summary_target_percent", "context_recovery_margin_percent"):
            self.assertNotIn(name, asdict(self.settings))
        for name in ("context_keep_recent_groups", "summary_history_max_output_tokens", "summary_run_max_output_tokens"):
            for value in (True, 0, -1, "3", 1.5):
                with self.subTest(name=name, value=value), self.assertRaises(ConfigurationError):
                    replace(self.settings, **{name: value})
        with TemporaryDirectory() as temporary, patch("os.getcwd", return_value=temporary):
            self.assertEqual(ModelSettings(base_url=self.settings.base_url, model_name="offline").application_root,
                             Path(temporary).resolve())
        for path in (None, "", "  ", 1):
            with self.subTest(path=path), self.assertRaises(ConfigurationError):
                replace(self.settings, application_root=path)
        small = replace(self.settings, context_budget_tokens=1000, max_output_tokens=128,
                        context_safety_margin_tokens=100)
        self.assertEqual(small.summary_history_max_output_tokens, 16384)
        self.assertEqual(small.summary_run_max_output_tokens, 8192)

    def test_old_turn_setting_is_rejected_in_constructor_and_environment(self):
        for field in ("context_keep_recent_turns", "summary_target_percent", "context_recovery_margin_percent",
                      "context_trigger_percent", "context_target_percent", "summary_max_output_tokens", "summary_max_bytes"):
            with self.subTest(field=field), self.assertRaisesRegex(ConfigurationError, field.upper()):
                replace(self.settings, **{field: 2})
            with TemporaryDirectory() as temporary, patch.dict(os.environ, {
                "BASE_URL": self.settings.base_url, "MODEL_NAME": self.settings.model_name, field.upper(): "2",
            }, clear=True):
                with self.assertRaisesRegex(ConfigurationError, "SUMMARY_HISTORY_MAX_OUTPUT_TOKENS"):
                    load_settings(Path(temporary) / "missing.env")

    def test_new_environment_values_are_loaded_without_old_field(self):
        with TemporaryDirectory() as temporary, patch.dict(os.environ, {
            "BASE_URL": self.settings.base_url, "MODEL_NAME": self.settings.model_name,
            "CONTEXT_KEEP_RECENT_GROUPS": "5", "MAX_OUTPUT_TOKENS": "256",
            "SUMMARY_HISTORY_MAX_OUTPUT_TOKENS": "3072", "SUMMARY_RUN_MAX_OUTPUT_TOKENS": "1024",
            "PROJECT_ROOT": "data",
        }, clear=True):
            configured = load_settings(Path(temporary) / "missing.env")
            self.assertEqual(configured.application_root, Path(temporary).resolve())
            self.assertEqual(configured.project_root, (Path(temporary) / "data").absolute())
        self.assertEqual(configured.context_keep_recent_groups, 5)
        self.assertEqual(configured.max_output_tokens, 256)
        self.assertEqual(configured.summary_history_max_output_tokens, 3072)
        self.assertEqual(configured.summary_run_max_output_tokens, 1024)


if __name__ == "__main__":
    unittest.main()
