"""Explicit migration and validation of compact protection/recovery settings."""

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
        self.assertEqual(self.settings.context_recovery_margin_percent, 10)
        self.assertEqual(self.settings.summary_target_percent, 40)
        self.assertNotIn("context_keep_recent_turns", asdict(self.settings))
        for value in (True, 0, -1, "3", 1.5):
            with self.subTest(groups=value), self.assertRaises(ConfigurationError):
                replace(self.settings, context_keep_recent_groups=value)
        for value in (True, -1, 100, "10", 1.5):
            with self.subTest(margin=value), self.assertRaises(ConfigurationError):
                replace(self.settings, context_recovery_margin_percent=value)
        for value in (0, 99):
            self.assertEqual(replace(self.settings, context_recovery_margin_percent=value).context_recovery_margin_percent, value)

    def test_old_turn_setting_is_rejected_in_constructor_and_environment(self):
        with self.assertRaisesRegex(ConfigurationError, "CONTEXT_KEEP_RECENT_GROUPS"):
            replace(self.settings, context_keep_recent_turns=2)
        with TemporaryDirectory() as temporary, patch.dict(os.environ, {
            "BASE_URL": self.settings.base_url, "MODEL_NAME": self.settings.model_name,
            "CONTEXT_KEEP_RECENT_TURNS": "2",
        }, clear=True):
            with self.assertRaisesRegex(ConfigurationError, "CONTEXT_KEEP_RECENT_GROUPS"):
                load_settings(Path(temporary) / "missing.env")

    def test_new_environment_values_are_loaded_without_old_field(self):
        with TemporaryDirectory() as temporary, patch.dict(os.environ, {
            "BASE_URL": self.settings.base_url, "MODEL_NAME": self.settings.model_name,
            "CONTEXT_KEEP_RECENT_GROUPS": "5", "CONTEXT_RECOVERY_MARGIN_PERCENT": "0",
        }, clear=True):
            configured = load_settings(Path(temporary) / "missing.env")
        self.assertEqual(configured.context_keep_recent_groups, 5)
        self.assertEqual(configured.context_recovery_margin_percent, 0)


if __name__ == "__main__":
    unittest.main()
