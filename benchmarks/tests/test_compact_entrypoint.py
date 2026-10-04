"""The legacy command delegates to one bounded Benchmark implementation."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import verify_compaction


class CompactEntrypointTests(unittest.TestCase):
    def test_explicit_config_is_forwarded_and_general_tasks_are_not_accepted(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "compact.json"
            path.write_text(json.dumps({"units": ["history-recall", "compaction-continuation"]}))
            with patch.object(verify_compaction.cli, "main", return_value=3) as execute:
                self.assertEqual(verify_compaction.main(["--config", str(path)]), 3)
                execute.assert_called_once_with(["run", "wechat-reporting", "--config", str(path)])
            path.write_text(json.dumps({"units": ["defect-reference"]}))
            with patch.object(verify_compaction.cli, "main") as execute, redirect_stderr(io.StringIO()):
                self.assertEqual(verify_compaction.main(["--config", str(path)]), 2)
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
