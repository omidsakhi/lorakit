"""Tests for output-folder helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lorakit.utils import ensure_fresh_experiment_folder


class EnsureFreshExperimentFolderTests(unittest.TestCase):
    def test_missing_folder_is_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            ensure_fresh_experiment_folder(Path(temp) / "new_run")

    def test_existing_folder_raises(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "jane_1.0"
            folder.mkdir()
            (folder / "logs").mkdir()
            with self.assertRaises(FileExistsError) as ctx:
                ensure_fresh_experiment_folder(folder)
            self.assertIn("already exists", str(ctx.exception))
            self.assertIn("Bump `version`", str(ctx.exception))

    def test_existing_folder_allowed_when_resuming(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "jane_1.0"
            folder.mkdir()
            ensure_fresh_experiment_folder(folder, resume_from_checkpoint="latest")


if __name__ == "__main__":
    unittest.main()
