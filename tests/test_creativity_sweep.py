"""Tests for creativity-sweep checkpoint resolution helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lorakit.creativity_sweep import (
    _parse_float_list,
    _parse_int_list,
    _sample_recipe,
    list_checkpoint_dirs,
    resolve_checkpoint,
)


class CreativitySweepHelpersTests(unittest.TestCase):
    def test_resolve_checkpoint_prefers_final_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "pytorch_lora_weights.safetensors").write_bytes(b"x")
            ckpt = root / "checkpoint_jane_1.0_0000000600"
            ckpt.mkdir()
            (ckpt / "pytorch_lora_weights.safetensors").write_bytes(b"y")
            path, label, step = resolve_checkpoint(root, "last")
            self.assertEqual(path, root.resolve())
            self.assertEqual(label, "final")
            self.assertIsNone(step)

    def test_resolve_checkpoint_falls_back_to_highest_step(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for step in (200, 600, 400):
                folder = root / f"checkpoint_jane_1.0_{step:010d}"
                folder.mkdir()
                (folder / "pytorch_lora_weights.safetensors").write_bytes(b"x")
            path, label, step = resolve_checkpoint(root, "latest")
            self.assertEqual(step, 600)
            self.assertTrue(path.name.endswith("0000000600"))
            self.assertEqual(label, path.name)

    def test_resolve_checkpoint_by_step(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "checkpoint_jane_1.0_0000000800"
            folder.mkdir()
            (folder / "pytorch_lora_weights.safetensors").write_bytes(b"x")
            path, _label, step = resolve_checkpoint(root, "800")
            self.assertEqual(path, folder.resolve())
            self.assertEqual(step, 800)

    def test_list_checkpoint_dirs_sorted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for step in (400, 200, 600):
                (root / f"checkpoint_x_{step:010d}").mkdir()
            steps = [int(path.name.split("_")[-1]) for path in list_checkpoint_dirs(root)]
            self.assertEqual(steps, [200, 400, 600])

    def test_sample_recipe_stays_in_bounds(self):
        import random

        rng = random.Random(0)
        for _ in range(20):
            recipe = _sample_recipe(
                rng,
                schedulers=("euler", "dpmpp"),
                steps_choices=(20, 30),
                guidance_min=3.0,
                guidance_max=9.0,
            )
            self.assertIn(recipe["scheduler"], {"euler", "dpmpp"})
            self.assertIn(recipe["steps"], {20, 30})
            self.assertGreaterEqual(recipe["guidance_scale"], 3.0)
            self.assertLessEqual(recipe["guidance_scale"], 9.0)

    def test_parse_helpers(self):
        self.assertEqual(_parse_float_list("3,9"), (3.0, 9.0))
        self.assertEqual(_parse_int_list("15,20,30"), (15, 20, 30))
        with self.assertRaises(ValueError):
            _parse_int_list("0,20")


if __name__ == "__main__":
    unittest.main()
