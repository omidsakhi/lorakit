"""Tests for restoring trained LoRA alpha/rank scaling at inference time."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from lorakit.lora_scaling import (
    find_training_target_modules,
    format_scaling_report,
    parse_rank_alpha,
    training_scaling_map,
)

TARGETS = {
    "ff.net.0.proj": [32, 32],
    "ff.net.2": [16, 16],
    "to_v": [16, 16],
    "to_out.0": [8, 8],
}


class LoraScalingTests(unittest.TestCase):
    def test_parse_rank_alpha_forms(self):
        self.assertEqual(parse_rank_alpha([16, 8]), (16, 8))
        self.assertEqual(parse_rank_alpha("(32, 16)"), (32, 16))
        self.assertEqual(parse_rank_alpha(4), (4, 4))
        with self.assertRaises(ValueError):
            parse_rank_alpha([1, 2, 3])

    def test_alpha_equals_rank_scales_every_module_to_one(self):
        scaling = training_scaling_map(TARGETS)
        self.assertEqual(
            scaling,
            {"ff.net.0.proj": 1.0, "ff.net.2": 1.0, "to_v": 1.0, "to_out.0": 1.0},
        )

    def test_scaling_map_uses_alpha_over_rank(self):
        scaling = training_scaling_map({"to_v": [16, 8], "to_out.0": [8, 16]})
        self.assertAlmostEqual(scaling["to_v"], 0.5)
        self.assertAlmostEqual(scaling["to_out.0"], 2.0)

    def test_find_training_target_modules_from_checkpoint_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            experiment = Path(temp)
            (experiment / "config.yaml").write_text(
                yaml.safe_dump({"config": {"train": {"lora": {"target_modules": TARGETS}}}}),
                encoding="utf-8",
            )
            checkpoint = experiment / "checkpoint_jane_1.0_0000000600"
            checkpoint.mkdir()
            weights = checkpoint / "pytorch_lora_weights.safetensors"
            weights.write_bytes(b"x")
            # Resolves by walking up from the checkpoint weight file.
            found = find_training_target_modules(weights)
            self.assertEqual(found, TARGETS)

    def test_find_training_target_modules_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(find_training_target_modules(Path(temp)))

    def test_format_scaling_report_lists_transitions(self):
        text = format_scaling_report(
            {
                "applied": {"to_v": 1.0},
                "previous": {"to_v": 0.5},
                "layers": {"to_v": 140},
            }
        )
        self.assertIn("to_v", text)
        self.assertIn("0.500->1.000", text)
        self.assertIn("140", text)


if __name__ == "__main__":
    unittest.main()
