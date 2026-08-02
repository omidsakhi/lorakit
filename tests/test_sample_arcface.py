"""Tests for sampling-time ArcFace scoring helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from lorakit.sample_arcface import (
    record_step_scores,
    reference_cohesion,
    score_generated_samples,
)


class _FakeEncoder(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Deterministic embedding from mean pixel value so same crops match.
        value = images.float().mean(dim=(1, 2, 3), keepdim=True).view(-1, 1)
        return value.expand(-1, 8)


class _DirectionEncoder(torch.nn.Module):
    """Maps mean pixel value onto an arc so distinct crops get distinct directions.

    Inputs arrive normalized to [-1, 1], so a black crop lands on the first axis
    and a white crop on the second.
    """

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        value = images.float().mean(dim=(1, 2, 3))
        theta = (value + 1.0) / 2.0 * (torch.pi / 2.0)
        embeds = torch.zeros(value.shape[0], 8)
        embeds[:, 0] = torch.cos(theta)
        embeds[:, 1] = torch.sin(theta)
        return embeds


class SampleArcFaceTests(unittest.TestCase):
    def test_record_step_scores_tracks_best_step(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "arcface_scores.json"
            record_step_scores(
                path,
                {
                    "step": 200,
                    "mean_max_similarity": 0.4,
                    "mean_similarity": 0.35,
                    "detection_rate": 1.0,
                    "n_samples": 2,
                    "n_detected": 2,
                },
            )
            summary = record_step_scores(
                path,
                {
                    "step": 400,
                    "mean_max_similarity": 0.7,
                    "mean_similarity": 0.6,
                    "detection_rate": 0.9,
                    "n_samples": 2,
                    "n_detected": 2,
                },
            )
            self.assertEqual(summary["best_step"], 400)
            self.assertEqual(summary["ranked_by"], "mean_max_similarity")
            self.assertAlmostEqual(summary["best_mean_max_similarity"], 0.7)
            self.assertAlmostEqual(summary["best_mean_similarity"], 0.6)
            # Upsert same step with a lower score should demote it.
            summary = record_step_scores(
                path,
                {
                    "step": 400,
                    "mean_max_similarity": 0.3,
                    "mean_similarity": 0.25,
                    "detection_rate": 0.9,
                    "n_samples": 2,
                    "n_detected": 2,
                },
            )
            self.assertEqual(summary["best_step"], 200)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([entry["step"] for entry in data["steps"]], [200, 400])

    def test_score_generated_samples_mean_over_detected(self):
        encoder = _FakeEncoder().eval()
        reference = torch.ones(1, 8)
        device = torch.device("cpu")

        def detector(_bgr):
            # Full-frame face.
            return [SimpleNamespace(bbox=[0.0, 0.0, 64.0, 64.0], det_score=0.99)]

        images = [
            Image.new("RGB", (64, 64), (255, 255, 255)),
            Image.new("RGB", (64, 64), (0, 0, 0)),
        ]

        def empty_detector(_bgr):
            return []

        scored = score_generated_samples(
            images=images,
            encoder=encoder,
            reference_embeddings=reference,
            detector=detector,
            device=device,
            crop_size=112,
            bbox_padding_fraction=0.0,
            image_names=["a.jpg", "b.jpg"],
        )
        self.assertEqual(scored["n_samples"], 2)
        self.assertEqual(scored["n_detected"], 2)
        self.assertIsNotNone(scored["mean_max_similarity"])
        self.assertEqual(scored["samples"][0]["image"], "a.jpg")

        missed = score_generated_samples(
            images=images,
            encoder=encoder,
            reference_embeddings=reference,
            detector=empty_detector,
            device=device,
            crop_size=112,
            bbox_padding_fraction=0.0,
        )
        self.assertEqual(missed["n_detected"], 0)
        self.assertIsNone(missed["mean_max_similarity"])
        self.assertIsNone(missed["mean_similarity"])
        self.assertEqual(missed["detection_rate"], 0.0)

    def test_score_generated_samples_takes_max_over_references(self):
        """Each sample scores against its best reference, not the centroid."""
        encoder = _DirectionEncoder().eval()
        device = torch.device("cpu")
        # One reference matching a black crop, one matching a white crop.
        references = torch.zeros(2, 8)
        references[0, 0] = 1.0
        references[1, 1] = 1.0

        def detector(_bgr):
            return [SimpleNamespace(bbox=[0.0, 0.0, 64.0, 64.0], det_score=0.99)]

        images = [
            Image.new("RGB", (64, 64), (0, 0, 0)),
            Image.new("RGB", (64, 64), (255, 255, 255)),
        ]

        scored = score_generated_samples(
            images=images,
            encoder=encoder,
            reference_embeddings=references,
            detector=detector,
            device=device,
            crop_size=112,
            bbox_padding_fraction=0.0,
            image_names=["black.jpg", "white.jpg"],
            reference_names=["ref_black.jpg", "ref_white.jpg"],
        )

        self.assertEqual(scored["n_references"], 2)
        self.assertEqual(scored["samples"][0]["best_reference"], "ref_black.jpg")
        self.assertEqual(scored["samples"][1]["best_reference"], "ref_white.jpg")
        for row in scored["samples"]:
            self.assertAlmostEqual(row["max_similarity"], 1.0, places=5)
        # Mean of per-sample maxima beats the centroid, which sits between refs.
        self.assertAlmostEqual(scored["mean_max_similarity"], 1.0, places=5)
        self.assertAlmostEqual(scored["mean_similarity"], 2.0**-0.5, places=5)
        self.assertAlmostEqual(scored["reference_cohesion"], 0.0, places=6)

    def test_reference_cohesion_needs_two_faces(self):
        self.assertIsNone(reference_cohesion(torch.ones(1, 8)))
        pair = torch.zeros(2, 4)
        pair[0, 0] = 1.0
        pair[1, 0] = 1.0
        self.assertAlmostEqual(reference_cohesion(pair), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
