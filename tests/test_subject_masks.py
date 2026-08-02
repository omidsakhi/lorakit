"""Tests for offline subject masks and mask-aware augmentation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lorakit.datasets import DreamBoothDataset, collate_fn
from lorakit.latent_face_detector import LatentBackgroundPreservationObjective
from lorakit.subject_masks import (
    BUILD_KEY,
    MANIFEST_NAME,
    generate_subject_masks,
    load_mask_build_settings,
    load_mask_manifest,
    subject_masks_need_build,
)


def _write_subject_image(path: Path, size=(200, 160)) -> None:
    """An image whose left half is the 'subject' and right half the background."""
    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, : width // 2] = (220, 180, 160)
    array[:, width // 2 :] = (20, 40, 90)
    Image.fromarray(array, mode="RGB").save(path)


def _left_half_segmenter(image: Image.Image) -> np.ndarray:
    width, height = image.size
    alpha = np.zeros((height, width), dtype=np.float32)
    alpha[:, : width // 2] = 1.0
    return alpha


class SubjectMaskManifestTests(unittest.TestCase):
    def test_generate_and_load_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            _write_subject_image(images / "a.png")
            _write_subject_image(images / "b.png")
            masks = Path(temp) / "masks"

            self.assertTrue(subject_masks_need_build(images, masks))
            manifest, missing = generate_subject_masks(images, masks, _left_half_segmenter)
            self.assertEqual(missing, [])
            self.assertEqual(sorted(manifest), ["a.png", "b.png"])
            self.assertAlmostEqual(manifest["a.png"]["coverage"], 0.5, places=2)
            (masks / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            loaded = load_mask_manifest(masks / MANIFEST_NAME)
            self.assertEqual(loaded["a.png"]["mask"], "a.png")
            self.assertFalse(subject_masks_need_build(images, masks))

    def _write_manifest(self, images: Path, masks: Path, build: dict | None) -> Path:
        manifest, _ = generate_subject_masks(images, masks, _left_half_segmenter)
        if build is not None:
            manifest = {BUILD_KEY: build, **manifest}
        path = masks / MANIFEST_NAME
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

    def test_masks_are_rebuilt_when_the_segmenter_changes(self):
        """A different model_id/input_size must not silently reuse cached alphas."""
        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            _write_subject_image(images / "a.png")
            masks = Path(temp) / "masks"
            build = {"model_id": "org/general", "input_size": 1024}
            path = self._write_manifest(images, masks, build)

            self.assertEqual(load_mask_build_settings(path), build)
            self.assertFalse(
                subject_masks_need_build(images, masks, model_id="org/general", input_size=1024)
            )
            self.assertTrue(
                subject_masks_need_build(images, masks, model_id="org/portrait", input_size=1024)
            )
            self.assertTrue(
                subject_masks_need_build(images, masks, model_id="org/general", input_size=2048)
            )

    def test_masks_without_a_build_record_are_rebuilt(self):
        """Masks predating the record cannot be verified, so they are not trusted."""
        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            _write_subject_image(images / "a.png")
            masks = Path(temp) / "masks"
            path = self._write_manifest(images, masks, None)

            self.assertIsNone(load_mask_build_settings(path))
            # Callers that do not care which model ran still see a fresh cache.
            self.assertFalse(subject_masks_need_build(images, masks))
            self.assertTrue(subject_masks_need_build(images, masks, model_id="org/general"))

    def test_empty_mask_is_reported_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            _write_subject_image(images / "a.png")
            masks = Path(temp) / "masks"

            def blank(image):
                return np.zeros((image.size[1], image.size[0]), dtype=np.float32)

            manifest, missing = generate_subject_masks(images, masks, blank)
            self.assertEqual(manifest, {})
            self.assertEqual([path.name for path in missing], ["a.png"])


class MaskAugmentationTests(unittest.TestCase):
    def _dataset(self, temp: Path, **augmentation):
        images = temp / "images"
        images.mkdir(exist_ok=True)
        _write_subject_image(images / "a.png")
        masks = temp / "masks"
        manifest, _ = generate_subject_masks(images, masks, _left_half_segmenter)
        return DreamBoothDataset(
            images,
            "sks",
            "person",
            resolution=64,
            subject_mask_manifest=manifest,
            subject_mask_folder=masks,
            mask_resolution=8,
            augmentation=augmentation or None,
        )

    def test_variants_expand_the_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(
                Path(temp), variants=4, scale_min=1.0, scale_max=1.5, random_flip=True, seed=7
            )
            self.assertEqual(dataset.num_instance_images, 4)
            self.assertEqual(len(dataset.instance_subject_masks), 4)
            self.assertTrue(all(dataset.instance_subject_mask_valid))
            for mask in dataset.instance_subject_masks:
                self.assertEqual(tuple(mask.shape), (1, 8, 8))
            # Random zoom/flip/translate must actually move the subject around.
            coverages = {round(float(m.mean()), 4) for m in dataset.instance_subject_masks}
            self.assertGreater(len(coverages), 1)

    def test_mask_tracks_pixels_through_augmentation(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(
                Path(temp), variants=6, scale_min=1.0, scale_max=1.4, random_flip=True, seed=3
            )
            for index in range(dataset.num_instance_images):
                pixels = dataset.pixel_values[index]
                mask = dataset.instance_subject_masks[index]
                alpha = torch.nn.functional.interpolate(
                    mask.unsqueeze(0), size=pixels.shape[-2:], mode="nearest"
                )[0, 0]
                # The subject half is bright/warm, the background dark blue. Where the
                # mask says "subject", the red channel must dominate the blue one.
                red, blue = pixels[0], pixels[2]
                subject = alpha > 0.5
                if subject.any():
                    self.assertGreater(
                        float((red - blue)[subject].mean()),
                        0.0,
                        f"variant {index} mask does not line up with subject pixels",
                    )
                background = alpha < 0.5
                if background.any():
                    self.assertLess(float((red - blue)[background].mean()), 0.0)

    def test_mask_tracks_pixels_through_rotation(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(
                Path(temp),
                variants=6,
                scale_min=1.0,
                scale_max=1.0,
                rotation_degrees=15,
                random_flip=False,
                seed=5,
            )
            for index in range(dataset.num_instance_images):
                pixels = dataset.pixel_values[index]
                mask = dataset.instance_subject_masks[index]
                alpha = torch.nn.functional.interpolate(
                    mask.unsqueeze(0), size=pixels.shape[-2:], mode="nearest"
                )[0, 0]
                red, blue = pixels[0], pixels[2]
                subject = alpha > 0.5
                if subject.any():
                    self.assertGreater(
                        float((red - blue)[subject].mean()),
                        0.0,
                        f"rotated variant {index} mask does not line up with subject pixels",
                    )

    def test_missing_mask_falls_back_to_ones(self):
        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            _write_subject_image(images / "a.png")
            _write_subject_image(images / "b.png")
            masks = Path(temp) / "masks"
            manifest, _ = generate_subject_masks(images, masks, _left_half_segmenter)
            del manifest["b.png"]

            dataset = DreamBoothDataset(
                images,
                "sks",
                "person",
                resolution=64,
                subject_mask_manifest=manifest,
                subject_mask_folder=masks,
                mask_resolution=8,
            )
            self.assertEqual(dataset.missing_subject_masks, ["b.png"])
            self.assertEqual(dataset.instance_subject_mask_valid, [True, False])
            batch = collate_fn([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["instance_subject_masks"].shape), (2, 1, 8, 8))
            torch.testing.assert_close(batch["instance_subject_masks"][1], torch.ones(1, 8, 8))
            self.assertEqual(batch["instance_subject_mask_valid"].tolist(), [True, False])


class SubjectMaskObjectiveTests(unittest.TestCase):
    def _objective(self):
        return LatentBackgroundPreservationObjective(mask_edge_sharpness=45.0)

    def test_background_loss_ignores_subject_region(self):
        objective = self._objective()
        # Subject occupies the left half of the latent grid.
        subject = torch.zeros(1, 1, 4, 4)
        subject[..., :2] = 1.0
        base = torch.zeros(1, 4, 4, 4)
        prediction = torch.zeros(1, 4, 4, 4)
        # Deviate only inside the subject: background loss must stay zero.
        prediction[..., :2] = 5.0
        loss, stats = objective(
            noisy_instance_latents=base,
            model_prediction=prediction,
            base_model_prediction=base,
            timesteps=torch.zeros(1, dtype=torch.long),
            scheduler=None,
            num_train_timesteps=1000,
            subject_mask=subject,
            subject_mask_valid=torch.tensor([True]),
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertEqual(stats["valid_faces"], 1.0)

        # Now deviate in the background only: the loss must be positive.
        prediction = torch.zeros(1, 4, 4, 4)
        prediction[..., 2:] = 5.0
        loss, _ = objective(
            noisy_instance_latents=base,
            model_prediction=prediction,
            base_model_prediction=base,
            timesteps=torch.zeros(1, dtype=torch.long),
            scheduler=None,
            num_train_timesteps=1000,
            subject_mask=subject,
            subject_mask_valid=torch.tensor([True]),
        )
        self.assertGreater(float(loss), 0.0)

    def test_invalid_rows_are_skipped(self):
        objective = self._objective()
        subject = torch.ones(2, 1, 4, 4)
        base = torch.zeros(2, 4, 4, 4)
        loss, stats = objective(
            noisy_instance_latents=base,
            model_prediction=torch.ones(2, 4, 4, 4),
            base_model_prediction=base,
            timesteps=torch.zeros(2, dtype=torch.long),
            scheduler=None,
            num_train_timesteps=1000,
            subject_mask=subject,
            subject_mask_valid=torch.tensor([False, False]),
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(stats["valid_faces"], 0.0)
        self.assertEqual(stats["skipped_faces"], 2.0)

    def test_face_focus_mask_uses_subject_alpha(self):
        objective = self._objective()
        subject = torch.zeros(2, 1, 4, 4)
        subject[..., :2] = 1.0
        mask, stats = objective.face_focus_mask(
            noisy_instance_latents=torch.zeros(2, 4, 4, 4),
            base_model_prediction=torch.zeros(2, 4, 4, 4),
            timesteps=torch.zeros(2, dtype=torch.long),
            scheduler=None,
            num_train_timesteps=1000,
            height=4,
            width=4,
            subject_mask=subject,
            subject_mask_valid=torch.tensor([True, False]),
        )
        # Row 0 uses the alpha; row 1 has no mask and stays unweighted (all ones).
        torch.testing.assert_close(mask[0], subject[0])
        torch.testing.assert_close(mask[1], torch.ones(1, 4, 4))
        self.assertEqual(stats["valid_faces"], 1.0)


if __name__ == "__main__":
    unittest.main()
