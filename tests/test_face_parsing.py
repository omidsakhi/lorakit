"""Tests for offline face parsing, its weight maps, and zoom-out augmentation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lorakit.datasets import DreamBoothDataset, collate_fn
from lorakit.face_parsing import (
    CLASS_INDEX,
    MANIFEST_NAME,
    PAD_LABEL,
    class_coverage,
    face_parses_need_build,
    generate_face_parses,
    load_parse_manifest,
    resolve_class_weights,
    subject_coverage,
)
from lorakit.latent_face_detector import LatentBackgroundPreservationObjective

SKIN = CLASS_INDEX["skin"]
HAIR = CLASS_INDEX["hair"]
CLOTH = CLASS_INDEX["cloth"]
BACKGROUND = CLASS_INDEX["background"]


def _write_portrait(path: Path, size=(64, 64)) -> None:
    """Rows of hair / skin / clothing over a dark background, top to bottom."""
    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :] = (10, 20, 40)
    quarter = height // 4
    array[:quarter] = (60, 40, 30)
    array[quarter : 2 * quarter] = (220, 180, 160)
    array[2 * quarter :] = (30, 120, 60)
    Image.fromarray(array, mode="RGB").save(path)


def _banded_parser(image: Image.Image) -> np.ndarray:
    """Label the same bands the test image draws."""
    width, height = image.size
    labels = np.full((height, width), BACKGROUND, dtype=np.uint8)
    quarter = height // 4
    labels[:quarter] = HAIR
    labels[quarter : 2 * quarter] = SKIN
    labels[2 * quarter :] = CLOTH
    return labels


class ClassWeightTests(unittest.TestCase):
    def test_defaults_halve_skin_and_drop_clothing(self):
        train, background = resolve_class_weights()
        self.assertEqual(train[SKIN], 0.5)
        self.assertEqual(train[HAIR], 1.0)
        self.assertEqual(train[CLASS_INDEX["neck"]], 1.0)
        self.assertEqual(train[CLASS_INDEX["nose"]], 1.0)
        self.assertEqual(train[CLOTH], 0.0)
        self.assertEqual(train[BACKGROUND], 0.0)
        # The two maps are not complements: half-weighted skin is not anchored.
        self.assertEqual(background[SKIN], 0.0)
        self.assertEqual(background[CLOTH], 1.0)
        self.assertEqual(background[BACKGROUND], 1.0)

    def test_padding_label_is_zero_in_both_maps(self):
        train, background = resolve_class_weights()
        self.assertEqual(train[PAD_LABEL], 0.0)
        self.assertEqual(background[PAD_LABEL], 0.0)

    def test_individual_class_overrides_its_group(self):
        train, background = resolve_class_weights({"features": 0.25, "nose": 1.0})
        self.assertEqual(train[CLASS_INDEX["l_eye"]], 0.25)
        self.assertEqual(train[CLASS_INDEX["nose"]], 1.0)
        self.assertEqual(background[CLASS_INDEX["l_eye"]], 0.0)

    def test_zeroing_a_group_moves_it_into_the_anchor(self):
        train, background = resolve_class_weights({"hair": 0.0})
        self.assertEqual(train[HAIR], 0.0)
        self.assertEqual(background[HAIR], 1.0)

    def test_invalid_names_and_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            resolve_class_weights({"eyebrows": 1.0})
        with self.assertRaises(ValueError):
            resolve_class_weights({"skin": 1.5})
        with self.assertRaises(ValueError):
            resolve_class_weights({"skin": "half"})


class ParseManifestTests(unittest.TestCase):
    def test_generate_and_load_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            _write_portrait(images / "a.png")
            _write_portrait(images / "b.png")
            parses = Path(temp) / "face_parse"

            self.assertTrue(face_parses_need_build(images, parses))
            manifest, missing = generate_face_parses(images, parses, _banded_parser)
            self.assertEqual(missing, [])
            self.assertEqual(sorted(manifest), ["a.png", "b.png"])
            # Hair and skin are a quarter of the frame each; clothing is a half.
            self.assertAlmostEqual(manifest["a.png"]["subject_coverage"], 0.5, places=3)
            self.assertAlmostEqual(manifest["a.png"]["coverage"]["cloth"], 0.5, places=3)
            (parses / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            loaded = load_parse_manifest(parses / MANIFEST_NAME)
            self.assertEqual(loaded["a.png"]["parse"], "a.png")
            self.assertFalse(face_parses_need_build(images, parses))

    def test_parse_map_round_trips_class_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            _write_portrait(images / "a.png")
            parses = Path(temp) / "face_parse"
            generate_face_parses(images, parses, _banded_parser)

            with Image.open(parses / "a.png") as raw:
                stored = np.asarray(raw.convert("L"), dtype=np.uint8)
            np.testing.assert_array_equal(stored, _banded_parser(Image.new("RGB", (64, 64))))

    def test_no_subject_is_reported_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            _write_portrait(images / "a.png")
            parses = Path(temp) / "face_parse"

            def all_cloth(image):
                return np.full((image.size[1], image.size[0]), CLOTH, dtype=np.uint8)

            manifest, missing = generate_face_parses(images, parses, all_cloth)
            self.assertEqual(manifest, {})
            self.assertEqual([path.name for path in missing], ["a.png"])

    def test_out_of_range_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            _write_portrait(images / "a.png")

            def bogus(image):
                return np.full((image.size[1], image.size[0]), 99, dtype=np.uint8)

            with self.assertRaises(ValueError):
                generate_face_parses(images, Path(temp) / "face_parse", bogus)

    def test_coverage_helpers(self):
        labels = _banded_parser(Image.new("RGB", (64, 64)))
        coverage = class_coverage(labels)
        self.assertAlmostEqual(coverage["hair"], 0.25, places=3)
        self.assertNotIn("neck", coverage)
        self.assertAlmostEqual(subject_coverage(labels), 0.5, places=3)


class ParseDatasetTests(unittest.TestCase):
    def _dataset(self, temp: Path, *, weights=None, **augmentation):
        images = temp / "images"
        images.mkdir(exist_ok=True)
        _write_portrait(images / "a.png")
        parses = temp / "face_parse"
        manifest, _ = generate_face_parses(images, parses, _banded_parser)
        train_lut, background_lut = resolve_class_weights(weights)
        return DreamBoothDataset(
            images,
            "sks",
            "person",
            resolution=64,
            parse_manifest=manifest,
            parse_folder=parses,
            parse_train_lut=train_lut,
            parse_background_lut=background_lut,
            mask_resolution=16,
            augmentation=augmentation or None,
        )

    def test_weights_follow_the_class_map(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(Path(temp))
            train = dataset.instance_subject_masks[0][0]
            background = dataset.instance_background_masks[0][0]
            self.assertEqual(tuple(train.shape), (16, 16))
            quarter = 16 // 4
            # Hair band trains at full weight, skin at half, clothing not at all.
            self.assertAlmostEqual(float(train[1, 8]), 1.0, places=4)
            self.assertAlmostEqual(float(train[quarter + 1, 8]), 0.5, places=4)
            self.assertAlmostEqual(float(train[2 * quarter + 1, 8]), 0.0, places=4)
            # Clothing is anchored to the base model instead.
            self.assertAlmostEqual(float(background[2 * quarter + 1, 8]), 1.0, places=4)
            self.assertAlmostEqual(float(background[quarter + 1, 8]), 0.0, places=4)

    def test_weights_are_not_complements(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(Path(temp))
            train = dataset.instance_subject_masks[0]
            background = dataset.instance_background_masks[0]
            self.assertGreater(float((train + background - 1.0).abs().max()), 0.1)

    def test_zoom_out_pads_are_anchored_not_trained(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(Path(temp), variants=6, scale_min=0.5, scale_max=0.8, seed=11)
            self.assertEqual(dataset.num_instance_images, 6)
            anchored_pads = 0
            for index in range(1, dataset.num_instance_images):
                train = dataset.instance_subject_masks[index]
                background = dataset.instance_background_masks[index]
                # Invented canvas: no diffusion loss, full background anchor.
                pad = (train <= 1e-4) & (background >= 1.0 - 1e-4)
                if bool(pad.any()):
                    anchored_pads += 1
                    self.assertFalse(bool(((train > 1e-4) & pad).any()))
            self.assertEqual(anchored_pads, dataset.num_instance_images - 1)

    def test_zoom_out_requires_a_pixel_mask_source(self):
        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            _write_portrait(images / "a.png")
            with self.assertRaises(ValueError):
                DreamBoothDataset(
                    images, "sks", "person", resolution=64, augmentation={"scale_min": 0.5}
                )

    def test_rotation_requires_a_pixel_mask_source(self):
        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            _write_portrait(images / "a.png")
            with self.assertRaises(ValueError):
                DreamBoothDataset(
                    images,
                    "sks",
                    "person",
                    resolution=64,
                    augmentation={"rotation_degrees": 15, "variants": 2},
                )

    def test_rotation_corners_are_filled_from_the_photo_not_black(self):
        # The portrait's darkest channel is 10/255, so an exactly-black pixel can
        # only come from a rotation fill leaking into the zoom-out padding.
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(
                Path(temp), variants=6, scale_min=0.5, scale_max=0.8, rotation_degrees=15, seed=4
            )
            for index in range(dataset.num_instance_images):
                raw = (dataset.pixel_values[index] * 0.5 + 0.5) * 255.0
                black = (raw <= 1.0).all(dim=0)
                self.assertFalse(
                    bool(black.any()),
                    f"variant {index} has {int(black.sum())} black rotation-fill pixels",
                )

    def test_rotation_keeps_the_frame_size(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(
                Path(temp), variants=4, scale_min=1.0, scale_max=1.0, rotation_degrees=15, seed=8
            )
            for pixels in dataset.pixel_values:
                self.assertEqual(tuple(pixels.shape[-2:]), (64, 64))

    def test_rotation_pads_are_anchored_not_trained(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(
                Path(temp), variants=6, scale_min=1.0, scale_max=1.0, rotation_degrees=15, seed=11
            )
            self.assertEqual(dataset.num_instance_images, 6)
            anchored_pads = 0
            for index in range(1, dataset.num_instance_images):
                train = dataset.instance_subject_masks[index]
                background = dataset.instance_background_masks[index]
                pad = (train <= 1e-4) & (background >= 1.0 - 1e-4)
                if bool(pad.any()):
                    anchored_pads += 1
            self.assertGreater(anchored_pads, 0)

    def test_pad_mode_is_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                self._dataset(Path(temp), scale_min=0.5, pad_mode="wrap")

    def test_batch_carries_both_weight_maps(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(Path(temp), variants=2, scale_min=1.0, scale_max=1.2)
            batch = collate_fn([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["instance_subject_masks"].shape), (2, 1, 16, 16))
            self.assertEqual(tuple(batch["instance_background_masks"].shape), (2, 1, 16, 16))
            self.assertEqual(batch["instance_subject_mask_valid"].tolist(), [True, True])

    def test_missing_parse_map_trains_unweighted(self):
        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            _write_portrait(images / "a.png")
            _write_portrait(images / "b.png")
            parses = Path(temp) / "face_parse"
            manifest, _ = generate_face_parses(images, parses, _banded_parser)
            del manifest["b.png"]
            train_lut, background_lut = resolve_class_weights()

            dataset = DreamBoothDataset(
                images,
                "sks",
                "person",
                resolution=64,
                parse_manifest=manifest,
                parse_folder=parses,
                parse_train_lut=train_lut,
                parse_background_lut=background_lut,
                mask_resolution=16,
            )
            self.assertEqual(dataset.missing_subject_masks, ["b.png"])
            self.assertEqual(dataset.instance_subject_mask_valid, [True, False])
            torch.testing.assert_close(dataset.instance_subject_masks[1], torch.ones(1, 16, 16))
            torch.testing.assert_close(dataset.instance_background_masks[1], torch.zeros(1, 16, 16))


class TrustRadiusTests(unittest.TestCase):
    """Both mask sources can over-claim together; the face box fences them off."""

    # A face filling the head band, and furniture in the bottom-right corner that
    # the parser calls `hair` while the alpha calls it subject.
    FACE_BOX = (20.0, 4.0, 44.0, 28.0)
    FURNITURE = slice(48, 64)

    def _parser(self, image: Image.Image) -> np.ndarray:
        labels = _banded_parser(image)
        labels[self.FURNITURE, self.FURNITURE] = HAIR
        return labels

    @staticmethod
    def _greedy_segmenter(image: Image.Image) -> np.ndarray:
        """A salient-object alpha that swallows the whole scene, couch included."""
        return np.ones((image.height, image.width), dtype=np.float32)

    def _dataset(self, temp: Path, *, trust_radius, face_manifest=True):
        from lorakit.subject_masks import generate_subject_masks

        images = temp / "images"
        images.mkdir(exist_ok=True)
        _write_portrait(images / "a.png")
        parses = temp / "face_parse"
        parse_manifest, _ = generate_face_parses(images, parses, self._parser)
        masks = temp / "masks"
        mask_manifest, _ = generate_subject_masks(images, masks, self._greedy_segmenter)
        train_lut, background_lut = resolve_class_weights()
        return DreamBoothDataset(
            images,
            "sks",
            "person",
            resolution=64,
            face_manifest=(
                {"a.png": {"bbox_xyxy": list(self.FACE_BOX), "source_size": [64, 64]}}
                if face_manifest
                else None
            ),
            parse_manifest=parse_manifest,
            parse_folder=parses,
            parse_train_lut=train_lut,
            parse_background_lut=background_lut,
            subject_mask_manifest=mask_manifest,
            subject_mask_folder=masks,
            mask_resolution=16,
            trust_radius=trust_radius,
        )

    def test_without_a_trust_radius_furniture_is_trained_and_unanchored(self):
        # The failure this exists to fix: the alpha says subject, the parser says
        # hair, so the couch takes full diffusion loss and never gets anchored.
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(Path(temp), trust_radius=0.0)
            self.assertAlmostEqual(float(dataset.instance_subject_masks[0][0][14, 14]), 1.0, 4)
            self.assertAlmostEqual(float(dataset.instance_background_masks[0][0][14, 14]), 0.0, 4)

    def test_trust_radius_anchors_furniture_instead_of_training_it(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(Path(temp), trust_radius=1.5)
            self.assertLess(float(dataset.instance_subject_masks[0][0][14, 14]), 0.01)
            self.assertGreater(float(dataset.instance_background_masks[0][0][14, 14]), 0.99)

    def test_trust_radius_leaves_the_face_region_alone(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(Path(temp), trust_radius=1.5)
            train = dataset.instance_subject_masks[0][0]
            # Hair on the head keeps full weight, skin its graded half weight.
            self.assertGreater(float(train[1, 8]), 0.9)
            self.assertGreater(float(train[5, 8]), 0.45)

    def test_trust_radius_requires_a_face_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                self._dataset(Path(temp), trust_radius=1.5, face_manifest=False)


class CombinedSourceTests(unittest.TestCase):
    """The alpha owns person vs scene; the parse map only grades the person."""

    # The parser is trained on face crops, so out in the scene it mislabels dark
    # furniture as hair.  Reproduce that as a block in the top-right corner.
    STRAY_COLUMN = 48
    PERSON = slice(16, 48)

    def _parser(self, image: Image.Image) -> np.ndarray:
        labels = _banded_parser(image)
        labels[: image.height // 4, self.STRAY_COLUMN :] = HAIR
        return labels

    def _segmenter(self, image: Image.Image) -> np.ndarray:
        alpha = np.zeros((image.height, image.width), dtype=np.float32)
        alpha[:, self.PERSON] = 1.0
        return alpha

    def _dataset(self, temp: Path, *, drop_alpha=False):
        from lorakit.subject_masks import generate_subject_masks

        images = temp / "images"
        images.mkdir(exist_ok=True)
        _write_portrait(images / "a.png")
        parses = temp / "face_parse"
        parse_manifest, _ = generate_face_parses(images, parses, self._parser)
        masks = temp / "masks"
        mask_manifest, _ = generate_subject_masks(images, masks, self._segmenter)
        if drop_alpha:
            mask_manifest = {}
        train_lut, background_lut = resolve_class_weights()
        return DreamBoothDataset(
            images,
            "sks",
            "person",
            resolution=64,
            parse_manifest=parse_manifest,
            parse_folder=parses,
            parse_train_lut=train_lut,
            parse_background_lut=background_lut,
            subject_mask_manifest=mask_manifest,
            subject_mask_folder=masks,
            mask_resolution=16,
        )

    def test_alpha_confines_training_to_the_person(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(Path(temp))
            train = dataset.instance_subject_masks[0][0]
            # Hair on the person keeps full weight...
            self.assertAlmostEqual(float(train[1, 6]), 1.0, places=4)
            # ...while the same label out in the scene is dropped.
            self.assertAlmostEqual(float(train[1, 14]), 0.0, places=4)
            self.assertAlmostEqual(float(train[5, 6]), 0.5, places=4)

    def test_anchor_is_the_complement_of_the_alpha(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(Path(temp))
            background = dataset.instance_background_masks[0][0]
            # The mislabeled furniture is anchored rather than left as a hole.
            self.assertAlmostEqual(float(background[1, 14]), 1.0, places=4)
            self.assertAlmostEqual(float(background[9, 14]), 1.0, places=4)
            self.assertAlmostEqual(float(background[1, 6]), 0.0, places=4)

    def test_clothing_is_neither_trained_nor_anchored(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(Path(temp))
            train = dataset.instance_subject_masks[0][0]
            background = dataset.instance_background_masks[0][0]
            self.assertAlmostEqual(float(train[9, 6]), 0.0, places=4)
            self.assertAlmostEqual(float(background[9, 6]), 0.0, places=4)

    def test_the_two_maps_never_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(Path(temp))
            train = dataset.instance_subject_masks[0]
            background = dataset.instance_background_masks[0]
            self.assertAlmostEqual(float((train * background).max()), 0.0, places=6)

    def test_missing_alpha_falls_back_to_the_parse_map(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self._dataset(Path(temp), drop_alpha=True)
            # Degraded, not invalid: the sample still trains off the parse map.
            self.assertEqual(dataset.missing_background_masks, ["a.png"])
            self.assertEqual(dataset.missing_subject_masks, [])
            self.assertEqual(dataset.instance_subject_mask_valid, [True])
            background = dataset.instance_background_masks[0][0]
            self.assertAlmostEqual(float(background[9, 6]), 1.0, places=4)

    def test_zoom_out_padding_is_background_with_alpha(self):
        from lorakit.subject_masks import generate_subject_masks

        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            _write_portrait(images / "a.png")
            parses = Path(temp) / "face_parse"
            parse_manifest, _ = generate_face_parses(images, parses, self._parser)
            masks = Path(temp) / "masks"
            mask_manifest, _ = generate_subject_masks(images, masks, self._segmenter)
            train_lut, background_lut = resolve_class_weights()
            dataset = DreamBoothDataset(
                images,
                "sks",
                "person",
                resolution=64,
                parse_manifest=parse_manifest,
                parse_folder=parses,
                parse_train_lut=train_lut,
                parse_background_lut=background_lut,
                subject_mask_manifest=mask_manifest,
                subject_mask_folder=masks,
                mask_resolution=16,
                augmentation={
                    "variants": 4,
                    "scale_min": 0.5,
                    "scale_max": 0.7,
                    "seed": 7,
                },
            )
            saw_pad = False
            for index in range(1, dataset.num_instance_images):
                train = dataset.instance_subject_masks[index]
                background = dataset.instance_background_masks[index]
                pad = (train <= 1e-4) & (background >= 1.0 - 1e-4)
                if bool(pad.any()):
                    saw_pad = True
                    # Invented canvas must never carry diffusion loss.
                    self.assertFalse(bool((train[pad] > 1e-4).any()))
            self.assertTrue(saw_pad)


class ParseObjectiveTests(unittest.TestCase):
    def _objective(self):
        return LatentBackgroundPreservationObjective(mask_edge_sharpness=45.0)

    def test_explicit_background_mask_replaces_the_complement(self):
        objective = self._objective()
        # Left column trains at half weight; only the right column is anchored.
        subject = torch.zeros(1, 1, 4, 4)
        subject[..., :2] = 0.5
        background = torch.zeros(1, 1, 4, 4)
        background[..., 2:] = 1.0
        base = torch.zeros(1, 4, 4, 4)

        prediction = torch.zeros(1, 4, 4, 4)
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
            background_mask=background,
        )
        # Deviating inside the half-weighted region must not be penalized, which
        # is exactly what `1 - subject` would have done.
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertAlmostEqual(stats["background_coverage"], 0.5, places=6)

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
            background_mask=background,
        )
        self.assertAlmostEqual(float(loss), 25.0, places=4)

    def test_omitted_background_mask_keeps_the_complement(self):
        objective = self._objective()
        subject = torch.zeros(1, 1, 4, 4)
        subject[..., :2] = 1.0
        base = torch.zeros(1, 4, 4, 4)
        prediction = torch.zeros(1, 4, 4, 4)
        prediction[..., 2:] = 5.0
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
        self.assertAlmostEqual(float(loss), 25.0, places=4)
        self.assertAlmostEqual(stats["background_coverage"], 0.5, places=6)

    def test_padding_included_in_background_is_anchored(self):
        objective = self._objective()
        # Column 0 trains, columns 1-3 are background (scene + zoom-out pad).
        subject = torch.zeros(1, 1, 4, 4)
        subject[..., 0] = 1.0
        background = torch.zeros(1, 1, 4, 4)
        background[..., 1:] = 1.0
        base = torch.zeros(1, 4, 4, 4)
        prediction = torch.zeros(1, 4, 4, 4)
        prediction[..., 2:] = 2.0
        loss, stats = objective(
            noisy_instance_latents=base,
            model_prediction=prediction,
            base_model_prediction=base,
            timesteps=torch.zeros(1, dtype=torch.long),
            scheduler=None,
            num_train_timesteps=1000,
            subject_mask=subject,
            subject_mask_valid=torch.tensor([True]),
            background_mask=background,
        )
        # MSE = 4 over half the anchored columns (2 of 3) → 4 * (2/3) = 8/3.
        self.assertAlmostEqual(float(loss), 8.0 / 3.0, places=4)
        self.assertAlmostEqual(stats["background_coverage"], 0.75, places=6)


if __name__ == "__main__":
    unittest.main()
