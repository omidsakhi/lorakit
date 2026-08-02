import itertools
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop, hflip, resize

from lorakit.face_parsing import PAD_LABEL, parse_map_for_image
from lorakit.identity import face_box_for_image
from lorakit.subject_masks import subject_mask_for_image

# Defaults reproduce the pre-augmentation behavior: one deterministic variant per
# image, no zoom/rotation, and flipping left to the legacy `random_flip` flag.
DEFAULT_AUGMENTATION = {
    "variants": 1,
    "scale_min": 1.0,
    "scale_max": 1.0,
    "rotation_degrees": 0.0,
    "random_flip": False,
    "seed": 42,
    "pad_mode": "reflect",
}
PAD_MODES = ("reflect", "edge", "constant")

# Falloff of the face-anchored trust region, in units of the trust radius.  Kept
# steep on purpose: a gradual fence hands the same pixel to the diffusion loss
# and the anchor at once, and those two then pull against each other.
TRUST_EDGE_SHARPNESS = 40.0


def _pad_array(array, pad, mode, fill=0):
    """Pad ``(left, top, right, bottom)`` around a H x W[ x C] array.

    NumPy is used directly rather than ``torchvision.transforms.functional.pad``
    because zoom-out can need a pad wider than the image itself, which the tensor
    backend rejects for reflect padding but NumPy handles by repeating the
    reflection.
    """
    left, top, right, bottom = pad
    width = ((top, bottom), (left, right))
    if array.ndim == 3:
        width = (*width, (0, 0))
    if mode == "constant":
        return np.pad(array, width, mode="constant", constant_values=fill)
    return np.pad(array, width, mode=mode)


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.

    Each source image expands into ``augmentation["variants"]`` pre-processed
    samples.  Variant 0 is the deterministic baseline; later variants apply random
    zoom, rotation, flip, and crop translation so the subject's face lands in a
    different place, scale, and orientation.  The subject alpha mask (or face
    parse map) and the ArcFace face box travel through the exact same geometry
    as the pixels.

    Every sample carries two latent-resolution weight maps.  ``subject`` scales
    the diffusion loss and ``background`` is the weight for anchoring a pixel to
    the frozen base model.  With a BiRefNet alpha alone they are complements.
    With a face parse map they are not: half-weighted skin must not be pulled
    halfway back toward the base prediction.  Zoom-out padding is never trained
    but is anchored as background, so the LoRA cannot drift on invented canvas.
    Passing *both* sources keeps the parse map for ``subject`` and takes
    ``background`` from the alpha, which is far cleaner at the person/scene
    boundary than the parser's own background class.
    """

    def __init__(
        self,
        dataset_folder,
        instance_prompt,
        class_prompt,
        class_data_root=None,
        class_num=None,
        resolution=1024,
        repeats=1,
        center_crop=False,
        random_flip=False,
        face_manifest=None,
        subject_mask_manifest=None,
        subject_mask_folder=None,
        parse_manifest=None,
        parse_folder=None,
        parse_train_lut=None,
        parse_background_lut=None,
        mask_resolution=None,
        augmentation=None,
        trust_radius=0.0,
    ):
        self.resolution = resolution
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt

        self.random_flip = random_flip
        self.augmentation = {**DEFAULT_AUGMENTATION, **(augmentation or {})}
        self.variants = int(self.augmentation["variants"])
        if self.variants < 1:
            raise ValueError("augmentation.variants must be >= 1")
        scale_min = float(self.augmentation["scale_min"])
        scale_max = float(self.augmentation["scale_max"])
        if not 0.0 < scale_min <= scale_max:
            raise ValueError("augmentation requires 0 < scale_min <= scale_max")
        rotation_degrees = float(self.augmentation["rotation_degrees"])
        if not math.isfinite(rotation_degrees) or rotation_degrees < 0.0:
            raise ValueError("augmentation.rotation_degrees must be a finite number >= 0")
        needs_pixel_mask = scale_min < 1.0 or rotation_degrees > 0.0
        if needs_pixel_mask and subject_mask_manifest is None and parse_manifest is None:
            # Zoom-out and rotation invent canvas pixels.  Without a per-pixel
            # mask those pixels would receive full diffusion loss and the model
            # would learn to reproduce the padding.
            raise ValueError(
                "augmentation.scale_min < 1.0 or rotation_degrees > 0 requires a "
                "per-pixel mask source (face_mask_source: face_parse or subject_mask)"
            )
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rotation_degrees = rotation_degrees
        self.augment_flip = bool(self.augmentation["random_flip"])
        self.pad_mode = str(self.augmentation["pad_mode"])
        if self.pad_mode not in PAD_MODES:
            raise ValueError(f"augmentation.pad_mode must be one of {PAD_MODES}")
        # Masks are only ever consumed in latent space, so store them at the
        # latent resolution instead of keeping a full-resolution copy per variant.
        self.mask_resolution = int(mask_resolution) if mask_resolution else resolution
        self.subject_mask_folder = (
            Path(subject_mask_folder) if subject_mask_folder is not None else None
        )
        self.parse_folder = Path(parse_folder) if parse_folder is not None else None
        if parse_manifest is not None:
            if self.parse_folder is None:
                raise ValueError("parse_folder is required when parse_manifest is set")
            if parse_train_lut is None or parse_background_lut is None:
                raise ValueError("parse weight lookup tables are required with parse_manifest")
        self.parse_train_lut = parse_train_lut
        self.parse_background_lut = parse_background_lut
        self.trust_radius = float(trust_radius or 0.0)
        if not math.isfinite(self.trust_radius) or self.trust_radius < 0.0:
            raise ValueError("trust_radius must be a finite number >= 0")
        if self.trust_radius > 0.0 and face_manifest is None:
            raise ValueError("trust_radius > 0 requires a face manifest (faces.json)")

        self.instance_data_root = Path(dataset_folder)
        if not self.instance_data_root.exists():
            raise ValueError(
                f"Instance images root '{self.instance_data_root.resolve()}' does not exist. "
                f"Set 'dataset_folder' in your config to the absolute path of the folder that "
                f"directly contains your training images."
            )
        if not self.instance_data_root.is_dir():
            raise ValueError(
                f"Instance images root '{self.instance_data_root.resolve()}' is not a directory. "
                f"Set 'dataset_folder' to a folder containing images, not a file."
            )

        supported_extensions = (".png", ".jpg", ".jpeg", ".webp")
        image_paths = sorted(
            path
            for path in self.instance_data_root.iterdir()
            if path.suffix.lower() in supported_extensions
        )
        if not image_paths:
            found = sorted(
                {p.suffix.lower() for p in self.instance_data_root.iterdir() if p.is_file()}
            )
            raise ValueError(
                f"No training images found in '{self.instance_data_root.resolve()}'. "
                f"Supported extensions are {supported_extensions}. "
                f"Make sure images are placed directly in this folder (not in subfolders)."
                + (
                    f" Files with these extensions were found and ignored: {found}."
                    if found
                    else ""
                )
            )
        instance_images = [(path, Image.open(path)) for path in image_paths]
        self.custom_instance_prompts = None

        self.instance_images = []
        for image_path, image in instance_images:
            self.instance_images.extend(itertools.repeat((image_path, image), repeats))

        # image processing to prepare for using SD-XL micro-conditioning
        self.original_sizes = []
        self.crop_top_lefts = []
        self.pixel_values = []
        self.instance_face_boxes = []
        self.instance_face_box_clipped = []
        self.instance_subject_masks = []
        self.instance_background_masks = []
        self.instance_subject_mask_valid = []
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        # A dedicated RNG keeps augmentation reproducible without disturbing the
        # global torch/random streams used elsewhere in training.
        rng = random.Random(int(self.augmentation["seed"]))
        self.missing_subject_masks = []
        self.missing_background_masks = []
        for image_path, source_image in self.instance_images:
            image = exif_transpose(source_image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            source_size = image.size
            face_box = (
                face_box_for_image(face_manifest, image_path, source_size)
                if face_manifest is not None
                else None
            )
            subject_mask = None
            if subject_mask_manifest is not None:
                if self.subject_mask_folder is None:
                    raise ValueError(
                        "subject_mask_folder is required when subject_mask_manifest is set"
                    )
                subject_mask = subject_mask_for_image(
                    subject_mask_manifest,
                    self.subject_mask_folder,
                    image_path,
                    source_size,
                )
                if subject_mask is None:
                    # With a parse map present the alpha only refines the anchor
                    # region, so its absence degrades rather than invalidates.
                    if parse_manifest is not None:
                        self.missing_background_masks.append(image_path.name)
                    else:
                        self.missing_subject_masks.append(image_path.name)
            parse_map = None
            if parse_manifest is not None:
                parse_map = parse_map_for_image(
                    parse_manifest, self.parse_folder, image_path, source_size
                )
                if parse_map is None:
                    self.missing_subject_masks.append(image_path.name)

            for variant_index in range(self.variants):
                plan = self._variant_plan(rng, variant_index)
                pixels, weights, box, clipped, crop_top_left = self._build_variant(
                    image, subject_mask, parse_map, face_box, plan, rng
                )
                self.original_sizes.append((source_size[1], source_size[0]))
                self.crop_top_lefts.append(crop_top_left)
                self.pixel_values.append(train_transforms(pixels))
                self.instance_face_boxes.append(box)
                self.instance_face_box_clipped.append(clipped)
                if weights is None:
                    # No mask for this image: train it unweighted and let the
                    # background objective skip the row via the valid flag.
                    self.instance_subject_masks.append(
                        torch.ones(1, self.mask_resolution, self.mask_resolution)
                    )
                    self.instance_background_masks.append(
                        torch.zeros(1, self.mask_resolution, self.mask_resolution)
                    )
                    self.instance_subject_mask_valid.append(False)
                else:
                    self.instance_subject_masks.append(weights[0])
                    self.instance_background_masks.append(weights[1])
                    self.instance_subject_mask_valid.append(True)

        self.num_instance_images = len(self.pixel_values)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution)
                if center_crop
                else transforms.RandomCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def _variant_plan(self, rng, variant_index):
        """Geometry for one variant: zoom, rotation, flip, and crop policy."""
        if variant_index == 0:
            # Baseline variant keeps the historical behavior so a run with
            # `variants: 1` is unchanged apart from the crop RNG source.
            return {
                "scale": 1.0,
                "angle": 0.0,
                "flip": self.random_flip and rng.random() < 0.5,
                "center_crop": self.center_crop,
            }
        angle = 0.0
        if self.rotation_degrees > 0.0:
            angle = rng.uniform(-self.rotation_degrees, self.rotation_degrees)
        return {
            "scale": rng.uniform(self.scale_min, self.scale_max),
            "angle": angle,
            "flip": self.augment_flip and rng.random() < 0.5,
            "center_crop": False,
        }

    def _build_variant(self, image, subject_mask, parse_map, face_box, plan, rng):
        """Apply one variant's geometry to the image, masks, and face box together."""
        scale = plan["scale"]
        target = int(round(self.resolution * scale))
        if scale >= 1.0:
            target = max(self.resolution, target)
        target = max(1, target)
        interpolation = transforms.InterpolationMode.BILINEAR
        source_width, source_height = image.size
        pixels = resize(image, target, interpolation=interpolation)
        mask = (
            resize(subject_mask, target, interpolation=interpolation)
            if subject_mask is not None
            else None
        )
        # Class indices must never be blended into each other.
        labels = (
            resize(parse_map, target, interpolation=transforms.InterpolationMode.NEAREST)
            if parse_map is not None
            else None
        )

        scale_x = pixels.width / source_width
        scale_y = pixels.height / source_height
        box = None
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            box = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)

        canvas = None
        pixels, mask, labels, canvas, box = self._rotate_variant(
            pixels, mask, labels, canvas, box, plan["angle"]
        )

        if pixels.width < self.resolution or pixels.height < self.resolution:
            pixels, mask, labels, canvas, box = self._pad_variant(pixels, mask, labels, canvas, box)

        if plan["flip"]:
            pixels = hflip(pixels)
            if mask is not None:
                mask = hflip(mask)
            if labels is not None:
                labels = hflip(labels)
            if canvas is not None:
                canvas = hflip(canvas)
            if box is not None:
                x1, y1, x2, y2 = box
                box = (pixels.width - x2, y1, pixels.width - x1, y2)

        max_top = max(0, pixels.height - self.resolution)
        max_left = max(0, pixels.width - self.resolution)
        if plan["center_crop"]:
            top = int(round(max_top / 2.0))
            left = int(round(max_left / 2.0))
        else:
            top = rng.randint(0, max_top) if max_top > 0 else 0
            left = rng.randint(0, max_left) if max_left > 0 else 0

        def crop_layer(layer):
            if layer is None:
                return None
            return crop(layer, top, left, self.resolution, self.resolution)

        pixels = crop_layer(pixels)
        mask = crop_layer(mask)
        labels = crop_layer(labels)
        canvas = crop_layer(canvas)

        box_out, clipped = self._crop_face_box(box, top, left)
        weights = self._weight_maps(mask, labels, canvas, box_out)
        return pixels, weights, box_out, clipped, (top, left)

    def _rotate_variant(self, pixels, mask, labels, canvas, box, angle):
        """Rotate every layer together, keeping the pre-rotation frame size.

        Rotation drags content in from outside the frame.  Each layer is padded
        first -- pixels with the configured pad mode, everything else with its
        own "invented" fill -- so the corners resolve to mirrored photo content.
        Rotating first and padding afterwards would tile the black rotation
        wedges across the whole zoom-out canvas.

        The corners are still marked invalid on ``canvas``, so like zoom-out
        padding they get zero diffusion loss and full background-anchor weight.
        """
        if abs(angle) < 1e-8:
            return pixels, mask, labels, canvas, box
        width, height = pixels.size
        radians = math.radians(angle)
        cos_a, sin_a = abs(math.cos(radians)), abs(math.sin(radians))
        # Widest source extent any output pixel can sample after rotating about
        # the centre, so the fill colour below never actually shows.
        pad_x = max(0, math.ceil((width * cos_a + height * sin_a - width) / 2.0))
        pad_y = max(0, math.ceil((width * sin_a + height * cos_a - height) / 2.0))
        pad = (pad_x, pad_y, pad_x, pad_y)

        if canvas is None:
            canvas = Image.new("L", (width, height), 255)
        box_mask = (
            Image.fromarray(self._rasterize_box_mask(box, (width, height)), mode="L")
            if box is not None
            else None
        )

        def turn(layer, pad_mode, fill, resample, image_mode, rotate_fill=None):
            if layer is None:
                return None
            padded = Image.fromarray(
                _pad_array(np.asarray(layer, dtype=np.uint8), pad, pad_mode, fill=fill),
                mode=image_mode,
            )
            rotated = padded.rotate(
                angle, resample=resample, fillcolor=fill if rotate_fill is None else rotate_fill
            )
            return crop(rotated, pad_y, pad_x, height, width)

        pixels = turn(pixels, self.pad_mode, 0, Image.BILINEAR, "RGB", rotate_fill=(0, 0, 0))
        mask = turn(mask, "constant", 0, Image.BILINEAR, "L")
        labels = turn(labels, "constant", PAD_LABEL, Image.NEAREST, "L")
        canvas = turn(canvas, "constant", 0, Image.NEAREST, "L")
        # The box travels as a raster through the identical transform, so its
        # bounds cannot drift from the pixels by a rounding step.
        if box_mask is not None:
            box = self._box_from_mask(turn(box_mask, "constant", 0, Image.NEAREST, "L"))
        return pixels, mask, labels, canvas, box

    @staticmethod
    def _rasterize_box_mask(box, size):
        """Binary mask covering the axis-aligned face box at ``size`` (W, H)."""
        width, height = size
        x1, y1, x2, y2 = box
        mask = np.zeros((height, width), dtype=np.uint8)
        xa = max(0, int(math.floor(x1)))
        xb = min(width, int(math.ceil(x2)))
        ya = max(0, int(math.floor(y1)))
        yb = min(height, int(math.ceil(y2)))
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = 255
        return mask

    @staticmethod
    def _box_from_mask(mask_image):
        """Axis-aligned bounds of non-zero pixels, or ``None`` if empty."""
        ys, xs = np.where(np.asarray(mask_image) > 0)
        if xs.size == 0:
            return None
        return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))

    def _pad_variant(self, pixels, mask, labels, canvas, box):
        """Grow a zoomed-out (or post-rotation) canvas back to training resolution.

        Each axis is padded by its full deficit on *both* sides so the random
        crop still has room to translate the subject.  ``canvas`` marks which
        pixels came from the source image; the invented ones get zero training
        weight and full background-anchor weight.
        """
        pad_x = max(0, self.resolution - pixels.width)
        pad_y = max(0, self.resolution - pixels.height)
        pad = (pad_x, pad_y, pad_x, pad_y)

        if canvas is None:
            canvas_array = np.full((pixels.height, pixels.width), 255, dtype=np.uint8)
        else:
            canvas_array = np.asarray(canvas, dtype=np.uint8)
        canvas = Image.fromarray(
            _pad_array(canvas_array, pad, "constant", fill=0),
            mode="L",
        )
        pixels = Image.fromarray(
            _pad_array(np.asarray(pixels, dtype=np.uint8), pad, self.pad_mode), mode="RGB"
        )
        if mask is not None:
            mask = Image.fromarray(
                _pad_array(np.asarray(mask, dtype=np.uint8), pad, "constant", fill=0), mode="L"
            )
        if labels is not None:
            labels = Image.fromarray(
                _pad_array(np.asarray(labels, dtype=np.uint8), pad, "constant", fill=PAD_LABEL),
                mode="L",
            )
        if box is not None:
            x1, y1, x2, y2 = box
            box = (x1 + pad_x, y1 + pad_y, x2 + pad_x, y2 + pad_y)
        return pixels, mask, labels, canvas, box

    def _weight_maps(self, mask, labels, canvas, box=None):
        """Build the ``(subject, background)`` latent-resolution weight pair.

        When a parse map and a subject alpha are both available, the alpha owns
        the person/scene split and the parse map only grades weights *inside*
        the person.  The parser is trained on face crops, so out in the scene it
        sprinkles stray ``hair`` and ``skin`` labels over dark furniture and bare
        arms; those punched holes in the anchor and, worse, pulled full-weight
        diffusion loss onto the couch.  Multiplying the trained region by the
        alpha and taking the anchor straight from its complement removes both.

        Both sources can still over-claim together: a salient-object alpha will
        happily swallow the couch the sitter leans on, and the parser calls that
        same couch ``hair``, so the region ends up trained *and* excluded from
        the anchor.  ``trust_radius`` fences that off -- past a multiple of the
        detected face box neither source is believed, so the pixels drop out of
        the trained region and are anchored like any other scene pixel.

        The person's non-face pixels (clothing, arms) are then neither trained
        nor anchored, which is the honest answer: the base model has no opinion
        about clothing it was never shown.  Zoom-out padding is the opposite
        case -- it is not the subject, so it is anchored like any other scene
        pixel while never contributing to the diffusion loss.
        """
        size = (self.mask_resolution, self.mask_resolution)
        if labels is not None:
            indices = np.asarray(labels, dtype=np.uint8)
            subject_weights = self.parse_train_lut[indices]
            background_weights = self.parse_background_lut[indices]
            if mask is not None:
                if mask.size != labels.size:
                    mask = mask.resize(labels.size, Image.BILINEAR)
                alpha = np.asarray(mask, dtype=np.float32) / 255.0
                subject_weights = subject_weights * alpha
                background_weights = 1.0 - alpha
            stacked = np.stack((subject_weights, background_weights))
            tensor = torch.from_numpy(stacked).float().unsqueeze(0)
            if tensor.shape[-2:] != size:
                # Area averaging turns the hard class boundaries into the same
                # soft edges the latent-space losses expect.
                tensor = torch.nn.functional.interpolate(tensor, size=size, mode="area")
            subject, background = tensor[0, 0:1], tensor[0, 1:2]
        elif mask is not None:
            if mask.size != size:
                mask = mask.resize(size, Image.BILINEAR)
            subject = transforms.functional.to_tensor(mask).float().clamp(0, 1)
            background = 1.0 - subject
        else:
            return None

        trust = self._trust_mask(box, size)
        if trust is not None:
            subject = subject * trust
            # Anything no longer believed to be the subject becomes scene, so it
            # is anchored rather than left unconstrained.
            background = 1.0 - (1.0 - background) * trust

        if canvas is not None:
            if canvas.size != size:
                canvas = canvas.resize(size, Image.BILINEAR)
            valid = transforms.functional.to_tensor(canvas).float().clamp(0, 1)
            # Invented canvas is never a training target, but it is still scene:
            # anchoring it holds the LoRA to the base prediction out there rather
            # than leaving a third of a zoomed-out frame with no constraint at all.
            subject = subject * valid
            background = background * valid + (1.0 - valid)
        return subject, background

    def _trust_mask(self, box, size):
        """Soft ellipse of ``trust_radius`` face boxes around the detected face.

        Returns ``None`` when the feature is off or the crop has no usable face,
        in which case the mask sources are taken at face value as before.
        """
        if self.trust_radius <= 0.0 or box is None:
            return None
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return None
        height, width = size
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half_width = (x2 - x1) / 2.0 * self.trust_radius
        half_height = (y2 - y1) / 2.0 * self.trust_radius
        # The box is in crop pixels; the weight maps live on the latent grid.
        xs = (np.arange(width, dtype=np.float32) + 0.5) * (self.resolution / width)
        ys = (np.arange(height, dtype=np.float32) + 0.5) * (self.resolution / height)
        dx = (xs - center_x) / half_width
        dy = (ys - center_y) / half_height
        distance = np.sqrt(dx[None, :] ** 2 + dy[:, None] ** 2)
        exponent = np.clip((distance - 1.0) * TRUST_EDGE_SHARPNESS, -60.0, 60.0)
        trust = 1.0 / (1.0 + np.exp(exponent))
        return torch.from_numpy(trust).float().unsqueeze(0)

    def _crop_face_box(self, box, top, left):
        """Translate a face box into crop coordinates, flagging unusable faces."""
        if box is None:
            return (-1.0, -1.0, -1.0, -1.0), False
        box_x1, box_y1, box_x2, box_y2 = box
        box_area = (box_x2 - box_x1) * (box_y2 - box_y1)
        visible_x1 = max(0.0, box_x1 - left)
        visible_y1 = max(0.0, box_y1 - top)
        visible_x2 = min(float(self.resolution), box_x2 - left)
        visible_y2 = min(float(self.resolution), box_y2 - top)
        visible_area = max(0.0, visible_x2 - visible_x1) * max(0.0, visible_y2 - visible_y1)
        # A face that is more than half outside the training crop is not a
        # meaningful ArcFace target.  Preserve the invalid sentinel rather
        # than silently replacing it with a full-frame crop.
        clipped = visible_area < box_area
        valid = box_area > 0.0 and visible_area / box_area >= 0.5
        if not valid:
            return (-1.0, -1.0, -1.0, -1.0), clipped
        return (visible_x1, visible_y1, visible_x2, visible_y2), clipped

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_index = index % self.num_instance_images
        example["instance_images"] = self.pixel_values[instance_index]
        example["original_size"] = self.original_sizes[instance_index]
        example["crop_top_left"] = self.crop_top_lefts[instance_index]
        example["index"] = instance_index
        example["instance_face_box"] = self.instance_face_boxes[instance_index]
        example["instance_face_box_clipped"] = self.instance_face_box_clipped[instance_index]
        example["instance_subject_mask"] = self.instance_subject_masks[instance_index]
        example["instance_background_mask"] = self.instance_background_masks[instance_index]
        example["instance_subject_mask_valid"] = self.instance_subject_mask_valid[instance_index]

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[instance_index]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # costum prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]
    indices = [example["index"] for example in examples]
    instance_face_boxes = [example["instance_face_box"] for example in examples]
    instance_face_box_clipped = [example["instance_face_box_clipped"] for example in examples]
    instance_subject_masks = [example["instance_subject_mask"] for example in examples]
    instance_background_masks = [example["instance_background_mask"] for example in examples]
    instance_subject_mask_valid = [example["instance_subject_mask_valid"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]
        original_sizes += [example["original_size"] for example in examples]
        crop_top_lefts += [example["crop_top_left"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
        "indices": indices,
        # These deliberately describe only the instance half of a prior batch.
        "instance_face_boxes": torch.tensor(instance_face_boxes, dtype=torch.float32),
        "instance_face_box_clipped": torch.tensor(instance_face_box_clipped, dtype=torch.bool),
        "instance_subject_masks": torch.stack(instance_subject_masks).float(),
        "instance_background_masks": torch.stack(instance_background_masks).float(),
        "instance_subject_mask_valid": torch.tensor(instance_subject_mask_valid, dtype=torch.bool),
    }
    return batch
