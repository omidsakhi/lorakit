"""Prepare offline face-parsing label maps for training (19-class BiSeNet).

Training reads the rendered ``*.png`` label maps plus ``parse.json``; it turns them
into per-pixel loss weights at dataset build time.  Storing raw class indices
rather than baked weights means retuning ``skin`` or excluding eyeglasses is a
config change, not a re-parse of the dataset.

Unlike the InsightFace manifest this needs no optional extras and no download:
the BiSeNet architecture lives in :mod:`lorakit.face_parsing_net`, depends only
on torch/torchvision, and reads a checkpoint vendored under ``models/``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose

from lorakit.face_manifest import IMAGE_EXTENSIONS, image_paths

MANIFEST_NAME = "parse.json"
DEFAULT_MODEL_PATH = "models/face-parsing-bisenet-resnet34/bisenet_resnet34.pt"
DEFAULT_BACKBONE = "resnet34"
# BiSeNet was trained at 512x512 with ImageNet normalization.
DEFAULT_INPUT_SIZE = 512
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# CelebAMask-HQ ordering: index 0 is background, 1..18 are the attributes.
CLASS_NAMES: tuple[str, ...] = (
    "background",
    "skin",
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "eye_g",
    "l_ear",
    "r_ear",
    "ear_r",
    "nose",
    "mouth",
    "u_lip",
    "l_lip",
    "neck",
    "neck_l",
    "cloth",
    "hair",
    "hat",
)
NUM_CLASSES = len(CLASS_NAMES)
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

# Reserved label for canvas that no source pixel maps to (zoom-out padding).  It
# is deliberately outside the class range so it can never be produced by argmax.
PAD_LABEL = 255

CLASS_GROUPS: dict[str, tuple[str, ...]] = {
    "background": ("background",),
    "skin": ("skin",),
    "features": (
        "l_brow",
        "r_brow",
        "l_eye",
        "r_eye",
        "l_ear",
        "r_ear",
        "nose",
        "mouth",
        "u_lip",
        "l_lip",
    ),
    "accessories": ("eye_g", "ear_r", "neck_l", "hat"),
    "neck": ("neck",),
    "cloth": ("cloth",),
    "hair": ("hair",),
}

# Skin is the largest and most repetitive region in a portrait, so at full weight
# it dominates the diffusion loss and drags the LoRA toward memorizing lighting
# and pose.  Halving it leaves hair, neck and the small features carrying the
# identity signal.  Clothing and accessories are excluded outright.
DEFAULT_CLASS_WEIGHTS: dict[str, float] = {
    "background": 0.0,
    "skin": 0.5,
    "features": 1.0,
    "accessories": 0.0,
    "neck": 1.0,
    "cloth": 0.0,
    "hair": 1.0,
}

# Classes that indicate the parser actually found a person, used for the
# min-coverage sanity check.  Accessories alone do not count.
_SUBJECT_GROUPS = ("skin", "features", "neck", "hair")

COLOR_LIST: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 0, 85),
    (255, 0, 170),
    (0, 255, 0),
    (85, 255, 0),
    (170, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (0, 85, 255),
    (0, 170, 255),
    (255, 255, 0),
    (255, 255, 85),
    (255, 255, 170),
    (255, 0, 255),
)


def resolve_class_weights(
    weights: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a group/class weight mapping into ``(train_lut, background_lut)``.

    Both are 256-entry float32 lookup tables indexed by parse label.  They are
    deliberately *not* complements:

    * ``train_lut`` scales the diffusion loss, so skin can sit at 0.5 while hair
      and neck stay at 1.0.
    * ``background_lut`` marks pixels that are *eligible* to be anchored to the
      frozen base model: 1.0 exactly where the training weight is zero, so
      half-weighted skin is not pulled halfway back toward the base prediction.
      It is an upper bound, not the final anchor weight -- the parser's own
      background class is unreliable around bare arms and busy scenery, so the
      dataset intersects this with a BiRefNet alpha when one is available.

    ``PAD_LABEL`` is zero in both LUTs; the dataset then promotes padded canvas
    to full background-anchor weight so invented pixels are preserved, not trained.
    Keys may name a group (``skin``, ``features``, ``accessories``, ``neck``,
    ``cloth``, ``hair``, ``background``) or an individual CelebAMask-HQ class;
    an individual class overrides the group it belongs to.
    """
    per_class: dict[str, float] = {}
    for group, weight in DEFAULT_CLASS_WEIGHTS.items():
        for name in CLASS_GROUPS[group]:
            per_class[name] = weight

    overrides: dict[str, float] = {}
    for key, value in (weights or {}).items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"face parse weight for {key!r} must be a number")
        weight = float(value)
        if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError(f"face parse weight for {key!r} must be in [0, 1]")
        if key in CLASS_GROUPS:
            for name in CLASS_GROUPS[key]:
                per_class[name] = weight
        elif key in CLASS_INDEX:
            overrides[key] = weight
        else:
            known = ", ".join(sorted(set(CLASS_GROUPS) | set(CLASS_INDEX)))
            raise ValueError(f"unknown face parse class or group {key!r}; use one of: {known}")
    per_class.update(overrides)

    train_lut = np.zeros(256, dtype=np.float32)
    background_lut = np.zeros(256, dtype=np.float32)
    for index, name in enumerate(CLASS_NAMES):
        weight = per_class[name]
        train_lut[index] = weight
        background_lut[index] = 1.0 if weight == 0.0 else 0.0
    return train_lut, background_lut


def subject_class_indices() -> tuple[int, ...]:
    """Label indices that mean 'a person was found here'."""
    return tuple(CLASS_INDEX[name] for group in _SUBJECT_GROUPS for name in CLASS_GROUPS[group])


def build_face_parser(
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    backbone: str = DEFAULT_BACKBONE,
    device: str = "cuda",
    input_size: int = DEFAULT_INPUT_SIZE,
) -> Callable[[Image.Image], np.ndarray]:
    """Return a callable mapping a PIL RGB image to a ``uint8`` label map.

    The returned array has the same height/width as the input image and holds
    class indices in ``[0, 18]``.
    """
    import torch
    from torchvision import transforms

    from lorakit.face_parsing_net import load_bisenet

    if input_size <= 0:
        raise ValueError("input_size must be positive")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"face-parsing checkpoint not found: {model_path}. "
            "See models/face-parsing-bisenet-resnet34/model_card.md."
        )

    torch_device = torch.device(device)
    model = load_bisenet(model_path, backbone_name=backbone, device=str(torch_device))

    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )

    def parse(image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        tensor = preprocess(rgb).unsqueeze(0).to(device=torch_device, dtype=torch.float32)
        with torch.no_grad():
            logits = model(tensor)[0]
        labels = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
        if labels.shape != (rgb.height, rgb.width):
            # Nearest resampling of the label map, not of the logits: upsampling
            # 19 channels to a multi-megapixel source would cost hundreds of MB.
            labels = np.asarray(
                Image.fromarray(labels, mode="L").resize((rgb.width, rgb.height), Image.NEAREST),
                dtype=np.uint8,
            )
        return labels

    return parse


def parse_path_for_image(parse_folder: str | Path, image_path: str | Path) -> Path:
    """Deterministic parse-map file location for a training image."""
    return Path(parse_folder) / f"{Path(image_path).stem}.png"


def load_parse_manifest(path: str | Path) -> dict[str, dict]:
    """Load and validate ``parse.json``.

    Entries map image file names to ``{"parse": <relative png>, "source_size":
    [w, h], "coverage": {<class>: <float>}, "subject_coverage": <float>}``.
    """
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"parse manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(data, dict) or not data:
        raise ValueError("parse manifest must be a non-empty JSON object")

    manifest: dict[str, dict] = {}
    for name, entry in data.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("parse manifest entries must map image names to objects")
        parse_name = entry.get("parse")
        if not isinstance(parse_name, str) or not parse_name:
            raise ValueError(f"parse manifest entry {name!r} needs a 'parse' file name")
        source_size = entry.get("source_size")
        if source_size is not None:
            if not isinstance(source_size, list) or len(source_size) != 2:
                raise ValueError(f"parse manifest source_size for {name!r} must be [width, height]")
            try:
                source_size = [int(value) for value in source_size]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"parse manifest source_size for {name!r} must be integral"
                ) from error
            if source_size[0] <= 0 or source_size[1] <= 0:
                raise ValueError(f"parse manifest source_size for {name!r} is invalid")
        subject_coverage = entry.get("subject_coverage")
        if subject_coverage is not None:
            try:
                subject_coverage = float(subject_coverage)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"parse manifest subject_coverage for {name!r} must be numeric"
                ) from error
        coverage = entry.get("coverage")
        if coverage is not None and not isinstance(coverage, dict):
            raise ValueError(f"parse manifest coverage for {name!r} must be an object")
        manifest[name] = {
            "parse": parse_name,
            "source_size": source_size,
            "coverage": coverage,
            "subject_coverage": subject_coverage,
        }
    return manifest


def parse_map_for_image(
    manifest: dict[str, dict],
    parse_folder: str | Path,
    image_path: str | Path,
    image_size: tuple[int, int],
) -> Image.Image | None:
    """Load one parse map as an ``L`` image, or ``None`` when absent."""
    image_path = Path(image_path)
    entry = manifest.get(image_path.name)
    if entry is None:
        return None
    expected_size = entry.get("source_size")
    if expected_size is not None and tuple(expected_size) != tuple(image_size):
        raise ValueError(
            f"parse manifest source_size for {image_path.name!r} is {expected_size}, "
            f"but the image is {list(image_size)}"
        )
    path = Path(parse_folder) / entry["parse"]
    if not path.is_file():
        raise FileNotFoundError(f"face parse map not found: {path}")
    with Image.open(path) as raw:
        labels = raw.convert("L")
        if labels.size != tuple(image_size):
            # Class indices must never be blended, so nearest only.
            labels = labels.resize(tuple(image_size), Image.NEAREST)
        return labels.copy()


def face_parses_need_build(image_folder: str | Path, parse_folder: str | Path) -> bool:
    """True when the manifest is missing/stale or any rendered parse map is gone."""
    image_folder = Path(image_folder)
    parse_folder = Path(parse_folder)
    manifest_path = parse_folder / MANIFEST_NAME
    if not manifest_path.is_file():
        return True
    try:
        manifest = load_parse_manifest(manifest_path)
    except (OSError, ValueError):
        return True
    manifest_mtime = manifest_path.stat().st_mtime
    for path in image_paths(image_folder):
        entry = manifest.get(path.name)
        if entry is None:
            return True
        if not (parse_folder / entry["parse"]).is_file():
            return True
        if path.stat().st_mtime > manifest_mtime:
            return True
    return False


def class_coverage(labels: np.ndarray) -> dict[str, float]:
    """Fraction of pixels per class name (classes with no pixels are omitted)."""
    counts = np.bincount(labels.reshape(-1), minlength=NUM_CLASSES)
    total = float(labels.size)
    return {
        CLASS_NAMES[index]: round(float(counts[index]) / total, 6)
        for index in range(NUM_CLASSES)
        if counts[index]
    }


def subject_coverage(labels: np.ndarray) -> float:
    """Fraction of pixels belonging to the person (skin, features, neck, hair)."""
    subject = np.isin(labels, np.array(subject_class_indices(), dtype=labels.dtype))
    return float(subject.mean())


def generate_face_parses(
    image_folder: str | Path,
    parse_folder: str | Path,
    parser: Callable[[Image.Image], np.ndarray],
    *,
    min_coverage: float = 0.01,
    preview: bool = False,
) -> tuple[dict[str, dict], list[Path]]:
    """Render one parse map per image; return the manifest plus failed images."""
    image_folder = Path(image_folder)
    parse_folder = Path(parse_folder)
    parse_folder.mkdir(parents=True, exist_ok=True)
    preview_folder = parse_folder / "previews"
    if preview:
        preview_folder.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    missing: list[Path] = []
    for path in image_paths(image_folder):
        try:
            with Image.open(path) as raw:
                image = exif_transpose(raw).convert("RGB")
                labels = parser(image)
                size = image.size
                rgb = np.asarray(image, dtype=np.uint8) if preview else None
        except OSError:
            missing.append(path)
            continue
        labels = np.asarray(labels, dtype=np.uint8)
        if labels.shape != (size[1], size[0]):
            raise ValueError(
                f"parser returned {labels.shape} for {path.name}, expected {(size[1], size[0])}"
            )
        if labels.max(initial=0) >= NUM_CLASSES:
            raise ValueError(
                f"parser returned label {int(labels.max())} for {path.name}, "
                f"expected values below {NUM_CLASSES}"
            )
        coverage = subject_coverage(labels)
        # A near-empty subject region means no face was found; training must not
        # silently anchor the whole frame to the base model.
        if coverage < min_coverage:
            missing.append(path)
            continue
        parse_file = parse_path_for_image(parse_folder, path)
        Image.fromarray(labels, mode="L").save(parse_file)
        if preview and rgb is not None:
            overlay = colorize_parse_map(labels, image=rgb)
            Image.fromarray(overlay, mode="RGB").save(
                preview_folder / f"{path.stem}.jpg", quality=92
            )
        manifest[path.name] = {
            "parse": parse_file.name,
            "source_size": list(size),
            "coverage": class_coverage(labels),
            "subject_coverage": round(coverage, 6),
        }
    return manifest, missing


def ensure_face_parses(
    image_folder: str | Path,
    parse_folder: str | Path | None = None,
    *,
    device: str = "cuda",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    backbone: str = DEFAULT_BACKBONE,
    input_size: int = DEFAULT_INPUT_SIZE,
    min_coverage: float = 0.01,
    allow_missing: bool = True,
    force: bool = False,
    preview: bool = False,
) -> Path:
    """Create face parse maps if missing/stale; return the parse folder."""
    image_folder = Path(image_folder)
    parse_folder = Path(parse_folder) if parse_folder is not None else image_folder / "face_parse"
    if not force and not face_parses_need_build(image_folder, parse_folder):
        return parse_folder

    print(f"Generating face parse maps for {image_folder} -> {parse_folder} ({backbone})")
    parser = build_face_parser(
        model_path=model_path, backbone=backbone, device=device, input_size=input_size
    )
    manifest, missing = generate_face_parses(
        image_folder, parse_folder, parser, min_coverage=min_coverage, preview=preview
    )
    if missing and not allow_missing:
        names = ", ".join(path.name for path in missing)
        raise RuntimeError(
            f"No usable face parse for {len(missing)} image(s): {names}. "
            "Fix the images/model settings or set allow_missing."
        )
    if missing:
        print(
            f"WARNING: {len(missing)} image(s) produced no usable face parse: "
            + ", ".join(path.name for path in missing)
        )
    if not manifest:
        raise RuntimeError(f"No face parse maps could be generated for {image_folder}")
    manifest_path = parse_folder / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} face parse map(s) to {parse_folder}")
    return parse_folder


def colorize_parse_map(labels: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
    """Render a label map with the CelebAMask-HQ palette, optionally blended."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[: len(COLOR_LIST)] = np.array(COLOR_LIST, dtype=np.uint8)
    colored = palette[labels]
    if image is None:
        return colored
    blended = image.astype(np.float32) * 0.6 + colored.astype(np.float32) * 0.4
    return np.clip(blended, 0, 255).astype(np.uint8)


def dump_dataset_parse_previews(
    output_dir: str | Path,
    dataset,
    *,
    limit: int = 24,
) -> int:
    """Write previews of the *augmented* samples and their two weight maps.

    These come from the dataset's stored variants rather than the source images,
    so they verify that zoom/flip/pad/crop kept the weights aligned with the
    pixels.  The overlay tints the trained region green and the anchored
    background region red; untinted areas are ignored by both losses.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def upsample(mask, size):
        return torch.nn.functional.interpolate(
            mask.float().unsqueeze(0), size=size, mode="bilinear", align_corners=False
        )[0].clamp(0, 1)

    written = 0
    total = min(len(dataset.pixel_values), limit)
    for index in range(total):
        pixels = dataset.pixel_values[index]
        valid = dataset.instance_subject_mask_valid[index]
        # Training tensors are normalized to [-1, 1]; recover displayable RGB.
        rgb = (pixels.float() * 0.5 + 0.5).clamp(0, 1)
        size = tuple(rgb.shape[-2:])

        train_full = upsample(dataset.instance_subject_masks[index], size)
        background_full = upsample(dataset.instance_background_masks[index], size)

        for name, mask in (("train", train_full), ("background", background_full)):
            mask_u8 = (mask[0] * 255.0).round().to(torch.uint8).cpu().numpy()
            Image.fromarray(mask_u8, mode="L").save(output_dir / f"{index:04d}_{name}.png")

        green = torch.tensor([0.15, 1.0, 0.35]).reshape(3, 1, 1)
        red = torch.tensor([1.0, 0.25, 0.15]).reshape(3, 1, 1)
        blended = rgb * (1.0 - 0.45 * train_full - 0.45 * background_full)
        blended = blended + green * (0.45 * train_full) + red * (0.45 * background_full)
        overlay = (blended.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0)
        suffix = "" if valid else "_noparse"
        Image.fromarray(overlay.cpu().numpy(), mode="RGB").save(
            output_dir / f"{index:04d}_overlay{suffix}.jpg", quality=92
        )
        written += 1
    return written


__all__ = [
    "CLASS_GROUPS",
    "CLASS_INDEX",
    "CLASS_NAMES",
    "COLOR_LIST",
    "DEFAULT_BACKBONE",
    "DEFAULT_CLASS_WEIGHTS",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_MODEL_PATH",
    "IMAGE_EXTENSIONS",
    "MANIFEST_NAME",
    "NUM_CLASSES",
    "PAD_LABEL",
    "build_face_parser",
    "class_coverage",
    "colorize_parse_map",
    "dump_dataset_parse_previews",
    "ensure_face_parses",
    "face_parses_need_build",
    "generate_face_parses",
    "load_parse_manifest",
    "parse_map_for_image",
    "parse_path_for_image",
    "resolve_class_weights",
    "subject_class_indices",
    "subject_coverage",
]
