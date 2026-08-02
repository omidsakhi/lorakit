"""Prepare offline subject alpha masks for training (BiRefNet background removal).

Training reads the rendered ``*.png`` masks plus ``masks.json``; it never imports
BiRefNet directly.  It does build them mid-run when they are missing, so
BiRefNet's requirements (timm, einops, kornia) are ordinary dependencies rather
than an extra -- a training run cannot install an extra for itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose

from lorakit.face_manifest import IMAGE_EXTENSIONS, image_paths

MANIFEST_NAME = "masks.json"
# Reserved manifest key recording which segmenter produced the masks.  Image
# entries are keyed by file name, which always carries an extension, so this
# cannot collide with one.
BUILD_KEY = "__build__"
DEFAULT_MODEL_ID = "ZhengPeng7/BiRefNet"
# BiRefNet is trained at 1024x1024 with ImageNet normalization.
DEFAULT_INPUT_SIZE = 1024
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_birefnet_segmenter(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    input_size: int = DEFAULT_INPUT_SIZE,
) -> Callable[[Image.Image], np.ndarray]:
    """Return a callable mapping a PIL RGB image to a float32 alpha in ``[0, 1]``.

    The returned alpha has the same height/width as the input image.
    """
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    if input_size <= 0:
        raise ValueError("input_size must be positive")

    try:
        model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
    except ImportError as error:
        # BiRefNet ships its architecture as remote code, so its imports are
        # resolved here rather than through lorakit's own import graph.
        raise RuntimeError(
            f"BiRefNet's remote code needs a package this environment is missing: {error} "
            "Run `uv sync` to install lorakit's dependencies."
        ) from error
    model.eval()
    torch_device = torch.device(device)
    # BiRefNet's swin backbone is numerically fragile in fp16; keep it fp32.
    model.to(torch_device, dtype=torch.float32)
    model.requires_grad_(False)

    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )

    def segment(image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        tensor = preprocess(rgb).unsqueeze(0).to(device=torch_device, dtype=torch.float32)
        with torch.no_grad():
            output = model(tensor)
        # BiRefNet returns a list of progressively refined maps; the last is final.
        logits = output[-1] if isinstance(output, (list, tuple)) else output
        while isinstance(logits, (list, tuple)):
            logits = logits[-1]
        logits = logits.float()
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        alpha = torch.sigmoid(logits)
        alpha = torch.nn.functional.interpolate(
            alpha, size=(rgb.height, rgb.width), mode="bilinear", align_corners=False
        )
        return alpha[0, 0].clamp(0, 1).cpu().numpy().astype(np.float32)

    return segment


def mask_path_for_image(mask_folder: str | Path, image_path: str | Path) -> Path:
    """Deterministic mask file location for a training image."""
    return Path(mask_folder) / f"{Path(image_path).stem}.png"


def load_mask_manifest(path: str | Path) -> dict[str, dict]:
    """Load and validate ``masks.json``.

    Entries map image file names to ``{"mask": <relative png>, "source_size": [w, h],
    "coverage": <float>}``.  The reserved ``BUILD_KEY`` entry is not a mask and is
    skipped; read it with :func:`load_mask_build_settings`.
    """
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"mask manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(data, dict) or not data:
        raise ValueError("mask manifest must be a non-empty JSON object")

    manifest: dict[str, dict] = {}
    for name, entry in data.items():
        if name == BUILD_KEY:
            continue
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("mask manifest entries must map image names to objects")
        mask_name = entry.get("mask")
        if not isinstance(mask_name, str) or not mask_name:
            raise ValueError(f"mask manifest entry {name!r} needs a 'mask' file name")
        source_size = entry.get("source_size")
        if source_size is not None:
            if not isinstance(source_size, list) or len(source_size) != 2:
                raise ValueError(f"mask manifest source_size for {name!r} must be [width, height]")
            try:
                source_size = [int(value) for value in source_size]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"mask manifest source_size for {name!r} must be integral"
                ) from error
            if source_size[0] <= 0 or source_size[1] <= 0:
                raise ValueError(f"mask manifest source_size for {name!r} is invalid")
        coverage = entry.get("coverage")
        if coverage is not None:
            try:
                coverage = float(coverage)
            except (TypeError, ValueError) as error:
                raise ValueError(f"mask manifest coverage for {name!r} must be numeric") from error
        manifest[name] = {
            "mask": mask_name,
            "source_size": source_size,
            "coverage": coverage,
        }
    if not manifest:
        raise ValueError("mask manifest must be a non-empty JSON object")
    return manifest


def load_mask_build_settings(path: str | Path) -> dict | None:
    """Return the segmenter settings recorded in ``masks.json``, if any.

    Masks written before this record existed return ``None``; the caller cannot
    tell which model produced them.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    build = data.get(BUILD_KEY)
    return build if isinstance(build, dict) else None


def subject_mask_for_image(
    manifest: dict[str, dict],
    mask_folder: str | Path,
    image_path: str | Path,
    image_size: tuple[int, int],
) -> Image.Image | None:
    """Load one subject mask as an ``L`` image, or ``None`` when absent."""
    image_path = Path(image_path)
    entry = manifest.get(image_path.name)
    if entry is None:
        return None
    expected_size = entry.get("source_size")
    if expected_size is not None and tuple(expected_size) != tuple(image_size):
        raise ValueError(
            f"mask manifest source_size for {image_path.name!r} is {expected_size}, "
            f"but the image is {list(image_size)}"
        )
    path = Path(mask_folder) / entry["mask"]
    if not path.is_file():
        raise FileNotFoundError(f"subject mask not found: {path}")
    with Image.open(path) as raw:
        mask = raw.convert("L")
        if mask.size != tuple(image_size):
            mask = mask.resize(tuple(image_size), Image.BILINEAR)
        return mask.copy()


def subject_masks_need_build(
    image_folder: str | Path,
    mask_folder: str | Path,
    *,
    model_id: str | None = None,
    input_size: int | None = None,
) -> bool:
    """True when the manifest is missing/stale or any rendered mask is gone.

    Passing ``model_id``/``input_size`` also rebuilds when the cached masks came
    from different segmenter settings.  Masks predating that record carry no
    settings to compare, so they are treated as stale rather than assumed to
    match: silently training against another model's alpha is the worse failure.
    """
    image_folder = Path(image_folder)
    mask_folder = Path(mask_folder)
    manifest_path = mask_folder / MANIFEST_NAME
    if not manifest_path.is_file():
        return True
    try:
        manifest = load_mask_manifest(manifest_path)
    except (OSError, ValueError):
        return True
    if model_id is not None or input_size is not None:
        build = load_mask_build_settings(manifest_path)
        if build is None:
            return True
        if model_id is not None and build.get("model_id") != model_id:
            return True
        if input_size is not None and build.get("input_size") != input_size:
            return True
    paths = image_paths(image_folder)
    manifest_mtime = manifest_path.stat().st_mtime
    for path in paths:
        entry = manifest.get(path.name)
        if entry is None:
            return True
        if not (mask_folder / entry["mask"]).is_file():
            return True
        if path.stat().st_mtime > manifest_mtime:
            return True
    return False


def generate_subject_masks(
    image_folder: str | Path,
    mask_folder: str | Path,
    segmenter: Callable[[Image.Image], np.ndarray],
    *,
    min_coverage: float = 0.01,
) -> tuple[dict[str, dict], list[Path]]:
    """Render one alpha mask per image; return the manifest plus failed images."""
    image_folder = Path(image_folder)
    mask_folder = Path(mask_folder)
    mask_folder.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    missing: list[Path] = []
    for path in image_paths(image_folder):
        try:
            with Image.open(path) as raw:
                image = exif_transpose(raw).convert("RGB")
                alpha = segmenter(image)
                size = image.size
        except OSError:
            missing.append(path)
            continue
        if alpha.shape != (size[1], size[0]):
            raise ValueError(
                f"segmenter returned {alpha.shape} for {path.name}, expected {(size[1], size[0])}"
            )
        coverage = float(np.clip(alpha, 0.0, 1.0).mean())
        # A near-empty mask means the subject was not found; training must not
        # silently anchor the whole frame to the base model.
        if coverage < min_coverage:
            missing.append(path)
            continue
        mask_file = mask_path_for_image(mask_folder, path)
        mask_u8 = np.clip(alpha * 255.0, 0, 255).round().astype(np.uint8)
        Image.fromarray(mask_u8, mode="L").save(mask_file)
        manifest[path.name] = {
            "mask": mask_file.name,
            "source_size": list(size),
            "coverage": round(coverage, 6),
        }
    return manifest, missing


def ensure_subject_masks(
    image_folder: str | Path,
    mask_folder: str | Path | None = None,
    *,
    device: str = "cuda",
    model_id: str = DEFAULT_MODEL_ID,
    input_size: int = DEFAULT_INPUT_SIZE,
    min_coverage: float = 0.01,
    allow_missing: bool = True,
    force: bool = False,
) -> Path:
    """Create subject masks if missing/stale; return the mask folder."""
    image_folder = Path(image_folder)
    mask_folder = Path(mask_folder) if mask_folder is not None else image_folder / "masks"
    if not force and not subject_masks_need_build(
        image_folder, mask_folder, model_id=model_id, input_size=input_size
    ):
        return mask_folder

    print(f"Generating subject masks for {image_folder} -> {mask_folder} ({model_id})")
    segmenter = build_birefnet_segmenter(model_id=model_id, device=device, input_size=input_size)
    manifest, missing = generate_subject_masks(
        image_folder, mask_folder, segmenter, min_coverage=min_coverage
    )
    if missing and not allow_missing:
        names = ", ".join(path.name for path in missing)
        raise RuntimeError(
            f"No usable subject mask for {len(missing)} image(s): {names}. "
            "Fix the images/model settings or set allow_missing."
        )
    if missing:
        print(
            f"WARNING: {len(missing)} image(s) produced no usable subject mask: "
            + ", ".join(path.name for path in missing)
        )
    if not manifest:
        raise RuntimeError(f"No subject masks could be generated for {image_folder}")
    manifest_path = mask_folder / MANIFEST_NAME
    recorded = {BUILD_KEY: {"model_id": model_id, "input_size": input_size}, **manifest}
    manifest_path.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} subject mask(s) to {mask_folder}")
    return mask_folder


def dump_dataset_mask_previews(
    output_dir: str | Path,
    dataset,
    *,
    limit: int = 24,
) -> int:
    """Write previews of the *augmented* training samples and their subject masks.

    These come from the dataset's stored variants rather than the source images, so
    they verify that zoom/flip/crop kept the mask aligned with the pixels.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    total = min(len(dataset.pixel_values), limit)
    for index in range(total):
        pixels = dataset.pixel_values[index]
        mask = dataset.instance_subject_masks[index]
        valid = dataset.instance_subject_mask_valid[index]
        # Training tensors are normalized to [-1, 1]; recover displayable RGB.
        rgb = (pixels.float() * 0.5 + 0.5).clamp(0, 1)
        height, width = rgb.shape[-2:]
        alpha = torch.nn.functional.interpolate(
            mask.float().unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        )[0].clamp(0, 1)

        mask_u8 = (alpha[0] * 255.0).round().to(torch.uint8).cpu().numpy()
        Image.fromarray(mask_u8, mode="L").save(output_dir / f"{index:04d}_mask.png")

        green = torch.tensor([0.15, 1.0, 0.35]).reshape(3, 1, 1)
        blended = (rgb * (1.0 - 0.45 * alpha) + green * (0.45 * alpha)).clamp(0, 1)
        overlay = (blended * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        suffix = "" if valid else "_nomask"
        Image.fromarray(overlay, mode="RGB").save(
            output_dir / f"{index:04d}_overlay{suffix}.jpg", quality=92
        )
        written += 1
    return written


__all__ = [
    "BUILD_KEY",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_MODEL_ID",
    "IMAGE_EXTENSIONS",
    "MANIFEST_NAME",
    "build_birefnet_segmenter",
    "dump_dataset_mask_previews",
    "ensure_subject_masks",
    "generate_subject_masks",
    "load_mask_build_settings",
    "load_mask_manifest",
    "mask_path_for_image",
    "subject_mask_for_image",
    "subject_masks_need_build",
]
