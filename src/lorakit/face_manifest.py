"""Offline face-manifest generation helpers.

This module deliberately has no InsightFace import.  Detection is injected by
the CLI so importing Lorakit's training code never requires ONNX or runs face
detection in the training process.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def image_paths(folder: str | Path) -> list[Path]:
    """Return supported top-level images in deterministic order."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Image folder does not exist or is not a directory: {folder}")
    paths = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No supported images found in {folder}")
    return paths


def _bbox(face: object) -> tuple[float, float, float, float]:
    values = getattr(face, "bbox", None)
    if values is None or len(values) != 4:
        raise ValueError("Face detector returned an invalid bounding box")
    x1, y1, x2, y2 = (float(value) for value in values)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Face detector returned an empty bounding box")
    return x1, y1, x2, y2


def select_face_box(
    faces: Sequence[object], image_size: tuple[int, int], selection: str
) -> list[float] | None:
    """Choose one valid detector box and clamp it to source-image bounds."""
    candidates: list[tuple[tuple[float, float], tuple[float, float, float, float]]] = []
    for face in faces:
        try:
            box = _bbox(face)
        except ValueError:
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        score = float(getattr(face, "det_score", 0.0))
        candidates.append(((area, score), box))
    if not candidates:
        return None
    if selection == "largest":
        _, box = max(candidates, key=lambda item: item[0])
    elif selection == "highest-confidence":
        _, box = max(candidates, key=lambda item: (item[0][1], item[0][0]))
    else:
        raise ValueError(f"Unsupported face selection policy: {selection}")
    width, height = image_size
    x1, y1, x2, y2 = box
    clamped = [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)]
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return None
    return [round(value, 3) for value in clamped]


def generate_face_manifest(
    paths: Sequence[Path],
    detector: Callable[[np.ndarray], Sequence[object]],
    *,
    selection: str,
) -> tuple[dict[str, dict[str, list[float] | list[int]]], list[Path]]:
    """Detect one face per image and return the manifest plus missing images."""
    manifest: dict[str, dict[str, list[float] | list[int]]] = {}
    missing: list[Path] = []
    for path in paths:
        try:
            with Image.open(path) as source_image:
                image = exif_transpose(source_image).convert("RGB")
                rgb = np.asarray(image)
                # Reversing channels creates a negative-stride view, which ONNX
                # Runtime's detector rejects for many otherwise-valid images.
                bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                box = select_face_box(detector(bgr), image.size, selection)
        except OSError:
            box = None
        if box is None:
            missing.append(path)
            continue
        manifest[path.name] = {"bbox_xyxy": box, "source_size": list(image.size)}
    return manifest, missing
