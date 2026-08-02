"""Sampling-time ArcFace identity scoring (decoded images only; no VAE / gradients).

Compares generated sample faces to the subject reference centroid so training runs
can pick the checkpoint step with the highest mean likeness.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from lorakit.identity import ArcFaceIdentityObjective, _crop_faces_roi_align


def detect_face_box(
    image: Image.Image,
    detector: Callable,
    *,
    selection: str = "largest",
) -> list[float] | None:
    """Run an InsightFace-style detector on a PIL image; return ``bbox_xyxy`` or None."""
    import numpy as np

    from lorakit.face_manifest import select_face_box

    rgb = image.convert("RGB")
    array = np.asarray(rgb)
    bgr = np.ascontiguousarray(array[:, :, ::-1])
    return select_face_box(detector(bgr), rgb.size, selection)


def reference_cohesion(reference_embeddings: torch.Tensor) -> float | None:
    """Mean pairwise cosine among the reference faces.

    Reports how tightly the subject's own photos cluster, which is the practical
    ceiling for any generated sample's ``max_similarity``.
    """
    refs = torch.nn.functional.normalize(reference_embeddings.float(), dim=-1)
    count = refs.shape[0]
    if count < 2:
        return None
    similarity = refs @ refs.t()
    off_diagonal = similarity.sum() - similarity.diagonal().sum()
    return float(off_diagonal / (count * (count - 1)))


@torch.no_grad()
def score_generated_samples(
    *,
    images: Sequence[Image.Image],
    encoder: torch.nn.Module,
    reference_embeddings: torch.Tensor,
    detector: Callable,
    device: torch.device,
    crop_size: int = 112,
    bbox_padding_fraction: float = 0.15,
    selection: str = "largest",
    image_names: Sequence[str] | None = None,
    reference_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score generated images against every individual ArcFace reference face.

    ``reference_embeddings`` is an ``[n_references, dim]`` matrix of per-image
    embeddings. For each sample this records the best-matching reference
    (``max_similarity``), the similarity to the reference centroid
    (``centroid_similarity``), and the mean across references. The step-level
    headline is ``mean_max_similarity``: the max is taken per sample first, then
    averaged over the samples where a face was detected.
    """
    refs = torch.nn.functional.normalize(
        reference_embeddings.float().to(device).reshape(reference_embeddings.shape[0], -1),
        dim=-1,
    )
    centroid = torch.nn.functional.normalize(refs.mean(dim=0, keepdim=True), dim=-1)
    names = list(reference_names) if reference_names is not None else None
    cohesion = reference_cohesion(refs)

    if not images:
        return {
            "mean_max_similarity": None,
            "mean_similarity": None,
            "detection_rate": 0.0,
            "n_samples": 0,
            "n_detected": 0,
            "n_references": int(refs.shape[0]),
            "reference_cohesion": cohesion,
            "samples": [],
        }

    from torchvision.transforms.functional import to_tensor

    encoder = encoder.to(device).eval()

    sample_rows: list[dict[str, Any]] = []
    max_scores: list[float] = []
    centroid_scores: list[float] = []

    for index, image in enumerate(images):
        name = (
            image_names[index]
            if image_names is not None and index < len(image_names)
            else f"sample_{index}"
        )
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        box = detect_face_box(rgb, detector, selection=selection)
        row: dict[str, Any] = {
            "index": index,
            "image": name,
            "detected": box is not None,
            "bbox_xyxy": box,
            "max_similarity": None,
            "best_reference": None,
            "centroid_similarity": None,
            "mean_reference_similarity": None,
        }
        if box is None:
            sample_rows.append(row)
            continue

        tensor = to_tensor(rgb).unsqueeze(0).to(device=device, dtype=torch.float32)
        boxes = torch.tensor([box], device=device, dtype=torch.float32)
        crop = _crop_faces_roi_align(
            tensor,
            boxes,
            crop_size=crop_size,
            bbox_padding_fraction=bbox_padding_fraction,
        )
        embed = ArcFaceIdentityObjective.embed_images(encoder, crop)
        per_reference = (embed @ refs.t()).reshape(-1)
        best = int(torch.argmax(per_reference).item())
        max_similarity = float(per_reference[best].item())
        centroid_similarity = float((embed * centroid).sum(dim=-1).item())
        mean_reference = float(per_reference.mean().item())
        if not math.isfinite(max_similarity):
            sample_rows.append(row)
            continue

        row["max_similarity"] = max_similarity
        row["best_reference"] = names[best] if names and best < len(names) else best
        row["centroid_similarity"] = centroid_similarity
        row["mean_reference_similarity"] = mean_reference
        max_scores.append(max_similarity)
        centroid_scores.append(centroid_similarity)
        sample_rows.append(row)

    n_samples = len(images)
    n_detected = len(max_scores)
    return {
        "mean_max_similarity": (float(sum(max_scores) / n_detected) if n_detected > 0 else None),
        "mean_similarity": (float(sum(centroid_scores) / n_detected) if n_detected > 0 else None),
        "detection_rate": float(n_detected / n_samples) if n_samples else 0.0,
        "n_samples": n_samples,
        "n_detected": n_detected,
        "n_references": int(refs.shape[0]),
        "reference_cohesion": cohesion,
        "samples": sample_rows,
    }


def record_step_scores(
    summary_path: str | Path,
    record: Mapping[str, Any],
    *,
    rank_by: str = "mean_max_similarity",
) -> dict[str, Any]:
    """Upsert one sampling-step record and refresh the best-step leaderboard.

    ``rank_by`` selects which similarity drives ``best_step``; it defaults to the
    mean of each sample's best-matching reference face.
    """
    summary_path = Path(summary_path)
    if summary_path.is_file():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    else:
        data = {}
    if not isinstance(data, dict):
        data = {}

    steps = data.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    step = int(record["step"])
    steps = [entry for entry in steps if not isinstance(entry, dict) or entry.get("step") != step]
    steps.append(dict(record))
    steps.sort(key=lambda entry: int(entry.get("step", -1)))

    eligible = [
        entry
        for entry in steps
        if isinstance(entry, dict)
        and int(entry.get("n_detected", 0) or 0) > 0
        and entry.get(rank_by) is not None
        and math.isfinite(float(entry[rank_by]))
    ]
    if eligible:
        best = max(eligible, key=lambda entry: float(entry[rank_by]))
        data["ranked_by"] = rank_by
        data["best_step"] = int(best["step"])
        data["best_mean_max_similarity"] = (
            float(best["mean_max_similarity"])
            if best.get("mean_max_similarity") is not None
            else None
        )
        data["best_mean_similarity"] = (
            float(best["mean_similarity"]) if best.get("mean_similarity") is not None else None
        )
        data["best_detection_rate"] = float(best.get("detection_rate", 0.0))
    else:
        data["ranked_by"] = rank_by
        data["best_step"] = None
        data["best_mean_max_similarity"] = None
        data["best_mean_similarity"] = None
        data["best_detection_rate"] = None

    data["steps"] = steps
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data
