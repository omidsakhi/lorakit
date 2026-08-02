"""Soft face masks and background-preserving LoRA training objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_face_mask(
    boxes: torch.Tensor,
    confidence: torch.Tensor,
    *,
    height: int,
    width: int,
    edge_sharpness: float,
) -> torch.Tensor:
    """Rasterize normalized boxes into a differentiable-looking, detached-use mask."""
    batch = boxes.shape[0]
    y = (torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5) / height
    x = (torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    x1, y1, x2, y2 = (boxes[:, index].reshape(batch, 1, 1) for index in range(4))
    inside = (
        torch.sigmoid(edge_sharpness * (xx - x1))
        * torch.sigmoid(edge_sharpness * (x2 - xx))
        * torch.sigmoid(edge_sharpness * (yy - y1))
        * torch.sigmoid(edge_sharpness * (y2 - yy))
    )
    return (confidence.reshape(batch, 1, 1) * inside).clamp(0, 1).unsqueeze(1)


def dump_manifest_soft_face_masks(
    output_dir: Path | str,
    dataset_folder: Path | str,
    face_manifest: Mapping[str, Mapping],
    *,
    edge_sharpness: float,
    face_focus_power: float = 1.0,
) -> int:
    """Write soft face-mask previews once from the offline ``faces.json`` boxes.

    For each training image with a manifest box, saves:
      - ``{stem}_mask.png`` — grayscale soft mask (after ``mask ** power``)
      - ``{stem}_overlay.jpg`` — training image with the mask as a red overlay

    Returns the number of images written.
    """
    from PIL import Image, ImageDraw
    from PIL.ImageOps import exif_transpose

    from lorakit.identity import face_box_for_image

    if not math.isfinite(edge_sharpness) or edge_sharpness <= 0:
        raise ValueError("edge_sharpness must be finite and positive")
    if not math.isfinite(face_focus_power) or not (0.0 <= face_focus_power <= 1.0):
        raise ValueError("face_focus_power must be finite and in [0, 1]")

    output_dir = Path(output_dir)
    dataset_folder = Path(dataset_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    supported = {".png", ".jpg", ".jpeg", ".webp"}
    written = 0

    for image_path in sorted(dataset_folder.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in supported:
            continue
        with Image.open(image_path) as raw:
            image = exif_transpose(raw).convert("RGB")
        box = face_box_for_image(face_manifest, image_path, image.size)
        if box is None:
            continue
        width, height = image.size
        x1, y1, x2, y2 = box
        norm = torch.tensor(
            [[x1 / width, y1 / height, x2 / width, y2 / height]],
            dtype=torch.float32,
        )
        mask = soft_face_mask(
            norm,
            torch.ones(1, dtype=torch.float32),
            height=height,
            width=width,
            edge_sharpness=edge_sharpness,
        )[0, 0]
        if face_focus_power != 1.0:
            mask = mask.clamp(0, 1).pow(face_focus_power)
        mask_u8 = (mask.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        Image.fromarray(mask_u8, mode="L").save(output_dir / f"{image_path.stem}_mask.png")

        rgb = torch.from_numpy(np.asarray(image, dtype=np.float32))
        red = torch.tensor([255.0, 40.0, 40.0])
        alpha = mask.unsqueeze(-1)
        blended = (rgb * (1.0 - 0.55 * alpha) + red * (0.55 * alpha)).clamp(0, 255)
        overlay = Image.fromarray(blended.round().to(torch.uint8).cpu().numpy(), mode="RGB")
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(
            [int(x1), int(y1), int(x2), int(y2)],
            outline=(255, 255, 0),
            width=max(2, min(width, height) // 256),
        )
        overlay.save(output_dir / f"{image_path.stem}_overlay.jpg", quality=92)
        written += 1

    return written


class LatentBackgroundPreservationObjective(nn.Module):
    """Keep LoRA predictions equal to the base model outside the trained region.

    The anchored region defaults to the complement of the face mask.  Callers
    with a face parse map pass ``background_mask`` instead, because there the
    trained region is graded (skin at reduced weight) and its complement would
    wrongly pull the face itself back toward the base prediction.
    """

    def __init__(self, *, mask_edge_sharpness: float) -> None:
        super().__init__()
        self.mask_edge_sharpness = mask_edge_sharpness

    @staticmethod
    def _graph_zero(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float().sum() * 0.0

    @staticmethod
    def _valid_manifest_rows(face_boxes: torch.Tensor) -> torch.Tensor:
        """Rows whose manifest box is real rather than the dataset's ``-1`` sentinel."""
        return (
            (face_boxes[:, 0] >= 0)
            & (face_boxes[:, 1] >= 0)
            & (face_boxes[:, 2] > face_boxes[:, 0])
            & (face_boxes[:, 3] > face_boxes[:, 1])
        )

    @staticmethod
    def _subject_mask_rows(
        subject_mask: torch.Tensor,
        subject_mask_valid: torch.Tensor | None,
        *,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select rows with a real subject mask and resample it to the latent grid.

        Unlike the face rectangle, a background-removal alpha already describes the
        exact subject silhouette, so it needs neither an edge-sharpness parameter
        nor a timestep gate.
        """
        batch = subject_mask.shape[0]
        if subject_mask_valid is None:
            valid = torch.ones(batch, dtype=torch.bool, device=subject_mask.device)
        else:
            valid = subject_mask_valid.to(device=subject_mask.device, dtype=torch.bool)
        indices = valid.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return indices, subject_mask.new_zeros((0, 1, height, width))
        return indices, LatentBackgroundPreservationObjective._resample_rows(
            subject_mask, indices, height=height, width=width
        )

    @staticmethod
    def _resample_rows(
        mask: torch.Tensor, indices: torch.Tensor, *, height: int, width: int
    ) -> torch.Tensor:
        """Select rows of a per-pixel mask and resample them to the latent grid."""
        selected = mask[indices].float()
        if selected.ndim == 3:
            selected = selected.unsqueeze(1)
        if selected.shape[-2:] != (height, width):
            selected = F.interpolate(
                selected, size=(height, width), mode="bilinear", align_corners=False
            )
        return selected.clamp(0, 1).detach()

    def _manifest_mask(
        self, face_boxes: torch.Tensor, *, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Soft mask for the rows with a usable manifest box.

        The manifest box comes from the training image itself, so it is exact at
        every noise level and needs no timestep gate.
        Returns the selected row indices and their ``[n, 1, height, width]`` mask.
        """
        valid = self._valid_manifest_rows(face_boxes)
        indices = valid.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return indices, face_boxes.new_zeros((0, 1, height, width))
        mask = soft_face_mask(
            face_boxes[indices].float(),
            torch.ones(indices.numel(), device=face_boxes.device, dtype=torch.float32),
            height=height,
            width=width,
            edge_sharpness=self.mask_edge_sharpness,
        )
        return indices, mask.detach()

    def forward(
        self,
        *,
        noisy_instance_latents: torch.Tensor,
        model_prediction: torch.Tensor,
        base_model_prediction: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler,
        num_train_timesteps: int,
        face_boxes: torch.Tensor | None = None,
        subject_mask: torch.Tensor | None = None,
        subject_mask_valid: torch.Tensor | None = None,
        background_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del noisy_instance_latents, timesteps, scheduler, num_train_timesteps
        stats = {
            "eligible_timesteps": 0.0,
            "valid_faces": 0.0,
            "skipped_faces": 0.0,
            "face_coverage": 0.0,
            "background_coverage": 0.0,
            "background_mse": 0.0,
        }
        anchor_mask = None
        if subject_mask is not None:
            # Background-removal alpha: preserve everything outside the subject
            # silhouette instead of everything outside the face rectangle.
            indices, face_mask = self._subject_mask_rows(
                subject_mask,
                subject_mask_valid,
                height=model_prediction.shape[-2],
                width=model_prediction.shape[-1],
            )
            stats["eligible_timesteps"] = float(model_prediction.shape[0])
            stats["valid_faces"] = float(indices.numel())
            stats["skipped_faces"] = float(model_prediction.shape[0] - indices.numel())
            if indices.numel() == 0:
                return self._graph_zero(model_prediction), stats
            if background_mask is not None:
                # A face parse map separates "trained at reduced weight" from
                # "not part of the subject", so the anchor region is supplied
                # explicitly instead of being inferred as 1 - face_mask.
                anchor_mask = self._resample_rows(
                    background_mask,
                    indices,
                    height=model_prediction.shape[-2],
                    width=model_prediction.shape[-1],
                )
        elif face_boxes is not None:
            # Manifest boxes are exact at every timestep, so every row is eligible.
            # Rows without a usable box are skipped rather than having the whole
            # frame anchored to the base model, which would cancel identity learning.
            indices, face_mask = self._manifest_mask(
                face_boxes,
                height=model_prediction.shape[-2],
                width=model_prediction.shape[-1],
            )
            stats["eligible_timesteps"] = float(model_prediction.shape[0])
            stats["valid_faces"] = float(indices.numel())
            stats["skipped_faces"] = float(model_prediction.shape[0] - indices.numel())
            if indices.numel() == 0:
                return self._graph_zero(model_prediction), stats
        else:
            raise RuntimeError("face_boxes or subject_mask is required for background preservation")

        anchor = anchor_mask if anchor_mask is not None else 1.0 - face_mask
        squared_error = (
            model_prediction[indices].float() - base_model_prediction[indices].float()
        ).square()
        background_loss = squared_error.mul(anchor).sum()
        background_loss = background_loss / (anchor.sum() * model_prediction.shape[1]).clamp_min(
            1.0
        )
        stats["face_coverage"] = float(face_mask.mean().item())
        stats["background_coverage"] = float(anchor.mean().item())
        stats["background_mse"] = float(background_loss.detach().item())
        return background_loss, stats

    @torch.no_grad()
    def face_focus_mask(
        self,
        *,
        noisy_instance_latents: torch.Tensor,
        base_model_prediction: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler,
        num_train_timesteps: int,
        height: int,
        width: int,
        face_boxes: torch.Tensor | None = None,
        subject_mask: torch.Tensor | None = None,
        subject_mask_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Soft per-row subject/face mask in ``[0, 1]`` for weighting the identity loss.

        Returns a detached ``[B, 1, height, width]`` mask that is ~1 on the region
        to train and falls off to 0 elsewhere. With ``subject_mask`` the region is
        the background-removal silhouette; with ``face_boxes`` it is the offline
        manifest rectangle. Both come from the training image itself and are exact
        at every noise level. Rows without a usable region fall back to an all-ones
        mask so the identity loss still trains normally (i.e. focusing is skipped
        rather than zeroing the gradient for that row).
        """
        del noisy_instance_latents, timesteps, scheduler, num_train_timesteps
        batch = base_model_prediction.shape[0]
        mask = base_model_prediction.new_ones((batch, 1, height, width))
        if subject_mask is not None:
            indices, subject = self._subject_mask_rows(
                subject_mask, subject_mask_valid, height=height, width=width
            )
            if indices.numel():
                mask[indices] = subject.to(mask.dtype)
            return mask, {
                "eligible_timesteps": float(batch),
                "valid_faces": float(indices.numel()),
                "face_coverage": float(mask.mean().item()),
            }
        if face_boxes is not None:
            indices, face = self._manifest_mask(face_boxes, height=height, width=width)
            if indices.numel():
                mask[indices] = face.to(mask.dtype)
            return mask, {
                "eligible_timesteps": float(batch),
                "valid_faces": float(indices.numel()),
                "face_coverage": float(mask.mean().item()),
            }
        raise RuntimeError("face_boxes or subject_mask is required for face focus masks")
