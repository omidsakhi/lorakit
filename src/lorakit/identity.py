"""Differentiable background-preservation supervision for DreamBooth LoRA training."""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint


def load_face_manifest(path: str | Path) -> dict[str, dict]:
    """Load and validate the offline face-box manifest.

    Entries use image file names as keys and ``bbox_xyxy`` in source-image pixel
    coordinates.  Optional ``source_size``/``image_size`` fields are ``[width,
    height]`` and are checked when the image is opened.
    """
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"face_manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(data, dict) or not data:
        raise ValueError("face_manifest must be a non-empty JSON object")

    manifest: dict[str, dict] = {}
    for name, entry in data.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("face_manifest entries must map image names to objects")
        box = entry.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"face_manifest entry {name!r} needs a four-value bbox_xyxy")
        try:
            box = [float(value) for value in box]
        except (TypeError, ValueError) as error:
            raise ValueError(f"face_manifest bbox_xyxy for {name!r} must be numeric") from error
        if not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"face_manifest bbox_xyxy for {name!r} is invalid")

        source_size = entry.get("source_size", entry.get("image_size"))
        if source_size is not None:
            if not isinstance(source_size, list) or len(source_size) != 2:
                raise ValueError(f"face_manifest source_size for {name!r} must be [width, height]")
            try:
                source_size = [int(value) for value in source_size]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"face_manifest source_size for {name!r} must be integral"
                ) from error
            if source_size[0] <= 0 or source_size[1] <= 0:
                raise ValueError(f"face_manifest source_size for {name!r} is invalid")
        manifest[name] = {"bbox_xyxy": box, "source_size": source_size}
    return manifest


def face_box_for_image(
    manifest: Mapping[str, Mapping], image_path: str | Path, image_size: tuple[int, int]
) -> tuple[float, float, float, float] | None:
    """Return a validated source box for an image, or ``None`` when it has none."""
    image_path = Path(image_path)
    entry = manifest.get(image_path.name)
    if entry is None:
        return None
    expected_size = entry.get("source_size")
    if expected_size is not None and tuple(expected_size) != image_size:
        raise ValueError(
            f"face_manifest source_size for {image_path.name!r} is {expected_size}, "
            f"but the image is {list(image_size)}"
        )
    x1, y1, x2, y2 = entry["bbox_xyxy"]
    width, height = image_size
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"face_manifest bbox_xyxy for {image_path.name!r} lies outside the image")
    return x1, y1, x2, y2


def reconstruct_clean_latent(
    noisy_latents: torch.Tensor,
    model_prediction: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    """Reconstruct ``z_0`` from an epsilon or velocity prediction in float32."""
    if prediction_type not in {"epsilon", "v_prediction"}:
        raise ValueError(
            f"background preservation does not support prediction_type={prediction_type!r}"
        )
    alphas = alphas_cumprod.to(device=noisy_latents.device, dtype=torch.float32)
    alpha = alphas[timesteps.long()].sqrt()
    sigma = (1.0 - alphas[timesteps.long()]).sqrt()
    while alpha.ndim < noisy_latents.ndim:
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    noisy_latents = noisy_latents.float()
    model_prediction = model_prediction.float()
    if prediction_type == "epsilon":
        return (noisy_latents - sigma * model_prediction) / alpha
    return alpha * noisy_latents - sigma * model_prediction


def _graph_zero(tensor: torch.Tensor) -> torch.Tensor:
    """A scalar zero which retains the model-prediction autograd connection."""
    return tensor.float().sum() * 0.0


def _padded_boxes(
    boxes: torch.Tensor, height: int, width: int, bbox_padding_fraction: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply crop padding without allowing an out-of-frame crop."""
    boxes = boxes.float()
    box_width = boxes[:, 2] - boxes[:, 0]
    box_height = boxes[:, 3] - boxes[:, 1]
    padding_x = box_width * bbox_padding_fraction
    padding_y = box_height * bbox_padding_fraction
    return (
        (boxes[:, 0] - padding_x).clamp(0, width - 1),
        (boxes[:, 1] - padding_y).clamp(0, height - 1),
        (boxes[:, 2] + padding_x).clamp(0, width - 1),
        (boxes[:, 3] + padding_y).clamp(0, height - 1),
    )


def _background_mask(
    boxes: torch.Tensor, height: int, width: int, bbox_padding_fraction: float
) -> torch.Tensor:
    """Return an image-space mask that excludes the padded face rectangle."""
    x1, y1, x2, y2 = _padded_boxes(boxes, height, width, bbox_padding_fraction)
    y, x = torch.meshgrid(
        torch.arange(height, device=boxes.device),
        torch.arange(width, device=boxes.device),
        indexing="ij",
    )
    in_face = (
        (x.unsqueeze(0) >= x1[:, None, None])
        & (x.unsqueeze(0) <= x2[:, None, None])
        & (y.unsqueeze(0) >= y1[:, None, None])
        & (y.unsqueeze(0) <= y2[:, None, None])
    )
    return (~in_face).unsqueeze(1).float()


def _decode_clean_prediction(
    *,
    noisy_instance_latents: torch.Tensor,
    model_prediction: torch.Tensor,
    timesteps: torch.Tensor,
    face_boxes: torch.Tensor,
    vae: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    prediction_type: str,
    latent_scaling_factor: float,
    latents_mean: torch.Tensor | None,
    latents_std: torch.Tensor | None,
    max_timestep_fraction: float,
    num_train_timesteps: int,
    decoder_gradient_checkpointing: bool,
) -> tuple[torch.Tensor | None, torch.Tensor, dict[str, float]]:
    """Decode eligible rows and return diagnostics for skipped rows."""
    max_timestep = math.floor(max_timestep_fraction * num_train_timesteps)
    eligible = timesteps <= max_timestep
    valid_box = (
        (face_boxes[:, 0] >= 0)
        & (face_boxes[:, 1] >= 0)
        & (face_boxes[:, 2] > face_boxes[:, 0])
        & (face_boxes[:, 3] > face_boxes[:, 1])
    )
    selected = eligible & valid_box
    skipped = eligible & ~valid_box
    stats = {
        "eligible_timesteps": float(eligible.sum().item()),
        "valid_faces": float(selected.sum().item()),
        "skipped_faces": float(skipped.sum().item()),
        "clipped_boxes": 0.0,
        "background_mse": 0.0,
    }
    if not selected.any():
        return None, _graph_zero(model_prediction), stats

    indices = selected.nonzero(as_tuple=True)[0]
    clean_latents = reconstruct_clean_latent(
        noisy_instance_latents[indices],
        model_prediction[indices],
        timesteps[indices],
        alphas_cumprod,
        prediction_type,
    )
    if latents_mean is None or latents_std is None:
        decoder_latents = clean_latents / latent_scaling_factor
    else:
        latents_mean = latents_mean.to(device=clean_latents.device, dtype=torch.float32)
        latents_std = latents_std.to(device=clean_latents.device, dtype=torch.float32)
        decoder_latents = clean_latents * latents_std / latent_scaling_factor
        decoder_latents = decoder_latents + latents_mean

    with torch.autocast(device_type=decoder_latents.device.type, enabled=False):
        decoder_latents = decoder_latents.float()
        if decoder_gradient_checkpointing:
            decoded = checkpoint(
                lambda latents: vae.decode(latents).sample,
                decoder_latents,
                use_reentrant=False,
            )
        else:
            decoded = vae.decode(decoder_latents).sample
    return decoded, indices, stats


class BackgroundPreservationObjective(torch.nn.Module):
    """Match the base model's low-noise background outside manifest face boxes."""

    def __init__(
        self,
        *,
        bbox_padding_fraction: float,
        max_timestep_fraction: float,
        decoder_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.bbox_padding_fraction = bbox_padding_fraction
        self.max_timestep_fraction = max_timestep_fraction
        self.decoder_gradient_checkpointing = decoder_gradient_checkpointing

    def forward(
        self,
        *,
        noisy_instance_latents: torch.Tensor,
        model_prediction: torch.Tensor,
        base_model_prediction: torch.Tensor,
        timesteps: torch.Tensor,
        face_boxes: torch.Tensor,
        face_box_clipped: torch.Tensor,
        vae: torch.nn.Module,
        alphas_cumprod: torch.Tensor,
        prediction_type: str,
        latent_scaling_factor: float,
        latents_mean: torch.Tensor | None,
        latents_std: torch.Tensor | None,
        num_train_timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the background loss and per-step diagnostics."""
        decoded, indices, stats = _decode_clean_prediction(
            noisy_instance_latents=noisy_instance_latents,
            model_prediction=model_prediction,
            timesteps=timesteps,
            face_boxes=face_boxes,
            vae=vae,
            alphas_cumprod=alphas_cumprod,
            prediction_type=prediction_type,
            latent_scaling_factor=latent_scaling_factor,
            latents_mean=latents_mean,
            latents_std=latents_std,
            max_timestep_fraction=self.max_timestep_fraction,
            num_train_timesteps=num_train_timesteps,
            decoder_gradient_checkpointing=self.decoder_gradient_checkpointing,
        )
        if decoded is None:
            return _graph_zero(model_prediction), stats

        with torch.no_grad():
            base_decoded, base_indices, _ = _decode_clean_prediction(
                noisy_instance_latents=noisy_instance_latents,
                model_prediction=base_model_prediction,
                timesteps=timesteps,
                face_boxes=face_boxes,
                vae=vae,
                alphas_cumprod=alphas_cumprod,
                prediction_type=prediction_type,
                latent_scaling_factor=latent_scaling_factor,
                latents_mean=latents_mean,
                latents_std=latents_std,
                max_timestep_fraction=self.max_timestep_fraction,
                num_train_timesteps=num_train_timesteps,
                decoder_gradient_checkpointing=False,
            )
        if base_decoded is None or not torch.equal(indices, base_indices):
            raise RuntimeError("base-model background teacher selected different training rows")

        stats["clipped_boxes"] = float(
            (
                (timesteps <= math.floor(self.max_timestep_fraction * num_train_timesteps))
                & face_box_clipped
            )
            .sum()
            .item()
        )
        background_mask = _background_mask(
            face_boxes[indices],
            decoded.shape[-2],
            decoded.shape[-1],
            self.bbox_padding_fraction,
        )
        if not background_mask.any():
            return _graph_zero(model_prediction), stats
        squared_error = (decoded.float() - base_decoded.float()).square() * background_mask
        background_loss = squared_error.sum() / (background_mask.sum() * decoded.shape[1])
        stats["background_mse"] = float(background_loss.detach().item())
        return background_loss, stats


def _crop_faces_roi_align(
    images: torch.Tensor,
    boxes: torch.Tensor,
    *,
    crop_size: int,
    bbox_padding_fraction: float,
) -> torch.Tensor:
    """Differentiable padded face crops resized to ``crop_size``."""
    from torchvision.ops import roi_align

    height, width = images.shape[-2:]
    x1, y1, x2, y2 = _padded_boxes(boxes, height, width, bbox_padding_fraction)
    # Reject degenerate padded boxes (can happen if a face is a single pixel).
    valid = (x2 > x1) & (y2 > y1)
    if not bool(valid.all()):
        raise RuntimeError("face crop produced a degenerate padded box")
    batch_index = torch.arange(boxes.shape[0], device=boxes.device, dtype=boxes.dtype)
    rois = torch.stack((batch_index, x1, y1, x2, y2), dim=-1)
    return roi_align(
        images.float(),
        rois,
        output_size=(crop_size, crop_size),
        spatial_scale=1.0,
        aligned=True,
    )


class ArcFaceIdentityObjective(torch.nn.Module):
    """Match decoded low-noise predictions to a frozen ArcFace reference embedding."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        reference_embedding: torch.Tensor,
        *,
        crop_size: int,
        bbox_padding_fraction: float,
        max_timestep_fraction: float,
        decoder_gradient_checkpointing: bool,
        background_loss_weight: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("reference_embedding", reference_embedding.float().reshape(1, -1))
        self.crop_size = crop_size
        self.bbox_padding_fraction = bbox_padding_fraction
        self.max_timestep_fraction = max_timestep_fraction
        self.decoder_gradient_checkpointing = decoder_gradient_checkpointing
        self.background_loss_weight = background_loss_weight

    @staticmethod
    def embed_images(encoder: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
        from lorakit.arcface import normalize_arcface_rgb

        # ArcFace BatchNorm1d needs batch>1 in train mode; keep encoder in eval.
        was_training = encoder.training
        encoder.eval()
        try:
            embeds = encoder(normalize_arcface_rgb(images))
        finally:
            encoder.train(was_training)
        return torch.nn.functional.normalize(embeds.float(), dim=-1)

    @classmethod
    def build_reference_embeddings(
        cls,
        encoder: torch.nn.Module,
        *,
        image_paths: list[Path],
        face_manifest: Mapping[str, Mapping],
        crop_size: int,
        bbox_padding_fraction: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, list[Path]]:
        """Per-image normalized reference face embeddings and their source paths.

        Images without a usable manifest box are skipped, so the returned paths
        align row-for-row with the embedding matrix.
        """
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        embeds: list[torch.Tensor] = []
        used_paths: list[Path] = []
        encoder = encoder.to(device)
        for image_path in image_paths:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                box = face_box_for_image(face_manifest, image_path, image.size)
                if box is None:
                    continue
                tensor = to_tensor(image).unsqueeze(0).to(device=device, dtype=torch.float32)
                boxes = torch.tensor([box], device=device, dtype=torch.float32)
                crop = _crop_faces_roi_align(
                    tensor,
                    boxes,
                    crop_size=crop_size,
                    bbox_padding_fraction=bbox_padding_fraction,
                )
                with torch.no_grad():
                    embeds.append(cls.embed_images(encoder, crop))
                used_paths.append(image_path)
        if not embeds:
            raise ValueError("identity_loss needs at least one valid reference face embedding")
        return torch.cat(embeds, dim=0), used_paths

    @classmethod
    def build_reference_embedding(
        cls,
        encoder: torch.nn.Module,
        *,
        image_paths: list[Path],
        face_manifest: Mapping[str, Mapping],
        crop_size: int,
        bbox_padding_fraction: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalized centroid of valid reference face embeddings."""
        stacked, _ = cls.build_reference_embeddings(
            encoder,
            image_paths=image_paths,
            face_manifest=face_manifest,
            crop_size=crop_size,
            bbox_padding_fraction=bbox_padding_fraction,
            device=device,
        )
        return torch.nn.functional.normalize(stacked.mean(dim=0), dim=0)

    def forward(
        self,
        *,
        noisy_instance_latents: torch.Tensor,
        model_prediction: torch.Tensor,
        timesteps: torch.Tensor,
        face_boxes: torch.Tensor,
        face_box_clipped: torch.Tensor,
        vae: torch.nn.Module,
        alphas_cumprod: torch.Tensor,
        prediction_type: str,
        latent_scaling_factor: float,
        latents_mean: torch.Tensor | None,
        latents_std: torch.Tensor | None,
        num_train_timesteps: int,
        pixel_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return identity loss (plus optional background MSE) and diagnostics."""
        stats = {
            "eligible_timesteps": 0.0,
            "valid_faces": 0.0,
            "skipped_faces": 0.0,
            "clipped_boxes": 0.0,
            "identity_cosine": 0.0,
            "identity_loss": 0.0,
            "background_mse": 0.0,
        }
        max_timestep = math.floor(self.max_timestep_fraction * num_train_timesteps)
        eligible = timesteps <= max_timestep
        clipped = face_box_clipped.bool()
        usable_boxes = face_boxes.clone()
        usable_boxes[clipped] = -1.0
        valid_box = (
            (usable_boxes[:, 0] >= 0)
            & (usable_boxes[:, 1] >= 0)
            & (usable_boxes[:, 2] > usable_boxes[:, 0])
            & (usable_boxes[:, 3] > usable_boxes[:, 1])
        )
        selected = eligible & valid_box
        stats["eligible_timesteps"] = float(eligible.sum().item())
        stats["valid_faces"] = float(selected.sum().item())
        stats["skipped_faces"] = float((eligible & ~valid_box).sum().item())
        stats["clipped_boxes"] = float((eligible & clipped).sum().item())
        if not selected.any():
            return _graph_zero(model_prediction), stats

        decoded, indices, _ = _decode_clean_prediction(
            noisy_instance_latents=noisy_instance_latents,
            model_prediction=model_prediction,
            timesteps=timesteps,
            face_boxes=usable_boxes,
            vae=vae,
            alphas_cumprod=alphas_cumprod,
            prediction_type=prediction_type,
            latent_scaling_factor=latent_scaling_factor,
            latents_mean=latents_mean,
            latents_std=latents_std,
            max_timestep_fraction=self.max_timestep_fraction,
            num_train_timesteps=num_train_timesteps,
            decoder_gradient_checkpointing=self.decoder_gradient_checkpointing,
        )
        if decoded is None:
            return _graph_zero(model_prediction), stats

        crops = _crop_faces_roi_align(
            decoded,
            usable_boxes[indices],
            crop_size=self.crop_size,
            bbox_padding_fraction=self.bbox_padding_fraction,
        )
        embeds = self.embed_images(self.encoder, crops)
        cosine = (embeds * self.reference_embedding).sum(dim=-1)
        identity_loss = (1.0 - cosine).mean()
        stats["identity_cosine"] = float(cosine.detach().mean().item())
        stats["identity_loss"] = float(identity_loss.detach().item())

        total = identity_loss
        if self.background_loss_weight > 0 and pixel_values is not None:
            refs = pixel_values[indices].float()
            if refs.shape[-2:] != decoded.shape[-2:]:
                refs = torch.nn.functional.interpolate(
                    refs, size=decoded.shape[-2:], mode="bilinear", align_corners=False
                )
            background_mask = _background_mask(
                usable_boxes[indices],
                decoded.shape[-2],
                decoded.shape[-1],
                self.bbox_padding_fraction,
            )
            if background_mask.any():
                background_mse = ((decoded.float() - refs).square() * background_mask).sum() / (
                    background_mask.sum() * decoded.shape[1]
                )
                stats["background_mse"] = float(background_mse.detach().item())
                total = total + self.background_loss_weight * background_mse
        return total, stats
