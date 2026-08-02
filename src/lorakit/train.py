import contextlib
import gc
import json
import logging
import math
import os
import shutil
import time
from functools import partial
from pathlib import Path

import diffusers
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_snr,
)
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import hf_hub_download
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from lorakit.arcface import load_arcface_encoder
from lorakit.callbacks import BaseCallback, LossCallback, ProgressCallback
from lorakit.datasets import DreamBoothDataset, collate_fn
from lorakit.ema import LoRAEMA
from lorakit.face_parsing import CLASS_INDEX, resolve_class_weights
from lorakit.identity import ArcFaceIdentityObjective
from lorakit.jobs import BaseJob
from lorakit.latent_face_detector import (
    LatentBackgroundPreservationObjective,
    dump_manifest_soft_face_masks,
)
from lorakit.models import get_model_type, make_scheduler, sample_defaults
from lorakit.profiling import StepProfiler
from lorakit.sample_arcface import (
    record_step_scores,
    reference_cohesion,
    score_generated_samples,
)


def _flush():
    torch.cuda.empty_cache()
    gc.collect()


def _gradient_norm(parameters):
    """Return the global L2 gradient norm without modifying gradients."""
    gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return None
    return torch.linalg.vector_norm(torch.stack([gradient.norm(2) for gradient in gradients]), 2)


@contextlib.contextmanager
def _adapters_disabled(*models):
    """Temporarily run Diffusers/PEFT models without their LoRA adapters."""
    disabled = []
    try:
        for model in models:
            model.disable_adapters()
            disabled.append(model)
        yield
    finally:
        for model in reversed(disabled):
            model.enable_adapters()


SCHEDULEFREE_OPTIMIZERS = frozenset({"adamwschedulefree", "adamwanchoredschedulefree"})


# copied from train_xl.py
def _tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# copied from train_xl.py
def _encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = _tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def _get_sigmas(accelerator, noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)

    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _masked_per_row_mean(squared_error, weight):
    """Per-row mean of ``squared_error`` weighted by a broadcastable spatial ``weight``.

    ``weight`` is a detached ``[B, 1, H, W]`` mask; the normalization by its own sum
    keeps the loss scale comparable regardless of how much of the frame is weighted.
    """
    flat_num = (squared_error * weight).reshape(squared_error.shape[0], -1).sum(dim=1)
    flat_den = (
        weight.expand_as(squared_error).reshape(squared_error.shape[0], -1).sum(dim=1)
    ).clamp_min(1.0)
    return flat_num / flat_den


def _compute_time_ids(original_size, crops_coords_top_left, resolution, device, dtype):
    # Computes additional embeddings/ids required by the SDXL UNet.
    # regular text embeddings (when `train_text_encoder` is not True)
    # pooled text embeddings
    # time ids
    # This function computes time IDs for Stable Diffusion XL (SDXL) training
    # It's adapted from the _get_add_time_ids method in StableDiffusionXLPipeline

    # Set the target size to a fixed resolution (presumably defined elsewhere)
    target_size = [resolution, resolution]

    # Combine original size, crop coordinates, and target size into a single list.
    # `original_size` and `crops_coords_top_left` arrive as tuples from the dataset, so we
    # normalize everything to lists before concatenating to avoid tuple/list type errors.
    add_time_ids = list(original_size) + list(crops_coords_top_left) + list(target_size)

    # Convert the list to a PyTorch tensor and add a batch dimension

    add_time_ids = torch.tensor([add_time_ids])

    # Move the tensor to the appropriate device and cast to the correct dtype
    add_time_ids = add_time_ids.to(device=device, dtype=dtype)

    return add_time_ids


def _save_model_hook(
    accelerator, unet_type, text_encoder_1_type, text_encoder_2_type, models, weights, output_folder
):
    if accelerator.is_main_process:
        unet_lora_layers_to_save = None
        text_encoder_one_lora_layers_to_save = None
        text_encoder_two_lora_layers_to_save = None
        for model in models:
            # `torch.compile` wraps the model in an `OptimizedModule`; unwrap it so
            # the isinstance checks (and peft state-dict extraction) see the real model.
            unwrapped = model._orig_mod if is_compiled_module(model) else model
            if isinstance(unwrapped, unet_type):
                unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped)
                )
            elif isinstance(unwrapped, text_encoder_1_type):
                text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped)
                )
            elif isinstance(unwrapped, text_encoder_2_type):
                text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped)
                )
            else:
                raise ValueError(f"Unexpected model type: {unwrapped.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        StableDiffusionXLPipeline.save_lora_weights(
            output_folder,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
        )


class TrainJob(BaseJob):
    @staticmethod
    def _parse_rank_alpha(value):
        """Parse a single ``(rank, lora_alpha)`` target value.

        Accepts a native YAML sequence (``[4, 4]``), a string tuple (``"(16, 32)"``),
        or a single int (rank, with ``lora_alpha`` defaulting to the same value).
        """
        if isinstance(value, (list, tuple)):
            pair = list(value)
        elif isinstance(value, str):
            stripped = value.strip().lstrip("(").rstrip(")")
            pair = [part for part in stripped.split(",") if part.strip()]
        elif isinstance(value, int):
            pair = [value, value]
        else:
            raise ValueError(f"target_modules values must be (rank, lora_alpha), got {value!r}")
        if len(pair) != 2:
            raise ValueError(f"target_modules values must be (rank, lora_alpha), got {value!r}")
        return int(pair[0]), int(pair[1])

    @staticmethod
    def _load_alpha_pattern_file(path):
        """Load a block-qualified ``alpha_pattern`` map (e.g. from perturbation-probe).

        The file is JSON with an ``"alpha_pattern"`` object mapping PEFT regex keys
        (matched via ``re.match(r"(.*\\.)?(KEY)$", layer_name)``) to ``lora_alpha``
        values. Used to scale each LoRA layer's strength by its measured identity
        impact so fine-tuning concentrates capacity in high-impact regions.
        """
        import json
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"alpha_pattern_file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        alpha_pattern = data.get("alpha_pattern", data)
        if not isinstance(alpha_pattern, dict) or not alpha_pattern:
            raise ValueError(f"alpha_pattern_file {p} has no non-empty 'alpha_pattern' mapping")
        return {str(k): float(v) for k, v in alpha_pattern.items()}

    @staticmethod
    def _parse_lora_targets(target_modules):
        """Normalize a LoRA ``target_modules`` mapping into PEFT-ready fields.

        ``target_modules`` is a mapping of ``module_name -> (rank, lora_alpha)``.
        Returns ``(modules, rank_pattern, alpha_pattern, default_rank, default_alpha)``,
        where the defaults (PEFT's ``r``/``lora_alpha`` fallback) are the max of the
        per-module values.
        """
        if not isinstance(target_modules, dict) or not target_modules:
            raise ValueError(
                "target_modules must be a non-empty mapping of module -> (rank, lora_alpha)"
            )
        modules = list(target_modules.keys())
        rank_pattern = {}
        alpha_pattern = {}
        for name, value in target_modules.items():
            rank, alpha = TrainJob._parse_rank_alpha(value)
            rank_pattern[name] = rank
            alpha_pattern[name] = alpha
        return (
            modules,
            rank_pattern,
            alpha_pattern,
            max(rank_pattern.values()),
            max(alpha_pattern.values()),
        )

    def _validate_face_detector_config(self, value, *, required=False):
        """Validate shared soft-mask settings used by face-focus / BG objectives.

        The block is still named ``face_detector`` for config compatibility; it
        carries ``mask_edge_sharpness`` for rasterizing manifest boxes and
        ``trust_radius`` for how far from the face a mask source is believed.
        """
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("face_detector must be a mapping")
        defaults = {
            "mask_edge_sharpness": value.get("mask_edge_sharpness", 80.0),
            "trust_radius": value.get("trust_radius", 0.0) or 0.0,
        }
        if (
            not isinstance(defaults["trust_radius"], (int, float))
            or isinstance(defaults["trust_radius"], bool)
            or not math.isfinite(defaults["trust_radius"])
            or defaults["trust_radius"] < 0
        ):
            raise ValueError("face_detector.trust_radius must be a finite number >= 0")
        defaults["trust_radius"] = float(defaults["trust_radius"])
        if not required:
            return defaults
        if self._do_edm_style_training:
            raise ValueError("face-mask objectives do not support do_edm_style_training")
        if (
            not isinstance(defaults["mask_edge_sharpness"], (int, float))
            or not math.isfinite(defaults["mask_edge_sharpness"])
            or defaults["mask_edge_sharpness"] <= 0
        ):
            raise ValueError("face_detector.mask_edge_sharpness must be finite and positive")
        return defaults

    def _validate_background_preservation_config(self, value):
        """Validate the background-preservation loss term (match base outside the face)."""
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("background_preservation must be a mapping")
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("background_preservation.enabled must be a boolean")
        defaults = {
            "enabled": enabled,
            "weight": value.get("weight", 0.0),
        }
        if not enabled:
            return defaults
        if (
            not isinstance(defaults["weight"], (int, float))
            or not math.isfinite(defaults["weight"])
            or defaults["weight"] <= 0
        ):
            raise ValueError("background_preservation.weight must be finite and positive")
        return defaults

    def _validate_identity_loss_config(self, value):
        """Validate the optional ArcFace subject-identity loss."""
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("identity_loss must be a mapping")
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("identity_loss.enabled must be a boolean")
        defaults = {
            "enabled": enabled,
            "backend": value.get("backend", "arcface_torch"),
            "model_id": value.get("model_id", "models/ms1mv3_arcface_r50_fp16/backbone.pth"),
            "face_manifest": value.get("face_manifest", None),
            "reference_images_folder": value.get("reference_images_folder", None),
            "weight": value.get("weight", 0.05),
            "background_loss_weight": value.get("background_loss_weight", 0.0),
            "max_timestep_fraction": value.get("max_timestep_fraction", 0.20),
            "crop_size": value.get("crop_size", 112),
            "bbox_padding_fraction": value.get("bbox_padding_fraction", 0.15),
            "decoder_gradient_checkpointing": value.get("decoder_gradient_checkpointing", True),
        }
        if not enabled:
            return defaults
        if self._do_edm_style_training:
            raise ValueError("identity_loss does not support do_edm_style_training yet")
        if defaults["backend"] != "arcface_torch":
            raise ValueError("identity_loss.backend must be 'arcface_torch'")
        if (
            not isinstance(defaults["weight"], (int, float))
            or not math.isfinite(defaults["weight"])
            or defaults["weight"] < 0
        ):
            raise ValueError("identity_loss.weight must be finite and non-negative")
        if not isinstance(defaults["background_loss_weight"], (int, float)) or not (
            math.isfinite(defaults["background_loss_weight"])
            and defaults["background_loss_weight"] >= 0
        ):
            raise ValueError("identity_loss.background_loss_weight must be finite and non-negative")
        if not isinstance(defaults["max_timestep_fraction"], (int, float)) or not (
            0 < defaults["max_timestep_fraction"] <= 1
        ):
            raise ValueError("identity_loss.max_timestep_fraction must be in (0, 1]")
        if not isinstance(defaults["crop_size"], int) or defaults["crop_size"] <= 0:
            raise ValueError("identity_loss.crop_size must be a positive integer")
        if not isinstance(defaults["bbox_padding_fraction"], (int, float)) or not (
            math.isfinite(defaults["bbox_padding_fraction"])
            and defaults["bbox_padding_fraction"] >= 0
        ):
            raise ValueError("identity_loss.bbox_padding_fraction must be finite and non-negative")
        if not isinstance(defaults["decoder_gradient_checkpointing"], bool):
            raise ValueError("identity_loss.decoder_gradient_checkpointing must be a boolean")

        from lorakit.config import resolve_user_path

        defaults["model_id"] = str(
            resolve_user_path(defaults["model_id"], config_path=self._config_path, must_exist=True)
        )
        if defaults["face_manifest"] is not None:
            defaults["face_manifest"] = str(
                resolve_user_path(
                    defaults["face_manifest"],
                    config_path=self._config_path,
                    must_exist=True,
                )
            )
        if defaults["reference_images_folder"] is not None:
            defaults["reference_images_folder"] = str(
                resolve_user_path(
                    defaults["reference_images_folder"],
                    config_path=self._config_path,
                    must_exist=True,
                )
            )
        return defaults

    def _validate_sample_arcface_config(self, value):
        """Validate optional sampling-time ArcFace likeness scoring."""
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("sample.arcface_metric must be a mapping")
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("sample.arcface_metric.enabled must be a boolean")
        defaults = {
            "enabled": enabled,
            "model_id": value.get("model_id", "models/ms1mv3_arcface_r50_fp16/backbone.pth"),
            "face_manifest": value.get("face_manifest", None),
            "reference_images_folder": value.get("reference_images_folder", None),
            "crop_size": value.get("crop_size", 112),
            "bbox_padding_fraction": value.get("bbox_padding_fraction", 0.15),
            "selection": value.get("selection", "largest"),
            "det_size": value.get("det_size", 640),
            "det_threshold": value.get("det_threshold", 0.5),
            "insightface_model": value.get("insightface_model", "buffalo_l"),
        }
        if not enabled:
            return defaults
        if not isinstance(defaults["crop_size"], int) or defaults["crop_size"] <= 0:
            raise ValueError("sample.arcface_metric.crop_size must be a positive integer")
        if not isinstance(defaults["bbox_padding_fraction"], (int, float)) or not (
            math.isfinite(defaults["bbox_padding_fraction"])
            and defaults["bbox_padding_fraction"] >= 0
        ):
            raise ValueError(
                "sample.arcface_metric.bbox_padding_fraction must be finite and non-negative"
            )
        if defaults["selection"] not in {"largest", "highest-confidence"}:
            raise ValueError(
                "sample.arcface_metric.selection must be 'largest' or 'highest-confidence'"
            )
        if not isinstance(defaults["det_size"], int) or defaults["det_size"] <= 0:
            raise ValueError("sample.arcface_metric.det_size must be a positive integer")
        if not isinstance(defaults["det_threshold"], (int, float)) or not (
            0.0 <= float(defaults["det_threshold"]) <= 1.0
        ):
            raise ValueError("sample.arcface_metric.det_threshold must be in [0, 1]")

        from lorakit.config import resolve_user_path

        defaults["model_id"] = str(
            resolve_user_path(defaults["model_id"], config_path=self._config_path, must_exist=True)
        )
        if defaults["face_manifest"] is not None:
            defaults["face_manifest"] = str(
                resolve_user_path(
                    defaults["face_manifest"],
                    config_path=self._config_path,
                    must_exist=True,
                )
            )
        if defaults["reference_images_folder"] is not None:
            defaults["reference_images_folder"] = str(
                resolve_user_path(
                    defaults["reference_images_folder"],
                    config_path=self._config_path,
                    must_exist=True,
                )
            )
        return defaults

    def _validate_subject_mask_config(self, value):
        """Validate offline background-removal (subject alpha) mask settings."""
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("subject_mask must be a mapping")
        from lorakit.subject_masks import DEFAULT_INPUT_SIZE, DEFAULT_MODEL_ID

        defaults = {
            "mask_folder": value.get("mask_folder", None),
            "auto_build": value.get("auto_build", True),
            "model_id": value.get("model_id", DEFAULT_MODEL_ID),
            "input_size": value.get("input_size", DEFAULT_INPUT_SIZE),
            "min_coverage": value.get("min_coverage", 0.01),
            "device": value.get("device", "cuda"),
        }
        if not isinstance(defaults["auto_build"], bool):
            raise ValueError("subject_mask.auto_build must be a boolean")
        if not isinstance(defaults["input_size"], int) or defaults["input_size"] <= 0:
            raise ValueError("subject_mask.input_size must be a positive integer")
        if not isinstance(defaults["min_coverage"], (int, float)) or not (
            0.0 <= float(defaults["min_coverage"]) < 1.0
        ):
            raise ValueError("subject_mask.min_coverage must be in [0, 1)")
        if defaults["device"] not in {"cuda", "cpu"}:
            raise ValueError("subject_mask.device must be 'cuda' or 'cpu'")
        if defaults["mask_folder"] is not None:
            from lorakit.config import resolve_user_path

            defaults["mask_folder"] = str(
                resolve_user_path(
                    defaults["mask_folder"], config_path=self._config_path, must_exist=False
                )
            )
        return defaults

    def _validate_face_parse_config(self, value):
        """Validate offline face-parsing (19-class segmentation) settings."""
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("face_parse must be a mapping")
        from lorakit.face_parsing import (
            DEFAULT_BACKBONE,
            DEFAULT_INPUT_SIZE,
            DEFAULT_MODEL_PATH,
            resolve_class_weights,
        )

        defaults = {
            "parse_folder": value.get("parse_folder", None),
            "auto_build": value.get("auto_build", True),
            "model_path": value.get("model_path", DEFAULT_MODEL_PATH),
            "backbone": value.get("backbone", DEFAULT_BACKBONE),
            "input_size": value.get("input_size", DEFAULT_INPUT_SIZE),
            "min_coverage": value.get("min_coverage", 0.01),
            "device": value.get("device", "cuda"),
            "class_weights": value.get("class_weights", {}) or {},
            "background_source": value.get("background_source", "subject_mask"),
        }
        if not isinstance(defaults["auto_build"], bool):
            raise ValueError("face_parse.auto_build must be a boolean")
        if defaults["background_source"] not in {"parse", "subject_mask"}:
            raise ValueError("face_parse.background_source must be 'parse' or 'subject_mask'")
        if defaults["backbone"] not in {"resnet18", "resnet34"}:
            raise ValueError("face_parse.backbone must be 'resnet18' or 'resnet34'")
        if not isinstance(defaults["input_size"], int) or defaults["input_size"] <= 0:
            raise ValueError("face_parse.input_size must be a positive integer")
        if not isinstance(defaults["min_coverage"], (int, float)) or not (
            0.0 <= float(defaults["min_coverage"]) < 1.0
        ):
            raise ValueError("face_parse.min_coverage must be in [0, 1)")
        if defaults["device"] not in {"cuda", "cpu"}:
            raise ValueError("face_parse.device must be 'cuda' or 'cpu'")
        if not isinstance(defaults["class_weights"], dict):
            raise ValueError("face_parse.class_weights must be a mapping")
        # Surface a bad class name or weight now rather than mid-dataset build.
        resolve_class_weights(defaults["class_weights"])

        from lorakit.config import resolve_user_path

        # The checkpoint is only read when the parse maps still have to be built;
        # a dataset parsed ahead of time with `lorakit-parse` does not need it.
        needs_checkpoint = (
            self._train_config.get("face_mask_source") == "face_parse" and defaults["auto_build"]
        )
        defaults["model_path"] = str(
            resolve_user_path(
                defaults["model_path"],
                config_path=self._config_path,
                must_exist=needs_checkpoint,
            )
        )
        if defaults["parse_folder"] is not None:
            defaults["parse_folder"] = str(
                resolve_user_path(
                    defaults["parse_folder"], config_path=self._config_path, must_exist=False
                )
            )
        return defaults

    def _resolve_face_parses(self, config):
        """Build (if needed) and load the offline face-parse manifest."""
        from lorakit.face_parsing import (
            MANIFEST_NAME,
            ensure_face_parses,
            face_parses_need_build,
            load_parse_manifest,
        )

        parse_folder = (
            Path(config["parse_folder"])
            if config["parse_folder"] is not None
            else Path(self._dataset_folder) / "face_parse"
        )
        if config["auto_build"] and face_parses_need_build(self._dataset_folder, parse_folder):
            ensure_face_parses(
                self._dataset_folder,
                parse_folder,
                device=config["device"],
                model_path=config["model_path"],
                backbone=config["backbone"],
                input_size=config["input_size"],
                min_coverage=config["min_coverage"],
                allow_missing=True,
            )
        manifest_path = parse_folder / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"face parse manifest not found: {manifest_path}. "
                "Run `lorakit-parse <dataset_folder>` or enable face_parse.auto_build."
            )
        return parse_folder, load_parse_manifest(manifest_path)

    def _validate_augmentation_config(self, value):
        """Validate pre-computed geometric augmentation variants."""
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("augmentation must be a mapping")
        from lorakit.datasets import PAD_MODES

        defaults = {
            "variants": value.get("variants", 1),
            "scale_min": value.get("scale_min", 1.0),
            "scale_max": value.get("scale_max", 1.0),
            "rotation_degrees": value.get("rotation_degrees", 0.0),
            "random_flip": value.get("random_flip", False),
            "pad_mode": value.get("pad_mode", "reflect"),
            "seed": value.get("seed", self._train_config.get("seed", 42)),
        }
        if not isinstance(defaults["variants"], int) or defaults["variants"] < 1:
            raise ValueError("augmentation.variants must be an integer >= 1")
        for key in ("scale_min", "scale_max", "rotation_degrees"):
            if not isinstance(defaults[key], (int, float)) or not math.isfinite(defaults[key]):
                raise ValueError(f"augmentation.{key} must be a finite number")
            defaults[key] = float(defaults[key])
        # Below 0.25 the subject is a thumbnail surrounded by invented canvas and
        # the VAE context stops resembling a photograph.
        if not 0.25 <= defaults["scale_min"] <= defaults["scale_max"]:
            raise ValueError("augmentation requires 0.25 <= scale_min <= scale_max")
        if defaults["rotation_degrees"] < 0.0:
            raise ValueError("augmentation.rotation_degrees must be >= 0")
        needs_pixel_mask = defaults["scale_min"] < 1.0 or defaults["rotation_degrees"] > 0.0
        if needs_pixel_mask and not self._use_pixel_masks:
            # Zoom-out and rotation invent canvas.  Without a per-pixel mask
            # those pixels get full diffusion loss and the model learns to
            # reproduce the padding.
            raise ValueError(
                "augmentation.scale_min < 1.0 or rotation_degrees > 0 requires "
                "face_mask_source: face_parse (or subject_mask) so invented "
                "canvas can be given zero weight"
            )
        if not isinstance(defaults["random_flip"], bool):
            raise ValueError("augmentation.random_flip must be a boolean")
        if defaults["pad_mode"] not in PAD_MODES:
            raise ValueError(f"augmentation.pad_mode must be one of {PAD_MODES}")
        if not isinstance(defaults["seed"], int):
            raise ValueError("augmentation.seed must be an integer")
        return defaults

    def _resolve_subject_masks(self, config):
        """Build (if needed) and load the offline subject-mask manifest."""
        from lorakit.subject_masks import (
            MANIFEST_NAME,
            ensure_subject_masks,
            load_mask_manifest,
            subject_masks_need_build,
        )

        mask_folder = (
            Path(config["mask_folder"])
            if config["mask_folder"] is not None
            else Path(self._dataset_folder) / "masks"
        )
        if config["auto_build"] and subject_masks_need_build(
            self._dataset_folder,
            mask_folder,
            model_id=config["model_id"],
            input_size=config["input_size"],
        ):
            ensure_subject_masks(
                self._dataset_folder,
                mask_folder,
                device=config["device"],
                model_id=config["model_id"],
                input_size=config["input_size"],
                min_coverage=config["min_coverage"],
                allow_missing=True,
            )
        manifest_path = mask_folder / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"subject mask manifest not found: {manifest_path}. "
                "Run `lorakit-masks <dataset_folder>` or enable subject_mask.auto_build."
            )
        return mask_folder, load_mask_manifest(manifest_path)

    def _validate_face_focus_config(self, value):
        """Validate the face-focused identity-loss settings."""
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("face_focus must be a mapping")
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("face_focus.enabled must be a boolean")
        power = value.get("power", 1.0)
        if (
            not isinstance(power, (int, float))
            or not math.isfinite(power)
            or not (0.0 <= power <= 1.0)
        ):
            raise ValueError("face_focus.power must be a finite number in [0, 1]")
        return {"enabled": enabled, "power": float(power)}

    EMA_STATE_FILE = "ema_lora.pt"

    def _save_ema_state(self, checkpoint_folder) -> None:
        """Persist LoRA EMA shadows so resume can continue the average."""
        if self._ema is None:
            return
        path = os.path.join(checkpoint_folder, self.EMA_STATE_FILE)
        torch.save(self._ema.state_dict(), path)

    def _load_ema_state(self, checkpoint_folder) -> None:
        if self._ema is None:
            return
        path = os.path.join(checkpoint_folder, self.EMA_STATE_FILE)
        if not os.path.isfile(path):
            # The shadows were seeded before `load_state`, so they still hold the
            # untrained adapter; reseed them from the resumed weights.
            self._ema.reset()
            print(
                f"WARNING: no {self.EMA_STATE_FILE} in {checkpoint_folder}; "
                "EMA restarted from the resumed LoRA weights."
            )
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._ema.load_state_dict(state)
        print(f"Restored LoRA EMA state from {path}")

    def _validate_timestep_range(self, value):
        """Validate the diffusion-timestep training band (fractions of num_train_timesteps)."""
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("timestep_range must be a mapping")
        min_fraction = value.get("min_fraction", 0.0)
        max_fraction = value.get("max_fraction", 1.0)
        for key, fraction in (("min_fraction", min_fraction), ("max_fraction", max_fraction)):
            if (
                not isinstance(fraction, (int, float))
                or not math.isfinite(fraction)
                or not (0.0 <= fraction <= 1.0)
            ):
                raise ValueError(f"timestep_range.{key} must be a finite number in [0, 1]")
        if min_fraction >= max_fraction:
            raise ValueError("timestep_range.min_fraction must be strictly less than max_fraction")
        return {"min_fraction": float(min_fraction), "max_fraction": float(max_fraction)}

    def _resolve_face_manifest(self, value, *, auto: bool):
        """Load the offline face-box manifest, optionally generating it first."""
        from lorakit.config import resolve_user_path
        from lorakit.identity import load_face_manifest

        if value is None:
            path = Path(self._dataset_folder) / "faces.json"
        else:
            path = resolve_user_path(value, config_path=self._config_path, must_exist=False)

        if auto:
            from lorakit.faces_prep import ensure_face_manifest

            device = "cuda" if str(self._device).startswith("cuda") else "cpu"
            path = ensure_face_manifest(
                self._dataset_folder,
                path,
                device=device,
                allow_missing=True,
            )
        elif not Path(path).is_file():
            if value is None:
                return None
            raise FileNotFoundError(f"face_manifest not found: {path}")

        manifest = load_face_manifest(path)
        print(f"Loaded {len(manifest)} face boxes from {path}")
        return manifest

    def __init__(self, config, version, name, root_folder, *, config_path=None):
        super().__init__(version, name, root_folder)
        self._config_path = config_path

        self._device = config.get("device", "cpu")
        if torch.cuda.is_available():
            if self._device == "cpu":
                print(
                    "Warning: CUDA is available, but CPU is being used. "
                    "Consider using a GPU for faster training."
                )
        else:
            if self._device != "cpu":
                raise RuntimeError(
                    "CUDA is not available, but GPU device was specified. "
                    "Please use a machine with CUDA support or specify 'cpu' as the device."
                )

        print(f"Using device: {self._device}")

        self._allow_tf32 = config.get("allow_tf32", False)

        if self._allow_tf32:
            print("Using TF32 for faster training on Ampere GPUs")
            # Enable TF32 for faster training on Ampere GPUs,
            # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._matmul_precision = config.get("matmul_precision", None)
        if self._matmul_precision:
            print(f"Using matmul precision: {self._matmul_precision}")
            torch.set_float32_matmul_precision(self._matmul_precision)

        self._cudnn_benchmark = config.get("cudnn_benchmark", None)
        if self._cudnn_benchmark:
            print("Using cuDNN benchmark")
            torch.backends.cudnn.benchmark = self._cudnn_benchmark

        self._logging_path = Path(self._experiment_folder, "logs")

        self._train_config = config.get("train", None)
        if self._train_config is None:
            raise ValueError("train is required")

        self._snr_gamma = self._train_config.get("snr_gamma", None)
        if self._snr_gamma is not None and self._snr_gamma <= 0:
            raise ValueError(
                "snr_gamma must be positive when set; omit the key or set it to null to "
                "disable SNR loss weighting. snr_gamma=0 zeroes the training loss."
            )

        self._do_edm_style_training = self._train_config.get("do_edm_style_training", False)
        if self._do_edm_style_training and self._snr_gamma is not None:
            raise ValueError("Cannot specify both do_edm_style_training and snr_gamma")

        dtype_str = self._train_config.get("dtype", None)
        if dtype_str is None:
            raise ValueError("dtype is required")
        dtype_str = dtype_str.lower()
        if dtype_str == "bfloat16" or dtype_str == "bf16":
            self._dtype = torch.bfloat16
            self._mixed_precision = "bf16"
        elif dtype_str == "fp16" or dtype_str == "float16":
            self._dtype = torch.float16
            self._mixed_precision = "fp16"
        elif dtype_str == "float32" or dtype_str == "fp32":
            self._dtype = torch.float32
            self._mixed_precision = "no"
        else:
            raise ValueError("Invalid dtype. Supported dtypes are bf16, fp16, fp32, and float32.")

        self._gradient_accumulation_steps = self._train_config.get("gradient_accumulation_steps", 1)
        if self._gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")

        self._train_seed = self._train_config.get("seed", None)
        if self._train_seed is not None:
            print(f"Using seed: {self._train_seed}")
        else:
            print("No seed specified, using random seed")

        self._model_config = config.get("model", None)
        if self._model_config is None:
            raise ValueError("model is required")

        self._model_name_or_path = self._model_config.get("name_or_path", None)
        if self._model_name_or_path is None:
            raise ValueError("name_or_path is required for model")

        self._local_files_only = self._model_config.get("local_files_only", None)
        if self._local_files_only is not None:
            print(f"Using local files only: {self._local_files_only}")

        self._revision = self._model_config.get("revision", None)
        if self._revision is not None:
            print(f"Using revision: {self._revision}")

        self._variant = self._model_config.get("variant", None)
        if self._variant is not None:
            print(f"Using variant: {self._variant}")

        self._model_type = get_model_type(self._model_config)
        print(f"Using model type: {self._model_type}")

        # Quantization configuration. When enabled, the base UNet (and optionally the
        # text encoders) are loaded in low-bit precision via bitsandbytes while the LoRA
        # adapters are still trained in higher precision (QLoRA-style training).
        self._quantization_config = self._model_config.get("quantization", None)
        self._quantize = False
        self._quantize_bits = None
        self._quantize_text_encoder = False
        if self._quantization_config is not None:
            self._quantize_bits = self._quantization_config.get("bits", 8)
            if self._quantize_bits not in (4, 8):
                raise ValueError("quantization.bits must be either 4 or 8")
            self._quantize = True
            self._quantize_text_encoder = self._quantization_config.get(
                "quantize_text_encoder", False
            )
            self._bnb_4bit_quant_type = self._quantization_config.get("bnb_4bit_quant_type", "nf4")
            if self._bnb_4bit_quant_type not in ("nf4", "fp4"):
                raise ValueError("quantization.bnb_4bit_quant_type must be either 'nf4' or 'fp4'")
            self._bnb_4bit_use_double_quant = self._quantization_config.get(
                "bnb_4bit_use_double_quant", True
            )
            print(
                f"Using {self._quantize_bits}-bit quantization for the base model"
                + (" (including text encoders)" if self._quantize_text_encoder else "")
            )

        self._gradient_checkpointing = self._train_config.get("gradient_checkpointing", False)
        if self._gradient_checkpointing:
            print("Using gradient checkpointing")

        self._train_text_encoder = self._train_config.get("train_text_encoder", False)
        if self._train_text_encoder:
            print("Text encoder will be fine-tuned")

        self._unet_lora = self._train_config.get("lora", self._train_config.get("unet_lora", None))
        if self._unet_lora is None:
            raise ValueError("unet_lora is required")

        (
            self._unet_lora_target_modules,
            self._unet_lora_rank_pattern,
            self._unet_lora_alpha_pattern,
            self._unet_lora_rank,
            self._unet_lora_alpha,
        ) = self._parse_lora_targets(
            self._unet_lora.get(
                "target_modules",
                {"to_k": (4, 4), "to_q": (4, 4), "to_v": (4, 4), "to_out.0": (4, 4)},
            )
        )

        # Optional per-(block x module_type) alpha mask (e.g. exported by
        # perturbation-probe). Its block-qualified regex keys are inserted *before*
        # the generic per-type keys so PEFT's get_pattern_key matches them first,
        # scaling each LoRA layer's alpha by its measured identity impact.
        alpha_pattern_file = self._unet_lora.get("alpha_pattern_file", None)
        if alpha_pattern_file:
            from lorakit.config import resolve_user_path

            alpha_pattern_file = resolve_user_path(
                alpha_pattern_file, config_path=self._config_path, must_exist=True
            )
            mask = self._load_alpha_pattern_file(alpha_pattern_file)
            self._unet_lora_alpha_pattern = {**mask, **self._unet_lora_alpha_pattern}
            self._unet_lora_alpha = max(
                self._unet_lora_alpha, *self._unet_lora_alpha_pattern.values()
            )
            print(
                f"Loaded LoRA alpha mask from {alpha_pattern_file} "
                f"({len(mask)} block-qualified alpha entries)"
            )

        self._unet_lora_init_weights = self._unet_lora.get("init_lora_weights", "gaussian")

        self._unet_lora_use_dora = self._unet_lora.get("use_dora", False)

        if self._train_text_encoder:
            self._text_encoder_lora = self._train_config.get("text_encoder_lora", None)
            if self._text_encoder_lora is None:
                raise ValueError("text_encoder_lora is required")

            (
                self._text_encoder_lora_target_modules,
                self._text_encoder_lora_rank_pattern,
                self._text_encoder_lora_alpha_pattern,
                self._text_encoder_lora_rank,
                self._text_encoder_lora_alpha,
            ) = self._parse_lora_targets(
                self._text_encoder_lora.get(
                    "target_modules",
                    {"q_proj": (4, 4), "k_proj": (4, 4), "v_proj": (4, 4), "out_proj": (4, 4)},
                )
            )

            self._text_encoder_lora_init_weights = self._text_encoder_lora.get(
                "init_lora_weights", "gaussian"
            )

            self._text_encoder_lora_use_dora = self._text_encoder_lora.get("use_dora", False)

        self._batch_size = self._train_config.get("batch_size", 1)
        self._optimizer_name = self._train_config.get("optimizer", None)
        if self._optimizer_name is None:
            raise ValueError("optimizer is required")
        self._optimizer_name = self._optimizer_name.lower()
        if self._optimizer_name not in [
            "adamw",
            "adamw8bit",
            "adamwschedulefree",
            "adamwanchored",
            "adamwanchoredschedulefree",
        ]:
            raise ValueError(
                "Invalid optimizer. Supported optimizers are adamw, adamw8bit, "
                "adamwschedulefree, adamwanchored, and adamwanchoredschedulefree."
            )

        self._optimizer_params = self._train_config.get("optimizer_params", None)
        if self._optimizer_params is None:
            raise ValueError("optimizer_params is required")

        self._resolution = self._train_config.get("resolution", None)
        if not isinstance(self._resolution, int):
            raise ValueError("Invalid resolution. resolution must be an integer")

        dataset_folder = self._train_config.get("dataset_folder", None)
        if dataset_folder is None:
            raise ValueError("dataset_folder is required")
        from lorakit.config import resolve_user_path

        self._dataset_folder = str(
            resolve_user_path(dataset_folder, config_path=self._config_path, must_exist=True)
        )

        # Dataset-level knobs. `DreamBoothDataset` has always accepted these; they
        # were simply unreachable from a config file.
        self._dataset_repeats = self._train_config.get("repeats", 1)
        if not isinstance(self._dataset_repeats, int) or self._dataset_repeats < 1:
            raise ValueError("repeats must be a positive integer")

        self._center_crop = self._train_config.get("center_crop", False)
        if not isinstance(self._center_crop, bool):
            raise ValueError("center_crop must be a boolean")

        # The flip is drawn once per dataset entry at load time, so `repeats: 2`
        # gives each image an independent draw per copy. Mirroring breaks
        # left/right-consistent artifacts in the training set (one-sided lens
        # glare, side lighting, hair part) that the subject token would otherwise
        # absorb as part of the identity.
        self._random_flip = self._train_config.get("random_flip", False)
        if not isinstance(self._random_flip, bool):
            raise ValueError("random_flip must be a boolean")

        # Offline InsightFace box per training image. Path may be filled later via
        # auto_face_manifest once we know face focus / identity need it.
        self._face_manifest_value = self._train_config.get("face_manifest", None)
        self._auto_face_manifest = self._train_config.get("auto_face_manifest", True)
        if not isinstance(self._auto_face_manifest, bool):
            raise ValueError("auto_face_manifest must be a boolean")
        self._face_manifest = None

        self._num_workers = self._train_config.get("num_workers", 1)
        if self._num_workers > 1:
            print(f"Using {self._num_workers} workers for data loading")

        # When enabled, every training image is encoded by the VAE exactly once and
        # the resulting latent is reused for the rest of training. This removes the
        # per-step fp32 VAE forward pass from the hot loop. Safe because the dataset
        # pre-computes a single deterministic crop/flip per image at load time.
        self._cache_latents = self._train_config.get("cache_latents", False)

        # Face-focused identity loss: weight the diffusion loss by a soft face mask
        # (high inside the detected face box, falling to 0 outside) so the LoRA learns
        # the face while leaving the rest of the image close to the base model. Parsed
        # before background preservation because it can also require the face detector.
        self._face_focus_config = self._validate_face_focus_config(
            self._train_config.get("face_focus", {})
        )
        self._face_focus_enabled = self._face_focus_config["enabled"]
        self._face_focus_power = self._face_focus_config["power"]

        background_preservation_raw = self._train_config.get("background_preservation", {}) or {}
        self._background_preservation_config = self._validate_background_preservation_config(
            background_preservation_raw
        )
        self._background_preservation_enabled = self._background_preservation_config["enabled"]

        # Where the trained-region mask comes from. "face_parse" uses an offline
        # 19-class segmentation, which is the only source that can tell clothing
        # apart from the subject and can weight face skin differently from hair.
        # "subject_mask" uses an offline background-removal alpha, so the
        # preserved region is everything outside the subject silhouette rather
        # than everything outside a face rectangle. "manifest" uses the offline
        # per-image face box, which belongs to the training image and is therefore
        # exact at every noise level. "auto" prefers the manifest when one is
        # available (and builds it when face-mask objectives need boxes).
        self._face_mask_source = self._train_config.get("face_mask_source", "auto")
        allowed_mask_sources = {"auto", "manifest", "subject_mask", "face_parse"}
        if self._face_mask_source not in allowed_mask_sources:
            raise ValueError(
                "face_mask_source must be one of: " + ", ".join(sorted(allowed_mask_sources))
            )
        self._use_subject_masks = self._face_mask_source == "subject_mask"
        self._use_face_parse = self._face_mask_source == "face_parse"
        # Both sources give the loss a per-pixel weight map instead of a box.
        self._use_pixel_masks = self._use_subject_masks or self._use_face_parse
        self._subject_mask_config = self._validate_subject_mask_config(
            self._train_config.get("subject_mask", {})
        )
        self._face_parse_config = self._validate_face_parse_config(
            self._train_config.get("face_parse", {})
        )
        self._augmentation_config = self._validate_augmentation_config(
            self._train_config.get("augmentation", {})
        )
        self._subject_mask_manifest = None
        self._subject_mask_folder = None
        self._parse_manifest = None
        self._parse_folder = None
        self._parse_train_lut = None
        self._parse_background_lut = None

        self._identity_loss_config = self._validate_identity_loss_config(
            self._train_config.get("identity_loss", {})
        )
        self._identity_loss_enabled = self._identity_loss_config["enabled"]

        # Sample ArcFace scoring needs the reference face manifest; validate early so
        # faces.json resolution below can see it (full sample.* parsing happens later).
        sample_config_early = config.get("sample") or {}
        if not isinstance(sample_config_early, dict):
            sample_config_early = {}
        self._sample_arcface_config = self._validate_sample_arcface_config(
            sample_config_early.get("arcface_metric", {})
        )
        self._sample_arcface_enabled = self._sample_arcface_config["enabled"]

        # Build/load faces.json as part of training when face-local objectives need
        # per-image boxes (no separate lorakit-faces step required).
        needs_face_manifest = (
            self._face_mask_source in {"auto", "manifest"}
            or self._identity_loss_enabled
            or self._sample_arcface_enabled
            or self._face_focus_enabled
            or self._background_preservation_enabled
        )
        if needs_face_manifest:
            identity_manifest_override = self._identity_loss_config.get("face_manifest")
            sample_manifest_override = (
                self._sample_arcface_config.get("face_manifest")
                if self._sample_arcface_enabled
                else None
            )
            manifest_value = (
                identity_manifest_override
                if identity_manifest_override is not None
                else sample_manifest_override
                if sample_manifest_override is not None
                else self._face_manifest_value
            )
            self._face_manifest = self._resolve_face_manifest(
                manifest_value,
                auto=self._auto_face_manifest,
            )
        if self._face_mask_source == "manifest" and self._face_manifest is None:
            raise ValueError(
                "face_mask_source: manifest requires a face_manifest "
                "(enable auto_face_manifest or run lorakit-faces)"
            )
        self._use_manifest_face_boxes = self._face_mask_source in {"auto", "manifest"} and (
            self._face_manifest is not None
        )

        if self._use_subject_masks:
            self._subject_mask_folder, self._subject_mask_manifest = self._resolve_subject_masks(
                self._subject_mask_config
            )
            print(
                f"Using {len(self._subject_mask_manifest)} offline subject mask(s) from "
                f"{self._subject_mask_folder} for background preservation"
            )
        if self._use_face_parse:
            self._parse_folder, self._parse_manifest = self._resolve_face_parses(
                self._face_parse_config
            )
            self._parse_train_lut, self._parse_background_lut = resolve_class_weights(
                self._face_parse_config["class_weights"]
            )
            print(
                f"Using {len(self._parse_manifest)} offline face parse map(s) from "
                f"{self._parse_folder} (skin weight "
                f"{self._parse_train_lut[CLASS_INDEX['skin']]:.2f}, clothing excluded)"
            )
            if self._face_parse_config["background_source"] == "subject_mask":
                # The parser's background class is unreliable at the person/scene
                # boundary; BiRefNet's alpha is not, so it decides what counts as
                # background while the parse map still decides what gets trained.
                self._subject_mask_folder, self._subject_mask_manifest = (
                    self._resolve_subject_masks(self._subject_mask_config)
                )
                print(
                    f"Anchoring background to {len(self._subject_mask_manifest)} BiRefNet "
                    f"alpha mask(s) from {self._subject_mask_folder}"
                )

        # Objective object is needed for any face-mask path (focus or background).
        # Masks must come from an offline face manifest, subject alpha, or parse map.
        self._needs_face_mask_objective = (
            self._face_focus_enabled or self._background_preservation_enabled
        )
        if (
            self._needs_face_mask_objective
            and not self._use_manifest_face_boxes
            and not self._use_pixel_masks
        ):
            raise ValueError(
                "face_focus / background_preservation require a mask source "
                "(face_mask_source: face_parse, subject_mask, or manifest with faces.json)"
            )
        face_detector_raw = self._train_config.get("face_detector")
        if face_detector_raw is None:
            # Legacy: soft-mask settings lived under background_preservation.
            face_detector_raw = background_preservation_raw
        self._face_detector_config = self._validate_face_detector_config(
            face_detector_raw,
            required=self._needs_face_mask_objective,
        )
        if self._face_detector_config["trust_radius"] > 0 and self._face_manifest is None:
            raise ValueError(
                "face_detector.trust_radius requires a face_manifest to anchor the "
                "region on (enable auto_face_manifest or set face_manifest)"
            )

        if self._identity_loss_enabled:
            if self._face_manifest is None:
                raise ValueError(
                    "identity_loss.enabled requires a face_manifest "
                    "(enable auto_face_manifest or set face_manifest)"
                )
            self._identity_face_manifest = self._face_manifest
        else:
            self._identity_face_manifest = None

        if self._sample_arcface_enabled and self._face_manifest is None:
            raise ValueError(
                "sample.arcface_metric.enabled requires a face_manifest "
                "(enable auto_face_manifest or set face_manifest)"
            )
        # Restrict which diffusion timesteps the loss trains on. Smaller timesteps are
        # lower noise; capping max_fraction below 1.0 keeps the base model's global
        # composition (decided at high noise) and only injects identity at low noise.
        self._timestep_range = self._validate_timestep_range(
            self._train_config.get("timestep_range", {})
        )

        # `channels_last` lays out the 4D conv activations for better tensor-core
        # utilization on Ampere/Ada. `torch_compile` JIT-compiles the UNet with
        # TorchInductor; `torch_compile_mode` selects the inductor preset.
        self._channels_last = self._train_config.get("channels_last", False)
        self._torch_compile = self._train_config.get("torch_compile", False)
        self._torch_compile_mode = self._train_config.get("torch_compile_mode", "default")

        self._with_prior_preservation = self._train_config.get("with_prior_preservation", False)
        # Folder of class images (e.g. base-model generations of the bare class
        # prompt). Required for prior preservation: each step trains on instance
        # and class images jointly so the LoRA keeps the base model's prior for
        # the class (backgrounds, scenes, textures) instead of collapsing onto
        # the instance dataset's look.
        self._class_data_folder = self._train_config.get("class_data_folder", None)
        self._num_class_images = self._train_config.get("num_class_images", None)
        if self._with_prior_preservation:
            if self._class_data_folder is None:
                raise ValueError(
                    "class_data_folder is required when with_prior_preservation is true"
                )
            print("Using prior preservation")

        self._prior_loss_weight = self._train_config.get("prior_loss_weight", 1.0)
        if self._prior_loss_weight < 0.0:
            raise ValueError("prior_loss_weight must be >= 0.0")

        self._lr_scheduler = self._train_config.get("lr_scheduler", None)
        if self._lr_scheduler is None:
            raise ValueError("lr_scheduler is required")

        self._lr_scheduler_params = self._train_config.get("lr_scheduler_params", None)
        if self._lr_scheduler_params is None:
            raise ValueError("lr_scheduler_params is required")

        self._class_prompt = config.get("class_prompt", None)
        if self._class_prompt is None:
            raise ValueError("class_prompt is required")

        self._instant_prompt = config.get("instant_prompt", None)
        if self._instant_prompt is None:
            raise ValueError("instant_prompt is required")

        self._save_every = self._train_config.get("save_every", 0)

        self._max_grad_norm = self._train_config.get("max_grad_norm", None)
        if self._max_grad_norm is not None and self._max_grad_norm < 0.0:
            raise ValueError("max_grad_norm must be >= 0.0 or None")

        self._max_train_steps = self._train_config.get("max_train_steps", None)
        if self._max_train_steps is not None:
            if not isinstance(self._max_train_steps, int) or self._max_train_steps < 0:
                raise ValueError("max_train_steps must be a non-negative integer or None")

        self._skip_pre_train_sample = self._train_config.get("skip_pre_train_sample", False)
        self._disable_sampling = self._train_config.get("disable_sampling", False)

        self._checkpoints_total_limit = self._train_config.get("checkpoints_total_limit", None)
        self._sample_config = config.get("sample", None)
        self._sample_prompt_seeds: list[int] | None = None
        if self._sample_config is not None:
            self._sample_every = self._sample_config.get("sample_every", 0)
            if self._sample_every < 0:
                raise ValueError("sample_every must be >= 0")

            prompt_file = self._sample_config.get("prompt_file", None)
            inline_prompts = self._sample_config.get("prompts", None)
            if prompt_file:
                from lorakit.config import resolve_user_path
                from lorakit.prompts import load_training_sample_prompts

                resolved = resolve_user_path(
                    prompt_file, config_path=self._config_path, must_exist=True
                )
                self._sample_prompts, self._sample_neg, self._sample_prompt_seeds = (
                    load_training_sample_prompts(
                        resolved,
                        trigger=self._instant_prompt,
                        class_word=self._class_prompt,
                    )
                )
                print(f"Loaded {len(self._sample_prompts)} sample prompts from {resolved}")
            elif inline_prompts is not None:
                self._sample_prompts = inline_prompts
                self._sample_neg = self._sample_config.get("neg", None)
                if self._sample_neg is None:
                    raise ValueError("sample.neg is required when sample.prompts is set")
                if len(self._sample_neg) != len(self._sample_prompts):
                    raise ValueError("neg and prompts must have the same length")
            else:
                raise ValueError("sample requires prompt_file or prompts")

            self._sample_seed = self._sample_config.get("seed", 42)
            default_guidance_scale, default_steps = sample_defaults(self._model_type)
            self._sample_guidance_scale = self._sample_config.get(
                "guidance_scale", default_guidance_scale
            )
            self._sample_steps = self._sample_config.get(
                "steps", self._sample_config.get("sample_steps", default_steps)
            )
            self._sample_scheduler = self._sample_config.get("scheduler", None)
            self._use_prompt_seeds = self._sample_config.get(
                "use_prompt_seeds", prompt_file is not None
            )
            self._walk_seed = self._sample_config.get("walk_seed", False)
            if self._walk_seed and self._use_prompt_seeds:
                print("Warning: walk_seed ignored when use_prompt_seeds is true")
            elif self._walk_seed:
                print("Walking seed")
            # Prefer sample.skip_initial; keep train.skip_pre_train_sample as a fallback.
            skip_initial = self._sample_config.get("skip_initial", None)
            if skip_initial is None:
                skip_initial = self._skip_pre_train_sample
            if not isinstance(skip_initial, bool):
                raise ValueError("sample.skip_initial must be a boolean")
            self._skip_pre_train_sample = skip_initial
            if self._skip_pre_train_sample:
                print("Skipping initial (step-0) sampling")
            elif self._use_prompt_seeds:
                print("Using per-prompt seeds from prompt file")
            if self._sample_arcface_enabled:
                print(
                    "Sample ArcFace metric enabled "
                    f"(model={self._sample_arcface_config['model_id']})"
                )
        elif self._sample_arcface_enabled:
            raise ValueError(
                "sample.arcface_metric.enabled requires a sample block (prompt_file or prompts)"
            )

        self._resume_from_checkpoint = self._train_config.get("resume_from_checkpoint", None)
        if self._resume_from_checkpoint is not None:
            print(f"Resuming from checkpoint: {self._resume_from_checkpoint}")

        # EMA configuration (shadow average of trainable LoRA / DoRA params).
        self._use_ema = bool(self._train_config.get("use_ema", False))
        self._ema_decay = float(self._train_config.get("ema_decay", 0.999))
        self._ema: LoRAEMA | None = None
        if self._use_ema:
            if not (0.0 <= self._ema_decay < 1.0):
                raise ValueError("ema_decay must be in [0, 1)")
            horizon = round(1.0 / (1.0 - self._ema_decay)) if self._ema_decay > 0 else 1
            print(f"Using EMA with decay {self._ema_decay} (~{horizon}-step horizon)")
            if self._max_train_steps is not None and horizon > self._max_train_steps:
                print(
                    f"WARNING: ema_decay {self._ema_decay} averages over ~{horizon} steps but "
                    f"max_train_steps is {self._max_train_steps}; samples will lag the live "
                    "weights badly. Aim for a horizon near 5-15% of max_train_steps "
                    f"(e.g. ema_decay {1.0 - 1.0 / max(1, self._max_train_steps // 10):.4g})."
                )

        self._loss_decay = self._train_config.get("loss_decay", 0.995)

        # Per-section training-loop profiler. Disabled by default; enable via the
        # `profile` block in the config (or the LORAKIT_PROFILE env override).
        profile_config = self._train_config.get("profile", None)
        if isinstance(profile_config, bool):
            profile_config = {"enabled": profile_config}
        profile_config = profile_config or {}
        self._profile_enabled = profile_config.get("enabled", False)
        self._profile_warmup = profile_config.get("warmup", 3)
        self._profile_report_every = profile_config.get("report_every", 0)
        if self._profile_enabled:
            print("Training-loop profiling enabled")

    def _get_bnb_config(self, framework: str):
        """
        Build a bitsandbytes quantization config for the requested framework.

        diffusers and transformers each ship their own thin `BitsAndBytesConfig`
        wrapper, so the correct one must be used depending on the component being
        loaded (`diffusers` for the UNet/VAE, `transformers` for the text encoders).
        """
        if not self._quantize:
            return None
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "bitsandbytes is required for quantization. Reinstall dependencies with `uv sync`."
            ) from err

        if framework == "diffusers":
            from diffusers import BitsAndBytesConfig as BnbConfig
        elif framework == "transformers":
            from transformers import BitsAndBytesConfig as BnbConfig
        else:
            raise ValueError(f"Unknown quantization framework: {framework}")

        if self._quantize_bits == 8:
            return BnbConfig(load_in_8bit=True)

        return BnbConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self._bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self._bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=self._dtype,
        )

    def _get_optimizer(self, params_to_optimize):
        if self._optimizer_name == "adamw":
            from torch.optim import AdamW

            optimizer = AdamW(params_to_optimize, **self._optimizer_params)
        elif self._optimizer_name == "adamw8bit":
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use AdamW8bit optimizer"
                ) from None
            optimizer = bnb.optim.AdamW8bit(params_to_optimize, **self._optimizer_params)
        elif self._optimizer_name == "adamwschedulefree":
            try:
                from schedulefree import AdamWScheduleFree
            except ImportError:
                raise ImportError(
                    "Please install schedulefree to use AdamWScheduleFree optimizer"
                ) from None
            optimizer = AdamWScheduleFree(params_to_optimize, **self._optimizer_params)
        elif self._optimizer_name == "adamwanchored":
            from lorakit.optimizers import AnchoredAdamW

            optimizer = AnchoredAdamW(params_to_optimize, **self._optimizer_params)
        elif self._optimizer_name == "adamwanchoredschedulefree":
            from lorakit.optimizers import AnchoredAdamWScheduleFree

            optimizer = AnchoredAdamWScheduleFree(params_to_optimize, **self._optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {self._optimizer_name}")

        return optimizer

    def _get_text_encoder(self, subfolder: str):
        text_encoder_config = AutoConfig.from_pretrained(
            self._model_name_or_path,
            subfolder=subfolder,
            revision=self._revision,
            local_files_only=self._local_files_only,
        )
        model_class = text_encoder_config.architectures[0]

        cls = None
        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            cls = CLIPTextModel
        elif model_class == "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection

            cls = CLIPTextModelWithProjection
        elif model_class == "T5EncoderModel":
            from transformers import T5EncoderModel

            cls = T5EncoderModel
        else:
            raise ValueError(f"{model_class} is not supported.")

        kwargs = {}
        if self._quantize and self._quantize_text_encoder:
            kwargs["quantization_config"] = self._get_bnb_config("transformers")
            kwargs["device_map"] = {"": self._device}
        return cls.from_pretrained(
            self._model_name_or_path,
            subfolder=subfolder,
            revision=self._revision,
            dtype=self._dtype,
            **kwargs,
        )

    def _get_tokenizer(self, subfolder: str):
        return AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            subfolder=subfolder,
            revision=self._revision,
            use_fast=False,
            local_files_only=self._local_files_only,
        )

    def _get_noise_scheduler(self):
        model_index_filename = "model_index.json"
        if Path(self._model_name_or_path).is_dir():
            model_index = Path(self._model_name_or_path) / model_index_filename
        else:
            model_index = hf_hub_download(
                repo_id=self._model_name_or_path,
                filename=model_index_filename,
                revision=self._revision,
            )

        with open(model_index) as f:
            scheduler_type = json.load(f)["scheduler"][1]
        if "EDM" in scheduler_type:
            from diffusers import EDMEulerScheduler

            noise_scheduler = EDMEulerScheduler.from_pretrained(
                self._model_name_or_path,
                subfolder="scheduler",
                local_files_only=self._local_files_only,
            )
            print("Using EDMEulerScheduler")
            self._do_edm_style_training = True
            is_edm_scheduler = True
        elif self._do_edm_style_training:
            from diffusers import EulerDiscreteScheduler

            noise_scheduler = EulerDiscreteScheduler.from_pretrained(
                self._model_name_or_path,
                subfolder="scheduler",
                local_files_only=self._local_files_only,
            )
            print("Using EulerDiscreteScheduler")
            is_edm_scheduler = False
        else:
            from diffusers import DDPMScheduler

            noise_scheduler = DDPMScheduler.from_pretrained(
                self._model_name_or_path,
                subfolder="scheduler",
                local_files_only=self._local_files_only,
            )
            print("Using DDPMScheduler")
            is_edm_scheduler = False
        if self._do_edm_style_training:
            print("Performing EDM-style training")

        return noise_scheduler, is_edm_scheduler

    def _get_vae(self):
        # The VAE is always in float32 to avoid NaN losses.
        return AutoencoderKL.from_pretrained(
            self._model_name_or_path,
            subfolder="vae",
            revision=self._revision,
            local_files_only=self._local_files_only,
        )

    def _get_latents(self, vae):
        latents_mean = latents_std = None
        if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
            latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
            latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)
        return latents_mean, latents_std

    def _get_unet(self):
        kwargs = {}
        if self._quantize:
            kwargs["quantization_config"] = self._get_bnb_config("diffusers")
            kwargs["device_map"] = {"": self._device}
        return UNet2DConditionModel.from_pretrained(
            self._model_name_or_path,
            subfolder="unet",
            torch_dtype=self._dtype,
            local_files_only=self._local_files_only,
            **kwargs,
        )

    def _cast_training_params_to_fp32(self, unet, text_encoder_1, text_encoder_2):
        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if self._dtype == torch.float16 or self._dtype == torch.bfloat16:
            models = [unet]
            if self._train_text_encoder:
                models.extend([text_encoder_1, text_encoder_2])
            cast_training_params(models, dtype=torch.float32)

    def _generate_samples(
        self, accelerator, global_step, vae, unet, text_encoder_one, text_encoder_two
    ):
        if text_encoder_one is None:
            text_encoder_one = self._get_text_encoder("text_encoder")

        if text_encoder_two is None:
            text_encoder_two = self._get_text_encoder("text_encoder_2")

        def _unwrap_for_pipeline(model):
            model = accelerator.unwrap_model(model)
            return model._orig_mod if is_compiled_module(model) else model

        with torch.no_grad():
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self._model_name_or_path,
                vae=vae,
                text_encoder=_unwrap_for_pipeline(text_encoder_one),
                text_encoder_2=_unwrap_for_pipeline(text_encoder_two),
                unet=_unwrap_for_pipeline(unet),
                revision=self._revision,
                torch_dtype=self._dtype,
                use_safetensors=True,
                local_files_only=self._local_files_only,
            )
            pipeline.set_progress_bar_config(disable=True)
            if self._sample_scheduler is not None:
                pipeline.scheduler = make_scheduler(pipeline.scheduler, self._sample_scheduler)
            # Quantized (bitsandbytes) components cannot be moved with `.to(device)`; they are
            # already placed on the target device at load time, so we only move the pipeline
            # when training without quantization.
            if not self._quantize:
                pipeline = pipeline.to(accelerator.device)
            else:
                vae.to(accelerator.device)
            # The base is trained in `self._dtype` (e.g. bf16), so the denoising loop
            # produces latents in that dtype. The training VAE is kept in fp32, and the
            # SDXL pipeline only auto-upcasts a *fp16* VAE, so a fp32 VAE would hit a
            # dtype mismatch at decode on CUDA. bf16 has the same exponent range as fp32
            # (no SDXL overflow), so casting the pipeline VAE to the training dtype for
            # sampling is both correct and safe. Restored to fp32 afterwards.
            pipeline.vae.to(device=accelerator.device, dtype=self._dtype)
            # Currently the context determination is a bit hand-wavy. We can improve it
            # in the future if there's a better way to condition it. Reference:
            # https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
            inference_ctx = (
                contextlib.nullcontext()
                # if "playground" in self.model_name_or_path
                # else torch.cuda.amp.autocast(accelerator.device)
            )
            samples_dir = Path(self._experiment_folder) / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            generated_images = []
            generated_paths = []
            with inference_ctx:
                if self._sample_neg:
                    sample_prompts = zip(self._sample_prompts, self._sample_neg, strict=False)
                else:
                    none_neg = [None] * len(self._sample_prompts)
                    sample_prompts = zip(self._sample_prompts, none_neg, strict=False)
                for i, (p, n) in enumerate(sample_prompts):
                    if self._use_prompt_seeds and self._sample_prompt_seeds:
                        gen_seed = self._sample_prompt_seeds[i]
                    elif self._walk_seed:
                        gen_seed = self._sample_seed + i
                    else:
                        gen_seed = self._sample_seed
                    generator = (
                        torch.Generator(device=accelerator.device).manual_seed(gen_seed)
                        if gen_seed is not None
                        else None
                    )
                    image = pipeline(
                        prompt=p,
                        negative_prompt=n,
                        height=self._resolution,
                        width=self._resolution,
                        guidance_scale=self._sample_guidance_scale,
                        num_inference_steps=self._sample_steps,
                        generator=generator,
                    ).images[0]
                    image_path = samples_dir / f"{int(time.time())}_{global_step:010d}_{i}.jpg"
                    image.save(image_path)
                    generated_images.append(image)
                    generated_paths.append(image_path)
            # Quantitative identity / consistency readout (CLIP-vision based).
            try:
                self._compute_identity_metrics(accelerator, generated_images, global_step)
            except Exception as e:  # noqa: BLE001 - metric is best-effort, never fail training
                print(f"IDENTITY step={global_step} metric_error={type(e).__name__}: {e}")
            if self._sample_arcface_enabled:
                try:
                    self._compute_sample_arcface_metrics(
                        accelerator,
                        generated_images,
                        generated_paths,
                        global_step,
                        samples_dir,
                    )
                except Exception as e:  # noqa: BLE001 - metric is best-effort, never fail training
                    print(f"ARCFACE step={global_step} metric_error={type(e).__name__}: {e}")
        # Restore the (shared) VAE to fp32 in case the training loop still uses it
        # (e.g. when latents are not cached).
        vae.to(dtype=torch.float32)
        del pipeline
        _flush()

    def _get_clip_identity_model(self, device):
        """Lazily load a CLIP-vision model used only for the identity metric."""
        if getattr(self, "_clip_model", None) is not None:
            return self._clip_model, self._clip_processor
        from transformers import CLIPImageProcessor, CLIPModel

        model_id = getattr(self, "_identity_clip_model_id", "openai/clip-vit-base-patch32")
        print(f"Loading CLIP identity model ({model_id}) for the identity metric")
        self._clip_model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self._clip_processor = CLIPImageProcessor.from_pretrained(model_id)
        return self._clip_model, self._clip_processor

    @torch.no_grad()
    def _embed_images_clip(self, images, device):
        model, processor = self._get_clip_identity_model(device)
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            device=device, dtype=next(model.parameters()).dtype
        )
        feats = model.get_image_features(pixel_values=pixel_values)
        # transformers >=5 returns a BaseModelOutputWithPooling whose `pooler_output`
        # holds the projected image embeds; older versions return the tensor directly.
        if not isinstance(feats, torch.Tensor):
            feats = getattr(feats, "pooler_output", None)
            if feats is None:
                raise AttributeError("CLIP get_image_features returned no usable embeds")
        return F.normalize(feats.float(), dim=-1)

    def _get_reference_identity_embeds(self, device):
        """CLIP embeddings of the training images (computed once, then cached)."""
        if getattr(self, "_ref_identity_embeds", None) is not None:
            return self._ref_identity_embeds
        from PIL import Image
        from PIL.ImageOps import exif_transpose

        exts = (".png", ".jpg", ".jpeg", ".webp")
        paths = sorted(p for p in Path(self._dataset_folder).iterdir() if p.suffix.lower() in exts)
        imgs = []
        for p in paths:
            img = exif_transpose(Image.open(p))
            if img.mode != "RGB":
                img = img.convert("RGB")
            imgs.append(img)
        self._ref_identity_embeds = self._embed_images_clip(imgs, device)
        return self._ref_identity_embeds

    @torch.no_grad()
    def _compute_identity_metrics(self, accelerator, generated_images, global_step):
        """
        Log a quantitative identity / consistency readout for the generated samples.

        - likeness: mean cosine sim of each generated sample to the training identity
          centroid (higher = generated subject looks more like the training photos).
        - consistency: mean pairwise cosine sim among the generated samples (higher =
          the subject renders consistently across different prompts/seeds).
        - max_match: mean over generated samples of the *best* match to any single
          training image (a high value alongside low diversity hints at memorization).
        - ref_cohesion: mean pairwise sim within the training set itself (context only).
        """
        if not generated_images:
            return
        device = accelerator.device
        ref = self._get_reference_identity_embeds(device)
        gen = self._embed_images_clip(generated_images, device)

        ref_mean = F.normalize(ref.mean(0, keepdim=True), dim=-1)
        likeness = (gen @ ref_mean.t()).squeeze(1).mean().item()

        if gen.shape[0] > 1:
            sim = gen @ gen.t()
            n = gen.shape[0]
            consistency = ((sim.sum() - n) / (n * (n - 1))).item()
        else:
            consistency = float("nan")

        if ref.shape[0] > 1:
            rsim = ref @ ref.t()
            rn = ref.shape[0]
            ref_cohesion = ((rsim.sum() - rn) / (rn * (rn - 1))).item()
        else:
            ref_cohesion = float("nan")

        max_match = (gen @ ref.t()).max(dim=1).values.mean().item()

        print(
            f"IDENTITY step={global_step} likeness={likeness:.4f} "
            f"consistency={consistency:.4f} max_match={max_match:.4f} "
            f"ref_cohesion={ref_cohesion:.4f} n_ref={ref.shape[0]} n_gen={gen.shape[0]}"
        )

    def _get_sample_arcface_encoder(self, device):
        """Lazily load the frozen ArcFace encoder used for sample scoring."""
        if getattr(self, "_sample_arcface_encoder", None) is not None:
            return self._sample_arcface_encoder
        model_id = self._sample_arcface_config["model_id"]
        print(f"Loading ArcFace sample metric encoder from {model_id}")
        self._sample_arcface_encoder = load_arcface_encoder(model_id).to(device).eval()
        return self._sample_arcface_encoder

    def _get_sample_arcface_detector(self, device):
        """Lazily build InsightFace detection used only for sample scoring."""
        if getattr(self, "_sample_arcface_detector", None) is not None:
            return self._sample_arcface_detector
        from lorakit.faces_prep import build_insightface_detector

        device_str = "cuda" if str(device).startswith("cuda") else "cpu"
        print(
            "Loading InsightFace detector for sample ArcFace metric "
            f"(model={self._sample_arcface_config['insightface_model']}, device={device_str})"
        )
        self._sample_arcface_detector = build_insightface_detector(
            model_name=self._sample_arcface_config["insightface_model"],
            device=device_str,
            det_size=self._sample_arcface_config["det_size"],
            det_threshold=self._sample_arcface_config["det_threshold"],
        )
        return self._sample_arcface_detector

    def _get_sample_arcface_reference(self, device):
        """Per-image subject reference embeddings for sample ArcFace scoring.

        Samples are scored against every input face individually, so the whole
        matrix is kept instead of collapsing it to a centroid.
        """
        if getattr(self, "_sample_arcface_reference", None) is not None:
            return self._sample_arcface_reference, self._sample_arcface_reference_names

        encoder = self._get_sample_arcface_encoder(device)
        reference_folder = Path(
            self._sample_arcface_config.get("reference_images_folder") or self._dataset_folder
        )
        exts = (".png", ".jpg", ".jpeg", ".webp")
        reference_paths = sorted(
            path for path in reference_folder.iterdir() if path.suffix.lower() in exts
        )
        if self._face_manifest is None:
            raise ValueError("sample ArcFace metric requires a face_manifest")
        print(
            f"Building sample ArcFace reference embeddings from {len(reference_paths)} images "
            f"in {reference_folder}"
        )
        embeddings, used_paths = ArcFaceIdentityObjective.build_reference_embeddings(
            encoder,
            image_paths=reference_paths,
            face_manifest=self._face_manifest,
            crop_size=self._sample_arcface_config["crop_size"],
            bbox_padding_fraction=self._sample_arcface_config["bbox_padding_fraction"],
            device=device,
        )
        self._sample_arcface_reference = embeddings
        self._sample_arcface_reference_names = [path.name for path in used_paths]
        cohesion = reference_cohesion(embeddings)
        cohesion_str = f"{cohesion:.4f}" if cohesion is not None else "n/a"
        print(
            f"Sample ArcFace references: {embeddings.shape[0]} faces "
            f"(mean pairwise cosine={cohesion_str})"
        )
        return self._sample_arcface_reference, self._sample_arcface_reference_names

    @torch.no_grad()
    def _compute_sample_arcface_metrics(
        self, accelerator, generated_images, generated_paths, global_step, samples_dir
    ):
        """Score samples with ArcFace and persist per-step + leaderboard JSON."""
        if not generated_images:
            return
        device = accelerator.device
        encoder = self._get_sample_arcface_encoder(device)
        reference, reference_names = self._get_sample_arcface_reference(device)
        detector = self._get_sample_arcface_detector(device)
        image_names = [str(path.name) for path in generated_paths]
        scores = score_generated_samples(
            images=generated_images,
            encoder=encoder,
            reference_embeddings=reference,
            detector=detector,
            device=device,
            crop_size=self._sample_arcface_config["crop_size"],
            bbox_padding_fraction=self._sample_arcface_config["bbox_padding_fraction"],
            selection=self._sample_arcface_config["selection"],
            image_names=image_names,
            reference_names=reference_names,
        )
        step_record = {
            "step": int(global_step),
            "mean_max_similarity": scores["mean_max_similarity"],
            "mean_similarity": scores["mean_similarity"],
            "detection_rate": scores["detection_rate"],
            "n_samples": scores["n_samples"],
            "n_detected": scores["n_detected"],
            "n_references": scores["n_references"],
            "reference_cohesion": scores["reference_cohesion"],
            "samples": scores["samples"],
        }
        step_path = samples_dir / f"{int(time.time())}_arcface_step_{int(global_step):010d}.json"
        step_path.write_text(json.dumps(step_record, indent=2) + "\n", encoding="utf-8")

        summary_path = Path(self._experiment_folder) / "arcface_scores.json"
        summary = record_step_scores(
            summary_path,
            {
                "step": int(global_step),
                "mean_max_similarity": scores["mean_max_similarity"],
                "mean_similarity": scores["mean_similarity"],
                "detection_rate": scores["detection_rate"],
                "n_samples": scores["n_samples"],
                "n_detected": scores["n_detected"],
                "detail": step_path.relative_to(self._experiment_folder).as_posix(),
            },
        )
        mean_max = scores["mean_max_similarity"]
        mean_max_str = f"{mean_max:.4f}" if mean_max is not None else "nan"
        centroid = scores["mean_similarity"]
        centroid_str = f"{centroid:.4f}" if centroid is not None else "nan"
        best_step = summary.get("best_step")
        best_mean = summary.get("best_mean_max_similarity")
        best_str = (
            f"best_step={best_step} best_mean_max={best_mean:.4f}"
            if best_step is not None and best_mean is not None
            else "best_step=none"
        )
        print(
            f"ARCFACE step={global_step} mean_max={mean_max_str} centroid={centroid_str} "
            f"detected={scores['n_detected']}/{scores['n_samples']} "
            f"detection_rate={scores['detection_rate']:.2f} {best_str}"
        )

    def _print_best_sample_arcface_step(self):
        """Print the leaderboard winner after training finishes."""
        if not self._sample_arcface_enabled:
            return
        summary_path = Path(self._experiment_folder) / "arcface_scores.json"
        if not summary_path.is_file():
            print("ARCFACE: no sample scores written; cannot recommend a checkpoint step")
            return
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"ARCFACE: failed to read {summary_path}: {error}")
            return
        best_step = summary.get("best_step")
        best_mean = summary.get("best_mean_max_similarity")
        best_centroid = summary.get("best_mean_similarity")
        best_rate = summary.get("best_detection_rate")
        if best_step is None or best_mean is None:
            print("ARCFACE: no step had a detectable face; cannot recommend a checkpoint")
            return
        rate_str = f"{best_rate:.2f}" if isinstance(best_rate, (int, float)) else "n/a"
        centroid_str = f"{best_centroid:.4f}" if isinstance(best_centroid, (int, float)) else "n/a"
        print(
            f"ARCFACE best checkpoint step={best_step} "
            f"mean_max_similarity={best_mean:.4f} mean_centroid_similarity={centroid_str} "
            f"detection_rate={rate_str} (see {summary_path})"
        )

    def run(self, callbacks: list[BaseCallback] = None):
        if callbacks is None:
            callbacks = []
        accelerator_project_config = ProjectConfiguration(
            project_dir=self._experiment_folder, logging_dir=self._logging_path
        )
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            mixed_precision=self._mixed_precision,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )

        def unwrap_model(model):
            """
            Unwraps a model from its accelerator and compilation wrappers.

            This function does two things:
            1. It uses the accelerator to unwrap the model, removing any distributed
                or parallel processing wrappers.
            2. It checks if the model is a compiled module (e.g., using torch.compile),
                and if so, it returns the original module instead of the compiled version.

            Args:
                model: The model to unwrap.

            Returns:
                The unwrapped model, free from accelerator and compilation wrappers.
            """
            model = accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()

            Path(self._experiment_folder, parents=True, exist_ok=True)
            Path(self._logging_path, parents=True, exist_ok=True)
            dump_face_masks = self._train_config.get("dump_face_masks", True)
            if not isinstance(dump_face_masks, bool):
                raise ValueError("dump_face_masks must be a boolean")
            # With per-pixel masks the rectangle preview would misrepresent the
            # trained region, so it is dumped from the dataset variants instead.
            if dump_face_masks and self._face_manifest is not None and not self._use_pixel_masks:
                masks_dir = Path(self._experiment_folder) / "face_masks"
                written = dump_manifest_soft_face_masks(
                    masks_dir,
                    self._dataset_folder,
                    self._face_manifest,
                    edge_sharpness=self._face_detector_config["mask_edge_sharpness"],
                    face_focus_power=(self._face_focus_power if self._face_focus_enabled else 1.0),
                )
                print(
                    f"Wrote {written} soft face-mask previews to {masks_dir} "
                    f"(edge_sharpness={self._face_detector_config['mask_edge_sharpness']})"
                )

        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if self._train_seed is not None:
            set_seed(self._train_seed)

        print("Loading noise scheduler")
        noise_scheduler, is_edm_scheduler = self._get_noise_scheduler()
        if self._needs_face_mask_objective and self._do_edm_style_training:
            raise ValueError("face-mask objectives do not support do_edm_style_training")

        # Quantized models are placed on the target device at load time via `device_map`,
        # so they must not be moved again with `.to(device)` afterwards.
        quantize_text_encoder = self._quantize and self._quantize_text_encoder

        print("Loading text encoder 1")
        tokenizer_1 = self._get_tokenizer("tokenizer")
        text_encoder_1 = self._get_text_encoder("text_encoder")
        text_encoder_1.requires_grad_(False)
        if not quantize_text_encoder:
            text_encoder_1.to(accelerator.device)

        print("Loading text encoder 2")
        tokenizer_2 = self._get_tokenizer("tokenizer_2")
        text_encoder_2 = self._get_text_encoder("text_encoder_2")
        text_encoder_2.requires_grad_(False)
        if not quantize_text_encoder:
            text_encoder_2.to(accelerator.device)

        print("Loading VAE")
        vae = self._get_vae()
        vae.requires_grad_(False)
        vae.to(accelerator.device, dtype=torch.float32)
        latents_mean, latents_std = self._get_latents(vae)

        background_objective = None
        if self._needs_face_mask_objective:
            if self._use_manifest_face_boxes:
                print("Using offline face manifest for face masks")
            background_objective = LatentBackgroundPreservationObjective(
                mask_edge_sharpness=self._face_detector_config["mask_edge_sharpness"],
            ).to(accelerator.device)

        identity_objective = None
        if self._identity_loss_enabled:
            print(f"Loading ArcFace identity encoder from {self._identity_loss_config['model_id']}")
            arcface_encoder = load_arcface_encoder(self._identity_loss_config["model_id"]).to(
                accelerator.device, dtype=torch.float32
            )
            reference_folder = (
                self._identity_loss_config["reference_images_folder"] or self._dataset_folder
            )
            supported = {".png", ".jpg", ".jpeg", ".webp"}
            reference_paths = sorted(
                path
                for path in Path(reference_folder).iterdir()
                if path.suffix.lower() in supported
            )
            reference_embedding = ArcFaceIdentityObjective.build_reference_embedding(
                arcface_encoder,
                image_paths=reference_paths,
                face_manifest=self._identity_face_manifest,
                crop_size=self._identity_loss_config["crop_size"],
                bbox_padding_fraction=self._identity_loss_config["bbox_padding_fraction"],
                device=accelerator.device,
            )
            print(
                f"Built ArcFace reference embedding from {len(reference_paths)} images "
                f"({int(reference_embedding.numel())}-d)"
            )
            identity_objective = ArcFaceIdentityObjective(
                arcface_encoder,
                reference_embedding,
                crop_size=self._identity_loss_config["crop_size"],
                bbox_padding_fraction=self._identity_loss_config["bbox_padding_fraction"],
                max_timestep_fraction=self._identity_loss_config["max_timestep_fraction"],
                decoder_gradient_checkpointing=self._identity_loss_config[
                    "decoder_gradient_checkpointing"
                ],
                background_loss_weight=self._identity_loss_config["background_loss_weight"],
            ).to(accelerator.device)

        print("Loading unet")
        unet = self._get_unet()
        unet.requires_grad_(False)
        if not self._quantize:
            unet.to(accelerator.device)

        if self._channels_last:
            try:
                unet = unet.to(memory_format=torch.channels_last)
                print("Using channels_last memory format for the UNet")
            except (NotImplementedError, RuntimeError, ValueError) as e:
                print(f"channels_last not applied to the UNet: {e}")

        if self._gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            if self._train_text_encoder:
                text_encoder_1.gradient_checkpointing_enable()
                text_encoder_2.gradient_checkpointing_enable()

        # Prepare quantized base models for k-bit (QLoRA-style) training. This upcasts the
        # layer norms to fp32 and freezes the base weights. Gradient checkpointing is enabled
        # explicitly above (and text-encoder input grads are handled in `set_to_train`), so we
        # disable peft's checkpointing hook here, which assumes a transformers-style model.
        if self._quantize:
            unet = prepare_model_for_kbit_training(unet, use_gradient_checkpointing=False)
            if quantize_text_encoder:
                text_encoder_1 = prepare_model_for_kbit_training(
                    text_encoder_1, use_gradient_checkpointing=False
                )
                text_encoder_2 = prepare_model_for_kbit_training(
                    text_encoder_2, use_gradient_checkpointing=False
                )

        print("Adding LoRA adapters")
        unet_lora_config = LoraConfig(
            r=self._unet_lora_rank,
            use_dora=self._unet_lora_use_dora,
            lora_alpha=self._unet_lora_alpha,
            init_lora_weights=self._unet_lora_init_weights,
            target_modules=self._unet_lora_target_modules,
            rank_pattern=self._unet_lora_rank_pattern,
            alpha_pattern=self._unet_lora_alpha_pattern,
        )
        unet.add_adapter(unet_lora_config)

        if self._train_text_encoder:
            print("Adding LoRA adapters for text encoders")
            text_encoder_lora_config = LoraConfig(
                r=self._text_encoder_lora_rank,
                use_dora=self._text_encoder_lora_use_dora,
                lora_alpha=self._text_encoder_lora_alpha,
                init_lora_weights=self._text_encoder_lora_init_weights,
                target_modules=self._text_encoder_lora_target_modules,
                rank_pattern=self._text_encoder_lora_rank_pattern,
                alpha_pattern=self._text_encoder_lora_alpha_pattern,
            )
            text_encoder_1.add_adapter(text_encoder_lora_config)
            text_encoder_2.add_adapter(text_encoder_lora_config)

        def load_model_hook(models, input_dir):
            unet_to_load = None
            text_encoder_one_to_load = None
            text_encoder_two_to_load = None
            unet_type = type(unwrap_model(unet))
            text_encoder_1_type = type(unwrap_model(text_encoder_1))
            text_encoder_2_type = type(unwrap_model(text_encoder_2))

            while len(models) > 0:
                model = models.pop()
                # Unwrap `torch.compile`'s OptimizedModule so the type checks match.
                model = model._orig_mod if is_compiled_module(model) else model

                if isinstance(model, unet_type):
                    unet_to_load = model
                elif isinstance(model, text_encoder_1_type):
                    text_encoder_one_to_load = model
                elif isinstance(model, text_encoder_2_type):
                    text_encoder_two_to_load = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(input_dir)

            unet_state_dict = {
                f"{k.replace('unet.', '')}": v
                for k, v in lora_state_dict.items()
                if k.startswith("unet.")
            }
            unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
            incompatible_keys = set_peft_model_state_dict(
                unet, unet_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    print(
                        "Loading adapter weights from state_dict led to unexpected keys "
                        f"not found in the model:  {unexpected_keys}. "
                    )

            if self._train_text_encoder:
                # Do we need to call `scale_lora_layers()` here?
                _set_state_dict_into_text_encoder(
                    lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_to_load
                )

                _set_state_dict_into_text_encoder(
                    lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_to_load
                )

            self._cast_training_params_to_fp32(
                unet_to_load, text_encoder_one_to_load, text_encoder_two_to_load
            )

        unet_type = type(unwrap_model(unet))
        text_encoder_1_type = type(unwrap_model(text_encoder_1))
        text_encoder_2_type = type(unwrap_model(text_encoder_2))

        accelerator.register_save_state_pre_hook(
            partial(
                _save_model_hook, accelerator, unet_type, text_encoder_1_type, text_encoder_2_type
            )
        )
        accelerator.register_load_state_pre_hook(load_model_hook)

        self._cast_training_params_to_fp32(unet, text_encoder_1, text_encoder_2)

        if self._torch_compile:
            print(f"Compiling the UNet with torch.compile (mode={self._torch_compile_mode})")
            try:
                # Mixed-rank LoRA/DoRA shares one peft DoRA.forward across adapters
                # whose lora_A shapes differ (rank 2 vs 8 vs 32, ...). Dynamo's
                # default static parameter-shape guards recompile per shape and,
                # after recompile_limit (8), permanently fall back to eager for
                # that frame — often 2-3x slower than never compiling. Allow
                # dynamic parameter shapes and keep enough cache slots for the
                # remaining specializations (layer widths, adapter on/off).
                torch._dynamo.config.force_parameter_static_shapes = False
                torch._dynamo.config.recompile_limit = max(
                    int(getattr(torch._dynamo.config, "recompile_limit", 8)), 64
                )
                torch._dynamo.config.cache_size_limit = max(
                    int(getattr(torch._dynamo.config, "cache_size_limit", 8)), 64
                )
                unet = torch.compile(unet, mode=self._torch_compile_mode)
            except Exception as e:  # noqa: BLE001 - compile is best-effort
                print(f"torch.compile failed, continuing uncompiled: {e}")

        unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
        unet_lora_parameters_with_lr = {"params": unet_lora_parameters}
        if self._train_text_encoder:
            text_lora_parameters_one = list(
                filter(lambda p: p.requires_grad, text_encoder_1.parameters())
            )
            text_lora_parameters_two = list(
                filter(lambda p: p.requires_grad, text_encoder_2.parameters())
            )
            text_lora_parameters_one_with_lr = {
                "params": text_lora_parameters_one,
            }
            text_lora_parameters_two_with_lr = {
                "params": text_lora_parameters_two,
            }
            params_to_optimize = [
                unet_lora_parameters_with_lr,
                text_lora_parameters_one_with_lr,
                text_lora_parameters_two_with_lr,
            ]
            params_to_clip = (
                unet_lora_parameters + text_lora_parameters_one + text_lora_parameters_two
            )
        else:
            params_to_optimize = [unet_lora_parameters_with_lr]
            params_to_clip = unet_lora_parameters

        print("Loading optimizer")
        optimizer = self._get_optimizer(params_to_optimize)

        print("Loading datasets")
        # Masks are only consumed in latent space, so build them at the latent grid
        # size instead of keeping a full-resolution copy for every variant.
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        train_dataset = DreamBoothDataset(
            self._dataset_folder,
            self._instant_prompt,
            self._class_prompt,
            class_data_root=self._class_data_folder if self._with_prior_preservation else None,
            class_num=self._num_class_images,
            resolution=self._resolution,
            repeats=self._dataset_repeats,
            center_crop=self._center_crop,
            random_flip=self._random_flip,
            face_manifest=self._face_manifest,
            subject_mask_manifest=self._subject_mask_manifest,
            subject_mask_folder=self._subject_mask_folder,
            parse_manifest=self._parse_manifest,
            parse_folder=self._parse_folder,
            parse_train_lut=self._parse_train_lut,
            parse_background_lut=self._parse_background_lut,
            mask_resolution=self._resolution // vae_scale_factor,
            augmentation=self._augmentation_config,
            trust_radius=self._face_detector_config["trust_radius"],
        )
        if self._face_detector_config["trust_radius"] > 0:
            print(
                "Mask trust region: "
                f"{self._face_detector_config['trust_radius']:.2f}x the face box; "
                "subject claims outside it are anchored as background"
            )
        if self._augmentation_config["variants"] > 1:
            print(
                f"Augmentation: {self._augmentation_config['variants']} variants/image "
                f"-> {train_dataset.num_instance_images} training samples "
                f"(scale {self._augmentation_config['scale_min']:.2f}"
                f"-{self._augmentation_config['scale_max']:.2f}, "
                f"rotation=±{self._augmentation_config['rotation_degrees']:.1f}°, "
                f"flip={self._augmentation_config['random_flip']})"
            )
        if train_dataset.missing_subject_masks:
            kind = "face parse map" if self._use_face_parse else "subject mask"
            print(
                f"WARNING: {len(train_dataset.missing_subject_masks)} image(s) have no {kind} "
                "and will train unweighted: " + ", ".join(train_dataset.missing_subject_masks)
            )
        if train_dataset.missing_background_masks:
            print(
                f"WARNING: {len(train_dataset.missing_background_masks)} image(s) have no "
                "subject alpha; their background anchor falls back to the parser's own "
                "background class: " + ", ".join(train_dataset.missing_background_masks)
            )
        if (
            accelerator.is_main_process
            and self._use_pixel_masks
            and self._train_config.get("dump_face_masks", True)
        ):
            if self._use_face_parse:
                from lorakit.face_parsing import dump_dataset_parse_previews

                previews_dir = Path(self._experiment_folder) / "face_parse"
                written = dump_dataset_parse_previews(previews_dir, train_dataset)
                print(f"Wrote {written} augmented face-parse preview(s) to {previews_dir}")
            else:
                from lorakit.subject_masks import dump_dataset_mask_previews

                previews_dir = Path(self._experiment_folder) / "subject_masks"
                written = dump_dataset_mask_previews(previews_dir, train_dataset)
                print(f"Wrote {written} augmented subject-mask preview(s) to {previews_dir}")
        # `persistent_workers` avoids respawning the worker processes (and re-running
        # the dataset's expensive __init__ image preprocessing) on every epoch, which
        # otherwise causes a multi-second stall at each epoch boundary for small
        # datasets. `prefetch_factor` lets workers stay ahead of the GPU.
        dataloader_kwargs = {}
        if self._num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True
            dataloader_kwargs["prefetch_factor"] = 4
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, with_prior_preservation=self._with_prior_preservation),
            num_workers=self._num_workers,
            pin_memory=True,
            **dataloader_kwargs,
        )

        # Pre-encode all training images to latents once (see `_cache_latents`).
        latent_cache = None
        if self._cache_latents:
            if self._with_prior_preservation:
                print(
                    "cache_latents is not supported with prior preservation; "
                    "encoding latents every step"
                )
            else:
                print(f"Caching VAE latents for {train_dataset.num_instance_images} image(s)")
                latent_cache = []
                with torch.no_grad():
                    for px in train_dataset.pixel_values:
                        px = px.unsqueeze(0).to(device=accelerator.device, dtype=vae.dtype)
                        model_input = vae.encode(px).latent_dist.sample()
                        if latents_mean is None and latents_std is None:
                            model_input = model_input * vae.config.scaling_factor
                        else:
                            lm = latents_mean.to(device=model_input.device, dtype=model_input.dtype)
                            ls = latents_std.to(device=model_input.device, dtype=model_input.dtype)
                            model_input = (model_input - lm) * vae.config.scaling_factor / ls
                        latent_cache.append(model_input.squeeze(0).to(dtype=self._dtype))

        if not self._train_text_encoder:
            tokenizers = [tokenizer_1, tokenizer_2]
            text_encoders = [text_encoder_1, text_encoder_2]

            def compute_text_embeddings(prompt, text_encoders, tokenizers):
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = _encode_prompt(
                        text_encoders, tokenizers, prompt
                    )
                    prompt_embeds = prompt_embeds.to(accelerator.device)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                return prompt_embeds, pooled_prompt_embeds

        # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
        # provided (i.e. the --instance_prompt is used for all images), we encode the instance
        # prompt once to avoid the redundant encoding.
        if not self._train_text_encoder:
            instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(
                self._instant_prompt, text_encoders, tokenizers
            )

        # Handle class prompt for prior-preservation.
        if self._with_prior_preservation:
            if not self._train_text_encoder:
                class_prompt_hidden_states, class_pooled_prompt_embeds = compute_text_embeddings(
                    self._class_prompt, text_encoders, tokenizers
                )

        # Clear the memory here
        if not self._train_text_encoder and not train_dataset.custom_instance_prompts:
            del tokenizers, text_encoders
            _flush()

        # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all
        # images), pack the statically computed variables appropriately here. This is so that we
        # don't have to pass them to the dataloader.

        if not train_dataset.custom_instance_prompts:
            if not self._train_text_encoder:
                prompt_embeds = instance_prompt_hidden_states
                unet_add_text_embeds = instance_pooled_prompt_embeds
                if self._with_prior_preservation:
                    prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
                    unet_add_text_embeds = torch.cat(
                        [unet_add_text_embeds, class_pooled_prompt_embeds], dim=0
                    )
            # if we're optmizing the text encoder (both if instance prompt is used for all
            # images or custom prompts) we need to tokenize and encode the batch prompts on
            # all training steps
            else:
                tokens_one = _tokenize_prompt(tokenizer_1, self._instance_prompt)
                tokens_two = _tokenize_prompt(tokenizer_2, self._instance_prompt)
                if self._with_prior_preservation:
                    class_tokens_one = _tokenize_prompt(tokenizer_1, self._class_prompt)
                    class_tokens_two = _tokenize_prompt(tokenizer_2, self._class_prompt)
                    tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                    tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self._gradient_accumulation_steps
        )
        if self._max_train_steps is None:
            self._max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        print("Loading lr scheduler")
        lr_scheduler = get_scheduler(
            self._lr_scheduler, optimizer=optimizer, **self._lr_scheduler_params
        )

        # Prepare everything with our `accelerator`.
        if self._train_text_encoder:
            unet, text_encoder_1, text_encoder_2, optimizer, train_dataloader, lr_scheduler = (
                accelerator.prepare(
                    unet, text_encoder_1, text_encoder_2, optimizer, train_dataloader, lr_scheduler
                )
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

        # Rebuild the trainable list after prepare so EMA / grad-clip track the
        # live Parameter objects (DDP / compile wrappers included).
        params_to_clip = list(filter(lambda p: p.requires_grad, unet.parameters()))
        if self._train_text_encoder:
            params_to_clip.extend(filter(lambda p: p.requires_grad, text_encoder_1.parameters()))
            params_to_clip.extend(filter(lambda p: p.requires_grad, text_encoder_2.parameters()))
        if self._use_ema:
            self._ema = LoRAEMA(params_to_clip, decay=self._ema_decay)

        # We need to recalculate our total training steps as the size of the training
        # dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self._gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            self._max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.num_train_epochs = math.ceil(self._max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            tracker_name = (
                "dreambooth-lora-sd-xl"
                if "playground" not in self._model_name_or_path
                else "dreambooth-lora-playground"
            )
            accelerator.init_trackers(tracker_name, config=vars(self))

        print("Training ...")

        global_step = 0
        first_epoch = 0
        ema_loss = None
        last_saved_step = -1
        last_sampled_step = -1

        # Potentially load in the weights and states from a previous save
        if self._resume_from_checkpoint:
            if self._resume_from_checkpoint != "latest":
                path = os.path.basename(self._resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self._experiment_folder)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("_")[-1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{self._resume_from_checkpoint}' does not exist. "
                    "Starting a new training run."
                )
                self._resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                checkpoint_dir = os.path.join(self._experiment_folder, path)
                accelerator.load_state(checkpoint_dir)
                self._load_ema_state(checkpoint_dir)
                global_step = int(path.split("_")[-1])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch

        else:
            initial_global_step = 0

        if not self._disable_sampling and not self._skip_pre_train_sample:
            self._generate_samples_with_optimizer_eval(
                accelerator, global_step, vae, unet, text_encoder_1, text_encoder_2, optimizer
            )

        progress_bar = tqdm(
            range(0, self._max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        profiler = StepProfiler(
            enabled=self._profile_enabled,
            warmup=self._profile_warmup,
            report_every=self._profile_report_every,
        )
        for _ in range(first_epoch, self.num_train_epochs):
            self.set_to_train(accelerator, text_encoder_1, text_encoder_2, unet, optimizer)

            profiler.wall_start("data_wait")
            for _step, batch in enumerate(train_dataloader):
                profiler.wall_stop("data_wait")
                with accelerator.accumulate(unet):
                    profiler.tic("vae_encode")
                    prompts = batch["prompts"]

                    # encode batch prompts when custom prompts are provided for each image -
                    if train_dataset.custom_instance_prompts:
                        if not self._train_text_encoder:
                            prompt_embeds, unet_add_text_embeds = compute_text_embeddings(
                                prompts, text_encoders, tokenizers
                            )
                        else:
                            tokens_one = _tokenize_prompt(tokenizer_1, prompts)
                            tokens_two = _tokenize_prompt(tokenizer_2, prompts)

                    if latent_cache is not None:
                        # Reuse the pre-computed latents instead of running the VAE.
                        model_input = torch.stack([latent_cache[i] for i in batch["indices"]]).to(
                            device=accelerator.device, dtype=self._dtype
                        )
                    else:
                        pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                        # Convert images to latent space
                        model_input = vae.encode(pixel_values).latent_dist.sample()

                        if latents_mean is None and latents_std is None:
                            model_input = model_input * vae.config.scaling_factor
                            model_input = model_input.to(self._dtype)
                        else:
                            latents_mean = latents_mean.to(
                                device=model_input.device, dtype=model_input.dtype
                            )
                            latents_std = latents_std.to(
                                device=model_input.device, dtype=model_input.dtype
                            )
                            model_input = (
                                (model_input - latents_mean)
                                * vae.config.scaling_factor
                                / latents_std
                            )
                            model_input = model_input.to(dtype=self._dtype)
                    profiler.toc("vae_encode")

                    profiler.tic("prep")
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]

                    # Sample a random timestep for each image, restricted to the
                    # configured [min_fraction, max_fraction] band of the schedule.
                    # Smaller indices are lower noise: capping the max keeps the base
                    # model's global composition and only trains identity at low noise.
                    num_train_timesteps = noise_scheduler.config.num_train_timesteps
                    timestep_low = int(self._timestep_range["min_fraction"] * num_train_timesteps)
                    timestep_high = max(
                        timestep_low + 1,
                        int(self._timestep_range["max_fraction"] * num_train_timesteps),
                    )
                    if not self._do_edm_style_training:
                        timesteps = torch.randint(
                            timestep_low,
                            timestep_high,
                            (bsz,),
                            device=model_input.device,
                        )
                        timesteps = timesteps.long()
                    else:
                        # in EDM formulation, the model is conditioned on the pre-conditioned
                        # noise levels instead of discrete timesteps, so here we sample indices
                        # to get the noise levels from `scheduler.timesteps`
                        indices = torch.randint(timestep_low, timestep_high, (bsz,))
                        timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                    # For EDM-style training, we first obtain the sigmas based on the continuous
                    # timesteps. We then precondition the final model inputs based on these
                    # sigmas instead of the timesteps. Follow: Section 5 of
                    # https://arxiv.org/abs/2206.00364.
                    if self._do_edm_style_training:
                        sigmas = _get_sigmas(
                            accelerator,
                            noise_scheduler,
                            timesteps,
                            len(noisy_model_input.shape),
                            noisy_model_input.dtype,
                        )
                        if is_edm_scheduler:
                            inp_noisy_latents = noise_scheduler.precondition_inputs(
                                noisy_model_input, sigmas
                            )
                        else:
                            inp_noisy_latents = noisy_model_input / ((sigmas**2 + 1) ** 0.5)

                    # time ids
                    add_time_ids = torch.cat(
                        [
                            _compute_time_ids(
                                original_size=s,
                                crops_coords_top_left=c,
                                resolution=self._resolution,
                                device=accelerator.device,
                                dtype=self._dtype,
                            )
                            for s, c in zip(
                                batch["original_sizes"], batch["crop_top_lefts"], strict=False
                            )
                        ]
                    )

                    # Calculate the elements to repeat depending on the use of prior-preservation
                    # and custom captions.
                    if not train_dataset.custom_instance_prompts:
                        elems_to_repeat_text_embeds = (
                            bsz // 2 if self._with_prior_preservation else bsz
                        )
                    else:
                        elems_to_repeat_text_embeds = 1
                    profiler.toc("prep")

                    # Predict the noise residual
                    profiler.tic("unet_forward")
                    if not self._train_text_encoder:
                        unet_added_conditions = {
                            "time_ids": add_time_ids,
                            "text_embeds": unet_add_text_embeds.repeat(
                                elems_to_repeat_text_embeds, 1
                            ),
                        }
                        prompt_embeds_input = prompt_embeds.repeat(
                            elems_to_repeat_text_embeds, 1, 1
                        )
                        model_pred = unet(
                            inp_noisy_latents if self._do_edm_style_training else noisy_model_input,
                            timesteps,
                            prompt_embeds_input,
                            added_cond_kwargs=unet_added_conditions,
                            return_dict=False,
                        )[0]
                    else:
                        unet_added_conditions = {"time_ids": add_time_ids}
                        prompt_embeds, pooled_prompt_embeds = _encode_prompt(
                            text_encoders=[text_encoder_1, text_encoder_2],
                            tokenizers=None,
                            prompt=None,
                            text_input_ids_list=[tokens_one, tokens_two],
                        )
                        unet_added_conditions.update(
                            {
                                "text_embeds": pooled_prompt_embeds.repeat(
                                    elems_to_repeat_text_embeds, 1
                                )
                            }
                        )
                        prompt_embeds_input = prompt_embeds.repeat(
                            elems_to_repeat_text_embeds, 1, 1
                        )
                        model_pred = unet(
                            inp_noisy_latents if self._do_edm_style_training else noisy_model_input,
                            timesteps,
                            prompt_embeds_input,
                            added_cond_kwargs=unet_added_conditions,
                            return_dict=False,
                        )[0]

                    base_model_pred = None
                    if background_objective is not None:
                        instance_batch_size = bsz // 2 if self._with_prior_preservation else bsz
                        teacher_unet = unwrap_model(unet)
                        teacher_models = [teacher_unet]
                        if self._train_text_encoder:
                            teacher_models.extend(
                                [unwrap_model(text_encoder_1), unwrap_model(text_encoder_2)]
                            )
                        training_states = [model.training for model in teacher_models]
                        for model in teacher_models:
                            model.eval()
                        try:
                            with torch.no_grad(), _adapters_disabled(*teacher_models):
                                teacher_input = (
                                    inp_noisy_latents
                                    if self._do_edm_style_training
                                    else noisy_model_input
                                )[:instance_batch_size]
                                teacher_time_ids = add_time_ids[:instance_batch_size]
                                if self._train_text_encoder:
                                    teacher_tokens_one = tokens_one
                                    teacher_tokens_two = tokens_two
                                    if train_dataset.custom_instance_prompts:
                                        teacher_tokens_one = teacher_tokens_one[
                                            :instance_batch_size
                                        ]
                                        teacher_tokens_two = teacher_tokens_two[
                                            :instance_batch_size
                                        ]
                                    else:
                                        teacher_tokens_one = teacher_tokens_one[:1]
                                        teacher_tokens_two = teacher_tokens_two[:1]
                                    teacher_prompt_embeds, teacher_pooled_prompt_embeds = (
                                        _encode_prompt(
                                            text_encoders=teacher_models[1:],
                                            tokenizers=None,
                                            prompt=None,
                                            text_input_ids_list=[
                                                teacher_tokens_one,
                                                teacher_tokens_two,
                                            ],
                                        )
                                    )
                                    if not train_dataset.custom_instance_prompts:
                                        teacher_prompt_embeds = teacher_prompt_embeds.repeat(
                                            instance_batch_size, 1, 1
                                        )
                                        teacher_pooled_prompt_embeds = (
                                            teacher_pooled_prompt_embeds.repeat(
                                                instance_batch_size, 1
                                            )
                                        )
                                    teacher_added_conditions = {
                                        "time_ids": teacher_time_ids,
                                        "text_embeds": teacher_pooled_prompt_embeds,
                                    }
                                else:
                                    teacher_prompt_embeds = prompt_embeds_input[
                                        :instance_batch_size
                                    ]
                                    teacher_added_conditions = {
                                        key: value[:instance_batch_size]
                                        if isinstance(value, torch.Tensor) and value.shape[0] == bsz
                                        else value
                                        for key, value in unet_added_conditions.items()
                                    }
                                base_model_pred = teacher_unet(
                                    teacher_input,
                                    timesteps[:instance_batch_size],
                                    teacher_prompt_embeds,
                                    added_cond_kwargs=teacher_added_conditions,
                                    return_dict=False,
                                )[0]
                        finally:
                            for model, was_training in zip(
                                teacher_models, training_states, strict=True
                            ):
                                model.train(was_training)
                    profiler.toc("unet_forward")

                    profiler.tic("loss")
                    weighting = None
                    prior_loss = torch.zeros((), device=model_pred.device, dtype=torch.float32)
                    background_loss = torch.zeros((), device=model_pred.device, dtype=torch.float32)
                    identity_loss = torch.zeros((), device=model_pred.device, dtype=torch.float32)
                    background_stats = {
                        "eligible_timesteps": 0.0,
                        "valid_faces": 0.0,
                        "skipped_faces": 0.0,
                        "face_coverage": 0.0,
                        "background_coverage": 0.0,
                        "background_mse": 0.0,
                    }
                    face_focus_stats = {
                        "eligible_timesteps": 0.0,
                        "valid_faces": 0.0,
                        "face_coverage": 1.0,
                    }
                    identity_stats = {
                        "eligible_timesteps": 0.0,
                        "valid_faces": 0.0,
                        "skipped_faces": 0.0,
                        "clipped_boxes": 0.0,
                        "identity_cosine": 0.0,
                        "identity_loss": 0.0,
                        "background_mse": 0.0,
                    }
                    effective_background_weight = float(
                        self._background_preservation_config["weight"]
                    )
                    if self._do_edm_style_training:
                        # Similar to the input preconditioning, the model predictions are also
                        # preconditioned on noised model inputs (before preconditioning) and the
                        # sigmas. Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                        if is_edm_scheduler:
                            model_pred = noise_scheduler.precondition_outputs(
                                noisy_model_input, model_pred, sigmas
                            )
                        else:
                            if noise_scheduler.config.prediction_type == "epsilon":
                                model_pred = model_pred * (-sigmas) + noisy_model_input
                            elif noise_scheduler.config.prediction_type == "v_prediction":
                                model_pred = model_pred * (-sigmas / (sigmas**2 + 1) ** 0.5) + (
                                    noisy_model_input / (sigmas**2 + 1)
                                )
                        # We are not doing weighting here because it tends result in numerical
                        # problems.
                        # See: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
                        # There might be other alternatives for weighting as well:
                        # https://github.com/huggingface/diffusers/pull/7126#discussion_r1505404686
                        if not is_edm_scheduler:
                            weighting = (sigmas**-2.0).float()

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = model_input if self._do_edm_style_training else noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = (
                            model_input
                            if self._do_edm_style_training
                            else noise_scheduler.get_velocity(model_input, noise, timesteps)
                        )
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        )

                    # Timesteps aligned with the (possibly chunked) instance loss.
                    # With prior preservation the batch is [instance, class]; the
                    # instance loss below is computed on the first half only, so the
                    # SNR weighting must use the first half's timesteps too (otherwise
                    # the instance gradient gets rescaled by the class image's
                    # unrelated timestep weight every step).
                    loss_timesteps = timesteps
                    if self._with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on
                        # each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)
                        loss_timesteps = torch.chunk(timesteps, 2, dim=0)[0]

                        # Compute prior loss
                        if weighting is not None:
                            prior_loss = torch.mean(
                                (
                                    weighting.float()
                                    * (model_pred_prior.float() - target_prior.float()) ** 2
                                ).reshape(target_prior.shape[0], -1),
                                1,
                            )
                            prior_loss = prior_loss.mean()
                        else:
                            prior_loss = F.mse_loss(
                                model_pred_prior.float(), target_prior.float(), reduction="mean"
                            )

                    # Face-focused identity loss: a detached soft face mask (raised to
                    # `power`, where lower power pulls every weight toward 1) reweights the
                    # diffusion loss so the LoRA learns mostly inside the face box. Rows
                    # without a usable face fall back to an all-ones weight (unweighted).
                    # Offline face boxes for the instance rows. Normalized [0, 1]
                    # boxes feed the latent soft-mask rasterizer; pixel boxes feed
                    # ArcFace roi_align on the decoded image.
                    # Invalid rows keep the dataset's negative sentinel and stay
                    # negative after the division, so both objectives skip them.
                    instance_face_boxes_px = batch["instance_face_boxes"][: model_pred.shape[0]].to(
                        device=accelerator.device, dtype=torch.float32
                    )
                    instance_face_box_clipped = batch["instance_face_box_clipped"][
                        : model_pred.shape[0]
                    ].to(device=accelerator.device)
                    manifest_face_boxes = None
                    if self._use_manifest_face_boxes and background_objective is not None:
                        manifest_face_boxes = instance_face_boxes_px / float(self._resolution)

                    # Per-pixel weights for the instance rows. When present they
                    # replace the face rectangle as the trained region, so the
                    # preserved area is the true background rather than everything
                    # outside a box around the face.  The anchor weights travel as
                    # their own channel because they are not the complement of the
                    # trained region: zoom-out padding and (with a parse map)
                    # clothing belong to neither.
                    subject_masks = None
                    subject_mask_valid = None
                    background_masks = None
                    if self._use_pixel_masks and background_objective is not None:
                        subject_masks = batch["instance_subject_masks"][: model_pred.shape[0]].to(
                            device=accelerator.device, dtype=torch.float32
                        )
                        subject_mask_valid = batch["instance_subject_mask_valid"][
                            : model_pred.shape[0]
                        ].to(device=accelerator.device)
                        background_masks = batch["instance_background_masks"][
                            : model_pred.shape[0]
                        ].to(device=accelerator.device, dtype=torch.float32)

                    face_weight = None
                    if self._face_focus_enabled and background_objective is not None:
                        face_mask, face_focus_stats = background_objective.face_focus_mask(
                            noisy_instance_latents=noisy_model_input[: model_pred.shape[0]],
                            base_model_prediction=base_model_pred,
                            timesteps=loss_timesteps,
                            scheduler=noise_scheduler,
                            num_train_timesteps=noise_scheduler.config.num_train_timesteps,
                            height=model_pred.shape[-2],
                            width=model_pred.shape[-1],
                            face_boxes=manifest_face_boxes,
                            subject_mask=subject_masks,
                            subject_mask_valid=subject_mask_valid,
                        )
                        # Mask power is fixed: flattening it would shift loss weight
                        # onto the background instead of easing identity pressure.
                        face_weight = face_mask.float().clamp(0, 1).pow(self._face_focus_power)

                    if self._snr_gamma is None:
                        if weighting is not None:
                            squared_error = (
                                weighting.float() * (model_pred.float() - target.float()) ** 2
                            )
                            if face_weight is not None:
                                loss = _masked_per_row_mean(squared_error, face_weight).mean()
                            else:
                                loss = torch.mean(
                                    squared_error.reshape(target.shape[0], -1), 1
                                ).mean()
                        else:
                            squared_error = (model_pred.float() - target.float()) ** 2
                            if face_weight is not None:
                                loss = _masked_per_row_mean(squared_error, face_weight).mean()
                            else:
                                loss = squared_error.mean()
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is
                        # slightly changed. This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, loss_timesteps)
                        base_weight = (
                            torch.stack([snr, self._snr_gamma * torch.ones_like(snr)], dim=1).min(
                                dim=1
                            )[0]
                            / snr
                        )

                        if noise_scheduler.config.prediction_type == "v_prediction":
                            # Velocity objective needs to be floored to an SNR weight of one.
                            mse_loss_weights = base_weight + 1
                        else:
                            # Epsilon and sample both use the same loss weights.
                            mse_loss_weights = base_weight

                        squared_error = F.mse_loss(
                            model_pred.float(), target.float(), reduction="none"
                        )
                        if face_weight is not None:
                            per_row = _masked_per_row_mean(squared_error, face_weight)
                        else:
                            per_row = squared_error.mean(
                                dim=list(range(1, len(squared_error.shape)))
                            )
                        loss = (per_row * mse_loss_weights).mean()

                    instance_diffusion_loss = loss
                    if self._with_prior_preservation:
                        # Add the prior loss to the instance loss.
                        loss = loss + self._prior_loss_weight * prior_loss

                    if background_objective is not None:
                        if base_model_pred is None:
                            raise RuntimeError(
                                "base-model background teacher prediction is missing"
                            )
                        # The background-preservation term (match the base model outside the
                        # face) is only added when explicitly enabled.
                        if self._background_preservation_enabled:
                            background_loss, background_stats = background_objective(
                                noisy_instance_latents=noisy_model_input[: model_pred.shape[0]],
                                model_prediction=model_pred,
                                base_model_prediction=base_model_pred,
                                timesteps=loss_timesteps,
                                scheduler=noise_scheduler,
                                num_train_timesteps=noise_scheduler.config.num_train_timesteps,
                                face_boxes=manifest_face_boxes,
                                subject_mask=subject_masks,
                                subject_mask_valid=subject_mask_valid,
                                background_mask=background_masks,
                            )
                            background_weight = self._background_preservation_config["weight"]
                            effective_background_weight = background_weight
                            loss = loss + background_weight * background_loss

                    if identity_objective is not None:
                        identity_loss, identity_stats = identity_objective(
                            noisy_instance_latents=noisy_model_input[: model_pred.shape[0]],
                            model_prediction=model_pred,
                            timesteps=loss_timesteps,
                            face_boxes=instance_face_boxes_px,
                            face_box_clipped=instance_face_box_clipped,
                            vae=vae,
                            alphas_cumprod=noise_scheduler.alphas_cumprod,
                            prediction_type=noise_scheduler.config.prediction_type,
                            latent_scaling_factor=vae.config.scaling_factor,
                            latents_mean=latents_mean,
                            latents_std=latents_std,
                            num_train_timesteps=noise_scheduler.config.num_train_timesteps,
                            pixel_values=batch["pixel_values"][: model_pred.shape[0]].to(
                                device=accelerator.device, dtype=torch.float32
                            ),
                        )
                        loss = loss + self._identity_loss_config["weight"] * identity_loss

                    profiler.toc("loss")

                    profiler.tic("backward")
                    accelerator.backward(loss)
                    gradient_norm = None
                    if accelerator.sync_gradients and self._max_grad_norm is not None:
                        gradient_norm = accelerator.clip_grad_norm_(
                            params_to_clip, self._max_grad_norm
                        )
                    elif accelerator.sync_gradients:
                        gradient_norm = _gradient_norm(params_to_clip)
                    profiler.toc("backward")

                    profiler.tic("optimizer")
                    optimizer.step()
                    if not self._is_schedulefree_optimizer():
                        lr_scheduler.step()
                    optimizer.zero_grad()
                    profiler.toc("optimizer")

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if self._ema is not None:
                        self._ema.update()
                    progress_bar.update(1)
                    global_step += 1
                    progress_callback = next(
                        (cb for cb in callbacks if isinstance(cb, ProgressCallback)), None
                    )
                    if progress_callback:
                        progress_callback.set_progress(global_step / self._max_train_steps)

                if ema_loss is None:
                    ema_loss = loss.detach().item()
                else:
                    ema_loss = (
                        self._loss_decay * ema_loss + (1 - self._loss_decay) * loss.detach().item()
                    )

                loss_callback = next((cb for cb in callbacks if isinstance(cb, LossCallback)), None)
                if loss_callback:
                    loss_callback.set_loss(ema_loss)

                logs = {
                    "loss": ema_loss,
                    "instance_diffusion_loss": instance_diffusion_loss.detach().item(),
                    "prior_loss": prior_loss.detach().item(),
                    "background_mse": background_stats["background_mse"],
                    "weighted_background_preservation": (
                        effective_background_weight * background_loss.detach().item()
                    ),
                    "background_eligible_timesteps": background_stats["eligible_timesteps"],
                    "background_valid_faces": background_stats["valid_faces"],
                    "background_skipped_faces": background_stats["skipped_faces"],
                    "background_face_coverage": background_stats["face_coverage"],
                    "background_anchor_coverage": background_stats["background_coverage"],
                    "face_focus_eligible_timesteps": face_focus_stats["eligible_timesteps"],
                    "face_focus_valid_faces": face_focus_stats["valid_faces"],
                    "face_focus_face_coverage": face_focus_stats["face_coverage"],
                    "identity_loss": identity_stats["identity_loss"],
                    "weighted_identity_loss": (
                        self._identity_loss_config["weight"] * identity_loss.detach().item()
                    ),
                    "identity_cosine": identity_stats["identity_cosine"],
                    "identity_eligible_timesteps": identity_stats["eligible_timesteps"],
                    "identity_valid_faces": identity_stats["valid_faces"],
                    "identity_skipped_faces": identity_stats["skipped_faces"],
                    "lr": self._get_logged_lr(optimizer, lr_scheduler),
                }
                if gradient_norm is not None:
                    logs["gradient_norm"] = gradient_norm.detach().item()
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process:
                    if (
                        last_saved_step < global_step
                        and self._save_every > 0
                        and global_step > 0
                        and global_step % self._save_every == 0
                    ):
                        last_saved_step = global_step
                        if self._checkpoints_total_limit is not None:
                            checkpoints = os.listdir(self._experiment_folder)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))

                            # before we save the new checkpoint, we need to have at _most_
                            # `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= self._checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - self._checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                print(
                                    f"{len(checkpoints)} checkpoints already exist, "
                                    f"removing {len(removing_checkpoints)} checkpoints"
                                )
                                print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        self._experiment_folder, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            self._experiment_folder,
                            f"checkpoint_{self._name}_{self._version}_{global_step:010d}",
                        )
                        self._optimizer_eval(optimizer)
                        try:
                            # Checkpoint the live training weights for optimizer
                            # continuity; EMA shadows are stored beside them.
                            accelerator.save_state(save_path)
                            self._save_ema_state(save_path)
                            print(f"Saved state to {save_path}")
                        finally:
                            self._optimizer_train(optimizer)

                    if (
                        last_sampled_step < global_step
                        and global_step > 0
                        and not self._disable_sampling
                        and self._sample_config is not None
                        and global_step % self._sample_every == 0
                    ):
                        last_sampled_step = global_step
                        self._generate_samples_with_optimizer_eval(
                            accelerator,
                            global_step,
                            vae,
                            unet,
                            text_encoder_1,
                            text_encoder_2,
                            optimizer,
                        )

                profiler.step_end()
                profiler.wall_start("data_wait")

                if global_step >= self._max_train_steps:
                    print("Max training steps reached")
                    break

        profiler.report()

        # Save the lora layers
        accelerator.wait_for_everyone()
        self._optimizer_eval(optimizer)
        if accelerator.is_main_process:
            # Export EMA weights when enabled — same tensors used for sampling.
            ema_context = (
                self._ema.average_parameters()
                if self._ema is not None
                else contextlib.nullcontext()
            )
            with ema_context:
                unet = unwrap_model(unet)
                # A quantized (bitsandbytes) base model cannot be cast with `.to(dtype)`.
                # The trainable LoRA params are already kept in fp32 (see
                # `_cast_training_params_to_fp32`), so the cast is only needed for the
                # non-quantized case.
                if not self._quantize:
                    unet = unet.to(torch.float32)
                unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

                if self._train_text_encoder:
                    text_encoder_1 = unwrap_model(text_encoder_1)
                    text_encoder_lora_layers = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(text_encoder_1.to(torch.float32))
                    )
                    text_encoder_2 = unwrap_model(text_encoder_2)
                    text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(text_encoder_2.to(torch.float32))
                    )
                else:
                    text_encoder_lora_layers = None
                    text_encoder_2_lora_layers = None

                StableDiffusionXLPipeline.save_lora_weights(
                    save_directory=self._experiment_folder,
                    unet_lora_layers=unet_lora_layers,
                    text_encoder_lora_layers=text_encoder_lora_layers,
                    text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                )
            self._print_best_sample_arcface_step()

        accelerator.end_training()

    def _is_schedulefree_optimizer(self) -> bool:
        return self._optimizer_name in SCHEDULEFREE_OPTIMIZERS

    def _optimizer_train(self, optimizer) -> None:
        if self._is_schedulefree_optimizer() and hasattr(optimizer, "train"):
            optimizer.train()

    def _optimizer_eval(self, optimizer) -> None:
        if self._is_schedulefree_optimizer() and hasattr(optimizer, "eval"):
            optimizer.eval()

    def _get_logged_lr(self, optimizer, lr_scheduler) -> float:
        if self._is_schedulefree_optimizer():
            param_groups = getattr(optimizer, "param_groups", None)
            if param_groups and "scheduled_lr" in param_groups[0]:
                return param_groups[0]["scheduled_lr"]
        return lr_scheduler.get_last_lr()[0]

    def _generate_samples_with_optimizer_eval(
        self, accelerator, global_step, vae, unet, text_encoder_one, text_encoder_two, optimizer
    ):
        self._optimizer_eval(optimizer)
        try:
            # Pipeline shares the live UNet / text-encoder modules, so install
            # EMA adapter weights for the duration of sampling when enabled.
            ema_context = (
                self._ema.average_parameters()
                if self._ema is not None
                else contextlib.nullcontext()
            )
            with ema_context:
                self._generate_samples(
                    accelerator, global_step, vae, unet, text_encoder_one, text_encoder_two
                )
        finally:
            self._optimizer_train(optimizer)

    def set_to_train(self, accelerator, text_encoder_1, text_encoder_2, unet, optimizer):
        unet.train()
        self._optimizer_train(optimizer)
        if self._train_text_encoder:
            text_encoder_1.train()
            text_encoder_2.train()

            # set top parameter requires_grad = True for gradient checkpointing works
            accelerator.unwrap_model(text_encoder_1).text_model.embeddings.requires_grad_(True)
            accelerator.unwrap_model(text_encoder_2).text_model.embeddings.requires_grad_(True)
