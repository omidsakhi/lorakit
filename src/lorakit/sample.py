"""Batch SDXL and SDXL Turbo inference from a JSON prompt file."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm

from lorakit.config import resolve_user_path
from lorakit.jobs import BaseJob
from lorakit.lora_scaling import (
    apply_training_lora_scaling,
    find_training_target_modules,
    format_scaling_report,
)
from lorakit.models import get_model_type, make_scheduler, sample_defaults
from lorakit.prompts import load_prompts, sample_prompts


def _flush():
    torch.cuda.empty_cache()
    gc.collect()


def _parse_dtype(dtype_str: str) -> tuple[torch.dtype, str]:
    dtype_str = dtype_str.lower()
    if dtype_str in ("bfloat16", "bf16"):
        return torch.bfloat16, "bf16"
    if dtype_str in ("fp16", "float16"):
        return torch.float16, "fp16"
    if dtype_str in ("float32", "fp32"):
        return torch.float32, "no"
    raise ValueError("Invalid dtype. Supported dtypes are bf16, fp16, fp32, and float32.")


def _resolve_lora_weights(path: str | Path) -> Path:
    """Return a folder or safetensors path suitable for ``load_lora_weights``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"lora_weights not found: {p}")
    if p.is_file():
        return p
    for name in ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin"):
        if (p / name).exists():
            return p
    raise FileNotFoundError(
        f"lora_weights folder {p} does not contain pytorch_lora_weights.safetensors"
    )


class SampleJob(BaseJob):
    def __init__(self, config, version, name, root_folder, *, config_path=None):
        super().__init__(version, name, root_folder)

        self._device = config.get("device", "cpu")
        if torch.cuda.is_available():
            if self._device == "cpu":
                print(
                    "Warning: CUDA is available, but CPU is being used. "
                    "Consider using a GPU for faster inference."
                )
        elif self._device != "cpu":
            raise RuntimeError(
                "CUDA is not available, but GPU device was specified. "
                "Use 'cpu' or run on a machine with CUDA."
            )
        print(f"Using device: {self._device}")

        self._allow_tf32 = config.get("allow_tf32", False)
        if self._allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        sample_config = config.get("sample", None)
        if sample_config is None:
            raise ValueError("sample is required")

        prompt_file = sample_config.get("prompt_file", None)
        if not prompt_file:
            raise ValueError("sample.prompt_file is required")
        self._prompt_file = resolve_user_path(prompt_file, config_path=config_path, must_exist=True)

        self._num_prompts = sample_config.get("num_prompts", None)
        # Only controls which prompts are selected from the file, not generation seeds.
        self._sampling_seed = sample_config.get("sampling_seed", 42)
        self._resolution = sample_config.get("resolution", 1024)
        self._scheduler = sample_config.get("scheduler", None)

        lora_weights = sample_config.get("lora_weights", None)
        self._lora_weights = (
            resolve_user_path(lora_weights, config_path=config_path, must_exist=True)
            if lora_weights
            else None
        )
        self._compare_base = sample_config.get("compare_base", True)

        model_config = config.get("model", None)
        if model_config is None:
            raise ValueError("model is required")
        self._model_name_or_path = model_config.get("name_or_path", None)
        if not self._model_name_or_path:
            raise ValueError("model.name_or_path is required")
        self._revision = model_config.get("revision", None)
        self._local_files_only = model_config.get("local_files_only", False)
        self._variant = model_config.get("variant", None)
        self._model_type = get_model_type(model_config)

        default_guidance_scale, default_steps = sample_defaults(self._model_type)
        self._guidance_scale = sample_config.get("guidance_scale", default_guidance_scale)
        self._sample_steps = sample_config.get(
            "steps", sample_config.get("sample_steps", default_steps)
        )

        dtype_str = config.get("dtype", sample_config.get("dtype", None))
        if dtype_str is None:
            raise ValueError("dtype is required (top-level or under sample)")
        self._dtype, _ = _parse_dtype(dtype_str)

    def _variants(self) -> list[str]:
        if self._lora_weights is None:
            return ["base"]
        if self._compare_base:
            return ["base", "lora"]
        return ["lora"]

    @staticmethod
    def _generation_seed(prompt_seed: int) -> int:
        """Use the dataset seed for torch.Generator (same as perturbation-probe)."""
        return int(prompt_seed)

    def run(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        all_prompts = load_prompts(self._prompt_file)
        prompts = sample_prompts(all_prompts, self._num_prompts, seed=self._sampling_seed)
        print(
            f"Loaded {len(all_prompts)} prompts from {self._prompt_file}, "
            f"running {len(prompts)} (sampling_seed={self._sampling_seed}, "
            f"generation seed=per-prompt from JSON)"
        )

        self._experiment_folder.mkdir(parents=True, exist_ok=True)
        manifest_path = self._experiment_folder / "manifest.jsonl"

        print(f"Loading pipeline: {self._model_name_or_path}")
        print(f"Model type: {self._model_type}")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._model_name_or_path,
            revision=self._revision,
            variant=self._variant,
            torch_dtype=self._dtype,
            use_safetensors=True,
            local_files_only=self._local_files_only,
        )
        pipeline.set_progress_bar_config(disable=True)
        pipeline = pipeline.to(self._device)
        if self._scheduler is not None:
            pipeline.scheduler = make_scheduler(pipeline.scheduler, self._scheduler)
            print(f"Scheduler: {self._scheduler}")

        lora_path = None
        if self._lora_weights is not None:
            lora_path = _resolve_lora_weights(self._lora_weights)
            print(f"LoRA weights: {lora_path}")

        variants = self._variants()
        print(f"Variants: {', '.join(variants)}")

        records: list[dict] = []
        for variant in variants:
            out_dir = self._experiment_folder / variant
            out_dir.mkdir(parents=True, exist_ok=True)

            if variant == "lora":
                pipeline.load_lora_weights(str(lora_path))
                # Exported LoRA weights carry no per-module alpha, so PEFT
                # reloads them at the wrong strength when ranks differ.
                targets = find_training_target_modules(lora_path, self._lora_weights)
                if targets:
                    report = apply_training_lora_scaling(pipeline, targets)
                    print(f"LoRA scaling restored ({format_scaling_report(report)})")
                else:
                    print(
                        "WARNING: no training config.yaml found next to the LoRA "
                        "weights; adapter strength may be weaker than training."
                    )

            for prompt in tqdm(prompts, desc=f"sample/{variant}"):
                gen_seed = self._generation_seed(prompt.seed)
                generator = torch.Generator(device=self._device).manual_seed(gen_seed)
                image = pipeline(
                    prompt=prompt.pos,
                    negative_prompt=prompt.neg or None,
                    height=self._resolution,
                    width=self._resolution,
                    guidance_scale=self._guidance_scale,
                    num_inference_steps=self._sample_steps,
                    generator=generator,
                ).images[0]

                safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt.id)
                image_path = out_dir / f"{safe_id}.jpg"
                image.save(image_path)
                records.append(
                    {
                        "variant": variant,
                        "id": prompt.id,
                        "pos": prompt.pos,
                        "neg": prompt.neg,
                        "seed": gen_seed,
                        "prompt_seed": prompt.seed,
                        "image": str(image_path.relative_to(self._experiment_folder)),
                    }
                )

        with manifest_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Saved {len(records)} images under {self._experiment_folder}")
        print(f"Manifest: {manifest_path}")

        del pipeline
        _flush()
