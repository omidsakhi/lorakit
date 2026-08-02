"""Creativity retention sweep: same prompts, random guidance / steps / scheduler."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm

from lorakit.lora_scaling import (
    apply_training_lora_scaling,
    format_scaling_report,
)
from lorakit.models import make_scheduler
from lorakit.prompts import load_training_sample_prompts
from lorakit.sample import _flush, _parse_dtype, _resolve_lora_weights

DEFAULT_SCHEDULERS = ("ddim", "euler", "euler_a", "dpmpp", "unipc")
DEFAULT_STEPS = (15, 20, 25, 30, 40)


def list_checkpoint_dirs(experiment: Path) -> list[Path]:
    """Checkpoint folders sorted by step ascending."""
    dirs = [
        path
        for path in experiment.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint_")
    ]
    return sorted(dirs, key=lambda path: int(path.name.split("_")[-1]))


def resolve_checkpoint(experiment: Path, checkpoint: str) -> tuple[Path, str, int | None]:
    """Resolve ``--checkpoint`` to a LoRA folder, label, and optional step.

    Accepts ``last`` / ``latest`` / ``final`` (experiment-root weights, else highest
    numbered checkpoint), a step integer like ``600``, a checkpoint folder name, or
    any path that contains ``pytorch_lora_weights.*``.
    """
    experiment = experiment.resolve()
    key = checkpoint.strip().lower()

    if key in {"last", "latest", "final"}:
        final = experiment / "pytorch_lora_weights.safetensors"
        if final.is_file():
            return experiment, "final", None
        dirs = list_checkpoint_dirs(experiment)
        if not dirs:
            raise FileNotFoundError(f"No final LoRA or checkpoint_* folders under {experiment}")
        chosen = dirs[-1]
        return chosen, chosen.name, int(chosen.name.split("_")[-1])

    if key.isdigit():
        step = int(key)
        matches = [
            path
            for path in list_checkpoint_dirs(experiment)
            if int(path.name.split("_")[-1]) == step
        ]
        if not matches:
            available = [int(path.name.split("_")[-1]) for path in list_checkpoint_dirs(experiment)]
            raise FileNotFoundError(
                f"No checkpoint for step {step} under {experiment}. "
                f"Available: {available or ['(none)']}"
            )
        return matches[0], matches[0].name, step

    candidate = Path(checkpoint)
    if not candidate.is_absolute():
        under_exp = experiment / checkpoint
        if under_exp.exists():
            candidate = under_exp
    candidate = candidate.resolve()
    lora = _resolve_lora_weights(candidate)
    label = candidate.name if candidate.is_dir() else candidate.parent.name
    step = None
    if candidate.is_dir() and candidate.name.startswith("checkpoint_"):
        step = int(candidate.name.split("_")[-1])
    return lora if lora.is_dir() else lora.parent, label, step


def load_experiment_config(experiment: Path) -> dict[str, Any]:
    path = experiment / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid experiment config: {path}")
    return data


def _prompt_slug(index: int, text: str, *, max_len: int = 32) -> str:
    words = "".join(c if c.isalnum() or c == " " else " " for c in text.lower()).split()
    body = "_".join(words[:6])[:max_len].strip("_") or "prompt"
    return f"{index:02d}_{body}"


def _sample_recipe(
    rng: random.Random,
    *,
    schedulers: tuple[str, ...],
    steps_choices: tuple[int, ...],
    guidance_min: float,
    guidance_max: float,
) -> dict[str, Any]:
    guidance = rng.uniform(guidance_min, guidance_max)
    # One decimal keeps filenames readable without pretending false precision.
    guidance = round(guidance, 1)
    return {
        "guidance_scale": guidance,
        "steps": int(rng.choice(steps_choices)),
        "scheduler": rng.choice(schedulers),
    }


def _assert_unet_lora_loaded(pipeline, lora_path: Path) -> None:
    """Confirm the UNet adapter is active; text-encoder warnings are expected."""
    adapters = {}
    if hasattr(pipeline, "get_list_adapters"):
        adapters = pipeline.get_list_adapters() or {}
    unet_adapters = list(adapters.get("unet") or [])
    if not unet_adapters:
        raise RuntimeError(
            f"LoRA load produced no UNet adapters from {lora_path}. "
            "Weights may be missing or incompatible."
        )

    from peft.tuners.lora import LoraLayer

    n_layers = 0
    n_dora = 0
    for module in pipeline.unet.modules():
        if isinstance(module, LoraLayer):
            n_layers += 1
            use_dora = getattr(module, "use_dora", None)
            if isinstance(use_dora, dict):
                if any(use_dora.values()):
                    n_dora += 1
            elif use_dora:
                n_dora += 1
            elif getattr(module, "lora_magnitude_vector", None) is not None:
                n_dora += 1

    kind = "DoRA" if n_dora else "LoRA"
    print(
        f"Loaded UNet {kind} adapter {unet_adapters} "
        f"({n_layers} layers, dora={n_dora}) from {lora_path}"
    )
    print(
        "Note: text_encoder LoRA warnings are normal — this run trained UNet only "
        "(train_text_encoder: false)."
    )


def run_creativity_sweep(
    *,
    experiment: Path,
    checkpoint: str = "last",
    variants_per_prompt: int = 4,
    seed: int = 42,
    device: str = "cuda:0",
    output_dir: Path | None = None,
    include_baseline: bool = True,
    schedulers: tuple[str, ...] = DEFAULT_SCHEDULERS,
    steps_choices: tuple[int, ...] = DEFAULT_STEPS,
    guidance_min: float = 3.0,
    guidance_max: float = 9.0,
    dtype: str | None = None,
    lora_scale: float = 1.0,
) -> Path:
    """Generate random inference recipes for every training sample prompt.

    Writes images plus ``manifest.json`` under the experiment folder (or
    ``output_dir``) so you can visually judge how much face creativity survives
    across guidance / steps / scheduler.
    """
    experiment = experiment.resolve()
    if not experiment.is_dir():
        raise FileNotFoundError(f"Experiment folder not found: {experiment}")
    if variants_per_prompt < 1:
        raise ValueError("variants_per_prompt must be >= 1")
    if not 0.0 <= guidance_min <= guidance_max:
        raise ValueError("require 0 <= guidance_min <= guidance_max")
    if not steps_choices:
        raise ValueError("steps_choices must not be empty")
    if not schedulers:
        raise ValueError("schedulers must not be empty")

    cfg = load_experiment_config(experiment)
    config = cfg.get("config") or {}
    sample_cfg = config.get("sample") or {}
    train_cfg = config.get("train") or {}
    model_cfg = config.get("model") or {}

    prompt_file = sample_cfg.get("prompt_file")
    if not prompt_file:
        raise ValueError(f"{experiment}/config.yaml has no sample.prompt_file")
    prompt_path = Path(prompt_file)
    if not prompt_path.is_file():
        # Resolve relative to repo root (cwd) first, then experiment parent chain.
        alt = Path.cwd() / prompt_file
        if alt.is_file():
            prompt_path = alt
        else:
            raise FileNotFoundError(f"prompt_file not found: {prompt_file}")

    trigger = config.get("instant_prompt")
    class_word = config.get("class_prompt")
    pos_list, neg_list, seed_list = load_training_sample_prompts(
        prompt_path,
        trigger=trigger,
        class_word=class_word,
    )

    model_id = model_cfg.get("name_or_path")
    if not model_id:
        raise ValueError("config.model.name_or_path is required in experiment config")
    resolution = int(train_cfg.get("resolution", sample_cfg.get("resolution", 1024)))
    dtype_str = dtype or config.get("dtype") or train_cfg.get("dtype") or "bf16"
    torch_dtype, _ = _parse_dtype(dtype_str)

    baseline_guidance = float(sample_cfg.get("guidance_scale", 7.0))
    baseline_steps = int(sample_cfg.get("steps", sample_cfg.get("sample_steps", 20)))
    baseline_scheduler = sample_cfg.get("scheduler") or "euler"

    lora_dir, checkpoint_label, checkpoint_step = resolve_checkpoint(experiment, checkpoint)
    lora_path = _resolve_lora_weights(lora_dir)

    stamp = int(time.time())
    out_root = (
        output_dir.resolve() if output_dir is not None else experiment / f"creativity_sweep_{stamp}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    print(f"Experiment: {experiment}")
    print(f"Checkpoint: {checkpoint_label} ({lora_path})")
    print(f"Prompts: {len(pos_list)} from {prompt_path}")
    print(
        f"Variants/prompt: {variants_per_prompt} (baseline={'on' if include_baseline else 'off'})"
    )
    print(f"Output: {out_root}")

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(device)
    pipeline.load_lora_weights(str(lora_path))
    _assert_unet_lora_loaded(pipeline, lora_path)

    # Exported weights carry no per-module alpha, so PEFT reloads them at the
    # wrong strength. Restore the trained alpha/rank ratio or the subject will
    # not look like itself.
    lora_targets = (train_cfg.get("lora") or train_cfg.get("unet_lora") or {}).get("target_modules")
    scaling_report: dict[str, Any] | None = None
    if lora_targets:
        scaling_report = apply_training_lora_scaling(pipeline, lora_targets, lora_scale=lora_scale)
        print(f"LoRA scaling restored ({format_scaling_report(scaling_report)})")
    else:
        print(
            "WARNING: no train.lora.target_modules in the experiment config; "
            "LoRA strength may be weaker than during training."
        )

    base_scheduler = pipeline.scheduler

    records: list[dict[str, Any]] = []
    total = len(pos_list) * (variants_per_prompt + (1 if include_baseline else 0))
    progress = tqdm(total=total, desc="creativity-sweep")

    try:
        for prompt_index, (pos, neg, prompt_seed) in enumerate(
            zip(pos_list, neg_list, seed_list, strict=True)
        ):
            slug = _prompt_slug(prompt_index, pos)
            prompt_dir = out_root / slug
            prompt_dir.mkdir(parents=True, exist_ok=True)

            recipes: list[dict[str, Any]] = []
            if include_baseline:
                recipes.append(
                    {
                        "kind": "baseline",
                        "guidance_scale": baseline_guidance,
                        "steps": baseline_steps,
                        "scheduler": baseline_scheduler,
                    }
                )
            for _ in range(variants_per_prompt):
                recipe = _sample_recipe(
                    rng,
                    schedulers=schedulers,
                    steps_choices=steps_choices,
                    guidance_min=guidance_min,
                    guidance_max=guidance_max,
                )
                recipe["kind"] = "random"
                recipes.append(recipe)

            for variant_index, recipe in enumerate(recipes):
                pipeline.scheduler = make_scheduler(base_scheduler, recipe["scheduler"])
                # The baseline keeps the exact training-sample seed so it is
                # directly comparable to samples/ images; random variants offset
                # it so each recipe explores different noise.
                gen_seed = (
                    int(prompt_seed)
                    if recipe["kind"] == "baseline"
                    else int(prompt_seed) + 10007 * variant_index + seed
                )
                generator = torch.Generator(device=device).manual_seed(gen_seed)
                image = pipeline(
                    prompt=pos,
                    negative_prompt=neg or None,
                    height=resolution,
                    width=resolution,
                    guidance_scale=float(recipe["guidance_scale"]),
                    num_inference_steps=int(recipe["steps"]),
                    generator=generator,
                ).images[0]

                g = recipe["guidance_scale"]
                g_tag = f"{g:.1f}".replace(".", "p")
                filename = (
                    f"v{variant_index:02d}_{recipe['kind']}"
                    f"_g{g_tag}_s{int(recipe['steps'])}_{recipe['scheduler']}.jpg"
                )
                image_path = prompt_dir / filename
                image.save(image_path, quality=92)

                records.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt_id": slug,
                        "variant_index": variant_index,
                        "kind": recipe["kind"],
                        "pos": pos,
                        "neg": neg,
                        "prompt_seed": int(prompt_seed),
                        "generation_seed": gen_seed,
                        "guidance_scale": float(recipe["guidance_scale"]),
                        "steps": int(recipe["steps"]),
                        "scheduler": recipe["scheduler"],
                        "image": image_path.relative_to(out_root).as_posix(),
                    }
                )
                progress.update(1)
    finally:
        progress.close()
        del pipeline
        _flush()

    summary = {
        "experiment": str(experiment),
        "checkpoint": checkpoint_label,
        "checkpoint_step": checkpoint_step,
        "lora_weights": str(lora_path),
        "prompt_file": str(prompt_path),
        "trigger": trigger,
        "class_word": class_word,
        "seed": seed,
        "variants_per_prompt": variants_per_prompt,
        "include_baseline": include_baseline,
        "guidance_range": [guidance_min, guidance_max],
        "steps_choices": list(steps_choices),
        "schedulers": list(schedulers),
        "resolution": resolution,
        "device": device,
        "dtype": dtype_str,
        "lora_scale": float(lora_scale),
        "lora_scaling": scaling_report,
        "n_prompts": len(pos_list),
        "n_images": len(records),
        "images": records,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} images + {manifest_path}")
    return out_root


def _parse_float_list(raw: str) -> tuple[float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("expected min,max")
    low, high = float(parts[0]), float(parts[1])
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError("guidance bounds must be finite")
    return low, high


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(value < 1 for value in values):
        raise ValueError("steps must be positive integers")
    return values
