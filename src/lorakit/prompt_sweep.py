"""Prompt corpus sweep: apply a gendered prompt list to an experiment checkpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm

from lorakit.creativity_sweep import (
    _assert_unet_lora_loaded,
    _prompt_slug,
    load_experiment_config,
    resolve_checkpoint,
)
from lorakit.lora_scaling import (
    apply_training_lora_scaling,
    format_scaling_report,
)
from lorakit.models import make_scheduler
from lorakit.prompts import (
    filter_prompts_by_gender,
    gender_class_words,
    inject_subject_trigger,
    load_prompts,
    normalize_gender,
    sample_prompts,
)
from lorakit.sample import _flush, _parse_dtype, _resolve_lora_weights


def run_prompt_sweep(
    *,
    experiment: Path,
    prompts_path: Path,
    gender: str,
    checkpoint: str = "last",
    limit: int | None = None,
    seed: int = 42,
    device: str = "cuda:0",
    output_dir: Path | None = None,
    dtype: str | None = None,
    lora_scale: float = 1.0,
    guidance_scale: float | None = None,
    steps: int | None = None,
    scheduler: str | None = None,
) -> Path:
    """Generate one image per gender-filtered prompt against an experiment LoRA.

    Female prompts match ``woman`` / ``girl`` / ``bride``; male prompts match
    ``man`` / ``boy`` / ``groom``. Each positive prompt is rewritten so the
    experiment trigger + class word (e.g. ``sks woman``) replace the matched
    subject noun.
    """
    experiment = experiment.resolve()
    if not experiment.is_dir():
        raise FileNotFoundError(f"Experiment folder not found: {experiment}")

    gender_key = normalize_gender(gender)
    prompts_path = prompts_path.resolve()
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    cfg = load_experiment_config(experiment)
    config = cfg.get("config") or {}
    sample_cfg = config.get("sample") or {}
    train_cfg = config.get("train") or {}
    model_cfg = config.get("model") or {}

    trigger = config.get("instant_prompt")
    class_word = config.get("class_prompt")
    if not trigger or not class_word:
        raise ValueError(f"{experiment}/config.yaml must define instant_prompt and class_prompt")

    model_id = model_cfg.get("name_or_path")
    if not model_id:
        raise ValueError("config.model.name_or_path is required in experiment config")

    resolution = int(train_cfg.get("resolution", sample_cfg.get("resolution", 1024)))
    dtype_str = dtype or config.get("dtype") or train_cfg.get("dtype") or "bf16"
    torch_dtype, _ = _parse_dtype(dtype_str)

    run_guidance = float(
        guidance_scale if guidance_scale is not None else sample_cfg.get("guidance_scale", 7.0)
    )
    run_steps = int(
        steps if steps is not None else sample_cfg.get("steps", sample_cfg.get("sample_steps", 20))
    )
    run_scheduler = scheduler or sample_cfg.get("scheduler") or "euler"

    all_prompts = load_prompts(prompts_path)
    filtered = filter_prompts_by_gender(all_prompts, gender_key)
    if not filtered:
        words = ", ".join(gender_class_words(gender_key))
        raise ValueError(f"No {gender_key} prompts found in {prompts_path} (looked for: {words})")
    selected = sample_prompts(filtered, limit, seed=seed)

    lora_dir, checkpoint_label, checkpoint_step = resolve_checkpoint(experiment, checkpoint)
    lora_path = _resolve_lora_weights(lora_dir)

    stamp = int(time.time())
    out_root = (
        output_dir.resolve()
        if output_dir is not None
        else experiment / f"prompt_sweep_{gender_key}_{stamp}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {experiment}")
    print(f"Checkpoint: {checkpoint_label} ({lora_path})")
    print(f"Prompts file: {prompts_path}")
    print(
        f"Gender: {gender_key} ({', '.join(gender_class_words(gender_key))}) "
        f"— {len(filtered)}/{len(all_prompts)} matched, running {len(selected)}"
    )
    print(f"Trigger: {trigger} {class_word}")
    print(
        f"Recipe: guidance={run_guidance}, steps={run_steps}, scheduler={run_scheduler}, "
        f"resolution={resolution}"
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

    pipeline.scheduler = make_scheduler(pipeline.scheduler, run_scheduler)

    records: list[dict[str, Any]] = []
    progress = tqdm(total=len(selected), desc="prompt-sweep")

    try:
        for index, prompt in enumerate(selected):
            pos = inject_subject_trigger(
                prompt.pos,
                trigger=str(trigger),
                class_word=str(class_word),
                gender=gender_key,
            )
            slug = _prompt_slug(index, pos)
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt.id)
            gen_seed = int(prompt.seed)
            generator = torch.Generator(device=device).manual_seed(gen_seed)
            image = pipeline(
                prompt=pos,
                negative_prompt=prompt.neg or None,
                height=resolution,
                width=resolution,
                guidance_scale=run_guidance,
                num_inference_steps=run_steps,
                generator=generator,
            ).images[0]

            filename = f"{index:04d}_{safe_id}.jpg"
            image_path = out_root / filename
            image.save(image_path, quality=92)

            records.append(
                {
                    "index": index,
                    "prompt_id": prompt.id,
                    "slug": slug,
                    "pos_original": prompt.pos,
                    "pos": pos,
                    "neg": prompt.neg,
                    "prompt_seed": int(prompt.seed),
                    "generation_seed": gen_seed,
                    "guidance_scale": run_guidance,
                    "steps": run_steps,
                    "scheduler": run_scheduler,
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
        "prompts_file": str(prompts_path),
        "gender": gender_key,
        "gender_class_words": list(gender_class_words(gender_key)),
        "trigger": trigger,
        "class_word": class_word,
        "seed": seed,
        "limit": limit,
        "n_prompts_file": len(all_prompts),
        "n_prompts_filtered": len(filtered),
        "n_prompts_run": len(selected),
        "guidance_scale": run_guidance,
        "steps": run_steps,
        "scheduler": run_scheduler,
        "resolution": resolution,
        "device": device,
        "dtype": dtype_str,
        "lora_scale": float(lora_scale),
        "lora_scaling": scaling_report,
        "n_images": len(records),
        "images": records,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # Keep a compact copy of the resolved config next to outputs for audits.
    (out_root / "experiment_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} images + {manifest_path}")
    return out_root
