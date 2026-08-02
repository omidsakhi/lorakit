"""CLI: apply a gendered prompt corpus to an experiment checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from lorakit.prompt_sweep import run_prompt_sweep
from lorakit.prompts import gender_class_words, normalize_gender


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reload an experiment checkpoint and generate images from a large "
            "prompt JSON, filtered by gender. Female keeps prompts with "
            "woman/girl/bride; male keeps man/boy/groom. Each prompt is "
            "rewritten to use the experiment trigger + class word "
            "(e.g. 'sks woman')."
        )
    )
    parser.add_argument(
        "experiment",
        type=Path,
        help="Experiment folder containing config.yaml and LoRA weights "
        "(e.g. output/jane/jane_1.0)",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="JSON prompt corpus (array of {id,pos,neg,seed}), e.g. "
        "data/prompts-20260609-152102.json",
    )
    parser.add_argument(
        "--gender",
        required=True,
        help="Subject gender filter: male or female",
    )
    parser.add_argument(
        "--checkpoint",
        default="last",
        help=(
            "Which weights to load: 'last'/'latest'/'final' (default), a step "
            "number like 600, a checkpoint_* folder name, or a path"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many filtered prompts to run (random sample)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed used when --limit samples a subset (default: 42)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device (default: cuda:0)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Override dtype (bf16/fp16/fp32). Default: experiment config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder (default: <experiment>/prompt_sweep_<gender>_<timestamp>)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="Override guidance scale (default: experiment sample.guidance_scale)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override inference steps (default: experiment sample steps)",
    )
    parser.add_argument(
        "--scheduler",
        default=None,
        help="Override scheduler name (default: experiment sample.scheduler or euler)",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier on the restored training LoRA strength. 1.0 matches "
            "training; lower values trade likeness for prompt freedom."
        ),
    )
    args = parser.parse_args()

    try:
        gender = normalize_gender(args.gender)
    except ValueError as error:
        parser.error(str(error))

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if not args.lora_scale > 0:
        parser.error("--lora-scale must be positive")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be >= 1")

    print(f"Filtering {gender} prompts ({', '.join(gender_class_words(gender))})")

    run_prompt_sweep(
        experiment=args.experiment,
        prompts_path=args.prompts,
        gender=gender,
        checkpoint=args.checkpoint,
        limit=args.limit,
        seed=args.seed,
        device=args.device,
        output_dir=args.output,
        dtype=args.dtype,
        lora_scale=args.lora_scale,
        guidance_scale=args.guidance,
        steps=args.steps,
        scheduler=args.scheduler,
    )


if __name__ == "__main__":
    main()
