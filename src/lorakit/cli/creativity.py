"""CLI: creativity retention sweep over guidance / steps / scheduler."""

from __future__ import annotations

import argparse
from pathlib import Path

from lorakit.creativity_sweep import (
    DEFAULT_SCHEDULERS,
    DEFAULT_STEPS,
    _parse_float_list,
    _parse_int_list,
    run_creativity_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reload an experiment checkpoint and regenerate the training "
            "sample prompts with randomly chosen guidance, step count, and "
            "noise scheduler — to eyeball how much face creativity survives."
        )
    )
    parser.add_argument(
        "experiment",
        type=Path,
        help="Experiment folder containing config.yaml and LoRA weights "
        "(e.g. output/jane/jane_1.0)",
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
        "--variants",
        type=int,
        default=4,
        help="Random recipes per prompt (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for recipe sampling (default: 42)",
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
        help="Output folder (default: <experiment>/creativity_sweep_<timestamp>)",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the train-time guidance/steps/scheduler recipe per prompt",
    )
    parser.add_argument(
        "--guidance",
        default="3.0,9.0",
        help="Random guidance range as min,max (default: 3.0,9.0)",
    )
    parser.add_argument(
        "--steps",
        default=",".join(str(value) for value in DEFAULT_STEPS),
        help=f"Comma-separated step choices (default: {','.join(map(str, DEFAULT_STEPS))})",
    )
    parser.add_argument(
        "--schedulers",
        default=",".join(DEFAULT_SCHEDULERS),
        help=f"Comma-separated schedulers (default: {','.join(DEFAULT_SCHEDULERS)})",
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
        guidance_min, guidance_max = _parse_float_list(args.guidance)
        steps_choices = _parse_int_list(args.steps)
    except ValueError as error:
        parser.error(str(error))

    schedulers = tuple(name.strip().lower() for name in args.schedulers.split(",") if name.strip())
    if not schedulers:
        parser.error("--schedulers must not be empty")
    if args.variants < 1:
        parser.error("--variants must be >= 1")
    if not args.lora_scale > 0:
        parser.error("--lora-scale must be positive")

    run_creativity_sweep(
        experiment=args.experiment,
        checkpoint=args.checkpoint,
        variants_per_prompt=args.variants,
        seed=args.seed,
        device=args.device,
        output_dir=args.output,
        include_baseline=not args.no_baseline,
        schedulers=schedulers,
        steps_choices=steps_choices,
        guidance_min=guidance_min,
        guidance_max=guidance_max,
        dtype=args.dtype,
        lora_scale=args.lora_scale,
    )


if __name__ == "__main__":
    main()
