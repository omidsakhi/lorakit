"""Restore per-module LoRA alpha scaling that is lost when exporting weights.

``StableDiffusionXLPipeline.save_lora_weights`` writes only ``lora.down`` /
``lora.up`` (and DoRA magnitude) tensors — no per-module ``alpha``. On reload,
PEFT infers ``rank_pattern`` from the tensor shapes but falls back to a single
default ``lora_alpha``, so every module whose training rank differed from that
default runs at the wrong strength (``scaling = alpha / rank``).

For a config like ``{ff.net.0.proj: [32, 32], to_v: [16, 16], to_out.0: [8, 8]}``
training scales every layer by 1.0, while a naive reload yields 0.25 / 0.5 / 1.0 —
i.e. the adapter is silently 2-4x too weak and the subject stops looking like
itself. These helpers recompute the intended ratio from the training config.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = "config.yaml"


def parse_rank_alpha(value: Any) -> tuple[int, int]:
    """Parse one ``(rank, lora_alpha)`` target value from a training config."""
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


def training_scaling_map(target_modules: Mapping[str, Any]) -> dict[str, float]:
    """Map each ``target_modules`` key to its trained ``alpha / rank`` scaling."""
    if not isinstance(target_modules, Mapping) or not target_modules:
        raise ValueError("target_modules must be a non-empty mapping")
    scaling: dict[str, float] = {}
    for name, value in target_modules.items():
        rank, alpha = parse_rank_alpha(value)
        if rank <= 0:
            raise ValueError(f"target_modules[{name!r}] rank must be positive")
        scaling[str(name)] = float(alpha) / float(rank)
    return scaling


def find_training_target_modules(*candidates: Path | str | None) -> dict[str, Any] | None:
    """Find ``config.train.lora.target_modules`` in the first readable candidate.

    Candidates may be experiment folders, checkpoint folders, or weight file
    paths; parent folders are searched too so a ``checkpoint_*`` directory finds
    the experiment-level ``config.yaml``.
    """
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        roots = [path] if path.is_dir() else [path.parent]
        roots.append(roots[0].parent)
        for root in roots:
            config_path = root / CONFIG_NAME
            if not config_path.is_file():
                continue
            try:
                data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if not isinstance(data, dict):
                continue
            config = data.get("config") or {}
            train = config.get("train") or {}
            lora = train.get("lora") or train.get("unet_lora") or {}
            target_modules = lora.get("target_modules")
            if isinstance(target_modules, Mapping) and target_modules:
                return dict(target_modules)
    return None


def apply_training_lora_scaling(
    pipeline,
    target_modules: Mapping[str, Any],
    *,
    lora_scale: float = 1.0,
) -> dict[str, Any]:
    """Set every UNet LoRA layer's scaling to its trained ``alpha / rank`` ratio.

    ``lora_scale`` multiplies the restored ratio, so 1.0 reproduces training and
    lower values weaken the adapter. Returns a report with the applied ratios and
    the number of layers touched.
    """
    if not (lora_scale > 0):
        raise ValueError("lora_scale must be positive")

    from peft.tuners.lora import LoraLayer

    scaling_map = training_scaling_map(target_modules)
    # Longest key first so `ff.net.0.proj` wins over a bare `proj`-style key.
    keys = sorted(scaling_map, key=len, reverse=True)

    applied: dict[str, float] = {}
    counts: dict[str, int] = {}
    before: dict[str, float] = {}
    unmatched = 0

    for name, module in pipeline.unet.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        key = next(
            (k for k in keys if re.match(rf"(.*\.)?{re.escape(k)}$", name)),
            None,
        )
        if key is None:
            unmatched += 1
            continue
        target = scaling_map[key] * float(lora_scale)
        for adapter_name in list(module.scaling.keys()):
            before.setdefault(key, float(module.scaling[adapter_name]))
            module.scaling[adapter_name] = target
        applied[key] = target
        counts[key] = counts.get(key, 0) + 1

    return {
        "applied": applied,
        "previous": before,
        "layers": counts,
        "n_layers": sum(counts.values()),
        "unmatched_layers": unmatched,
        "lora_scale": float(lora_scale),
    }


def format_scaling_report(report: Mapping[str, Any]) -> str:
    """One-line summary of what `apply_training_lora_scaling` changed."""
    applied = report.get("applied") or {}
    previous = report.get("previous") or {}
    counts = report.get("layers") or {}
    parts = []
    for key in sorted(applied):
        was = previous.get(key)
        was_text = f"{was:.3f}" if isinstance(was, (int, float)) else "?"
        parts.append(f"{key}: {was_text}->{applied[key]:.3f} x{counts.get(key, 0)}")
    return ", ".join(parts) if parts else "no LoRA layers matched"


__all__ = [
    "apply_training_lora_scaling",
    "find_training_target_modules",
    "format_scaling_report",
    "parse_rank_alpha",
    "training_scaling_map",
]
