"""Model-family helpers shared by training and batch generation."""

from __future__ import annotations

from collections.abc import Mapping

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    UniPCMultistepScheduler,
)

SDXL = "sdxl"
SDXL_TURBO = "sdxl-turbo"

_MODEL_TYPE_ALIASES = {
    "sdxl": SDXL,
    "sdxl-turbo": SDXL_TURBO,
    "sdxl_turbo": SDXL_TURBO,
    "turbo": SDXL_TURBO,
}


def get_model_type(model_config: Mapping) -> str:
    """Normalize the optional model-family setting.

    SDXL Turbo has the same component layout as SDXL, so it uses the same
    loading and LoRA-training code.  It does, however, require very different
    sampling defaults, which are selected through this explicit model type.
    """
    value = str(model_config.get("type", SDXL)).lower()
    try:
        return _MODEL_TYPE_ALIASES[value]
    except KeyError as err:
        supported = ", ".join(sorted({SDXL, SDXL_TURBO}))
        raise ValueError(f"Unsupported model.type {value!r}; use one of: {supported}") from err


def sample_defaults(model_type: str) -> tuple[float, int]:
    """Return ``(guidance_scale, inference_steps)`` for a model family."""
    if model_type == SDXL_TURBO:
        # Turbo was distilled for CFG-free, one-to-four-step inference.
        return 0.0, 4
    return 7.0, 20


def make_scheduler(scheduler, name: str):
    """Build a user-selected inference scheduler from an existing scheduler."""
    cfg = scheduler.config
    name = name.lower()
    if name == "ddpm":
        return DDPMScheduler.from_config(cfg)
    if name == "ddim":
        return DDIMScheduler.from_config(cfg)
    if name == "euler":
        return EulerDiscreteScheduler.from_config(cfg)
    if name in {"euler_a", "euler-a", "euler_ancestral"}:
        return EulerAncestralDiscreteScheduler.from_config(cfg)
    if name == "dpmpp":
        return DPMSolverMultistepScheduler.from_config(
            cfg, algorithm_type="dpmsolver++", use_karras_sigmas=True
        )
    if name == "unipc":
        return UniPCMultistepScheduler.from_config(cfg)
    raise ValueError(
        f"Unsupported scheduler: {name!r} (use ddpm, ddim, euler, euler_a, dpmpp, or unipc)"
    )
