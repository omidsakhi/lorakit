"""Exponential moving average over trainable LoRA / DoRA parameters."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch import Tensor
from torch.nn import Parameter


class LoRAEMA:
    """Shadow copy of trainable LoRA tensors updated after each optimizer step.

    Diffusers' ``EMAModel`` shadows a full module; here only PEFT adapter
    parameters are trainable, so a plain param-list average is enough and stays
    compatible with ``torch.compile``, quantization, and DoRA magnitudes.

    The decay is bias-corrected during warmup (see :attr:`current_decay`). Without
    that correction a LoRA EMA is unusable on short DreamBooth runs: PEFT
    zero-initializes the ``lora_B`` / up projections, so a raw ``0.9999`` average
    still carries ~96% of the *untrained* adapter at step 400 and every sample
    looks like the base model.
    """

    def __init__(self, parameters: Iterable[Parameter], decay: float = 0.999) -> None:
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("LoRAEMA requires at least one trainable parameter")
        self.num_updates = 0
        self.shadow: list[Tensor] = [
            parameter.detach().float().clone() for parameter in self.parameters
        ]
        self._backup: list[Tensor] | None = None

    @property
    def current_decay(self) -> float:
        """Decay for the next update, warmed up from 0 to :attr:`decay`.

        ``num_updates / (num_updates + 1)`` makes the early shadow an exact
        running mean of the observed weights, so the average never lags behind
        the initialization; it switches to the configured exponential decay once
        the averaging horizon (``1 / (1 - decay)`` steps) has elapsed.
        """
        return min(self.decay, self.num_updates / (self.num_updates + 1))

    @torch.no_grad()
    def update(self) -> None:
        """Blend current trainable weights into the shadow buffers."""
        decay = self.current_decay
        one_minus_decay = 1.0 - decay
        for shadow, parameter in zip(self.shadow, self.parameters, strict=True):
            shadow.mul_(decay).add_(parameter.detach().float(), alpha=one_minus_decay)
        self.num_updates += 1

    @torch.no_grad()
    def reset(self) -> None:
        """Reseed the shadows from the live weights and restart the warmup.

        Used when resuming a checkpoint that has no saved EMA state, where the
        shadows would otherwise still hold the pre-resume initialization.
        """
        self.num_updates = 0
        self.shadow = [parameter.detach().float().clone() for parameter in self.parameters]

    @torch.no_grad()
    def copy_to(self) -> None:
        """Overwrite live parameters with EMA shadows (e.g. before sampling)."""
        for shadow, parameter in zip(self.shadow, self.parameters, strict=True):
            parameter.data.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))

    @torch.no_grad()
    def store(self) -> None:
        """Snapshot live parameters so :meth:`restore` can undo :meth:`copy_to`."""
        self._backup = [parameter.detach().clone() for parameter in self.parameters]

    @torch.no_grad()
    def restore(self) -> None:
        """Restore parameters saved by :meth:`store`."""
        if self._backup is None:
            raise RuntimeError("LoRAEMA.restore() called without a prior store()")
        for backup, parameter in zip(self._backup, self.parameters, strict=True):
            parameter.data.copy_(backup)
        self._backup = None

    @contextlib.contextmanager
    def average_parameters(self) -> Iterator[None]:
        """Temporarily install EMA weights, then restore the training weights."""
        self.store()
        self.copy_to()
        try:
            yield
        finally:
            self.restore()

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": [tensor.detach().cpu().clone() for tensor in self.shadow],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        decay = float(state["decay"])
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {decay}")
        shadows = state["shadow"]
        if len(shadows) != len(self.shadow):
            raise ValueError(
                f"EMA shadow count mismatch: checkpoint has {len(shadows)}, "
                f"model has {len(self.shadow)}"
            )
        self.decay = decay
        loaded: list[Tensor] = []
        for tensor, parameter in zip(shadows, self.parameters, strict=True):
            shadow = torch.as_tensor(tensor).detach().float().clone()
            if shadow.shape != parameter.shape:
                raise ValueError(
                    f"EMA shadow shape mismatch: got {tuple(shadow.shape)}, "
                    f"expected {tuple(parameter.shape)}"
                )
            loaded.append(shadow)
        self.shadow = loaded
        self.num_updates = int(state.get("num_updates", 0))


__all__ = ["LoRAEMA"]
