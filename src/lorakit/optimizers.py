"""Custom optimizers for lorakit.

This module provides :class:`AnchoredAdamW`, an AdamW variant whose decoupled
weight decay pulls each parameter back toward its *initial* value instead of
toward zero, and :class:`AnchoredAdamWScheduleFree`, the same anchored decay
combined with the Schedule-Free AdamW update of Defazio et al.

Standard AdamW applies decoupled weight decay of the form::

    p <- p - lr * weight_decay * p

which biases every weight toward the origin. For fine-tuning (and LoRA in
particular) that is often the wrong prior: we do not believe the "best" weights
live near zero, we believe they live near where we started. ``AnchoredAdamW``
implements an L2-SP ("L2 toward Starting Point") penalty instead::

    p <- p - lr * weight_decay * (p - p0)

where ``p0`` is the value of the parameter the first time the optimizer sees it.
This keeps the solution close to the initialization (anchor) while still letting
Adam's adaptive step move it where the gradient demands. Setting
``weight_decay=0`` recovers plain Adam with no anchoring.

References:
    Li, Grandvalet & Davoine, "Explicit Inductive Bias for Transfer Learning
    with Convolutional Networks" (ICML 2018).
    Defazio et al., "The Road Less Scheduled" (2024), https://arxiv.org/abs/2405.15682.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ["AnchoredAdamW", "AnchoredAdamWScheduleFree"]


class AnchoredAdamW(Optimizer):
    """AdamW with decoupled weight decay toward the initial weights (L2-SP).

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate.
        betas: Coefficients for the running averages of gradient and its square.
        eps: Term added to the denominator for numerical stability.
        weight_decay: Strength of the pull toward the anchor (initial weights).
            With ``0`` this behaves like plain Adam.
        amsgrad: Whether to use the AMSGrad variant.
        anchor: Optional initial anchor behaviour. If ``None`` (default) each
            parameter is anchored to its value the first time :meth:`step` is
            called. The captured anchors are stored in the optimizer state, so
            they are saved/restored with ``state_dict``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AnchoredAdamW does not support sparse gradients")

                state = self.state[p]

                # Lazy state init. The anchor is the parameter value the first
                # time we see it, i.e. the initialization we want to stay near.
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    state["anchor"] = p.detach().clone()

                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]
                anchor: Tensor = state["anchor"]
                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay toward the anchor instead of zero.
                # p <- p - lr * weight_decay * (p - anchor)
                if weight_decay != 0:
                    p.add_(p - anchor, alpha=-lr * weight_decay)

                # Standard Adam moment updates.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                if amsgrad:
                    max_exp_avg_sq: Tensor = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class AnchoredAdamWScheduleFree(Optimizer):
    """Schedule-Free AdamW with decoupled weight decay toward the initial weights.

    Combines the Schedule-Free update of Defazio et al. ("The Road Less
    Scheduled", 2024) with the L2-SP anchored weight decay of
    :class:`AnchoredAdamW`: instead of decaying toward zero, the decay term
    computed at ``y`` pulls toward each parameter's initial value (anchor)::

        decay_term = weight_decay * (y - anchor)

    As with the reference Schedule-Free implementation, no LR scheduler should
    be used; warmup is handled via ``warmup_steps``. The optimizer requires
    ``.train()`` and ``.eval()`` calls: call ``.train()`` before training steps
    and ``.eval()`` before evaluation/sampling and before saving checkpoints,
    so the parameters hold the averaged iterate ``x`` rather than ``y``.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate.
        betas: Coefficients for interpolation/averaging (beta1) and the running
            average of the squared gradient (beta2).
        eps: Term added to the denominator for numerical stability.
        weight_decay: Strength of the pull toward the anchor (initial weights).
            With ``0`` this behaves like plain Schedule-Free Adam.
        warmup_steps: Enables a linear learning rate warmup.
        r: Use polynomial weighting in the average with power ``r``.
        weight_lr_power: During warmup, the weights in the average will be
            equal to lr raised to this power. Set to 0 for no weighting.
    """

    def __init__(
        self,
        params,
        lr: float = 0.0025,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps value: {warmup_steps}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            r=r,
            k=0,
            train_mode=False,
            weight_sum=0.0,
            lr_max=-1.0,
            scheduled_lr=0.0,
            weight_lr_power=weight_lr_power,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            beta1, _ = group["betas"]
            if group["train_mode"]:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p to x (the averaged iterate).
                        p.lerp_(end=state["z"].to(p.device), weight=1 - 1 / beta1)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            beta1, _ = group["betas"]
            if not group["train_mode"]:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p to y (the gradient-evaluation point).
                        p.lerp_(end=state["z"].to(p.device), weight=1 - beta1)
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                "Optimizer was not in train mode when step is called. Please "
                "insert .train() and .eval() calls on the optimizer."
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            k = group["k"]
            r = group["r"]
            warmup_steps = group["warmup_steps"]
            weight_lr_power = group["weight_lr_power"]

            sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
            bias_correction2 = 1 - beta2 ** (k + 1)
            lr = group["lr"] * sched
            group["scheduled_lr"] = lr  # For logging purposes.

            lr_max = group["lr_max"] = max(lr, group["lr_max"])

            weight = ((k + 1) ** r) * (lr_max**weight_lr_power)
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight
            ckp1 = weight / weight_sum if weight_sum != 0 else 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "AnchoredAdamWScheduleFree does not support sparse gradients"
                    )

                state = self.state[p]

                # Lazy state init. At the first step p == y == z == x, so the
                # anchor captures the initialization we want to stay near.
                if "z" not in state:
                    state["z"] = torch.clone(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["anchor"] = p.detach().clone()

                y = p  # Notation to match the schedule-free theory.
                z: Tensor = state["z"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]
                anchor: Tensor = state["anchor"]

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                # Reuse the grad buffer for memory efficiency.
                grad_normalized = grad.div_(denom)

                # Anchored weight decay calculated at y: decay toward the
                # initial weights instead of toward zero.
                if weight_decay != 0:
                    grad_normalized.add_(y - anchor, alpha=weight_decay)

                # These operations update y in-place, without computing x
                # explicitly.
                y.lerp_(end=z, weight=ckp1)
                y.add_(grad_normalized, alpha=lr * (beta1 * (1 - ckp1) - 1))

                # z step
                z.sub_(grad_normalized, alpha=lr)

            group["k"] = k + 1

        return loss
