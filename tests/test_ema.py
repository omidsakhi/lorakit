"""Tests for LoRA EMA weight averaging."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from lorakit.ema import LoRAEMA


class _TinyLoRA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down = nn.Linear(4, 2, bias=False)
        self.up = nn.Linear(2, 4, bias=False)


def _fill(model: nn.Module, value: float) -> None:
    for parameter in model.parameters():
        parameter.data.fill_(value)


class LoRAEMATests(unittest.TestCase):
    def test_first_update_discards_initialization(self):
        # PEFT zero-initializes lora_B, so an EMA anchored to the init would keep
        # emitting base-model samples for thousands of steps.
        model = _TinyLoRA()
        _fill(model, 0.0)
        ema = LoRAEMA(model.parameters(), decay=0.9999)
        _fill(model, 1.0)
        ema.update()
        for shadow in ema.shadow:
            self.assertTrue(torch.allclose(shadow, torch.ones_like(shadow)))

    def test_warmup_is_an_exact_running_mean(self):
        model = _TinyLoRA()
        _fill(model, 0.0)
        ema = LoRAEMA(model.parameters(), decay=0.9999)
        for value in (1.0, 2.0, 3.0):
            _fill(model, value)
            ema.update()
        for shadow in ema.shadow:
            self.assertTrue(torch.allclose(shadow, torch.full_like(shadow, 2.0)))

    def test_decay_caps_the_warmup(self):
        model = _TinyLoRA()
        ema = LoRAEMA(model.parameters(), decay=0.5)
        self.assertEqual(ema.current_decay, 0.0)
        ema.update()
        self.assertEqual(ema.current_decay, 0.5)
        ema.update()
        self.assertEqual(ema.current_decay, 0.5)

    def test_update_blends_toward_live_weights_after_warmup(self):
        model = _TinyLoRA()
        _fill(model, 0.0)
        ema = LoRAEMA(model.parameters(), decay=0.5)
        ema.update()  # warmup: shadow = 0.0
        _fill(model, 1.0)
        ema.update()  # 0.5*0.0 + 0.5*1.0
        for shadow in ema.shadow:
            self.assertTrue(torch.allclose(shadow, torch.full_like(shadow, 0.5)))

    def test_average_parameters_swaps_and_restores(self):
        model = _TinyLoRA()
        _fill(model, 0.0)
        ema = LoRAEMA(model.parameters(), decay=0.9)
        ema.update()
        _fill(model, 2.0)

        live_before = [parameter.detach().clone() for parameter in model.parameters()]
        with ema.average_parameters():
            for parameter, shadow in zip(model.parameters(), ema.shadow, strict=True):
                self.assertTrue(torch.allclose(parameter, shadow.to(dtype=parameter.dtype)))
        for parameter, expected in zip(model.parameters(), live_before, strict=True):
            self.assertTrue(torch.allclose(parameter, expected))

    def test_reset_reseeds_from_live_weights(self):
        model = _TinyLoRA()
        _fill(model, 0.0)
        ema = LoRAEMA(model.parameters(), decay=0.9)
        ema.update()
        _fill(model, 5.0)
        ema.reset()
        self.assertEqual(ema.num_updates, 0)
        for shadow in ema.shadow:
            self.assertTrue(torch.allclose(shadow, torch.full_like(shadow, 5.0)))

    def test_state_dict_roundtrip(self):
        model = _TinyLoRA()
        for parameter in model.parameters():
            parameter.data.uniform_(-1.0, 1.0)
        ema = LoRAEMA(model.parameters(), decay=0.99)
        for parameter in model.parameters():
            parameter.data.add_(1.0)
        ema.update()
        ema.update()

        other = _TinyLoRA()
        other_ema = LoRAEMA(other.parameters(), decay=0.5)
        other_ema.load_state_dict(ema.state_dict())
        self.assertEqual(other_ema.decay, 0.99)
        self.assertEqual(other_ema.num_updates, ema.num_updates)
        for left, right in zip(ema.shadow, other_ema.shadow, strict=True):
            self.assertTrue(torch.allclose(left, right))

    def test_rejects_invalid_decay(self):
        model = _TinyLoRA()
        with self.assertRaises(ValueError):
            LoRAEMA(model.parameters(), decay=1.0)


if __name__ == "__main__":
    unittest.main()
