import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from lorakit.datasets import DreamBoothDataset
from lorakit.identity import BackgroundPreservationObjective, reconstruct_clean_latent
from lorakit.latent_face_detector import LatentBackgroundPreservationObjective


class _BackgroundVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)

    def decode(self, latents):
        return SimpleNamespace(sample=latents[:, :3] * self.scale)


class BackgroundPreservationTests(unittest.TestCase):
    def test_clean_latent_reconstruction_for_epsilon_and_velocity(self):
        z0 = torch.tensor([[[[0.3]]]])
        epsilon = torch.tensor([[[[-0.2]]]])
        alphas = torch.tensor([0.36])
        alpha = alphas.sqrt().view(1, 1, 1, 1)
        sigma = (1 - alphas).sqrt().view(1, 1, 1, 1)
        noisy = alpha * z0 + sigma * epsilon
        epsilon_result = reconstruct_clean_latent(
            noisy, epsilon, torch.tensor([0]), alphas, "epsilon"
        )
        velocity = alpha * epsilon - sigma * z0
        velocity_result = reconstruct_clean_latent(
            noisy, velocity, torch.tensor([0]), alphas, "v_prediction"
        )
        torch.testing.assert_close(epsilon_result, z0)
        torch.testing.assert_close(velocity_result, z0)

    def test_out_of_crop_face_is_an_invalid_sentinel(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            image_path = directory / "subject.png"
            Image.new("RGB", (200, 100), (127, 127, 127)).save(image_path)
            manifest_path = directory / "faces.json"
            manifest_path.write_text(
                json.dumps({"subject.png": {"bbox_xyxy": [0, 0, 20, 20]}}), encoding="utf-8"
            )
            dataset = DreamBoothDataset(
                directory,
                "subject",
                "person",
                resolution=50,
                center_crop=True,
                face_manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
            )
            self.assertEqual(dataset[0]["instance_face_box"], (-1.0, -1.0, -1.0, -1.0))
            self.assertTrue(dataset[0]["instance_face_box_clipped"])

    def test_background_objective_backpropagates_through_prediction(self):
        objective = BackgroundPreservationObjective(
            bbox_padding_fraction=0.0,
            max_timestep_fraction=1.0,
            decoder_gradient_checkpointing=True,
        )
        desired_latents = torch.tensor([[[[0.8] * 8] * 8, [[-0.8] * 8] * 8, [[-0.6] * 8] * 8]])
        alphas = torch.tensor([0.25])
        alpha = alphas.sqrt()
        sigma = (1 - alphas).sqrt()
        noise = torch.full_like(desired_latents, 0.1)
        noisy = alpha * desired_latents + sigma * noise
        prediction = noise.detach().clone().requires_grad_(True)
        base_prediction = (noisy / sigma).detach()
        vae = _BackgroundVAE()
        background_loss, stats = objective(
            noisy_instance_latents=noisy,
            model_prediction=prediction,
            base_model_prediction=base_prediction,
            timesteps=torch.zeros(1, dtype=torch.long),
            face_boxes=torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
            face_box_clipped=torch.tensor([False]),
            vae=vae,
            alphas_cumprod=alphas,
            prediction_type="epsilon",
            latent_scaling_factor=1.0,
            latents_mean=None,
            latents_std=None,
            num_train_timesteps=1,
        )
        background_loss.backward()
        self.assertGreater(background_loss.item(), 0.0)
        self.assertGreater(stats["background_mse"], 0.0)
        self.assertGreater(torch.count_nonzero(prediction.grad).item(), 0)
        self.assertIsNone(vae.scale.grad)

    def test_latent_background_objective_only_penalizes_manifest_background(self):
        objective = LatentBackgroundPreservationObjective(mask_edge_sharpness=1_000.0)
        # Normalized box covering the central half of the latent grid.
        face_boxes = torch.tensor([[0.25, 0.25, 0.75, 0.75]])
        prediction = torch.zeros(1, 4, 8, 8, requires_grad=True)
        with torch.no_grad():
            prediction[:, :, 3, 3] = 1.0  # face center: excluded from the loss
            prediction[:, :, 0, 0] = 1.0  # background: included in the loss
        loss, stats = objective(
            noisy_instance_latents=torch.zeros_like(prediction),
            model_prediction=prediction,
            base_model_prediction=torch.zeros_like(prediction),
            timesteps=torch.tensor([0]),
            scheduler=None,
            num_train_timesteps=1,
            face_boxes=face_boxes,
        )
        loss.backward()
        self.assertGreater(loss.item(), 0.0)
        self.assertEqual(stats["valid_faces"], 1.0)
        self.assertEqual(prediction.grad[:, :, 3, 3].abs().max().item(), 0.0)
        self.assertGreater(prediction.grad[:, :, 0, 0].abs().max().item(), 0.0)

    def test_face_focus_mask_is_high_inside_box_and_low_outside(self):
        objective = LatentBackgroundPreservationObjective(mask_edge_sharpness=1_000.0)
        face_boxes = torch.tensor([[0.25, 0.25, 0.75, 0.75]])
        base_prediction = torch.zeros(1, 4, 8, 8)
        mask, stats = objective.face_focus_mask(
            noisy_instance_latents=torch.zeros_like(base_prediction),
            base_model_prediction=base_prediction,
            timesteps=torch.tensor([0]),
            scheduler=None,
            num_train_timesteps=1,
            height=8,
            width=8,
            face_boxes=face_boxes,
        )
        self.assertEqual(mask.shape, (1, 1, 8, 8))
        self.assertEqual(stats["valid_faces"], 1.0)
        self.assertGreater(mask[0, 0, 3, 3].item(), 0.9)
        self.assertLess(mask[0, 0, 0, 0].item(), 0.1)

    def test_face_focus_mask_falls_back_to_ones_without_usable_box(self):
        objective = LatentBackgroundPreservationObjective(mask_edge_sharpness=1_000.0)
        # Dataset sentinel for an out-of-crop / missing face.
        face_boxes = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
        base_prediction = torch.zeros(1, 4, 8, 8)
        mask, stats = objective.face_focus_mask(
            noisy_instance_latents=torch.zeros_like(base_prediction),
            base_model_prediction=base_prediction,
            timesteps=torch.tensor([0]),
            scheduler=None,
            num_train_timesteps=1,
            height=8,
            width=8,
            face_boxes=face_boxes,
        )
        self.assertEqual(stats["valid_faces"], 0.0)
        torch.testing.assert_close(mask, torch.ones_like(mask))


if __name__ == "__main__":
    unittest.main()
