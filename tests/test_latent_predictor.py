"""
Unit tests for LatentPredictor.
"""

import unittest
import torch
from configs.config import PredictorConfig
from src.models.predictor import LatentPredictor


class TestLatentPredictor(unittest.TestCase):
    def setUp(self):
        self.config = PredictorConfig(
            latent_dim=64,
            max_steps=8,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            dropout=0.1,
            reg_weight=0.05,
        )
        self.predictor = LatentPredictor(self.config)

    def test_predictor_forward(self):
        B, T, D = 4, 6, self.config.latent_dim
        z_seq = torch.randn(B, T, D)
        preds = self.predictor(z_seq)
        self.assertEqual(preds.shape, (B, T, D))

    def test_compute_loss_and_gradients(self):
        B, T, D = 4, 5, self.config.latent_dim
        z_seq = torch.randn(B, T, D)
        valid_mask = torch.ones(B, T, dtype=torch.bool)
        valid_mask[:, 4] = False  # 5th step is padding

        total_loss, mse_loss, reg_loss = self.predictor.compute_loss(z_seq, valid_mask=valid_mask)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(torch.isfinite(mse_loss))
        self.assertTrue(torch.isfinite(reg_loss))

        total_loss.backward()
        for param in self.predictor.parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_latent_rollout(self):
        D = self.config.latent_dim
        z0 = torch.randn(1, D)
        traj = self.predictor.rollout(z0, num_steps=4)
        self.assertEqual(traj.shape, (1, 5, D))  # initial step + 4 rolled-out steps


if __name__ == "__main__":
    unittest.main()
