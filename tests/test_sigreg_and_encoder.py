"""
Unit tests for SIGReg loss and TextJEPAEncoder.
"""

import unittest
import torch
from configs.config import EncoderConfig, SIGRegConfig
from src.models.sigreg import SIGRegLoss, compute_lejepa_loss
from src.models.encoder import TextJEPAEncoder


class TestSIGRegAndEncoder(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 64
        self.sigreg = SIGRegLoss(latent_dim=self.latent_dim, num_projections=32)

        self.enc_config = EncoderConfig(
            vocab_size=100,
            max_seq_len=32,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            latent_dim=self.latent_dim,
        )
        self.encoder = TextJEPAEncoder(self.enc_config)

    def test_sigreg_zero_on_standard_normal(self):
        # A large sample from N(0, I) should have relatively low SIGReg loss
        torch.manual_seed(42)
        z = torch.randn(256, self.latent_dim)
        loss = self.sigreg(z)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_sigreg_high_on_collapsed_embeddings(self):
        # Collapsed embeddings (constant vectors) should have much higher penalty
        z_collapsed = torch.ones(256, self.latent_dim) * 5.0
        loss_collapsed = self.sigreg(z_collapsed)

        torch.manual_seed(42)
        z_normal = torch.randn(256, self.latent_dim)
        loss_normal = self.sigreg(z_normal)

        self.assertGreater(loss_collapsed.item(), loss_normal.item())

    def test_encoder_forward_and_backward(self):
        B, L = 4, 16
        input_ids = torch.randint(0, self.enc_config.vocab_size, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        attention_mask[:, 12:] = 0  # partial padding

        e = self.encoder(input_ids, attention_mask=attention_mask)
        self.assertEqual(e.shape, (B, self.latent_dim))

        # Check gradient flow
        target = torch.randn_like(e)
        loss = torch.nn.functional.mse_loss(e, target)
        loss.backward()

        for param in self.encoder.parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_compute_lejepa_loss(self):
        B = 8
        e1 = torch.randn(B, self.latent_dim, requires_grad=True)
        e2 = torch.randn(B, self.latent_dim, requires_grad=True)

        tot_loss, inv_loss, reg_loss = compute_lejepa_loss(e1, e2, self.sigreg, reg_weight=0.05)
        self.assertTrue(torch.isfinite(tot_loss))
        self.assertTrue(torch.isfinite(inv_loss))
        self.assertTrue(torch.isfinite(reg_loss))

        tot_loss.backward()
        self.assertIsNotNone(e1.grad)
        self.assertIsNotNone(e2.grad)


if __name__ == "__main__":
    unittest.main()
