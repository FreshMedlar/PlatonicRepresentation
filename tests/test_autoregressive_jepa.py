"""
Unit tests for Autoregressive Predictive JEPA components:
- Sentence splitting & sliding contiguous windows
- Pure MSE predictor loss without SIGReg interference
- Joint autoregressive + good-view forward/backward autograd flow
"""

import unittest
import torch

from configs.config import EncoderConfig, PredictorConfig
from src.data.autoregressive_corpus_loader import split_clean_sentences
from src.models.encoder import TextJEPAEncoder
from src.models.predictor import LatentPredictor
from src.models.sigreg import SIGRegLoss
from src.models.sadl import SADLLoss
from src.training.autoregressive_trainer import compute_joint_autoregressive_step, compute_symmetric_infonce


class TestAutoregressiveJEPA(unittest.TestCase):
    def test_split_clean_sentences_filters_noise(self):
        doc = (
            "First, heat the pan over medium heat until warm. "
            "Then add two tablespoons of olive oil slowly.\n"
            "Visit https://example.com for more tips.\n"
            "Short.\n"
            "Finally, stir the vegetables continuously for five minutes."
        )
        sents = split_clean_sentences(doc)
        self.assertEqual(len(sents), 3)
        self.assertTrue(all("https://" not in s for s in sents))

    def test_predictor_pure_mse_when_reg_weight_zero(self):
        cfg = PredictorConfig(
            latent_dim=32, max_steps=6, d_model=32, num_layers=1, num_heads=2, d_ff=64, reg_weight=0.0
        )
        pred = LatentPredictor(cfg)
        z_seq = torch.randn(4, 5, 32)
        valid = torch.ones(4, 5, dtype=torch.bool)
        tot, mse, reg = pred.compute_loss(z_seq, valid_mask=valid)
        self.assertAlmostEqual(tot.item(), mse.item(), places=6)
        self.assertEqual(reg.item(), 0.0)

    def test_joint_autoregressive_step_gradients(self):
        enc_cfg = EncoderConfig(
            vocab_size=50, max_seq_len=12, d_model=32, num_layers=1, num_heads=2, d_ff=64, latent_dim=16
        )
        pred_cfg = PredictorConfig(
            latent_dim=16, max_steps=4, d_model=32, num_layers=1, num_heads=2, d_ff=64, reg_weight=0.0
        )
        encoder = TextJEPAEncoder(enc_cfg)
        predictor = LatentPredictor(pred_cfg)
        sigreg = SIGRegLoss(latent_dim=16, num_projections=32, knots=9)
        encoder.train()
        predictor.train()

        doc_ids = torch.randint(1, 50, (3, 4, 12))
        doc_masks = torch.ones(3, 4, 12, dtype=torch.long)
        doc_valid = torch.tensor(
            [[True, True, True, True], [True, True, True, False], [True, True, True, True]],
            dtype=torch.bool,
        )
        gv1_ids = torch.randint(1, 50, (4, 12))
        gv1_masks = torch.ones(4, 12, dtype=torch.long)
        gv2_ids = torch.randint(1, 50, (4, 12))
        gv2_masks = torch.ones(4, 12, dtype=torch.long)

        out = compute_joint_autoregressive_step(
            encoder, predictor, sigreg,
            doc_ids, doc_masks, doc_valid,
            gv1_ids, gv1_masks, gv2_ids, gv2_masks,
        )
        for key in ("total_loss", "ar_mse", "ar_cos", "gv_mse", "reg_loss"):
            self.assertTrue(torch.isfinite(out[key]))

        out["total_loss"].backward()
        enc_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
        pred_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.parameters())
        self.assertTrue(enc_has_grad)
        self.assertTrue(pred_has_grad)

    def test_joint_autoregressive_step_with_sadl_and_infonce(self):
        enc_cfg = EncoderConfig(
            vocab_size=50, max_seq_len=12, d_model=32, num_layers=1, num_heads=2, d_ff=64, latent_dim=16, dropout=0.0
        )
        pred_cfg = PredictorConfig(
            latent_dim=16, max_steps=4, d_model=32, num_layers=1, num_heads=2, d_ff=64, reg_weight=0.0
        )
        encoder = TextJEPAEncoder(enc_cfg)
        predictor = LatentPredictor(pred_cfg)
        sigreg = SIGRegLoss(latent_dim=16, num_projections=32, knots=9)
        sadl = SADLLoss()
        encoder.train()
        predictor.train()

        doc_ids = torch.randint(1, 50, (4, 4, 12))
        doc_masks = torch.ones(4, 4, 12, dtype=torch.long)
        doc_valid = torch.ones(4, 4, dtype=torch.bool)
        gv1_ids = torch.randint(1, 50, (6, 12))
        gv1_masks = torch.ones(6, 12, dtype=torch.long)
        gv2_ids = torch.randint(1, 50, (6, 12))
        gv2_masks = torch.ones(6, 12, dtype=torch.long)

        nce = compute_symmetric_infonce(torch.randn(6, 16), torch.randn(6, 16))
        self.assertTrue(torch.isfinite(nce) and nce.item() > 0.0)

        out = compute_joint_autoregressive_step(
            encoder, predictor, sigreg,
            doc_ids, doc_masks, doc_valid,
            gv1_ids, gv1_masks, gv2_ids, gv2_masks,
            reg_weight=0.10, sadl=sadl, supply_weight=0.10,
            ar_nce_weight=0.35, gv_nce_weight=0.25,
        )
        for key in ("total_loss", "ar_mse", "ar_cos", "gv_mse", "reg_loss", "supply_loss"):
            self.assertTrue(torch.isfinite(out[key]))
        out["total_loss"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.parameters()))


if __name__ == "__main__":
    unittest.main()
