"""
Unit tests for SADL (Supply And Demand Loss) and encoder/benchmark integration.
"""

import unittest
import torch

from configs.config import DecoderConfig, EncoderConfig, PredictorConfig, SADLConfig, SIGRegConfig, SystemConfig
from src.data.view_generator import GoodViewGenerator
from src.evaluation.benchmark import benchmark_encoder_geometry
from src.models.decoder import ConditionalTextDecoder
from src.models.encoder import BatchStandardization1d, TextJEPAEncoder
from src.models.pipeline import GoodViewTextJEPA
from src.models.predictor import LatentPredictor
from src.models.sadl import (
    SADLLoss,
    compute_excitation,
    compute_representation_stats,
    compute_transmission,
)
from src.models.sigreg import SIGRegLoss, compute_lejepa_loss
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.autoregressive_trainer import compute_joint_autoregressive_step


class TestSADL(unittest.TestCase):
    def test_excitation_bounds_and_shared_mean_sensitivity(self):
        torch.manual_seed(0)
        h_iso = torch.randn(64, 16)
        k_iso = compute_excitation(h_iso)
        self.assertGreater(k_iso.item(), 0.7)
        self.assertLessEqual(k_iso.item(), 1.0)

        # Adding a large shared vector across all samples must drop excitation sharply
        h_shared = h_iso + 20.0 * torch.ones(1, 16)
        k_shared = compute_excitation(h_shared)
        self.assertLess(k_shared.item(), 0.1)

        # Constant representation has zero excitation
        h_const = torch.ones(32, 16) * 3.0
        self.assertAlmostEqual(compute_excitation(h_const).item(), 0.0, places=5)

    def test_excitation_scale_and_orthogonal_invariance(self):
        torch.manual_seed(1)
        u = torch.randn(48, 12)
        q_mat, _ = torch.linalg.qr(torch.randn(12, 12))
        base = compute_excitation(u).item()
        scaled_rot = compute_excitation(3.7 * (u @ q_mat)).item()
        self.assertAlmostEqual(base, scaled_rot, places=5)

    def test_transmission_bounds_and_radial_invariance(self):
        torch.manual_seed(2)
        h = torch.randn(64, 24, requires_grad=True)
        proj = torch.randn(24, 16)
        z = (h.detach() @ proj)

        k_tr_aligned = compute_transmission(h, z)
        self.assertGreater(k_tr_aligned.item(), 0.7)
        self.assertLessEqual(k_tr_aligned.item(), 1.0)

        # Radial scaling of H should preserve k_tr and yield zero radial gradient
        h_scale = torch.randn(64, 24, requires_grad=True)
        val = compute_transmission(h_scale, z)
        val.backward()
        radial_dot = (h_scale * h_scale.grad).sum().item()
        self.assertAlmostEqual(radial_dot, 0.0, places=4)

        # Independent random variables with sufficient samples (N >> d_H + d_Z) yield low transmission
        h_ind = torch.randn(512, 24)
        z_ind = torch.randn(512, 16)
        k_tr_rand = compute_transmission(h_ind, z_ind)
        self.assertLess(k_tr_rand.item(), 0.25)

    def test_sadl_one_sided_gate_stop_grad_and_rng_safety(self):
        torch.manual_seed(3)
        sigreg = SIGRegLoss(latent_dim=16, num_projections=32, knots=9)
        sadl = SADLLoss()

        # 1. Active deficit (shared component dominates H, while Z is standard normal)
        h = (0.05 * torch.randn(64, 24) + 10.0).requires_grad_(True)
        z = torch.randn(64, 16, requires_grad=True)
        d_loss = sigreg(z)

        torch.manual_seed(999)
        rng_before = torch.get_rng_state()
        s_loss, stats = sadl(h, z, demand_loss=d_loss, demand_fn=sigreg, return_stats=True)
        # Check D(0) caching didn't alter RNG state (call again with fresh cache too)
        sadl_fresh = SADLLoss()
        torch.manual_seed(999)
        _ = sadl_fresh(h.detach(), z.detach(), demand_loss=d_loss.detach(), demand_fn=sigreg)
        rng_after = torch.get_rng_state()
        self.assertTrue(torch.equal(rng_before, rng_after))

        self.assertLess(stats["k"], stats["q"])
        self.assertGreater(s_loss.item(), 0.0)

        # Verify gradient flows only to H, never to Z through S
        s_loss.backward()
        self.assertIsNotNone(h.grad)
        self.assertGreater(h.grad.abs().sum().item(), 0.0)
        self.assertIsNone(z.grad)

        # 2. Sufficient supply (k >= q -> S == 0)
        h_good = torch.randn(128, 16, requires_grad=True)
        # Shifted/scaled z keeps linear correlation (high k_tr) while having low q_exc and high D(Z)
        z_low_demand = (0.05 * h_good.detach() + 10.0).requires_grad_(True)
        d_bad = sigreg(z_low_demand)
        s_inactive, stats_ok = sadl(h_good, z_low_demand, demand_loss=d_bad, demand_fn=sigreg, return_stats=True)
        self.assertGreaterEqual(stats_ok["k"], stats_ok["q"])
        self.assertEqual(s_inactive.item(), 0.0)

    def test_batch_standardization_no_nan_on_constant_input(self):
        bs = BatchStandardization1d(8)
        x = torch.ones(16, 8, requires_grad=True)
        y = bs(x)
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_encoder_sadl_integration_and_repair(self):
        torch.manual_seed(4)
        enc_cfg = EncoderConfig(
            vocab_size=50, max_seq_len=12, d_model=32, num_layers=1, num_heads=2, d_ff=64, latent_dim=16
        )
        pred_cfg = PredictorConfig(
            latent_dim=16, max_steps=4, d_model=32, num_layers=1, num_heads=2, d_ff=64, reg_weight=0.0
        )
        encoder = TextJEPAEncoder(enc_cfg)
        predictor = LatentPredictor(pred_cfg)
        sigreg = SIGRegLoss(latent_dim=16, num_projections=32, knots=9)
        sadl = SADLLoss()

        ids1 = torch.randint(1, 50, (16, 12))
        ids2 = torch.randint(1, 50, (16, 12))
        mask = torch.ones(16, 12, dtype=torch.long)

        h1, e1 = encoder(ids1, attention_mask=mask, return_both=True)
        h2, e2 = encoder(ids2, attention_mask=mask, return_both=True)
        self.assertEqual(h1.shape, (16, 32))
        self.assertEqual(e1.shape, (16, 16))

        tot, inv, reg = compute_lejepa_loss(
            e1, e2, sigreg, reg_weight=0.02, h1=h1, h2=h2, sadl_loss_fn=sadl, supply_weight=0.01
        )
        self.assertTrue(torch.isfinite(tot))
        tot.backward()

        # Verify joint autoregressive step with SADL
        encoder.zero_grad()
        predictor.zero_grad()
        doc_ids = torch.randint(1, 50, (3, 4, 12))
        doc_masks = torch.ones(3, 4, 12, dtype=torch.long)
        doc_valid = torch.ones(3, 4, dtype=torch.bool)
        out = compute_joint_autoregressive_step(
            encoder, predictor, sigreg,
            doc_ids, doc_masks, doc_valid,
            ids1[:4], mask[:4], ids2[:4], mask[:4],
            sadl=sadl, supply_weight=0.01,
        )
        self.assertIn("supply_loss", out)
        self.assertTrue(torch.isfinite(out["total_loss"]))
        out["total_loss"].backward()

    def test_benchmark_reports_supply_demand_metrics(self):
        gen = GoodViewGenerator(seed=7)
        texts = gen.get_all_texts()
        tok = BPETokenizer()
        tok.train(texts, target_vocab_size=120)
        cfg = SystemConfig(
            encoder=EncoderConfig(vocab_size=tok.vocab_size, d_model=32, num_heads=2, num_layers=1, d_ff=64, latent_dim=16, max_seq_len=24),
            sigreg=SIGRegConfig(),
            sadl=SADLConfig(),
            decoder=DecoderConfig(vocab_size=tok.vocab_size, d_model=32, num_heads=2, num_layers=1, d_ff=64, latent_dim=16, max_seq_len=24, num_soft_tokens=2),
            predictor=PredictorConfig(latent_dim=16, d_model=32, num_heads=2, num_layers=1, d_ff=64, max_steps=4),
        )
        pipe = GoodViewTextJEPA(cfg, tok)
        geom = benchmark_encoder_geometry(pipe, [(texts[0], texts[1]), (texts[2], texts[3])], num_samples=2)
        for key in ("f_H", "f_Z", "erank_H", "erank_Z", "mean_cos_H", "mean_cos_Z", "k_exc", "k_tr", "q_exc"):
            self.assertIn(key, geom)


if __name__ == "__main__":
    unittest.main()
