import os
import tempfile
import unittest
import torch

from configs.baby_config import (
    ACTIONS,
    BabyJEPAConfig,
    BabyPredictorConfig,
    BabyTrainingConfig,
    BabyVisionConfig,
    BabyWordConfig,
    TabletopEnvConfig,
)
from src.env.tabletop_world import TabletopWorld
from src.evaluation.baby_benchmark import (
    FrozenDINOEncoder,
    build_eval_datasets,
    compute_linear_cka,
    compute_mutual_knn,
    evaluate_baby_jepa,
    evaluate_ridge_probe,
    fit_orthogonal_procrustes,
)
from src.models.baby_jepa import BabyJEPASystem, whitened_mse
from src.training.baby_jepa_trainer import BabyJEPATrainer


class TestTabletopWorld(unittest.TestCase):
    def test_render_all_shapes_and_actions(self):
        cfg = TabletopEnvConfig(image_size=32, num_shapes=5, teacher_word_prob=0.5)
        world = TabletopWorld(cfg, batch_size=5, device="cpu", seed=7)
        obs = world.reset(confound_prob=0.0, biased_view=False)
        # Force one of each shape 0..4
        world.shape_id = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        imgs = world.render()
        self.assertEqual(imgs.shape, (5, 3, 32, 32))
        self.assertTrue(torch.isfinite(imgs).all())
        self.assertGreaterEqual(float(imgs.min()), 0.0)
        self.assertLessEqual(float(imgs.max()), 1.0)

        for aid in range(len(ACTIONS)):
            step_obs = world.step(torch.full((5,), aid, dtype=torch.long))
            self.assertEqual(step_obs["images"].shape, (5, 3, 32, 32))
            self.assertTrue(torch.isfinite(step_obs["images"]).all())

    def test_confounder_coupling_and_curriculum(self):
        cfg = TabletopEnvConfig(image_size=32, num_shapes=4, confound_prob=1.0)
        world = TabletopWorld(cfg, batch_size=32, device="cpu", seed=11)
        obs = world.reset(confound_prob=1.0, biased_view=True)
        self.assertTrue(torch.equal(obs["color_id"], obs["shape_id"] % cfg.num_colors))

        world.set_num_shapes(5)
        obs2 = world.reset(confound_prob=0.0, biased_view=False)
        self.assertTrue((obs2["shape_id"] < 5).all())


class TestBabyJEPAModels(unittest.TestCase):
    def _small_config(self, arm: str = "active") -> BabyJEPAConfig:
        return BabyJEPAConfig(
            env=TabletopEnvConfig(image_size=32, episode_length=4, teacher_word_prob=0.5),
            vision=BabyVisionConfig(image_size=32, d_model=64, latent_dim=32),
            predictor=BabyPredictorConfig(
                latent_dim=32, d_model=48, num_layers=2, num_heads=4, d_ff=96, ensemble_size=3, max_steps=6
            ),
            word=BabyWordConfig(d_word=32, latent_dim=32),
            training=BabyTrainingConfig(
                arm=arm, total_episodes=8, batch_envs=4, batch_size=4, eval_every_episodes=4, device="cpu"
            ),
        )

    def test_forward_loss_and_action_scoring(self):
        cfg = self._small_config()
        sys = BabyJEPASystem(cfg)
        imgs = torch.rand(4, 4, 3, 32, 32)
        acts = torch.randint(0, len(ACTIONS), (4, 3))
        wids = torch.randint(1, cfg.word.vocab_size, (4, 4))
        hw = torch.tensor([[True, False, True, False]] * 4)

        loss, metrics = sys.compute_training_loss(imgs, acts, wids, hw)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIn("gv_inv_loss", metrics)
        self.assertIn("temporal_wmse", metrics)
        self.assertIn("cross_modal_loss", metrics)

        # Verify action scoring shapes (disagreement + spectral)
        z_ctx = torch.randn(4, 2, 32)
        a_ctx = torch.randint(0, len(ACTIONS), (4, 1))
        recent_z = torch.randn(20, 32)
        scores, dis = sys.predictor_ensemble.score_candidate_actions(
            z_ctx, a_ctx, recent_z=recent_z, spectral_bonus_weight=0.25
        )
        stats = sys.predictor_ensemble.compute_spectral_stats(recent_z)
        scores_cached, dis_cached = sys.predictor_ensemble.score_candidate_actions(
            z_ctx, a_ctx, spectral_bonus_weight=0.25, spectral_stats=stats
        )
        self.assertTrue(torch.allclose(scores, scores_cached, atol=1e-5))
        self.assertTrue(torch.allclose(dis, dis_cached, atol=1e-5))
        self.assertEqual(scores.shape, (4, len(ACTIONS)))
        self.assertEqual(dis.shape, (4, len(ACTIONS)))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue((dis >= 0).all())

        # Boredom/habituation check: if visited_z contains a state predicted for action 0,
        # action 0 should be discounted when re-scored with visited_z.
        with torch.no_grad():
            z_step1 = z_ctx[:, -1:, :]
            cand_a = torch.arange(len(ACTIONS)).view(1, -1).expand(4, -1)
            a_full = torch.cat([a_ctx.unsqueeze(1).expand(4, len(ACTIONS), 1).reshape(-1, 1), cand_a.reshape(-1, 1)], dim=1)
            z_exp = z_ctx.unsqueeze(1).expand(4, len(ACTIONS), 2, 32).reshape(-1, 2, 32)
            mean_next = sys.predictor_ensemble.forward_all(z_exp, a_full)[:, :, -1, :].mean(dim=0).view(4, len(ACTIONS), 32)
        visited_with_a0 = torch.cat([z_step1, mean_next[:, 0:1, :]], dim=1)
        scores_bored, _ = sys.predictor_ensemble.score_candidate_actions(
            z_ctx, a_ctx, spectral_bonus_weight=0.25, spectral_stats=stats, visited_z=visited_with_a0
        )
        self.assertTrue((scores_bored[:, 0] < scores_cached[:, 0]).all())

    def test_whitened_mse_matches_zero_on_identical(self):
        x = torch.randn(16, 12)
        self.assertLess(float(whitened_mse(x, x)), 1e-6)


class TestBabyBenchmarkAndTrainer(unittest.TestCase):
    def test_cka_knn_and_procrustes_orthogonal_recovery(self):
        torch.manual_seed(0)
        x = torch.randn(80, 16)
        q_true, _ = torch.linalg.qr(torch.randn(16, 16))
        y = x @ q_true

        self.assertAlmostEqual(compute_linear_cka(x, y), 1.0, places=4)
        self.assertAlmostEqual(compute_mutual_knn(x, y, k=5), 1.0, places=4)

        q_fit = fit_orthogonal_procrustes(x, y)
        recon_err = (x @ q_fit - y).pow(2).mean().item()
        self.assertLess(recon_err, 1e-5)

        labels = torch.randint(0, 4, (80,))
        acc = evaluate_ridge_probe(x[:40], labels[:40], x[40:], labels[40:], num_classes=4)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_trainer_smoke_all_arms(self):
        for arm in ("active", "active_spectral", "random", "passive"):
            cfg = BabyJEPAConfig(
                env=TabletopEnvConfig(image_size=32, episode_length=4, teacher_word_prob=0.4),
                vision=BabyVisionConfig(image_size=32, d_model=48, latent_dim=24),
                predictor=BabyPredictorConfig(
                    latent_dim=24, d_model=32, num_layers=1, num_heads=4, d_ff=64, ensemble_size=3, max_steps=6
                ),
                word=BabyWordConfig(d_word=24, latent_dim=24),
                training=BabyTrainingConfig(
                    arm=arm,
                    total_episodes=8,
                    batch_envs=4,
                    batch_size=4,
                    train_steps_per_episode=1,
                    eval_every_episodes=4,
                    device="cpu",
                ),
            )
            splits = build_eval_datasets(cfg.env, device="cpu", num_probe_train=16, num_eval=16, seed=123)
            trainer = BabyJEPATrainer(cfg, eval_splits=splits)
            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_dir = tmpdir if arm == "active_spectral" else None
                res = trainer.run(checkpoint_dir=ckpt_dir)
                if ckpt_dir is not None:
                    self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "checkpoint_active_spectral.pt")))
                    self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "best_active_spectral.pt")))
            self.assertEqual(res["arm"], arm)
            self.assertGreaterEqual(len(res["history"]), 2)
            self.assertIn("procrustes_shape_naming_acc", res["final"])
            if arm != "passive":
                self.assertEqual(sum(res["action_counts"].values()), 8 * 3)

    def test_cached_dino_loader_if_present(self):
        dino = FrozenDINOEncoder.from_local_cache()
        if dino is not None:
            out = dino(torch.rand(2, 3, 64, 64))
            self.assertEqual(out.shape, (2, 384))
            self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
