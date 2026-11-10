"""
Unit tests for Hugging Face loader and Benchmark evaluation suite.
"""

import unittest
import torch

from configs.config import (
    SystemConfig,
    EncoderConfig,
    SIGRegConfig,
    DecoderConfig,
    PredictorConfig,
)
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.models.encoder import TextJEPAEncoder
from src.models.decoder import ConditionalTextDecoder
from src.models.predictor import LatentPredictor
from src.models.pipeline import GoodViewTextJEPA
from src.evaluation.benchmark import (
    compute_effective_rank,
    benchmark_encoder_geometry,
    benchmark_decoder_reconstruction,
    benchmark_predictor_reasoning,
)


class TestBenchmarkAndLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = [
            "Water freezes into solid ice at zero degrees Celsius.",
            "Water boils and creates steam once it gets to one hundred degrees.",
            "Massive objects exert a gravitational force that attracts bodies.",
            "Light travels through a vacuum at three hundred thousand km/s.",
            "Plants convert sunlight and carbon dioxide into oxygen.",
        ]
        cls.tokenizer = BPETokenizer()
        cls.tokenizer.train(cls.texts, target_vocab_size=100)

        vocab_sz = cls.tokenizer.vocab_size
        cls.config = SystemConfig(
            encoder=EncoderConfig(vocab_size=vocab_sz, d_model=32, latent_dim=16, max_seq_len=24),
            sigreg=SIGRegConfig(),
            decoder=DecoderConfig(vocab_size=vocab_sz, d_model=32, latent_dim=16, max_seq_len=24, num_soft_tokens=2),
            predictor=PredictorConfig(latent_dim=16, d_model=32, max_steps=4),
            device="cpu",
        )

        cls.encoder = TextJEPAEncoder(cls.config.encoder)
        cls.decoder = ConditionalTextDecoder(cls.config.decoder)
        cls.predictor = LatentPredictor(cls.config.predictor)

        cls.pipeline = GoodViewTextJEPA(
            config=cls.config,
            tokenizer=cls.tokenizer,
            encoder=cls.encoder,
            decoder=cls.decoder,
            predictor=cls.predictor,
        )

    def test_effective_rank(self):
        # Orthogonal / random embeddings should have high effective rank
        embeds = torch.randn(100, 16)
        eff_rank = compute_effective_rank(embeds)
        self.assertGreater(eff_rank, 5.0)

        # Rank-1 collapsed embeddings should have effective rank close to 1
        collapsed = torch.ones(100, 1) * torch.randn(1, 16)
        eff_rank_col = compute_effective_rank(collapsed)
        self.assertLess(eff_rank_col, 2.0)

    def test_geometry_benchmark(self):
        pairs = [
            (self.texts[0], self.texts[1]),
            (self.texts[2], self.texts[3]),
        ]
        geom = benchmark_encoder_geometry(self.pipeline, pairs, num_samples=2)
        self.assertIn("alignment", geom)
        self.assertIn("uniformity", geom)
        self.assertIn("effective_rank", geom)
        self.assertGreater(geom["effective_rank"], 0.0)

    def test_decoder_reconstruction_benchmark(self):
        dec_metrics = benchmark_decoder_reconstruction(
            self.pipeline, self.texts[:3], max_samples=3, max_tokens=10
        )
        self.assertIn("latent_cosine_recovery", dec_metrics)
        self.assertIn("token_overlap_f1", dec_metrics)

    def test_predictor_reasoning_benchmark(self):
        trajectories = [
            [self.texts[0], self.texts[1], self.texts[2]],
            [self.texts[2], self.texts[3], self.texts[4]],
        ]
        pred_metrics = benchmark_predictor_reasoning(
            self.pipeline, trajectories, pool_size=5, max_samples=2
        )
        self.assertIn("next_step_p@1", pred_metrics)
        self.assertIn("next_step_mrr", pred_metrics)
        self.assertIn("drift_cosine_step1", pred_metrics)


if __name__ == "__main__":
    unittest.main()

