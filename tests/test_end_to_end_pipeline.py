"""
Full end-to-end integration test for Good-View Text JEPA System.
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
from src.data.view_generator import GoodViewGenerator
from src.data.parallel_dataset import GoodViewPairDataset
from src.data.sequential_dataset import SequentialReasoningDataset
from src.models.encoder import TextJEPAEncoder
from src.models.decoder import ConditionalTextDecoder
from src.models.predictor import LatentPredictor
from src.models.pipeline import GoodViewTextJEPA
from src.training.train_encoder import train_jepa_encoder
from src.training.train_decoder import train_conditional_decoder
from src.training.train_predictor import train_latent_predictor


class TestEndToEndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Setup Generator & Tokenizer
        cls.generator = GoodViewGenerator(seed=42)
        corpus = cls.generator.get_all_texts()

        cls.tokenizer = BPETokenizer()
        cls.tokenizer.train(corpus, target_vocab_size=150)
        actual_vocab_size = cls.tokenizer.vocab_size

        # 2. Compact configuration for fast integration test
        cls.latent_dim = 32
        cls.d_model = 32
        cls.config = SystemConfig(
            encoder=EncoderConfig(
                vocab_size=actual_vocab_size,
                max_seq_len=24,
                d_model=cls.d_model,
                num_layers=1,
                num_heads=2,
                d_ff=64,
                latent_dim=cls.latent_dim,
            ),
            sigreg=SIGRegConfig(
                lam=1.0,
                num_projections=16,
                num_integration_steps=9,
                reg_weight=0.05,
            ),
            decoder=DecoderConfig(
                vocab_size=actual_vocab_size,
                max_seq_len=24,
                d_model=cls.d_model,
                num_layers=1,
                num_heads=2,
                d_ff=64,
                latent_dim=cls.latent_dim,
                num_soft_tokens=2,
                noise_scale=0.05,
                num_styles=4,
                style_dim=8,
            ),
            predictor=PredictorConfig(
                latent_dim=cls.latent_dim,
                max_steps=4,
                d_model=cls.d_model,
                num_layers=1,
                num_heads=2,
                d_ff=64,
                reg_weight=0.05,
            ),
            device="cpu",
        )

        # 3. Instantiate models
        cls.encoder = TextJEPAEncoder(cls.config.encoder)
        cls.decoder = ConditionalTextDecoder(cls.config.decoder)
        cls.predictor = LatentPredictor(cls.config.predictor)

    def test_full_pipeline_training_and_inference(self):
        # Create datasets
        pair_dataset = GoodViewPairDataset(
            tokenizer=self.tokenizer,
            num_samples=32,
            generator=self.generator,
            max_seq_len=24,
        )
        seq_dataset = SequentialReasoningDataset(
            tokenizer=self.tokenizer,
            num_samples=32,
            generator=self.generator,
            max_steps=4,
            max_seq_len=24,
        )

        # Stage 1: Train Encoder
        enc_hist = train_jepa_encoder(
            encoder=self.encoder,
            dataset=pair_dataset,
            config=self.config,
            epochs=1,
            batch_size=8,
            verbose=False,
        )
        self.assertTrue(len(enc_hist["total_loss"]) > 0)

        # Stage 2: Train Decoder
        dec_hist = train_conditional_decoder(
            decoder=self.decoder,
            encoder=self.encoder,
            dataset=pair_dataset,
            tokenizer=self.tokenizer,
            config=self.config,
            epochs=1,
            batch_size=8,
            verbose=False,
        )
        self.assertTrue(len(dec_hist["ce_loss"]) > 0)

        # Stage 3: Train Predictor
        pred_hist = train_latent_predictor(
            predictor=self.predictor,
            encoder=self.encoder,
            dataset=seq_dataset,
            config=self.config,
            epochs=1,
            batch_size=8,
            verbose=False,
        )
        self.assertTrue(len(pred_hist["pred_total_loss"]) > 0)

        # Assemble full pipeline
        pipeline = GoodViewTextJEPA(
            config=self.config,
            tokenizer=self.tokenizer,
            encoder=self.encoder,
            decoder=self.decoder,
            predictor=self.predictor,
        )

        # Test inference
        prompt = "Thermal energy is supplied to the liquid container."
        result = pipeline.run_inference(prompt=prompt, reasoning_steps=2, style_id=0, max_tokens_per_step=10)

        self.assertIn("prompt", result)
        self.assertIn("latent_trajectory", result)
        self.assertIn("rendered_steps", result)
        self.assertIn("full_response", result)
        self.assertEqual(len(result["rendered_steps"]), 2)
        self.assertEqual(result["latent_trajectory"].shape, (1, 3, self.latent_dim))

        # Test semantic invariance comparison
        p1 = "Water freezes into solid ice at zero degrees Celsius."
        p2 = "Whenever water hits zero degrees Celsius it turns into solid ice."
        inv_metrics = pipeline.compare_semantic_invariance(p1, p2)
        self.assertIn("cosine_similarity", inv_metrics)
        self.assertIn("l2_distance", inv_metrics)


if __name__ == "__main__":
    unittest.main()
