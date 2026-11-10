"""
Unit tests for ConditionalTextDecoder.
"""

import unittest
import torch
import torch.nn.functional as F
from configs.config import DecoderConfig
from src.models.decoder import ConditionalTextDecoder


class TestConditionalDecoder(unittest.TestCase):
    def setUp(self):
        self.config = DecoderConfig(
            vocab_size=100,
            max_seq_len=32,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            latent_dim=64,
            num_soft_tokens=4,
            noise_scale=0.05,
            num_styles=4,
            style_dim=16,
        )
        self.decoder = ConditionalTextDecoder(self.config)

    def test_soft_prompt_projection(self):
        B = 4
        e = torch.randn(B, self.config.latent_dim)
        style_ids = torch.randint(0, self.config.num_styles, (B,))
        soft_tokens = self.decoder.project_soft_prompts(e, style_ids=style_ids)
        self.assertEqual(soft_tokens.shape, (B, self.config.num_soft_tokens, self.config.d_model))

    def test_decoder_forward(self):
        B, L = 4, 16
        e = torch.randn(B, self.config.latent_dim)
        input_ids = torch.randint(0, self.config.vocab_size, (B, L))
        attn_mask = torch.ones(B, L, dtype=torch.long)
        attn_mask[:, 12:] = 0
        style_ids = torch.zeros(B, dtype=torch.long)

        logits = self.decoder(e, input_ids, attention_mask=attn_mask, style_ids=style_ids)
        self.assertEqual(logits.shape, (B, L, self.config.vocab_size))

        # Check loss backpropagation
        targets = torch.randint(0, self.config.vocab_size, (B, L))
        loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        loss.backward()

        for param in self.decoder.parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_autoregressive_generation(self):
        e = torch.randn(1, self.config.latent_dim)
        gen = self.decoder.generate(
            e,
            bos_id=1,
            eos_id=2,
            pad_id=0,
            style_id=0,
            max_new_tokens=8,
            temperature=1.0,
        )
        self.assertEqual(gen.dim(), 2)
        self.assertEqual(gen.size(0), 1)
        self.assertLessEqual(gen.size(1), 9)  # bos + up to 8 tokens
        self.assertEqual(gen[0, 0].item(), 1)  # BOS token


if __name__ == "__main__":
    unittest.main()
