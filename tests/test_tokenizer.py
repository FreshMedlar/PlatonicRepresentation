"""
Unit tests for BPETokenizer.
"""

import os
import tempfile
import unittest
import torch
from src.tokenizer.bpe_tokenizer import BPETokenizer


class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast dark fox leaped across a sleeping hound.",
            "Water boils at one hundred degrees Celsius at sea level.",
            "The boiling point of water is 100 C under standard atmospheric pressure.",
            "Platonic representation hypothesis states that models converge to the same representation.",
            "Different neural networks map the world into identical invariant semantic geometries.",
        ]
        self.tokenizer = BPETokenizer()
        self.tokenizer.train(self.corpus, target_vocab_size=120)

    def test_vocab_size_and_special_tokens(self):
        self.assertGreater(self.tokenizer.vocab_size, len(self.tokenizer.special_tokens))
        self.assertIn("[PAD]", self.tokenizer.vocab)
        self.assertIn("[UNK]", self.tokenizer.vocab)
        self.assertIn("[BOS]", self.tokenizer.vocab)
        self.assertIn("[EOS]", self.tokenizer.vocab)
        self.assertIn("[MASK]", self.tokenizer.vocab)

    def test_encode_and_decode(self):
        text = "The quick brown fox jumps."
        enc = self.tokenizer.encode(text, add_bos=True, add_eos=True, max_len=16, pad=True)
        self.assertEqual(enc["input_ids"].shape, (16,))
        self.assertEqual(enc["attention_mask"].shape, (16,))
        self.assertEqual(enc["input_ids"][0].item(), self.tokenizer.bos_id)

        decoded = self.tokenizer.decode(enc["input_ids"])
        self.assertIn("fox", decoded.lower())

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.tokenizer.save(tmp_path)
            loaded_tok = BPETokenizer.load(tmp_path)
            self.assertEqual(self.tokenizer.vocab_size, loaded_tok.vocab_size)
            self.assertEqual(self.tokenizer.merges, loaded_tok.merges)

            text = "Water boils at one hundred degrees."
            enc1 = self.tokenizer.encode(text)
            enc2 = loaded_tok.encode(text)
            self.assertTrue(torch.equal(enc1["input_ids"], enc2["input_ids"]))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
