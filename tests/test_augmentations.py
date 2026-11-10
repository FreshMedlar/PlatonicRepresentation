"""
Unit tests for data augmentations, view generator, and datasets.
"""

import unittest
import torch
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.augmentations import TextSpanTrimmer, LightTokenMasker, augment_token_pair
from src.data.view_generator import GoodViewGenerator
from src.data.parallel_dataset import GoodViewPairDataset
from src.data.sequential_dataset import SequentialReasoningDataset


class TestDataComponents(unittest.TestCase):
    def setUp(self):
        self.generator = GoodViewGenerator(seed=123)
        self.corpus = self.generator.get_all_texts()
        self.tokenizer = BPETokenizer()
        self.tokenizer.train(self.corpus, target_vocab_size=200)

    def test_view_generator_pair(self):
        v1, v2, concept, s1, s2 = self.generator.sample_positive_pair()
        self.assertTrue(len(v1) > 0)
        self.assertTrue(len(v2) > 0)
        self.assertTrue(len(concept) > 0)

    def test_view_generator_trajectory(self):
        steps, styles, topic = self.generator.sample_reasoning_trajectory()
        self.assertGreaterEqual(len(steps), 2)
        self.assertEqual(len(steps), len(styles))

    def test_span_trimmer_bounds(self):
        trimmer = TextSpanTrimmer(min_ratio=0.5, max_ratio=0.8)
        tokens = list(range(100))
        trimmed = trimmer(tokens)
        self.assertGreaterEqual(len(trimmed), 50)
        self.assertLessEqual(len(trimmed), 80)

    def test_token_masker(self):
        masker = LightTokenMasker(mask_id=999, mask_prob=0.5)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        masked = masker(tokens, protected_tokens=[1, 10])
        self.assertEqual(masked[0], 1)
        self.assertEqual(masked[-1], 10)
        self.assertIn(999, masked)

    def test_parallel_dataset(self):
        dataset = GoodViewPairDataset(
            self.tokenizer,
            num_samples=10,
            generator=self.generator,
            max_seq_len=32,
            apply_augmentation=True,
        )
        item = dataset[0]
        self.assertEqual(item["view1_ids"].shape, (32,))
        self.assertEqual(item["view2_ids"].shape, (32,))
        self.assertEqual(item["view1_mask"].shape, (32,))
        self.assertEqual(item["view2_mask"].shape, (32,))

    def test_sequential_dataset(self):
        dataset = SequentialReasoningDataset(
            self.tokenizer,
            num_samples=5,
            generator=self.generator,
            max_steps=4,
            max_seq_len=24,
        )
        item = dataset[0]
        self.assertEqual(item["step_input_ids"].shape, (4, 24))
        self.assertEqual(item["step_attention_mask"].shape, (4, 24))
        self.assertEqual(item["step_valid_mask"].shape, (4,))


if __name__ == "__main__":
    unittest.main()
