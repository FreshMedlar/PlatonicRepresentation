"""
PyTorch Dataset and DataLoader for Good-View text pairs.
Applies span trimming and light masking augmentations.
"""

from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.view_generator import GoodViewGenerator
from src.data.augmentations import TextSpanTrimmer, LightTokenMasker, augment_token_pair


class GoodViewPairDataset(Dataset):
    """
    Dataset generating or loading good-view positive pairs (x1, x2).
    Both views observe the same world state z with independent nuisances.
    """

    def __init__(
        self,
        tokenizer: BPETokenizer,
        num_samples: int = 1000,
        generator: Optional[GoodViewGenerator] = None,
        max_seq_len: int = 64,
        apply_augmentation: bool = True,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.generator = generator or GoodViewGenerator(seed=seed)
        self.max_seq_len = max_seq_len
        self.apply_augmentation = apply_augmentation

        self.global_trimmer = TextSpanTrimmer(min_ratio=0.5, max_ratio=1.0)
        self.local_trimmer = TextSpanTrimmer(min_ratio=0.15, max_ratio=0.5)
        self.masker = LightTokenMasker(mask_id=tokenizer.mask_id, mask_prob=0.15)
        self.style_to_id = {s: i for i, s in enumerate(GoodViewGenerator.STYLES)}

    def __len__(self) -> int:
        return self.num_samples

    def _pad_or_truncate(self, token_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]
        mask = [1] * len(token_ids)
        if len(token_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(token_ids)
            token_ids = token_ids + [self.tokenizer.pad_id] * pad_len
            mask = mask + [0] * pad_len
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t1, t2, concept, s1, s2 = self.generator.sample_positive_pair()

        # Clean unaugmented tokens
        clean_toks1 = self.tokenizer.encode(t1, add_bos=True, add_eos=True)["input_ids"].tolist()
        clean_toks2 = self.tokenizer.encode(t2, add_bos=True, add_eos=True)["input_ids"].tolist()

        toks1 = list(clean_toks1)
        toks2 = list(clean_toks2)

        if self.apply_augmentation:
            prot = [self.tokenizer.bos_id, self.tokenizer.eos_id]
            toks1, toks2 = augment_token_pair(
                toks1,
                toks2,
                mask_id=self.tokenizer.mask_id,
                global_trimmer=self.global_trimmer,
                local_trimmer=self.local_trimmer,
                masker=self.masker,
                protected_tokens=prot,
            )

        ids1, mask1 = self._pad_or_truncate(toks1)
        ids2, mask2 = self._pad_or_truncate(toks2)
        clean_ids1, clean_mask1 = self._pad_or_truncate(clean_toks1)
        clean_ids2, clean_mask2 = self._pad_or_truncate(clean_toks2)

        return {
            "view1_ids": ids1,
            "view1_mask": mask1,
            "view2_ids": ids2,
            "view2_mask": mask2,
            "clean1_ids": clean_ids1,
            "clean1_mask": clean_mask1,
            "clean2_ids": clean_ids2,
            "clean2_mask": clean_mask2,
            "style1_id": torch.tensor(self.style_to_id.get(s1, 0), dtype=torch.long),
            "style2_id": torch.tensor(self.style_to_id.get(s2, 0), dtype=torch.long),
        }
