"""
PyTorch Dataset and DataLoader for sequential multi-step reasoning traces.
Used to train the Latent State-Transition Predictor P_phi.
"""

from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.view_generator import GoodViewGenerator


class SequentialReasoningDataset(Dataset):
    """
    Dataset representing multi-step reasoning trajectories.
    Each item contains T sequential semantic steps [x^(1), ..., x^(T)].
    """

    def __init__(
        self,
        tokenizer: BPETokenizer,
        num_samples: int = 500,
        generator: Optional[GoodViewGenerator] = None,
        max_steps: int = 8,
        max_seq_len: int = 48,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.generator = generator or GoodViewGenerator(seed=seed)
        self.max_steps = max_steps
        self.max_seq_len = max_seq_len
        self.style_to_id = {s: i for i, s in enumerate(GoodViewGenerator.STYLES)}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        step_texts, step_styles, topic = self.generator.sample_reasoning_trajectory()

        # Truncate steps to max_steps if needed
        actual_steps = min(len(step_texts), self.max_steps)
        step_texts = step_texts[:actual_steps]
        step_styles = step_styles[:actual_steps]

        all_input_ids = torch.full((self.max_steps, self.max_seq_len), self.tokenizer.pad_id, dtype=torch.long)
        all_masks = torch.zeros((self.max_steps, self.max_seq_len), dtype=torch.long)
        all_styles = torch.zeros(self.max_steps, dtype=torch.long)
        step_valid_mask = torch.zeros(self.max_steps, dtype=torch.bool)

        for step_idx in range(actual_steps):
            enc = self.tokenizer.encode(
                step_texts[step_idx],
                add_bos=True,
                add_eos=True,
                max_len=self.max_seq_len,
                pad=True,
            )
            all_input_ids[step_idx] = enc["input_ids"]
            all_masks[step_idx] = enc["attention_mask"]
            all_styles[step_idx] = self.style_to_id.get(step_styles[step_idx], 0)
            step_valid_mask[step_idx] = True

        return {
            "step_input_ids": all_input_ids,  # (max_steps, max_seq_len)
            "step_attention_mask": all_masks,  # (max_steps, max_seq_len)
            "step_styles": all_styles,  # (max_steps,)
            "step_valid_mask": step_valid_mask,  # (max_steps,)
            "actual_steps": torch.tensor(actual_steps, dtype=torch.long),
        }
