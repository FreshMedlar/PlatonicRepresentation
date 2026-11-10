"""
Training utilities for optimization, learning rate schedules, and checkpointing.
"""

import math
import os
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.01) -> AdamW:
    """
    Creates AdamW optimizer with weight decay applied only to weights, not biases or LayerNorms.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(optim_groups, lr=lr)


def get_cosine_schedule_with_warmup(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Cosine annealing learning rate scheduler with linear warmup.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    state_dict: Dict,
    save_dir: str,
    filename: str = "checkpoint.pt",
):
    """Saves model/training state to disk."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state_dict, filepath)
    return filepath


def load_checkpoint(filepath: str, map_location: str = "cpu") -> Dict:
    """Loads checkpoint dictionary from disk."""
    return torch.load(filepath, map_location=map_location, weights_only=True)
