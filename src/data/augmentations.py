"""
Augmentations for Good-View Text JEPA.
Implements:
1. Random contiguous span trimming (Section 3.2: 50-100% for global views, 15-50% for local views).
2. Light token masking (15% random replacement with [MASK]).
"""

import random
from typing import Dict, List, Optional, Tuple
import torch


class TextSpanTrimmer:
    """
    Trims a random contiguous span from a sequence of tokens or words.
    """

    def __init__(self, min_ratio: float = 0.5, max_ratio: float = 1.0):
        assert 0.0 < min_ratio <= max_ratio <= 1.0
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, tokens: List[int], protected_tokens: Optional[List[int]] = None) -> List[int]:
        if len(tokens) <= 2:
            return list(tokens)

        protected = set(protected_tokens or [])
        # Extract indices that can be trimmed
        valid_indices = [i for i, t in enumerate(tokens) if t not in protected]
        if not valid_indices:
            return list(tokens)

        seq_len = len(valid_indices)
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        span_len = max(1, int(round(seq_len * ratio)))

        max_start = max(0, seq_len - span_len)
        start_offset = random.randint(0, max_start)
        chosen_indices = set(valid_indices[start_offset : start_offset + span_len])

        # Keep protected tokens and the selected span
        result = [t for i, t in enumerate(tokens) if i in chosen_indices or t in protected]
        return result


class LightTokenMasker:
    """
    Applies light token masking (default 15%) by replacing selected tokens with mask_id.
    """

    def __init__(self, mask_id: int, mask_prob: float = 0.15):
        self.mask_id = mask_id
        self.mask_prob = mask_prob

    def __call__(self, token_ids: List[int], protected_tokens: Optional[List[int]] = None) -> List[int]:
        protected = set(protected_tokens or [])
        masked = []
        for t in token_ids:
            if t not in protected and random.random() < self.mask_prob:
                masked.append(self.mask_id)
            else:
                masked.append(t)
        return masked


def augment_token_pair(
    tokens1: List[int],
    tokens2: List[int],
    mask_id: int,
    global_trimmer: Optional[TextSpanTrimmer] = None,
    local_trimmer: Optional[TextSpanTrimmer] = None,
    masker: Optional[LightTokenMasker] = None,
    protected_tokens: Optional[List[int]] = None,
) -> Tuple[List[int], List[int]]:
    """
    Augments a pair of positive text token sequences.
    tokens1 is treated as global view, tokens2 as local (or global) view.
    """
    if global_trimmer is None:
        global_trimmer = TextSpanTrimmer(min_ratio=0.5, max_ratio=1.0)
    if local_trimmer is None:
        local_trimmer = TextSpanTrimmer(min_ratio=0.15, max_ratio=0.5)
    if masker is None:
        masker = LightTokenMasker(mask_id=mask_id, mask_prob=0.15)

    v1 = global_trimmer(tokens1, protected_tokens=protected_tokens)
    v2 = local_trimmer(tokens2, protected_tokens=protected_tokens)

    v1 = masker(v1, protected_tokens=protected_tokens)
    v2 = masker(v2, protected_tokens=protected_tokens)

    return v1, v2
