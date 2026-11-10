"""
Self-contained Byte-Pair Encoding (BPE) Tokenizer.
Operates completely offline without external dependencies.
"""

import json
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union
import torch


class BPETokenizer:
    """
    A lightweight, robust subword BPE tokenizer.
    Supports special tokens, fixed padding/truncation, and JSON serialization.
    """

    def __init__(
        self,
        special_tokens: Optional[List[str]] = None,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        mask_token: str = "[MASK]",
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token

        if special_tokens is None:
            self.special_tokens = [pad_token, unk_token, bos_token, eos_token, mask_token]
        else:
            self.special_tokens = list(special_tokens)
            for st in [pad_token, unk_token, bos_token, eos_token, mask_token]:
                if st not in self.special_tokens:
                    self.special_tokens.append(st)

        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._init_special_tokens()

    def _init_special_tokens(self):
        self.vocab = {}
        self.inverse_vocab = {}
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

    @property
    def pad_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.vocab[self.unk_token]

    @property
    def bos_id(self) -> int:
        return self.vocab[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.vocab[self.eos_token]

    @property
    def mask_id(self) -> int:
        return self.vocab[self.mask_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _pre_tokenize(self, text: str) -> List[str]:
        """Splits into alphanumeric tokens and single punctuation symbols."""
        # Find all words or punctuation marks
        tokens = re.findall(r"\w+|[^\w\s]", text.strip(), re.UNICODE)
        return tokens

    def train(self, texts: List[str], target_vocab_size: int = 500, min_freq: int = 1):
        """
        Trains BPE merges over a collection of texts until reaching target_vocab_size.
        """
        self._init_special_tokens()

        # Build initial word counts
        word_freqs = Counter()
        for text in texts:
            for word in self._pre_tokenize(text):
                # Represent word as tuple of characters + end of word marker '</w>'
                chars = tuple(list(word) + ["</w>"])
                word_freqs[chars] += 1

        # Add all individual characters to vocabulary
        char_set = set()
        for chars in word_freqs:
            for c in chars:
                char_set.add(c)

        for c in sorted(char_set):
            if c not in self.vocab:
                idx = len(self.vocab)
                self.vocab[c] = idx
                self.inverse_vocab[idx] = c

        self.merges = []
        num_merges = max(0, target_vocab_size - len(self.vocab))

        # Iteratively find best pair and merge
        for _ in range(num_merges):
            pair_counts = Counter()
            for chars, freq in word_freqs.items():
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_freq:
                break

            # Create new token
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[merged_token] = idx
                self.inverse_vocab[idx] = merged_token
                self.merges.append(best_pair)

            # Update word frequencies with merge applied
            new_word_freqs = Counter()
            p0, p1 = best_pair
            for chars, freq in word_freqs.items():
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == p0 and chars[i + 1] == p1:
                        new_chars.append(merged_token)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                new_word_freqs[tuple(new_chars)] = freq
            word_freqs = new_word_freqs

            if len(self.vocab) >= target_vocab_size:
                break

    def _tokenize_word(self, word: str) -> List[str]:
        chars = list(word) + ["</w>"]
        for p0, p1 in self.merges:
            merged = p0 + p1
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == p0 and chars[i + 1] == p1:
                    new_chars.append(merged)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        return chars

    def tokenize(self, text: str) -> List[str]:
        words = self._pre_tokenize(text)
        subwords = []
        for word in words:
            subwords.extend(self._tokenize_word(word))
        return subwords

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: Optional[int] = None,
        pad: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes a string into input_ids and attention_mask tensors.
        """
        subwords = self.tokenize(text)
        ids = [self.vocab.get(sw, self.unk_id) for sw in subwords]

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
                if add_eos and ids[-1] != self.eos_id:
                    ids[-1] = self.eos_id
            mask = [1] * len(ids)
            if pad and len(ids) < max_len:
                pad_len = max_len - len(ids)
                ids = ids + [self.pad_id] * pad_len
                mask = mask + [0] * pad_len
        else:
            mask = [1] * len(ids)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list or tensor of token IDs back into text.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = []
        for tid in token_ids:
            if tid in self.inverse_vocab:
                token = self.inverse_vocab[tid]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                if not skip_special_tokens:
                    tokens.append(self.unk_token)

        # Reconstruct text by joining subwords and replacing </w> with spaces
        full_text = "".join(tokens).replace("</w>", " ").strip()
        # Clean up punctuation spacing
        full_text = re.sub(r"\s+([.,!?;:])", r"\1", full_text)
        return full_text

    def save(self, filepath: str):
        """Saves tokenizer state to JSON."""
        data = {
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": self.merges,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Loads tokenizer state from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(special_tokens=data.get("special_tokens"))
        tok.vocab = data["vocab"]
        tok.inverse_vocab = {int(v): k for k, v in data["vocab"].items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        return tok
