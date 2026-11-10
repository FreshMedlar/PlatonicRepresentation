"""
Good-View Text JEPA Backbone Encoder f_theta: X_text -> R^k.
Bidirectional Transformer encoder + MLP projector mapping text to Platonic semantic space.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import EncoderConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding that dynamically extends if sequence length exceeds buffer."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        pe = self._build_pe(d_model, max_len)
        self.register_buffer("pe", pe)

    def _build_pe(self, d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            new_pe = self._build_pe(self.d_model, max(seq_len, self.pe.size(1) * 2)).to(x.device)
            self.pe = new_pe
        return x + self.pe[:, :seq_len]


class TextJEPAEncoder(nn.Module):
    """
    Bidirectional Transformer Encoder with MLP Projector into R^k.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, max_len=config.max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Projection head to latent dimension k (Platonic semantic space)
        self.projector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.latent_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) 1 for valid tokens, 0 for padding
            return_hidden: if True, returns pooled representation before projector
        Returns:
            Latent embedding e in R^k of shape (B, latent_dim)
        """
        B, L = input_ids.shape

        # Token + positional embedding
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)

        # PyTorch TransformerEncoder expects src_key_padding_mask where True indicates PADDED positions
        if attention_mask is not None:
            is_all_pad = (attention_mask.sum(dim=1) == 0)
            src_key_padding_mask = (attention_mask == 0).clone()
            if is_all_pad.any():
                # Prevent all-True rows from producing NaN in PyTorch multihead attention
                src_key_padding_mask[is_all_pad, 0] = False
        else:
            is_all_pad = None
            src_key_padding_mask = None

        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        h = self.layer_norm(h)

        # Mean pooling over valid non-padded tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            sum_h = (h * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-6)
            pooled = sum_h / sum_mask
            if is_all_pad is not None and is_all_pad.any():
                pooled = torch.where(is_all_pad.unsqueeze(-1), torch.zeros_like(pooled), pooled)
        else:
            pooled = h.mean(dim=1)

        if return_hidden:
            return pooled

        # Project into Platonic semantic space R^k
        e = self.projector(pooled)
        return e
