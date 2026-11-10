"""
Conditional Text Decoder (Vector-to-Text) d_psi.
Autoregressive Causal Transformer decoder conditioned on soft prompt prefix
projected from Platonic semantic state vector e (and optional style vector).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import DecoderConfig
from src.models.encoder import PositionalEncoding


class ConditionalTextDecoder(nn.Module):
    """
    Causal Transformer Decoder conditioned on soft prefix tokens projected from state vector e.
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_soft_tokens = config.num_soft_tokens
        self.noise_scale = config.noise_scale

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(
            config.d_model, max_len=config.max_seq_len + config.num_soft_tokens + 16
        )

        # Style embedding for nuisance conditioning
        self.style_embedding = nn.Embedding(config.num_styles, config.style_dim)

        # MLP projector mapping [latent_dim + style_dim] -> (num_soft_tokens * d_model)
        input_proj_dim = config.latent_dim + config.style_dim
        self.soft_prompt_mlp = nn.Sequential(
            nn.Linear(input_proj_dim, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.num_soft_tokens * config.d_model),
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Using TransformerEncoder with causal attention mask implements standard causal GPT-style decoder
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=config.num_layers)
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # Tie weights between token embedding and lm head
        self.lm_head.weight = self.token_embedding.weight

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

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates boolean upper triangular causal mask (True = disallowed)."""
        return torch.triu(torch.ones((sz, sz), dtype=torch.bool, device=device), diagonal=1)

    def project_soft_prompts(
        self,
        e: torch.Tensor,
        style_ids: Optional[torch.Tensor] = None,
        inject_noise: bool = True,
    ) -> torch.Tensor:
        """
        Projects state embedding e into M soft prompt prefix tokens.
        Args:
            e: (B, latent_dim)
            style_ids: (B,) optional discrete style class
            inject_noise: whether to add Gaussian noise N(0, noise_scale^2)
        Returns:
            soft_tokens: (B, M, d_model)
        """
        B = e.size(0)
        device = e.device

        # State perturbation e_tilde = e + epsilon
        if inject_noise and self.noise_scale > 0.0 and self.training:
            noise = torch.randn_like(e) * self.noise_scale
            e_tilde = e + noise
        else:
            e_tilde = e

        # Style embedding
        if style_ids is None:
            style_ids = torch.zeros(B, dtype=torch.long, device=device)
        s_emb = self.style_embedding(style_ids)  # (B, style_dim)

        combined = torch.cat([e_tilde, s_emb], dim=-1)  # (B, latent_dim + style_dim)
        soft_flat = self.soft_prompt_mlp(combined)  # (B, M * d_model)
        soft_tokens = soft_flat.view(B, self.num_soft_tokens, self.d_model)
        return soft_tokens

    def forward(
        self,
        e: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        style_ids: Optional[torch.Tensor] = None,
        inject_noise: bool = True,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass for training the conditional decoder.
        Args:
            e: (B, latent_dim) state vector
            input_ids: (B, L) target token IDs
            attention_mask: (B, L) attention mask for target tokens
            style_ids: (B,) style conditioning IDs
        Returns:
            logits: (B, L, vocab_size) predictions for target tokens
        """
        B, L = input_ids.shape
        device = input_ids.device
        M = self.num_soft_tokens

        # Project state into soft prompt prefix tokens
        soft_tokens = self.project_soft_prompts(e, style_ids=style_ids, inject_noise=inject_noise)  # (B, M, D)
        text_tokens = self.token_embedding(input_ids)  # (B, L, D)

        # Concatenate prefix soft tokens with text tokens
        seq = torch.cat([soft_tokens, text_tokens], dim=1)  # (B, M + L, D)
        seq = self.pos_encoding(seq)

        # Build causal mask for sequence length (M + L)
        total_len = M + L
        causal_mask = self._generate_causal_mask(total_len, device=device)

        # Build key padding mask: soft tokens are always valid (False), text tokens follow attention_mask
        if attention_mask is not None:
            prefix_mask = torch.zeros(B, M, dtype=torch.bool, device=device)  # False = not padded
            text_pad_mask = (attention_mask == 0)  # True = padded
            src_key_padding_mask = torch.cat([prefix_mask, text_pad_mask], dim=1)
        else:
            src_key_padding_mask = None

        h = self.transformer(seq, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        h = self.layer_norm(h)

        # Extract only the text positions [M : M + L]
        h_text = h[:, M:]
        logits = self.lm_head(h_text)  # (B, L, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        e: torch.Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        style_id: int = 0,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 20,
    ) -> torch.Tensor:
        """
        Autoregressively renders text from Platonic state vector e.
        Args:
            e: (1, latent_dim) or (latent_dim,)
            bos_id: beginning of sentence token ID
            eos_id: end of sentence token ID
        Returns:
            generated_ids: (1, generated_len)
        """
        self.eval()
        if e.dim() == 1:
            e = e.unsqueeze(0)
        B = e.size(0)
        device = e.device
        M = self.num_soft_tokens

        style_tensor = torch.tensor([style_id] * B, dtype=torch.long, device=device)
        soft_tokens = self.project_soft_prompts(e, style_ids=style_tensor, inject_noise=False)  # (B, M, D)

        generated = torch.tensor([[bos_id] * B], dtype=torch.long, device=device).transpose(0, 1)  # (B, 1)

        for _ in range(max_new_tokens):
            cur_text_emb = self.token_embedding(generated)
            seq = torch.cat([soft_tokens, cur_text_emb], dim=1)
            seq = self.pos_encoding(seq)

            cur_total_len = seq.size(1)
            causal_mask = self._generate_causal_mask(cur_total_len, device=device)

            h = self.transformer(seq, mask=causal_mask)
            h = self.layer_norm(h)

            next_token_logits = self.lm_head(h[:, -1, :]) / max(temperature, 1e-4)

            # Numerical stability guard
            if not torch.isfinite(next_token_logits).all():
                next_token_logits = torch.nan_to_num(next_token_logits, nan=0.0, posinf=10.0, neginf=-10.0)

            # Top-k filtering
            if top_k > 0 and top_k < next_token_logits.size(-1):
                val, _ = torch.topk(next_token_logits, top_k)
                min_val = val[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_val,
                    torch.tensor(-1e4, device=device),
                    next_token_logits,
                )

            probs = F.softmax(next_token_logits, dim=-1)
            # Guard against zero or NaN probabilities
            if torch.isnan(probs).any() or (probs.sum(dim=-1) == 0).any():
                probs = torch.ones_like(probs) / probs.size(-1)

            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_id).all():
                break

        return generated
