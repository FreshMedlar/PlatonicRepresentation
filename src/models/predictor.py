"""
Latent State-Transition Predictor (Latent Reasoning) P_phi.
Operates purely in Platonic vector space R^k to model semantic state transitions
(reasoning, deduction, and narrative trajectories) without generating words.
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import PredictorConfig
from src.models.encoder import PositionalEncoding
from src.models.sigreg import SIGRegLoss


class LatentPredictor(nn.Module):
    """
    Causal Transformer operating in continuous vector space R^k.
    Given (z_1, ..., z_t), predicts next semantic state z_hat_{t+1} in R^k.
    """

    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.d_model = config.d_model

        # Projection from vector space R^k to model dimension
        self.in_proj = nn.Linear(config.latent_dim, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, max_len=config.max_steps + 8)

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Output projection back to vector space R^k
        self.out_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.latent_dim),
        )

        self.sigreg = SIGRegLoss(latent_dim=config.latent_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates boolean upper triangular causal mask (True = disallowed)."""
        return torch.triu(torch.ones((sz, sz), dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        z_seq: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_seq: (B, T, latent_dim) sequence of semantic state vectors
            valid_mask: (B, T) boolean mask where True indicates valid step
        Returns:
            z_pred: (B, T, latent_dim) where z_pred[:, t] predicts z_{t+1}
        """
        B, T, D = z_seq.shape
        device = z_seq.device

        x = self.in_proj(z_seq)  # (B, T, d_model)
        x = self.pos_encoding(x)

        causal_mask = self._generate_causal_mask(T, device=device)

        if valid_mask is not None:
            # TransformerEncoder expects True for PADDED/ignored positions
            is_all_pad = (~valid_mask).all(dim=1)
            src_key_padding_mask = (~valid_mask).clone()
            if is_all_pad.any():
                src_key_padding_mask[is_all_pad, 0] = False
        else:
            is_all_pad = None
            src_key_padding_mask = None

        h = self.transformer(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        h = self.layer_norm(h)

        z_pred = self.out_proj(h)  # (B, T, latent_dim)
        return z_pred

    def compute_loss(
        self,
        z_seq: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes predictive transition loss + SIGReg anti-collapse on predicted states.
        Predicts z_{t+1} from z_{1...t}.
        """
        B, T, D = z_seq.shape
        if T < 2:
            raise ValueError("Sequence length T must be at least 2 for state transition modeling.")

        # Input is steps 0 to T-2
        inputs = z_seq[:, :-1, :]  # (B, T-1, D)
        targets = z_seq[:, 1:, :]  # (B, T-1, D)

        if valid_mask is not None:
            in_mask = valid_mask[:, :-1]
            target_mask = valid_mask[:, 1:]
            combined_mask = in_mask & target_mask  # (B, T-1)
        else:
            combined_mask = None

        preds = self.forward(inputs, valid_mask=in_mask if valid_mask is not None else None)

        if combined_mask is not None:
            mask_exp = combined_mask.unsqueeze(-1).float()
            mse = ((preds - targets) ** 2 * mask_exp).sum() / (mask_exp.sum() * D + 1e-6)
            valid_preds = preds[combined_mask]  # (N_valid, D)
        else:
            mse = F.mse_loss(preds, targets)
            valid_preds = preds.reshape(-1, D)

        # Apply SIGReg to predicted latent states to prevent latent collapse
        reg_loss = self.sigreg(valid_preds)
        total_loss = mse + self.config.reg_weight * reg_loss

        return total_loss, mse, reg_loss

    @torch.no_grad()
    def rollout(
        self,
        initial_z: torch.Tensor,
        num_steps: int = 3,
    ) -> torch.Tensor:
        """
        Autoregressively rolls out future semantic states in latent space R^k.
        Args:
            initial_z: (1, latent_dim) or (1, t_init, latent_dim)
            num_steps: number of future states to generate
        Returns:
            trajectory: (1, t_init + num_steps, latent_dim)
        """
        self.eval()
        if initial_z.dim() == 2:
            z_traj = initial_z.unsqueeze(1)  # (B, 1, D)
        else:
            z_traj = initial_z.clone()  # (B, T, D)

        for _ in range(num_steps):
            preds = self.forward(z_traj)
            next_z = preds[:, -1:, :]  # (B, 1, D)
            z_traj = torch.cat([z_traj, next_z], dim=1)

        return z_traj
