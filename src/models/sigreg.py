"""
Sketched Isotropic Gaussian Regularization (SIGReg) Loss.
From LeJEPA (Balestriero & LeCun, 2025).

Prevents representation collapse by enforcing the batch embedding distribution
to match an isotropic Gaussian N(0, I) across random 1D projections via the
Epps-Pulley empirical characteristic function test.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGRegLoss(nn.Module):
    """
    Computes SIGReg anti-collapse loss in O(B * k) time and memory complexity.
    """

    def __init__(
        self,
        latent_dim: int,
        lam: float = 1.0,
        num_projections: int = 64,
        num_integration_steps: int = 17,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.lam = lam
        self.num_projections = num_projections
        self.num_integration_steps = num_integration_steps

        # Setup integration knots and trapezoidal weights: t in [-4*lam, 4*lam]
        t_knots = torch.linspace(-4.0 * lam, 4.0 * lam, num_integration_steps)
        dt = (t_knots[-1] - t_knots[0]) / (num_integration_steps - 1)
        weights = torch.ones(num_integration_steps) * dt
        weights[0] /= 2.0
        weights[-1] /= 2.0
        target_cf = torch.exp(-0.5 * (t_knots ** 2))

        # Register non-parameter buffers
        self.register_buffer("t_knots", t_knots)
        self.register_buffer("quad_weights", weights)
        self.register_buffer("target_cf", target_cf)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Tensor of shape (B, latent_dim), model embeddings.
        Returns:
            Scalar tensor representing the SIGReg loss penalty.
        """
        B, D = z.shape
        if B < 2:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        device = z.device

        # 1. Sample random 1D projection vectors V ~ N(0, I), normalized to unit sphere
        V = torch.randn(D, self.num_projections, device=device)
        V = F.normalize(V, p=2, dim=0)

        # 2. Project embeddings onto 1D: shape (B, num_projections)
        proj = z @ V

        # 3. Compute Empirical Characteristic Function (ECF) across batch
        # t_knots: (T,), proj: (B, P) -> arg: (T, B, P)
        arg = self.t_knots.view(-1, 1, 1) * proj.view(1, B, self.num_projections)

        # Average over batch: shape (T, P)
        ecf_real = torch.cos(arg).mean(dim=1)
        ecf_imag = torch.sin(arg).mean(dim=1)

        # 4. Difference from target standard normal CF (target CF is real, imaginary is 0)
        diff_real = ecf_real - self.target_cf.view(-1, 1)
        diff_imag = ecf_imag

        # 5. Integrated squared error along each projection direction using trapezoidal quadrature
        err_sq = diff_real ** 2 + diff_imag ** 2
        loss_per_proj = (self.quad_weights.view(-1, 1) * err_sq).sum(dim=0)

        return loss_per_proj.mean()


def compute_lejepa_loss(
    e1: torch.Tensor,
    e2: torch.Tensor,
    sigreg_loss_fn: SIGRegLoss,
    reg_weight: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total LeJEPA objective: Invariance MSE + lambda * SIGReg(e1, e2).
    """
    # Invariance loss: distance between two positive views of same world state z
    inv_loss = F.mse_loss(e1, e2)

    # Combined batch for isotropic Gaussian regularization
    all_embeddings = torch.cat([e1, e2], dim=0)
    reg_loss = sigreg_loss_fn(all_embeddings)

    total_loss = inv_loss + reg_weight * reg_loss
    return total_loss, inv_loss, reg_loss
