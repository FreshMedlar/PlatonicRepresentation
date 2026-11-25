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
    Matches official LeJEPA formulation (Balestriero & LeCun, 2025).
    """

    def __init__(
        self,
        latent_dim: int = 256,
        knots: int = 17,
        num_projections: int = 512,
        lam: float = 1.0,
        num_integration_steps: Optional[int] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_projections = num_projections
        knots = num_integration_steps if num_integration_steps is not None else knots

        # Setup integration knots on [0, 3] with Gaussian window weighting
        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-0.5 * (t ** 2))

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: Tensor of shape (B, D) or (T, B, D).
        Returns:
            Scalar tensor representing the SIGReg loss penalty.
        """
        if proj.dim() == 2:
            B, D = proj.shape
            if B < 2:
                return torch.tensor(0.0, device=proj.device, requires_grad=True)
            A = torch.randn(D, self.num_projections, device=proj.device)
            A = A.div_(A.norm(p=2, dim=0, keepdim=True))

            # (B, P, 1) * (1, 1, knots) -> (B, P, knots)
            x_t = (proj @ A).unsqueeze(-1) * self.t
            err = (x_t.cos().mean(dim=0) - self.phi).square() + x_t.sin().mean(dim=0).square()
            statistic = (err @ self.weights) * B
            return statistic.mean()
        elif proj.dim() == 3:
            # (T, B, D)
            T, B, D = proj.shape
            if B < 2:
                return torch.tensor(0.0, device=proj.device, requires_grad=True)
            A = torch.randn(D, self.num_projections, device=proj.device)
            A = A.div_(A.norm(p=2, dim=0, keepdim=True))

            x_t = (proj @ A).unsqueeze(-1) * self.t
            err = (x_t.cos().mean(dim=1) - self.phi).square() + x_t.sin().mean(dim=1).square()
            statistic = (err @ self.weights) * B
            return statistic.mean()
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {proj.shape}")


def compute_lejepa_loss(
    e1: torch.Tensor,
    e2: torch.Tensor,
    sigreg_loss_fn: SIGRegLoss,
    reg_weight: float = 0.05,
    h1: Optional[torch.Tensor] = None,
    h2: Optional[torch.Tensor] = None,
    sadl_loss_fn: Optional[nn.Module] = None,
    supply_weight: float = 0.01,
    sadl_view_mean: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total LeJEPA objective: Invariance MSE + lambda * SIGReg(e1, e2) [+ lambda_S * SADL(H, sg(Z))].
    When h1, h2, and sadl_loss_fn are provided, applies the one-sided SADL Supply Regularizer
    directly on pre-projector encoder representations H to prevent null-space collapse.
    """
    # Invariance loss: distance between two positive views of same world state z
    inv_loss = F.mse_loss(e1, e2)

    # Combined batch for isotropic Gaussian regularization
    all_embeddings = torch.cat([e1, e2], dim=0)
    reg_loss = sigreg_loss_fn(all_embeddings)

    total_loss = inv_loss + reg_weight * reg_loss

    if (
        sadl_loss_fn is not None
        and h1 is not None
        and h2 is not None
        and supply_weight > 0.0
    ):
        if sadl_view_mean:
            h_sadl = 0.5 * (h1 + h2)
            z_sadl = 0.5 * (e1 + e2)
            d_zero = sadl_loss_fn._get_d_zero(sigreg_loss_fn, all_embeddings)
            supply_loss = sadl_loss_fn(h_sadl, z_sadl, demand_loss=reg_loss, demand_zero=d_zero)
        else:
            h_sadl = torch.cat([h1, h2], dim=0)
            supply_loss = sadl_loss_fn(
                h_sadl, all_embeddings, demand_loss=reg_loss, demand_fn=sigreg_loss_fn
            )
        total_loss = total_loss + supply_weight * supply_loss.to(total_loss.dtype)

    return total_loss, inv_loss, reg_loss
