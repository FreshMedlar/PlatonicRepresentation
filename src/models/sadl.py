"""
Supply And Demand Loss (SADL) for Joint-Embedding Predictive Architectures.

Implements the adaptive, one-sided Supply Regularizer from:
  "SADL: On the Demand-Supply Geometry of Representation Collapse" (arXiv:2611.21596v1)

Diagnoses and prevents encoder representation collapse behind batch-normalized
projectors by matching the encoder's supply capacity k = min(k_exc, k_tr) to the
detached endpoint demand capacity q = sqrt(q_reg * q_exc) (or log-geodesic alpha).
"""

import math
from typing import Callable, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


def _flatten_aligned(u: torch.Tensor) -> torch.Tensor:
    """Flattens (B, S, D) -> (B * S, D) while preserving batch-time alignment."""
    if u.dim() == 2:
        return u
    if u.dim() == 3:
        return u.reshape(-1, u.size(-1))
    raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(u.shape)}")


def compute_excitation(
    u: torch.Tensor,
    gamma: float = 9.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes normalized excitation score F_exc(U) in [0, 1] (Eq. 3).

    F_exc(U) = log det(I + gamma * Sigma_U / e_U) / (m_U * log(1 + gamma * d_U / m_U))
    where e_U = max(||U||_F^2 / (N * d_U), eps) and m_U = min(d_U, N - 1).
    """
    u_fp32 = _flatten_aligned(u).float()
    n, d_u = u_fp32.shape
    if n < 2 or d_u == 0:
        return torch.zeros((), device=u.device, dtype=torch.float32)

    m_u = min(d_u, n - 1)
    mu_u = u_fp32.mean(dim=0, keepdim=True)
    u_bar = u_fp32 - mu_u
    e_u = (u_fp32.pow(2).sum() / (n * d_u)).clamp(min=eps)

    # Use smaller Gram matrix via Sylvester's determinant identity
    scale = gamma / (n * e_u)
    if n <= d_u:
        gram = scale * (u_bar @ u_bar.transpose(0, 1))
        dim = n
    else:
        gram = scale * (u_bar.transpose(0, 1) @ u_bar)
        dim = d_u

    gram = 0.5 * (gram + gram.transpose(0, 1))
    eye = torch.eye(dim, device=u_fp32.device, dtype=torch.float32)
    chol = torch.linalg.cholesky(eye + gram)
    logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()

    denom = m_u * math.log(1.0 + gamma * d_u / m_u)
    return (logdet / denom).clamp(0.0, 1.0)


def compute_transmission(
    h: torch.Tensor,
    z: torch.Tensor,
    gamma: float = 9.0,
    rho: float = 1e-3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes regularized canonical correlation transmission F_tr(H, Z) in [0, 1] (Eqs. 4-5).

    Uses Cholesky whitening (Appendix B) on the smaller Gram matrix of K_tilde = L_H^{-1} Sigma_HZ L_Z^{-T}.
    Relative ridge scales tau_H and tau_Z retain their gradients when active above eps.
    """
    h_fp32 = _flatten_aligned(h).float()
    z_fp32 = _flatten_aligned(z).float()

    if h_fp32.size(0) != z_fp32.size(0):
        raise ValueError(
            f"Aligned batch sizes must match for transmission, got {h_fp32.size(0)} and {z_fp32.size(0)}"
        )

    n, d_h = h_fp32.shape
    d_z = z_fp32.shape[1]
    if n < 2 or d_h == 0 or d_z == 0:
        return torch.zeros((), device=h.device, dtype=torch.float32)

    h_bar = h_fp32 - h_fp32.mean(dim=0, keepdim=True)
    z_bar = z_fp32 - z_fp32.mean(dim=0, keepdim=True)

    sigma_h = (h_bar.transpose(0, 1) @ h_bar) / n
    sigma_z = (z_bar.transpose(0, 1) @ z_bar) / n
    sigma_hz = (h_bar.transpose(0, 1) @ z_bar) / n

    sigma_h = 0.5 * (sigma_h + sigma_h.transpose(0, 1))
    sigma_z = 0.5 * (sigma_z + sigma_z.transpose(0, 1))

    tau_h = (torch.trace(sigma_h) / d_h).clamp(min=eps)
    tau_z = (torch.trace(sigma_z) / d_z).clamp(min=eps)

    eye_h = torch.eye(d_h, device=h_fp32.device, dtype=torch.float32)
    eye_z = torch.eye(d_z, device=z_fp32.device, dtype=torch.float32)

    a_h = sigma_h + (rho * tau_h) * eye_h
    a_z = sigma_z + (rho * tau_z) * eye_z

    l_h = torch.linalg.cholesky(a_h)
    l_z = torch.linalg.cholesky(a_z)

    # K_tilde = L_H^{-1} Sigma_HZ L_Z^{-T}
    left_solved = torch.linalg.solve_triangular(l_h, sigma_hz, upper=False)
    k_tilde = torch.linalg.solve_triangular(l_z, left_solved.transpose(0, 1), upper=False).transpose(0, 1)

    if d_z <= d_h:
        gram = k_tilde.transpose(0, 1) @ k_tilde
        gram_dim = d_z
    else:
        gram = k_tilde @ k_tilde.transpose(0, 1)
        gram_dim = d_h

    gram = 0.5 * (gram + gram.transpose(0, 1))
    eye_g = torch.eye(gram_dim, device=h_fp32.device, dtype=torch.float32)
    chol_tr = torch.linalg.cholesky(eye_g + gamma * gram)
    logdet = 2.0 * torch.log(torch.diagonal(chol_tr)).sum()

    m = min(d_h, d_z, n - 1)
    denom = m * math.log(1.0 + gamma)
    return (logdet / denom).clamp(0.0, 1.0)


@torch.no_grad()
def compute_representation_stats(u: torch.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    """
    Computes paper diagnostic statistics (Section 4.2, Table 1) on representation U:
      - f: fraction of total energy in sample variation tr(Sigma) / (d * e)
      - erank: entropy-based effective rank of centered covariance Sigma
      - mean_cos: mean off-diagonal sample cosine similarity
      - mean_norm: L2 norm of batch mean ||mu||_2
    """
    u_fp32 = _flatten_aligned(u).float()
    n, d = u_fp32.shape
    if n < 2 or d == 0:
        return {"f": 0.0, "erank": 0.0, "mean_cos": 1.0, "mean_norm": 0.0}

    mu = u_fp32.mean(dim=0, keepdim=True)
    u_bar = u_fp32 - mu
    sigma = (u_bar.transpose(0, 1) @ u_bar) / n
    sigma = 0.5 * (sigma + sigma.transpose(0, 1))
    e_u = (u_fp32.pow(2).sum() / (n * d)).clamp(min=eps)
    f_val = (torch.trace(sigma) / (d * e_u)).clamp(0.0, 1.0).item()

    evals = torch.linalg.eigvalsh(sigma).clamp(min=0.0)
    ev_sum = evals.sum()
    if ev_sum <= 1e-12:
        erank = 0.0
    else:
        p = evals / ev_sum
        p = p[p > 1e-12]
        erank = math.exp(-(p * p.log()).sum().item())

    norms = u_fp32.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    u_norm = u_fp32 / norms
    cos_mat = u_norm @ u_norm.transpose(0, 1)
    mean_cos = ((cos_mat.sum() - n) / (n * (n - 1))).item()

    return {
        "f": float(f_val),
        "erank": float(erank),
        "mean_cos": float(mean_cos),
        "mean_norm": float(mu.norm().item()),
    }


class SADLLoss(nn.Module):
    """
    Supply And Demand Loss (SADL) module (Eqs. 6-10, Algorithm 1).

    Computes the one-sided Supply Regularizer S(H, sg(Z)) = [log((q + eps) / (k + eps))]_+^2
    where demand capacity q combines regularizer satisfaction q_reg and endpoint excitation q_exc,
    and supply capacity k = min(k_exc, k_tr).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 9.0,
        rho: float = 1e-3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.eps = eps
        self._d_zero_cache: Dict[Tuple[int, Tuple[int, ...], str, str], torch.Tensor] = {}

    def _get_d_zero(
        self,
        demand_fn: Callable[[torch.Tensor], torch.Tensor],
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluates D(0) on a shape-matched zero tensor and caches it without altering RNG state.
        """
        key = (id(demand_fn), tuple(z.shape), str(z.device), str(z.dtype))
        cached = self._d_zero_cache.get(key)
        if cached is not None:
            return cached

        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            with torch.no_grad():
                zeros = torch.zeros_like(z)
                d_zero = demand_fn(zeros).detach().float().clamp(min=self.eps)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        self._d_zero_cache[key] = d_zero
        return d_zero

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        demand_loss: torch.Tensor,
        demand_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        demand_zero: Optional[Union[float, torch.Tensor]] = None,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Args:
            h: Encoder representation H of shape (N, d_H) or (B, S, d_H)
            z: Projector endpoint Z of shape (N, d_Z) or (B, S, d_Z)
            demand_loss: Scalar Demand Regularizer value D(Z) (already computed)
            demand_fn: Callable D(.) used to evaluate & cache D(0) when demand_zero is not given
            demand_zero: Explicit reference value D(0), optional if demand_fn is provided
            return_stats: If True, returns (supply_loss, stats_dict)
        """
        device_type = h.device.type if h.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            h_fp32 = _flatten_aligned(h).float()
            z_sg = _flatten_aligned(z).detach().float()

            if h_fp32.size(0) < 2:
                zero = torch.zeros((), device=h.device, dtype=torch.float32)
                if return_stats:
                    return zero, {"q_reg": 0.0, "q_exc": 0.0, "q": 0.0, "k_exc": 0.0, "k_tr": 0.0, "k": 0.0}
                return zero

            if demand_zero is not None:
                d_zero = torch.as_tensor(demand_zero, device=h.device, dtype=torch.float32).clamp(min=self.eps)
            elif demand_fn is not None:
                d_zero = self._get_d_zero(demand_fn, z)
            else:
                raise ValueError("Either demand_fn or demand_zero must be provided to SADLLoss.")

            q_reg = (1.0 - demand_loss.detach().float() / d_zero).clamp(0.0, 1.0)
            q_exc = compute_excitation(z_sg, gamma=self.gamma, eps=self.eps).detach()
            q = torch.exp(
                (1.0 - self.alpha) * torch.log(q_reg + self.eps)
                + self.alpha * torch.log(q_exc + self.eps)
            ).detach()

            k_exc = compute_excitation(h_fp32, gamma=self.gamma, eps=self.eps)
            k_tr = compute_transmission(h_fp32, z_sg, gamma=self.gamma, rho=self.rho, eps=self.eps)
            k = torch.minimum(k_exc, k_tr)

            deficit = torch.clamp(torch.log(q + self.eps) - torch.log(k + self.eps), min=0.0)
            supply_loss = deficit.square()

        if return_stats:
            stats = {
                "q_reg": float(q_reg.item()),
                "q_exc": float(q_exc.item()),
                "q": float(q.item()),
                "k_exc": float(k_exc.item()),
                "k_tr": float(k_tr.item()),
                "k": float(k.item()),
            }
            return supply_loss, stats
        return supply_loss
