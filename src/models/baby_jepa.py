"""
Baby-JEPA Models and Objectives:
- BabyVisionEncoder: ResNet-style visual encoder f_v producing pre-projector H and latent Z.
- ActionConditionedPredictor & PredictorEnsemble: K=5 causal latent world-model ensemble.
- BabyWordEncoder & CrossModalHead: grounded word pathway f_w and cross-modal predictor.
- Whitened JEPA losses + SIGReg/SADL anti-collapse + epistemic & spectral exploration scores.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.baby_config import (
    BabyJEPAConfig,
    BabyPredictorConfig,
    BabyVisionConfig,
    BabyWordConfig,
)
from src.models.encoder import BatchStandardization1d, PositionalEncoding
from src.models.sadl import SADLLoss
from src.models.sigreg import SIGRegLoss


class _ResBlock(nn.Module):
    """Residual convolutional block with GroupNorm for batch-size-invariant stability."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        groups = min(16, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_ch),
            )
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class BabyVisionEncoder(nn.Module):
    """
    Visual encoder f_v: RGB frame (B, 3, H, W) -> pre-projector H (B, d_model) & latent Z (B, latent_dim).
    """

    def __init__(self, config: BabyVisionConfig):
        super().__init__()
        self.config = config
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )
        self.stage1 = _ResBlock(48, 64, stride=2)     # 32x32
        self.stage2 = _ResBlock(64, 128, stride=2)    # 16x16
        self.stage3 = _ResBlock(128, 192, stride=2)   # 8x8
        self.stage4 = _ResBlock(192, 256, stride=2)   # 4x4

        self.head_h = nn.Sequential(
            nn.Linear(256 * 2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        # Semantic Good-View projector (pose/lighting-invariant world state)
        self.projector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.latent_dim),
            BatchStandardization1d(config.latent_dim),
        )
        # Equivariant dynamics projector (models view/pose/lighting transitions under actions)
        self.dyn_projector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.latent_dim),
            BatchStandardization1d(config.latent_dim),
        )

    def forward(
        self, images: torch.Tensor, return_both: bool = False, return_all: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        x = (images - self.img_mean) / self.img_std
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        avg_p = x.mean(dim=(-2, -1))
        max_p = x.amax(dim=(-2, -1))
        h = self.head_h(torch.cat([avg_p, max_p], dim=-1))
        z_sem = self.projector(h)
        if return_all:
            z_dyn = self.dyn_projector(h)
            return h, z_sem, z_dyn
        if return_both:
            return h, z_sem
        return z_sem


class ActionConditionedPredictor(nn.Module):
    """
    Causal latent world-model predictor: (z_{<=t}, a_{<=t}) -> z_hat_{t+1}.
    """

    def __init__(self, config: BabyPredictorConfig):
        super().__init__()
        self.config = config
        self.z_proj = nn.Linear(config.latent_dim, config.d_model)
        self.act_emb = nn.Embedding(config.num_actions, config.d_model)
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
        self.norm = nn.LayerNorm(config.d_model)
        self.out_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.latent_dim),
        )

    def forward(self, z_seq: torch.Tensor, a_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_seq: (B, T, latent_dim)
            a_seq: (B, T) action taken at each step t
        Returns:
            z_pred: (B, T, latent_dim) where z_pred[:, t] predicts z_{t+1}
        """
        T = z_seq.size(1)
        x = self.z_proj(z_seq) + self.act_emb(a_seq)
        x = self.pos_encoding(x)
        causal_mask = torch.triu(
            torch.ones((T, T), dtype=torch.bool, device=z_seq.device), diagonal=1
        )
        h = self.transformer(x, mask=causal_mask)
        h = self.norm(h)
        return self.out_proj(h)


class PredictorEnsemble(nn.Module):
    """Ensemble of K independent ActionConditionedPredictors for epistemic uncertainty."""

    def __init__(self, config: BabyPredictorConfig):
        super().__init__()
        self.config = config
        self.members = nn.ModuleList(
            [ActionConditionedPredictor(config) for _ in range(config.ensemble_size)]
        )

    def forward_all(self, z_seq: torch.Tensor, a_seq: torch.Tensor) -> torch.Tensor:
        """Returns predictions of shape (M, B, T, D)."""
        preds = [m(z_seq, a_seq) for m in self.members]
        return torch.stack(preds, dim=0)

    @staticmethod
    @torch.no_grad()
    def compute_spectral_stats(
        recent_z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precomputes (mean, eigvecs, inv_eigval_weights) from recent latent buffer."""
        rz = recent_z.float()
        mu = rz.mean(dim=0, keepdim=True)
        centered = rz - mu
        cov = (centered.transpose(0, 1) @ centered) / max(1, rz.size(0) - 1)
        cov = 0.5 * (cov + cov.transpose(0, 1))
        eigvals, eigvecs = torch.linalg.eigh(cov)
        inv_w = 1.0 / (eigvals.clamp(min=1e-3) + 0.05)
        inv_w = inv_w / inv_w.mean()
        return mu, eigvecs, inv_w

    @torch.no_grad()
    def score_candidate_actions(
        self,
        z_ctx: torch.Tensor,
        a_ctx: Optional[torch.Tensor],
        recent_z: Optional[torch.Tensor] = None,
        spectral_bonus_weight: float = 0.0,
        spectral_stats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        visited_z: Optional[torch.Tensor] = None,
        novelty_tau: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates all candidate actions a in [0, num_actions) from context (z_ctx, a_ctx).
        First-principles curiosity without action-specific rules:
        - Epistemic disagreement: Var_m(z_hat^(m)_{t+1}(a))
        - Spectral displacement gain: energy of predicted change (z_hat_{t+1} - z_t)
          along underrepresented eigendirections of recent_z (rewards real, informative change).
        - Episodic boredom / habituation: discounts actions whose predicted next latent
          revisits states already seen in visited_z (e.g. clamped stops or ping-pong loops).

        Args:
            z_ctx: (B, t, D) where t >= 1
            a_ctx: (B, t-1) previous actions or None if t == 1
            recent_z: optional (N, D) buffer of recent vision latents for spectral proxy
            spectral_bonus_weight: weight on underrepresented-eigendirection displacement bonus
            spectral_stats: optional precomputed (mu, eigvecs, inv_w) from compute_spectral_stats
            visited_z: optional (B, T_vis, D) states already visited in the current episode
            novelty_tau: kernel bandwidth controlling episodic boredom decay
        Returns:
            scores: (B, A) total exploration utility per action
            disagreement: (B, A) mean ensemble variance per action
        """
        B, t, D = z_ctx.shape
        A = self.config.num_actions
        device = z_ctx.device
        was_training = self.training
        if was_training:
            self.eval()

        try:
            cand_a = torch.arange(A, device=device, dtype=torch.long).view(1, A).expand(B, A)
            z_exp = z_ctx.unsqueeze(1).expand(B, A, t, D).reshape(B * A, t, D)
            if a_ctx is not None and a_ctx.size(1) > 0:
                a_prev = a_ctx.unsqueeze(1).expand(B, A, t - 1).reshape(B * A, t - 1)
                a_full = torch.cat([a_prev, cand_a.reshape(B * A, 1)], dim=1)
            else:
                a_full = cand_a.reshape(B * A, 1)

            preds = self.forward_all(z_exp, a_full)  # (M, B*A, t, D)
            next_preds = preds[:, :, -1, :].view(len(self.members), B, A, D)
            mean_pred = next_preds.mean(dim=0)  # (B, A, D)
            disagreement = (next_preds - mean_pred.unsqueeze(0)).pow(2).sum(dim=-1).mean(dim=0)  # (B, A)

            scores = disagreement.clone()
            if spectral_bonus_weight > 0.0 and (
                spectral_stats is not None or (recent_z is not None and recent_z.size(0) >= 8)
            ):
                # Spectral displacement: reward predicted state CHANGE (z_hat - z_t) along
                # underrepresented eigendirections so stationary/clamped states get zero bonus.
                if spectral_stats is not None:
                    _, eigvecs, inv_w = spectral_stats
                else:
                    _, eigvecs, inv_w = self.compute_spectral_stats(recent_z)
                z_curr = z_ctx[:, -1, :].unsqueeze(1).float()  # (B, 1, D)
                delta_z = mean_pred.float() - z_curr             # (B, A, D)
                proj = delta_z @ eigvecs                         # (B, A, D)
                spectral_gain = (proj.pow(2) * inv_w.view(1, 1, D)).mean(dim=-1)
                scale = disagreement.mean().clamp(min=1e-6) / spectral_gain.mean().clamp(min=1e-6)
                scores = disagreement + spectral_bonus_weight * scale * spectral_gain

            # Episodic boredom / habituation: discount actions landing near visited episode states
            vis = visited_z if visited_z is not None else z_ctx
            d2 = (
                mean_pred.float().unsqueeze(2) - vis.float().unsqueeze(1)
            ).pow(2).mean(dim=-1)  # (B, A, T_vis)
            familiarity = torch.exp(-d2 / (2.0 * novelty_tau * novelty_tau)).max(dim=-1).values
            novelty = (1.0 - familiarity).clamp(min=0.02)
            scores = scores * novelty

            return scores, disagreement
        finally:
            if was_training:
                self.train()


class BabyWordEncoder(nn.Module):
    """Word encoder f_w embedding sparse grounded concept words into R^k."""

    def __init__(self, config: BabyWordConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_word)
        self.proj = nn.Sequential(
            nn.Linear(config.d_word, config.d_word),
            nn.GELU(),
            nn.Linear(config.d_word, config.latent_dim),
            BatchStandardization1d(config.latent_dim),
        )

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embedding(word_ids))


class CrossModalHead(nn.Module):
    """Predicts word-latent c_t from visual latent z_t."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def whitened_mse(pred: torch.Tensor, target: torch.Tensor, ridge: float = 1e-2) -> torch.Tensor:
    """
    Computes ||pred - sg(target)||^2_whitened using target batch covariance whitening.
    Falls back to standard MSE when batch size is too small to estimate covariance.
    """
    p = pred.reshape(-1, pred.size(-1)).float()
    t = target.detach().reshape(-1, target.size(-1)).float()
    N, D = t.shape
    if N < 4:
        return F.mse_loss(p, t)

    t_centered = t - t.mean(dim=0, keepdim=True)
    cov = (t_centered.transpose(0, 1) @ t_centered) / N
    cov = 0.5 * (cov + cov.transpose(0, 1))
    diag_scale = (torch.trace(cov) / D).clamp(min=1e-4)
    cov_reg = cov + (ridge * diag_scale) * torch.eye(D, device=t.device, dtype=torch.float32)

    chol = torch.linalg.cholesky(cov_reg)
    diff = (p - t).transpose(0, 1)  # (D, N)
    whitened_diff = torch.linalg.solve_triangular(chol, diff, upper=False)
    return whitened_diff.pow(2).mean()


class BabyJEPASystem(nn.Module):
    """Complete Baby-JEPA system: vision encoder, predictor ensemble, word encoder, cross-modal head."""

    def __init__(self, config: BabyJEPAConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = BabyVisionEncoder(config.vision)
        self.predictor_ensemble = PredictorEnsemble(config.predictor)
        self.word_encoder = BabyWordEncoder(config.word)
        self.cross_modal_head = CrossModalHead(config.vision.latent_dim)

        self.sigreg = SIGRegLoss(latent_dim=config.vision.latent_dim, num_projections=128)
        self.word_sigreg = SIGRegLoss(latent_dim=config.word.latent_dim, num_projections=64)
        self.sadl = SADLLoss()

    def encode_images(
        self, images: torch.Tensor, return_both: bool = False, return_all: bool = False
    ):
        return self.vision_encoder(images, return_both=return_both, return_all=return_all)

    def compute_training_loss(
        self,
        images_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        word_ids_seq: torch.Tensor,
        has_word_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes joint Baby-JEPA loss on trajectory batch:
        - Good-View invariance on semantic head z_sem across distinct trajectory viewpoints
        - Temporal JEPA prediction on dynamics head z_dyn (whitened + unwhitened MSE vs sg(z_dyn_{t+1}))
        - Vision SIGReg on z_sem & z_dyn + SADL supply regularization on (H, sg(z_sem))
        - Cross-modal grounded word alignment on z_sem at sparse co-occurrence steps + word SIGReg

        Args:
            images_seq: (B, T, 3, H, W)
            actions_seq: (B, T-1) actions transitioning t -> t+1
            word_ids_seq: (B, T) teacher word IDs
            has_word_seq: (B, T) boolean mask where teacher emitted a word
        """
        B, T, C, H, W = images_seq.shape
        flat_imgs = images_seq.view(B * T, C, H, W)
        h_flat, z_sem_flat, z_dyn_flat = self.vision_encoder(flat_imgs, return_all=True)
        z_sem_seq = z_sem_flat.view(B, T, -1)
        z_dyn_seq = z_dyn_flat.view(B, T, -1)

        # 1. Good-View semantic invariance across different viewpoints within each episode
        if T > 1:
            shift = torch.randint(1, T, (B, T), device=images_seq.device)
            base_idx = torch.arange(T, device=images_seq.device).view(1, T).expand(B, T)
            pair_idx = (base_idx + shift) % T
            batch_idx = torch.arange(B, device=images_seq.device).view(B, 1).expand(B, T)
            z_sem_pair = z_sem_seq[batch_idx, pair_idx]
            gv_inv_loss = F.mse_loss(z_sem_seq, z_sem_pair)
        else:
            gv_inv_loss = torch.tensor(0.0, device=images_seq.device)

        # 2. Temporal prediction on z_dyn: predict z_dyn_{1..T-1} from (z_dyn_{0..T-2}, a_{0..T-2})
        z_ctx = z_dyn_seq[:, :-1, :]
        z_tgt = z_dyn_seq[:, 1:, :].detach()
        preds_all = self.predictor_ensemble.forward_all(z_ctx, actions_seq)  # (M, B, T-1, D)

        temp_mse = torch.tensor(0.0, device=images_seq.device)
        temp_wmse = torch.tensor(0.0, device=images_seq.device)
        M = preds_all.size(0)
        for m in range(M):
            # Bootstrap weighting per ensemble member preserves genuine epistemic
            # disagreement on under-explored state-action transitions.
            boot = (torch.rand(B, device=images_seq.device) < 0.8).float()
            if boot.sum() < 2:
                boot = torch.ones(B, device=images_seq.device)
            w = (boot / boot.mean()).view(B, 1, 1)
            pred_m = preds_all[m] * w
            tgt_m = z_tgt * w
            temp_mse = temp_mse + F.mse_loss(pred_m, tgt_m)
            temp_wmse = temp_wmse + whitened_mse(preds_all[m][boot.bool()], z_tgt[boot.bool()])
        temp_mse = temp_mse / M
        temp_wmse = temp_wmse / M

        w_weight = self.config.training.whitened_loss_weight
        temporal_loss = (1.0 - w_weight) * temp_mse + w_weight * temp_wmse

        # 3. Vision anti-collapse: SIGReg on both z_sem and z_dyn, plus SADL on (H, sg(z_sem))
        reg_sem = self.sigreg(z_sem_flat)
        reg_dyn = self.sigreg(z_dyn_flat)
        reg_loss = reg_sem + 0.5 * reg_dyn
        d_zero = self.sadl._get_d_zero(self.sigreg, z_sem_flat)
        sadl_loss = self.sadl(h_flat, z_sem_flat, demand_loss=reg_sem, demand_zero=d_zero)

        # 4. Cross-modal sparse grounded word loss on invariant semantic space z_sem
        cm_loss = torch.tensor(0.0, device=images_seq.device)
        word_reg = torch.tensor(0.0, device=images_seq.device)
        if has_word_seq.any():
            active_z = z_sem_seq[has_word_seq]       # (N_w, D)
            active_wids = word_ids_seq[has_word_seq] # (N_w,)
            c_tgt = self.word_encoder(active_wids)   # (N_w, D)
            c_hat = self.cross_modal_head(active_z)  # (N_w, D)

            cm_mse = F.mse_loss(c_hat, c_tgt.detach())
            cm_wmse = whitened_mse(c_hat, c_tgt)
            cm_loss = (1.0 - w_weight) * cm_mse + w_weight * cm_wmse

            # Regularize active concept word embeddings
            uniq_wids = torch.arange(1, self.config.word.vocab_size, device=images_seq.device)
            all_c = self.word_encoder(uniq_wids)
            word_reg = self.word_sigreg(all_c)

        total_loss = (
            gv_inv_loss
            + temporal_loss
            + self.config.training.sigreg_weight * reg_loss
            + self.config.training.sadl_weight * sadl_loss
            + self.config.training.cross_modal_weight * cm_loss
            + self.config.training.word_sigreg_weight * word_reg
        )

        # Ensemble disagreement diagnostic on context trajectories
        with torch.no_grad():
            ens_var = (preds_all - preds_all.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1).mean()

        metrics = {
            "total_loss": float(total_loss.item()),
            "gv_inv_loss": float(gv_inv_loss.item()),
            "temporal_mse": float(temp_mse.item()),
            "temporal_wmse": float(temp_wmse.item()),
            "sigreg_loss": float(reg_loss.item()),
            "sadl_loss": float(sadl_loss.item()),
            "cross_modal_loss": float(cm_loss.item()),
            "word_sigreg_loss": float(word_reg.item()),
            "ensemble_disagreement": float(ens_var.item()),
        }
        return total_loss, metrics
