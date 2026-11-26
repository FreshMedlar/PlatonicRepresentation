"""
Evaluation Benchmark Suite for Baby-JEPA:
- Linear CKA and Mutual k-NN alignment against frozen independent encoders (DINO ViT-S/8)
- Single-rotation Orthogonal Procrustes zero-shot naming (internal word space & frozen MiniLM)
- False-belief / confounder disentanglement linear probes (shape, color, texture, light_tint)
- Latent geometry diagnostics (effective rank, f_H, erank_H, erank_Z)
"""

import glob
import math
import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.baby_config import COLORS, SHAPES, TEXTURES, TabletopEnvConfig
from src.env.tabletop_world import TabletopWorld, build_word_vocabulary
from src.evaluation.benchmark import compute_effective_rank
from src.models.baby_jepa import BabyJEPASystem
from src.models.sadl import compute_representation_stats


def compute_linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Computes linear Centered Kernel Alignment (CKA) in [0, 1] between (N, D1) and (N, D2).
    Invariant to orthogonal transformations and isotropic scaling.
    """
    x_c = x.float() - x.float().mean(dim=0, keepdim=True)
    y_c = y.float() - y.float().mean(dim=0, keepdim=True)
    if x_c.size(0) < 2:
        return 0.0

    xty = x_c.transpose(0, 1) @ y_c
    xtx = x_c.transpose(0, 1) @ x_c
    yty = y_c.transpose(0, 1) @ y_c
    num = (xty * xty).sum()
    denom = torch.sqrt((xtx * xtx).sum().clamp(min=eps) * (yty * yty).sum().clamp(min=eps))
    return float((num / denom).clamp(0.0, 1.0).item())


def compute_mutual_knn(x: torch.Tensor, y: torch.Tensor, k: int = 10) -> float:
    """
    Computes mutual k-nearest-neighbor overlap (the Platonic Representation Hypothesis metric).
    """
    n = x.size(0)
    if n <= k + 1:
        k = max(1, n - 2)
    if n < 3:
        return 0.0

    x_n = F.normalize(x.float(), p=2, dim=-1)
    y_n = F.normalize(y.float(), p=2, dim=-1)

    sim_x = x_n @ x_n.transpose(0, 1)
    sim_y = y_n @ y_n.transpose(0, 1)
    eye = torch.eye(n, device=x.device, dtype=torch.bool)
    sim_x = sim_x.masked_fill(eye, -1e9)
    sim_y = sim_y.masked_fill(eye, -1e9)

    nn_x = sim_x.topk(k, dim=-1).indices  # (N, k)
    nn_y = sim_y.topk(k, dim=-1).indices  # (N, k)

    # Overlap per sample
    overlap = (nn_x.unsqueeze(-1) == nn_y.unsqueeze(1)).any(dim=-1).float().mean().item()
    return float(overlap)


def fit_orthogonal_procrustes(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Fits one orthogonal rotation matrix Q maximizing <source @ Q, target>
    via SVD of source^T @ target. Handles differing dimensions via zero-padding if needed.
    """
    s = source.float()
    t = target.float()
    # Center both spaces before rotation fitting
    s = s - s.mean(dim=0, keepdim=True)
    t = t - t.mean(dim=0, keepdim=True)

    d_s, d_t = s.size(1), t.size(1)
    if d_s != d_t:
        d_max = max(d_s, d_t)
        if d_s < d_max:
            s = F.pad(s, (0, d_max - d_s))
        if d_t < d_max:
            t = F.pad(t, (0, d_max - d_t))

    m = s.transpose(0, 1) @ t
    u, _, vt = torch.linalg.svd(m, full_matrices=False)
    q = u @ vt
    return q[:d_s, :d_t]


def evaluate_ridge_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    ridge: float = 1.0,
) -> float:
    """Closed-form multi-class ridge linear probe on frozen embeddings."""
    x_tr = train_x.float()
    x_te = test_x.float()
    mu = x_tr.mean(dim=0, keepdim=True)
    std = x_tr.std(dim=0, keepdim=True).clamp(min=1e-4)
    x_tr_n = (x_tr - mu) / std
    x_te_n = (x_te - mu) / std

    # Augment with bias column
    ones_tr = torch.ones((x_tr_n.size(0), 1), device=x_tr.device)
    ones_te = torch.ones((x_te_n.size(0), 1), device=x_te.device)
    phi_tr = torch.cat([x_tr_n, ones_tr], dim=1)
    phi_te = torch.cat([x_te_n, ones_te], dim=1)

    y_onehot = F.one_hot(train_y.long(), num_classes=num_classes).float()
    dim = phi_tr.size(1)
    a = phi_tr.transpose(0, 1) @ phi_tr + ridge * torch.eye(dim, device=x_tr.device)
    b = phi_tr.transpose(0, 1) @ y_onehot
    w = torch.linalg.solve(a, b)
    preds = (phi_te @ w).argmax(dim=-1)
    return float((preds == test_y.long()).float().mean().item())


class _DINOViTBlock(nn.Module):
    """Exact timm-compatible ViT block for offline loading of DINO ViT-S/8."""

    def __init__(self, dim: int = 384, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.Module()
        self.attn.num_heads = num_heads
        self.attn.head_dim = dim // num_heads
        self.attn.qkv = nn.Linear(dim, dim * 3)
        self.attn.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, hidden)
        self.mlp.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        h = self.norm1(x)
        qkv = self.attn.qkv(h).reshape(B, N, 3, self.attn.num_heads, self.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        x = x + self.attn.proj(attn_out)

        h2 = self.norm2(x)
        x = x + self.mlp.fc2(F.gelu(self.mlp.fc1(h2)))
        return x


class FrozenDINOEncoder(nn.Module):
    """
    Pure-PyTorch loader for cached timm/vit_small_patch8_224.dino weights (no timm needed).
    """

    def __init__(self, weights_path: str):
        super().__init__()
        from safetensors.torch import load_file

        dim = 384
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=8, stride=8)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 785, dim))
        self.blocks = nn.ModuleList([_DINOViTBlock(dim=dim, num_heads=6) for _ in range(12)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        state = load_file(weights_path)
        self.load_state_dict(state, strict=True)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @classmethod
    def from_local_cache(cls) -> Optional["FrozenDINOEncoder"]:
        pattern = os.path.expanduser(
            "~/.cache/huggingface/hub/models--timm--vit_small_patch8_224.dino/snapshots/*/model.safetensors"
        )
        matches = sorted(glob.glob(pattern))
        if not matches:
            return None
        try:
            return cls(matches[-1])
        except Exception:
            return None

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        x = (images - self.mean) / self.std
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)  # (B, 784, 384)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


class FrozenMiniLMEncoder:
    """Loads locally cached sentence-transformers/all-MiniLM-L6-v2 for independent text geometry."""

    def __init__(self, device: torch.device):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_local_cache(cls, device: torch.device) -> Optional["FrozenMiniLMEncoder"]:
        try:
            return cls(device=device)
        except Exception:
            return None

    @torch.no_grad()
    def encode_words(self, words: List[str]) -> torch.Tensor:
        tok = self.tokenizer(words, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.model(**tok).last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1).float()
        emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(emb, p=2, dim=-1)


@torch.no_grad()
def build_eval_datasets(
    env_cfg: TabletopEnvConfig,
    device: str,
    num_probe_train: int = 160,
    num_eval: int = 192,
    seed: int = 999,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Generates fixed held-out probe/evaluation datasets:
    - 'confounded': matches training confounder distribution (for probe training / false-belief check)
    - 'decorrelated': completely unconfounded random colors, shapes, poses, and lights
    """
    datasets = {}
    for split_name, c_prob, biased in [
        ("confounded_train", env_cfg.confound_prob, True),
        ("decorrelated_eval", 0.0, False),
    ]:
        n = num_probe_train if split_name == "confounded_train" else num_eval
        world = TabletopWorld(env_cfg, batch_size=n, device=device, seed=seed + (0 if biased else 17))
        obs = world.reset(confound_prob=c_prob, biased_view=biased)
        datasets[split_name] = obs
    return datasets


@torch.no_grad()
def evaluate_baby_jepa(
    system: BabyJEPASystem,
    eval_splits: Dict[str, Dict[str, torch.Tensor]],
    dino_encoder: Optional[FrozenDINOEncoder] = None,
    minilm_encoder: Optional[FrozenMiniLMEncoder] = None,
    dino_eval_feats: Optional[torch.Tensor] = None,
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Runs the full Baby-JEPA evaluation suite on held-out splits.
    """
    was_training = system.training
    system.eval()
    device = next(system.parameters()).device

    def _encode_split(imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hs, zs = [], []
        for i in range(0, imgs.size(0), batch_size):
            h_b, z_b = system.encode_images(imgs[i : i + batch_size].to(device), return_both=True)
            hs.append(h_b)
            zs.append(z_b)
        return torch.cat(hs, dim=0), torch.cat(zs, dim=0)

    tr_split = eval_splits["confounded_train"]
    ev_split = eval_splits["decorrelated_eval"]

    h_tr, z_tr = _encode_split(tr_split["images"])
    h_ev, z_ev = _encode_split(ev_split["images"])

    num_shapes = system.config.env.num_shapes
    num_colors = system.config.env.num_colors
    num_textures = system.config.env.num_textures

    # 1. False-belief / confounder disentanglement probes (trained on confounded split, tested on decorrelated!)
    shape_probe_dec = evaluate_ridge_probe(h_tr, tr_split["shape_id"].to(device), h_ev, ev_split["shape_id"].to(device), num_shapes)
    color_probe_dec = evaluate_ridge_probe(h_tr, tr_split["color_id"].to(device), h_ev, ev_split["color_id"].to(device), num_colors)
    tex_probe_dec = evaluate_ridge_probe(h_tr, tr_split["texture_id"].to(device), h_ev, ev_split["texture_id"].to(device), num_textures)

    # Also evaluate probes trained directly on half of decorrelated split (representational capacity check)
    half = ev_split["images"].size(0) // 2
    shape_probe_clean = evaluate_ridge_probe(
        h_ev[:half], ev_split["shape_id"][:half].to(device),
        h_ev[half:], ev_split["shape_id"][half:].to(device),
        num_shapes,
    )
    color_probe_clean = evaluate_ridge_probe(
        h_ev[:half], ev_split["color_id"][:half].to(device),
        h_ev[half:], ev_split["color_id"][half:].to(device),
        num_colors,
    )
    shape_probe_z = evaluate_ridge_probe(
        z_ev[:half], ev_split["shape_id"][:half].to(device),
        z_ev[half:], ev_split["shape_id"][half:].to(device),
        num_shapes,
    )
    color_probe_z = evaluate_ridge_probe(
        z_ev[:half], ev_split["color_id"][:half].to(device),
        z_ev[half:], ev_split["color_id"][half:].to(device),
        num_colors,
    )

    # 2. Geometry metrics
    h_stats = compute_representation_stats(h_ev)
    z_stats = compute_representation_stats(z_ev)
    eff_rank_z = compute_effective_rank(F.normalize(z_ev, p=2, dim=-1))

    metrics: Dict[str, float] = {
        "shape_probe_from_confounded": shape_probe_dec,
        "color_probe_from_confounded": color_probe_dec,
        "texture_probe_from_confounded": tex_probe_dec,
        "shape_probe_decorrelated": shape_probe_clean,
        "color_probe_decorrelated": color_probe_clean,
        "shape_probe_z_decorrelated": shape_probe_z,
        "color_probe_z_decorrelated": color_probe_z,
        "effective_rank_z": float(eff_rank_z),
        "f_H": float(h_stats["f"]),
        "f_Z": float(z_stats["f"]),
        "erank_H": float(h_stats["erank"]),
        "erank_Z": float(z_stats["erank"]),
        "mean_cos_Z": float(z_stats["mean_cos"]),
    }

    # 3. Single-rotation Procrustes zero-shot naming (in internal word space)
    word_to_id, _ = build_word_vocabulary()
    shape_wids = torch.tensor([word_to_id[s] for s in SHAPES[:num_shapes]], device=device)
    color_wids = torch.tensor([word_to_id[c] for c in COLORS[:num_colors]], device=device)
    shape_word_vecs = system.word_encoder(shape_wids)  # (S, D)
    color_word_vecs = system.word_encoder(color_wids)  # (C, D)

    cal_shape_targets = shape_word_vecs[ev_split["shape_id"][:half].to(device)]
    q_shape = fit_orthogonal_procrustes(z_ev[:half], cal_shape_targets)
    z_rot_shape = F.normalize(z_ev[half:] @ q_shape, p=2, dim=-1)
    s_norm = F.normalize(shape_word_vecs, p=2, dim=-1)
    shape_naming_acc = float(((z_rot_shape @ s_norm.T).argmax(dim=-1) == ev_split["shape_id"][half:].to(device)).float().mean().item())

    cal_color_targets = color_word_vecs[ev_split["color_id"][:half].to(device)]
    q_color = fit_orthogonal_procrustes(z_ev[:half], cal_color_targets)
    z_rot_color = F.normalize(z_ev[half:] @ q_color, p=2, dim=-1)
    c_norm = F.normalize(color_word_vecs, p=2, dim=-1)
    color_naming_acc = float(((z_rot_color @ c_norm.T).argmax(dim=-1) == ev_split["color_id"][half:].to(device)).float().mean().item())

    metrics["procrustes_shape_naming_acc"] = shape_naming_acc
    metrics["procrustes_color_naming_acc"] = color_naming_acc

    # 4. Procrustes zero-shot naming against frozen independent MiniLM embeddings
    if minilm_encoder is not None:
        ext_shape_vecs = minilm_encoder.encode_words(list(SHAPES[:num_shapes]))
        q_ext = fit_orthogonal_procrustes(z_ev[:half], ext_shape_vecs[ev_split["shape_id"][:half].to(device)])
        z_ext_rot = F.normalize(z_ev[half:] @ q_ext, p=2, dim=-1)
        ext_acc = float(((z_ext_rot @ ext_shape_vecs.T).argmax(dim=-1) == ev_split["shape_id"][half:].to(device)).float().mean().item())
        metrics["minilm_procrustes_shape_acc"] = ext_acc

    # 5. Alignment against frozen independent DINO ViT-S/8
    if dino_eval_feats is None and dino_encoder is not None:
        feats = []
        imgs_ev = ev_split["images"]
        for i in range(0, imgs_ev.size(0), 32):
            feats.append(dino_encoder(imgs_ev[i : i + 32].to(device)))
        dino_eval_feats = torch.cat(feats, dim=0)

    if dino_eval_feats is not None:
        metrics["dino_cka"] = compute_linear_cka(z_ev, dino_eval_feats)
        metrics["dino_mutual_10nn"] = compute_mutual_knn(z_ev, dino_eval_feats, k=10)
        metrics["dino_cka_h"] = compute_linear_cka(h_ev, dino_eval_feats)

    if was_training:
        system.train()
    return metrics
