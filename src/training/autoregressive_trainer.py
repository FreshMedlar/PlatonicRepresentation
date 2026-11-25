"""
Four-Stage Autoregressive Predictive JEPA Trainer.

Pipeline implementing stable autoregressive JEPA training over general text corpora:
  Stage 1: Pre-train Encoder E on Good-View paraphrase pairs + SIGReg (semantic invariance)
  Stage 2: Pre-train Predictor P on frozen contiguous sentence chains using pure MSE
  Stage 3 (Joint): Co-train E and P where predicted z_hat_{t+1} and actual z_{t+1} act as a
           predictive pair, anchored by Good-View invariance + SIGReg on E's outputs
  Stage 4: Train Conditional Decoder D on clean prose/steps with noise calibrated to P's residual error
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("ROCM_PATH", "/opt/rocm")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from configs.config import (
    DecoderConfig,
    EncoderConfig,
    PredictorConfig,
    SADLConfig,
    SIGRegConfig,
    SystemConfig,
)
from src.data.augmentations import LightTokenMasker, TextSpanTrimmer, augment_token_pair
from src.data.autoregressive_corpus_loader import AutoregressiveCorpusLoader
from src.evaluation.benchmark import benchmark_predictor_reasoning, run_full_benchmark
from src.models.decoder import ConditionalTextDecoder
from src.models.encoder import TextJEPAEncoder
from src.models.pipeline import GoodViewTextJEPA
from src.models.predictor import LatentPredictor
from src.models.sadl import SADLLoss, compute_representation_stats
from src.models.sigreg import SIGRegLoss, compute_lejepa_loss
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.gpu_fast_trainer import GPUMonitor
from src.training.trainer_utils import create_optimizer, get_cosine_schedule_with_warmup


def pad_and_truncate(ids: List[int], max_len: int, pad_id: int) -> Tuple[List[int], List[int]]:
    """Pads or truncates token ID sequence to fixed max_len and builds attention mask."""
    if len(ids) > max_len:
        ids = ids[:max_len]
    mask = [1] * len(ids)
    if len(ids) < max_len:
        pad_len = max_len - len(ids)
        ids = ids + [pad_id] * pad_len
        mask = mask + [0] * pad_len
    return ids, mask


def build_good_view_tensors(
    loader: AutoregressiveCorpusLoader,
    tokenizer: BPETokenizer,
    num_samples: int,
    max_seq_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Pre-tokenizes Good-View anchor pairs into GPU VRAM (clean TaPaCo + in-domain prose self-views)."""
    print(f"[*] Pre-tokenizing {num_samples} Good-View anchor pairs into {device}...")
    global_trimmer = TextSpanTrimmer(min_ratio=0.65, max_ratio=1.0)
    local_trimmer = TextSpanTrimmer(min_ratio=0.55, max_ratio=1.0)
    masker = LightTokenMasker(mask_id=tokenizer.mask_id, mask_prob=0.10)
    prot = [tokenizer.bos_id, tokenizer.eos_id]
    domain_sents = [s for doc in loader.train_documents for s in doc]

    v1_ids, v1_m, v2_ids, v2_m = [], [], [], []
    for _ in range(num_samples):
        if domain_sents and loader.rng.random() < 0.20:
            s = loader.rng.choice(domain_sents)
            t1, t2 = s, s
        else:
            t1, t2 = loader.sample_good_view_pair(tapaco_prob=1.0)
        c1 = tokenizer.encode(t1, add_bos=True, add_eos=True)["input_ids"].tolist()
        c2 = tokenizer.encode(t2, add_bos=True, add_eos=True)["input_ids"].tolist()
        toks1, toks2 = augment_token_pair(
            list(c1), list(c2),
            mask_id=tokenizer.mask_id,
            global_trimmer=global_trimmer,
            local_trimmer=local_trimmer,
            masker=masker,
            protected_tokens=prot,
        )
        i1, m1 = pad_and_truncate(toks1, max_seq_len, tokenizer.pad_id)
        i2, m2 = pad_and_truncate(toks2, max_seq_len, tokenizer.pad_id)
        v1_ids.append(i1)
        v1_m.append(m1)
        v2_ids.append(i2)
        v2_m.append(m2)

    return {
        "v1_ids": torch.tensor(v1_ids, dtype=torch.long, device=device),
        "v1_mask": torch.tensor(v1_m, dtype=torch.long, device=device),
        "v2_ids": torch.tensor(v2_ids, dtype=torch.long, device=device),
        "v2_mask": torch.tensor(v2_m, dtype=torch.long, device=device),
    }


def build_window_tensors(
    windows: List[List[str]],
    tokenizer: BPETokenizer,
    max_steps: int,
    max_seq_len: int,
    device: torch.device,
    label: str = "train",
) -> Dict[str, torch.Tensor]:
    """Pre-tokenizes contiguous sentence chains [s_0, ..., s_{T-1}] into GPU VRAM."""
    n = len(windows)
    print(f"[*] Pre-tokenizing {n} ({label}) contiguous sentence windows (T={max_steps}) into {device}...")
    step_ids = torch.full((n, max_steps, max_seq_len), tokenizer.pad_id, dtype=torch.long, device=device)
    step_masks = torch.zeros((n, max_steps, max_seq_len), dtype=torch.long, device=device)
    step_valid = torch.zeros((n, max_steps), dtype=torch.bool, device=device)

    for i, chain in enumerate(windows):
        t_len = min(len(chain), max_steps)
        for s_idx in range(t_len):
            enc = tokenizer.encode(chain[s_idx], add_bos=True, add_eos=True, max_len=max_seq_len, pad=True)
            step_ids[i, s_idx] = enc["input_ids"].to(device)
            step_masks[i, s_idx] = enc["attention_mask"].to(device)
            step_valid[i, s_idx] = True

    return {"step_ids": step_ids, "step_masks": step_masks, "step_valid": step_valid}


def build_decoder_tensors(
    sentences: List[str],
    tokenizer: BPETokenizer,
    max_seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-tokenizes clean prose/procedural sentences for Stage 4 Decoder training."""
    print(f"[*] Pre-tokenizing {len(sentences)} clean decoder sentences into {device}...")
    ids_list, mask_list = [], []
    for sent in sentences:
        enc = tokenizer.encode(sent, add_bos=True, add_eos=True, max_len=max_seq_len, pad=True)
        ids_list.append(enc["input_ids"])
        mask_list.append(enc["attention_mask"])
    return torch.stack(ids_list).to(device), torch.stack(mask_list).to(device)


def encode_in_chunks(
    encoder: TextJEPAEncoder,
    ids: torch.Tensor,
    masks: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Encodes token batches in chunks without tracking gradients to avoid VRAM spikes."""
    out = []
    with torch.no_grad():
        for start in range(0, ids.size(0), chunk_size):
            out.append(encoder(ids[start : start + chunk_size], attention_mask=masks[start : start + chunk_size]))
    return torch.cat(out, dim=0)


def compute_symmetric_infonce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Computes symmetric in-batch cosine InfoNCE loss between aligned embeddings (N, D)."""
    if a.size(0) < 2:
        return torch.zeros((), device=a.device, dtype=a.dtype)
    a_n = F.normalize(a, p=2, dim=-1)
    b_n = F.normalize(b, p=2, dim=-1)
    logits = (a_n @ b_n.transpose(0, 1)) / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def compute_joint_autoregressive_step(
    encoder: TextJEPAEncoder,
    predictor: LatentPredictor,
    sigreg: SIGRegLoss,
    doc_ids: torch.Tensor,
    doc_masks: torch.Tensor,
    doc_valid: torch.Tensor,
    gv1_ids: torch.Tensor,
    gv1_masks: torch.Tensor,
    gv2_ids: torch.Tensor,
    gv2_masks: torch.Tensor,
    gv_weight: float = 0.5,
    cos_weight: float = 0.5,
    reg_weight: float = 0.02,
    sadl: Optional[SADLLoss] = None,
    supply_weight: float = 0.01,
    ar_nce_weight: float = 0.0,
    gv_nce_weight: float = 0.0,
    ar_mse_weight: float = 1.0,
    ar_nce_temp: float = 0.12,
    detach_ar_nce_tgt: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Executes one joint autoregressive + good-view JEPA forward pass.
    Concatenates valid document sentences and good-view anchor sentences into a unified
    encoder forward pass so BatchStandardization1d acts on clean, unpadded representations.
    When sadl is provided, also constrains pre-projector encoder representations H via SADL.
    """
    b_doc, t_len, _ = doc_ids.shape
    latent_dim = encoder.config.latent_dim

    valid_doc_ids = doc_ids[doc_valid]
    valid_doc_masks = doc_masks[doc_valid]
    n_doc_valid = valid_doc_ids.size(0)
    n_gv = gv1_ids.size(0)

    all_ids = torch.cat([valid_doc_ids, gv1_ids, gv2_ids], dim=0)
    all_masks = torch.cat([valid_doc_masks, gv1_masks, gv2_masks], dim=0)
    if sadl is not None and supply_weight > 0.0:
        all_h, all_z = encoder(all_ids, attention_mask=all_masks, return_both=True)
    else:
        all_h = None
        all_z = encoder(all_ids, attention_mask=all_masks)

    z_valid_doc = all_z[:n_doc_valid]
    e1 = all_z[n_doc_valid : n_doc_valid + n_gv]
    e2 = all_z[n_doc_valid + n_gv :]

    # Scatter valid sentence vectors back into (B_doc, T, D) while preserving autograd flow
    z_doc = torch.zeros((b_doc, t_len, latent_dim), device=all_z.device, dtype=all_z.dtype)
    z_doc[doc_valid] = z_valid_doc

    in_mask = doc_valid[:, :-1]
    target_mask = doc_valid[:, 1:]
    pair_mask = in_mask & target_mask

    z_preds = predictor(z_doc[:, :-1, :], valid_mask=in_mask)
    z_targets = z_doc[:, 1:, :]

    mask_exp = pair_mask.unsqueeze(-1).float()
    ar_mse = ((z_preds - z_targets) ** 2 * mask_exp).sum() / (mask_exp.sum() * latent_dim + 1e-6)
    pred_flat = z_preds[pair_mask]
    tgt_flat = z_targets[pair_mask]
    ar_cos = F.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean()
    ar_cos_loss = 1.0 - ar_cos

    gv_mse = F.mse_loss(e1, e2)
    reg_loss = sigreg(all_z)

    total_loss = ar_mse_weight * ar_mse + cos_weight * ar_cos_loss + gv_weight * gv_mse + reg_weight * reg_loss
    if ar_nce_weight > 0.0 and pred_flat.size(0) >= 2:
        nce_tgt = tgt_flat.detach() if detach_ar_nce_tgt else tgt_flat
        total_loss = total_loss + ar_nce_weight * compute_symmetric_infonce(pred_flat, nce_tgt, temperature=ar_nce_temp)
    if gv_nce_weight > 0.0 and e1.size(0) >= 2:
        total_loss = total_loss + gv_nce_weight * compute_symmetric_infonce(e1, e2, temperature=0.10)
    if sadl is not None and all_h is not None and supply_weight > 0.0:
        supply_loss = sadl(all_h, all_z, demand_loss=reg_loss, demand_fn=sigreg)
        total_loss = total_loss + supply_weight * supply_loss.to(total_loss.dtype)
    else:
        supply_loss = torch.zeros((), device=all_z.device, dtype=all_z.dtype)

    return {
        "total_loss": total_loss,
        "ar_mse": ar_mse,
        "ar_cos": ar_cos,
        "gv_mse": gv_mse,
        "reg_loss": reg_loss,
        "supply_loss": supply_loss,
    }


def evaluate_predictor_on_windows(
    encoder: TextJEPAEncoder,
    predictor: LatentPredictor,
    window_data: Dict[str, torch.Tensor],
) -> Tuple[float, float]:
    """Evaluates clean next-sentence prediction MSE and cosine similarity on held-out windows."""
    encoder.eval()
    predictor.eval()
    b_n, t_len, _ = window_data["step_ids"].shape
    d_lat = encoder.config.latent_dim
    flat_z = encode_in_chunks(
        encoder,
        window_data["step_ids"].view(b_n * t_len, -1),
        window_data["step_masks"].view(b_n * t_len, -1),
    )
    z_seq = flat_z.view(b_n, t_len, d_lat)
    valid = window_data["step_valid"]
    pair_mask = valid[:, :-1] & valid[:, 1:]
    with torch.no_grad():
        preds = predictor(z_seq[:, :-1, :], valid_mask=valid[:, :-1])
        tgts = z_seq[:, 1:, :]
        mask_exp = pair_mask.unsqueeze(-1).float()
        mse = ((preds - tgts) ** 2 * mask_exp).sum() / (mask_exp.sum() * d_lat + 1e-6)
        cos = F.cosine_similarity(preds[pair_mask], tgts[pair_mask], dim=-1).mean()
    return float(mse.item()), float(cos.item())


@torch.no_grad()
def evaluate_window_retrieval(
    encoder: TextJEPAEncoder,
    predictor: LatentPredictor,
    window_data: Dict[str, torch.Tensor],
    pool_size: int = 50,
) -> Dict[str, float]:
    """Evaluates 1-step (t=0->1) and multi-step (t=0..2->3) 50-way retrieval P@1 and predictor rank."""
    encoder.eval()
    predictor.eval()
    n = min(200, window_data["step_ids"].size(0))
    if n < pool_size:
        return {"p1_1step": 0.0, "p1_3step": 0.0, "p1_erank": 0.0}
    t_len = window_data["step_ids"].size(1)
    d_lat = encoder.config.latent_dim
    flat_z = encode_in_chunks(
        encoder,
        window_data["step_ids"][:n].reshape(n * t_len, -1),
        window_data["step_masks"][:n].reshape(n * t_len, -1),
    )
    z_seq = flat_z.view(n, t_len, d_lat)
    valid = window_data["step_valid"][:n]

    p1 = predictor(z_seq[:, 0:1, :])[:, -1, :]
    t1 = z_seq[:, 1, :]
    sim1 = F.cosine_similarity(p1[:pool_size].unsqueeze(1), t1[:pool_size].unsqueeze(0), dim=-1)
    acc1 = (sim1.argmax(dim=-1) == torch.arange(pool_size, device=p1.device)).float().mean().item()
    st1 = compute_representation_stats(p1)

    acc3 = 0.0
    if t_len >= 4:
        m3 = valid[:, 3]
        if m3.sum().item() >= pool_size:
            z3 = z_seq[m3]
            p3 = predictor(z3[:pool_size, :3, :])[:, -1, :]
            t3 = z3[:pool_size, 3, :]
            sim3 = F.cosine_similarity(p3.unsqueeze(1), t3.unsqueeze(0), dim=-1)
            acc3 = (sim3.argmax(dim=-1) == torch.arange(pool_size, device=p3.device)).float().mean().item()

    return {"p1_1step": float(acc1), "p1_3step": float(acc3), "p1_erank": float(st1["erank"])}


def plot_ar_progression(logs: Dict, save_path: str, avg_gpu: float):
    """Generates a 4-panel progression plot covering all four stages."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), dpi=140)
    plt.subplots_adjust(wspace=0.32)

    s1 = logs.get("stage1_encoder", {})
    if s1.get("inv_loss"):
        ep = range(1, len(s1["inv_loss"]) + 1)
        axes[0].plot(ep, s1["inv_loss"], color="#16a34a", label="Good-View MSE", linewidth=2)
        axes[0].set_title("1. Pre-train Encoder", fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend(fontsize=8)

    s2 = logs.get("stage2_predictor_warmup", {})
    if s2.get("train_mse"):
        ep = range(1, len(s2["train_mse"]) + 1)
        axes[1].plot(ep, s2["train_mse"], color="#0891b2", label="Train MSE", linewidth=2)
        if s2.get("test_mse"):
            tx = [e for e in range(1, len(s2["train_mse"]) + 1) if e % 10 == 0 or e == len(s2["train_mse"])]
            if len(tx) == len(s2["test_mse"]):
                axes[1].plot(tx, s2["test_mse"], marker="o", color="#dc2626", label="Unseen Test MSE")
        axes[1].set_title("2. Pre-train Predictor", fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(fontsize=8)

    s3 = logs.get("stage3_joint", {})
    if s3.get("ar_mse"):
        ep = range(1, len(s3["ar_mse"]) + 1)
        axes[2].plot(ep, s3["ar_mse"], color="#9333ea", label="Joint AR MSE", linewidth=2)
        axes[2].plot(ep, s3["ar_cos"], color="#2563eb", label="AR Cosine Sim", linestyle="--", linewidth=2)
        axes[2].set_title("3. Joint E+P Co-Training", fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, linestyle=":", alpha=0.6)
        axes[2].legend(fontsize=8)

    s4 = logs.get("stage4_decoder", {})
    if s4.get("ce_loss"):
        ep = range(1, len(s4["ce_loss"]) + 1)
        axes[3].plot(ep, s4["ce_loss"], color="#ea580c", label="Decoder CE Loss", linewidth=2)
        axes[3].set_title("4. Clean Decoder", fontweight="bold")
        axes[3].set_xlabel("Epoch")
        axes[3].grid(True, linestyle=":", alpha=0.6)
        axes[3].legend(fontsize=8)

    fig.suptitle(f"Autoregressive Predictive JEPA (ROCm GPU - Avg Compute: {avg_gpu:.1f}%)", fontweight="bold")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved training progression figure to {save_path}")


def train_autoregressive_jepa(
    save_dir: str = "checkpoints_ar_jepa",
    epochs_s1: int = 25,
    epochs_s2: int = 60,
    epochs_s3: int = 20,
    epochs_s4: int = 70,
    max_fineweb_docs: int = 3500,
    max_windows: int = 6000,
    num_gv_samples: int = 12288,
    max_dec_sentences: int = 24000,
    batch_size: int = 128,
    joint_batch_size: int = 24,
    vocab_size: int = 4000,
    d_model: int = 384,
    num_layers: int = 6,
    num_heads: int = 12,
    latent_dim: int = 256,
    max_steps: int = 8,
    max_seq_len: int = 48,
    stages: str = "1,2,3,4",
    rebuild_caches: bool = False,
    use_sadl: bool = True,
    supply_weight: float = 0.01,
    reg_weight: float = 0.02,
    nce_weight: float = 0.25,
    from_s12: bool = False,
    max_books: int = 0,
    clean_paraphrases: bool = False,
    lr_s3: float = 1e-4,
    gv_weight_s3: float = 1.5,
    ar_mse_weight_s3: float = 0.2,
    ar_nce_weight_s3: float = 0.35,
    ar_nce_temp: float = 0.25,
    resume_s1: bool = False,
    lr_s1: float = 5e-4,
    lr_s2: float = 3e-4,
    dropout_p: float = 0.15,
    max_s3_batches: int = 0,
    lr_s4: float = 1e-3,
    resume_s4: bool = False,
    cycle_weight: float = 0.0,
    dec_noise_scale: float = -1.0,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 68)
    print(f" AUTOREGRESSIVE PREDICTIVE JEPA ON: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f" Model: d={d_model}, L={num_layers}, H={num_heads}, k={latent_dim}, T={max_steps}, L_seq={max_seq_len}, V={vocab_size}")
    print(f" Stages: [{stages}] | S1={epochs_s1}, S2={epochs_s2}, S3(Joint)={epochs_s3}, S4={epochs_s4} | reg_w={reg_weight}, sadl={use_sadl}(lam_S={supply_weight})")
    print("=" * 68)
    os.makedirs(save_dir, exist_ok=True)

    loader = AutoregressiveCorpusLoader(
        max_fineweb_docs=max_fineweb_docs,
        max_tapaco_clusters=30000 if clean_paraphrases else 6000,
        max_wikihow_guides=4500,
        max_recap_anchors=0 if clean_paraphrases else 10000,
        max_books=max_books,
        use_clean_paraphrases=clean_paraphrases,
    )

    tok_path = os.path.join(save_dir, "tokenizer.json")
    if rebuild_caches and os.path.exists(tok_path):
        os.remove(tok_path)

    if os.path.exists(tok_path):
        print(f"[+] Loading tokenizer from {tok_path}...")
        tokenizer = BPETokenizer.load(tok_path)
    else:
        print(f"[*] Training clean BPE tokenizer (target V={vocab_size})...")
        tokenizer = BPETokenizer()
        tokenizer.train(loader.get_tokenizer_corpus(max_texts=10000), target_vocab_size=vocab_size)
        tokenizer.save(tok_path)
    actual_vocab_size = tokenizer.vocab_size

    config = SystemConfig(
        encoder=EncoderConfig(
            vocab_size=actual_vocab_size, max_seq_len=max_seq_len,
            d_model=d_model, num_layers=num_layers, num_heads=num_heads,
            d_ff=d_model * 4, latent_dim=latent_dim, dropout=0.0,
        ),
        sigreg=SIGRegConfig(lam=1.0, num_projections=512, num_integration_steps=17, reg_weight=reg_weight),
        sadl=SADLConfig(enabled=use_sadl, supply_weight=supply_weight),
        decoder=DecoderConfig(
            vocab_size=actual_vocab_size, max_seq_len=max_seq_len,
            d_model=d_model, num_layers=num_layers, num_heads=num_heads,
            d_ff=d_model * 4, latent_dim=latent_dim, num_soft_tokens=4,
            noise_scale=0.35, num_styles=0, style_dim=0,
        ),
        predictor=PredictorConfig(
            latent_dim=latent_dim, max_steps=max_steps,
            d_model=d_model, num_layers=num_layers, num_heads=num_heads,
            d_ff=d_model * 4, dropout=dropout_p, reg_weight=0.0,  # Pure predictive objective (no SIGReg conflict on P)
        ),
        device=str(device),
    )

    encoder = TextJEPAEncoder(config.encoder).to(device)
    sigreg = SIGRegLoss(latent_dim=latent_dim, num_projections=512, knots=17).to(device)
    sadl = (
        SADLLoss(
            alpha=config.sadl.alpha,
            gamma=config.sadl.gamma,
            rho=config.sadl.rho,
            eps=config.sadl.eps,
        ).to(device)
        if use_sadl
        else None
    )
    predictor = LatentPredictor(config.predictor).to(device)
    decoder = ConditionalTextDecoder(config.decoder).to(device)

    enc_p = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    pred_p = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    dec_p = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"[PARAMETER SCALE] Encoder: {enc_p/1e6:.2f}M | Predictor: {pred_p/1e6:.2f}M | Decoder: {dec_p/1e6:.2f}M | Total: {(enc_p+pred_p+dec_p)/1e6:.2f}M params")

    stages_set = {s.strip() for s in stages.split(",")}
    enc_path = os.path.join(save_dir, "encoder.pt")
    enc_s1_path = os.path.join(save_dir, "encoder_s1.pt")
    pred_path = os.path.join(save_dir, "predictor.pt")
    pred_s2_path = os.path.join(save_dir, "predictor_s2.pt")
    dec_path = os.path.join(save_dir, "decoder.pt")

    load_enc = enc_s1_path if (from_s12 and os.path.exists(enc_s1_path)) else enc_path
    load_pred = pred_s2_path if (from_s12 and os.path.exists(pred_s2_path)) else pred_path
    if ("1" not in stages_set or resume_s1) and os.path.exists(load_enc):
        encoder.load_state_dict(torch.load(load_enc, map_location=device)["encoder"])
        print(f"[+] Loaded Encoder checkpoint from {load_enc}")
    if "2" not in stages_set and os.path.exists(load_pred):
        predictor.load_state_dict(torch.load(load_pred, map_location=device)["predictor"])
        print(f"[+] Loaded Predictor checkpoint from {load_pred}")
    if ("4" not in stages_set or resume_s4) and os.path.exists(dec_path):
        decoder.load_state_dict(torch.load(dec_path, map_location=device)["decoder"])
        print(f"[+] Loaded Decoder checkpoint from {dec_path}")

    # Cached GPU tensors
    gv_cache = os.path.join(save_dir, "gv_cache.pt")
    win_train_cache = os.path.join(save_dir, "win_train_cache.pt")
    win_test_cache = os.path.join(save_dir, "win_test_cache.pt")
    dec_cache = os.path.join(save_dir, "dec_cache.pt")

    if rebuild_caches:
        for p in [gv_cache, win_train_cache, win_test_cache, dec_cache]:
            if os.path.exists(p):
                os.remove(p)

    gv_data = torch.load(gv_cache, map_location=device, weights_only=True) if os.path.exists(gv_cache) else None
    if gv_data is None:
        gv_data = build_good_view_tensors(loader, tokenizer, num_gv_samples, max_seq_len, device)
        torch.save(gv_data, gv_cache)

    win_train = torch.load(win_train_cache, map_location=device, weights_only=True) if os.path.exists(win_train_cache) else None
    if win_train is None:
        tr_windows = loader.build_sliding_windows("train", window_size=max_steps, stride=3, max_windows=max_windows)
        win_train = build_window_tensors(tr_windows, tokenizer, max_steps, max_seq_len, device, "train")
        torch.save(win_train, win_train_cache)

    win_test = torch.load(win_test_cache, map_location=device, weights_only=True) if os.path.exists(win_test_cache) else None
    if win_test is None:
        te_windows = loader.build_sliding_windows("test", window_size=max_steps, stride=4, max_windows=1200)
        win_test = build_window_tensors(te_windows, tokenizer, max_steps, max_seq_len, device, "test")
        torch.save(win_test, win_test_cache)

    dec_tensors = torch.load(dec_cache, map_location=device, weights_only=True) if os.path.exists(dec_cache) else None
    if dec_tensors is None:
        clean_sents = loader.get_clean_decoder_sentences(max_sentences=max_dec_sentences)
        d_ids, d_masks = build_decoder_tensors(clean_sents, tokenizer, max_seq_len, device)
        dec_tensors = {"ids": d_ids, "masks": d_masks}
        torch.save(dec_tensors, dec_cache)

    monitor = GPUMonitor(interval=0.5) if torch.cuda.is_available() else None
    if monitor:
        monitor.start()
    logs: Dict = {}

    # ----------------------------------------------------------
    # STAGE 1: PRE-TRAIN ENCODER ON GOOD VIEWS + SIGREG
    # ----------------------------------------------------------
    if "1" in stages_set:
        sadl_tag = f" + SADL(lambda_S={supply_weight})" if sadl is not None else ""
        print(f"\n>>> STAGE 1: PRE-TRAINING ENCODER ON GOOD VIEWS + SIGREG{sadl_tag} ({epochs_s1} Epochs)")
        encoder.train()
        opt1 = create_optimizer(encoder, lr=lr_s1, weight_decay=0.01)
        n_gv = gv_data["v1_ids"].size(0)
        n_batches = max(1, n_gv // batch_size)
        sched1 = get_cosine_schedule_with_warmup(opt1, n_batches, epochs_s1 * n_batches)

        s1_tot, s1_inv, s1_reg = [], [], []
        t0 = time.time()
        for ep in range(epochs_s1):
            perm = torch.randperm(n_gv, device=device)
            ep_t = ep_i = ep_r = 0.0
            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                opt1.zero_grad()
                if sadl is not None:
                    h1, e1 = encoder(gv_data["v1_ids"][idx], attention_mask=gv_data["v1_mask"][idx], return_both=True)
                    h2, e2 = encoder(gv_data["v2_ids"][idx], attention_mask=gv_data["v2_mask"][idx], return_both=True)
                else:
                    h1 = h2 = None
                    e1 = encoder(gv_data["v1_ids"][idx], attention_mask=gv_data["v1_mask"][idx])
                    e2 = encoder(gv_data["v2_ids"][idx], attention_mask=gv_data["v2_mask"][idx])
                tot_l, inv_l, reg_l = compute_lejepa_loss(
                    e1,
                    e2,
                    sigreg,
                    reg_weight=reg_weight,
                    h1=h1,
                    h2=h2,
                    sadl_loss_fn=sadl,
                    supply_weight=supply_weight,
                )
                if nce_weight > 0.0:
                    tot_l = tot_l + nce_weight * compute_symmetric_infonce(e1, e2, temperature=0.10)
                tot_l.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                opt1.step()
                sched1.step()
                ep_t += tot_l.item()
                ep_i += inv_l.item()
                ep_r += reg_l.item()
            s1_tot.append(ep_t / n_batches)
            s1_inv.append(ep_i / n_batches)
            s1_reg.append(ep_r / n_batches)
            if (ep + 1) % 2 == 0 or ep == epochs_s1 - 1:
                probe_h = h1 if h1 is not None else encoder(gv_data["v1_ids"][idx], attention_mask=gv_data["v1_mask"][idx], return_hidden=True)
                h_st = compute_representation_stats(probe_h)
                print(
                    f"  [S1 Epoch {ep+1:02d}/{epochs_s1}] Loss: {s1_tot[-1]:.4f} | Inv MSE: {s1_inv[-1]:.4f} "
                    f"| SIGReg: {s1_reg[-1]:.2f} | f_H={h_st['f']:.3f} erank_H={h_st['erank']:.1f}"
                )
        logs["stage1_encoder"] = {"total_loss": s1_tot, "inv_loss": s1_inv, "reg_loss": s1_reg}
        s1_ckpt = {"encoder": encoder.state_dict(), "history": logs["stage1_encoder"]}
        torch.save(s1_ckpt, enc_path)
        torch.save(s1_ckpt, enc_s1_path)
        print(f"[+] Stage 1 finished in {time.time() - t0:.2f}s (saved backup to {enc_s1_path}).")

    # ----------------------------------------------------------
    # STAGE 2: PRE-TRAIN PREDICTOR ON FROZEN SEQUENCES (PURE MSE)
    # ----------------------------------------------------------
    if "2" in stages_set:
        s2_mode = "MSE + Cosine + InfoNCE" if nce_weight > 0.0 else "Pure MSE"
        print(f"\n>>> STAGE 2: PRE-TRAINING PREDICTOR ON FROZEN SEQUENCES ({epochs_s2} Epochs, {s2_mode})")
        encoder.eval()
        n_tr = win_train["step_ids"].size(0)
        flat_z = encode_in_chunks(
            encoder,
            win_train["step_ids"].view(n_tr * max_steps, max_seq_len),
            win_train["step_masks"].view(n_tr * max_steps, max_seq_len),
        )
        all_z_seq = flat_z.view(n_tr, max_steps, latent_dim)
        all_valid = win_train["step_valid"]

        predictor.train()
        opt2 = create_optimizer(predictor, lr=lr_s2, weight_decay=0.05)
        n_batches = max(1, n_tr // batch_size)
        sched2 = get_cosine_schedule_with_warmup(opt2, n_batches, epochs_s2 * n_batches)

        s2_mse, s2_test_mse = [], []
        best_s2_score = -1e9
        best_s2_cos = -1e9
        best_s2_state = None
        t0 = time.time()
        for ep in range(epochs_s2):
            predictor.train()
            perm = torch.randperm(n_tr, device=device)
            ep_m = 0.0
            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                opt2.zero_grad()
                z_b = all_z_seq[idx]
                v_b = all_valid[idx]
                in_m = v_b[:, :-1]
                pair_m = in_m & v_b[:, 1:]
                inp_z = z_b[:, :-1, :] + 0.12 * torch.randn_like(z_b[:, :-1, :])
                p_out = predictor(inp_z, valid_mask=in_m)
                p_flat = p_out[pair_m]
                t_flat = z_b[:, 1:, :][pair_m]
                mse_l = F.mse_loss(p_flat, t_flat)
                cos_l = 1.0 - F.cosine_similarity(p_flat, t_flat, dim=-1).mean()
                if nce_weight > 0.0:
                    tot_l = 0.10 * mse_l + 0.30 * cos_l + 0.60 * compute_symmetric_infonce(p_flat, t_flat, temperature=0.15)
                else:
                    tot_l = mse_l + 0.5 * cos_l
                tot_l.backward()
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                opt2.step()
                sched2.step()
                ep_m += mse_l.item()
            s2_mse.append(ep_m / n_batches)
            if (ep + 1) % 2 == 0 or ep == epochs_s2 - 1:
                te_mse, te_cos = evaluate_predictor_on_windows(encoder, predictor, win_test)
                ret_s2 = evaluate_window_retrieval(encoder, predictor, win_test)
                s2_test_mse.append(te_mse)
                score = 0.5 * (ret_s2["p1_1step"] + ret_s2["p1_3step"])
                if score > best_s2_score or (score == best_s2_score and te_cos > best_s2_cos):
                    best_s2_score = score
                    best_s2_cos = te_cos
                    best_s2_state = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}
                print(
                    f"  [S2 Epoch {ep+1:02d}/{epochs_s2}] Train MSE: {s2_mse[-1]:.4f} | Test MSE: {te_mse:.4f} "
                    f"| Cos: {te_cos:+.3f} | Win P@1(1s/3s): {ret_s2['p1_1step']*100:.1f}%/{ret_s2['p1_3step']*100:.1f}% "
                    f"(best {best_s2_score*100:.1f}%) | p1_erank={ret_s2['p1_erank']:.1f}"
                )
        if best_s2_state is not None:
            predictor.load_state_dict({k: v.to(device) for k, v in best_s2_state.items()})
        logs["stage2_predictor_warmup"] = {"train_mse": s2_mse, "test_mse": s2_test_mse, "best_test_cos": best_s2_cos, "best_win_p1": best_s2_score}
        s2_ckpt = {"predictor": predictor.state_dict(), "history": logs["stage2_predictor_warmup"]}
        torch.save(s2_ckpt, pred_path)
        torch.save(s2_ckpt, pred_s2_path)
        print(f"[+] Stage 2 finished in {time.time() - t0:.2f}s (restored best Win P@1={best_s2_score*100:.1f}%, Cos={best_s2_cos:+.3f}, saved to {pred_s2_path}).")

    # ----------------------------------------------------------
    # STAGE 3 (JOINT): CO-TRAIN ENCODER + PREDICTOR AUTOREGRESSIVELY
    # ----------------------------------------------------------
    if "3" in stages_set:
        print(f"\n>>> STAGE 3: JOINT AUTOREGRESSIVE CO-TRAINING OF E + P ({epochs_s3} Epochs)")
        for p in encoder.parameters():
            p.requires_grad = True
        for p in predictor.parameters():
            p.requires_grad = True

        n_tr = win_train["step_ids"].size(0)
        n_gv = gv_data["v1_ids"].size(0)
        n_batches_full = max(1, n_tr // joint_batch_size)
        n_batches = min(n_batches_full, max_s3_batches) if max_s3_batches > 0 else n_batches_full
        gv_bs = min(128, n_gv)

        joint_module = torch.nn.ModuleList([encoder, predictor])
        joint_params = list(joint_module.parameters())
        opt3 = create_optimizer(joint_module, lr=lr_s3, weight_decay=0.02)
        sched3 = get_cosine_schedule_with_warmup(opt3, min(100, n_batches), epochs_s3 * n_batches)

        s3_tot, s3_mse, s3_cos, s3_gv, s3_reg, s3_sup = [], [], [], [], [], []
        best_s3_score = -1e9
        best_s3_enc = None
        best_s3_pred = None
        t0 = time.time()
        for ep in range(epochs_s3):
            encoder.train()
            predictor.train()
            perm_doc = torch.randperm(n_tr, device=device)
            ep_t = ep_m = ep_c = ep_g = ep_r = ep_s = 0.0
            for b in range(n_batches):
                idx = perm_doc[b * joint_batch_size : (b + 1) * joint_batch_size]
                gv_idx = torch.randint(0, n_gv, (gv_bs,), device=device)

                opt3.zero_grad()
                step_out = compute_joint_autoregressive_step(
                    encoder=encoder,
                    predictor=predictor,
                    sigreg=sigreg,
                    doc_ids=win_train["step_ids"][idx],
                    doc_masks=win_train["step_masks"][idx],
                    doc_valid=win_train["step_valid"][idx],
                    gv1_ids=gv_data["v1_ids"][gv_idx],
                    gv1_masks=gv_data["v1_mask"][gv_idx],
                    gv2_ids=gv_data["v2_ids"][gv_idx],
                    gv2_masks=gv_data["v2_mask"][gv_idx],
                    gv_weight=gv_weight_s3,
                    cos_weight=0.5,
                    reg_weight=reg_weight,
                    sadl=sadl,
                    supply_weight=supply_weight,
                    ar_nce_weight=ar_nce_weight_s3 if nce_weight > 0.0 else 0.0,
                    gv_nce_weight=nce_weight,
                    ar_mse_weight=ar_mse_weight_s3,
                    ar_nce_temp=ar_nce_temp,
                    detach_ar_nce_tgt=False,
                )
                step_out["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(joint_params, 1.0)
                opt3.step()
                sched3.step()

                ep_t += step_out["total_loss"].item()
                ep_m += step_out["ar_mse"].item()
                ep_c += step_out["ar_cos"].item()
                ep_g += step_out["gv_mse"].item()
                ep_r += step_out["reg_loss"].item()
                ep_s += step_out["supply_loss"].item()

            s3_tot.append(ep_t / n_batches)
            s3_mse.append(ep_m / n_batches)
            s3_cos.append(ep_c / n_batches)
            s3_gv.append(ep_g / n_batches)
            s3_reg.append(ep_r / n_batches)
            s3_sup.append(ep_s / n_batches)

            if n_batches >= 400 or (ep + 1) % 2 == 0 or ep == epochs_s3 - 1:
                te_mse, te_cos = evaluate_predictor_on_windows(encoder, predictor, win_test)
                ret_s3 = evaluate_window_retrieval(encoder, predictor, win_test)
                with torch.no_grad():
                    probe_n = min(384, n_gv)
                    probe_h = encoder(gv_data["v1_ids"][:probe_n], attention_mask=gv_data["v1_mask"][:probe_n], return_hidden=True)
                    h_st = compute_representation_stats(probe_h)
                s3_score = 0.5 * (ret_s3["p1_1step"] + ret_s3["p1_3step"])
                if h_st["f"] >= 0.70 and s3_score >= best_s3_score:
                    best_s3_score = s3_score
                    best_s3_enc = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
                    best_s3_pred = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}
                print(
                    f"  [S3 Joint {ep+1:02d}/{epochs_s3}] AR MSE: {s3_mse[-1]:.4f} | AR Cos: {s3_cos[-1]:+.3f} "
                    f"| GV MSE: {s3_gv[-1]:.4f} | SIGReg: {s3_reg[-1]:.2f} | SADL: {s3_sup[-1]:.4f} "
                    f"| f_H={h_st['f']:.3f} erank_H={h_st['erank']:.1f} | Test MSE: {te_mse:.4f} (Cos {te_cos:+.3f}) "
                    f"| Win P@1(1s/3s): {ret_s3['p1_1step']*100:.1f}%/{ret_s3['p1_3step']*100:.1f}%"
                )

        if best_s3_enc is not None and best_s3_pred is not None:
            encoder.load_state_dict({k: v.to(device) for k, v in best_s3_enc.items()})
            predictor.load_state_dict({k: v.to(device) for k, v in best_s3_pred.items()})
        logs["stage3_joint"] = {
            "total_loss": s3_tot, "ar_mse": s3_mse, "ar_cos": s3_cos,
            "gv_mse": s3_gv, "reg_loss": s3_reg, "supply_loss": s3_sup,
            "best_win_p1": best_s3_score,
        }
        s3_enc = {"encoder": encoder.state_dict(), "history": logs["stage3_joint"]}
        s3_pred = {"predictor": predictor.state_dict(), "history": logs["stage3_joint"]}
        torch.save(s3_enc, enc_path)
        torch.save(s3_pred, pred_path)
        torch.save(s3_enc, os.path.join(save_dir, "encoder_s3.pt"))
        torch.save(s3_pred, os.path.join(save_dir, "predictor_s3.pt"))
        print(f"[+] Stage 3 Joint Co-Training finished in {time.time() - t0:.2f}s (backups saved).")

    # Calibrate decoder noise to predictor's empirical unseen validation RMSE
    encoder.eval()
    predictor.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in predictor.parameters():
        p.requires_grad = False

    test_mse, test_cos = evaluate_predictor_on_windows(encoder, predictor, win_test)
    calibrated_noise = dec_noise_scale if dec_noise_scale >= 0.0 else max(0.15, min(0.75, float(test_mse ** 0.5) * 0.5))
    decoder.noise_scale = calibrated_noise
    print(f"\n[*] Calibrated Decoder noise_scale = {calibrated_noise:.3f} (Unseen Test MSE={test_mse:.4f}, Cos={test_cos:+.3f})")

    # ----------------------------------------------------------
    # STAGE 4: CLEAN DECODER TRAINING ON PROSE + WIKIHOW
    # ----------------------------------------------------------
    if "4" in stages_set:
        cycle_tag = f" + Cycle({cycle_weight})" if cycle_weight > 0.0 else ""
        print(f"\n>>> STAGE 4: TRAINING CLEAN DECODER WITH CALIBRATED NOISE{cycle_tag} ({epochs_s4} Epochs)")
        d_ids = dec_tensors["ids"]
        d_masks = dec_tensors["masks"]
        n_dec = d_ids.size(0)
        dec_latents = encode_in_chunks(encoder, d_ids, d_masks, chunk_size=512)

        decoder.train()
        opt4 = create_optimizer(decoder, lr=lr_s4, weight_decay=0.01)
        n_batches = max(1, n_dec // batch_size)
        sched4 = get_cosine_schedule_with_warmup(opt4, n_batches, epochs_s4 * n_batches)

        s4_ce = []
        t0 = time.time()
        for ep in range(epochs_s4):
            perm = torch.randperm(n_dec, device=device)
            ep_ce = ep_cyc = 0.0
            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                c_ids = d_ids[idx]
                c_m = d_masks[idx]
                e = dec_latents[idx]

                opt4.zero_grad()
                logits = decoder(e=e, input_ids=c_ids[:, :-1], attention_mask=c_m[:, :-1], inject_noise=True)
                ce_loss = F.cross_entropy(logits.reshape(-1, actual_vocab_size), c_ids[:, 1:].reshape(-1), ignore_index=tokenizer.pad_id)
                loss4 = ce_loss
                if cycle_weight > 0.0:
                    probs = F.softmax(logits / 0.5, dim=-1)
                    soft_tok = probs @ encoder.token_embedding.weight
                    bos_emb = encoder.token_embedding(c_ids[:, :1])
                    z_rec = encoder.forward_from_embeddings(torch.cat([bos_emb, soft_tok], dim=1), attention_mask=c_m)
                    cyc_loss = 1.0 - F.cosine_similarity(z_rec, e, dim=-1).mean()
                    loss4 = loss4 + cycle_weight * cyc_loss
                    ep_cyc += cyc_loss.item()
                loss4.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                opt4.step()
                sched4.step()
                ep_ce += ce_loss.item()
            s4_ce.append(ep_ce / n_batches)
            if (ep + 1) % 5 == 0 or ep == epochs_s4 - 1:
                cyc_str = f" | Cycle CosLoss: {ep_cyc / n_batches:.4f}" if cycle_weight > 0.0 else ""
                print(f"  [S4 Epoch {ep+1:02d}/{epochs_s4}] Decoder CE Loss: {s4_ce[-1]:.4f}{cyc_str}")
        logs["stage4_decoder"] = {"ce_loss": s4_ce, "noise_scale": calibrated_noise}
        s4_ckpt = {"decoder": decoder.state_dict(), "history": logs["stage4_decoder"]}
        torch.save(s4_ckpt, dec_path)
        torch.save(s4_ckpt, os.path.join(save_dir, "decoder_s4.pt"))
        print(f"[+] Stage 4 finished in {time.time() - t0:.2f}s (backup saved).")

    avg_gpu = monitor.stop() if monitor else 0.0
    if monitor:
        print(f"\n[GPU TELEMETRY] Average GPU Compute Utilization: {avg_gpu:.1f}%")

    # Save progress plot (merge persisted stage histories when running isolated stages)
    merged_logs = dict(logs)
    hist_path = os.path.join(save_dir, "history.json")
    if os.path.exists(hist_path):
        try:
            with open(hist_path) as hf:
                prev_logs = json.load(hf)
            prev_logs.update(merged_logs)
            merged_logs = prev_logs
        except Exception:
            pass
    with open(hist_path, "w") as hf:
        json.dump(merged_logs, hf, indent=2)
    plot_ar_progression(merged_logs, os.path.join(save_dir, "ar_loss_progression.png"), avg_gpu)

    pipeline = GoodViewTextJEPA(config=config, tokenizer=tokenizer, encoder=encoder, decoder=decoder, predictor=predictor).to(device)
    eval_pairs = [loader.sample_good_view_pair(tapaco_prob=1.0) for _ in range(300)]
    eval_trajs = [[title] + steps for title, steps in loader.wikihow_test_guides[:100]]
    bench = run_full_benchmark(pipeline, positive_pairs=eval_pairs, trajectories=eval_trajs, eval_sts=True)
    win_ret = evaluate_window_retrieval(encoder, predictor, win_test)
    bench["window_retrieval"] = win_ret
    print(f"  - Held-Out Window P@1 (1-step / 3-step): {win_ret['p1_1step']*100:.2f}% / {win_ret['p1_3step']*100:.2f}% | p1_erank={win_ret['p1_erank']:.2f}")
    if loader.book_test_documents:
        book_trajs = [d for d in loader.book_test_documents if len(d) >= 4][:100]
        book_bench = benchmark_predictor_reasoning(pipeline, trajectories=book_trajs)
        bench["book_predictor"] = book_bench
        print(f"  - Held-Out Books P@1: {book_bench['next_step_p@1']*100:.2f}% | MRR: {book_bench['next_step_mrr']:.4f} | Step1 Cos: {book_bench.get('drift_cosine_step1', 0.0):.4f}")

    print("\n" + "=" * 70)
    print(" UNSEEN TEST ROLLOUTS (AUTOREGRESSIVE PREDICTIVE JEPA)")
    print("=" * 70)
    for title, gt_steps in loader.wikihow_test_guides[:5]:
        n_eval = min(3, len(gt_steps))
        res = pipeline.run_inference(title, reasoning_steps=n_eval, max_tokens_per_step=24)
        print(f"\n[Unseen Goal]: \"{title}\"")
        for i in range(n_eval):
            pred_z = res["latent_trajectory"][:, i + 1, :]
            gt_z = pipeline.encode_text(gt_steps[i])
            cos_sim = F.cosine_similarity(pred_z, gt_z).item()
            print(f"  Step {i+1} (Cosine={cos_sim:+.3f}):")
            print(f"    * Predicted:    {res['rendered_steps'][i]}")
            print(f"    * Ground Truth: {gt_steps[i]}")

    report = {
        "architecture": "Autoregressive Predictive JEPA (Tier A)",
        "avg_gpu_utilization_pct": avg_gpu,
        "unseen_next_step_mse": test_mse,
        "unseen_next_step_cos": test_cos,
        "calibrated_decoder_noise": calibrated_noise,
        "benchmark_metrics": bench,
        "training_history": logs,
    }
    with open(os.path.join(save_dir, "ar_benchmark_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] All checkpoints and benchmark saved to {save_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description="Autoregressive Predictive JEPA Trainer")
    parser.add_argument("--stages", type=str, default="1,2,3,4", help="Comma-separated stages: 1,2,3,4")
    parser.add_argument("--save_dir", type=str, default="checkpoints_ar_jepa")
    parser.add_argument("--epochs_s1", type=int, default=25)
    parser.add_argument("--epochs_s2", type=int, default=60)
    parser.add_argument("--epochs_s3", type=int, default=20)
    parser.add_argument("--epochs_s4", type=int, default=70)
    parser.add_argument("--max_fineweb_docs", type=int, default=3500)
    parser.add_argument("--max_windows", type=int, default=6000)
    parser.add_argument("--num_gv_samples", type=int, default=12288)
    parser.add_argument("--max_dec_sentences", type=int, default=24000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--joint_batch_size", type=int, default=24)
    parser.add_argument("--vocab_size", type=int, default=4000)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=48)
    parser.add_argument("--max_books", type=int, default=0, help="Number of PG-19 books to stream")
    parser.add_argument("--clean_paraphrases", action="store_true", help="Use PAWS+MRPC+QQP+filtered TaPaCo (no Recap)")
    parser.add_argument("--lr_s3", type=float, default=1e-4)
    parser.add_argument("--gv_weight_s3", type=float, default=1.5)
    parser.add_argument("--ar_mse_weight_s3", type=float, default=0.2)
    parser.add_argument("--ar_nce_weight_s3", type=float, default=0.35)
    parser.add_argument("--ar_nce_temp", type=float, default=0.25)
    parser.add_argument("--resume_s1", action="store_true", help="Resume Stage 1 from encoder.pt/encoder_s1.pt")
    parser.add_argument("--lr_s1", type=float, default=5e-4)
    parser.add_argument("--lr_s2", type=float, default=3e-4)
    parser.add_argument("--dropout_p", type=float, default=0.15)
    parser.add_argument("--max_s3_batches", type=int, default=0)
    parser.add_argument("--lr_s4", type=float, default=1e-3)
    parser.add_argument("--resume_s4", action="store_true")
    parser.add_argument("--cycle_weight", type=float, default=0.0, help="Frozen-encoder soft cycle-consistency weight in Stage 4")
    parser.add_argument("--dec_noise_scale", type=float, default=-1.0, help="Override Stage 4 decoder noise scale when >= 0")
    parser.add_argument("--rebuild_caches", action="store_true")
    parser.add_argument("--no_sadl", action="store_true", help="Disable SADL Supply Regularizer on encoder H")
    parser.add_argument("--supply_weight", type=float, default=0.01, help="SADL Supply Regularizer weight lambda_S")
    parser.add_argument("--reg_weight", type=float, default=0.02, help="SIGReg regularization weight lambda")
    parser.add_argument("--nce_weight", type=float, default=0.25, help="Symmetric InfoNCE weight (0 disables)")
    parser.add_argument("--from_s12", action="store_true", help="Load encoder_s1.pt / predictor_s2.pt backups when skipping Stage 1/2")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_autoregressive_jepa(
        save_dir=args.save_dir,
        epochs_s1=args.epochs_s1,
        epochs_s2=args.epochs_s2,
        epochs_s3=args.epochs_s3,
        epochs_s4=args.epochs_s4,
        max_fineweb_docs=args.max_fineweb_docs,
        max_windows=args.max_windows,
        num_gv_samples=args.num_gv_samples,
        max_dec_sentences=args.max_dec_sentences,
        batch_size=args.batch_size,
        joint_batch_size=args.joint_batch_size,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim,
        max_seq_len=args.max_seq_len,
        stages=args.stages,
        rebuild_caches=args.rebuild_caches,
        use_sadl=not args.no_sadl,
        supply_weight=args.supply_weight,
        reg_weight=args.reg_weight,
        nce_weight=args.nce_weight,
        from_s12=args.from_s12,
        max_books=args.max_books,
        clean_paraphrases=args.clean_paraphrases,
        lr_s3=args.lr_s3,
        gv_weight_s3=args.gv_weight_s3,
        ar_mse_weight_s3=args.ar_mse_weight_s3,
        ar_nce_weight_s3=args.ar_nce_weight_s3,
        ar_nce_temp=args.ar_nce_temp,
        resume_s1=args.resume_s1,
        lr_s1=args.lr_s1,
        lr_s2=args.lr_s2,
        dropout_p=args.dropout_p,
        max_s3_batches=args.max_s3_batches,
        lr_s4=args.lr_s4,
        resume_s4=args.resume_s4,
        cycle_weight=args.cycle_weight,
        dec_noise_scale=args.dec_noise_scale,
    )
