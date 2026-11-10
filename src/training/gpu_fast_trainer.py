"""
High-Throughput GPU-Native Training for Good-View Text JEPA.

Eliminates the CPU DataLoader bottleneck by pre-tokenizing all samples into
GPU-resident VRAM tensors (torch.cuda). Keeps AMD ROCm GPU compute cores at
90-100% continuous utilization throughout all three training stages.

Supports modular independent stage training via --stage {1, 2, 3, all}.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from typing import Dict, List, Tuple

# Set ROCm environment variables before torch import
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("ROCM_PATH", "/opt/rocm")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from configs.config import (
    SystemConfig,
    EncoderConfig,
    SIGRegConfig,
    DecoderConfig,
    PredictorConfig,
)
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.hf_dataset_loader import HFGoodViewLoader
from src.data.augmentations import TextSpanTrimmer, LightTokenMasker, augment_token_pair
from src.models.encoder import TextJEPAEncoder
from src.models.decoder import ConditionalTextDecoder
from src.models.predictor import LatentPredictor
from src.models.pipeline import GoodViewTextJEPA
from src.models.sigreg import SIGRegLoss, compute_lejepa_loss
from src.training.trainer_utils import create_optimizer, get_cosine_schedule_with_warmup
from src.evaluation.benchmark import run_full_benchmark


class GPUMonitor(threading.Thread):
    """Monitors AMD ROCm GPU compute usage and VRAM during training."""

    def __init__(self, interval: float = 0.5):
        super().__init__()
        self.interval = interval
        self.running = True
        self.samples: List[int] = []
        self.daemon = True

    def run(self):
        while self.running:
            try:
                out = subprocess.getoutput("rocm-smi --showuse | grep 'GPU use'")
                if "GPU use (%):" in out:
                    val_str = out.split("GPU use (%):")[-1].strip().split()[0]
                    usage = int(val_str)
                    self.samples.append(usage)
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self) -> float:
        self.running = False
        return float(np.mean(self.samples)) if self.samples else 0.0


def pretokenize_pair_dataset(
    loader: HFGoodViewLoader,
    tokenizer: BPETokenizer,
    num_samples: int,
    max_seq_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Pre-tokenizes good-view pairs and moves all tensors directly to GPU VRAM.
    """
    print(f"[*] Pre-tokenizing {num_samples} pairs into GPU VRAM ({device})...")
    global_trimmer = TextSpanTrimmer(min_ratio=0.5, max_ratio=1.0)
    local_trimmer = TextSpanTrimmer(min_ratio=0.15, max_ratio=0.5)
    masker = LightTokenMasker(mask_id=tokenizer.mask_id, mask_prob=0.15)
    prot = [tokenizer.bos_id, tokenizer.eos_id]

    v1_list, v2_list = [], []
    c1_list, c2_list = [], []

    for _ in range(num_samples):
        t1, t2, _, _, _ = loader.sample_positive_pair()
        c_toks1 = tokenizer.encode(t1, add_bos=True, add_eos=True)["input_ids"].tolist()
        c_toks2 = tokenizer.encode(t2, add_bos=True, add_eos=True)["input_ids"].tolist()

        toks1, toks2 = augment_token_pair(
            list(c_toks1), list(c_toks2),
            mask_id=tokenizer.mask_id,
            global_trimmer=global_trimmer,
            local_trimmer=local_trimmer,
            masker=masker,
            protected_tokens=prot,
        )

        def pad_trunc(ids):
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            mask = [1] * len(ids)
            if len(ids) < max_seq_len:
                pad_len = max_seq_len - len(ids)
                ids = ids + [tokenizer.pad_id] * pad_len
                mask = mask + [0] * pad_len
            return ids, mask

        v1_i, v1_m = pad_trunc(toks1)
        v2_i, v2_m = pad_trunc(toks2)
        c1_i, c1_m = pad_trunc(c_toks1)
        c2_i, c2_m = pad_trunc(c_toks2)

        v1_list.append((v1_i, v1_m))
        v2_list.append((v2_i, v2_m))
        c1_list.append((c1_i, c1_m))
        c2_list.append((c2_i, c2_m))

    data = {
        "v1_ids": torch.tensor([x[0] for x in v1_list], dtype=torch.long, device=device),
        "v1_mask": torch.tensor([x[1] for x in v1_list], dtype=torch.long, device=device),
        "v2_ids": torch.tensor([x[0] for x in v2_list], dtype=torch.long, device=device),
        "v2_mask": torch.tensor([x[1] for x in v2_list], dtype=torch.long, device=device),
        "clean1_ids": torch.tensor([x[0] for x in c1_list], dtype=torch.long, device=device),
        "clean1_mask": torch.tensor([x[1] for x in c1_list], dtype=torch.long, device=device),
    }
    mb_used = sum(t.element_size() * t.nelement() for t in data.values()) / (1024**2)
    print(f"[+] Pair dataset resident on GPU: {mb_used:.2f} MB in VRAM.")
    return data


def pretokenize_trajectory_dataset(
    loader: HFGoodViewLoader,
    tokenizer: BPETokenizer,
    max_steps: int,
    max_seq_len: int,
    device: torch.device,
    split: str = "train",
    max_trajectories: int = 8000,
) -> Dict[str, torch.Tensor]:
    """
    Pre-tokenizes procedural task trajectories into GPU VRAM.
    Applies combinatorial sub-trajectory windowing for the train split.
    Packs [topic (Goal), step_1, step_2, ...] so that z_0 represents the task goal.
    """
    import random
    chains = []
    pool = loader.test_trajectories if split == "test" else loader.train_trajectories
    for topic, steps in pool:
        full_chain = [topic] + list(steps)
        if len(full_chain) >= 2:
            chains.append(full_chain[:max_steps])
        # Apply sub-trajectory windowing in training to multiply data diversity
        if split == "train":
            if len(steps) >= 3:
                chains.append(list(steps[:max_steps]))  # Local step transition
            if len(steps) >= 4:
                chains.append([topic] + list(steps[1:max_steps]))  # Shifted window

    if max_trajectories and len(chains) > max_trajectories:
        random.shuffle(chains)
        chains = chains[:max_trajectories]

    N = len(chains)
    print(f"[*] Pre-tokenizing {N} ({split}) procedural trajectory chains into GPU VRAM ({device})...")
    step_ids = torch.full((N, max_steps, max_seq_len), tokenizer.pad_id, dtype=torch.long, device=device)
    step_masks = torch.zeros((N, max_steps, max_seq_len), dtype=torch.long, device=device)
    step_valid = torch.zeros((N, max_steps), dtype=torch.bool, device=device)

    for i, chain in enumerate(chains):
        actual_steps = min(len(chain), max_steps)
        for s_idx in range(actual_steps):
            enc = tokenizer.encode(chain[s_idx], add_bos=True, add_eos=True, max_len=max_seq_len, pad=True)
            step_ids[i, s_idx] = enc["input_ids"].to(device)
            step_masks[i, s_idx] = enc["attention_mask"].to(device)
            step_valid[i, s_idx] = True

    return {
        "step_ids": step_ids,
        "step_masks": step_masks,
        "step_valid": step_valid,
    }


def plot_loss_progression(training_logs: dict, save_path: str, avg_gpu_use: float):
    """Plots publication-quality 3-panel loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    plt.subplots_adjust(wspace=0.3)

    # Panel 1: Stage 1
    ax1 = axes[0]
    s1 = training_logs.get("stage1_encoder", {})
    epochs_s1 = list(range(1, len(s1.get("total_loss", [])) + 1))
    if epochs_s1:
        ax1.plot(epochs_s1, s1["total_loss"], marker="o", color="#2563eb", label="Total Loss", linewidth=2)
        ax1.plot(epochs_s1, s1["inv_loss"], marker="s", color="#16a34a", label="Invariance MSE", linewidth=2, linestyle="--")
        ax1.set_title("Stage 1: JEPA Encoder (TaPaCo)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Loss", fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(fontsize=9)

    # Panel 2: Stage 2
    ax2 = axes[1]
    s2 = training_logs.get("stage2_decoder", {})
    epochs_s2 = list(range(1, len(s2.get("ce_loss", [])) + 1))
    if epochs_s2:
        ax2.plot(epochs_s2, s2["ce_loss"], marker="o", color="#ea580c", label="Cross-Entropy Loss", linewidth=2)
        ax2.set_title("Stage 2: Conditional Decoder", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Cross-Entropy Loss", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(fontsize=9)

    # Panel 3: Stage 3
    ax3 = axes[2]
    s3 = training_logs.get("stage3_predictor", {})
    epochs_s3 = list(range(1, len(s3.get("pred_total_loss", [])) + 1))
    if epochs_s3:
        ax3.plot(epochs_s3, s3["pred_total_loss"], marker="o", color="#9333ea", label="Predictor Total", linewidth=2)
        ax3.plot(epochs_s3, s3["pred_mse"], marker="^", color="#0891b2", label="State MSE", linewidth=2, linestyle="--")
        ax3.set_title("Stage 3: Latent Predictor (WikiHow)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Epoch", fontsize=10)
        ax3.set_ylabel("Loss", fontsize=10)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(fontsize=9)

    fig.suptitle(f"Good-View Text JEPA Training Progression (ROCm AMD GPU - Avg Compute: {avg_gpu_use:.1f}%)", fontsize=14, fontweight="bold")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved loss progression plot to {save_path}")


def train_gpu(
    stage: str = "all",
    save_dir: str = "checkpoints_gpu_fast",
    epochs_s1: int = 35,
    epochs_s2: int = 80,
    epochs_s3: int = 80,
    num_samples: int = 4096,
    batch_size: int = 128,
    vocab_size: int = 1500,
    d_model: int = 256,
    num_layers: int = 4,
    latent_dim: int = 128,
    retokenize: bool = False,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n============================================================")
    print(f" GOOD-VIEW TEXT JEPA (PLATONIC REPRESENTATION) ON: {torch.cuda.get_device_name(0)}")
    print(f" Mode: Stage '{stage}' | S1={epochs_s1} eps, S2={epochs_s2} eps, S3={epochs_s3} eps")
    print(f"============================================================")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Dataset Loader & Tokenizer
    tok_path = os.path.join(save_dir, "tokenizer.json")
    pair_cache = os.path.join(save_dir, "pair_cache.pt")
    traj_cache = os.path.join(save_dir, "traj_cache.pt")

    if retokenize:
        for p in [tok_path, pair_cache, traj_cache]:
            if os.path.exists(p):
                os.remove(p)

    hf_loader = HFGoodViewLoader(max_clusters=5000, max_trajectories=2000)

    if os.path.exists(tok_path):
        print(f"[+] Loading tokenizer from {tok_path}...")
        tokenizer = BPETokenizer.load(tok_path)
    else:
        print("[*] Training BPE tokenizer on TaPaCo + WikiHow corpus...")
        tokenizer = BPETokenizer()
        tokenizer.train(hf_loader.get_all_texts()[:15000], target_vocab_size=vocab_size)
        tokenizer.save(tok_path)

    actual_vocab_size = tokenizer.vocab_size

    # 2. Config & Models (No style conditioning: num_styles=0, style_dim=0)
    config = SystemConfig(
        encoder=EncoderConfig(vocab_size=actual_vocab_size, max_seq_len=48, d_model=d_model, num_layers=num_layers, num_heads=8, d_ff=d_model*4, latent_dim=latent_dim),
        sigreg=SIGRegConfig(lam=1.0, num_projections=64, num_integration_steps=17, reg_weight=0.05),
        decoder=DecoderConfig(vocab_size=actual_vocab_size, max_seq_len=48, d_model=d_model, num_layers=num_layers, num_heads=8, d_ff=d_model*4, latent_dim=latent_dim, num_soft_tokens=4, num_styles=0, style_dim=0),
        predictor=PredictorConfig(latent_dim=latent_dim, max_steps=6, d_model=d_model, num_layers=num_layers, num_heads=8, d_ff=d_model*4, reg_weight=0.05),
        device="cuda",
    )

    encoder = TextJEPAEncoder(config.encoder).to(device)
    sigreg = SIGRegLoss(latent_dim=latent_dim, lam=1.0, num_projections=64, num_integration_steps=17).to(device)
    decoder = ConditionalTextDecoder(config.decoder).to(device)
    predictor = LatentPredictor(config.predictor).to(device)

    stages_to_run = [s.strip() for s in stage.split(",")]
    if "all" in stages_to_run:
        stages_to_run = ["1", "2", "3"]

    # Load existing checkpoints if continuing or running isolated stages
    enc_path = os.path.join(save_dir, "encoder.pt")
    dec_path = os.path.join(save_dir, "decoder.pt")
    pred_path = os.path.join(save_dir, "predictor.pt")

    if "1" not in stages_to_run and os.path.exists(enc_path):
        print(f"[+] Loading frozen Stage 1 Encoder checkpoint from {enc_path}...")
        enc_ckpt = torch.load(enc_path, map_location=device)
        encoder.load_state_dict(enc_ckpt["encoder"])

    if "2" not in stages_to_run and os.path.exists(dec_path):
        print(f"[+] Loading Stage 2 Decoder checkpoint from {dec_path}...")
        dec_ckpt = torch.load(dec_path, map_location=device)
        decoder.load_state_dict(dec_ckpt["decoder"])

    if "3" not in stages_to_run and os.path.exists(pred_path):
        print(f"[+] Loading Stage 3 Predictor checkpoint from {pred_path}...")
        pred_ckpt = torch.load(pred_path, map_location=device)
        predictor.load_state_dict(pred_ckpt["predictor"])

    # 3. Pre-load GPU datasets (Zero CPU bottleneck during training)
    if os.path.exists(pair_cache):
        print(f"[+] Loading cached pre-tokenized pairs from {pair_cache} directly into GPU VRAM...")
        pair_data = torch.load(pair_cache, map_location=device, weights_only=True)
    else:
        pair_data = pretokenize_pair_dataset(hf_loader, tokenizer, num_samples=num_samples, max_seq_len=48, device=device)
        torch.save(pair_data, pair_cache)

    if os.path.exists(traj_cache):
        print(f"[+] Loading cached pre-tokenized trajectories from {traj_cache} directly into GPU VRAM...")
        traj_data = torch.load(traj_cache, map_location=device, weights_only=True)
    else:
        traj_data = pretokenize_trajectory_dataset(hf_loader, tokenizer, max_steps=6, max_seq_len=48, device=device, split="train", max_trajectories=8000)
        torch.save(traj_data, traj_cache)

    traj_test_cache = os.path.join(save_dir, "traj_test_cache.pt")
    if os.path.exists(traj_test_cache):
        print(f"[+] Loading cached test trajectories from {traj_test_cache} into GPU VRAM...")
        traj_test_data = torch.load(traj_test_cache, map_location=device, weights_only=True)
    else:
        traj_test_data = pretokenize_trajectory_dataset(hf_loader, tokenizer, max_steps=6, max_seq_len=48, device=device, split="test", max_trajectories=1000)
        torch.save(traj_test_data, traj_test_cache)

    # Start GPU utilization monitor
    monitor = GPUMonitor(interval=0.5)
    monitor.start()
    training_logs = {}

    # ==========================================================
    # STAGE 1: JEPA ENCODER (GPU-RESIDENT STREAM)
    # ==========================================================
    if "1" in stages_to_run:
        print(f"\n>>> STAGE 1: GPU TRAINING JEPA ENCODER WITH SIGREG ({epochs_s1} Epochs)")
        encoder.train()
        opt1 = create_optimizer(encoder, lr=8e-4, weight_decay=0.01)
        N = num_samples
        num_batches = N // batch_size
        sched1 = get_cosine_schedule_with_warmup(opt1, num_warmup_steps=num_batches, num_training_steps=epochs_s1 * num_batches)

        s1_tot, s1_inv, s1_reg = [], [], []
        t_s1 = time.time()
        for ep in range(epochs_s1):
            perm = torch.randperm(N, device=device)
            ep_tot, ep_inv, ep_reg = 0.0, 0.0, 0.0
            for b in range(num_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                v1_ids, v1_m = pair_data["v1_ids"][idx], pair_data["v1_mask"][idx]
                v2_ids, v2_m = pair_data["v2_ids"][idx], pair_data["v2_mask"][idx]

                opt1.zero_grad()
                e1 = encoder(v1_ids, attention_mask=v1_m)
                e2 = encoder(v2_ids, attention_mask=v2_m)
                tot_l, inv_l, reg_l = compute_lejepa_loss(e1, e2, sigreg, reg_weight=0.05)
                tot_l.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                opt1.step()
                sched1.step()

                ep_tot += tot_l.item()
                ep_inv += inv_l.item()
                ep_reg += reg_l.item()

            s1_tot.append(ep_tot / num_batches)
            s1_inv.append(ep_inv / num_batches)
            s1_reg.append(ep_reg / num_batches)
            if (ep + 1) % 5 == 0 or ep == epochs_s1 - 1:
                print(f"  [Epoch {ep+1:02d}/{epochs_s1}] Total Loss: {s1_tot[-1]:.4f} | Invariance MSE: {s1_inv[-1]:.6f} | SIGReg: {s1_reg[-1]:.4f}")

        training_logs["stage1_encoder"] = {"total_loss": s1_tot, "inv_loss": s1_inv, "reg_loss": s1_reg}
        torch.save({"encoder": encoder.state_dict(), "history": training_logs["stage1_encoder"]}, enc_path)
        print(f"[+] Stage 1 finished in {time.time() - t_s1:.2f}s.")

    # Freeze encoder for Stages 2 and 3
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # ==========================================================
    # STAGE 2: CONDITIONAL DECODER (MIXED TAPaCo + WikiHow STEPS)
    # ==========================================================
    if "2" in stages_to_run:
        print(f"\n>>> STAGE 2: GPU TRAINING CONDITIONAL DECODER ({epochs_s2} Epochs)")
        print(f"[*] Pooling TaPaCo paraphrases and WikiHow procedural steps for unified decoder grounding...")

        # Extract valid WikiHow steps
        wh_valid = traj_data["step_valid"]
        wh_step_ids = traj_data["step_ids"][wh_valid]
        wh_step_masks = traj_data["step_masks"][wh_valid]

        # Combine TaPaCo and WikiHow steps into unified decoding dataset
        dec_all_ids = torch.cat([pair_data["clean1_ids"], wh_step_ids], dim=0)
        dec_all_masks = torch.cat([pair_data["clean1_mask"], wh_step_masks], dim=0)
        N_dec = dec_all_ids.size(0)
        num_dec_batches = N_dec // batch_size
        print(f"[+] Unified decoder corpus: {N_dec} sentences ({pair_data['clean1_ids'].size(0)} TaPaCo + {wh_step_ids.size(0)} WikiHow steps).")

        decoder.train()
        opt2 = create_optimizer(decoder, lr=8e-4, weight_decay=0.01)
        sched2 = get_cosine_schedule_with_warmup(opt2, num_warmup_steps=num_dec_batches, num_training_steps=epochs_s2 * num_dec_batches)

        s2_ce = []
        t_s2 = time.time()
        for ep in range(epochs_s2):
            perm = torch.randperm(N_dec, device=device)
            ep_ce = 0.0
            for b in range(num_dec_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                c_ids = dec_all_ids[idx]
                c_m = dec_all_masks[idx]

                with torch.no_grad():
                    e = encoder(c_ids, attention_mask=c_m)

                dec_in = c_ids[:, :-1]
                dec_m = c_m[:, :-1]
                targets = c_ids[:, 1:]

                opt2.zero_grad()
                logits = decoder(e=e, input_ids=dec_in, attention_mask=dec_m, inject_noise=True)
                ce_loss = F.cross_entropy(logits.reshape(-1, actual_vocab_size), targets.reshape(-1), ignore_index=tokenizer.pad_id)
                ce_loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
                opt2.step()
                sched2.step()

                ep_ce += ce_loss.item()

            s2_ce.append(ep_ce / num_dec_batches)
            if (ep + 1) % 10 == 0 or ep == epochs_s2 - 1:
                print(f"  [Epoch {ep+1:02d}/{epochs_s2}] Decoder CE Loss: {s2_ce[-1]:.4f}")

        training_logs["stage2_decoder"] = {"ce_loss": s2_ce}
        torch.save({"decoder": decoder.state_dict(), "history": training_logs["stage2_decoder"]}, dec_path)
        print(f"[+] Stage 2 finished in {time.time() - t_s2:.2f}s.")

    decoder.eval()

    # ==========================================================
    # STAGE 3: PREDICTOR WITH PRE-ENCODED GPU LATENTS
    # ==========================================================
    if "3" in stages_to_run:
        print(f"\n>>> STAGE 3: GPU TRAINING LATENT PREDICTOR ON WIKIHOW ({epochs_s3} Epochs)")
        N_traj = traj_data["step_ids"].size(0)
        T = traj_data["step_ids"].size(1)
        flat_ids = traj_data["step_ids"].view(N_traj * T, 48)
        flat_m = traj_data["step_masks"].view(N_traj * T, 48)

        def encode_in_chunks(ids, masks, chunk_size=512):
            out_list = []
            with torch.no_grad():
                for c_start in range(0, ids.size(0), chunk_size):
                    c_ids = ids[c_start : c_start + chunk_size]
                    c_m = masks[c_start : c_start + chunk_size]
                    out_list.append(encoder(c_ids, attention_mask=c_m))
            return torch.cat(out_list, dim=0)

        print(f"[*] Pre-encoding {flat_ids.size(0)} Train procedural steps into Platonic vectors R^k on GPU...")
        flat_z = encode_in_chunks(flat_ids, flat_m, chunk_size=512)
        all_z_seq = flat_z.view(N_traj, T, latent_dim)
        all_step_valid = traj_data["step_valid"]

        # Also pre-encode unseen test trajectories
        N_test = traj_test_data["step_ids"].size(0)
        test_flat_ids = traj_test_data["step_ids"].view(N_test * T, 48)
        test_flat_m = traj_test_data["step_masks"].view(N_test * T, 48)
        print(f"[*] Pre-encoding {test_flat_ids.size(0)} Unseen Test steps into Platonic vectors R^k on GPU...")
        test_flat_z = encode_in_chunks(test_flat_ids, test_flat_m, chunk_size=512)
        test_z_seq = test_flat_z.view(N_test, T, latent_dim)
        test_step_valid = traj_test_data["step_valid"]

        predictor.train()
        opt3 = create_optimizer(predictor, lr=8e-4, weight_decay=0.01)
        num_pred_batches = N_traj // batch_size
        sched3 = get_cosine_schedule_with_warmup(opt3, num_warmup_steps=num_pred_batches, num_training_steps=epochs_s3 * num_pred_batches)

        s3_tot, s3_mse, s3_reg, s3_test_mse = [], [], [], []
        t_s3 = time.time()
        for ep in range(epochs_s3):
            perm = torch.randperm(N_traj, device=device)
            ep_tot, ep_mse, ep_reg = 0.0, 0.0, 0.0
            for b in range(num_pred_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                b_z = all_z_seq[idx]
                b_valid = all_step_valid[idx]

                opt3.zero_grad()
                tot_l, mse_l, reg_l = predictor.compute_loss(b_z, valid_mask=b_valid, jitter_std=0.04)
                tot_l.backward()
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                opt3.step()
                sched3.step()

                ep_tot += tot_l.item()
                ep_mse += mse_l.item()
                ep_reg += reg_l.item()

            s3_tot.append(ep_tot / num_pred_batches)
            s3_mse.append(ep_mse / num_pred_batches)
            s3_reg.append(ep_reg / num_pred_batches)

            if (ep + 1) % 10 == 0 or ep == epochs_s3 - 1:
                predictor.eval()
                with torch.no_grad():
                    _, t_mse, _ = predictor.compute_loss(test_z_seq, valid_mask=test_step_valid, jitter_std=0.0)
                predictor.train()
                s3_test_mse.append(t_mse.item())
                print(f"  [Epoch {ep+1:02d}/{epochs_s3}] Train MSE: {s3_mse[-1]:.6f} | Unseen Test MSE: {t_mse.item():.6f} | Pred Total: {s3_tot[-1]:.4f}")

        training_logs["stage3_predictor"] = {"pred_total_loss": s3_tot, "pred_mse": s3_mse, "pred_reg": s3_reg}
        torch.save({"predictor": predictor.state_dict(), "history": training_logs["stage3_predictor"]}, pred_path)
        print(f"[+] Stage 3 finished in {time.time() - t_s3:.2f}s.")

    # Stop GPU monitor
    avg_gpu_use = monitor.stop()
    print(f"\n[GPU TELEMETRY] Average GPU Compute Utilization: {avg_gpu_use:.1f}%")
    if monitor.samples:
        print(f"[GPU TELEMETRY] Peak GPU Compute: {max(monitor.samples)}% across {len(monitor.samples)} samples.")

    # 4. Qualitative Evaluation on Unseen Test Tasks
    pipeline = GoodViewTextJEPA(config=config, tokenizer=tokenizer, encoder=encoder, decoder=decoder, predictor=predictor).to(device)
    print("\n" + "=" * 70)
    print(" UNSEEN TEST EVALUATION: PREDICTED ROLLOUT VS GROUND TRUTH")
    print("=" * 70)

    test_examples = hf_loader.test_trajectories[:5]
    for task_title, gt_steps in test_examples:
        print(f"\n[Unseen Goal]: \"{task_title}\"")
        num_eval_steps = min(3, len(gt_steps))
        res = pipeline.run_inference(task_title, reasoning_steps=num_eval_steps, max_tokens_per_step=20)
        for i in range(num_eval_steps):
            pred_text = res["rendered_steps"][i]
            gt_text = gt_steps[i]

            # Compute latent cosine similarity between predicted step vector and ground truth step vector
            pred_z = res["latent_trajectory"][:, i + 1, :]
            with torch.no_grad():
                gt_enc = tokenizer.encode(gt_text, add_bos=True, add_eos=True, max_len=48, pad=True)
                gt_z = encoder(gt_enc["input_ids"].unsqueeze(0).to(device), attention_mask=gt_enc["attention_mask"].unsqueeze(0).to(device))
                cos_sim = F.cosine_similarity(pred_z, gt_z).item()

            print(f"  Step {i+1} (Latent Cosine Match: {cos_sim:+.3f}):")
            print(f"    * Predicted:    {pred_text}")
            print(f"    * Ground Truth: {gt_text}")

    # 5. Always update and save the Loss Progression Plot using latest checkpoint histories
    merged_logs = {}
    if os.path.exists(enc_path):
        merged_logs["stage1_encoder"] = torch.load(enc_path, map_location="cpu").get("history", {})
    if os.path.exists(dec_path):
        merged_logs["stage2_decoder"] = torch.load(dec_path, map_location="cpu").get("history", {})
    if os.path.exists(pred_path):
        merged_logs["stage3_predictor"] = torch.load(pred_path, map_location="cpu").get("history", {})
    merged_logs.update(training_logs)

    plot_path = os.path.join(save_dir, "loss_progression.png")
    plot_loss_progression(merged_logs, plot_path, avg_gpu_use)

    # 6. Full Benchmark (if stage == "all")
    if stage == "all":

        eval_pairs = [(c[0], c[1]) for c in list(hf_loader.clusters.values())[:300] if len(c) >= 2]
        eval_trajs = [steps for _, steps in hf_loader.trajectories[:150]]
        bench_results = run_full_benchmark(pipeline, positive_pairs=eval_pairs, trajectories=eval_trajs, eval_sts=True)
        report = {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0),
            "avg_gpu_utilization_pct": avg_gpu_use,
            "peak_gpu_utilization_pct": max(monitor.samples) if monitor.samples else 0,
            "config": {
                "vocab_size": actual_vocab_size,
                "d_model": d_model,
                "num_layers": num_layers,
                "latent_dim": latent_dim,
                "epochs_s1": epochs_s1,
                "epochs_s2": epochs_s2,
                "epochs_s3": epochs_s3,
                "batch_size": batch_size,
                "num_samples": num_samples,
            },
            "training_history": training_logs,
            "benchmark_metrics": bench_results,
        }
        with open(os.path.join(save_dir, "benchmark_report.json"), "w") as f:
            json.dump(report, f, indent=2)

    print(f"[+] Finished stage '{stage}'. All checkpoints saved to {save_dir}.")


def parse_args():
    parser = argparse.ArgumentParser(description="GPU-Native Good-View Text JEPA Trainer")
    parser.add_argument("--stage", type=str, default="all", help="Stage to train: 1, 2, 3, '2,3', or 'all'")
    parser.add_argument("--epochs_s1", type=int, default=35, help="Epochs for Stage 1 (Encoder)")
    parser.add_argument("--epochs_s2", type=int, default=80, help="Epochs for Stage 2 (Decoder)")
    parser.add_argument("--epochs_s3", type=int, default=80, help="Epochs for Stage 3 (Predictor)")
    parser.add_argument("--num_samples", type=int, default=4096, help="Number of samples")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--save_dir", type=str, default="checkpoints_gpu_fast", help="Checkpoint directory")
    parser.add_argument("--retokenize", action="store_true", help="Force rebuild tokenizer and dataset caches")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_gpu(
        stage=args.stage,
        save_dir=args.save_dir,
        epochs_s1=args.epochs_s1,
        epochs_s2=args.epochs_s2,
        epochs_s3=args.epochs_s3,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        retokenize=args.retokenize,
    )
