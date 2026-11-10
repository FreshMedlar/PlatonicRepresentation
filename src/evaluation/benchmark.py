"""
Comprehensive Benchmark Suite for Good-View Text JEPA / Platonic Representations.

Evaluates:
1. Stage 1: Encoder Geometry & Semantic Alignment
   - Alignment & Uniformity (Wang & Isola, 2020)
   - Effective Rank / SVD Singular Value Entropy (Anti-collapse check)
   - Semantic Textual Similarity (STS-B Spearman rho)
2. Stage 2: Conditional Text Decoder Fidelity
   - Round-trip Latent Cosine Recovery: cos(f(x), f(d(f(x))))
   - Word Overlap & ROUGE-L Token Reconstruction
3. Stage 3: Latent State-Transition Predictor Reasoning
   - Next-Step State Retrieval: P@1 and MRR against distractor states
   - Autoregressive Latent Drift Curves: MSE & Cosine across rollout steps k
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from src.models.pipeline import GoodViewTextJEPA


def compute_effective_rank(embeddings: torch.Tensor) -> float:
    """
    Computes effective rank = exp(Entropy(singular values)).
    embeddings: (N, D)
    """
    # Center embeddings
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(centered, full_matrices=False)
    s = s[s > 1e-7]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
    return math.exp(entropy)


def benchmark_encoder_geometry(
    pipeline: GoodViewTextJEPA,
    positive_pairs: List[Tuple[str, str]],
    all_texts: Optional[List[str]] = None,
    num_samples: int = 500,
) -> Dict[str, float]:
    """
    Computes Wang-Isola Alignment and Uniformity, plus Effective Rank in Platonic space R^k.
    - Alignment: E_{(x, x+)} [ ||f(x) - f(x+)||^2 ] (Lower is better)
    - Uniformity: log E_{x, y} [ exp(-2 * ||f(x) - f(y)||^2) ] (Lower is better)
    - Effective Rank: Dimensional usage in R^k (Higher is better, max = latent_dim)
    """
    pairs = positive_pairs[:num_samples]
    if not pairs:
        return {"alignment": 0.0, "uniformity": 0.0, "effective_rank": 0.0}

    vecs1, vecs2 = [], []
    for x1, x2 in pairs:
        v1 = pipeline.encode_text(x1)
        v2 = pipeline.encode_text(x2)
        # Normalize to unit sphere for metric consistency
        v1 = F.normalize(v1, p=2, dim=-1)
        v2 = F.normalize(v2, p=2, dim=-1)
        vecs1.append(v1)
        vecs2.append(v2)

    t1 = torch.cat(vecs1, dim=0)  # (N, D)
    t2 = torch.cat(vecs2, dim=0)  # (N, D)

    # 1. Alignment
    align = torch.norm(t1 - t2, dim=-1).pow(2).mean().item()

    # 2. Uniformity
    all_vecs = torch.cat([t1, t2], dim=0)
    # Subsample if too large for pairwise distance
    if len(all_vecs) > 1000:
        perm = torch.randperm(len(all_vecs))[:1000]
        sub_vecs = all_vecs[perm]
    else:
        sub_vecs = all_vecs

    pdist_sq = torch.cdist(sub_vecs, sub_vecs, p=2).pow(2)
    # Mask diagonal
    eye_mask = torch.eye(len(sub_vecs), device=sub_vecs.device, dtype=torch.bool)
    off_diag = pdist_sq[~eye_mask]
    uniform = torch.log(torch.exp(-2.0 * off_diag).mean() + 1e-12).item()

    # 3. Effective Rank of representation batch
    eff_rank = compute_effective_rank(all_vecs)

    return {
        "alignment": float(align),
        "uniformity": float(uniform),
        "effective_rank": float(eff_rank),
        "latent_dim": int(all_vecs.size(-1)),
    }


def benchmark_encoder_sts(
    pipeline: GoodViewTextJEPA,
    dataset_name: str = "sentence-transformers/stsb",
    split: str = "validation",
    max_samples: int = 1000,
) -> Dict[str, float]:
    """
    Evaluates Stage 1 Platonic encoder on Semantic Textual Similarity Benchmark (STS-B).
    Returns Spearman's rank correlation rho between latent cosine similarity and gold score.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return {"spearman_rho": 0.0, "samples_evaluated": 0, "error": "datasets not installed"}

    try:
        ds = load_dataset(dataset_name, split=f"{split}[:{max_samples}]")
    except Exception as e:
        return {"spearman_rho": 0.0, "samples_evaluated": 0, "error": str(e)}

    gold_scores = []
    pred_sims = []

    for row in ds:
        s1 = row.get("sentence1")
        s2 = row.get("sentence2")
        score = row.get("score")
        if s1 is None or s2 is None or score is None:
            continue

        e1 = pipeline.encode_text(s1)
        e2 = pipeline.encode_text(s2)
        sim = F.cosine_similarity(e1, e2).item()

        gold_scores.append(float(score))
        pred_sims.append(sim)

    if len(pred_sims) < 10:
        return {"spearman_rho": 0.0, "samples_evaluated": len(pred_sims)}

    corr, pval = spearmanr(gold_scores, pred_sims)
    return {
        "spearman_rho": float(corr),
        "p_value": float(pval),
        "samples_evaluated": len(pred_sims),
    }


def benchmark_decoder_reconstruction(
    pipeline: GoodViewTextJEPA,
    test_texts: List[str],
    max_samples: int = 100,
    max_tokens: int = 24,
) -> Dict[str, float]:
    """
    Evaluates Stage 2 Conditional Text Decoder round-trip reconstruction:
    x -> z = f(x) -> x_hat = d(z)
    Measures:
    - latent_cosine_recovery: cos(f(x), f(x_hat))
    - token_overlap_f1: unigram token F1 between x and x_hat
    """
    samples = test_texts[:max_samples]
    if not samples:
        return {"latent_cosine_recovery": 0.0, "token_overlap_f1": 0.0}

    cos_recoveries = []
    f1_scores = []

    for text in samples:
        z = pipeline.encode_text(text)
        rendered = pipeline.decode_state(z, style_id=0, max_new_tokens=max_tokens)
        z_rec = pipeline.encode_text(rendered)

        cos_sim = F.cosine_similarity(z, z_rec).item()
        cos_recoveries.append(cos_sim)

        # Unigram F1 score
        ref_tokens = text.lower().split()
        cand_tokens = rendered.lower().split()
        if not ref_tokens or not cand_tokens:
            f1 = 0.0
        else:
            ref_set = set(ref_tokens)
            cand_set = set(cand_tokens)
            overlap = len(ref_set & cand_set)
            prec = overlap / len(cand_set) if cand_set else 0.0
            rec = overlap / len(ref_set) if ref_set else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

    return {
        "latent_cosine_recovery": float(np.mean(cos_recoveries)),
        "token_overlap_f1": float(np.mean(f1_scores)),
        "samples_evaluated": len(samples),
    }


def benchmark_predictor_reasoning(
    pipeline: GoodViewTextJEPA,
    trajectories: List[List[str]],
    pool_size: int = 50,
    max_samples: int = 100,
) -> Dict[str, Any]:
    """
    Evaluates Stage 3 Latent Predictor on Next-Step Retrieval and Multi-Step Drift:
    1. Next-Step Retrieval: Given z_(1:t), does predicted z_hat_(t+1) rank true z_(t+1)
       ahead of random distractor steps in the candidate pool?
    2. Multi-step Drift: MSE and Cosine similarity for horizon k in [1, 2, 3].
    """
    valid_trajs = [t for t in trajectories if len(t) >= 2][:max_samples]
    if not valid_trajs:
        return {
            "next_step_p@1": 0.0,
            "next_step_mrr": 0.0,
            "drift_cosine_step1": 0.0,
            "samples_evaluated": 0,
        }

    # Gather distractor pool
    all_steps = [step for t in valid_trajs for step in t]
    distractor_vecs = torch.cat(
        [pipeline.encode_text(s) for s in all_steps[: max(pool_size * 2, 100)]], dim=0
    )

    p1_correct = 0
    mrr_accum = 0.0
    drift_cos = {1: [], 2: [], 3: []}
    drift_mse = {1: [], 2: [], 3: []}
    count = 0

    pipeline.predictor.eval()
    with torch.no_grad():
        for traj in valid_trajs:
            # Encode each step into latent vector
            z_seq = torch.cat([pipeline.encode_text(s) for s in traj], dim=0)  # (T, latent_dim)
            T = z_seq.size(0)

            # 1. Next-step retrieval evaluation at t = 1
            z_hist = z_seq[0:1].unsqueeze(0)  # (1, 1, latent_dim)
            z_pred_next = pipeline.predictor(z_hist)[:, -1, :]  # (1, latent_dim)
            z_target = z_seq[1:2]  # (1, latent_dim)

            # Build candidates: 1 true target + pool_size - 1 distractors
            distractors = distractor_vecs[: pool_size - 1]
            candidates = torch.cat([z_target, distractors], dim=0)  # (pool_size, latent_dim)
            sims = F.cosine_similarity(z_pred_next, candidates)

            ranked_indices = torch.argsort(sims, descending=True)
            rank_of_gold = (ranked_indices == 0).nonzero(as_tuple=True)[0].item() + 1

            if rank_of_gold == 1:
                p1_correct += 1
            mrr_accum += 1.0 / rank_of_gold
            count += 1

            # 2. Multi-step rollout drift evaluation
            current_traj = z_seq[0:1].unsqueeze(0)  # (1, 1, latent_dim)
            for k in range(1, min(4, T)):
                pred_step = pipeline.predictor(current_traj)[:, -1:, :]
                target_k = z_seq[k : k + 1].unsqueeze(0)
                cos_k = F.cosine_similarity(
                    pred_step.squeeze(1), target_k.squeeze(1)
                ).item()
                mse_k = F.mse_loss(pred_step, target_k).item()

                drift_cos[k].append(cos_k)
                drift_mse[k].append(mse_k)
                # Autoregressive update for multi-step rollout
                current_traj = torch.cat([current_traj, pred_step], dim=1)

    results = {
        "next_step_p@1": float(p1_correct / count) if count > 0 else 0.0,
        "next_step_mrr": float(mrr_accum / count) if count > 0 else 0.0,
        "samples_evaluated": count,
    }

    for k in [1, 2, 3]:
        if drift_cos[k]:
            results[f"drift_cosine_step{k}"] = float(np.mean(drift_cos[k]))
            results[f"drift_mse_step{k}"] = float(np.mean(drift_mse[k]))

    return results


def run_full_benchmark(
    pipeline: GoodViewTextJEPA,
    positive_pairs: List[Tuple[str, str]],
    trajectories: List[List[str]],
    eval_sts: bool = True,
) -> Dict[str, Any]:
    """
    Executes the full evaluation suite and returns a unified benchmark dictionary.
    """
    print("\n" + "=" * 65)
    print(" GOOD-VIEW TEXT JEPA / PLATONIC REPRESENTATION BENCHMARK")
    print("=" * 65)

    report = {}

    # 1. Geometry
    print("\n[1/4] Evaluating Stage 1 Geometry (Alignment, Uniformity, Effective Rank)...")
    geom = benchmark_encoder_geometry(pipeline, positive_pairs=positive_pairs)
    report["geometry"] = geom
    print(f"  - Alignment L_align:   {geom['alignment']:.4f} (lower is better)")
    print(f"  - Uniformity L_unif:   {geom['uniformity']:.4f} (lower is better)")
    print(
        f"  - Effective Rank:      {geom['effective_rank']:.2f} / {geom['latent_dim']} (higher is better)"
    )

    # 2. STS-B
    if eval_sts:
        print("\n[2/4] Evaluating Stage 1 Semantic Textual Similarity (STS-B)...")
        sts = benchmark_encoder_sts(pipeline)
        report["sts"] = sts
        if "error" in sts:
            print(f"  - STS-B skipped/error: {sts['error']}")
        else:
            print(f"  - STS-B Spearman rho:  {sts['spearman_rho']:.4f}")

    # 3. Decoder Reconstruction
    print("\n[3/4] Evaluating Stage 2 Conditional Text Decoder...")
    sample_texts = [p[0] for p in positive_pairs[:50]]
    dec = benchmark_decoder_reconstruction(pipeline, test_texts=sample_texts)
    report["decoder"] = dec
    print(f"  - Latent Cosine Recovery: {dec['latent_cosine_recovery']:.4f}")
    print(f"  - Token Overlap F1:       {dec['token_overlap_f1']:.4f}")

    # 4. Predictor Reasoning
    print("\n[4/4] Evaluating Stage 3 Latent Predictor Reasoning...")
    pred = benchmark_predictor_reasoning(pipeline, trajectories=trajectories)
    report["predictor"] = pred
    print(f"  - Next-Step P@1 (top-1):  {pred['next_step_p@1'] * 100:.2f}%")
    print(f"  - Next-Step MRR:          {pred['next_step_mrr']:.4f}")
    if "drift_cosine_step1" in pred:
        print(f"  - Step 1 Rollout Cosine:  {pred['drift_cosine_step1']:.4f}")
    if "drift_cosine_step2" in pred:
        print(f"  - Step 2 Rollout Cosine:  {pred['drift_cosine_step2']:.4f}")
    if "drift_cosine_step3" in pred:
        print(f"  - Step 3 Rollout Cosine:  {pred['drift_cosine_step3']:.4f}")

    print("\n" + "=" * 65)
    print(" BENCHMARK COMPLETE")
    print("=" * 65)
    return report

