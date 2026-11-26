"""
CLI Runner for the Baby-JEPA Three-Arm Experiment (Active vs. Random vs. Passive).
Saves JSON metrics and sample-efficiency plots to checkpoints_baby_jepa/.
"""

import argparse
import json
import os
from typing import Dict, List
import torch

from configs.baby_config import BabyJEPAConfig, BabyTrainingConfig, TabletopEnvConfig
from src.evaluation.baby_benchmark import (
    FrozenDINOEncoder,
    FrozenMiniLMEncoder,
    build_eval_datasets,
)
from src.training.baby_jepa_trainer import BabyJEPATrainer


def _save_plots(results: Dict[str, Dict], output_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_to_plot = [
        ("shape_probe_from_confounded", "Shape Probe (Trained Confounded -> Tested Decorrelated)"),
        ("procrustes_shape_naming_acc", "Zero-Shot Procrustes Naming Acc (Shape)"),
        ("dino_cka", "Alignment vs Frozen DINO ViT-S/8 (Linear CKA)"),
        ("dino_mutual_10nn", "Alignment vs Frozen DINO ViT-S/8 (Mutual 10-NN)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, title) in zip(axes.flat, metrics_to_plot):
        for arm, res in results.items():
            hist = res["history"]
            xs = [h["interactions"] for h in hist if key in h]
            ys = [h[key] for h in hist if key in h]
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2, label=arm)
        ax.set_title(title)
        ax.set_xlabel("Environment Frames / Interactions")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "baby_jepa_sample_efficiency.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    # Also save a single-arm / detailed diagnostics & action-histogram plot if 1 arm or first arm
    first_arm = next(iter(results))
    res0 = results[first_arm]
    hist0 = res0["history"]
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))

    xs_tr = [h["interactions"] for h in hist0 if "train_temporal_wmse" in h]
    if xs_tr:
        if "train_gv_inv_loss" in hist0[1] if len(hist0) > 1 else False:
            axes2[0, 0].plot(xs_tr, [h["train_gv_inv_loss"] for h in hist0 if "train_gv_inv_loss" in h], marker="d", label="gv_inv")
        axes2[0, 0].plot(xs_tr, [h["train_temporal_wmse"] for h in hist0 if "train_temporal_wmse" in h], marker="o", label="temporal_wmse")
        axes2[0, 0].plot(xs_tr, [h["train_cross_modal_loss"] for h in hist0 if "train_cross_modal_loss" in h], marker="s", label="cross_modal")
        axes2[0, 0].plot(xs_tr, [h["train_sigreg_loss"] for h in hist0 if "train_sigreg_loss" in h], marker="^", label="sigreg")
        axes2[0, 0].set_title(f"{first_arm}: Training Losses")
        axes2[0, 0].set_xlabel("Interactions")
        axes2[0, 0].grid(alpha=0.3)
        axes2[0, 0].legend()

    xs_all = [h["interactions"] for h in hist0]
    axes2[0, 1].plot(xs_all, [h["erank_Z"] for h in hist0], marker="o", label="erank_Z")
    axes2[0, 1].plot(xs_all, [h["erank_H"] for h in hist0], marker="s", label="erank_H")
    axes2[0, 1].set_title(f"{first_arm}: Effective Rank (Z & H)")
    axes2[0, 1].set_xlabel("Interactions")
    axes2[0, 1].grid(alpha=0.3)
    axes2[0, 1].legend()

    axes2[1, 0].plot(xs_all, [h["f_H"] for h in hist0], marker="o", label="f_H (pre-proj)")
    axes2[1, 0].plot(xs_all, [h["f_Z"] for h in hist0], marker="s", label="f_Z (post-proj)")
    axes2[1, 0].plot(xs_all, [h["mean_cos_Z"] for h in hist0], marker="^", label="mean_cos_Z")
    axes2[1, 0].set_title(f"{first_arm}: Supply/Demand & Isotropy")
    axes2[1, 0].set_xlabel("Interactions")
    axes2[1, 0].grid(alpha=0.3)
    axes2[1, 0].legend()

    act_counts = res0.get("action_counts", {})
    if act_counts:
        names = list(act_counts.keys())
        vals = [act_counts[k] for k in names]
        axes2[1, 1].barh(names, vals)
        axes2[1, 1].set_title(f"{first_arm}: Selected Action Distribution")
        axes2[1, 1].set_xlabel("Count")
        axes2[1, 1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "baby_jepa_diagnostics.png"), dpi=150)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Run Baby-JEPA 3-arm active vs passive benchmark")
    parser.add_argument("--arms", nargs="+", default=["active", "active_spectral", "random", "passive"])
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--batch_envs", type=int, default=16)
    parser.add_argument("--episode_length", type=int, default=16)
    parser.add_argument("--train_steps", type=int, default=4)
    parser.add_argument("--sigreg_weight", type=float, default=0.05)
    parser.add_argument("--sadl_weight", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="checkpoints_baby_jepa")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Baby-JEPA] Running on device={device} | arms={args.arms} | episodes={args.episodes}")

    env_cfg = TabletopEnvConfig(episode_length=args.episode_length)
    eval_splits = build_eval_datasets(env_cfg, device=device, seed=999)

    dino_enc = FrozenDINOEncoder.from_local_cache()
    dino_feats = None
    if dino_enc is not None:
        dino_enc = dino_enc.to(device)
        with torch.no_grad():
            imgs = eval_splits["decorrelated_eval"]["images"]
            dino_feats = torch.cat(
                [dino_enc(imgs[i : i + 32].to(device)) for i in range(0, imgs.size(0), 32)], dim=0
            )
        print("[Baby-JEPA] Loaded frozen DINO ViT-S/8 from local cache.")
    else:
        print("[Baby-JEPA] DINO cache not found; skipping DINO CKA/kNN metrics.")

    minilm_enc = FrozenMiniLMEncoder.from_local_cache(torch.device(device))
    if minilm_enc is not None:
        print("[Baby-JEPA] Loaded frozen all-MiniLM-L6-v2 from local cache.")

    all_results: Dict[str, Dict] = {}
    for arm in args.arms:
        print(f"\n--- Running arm: {arm} ---")
        cfg = BabyJEPAConfig(
            env=env_cfg,
            training=BabyTrainingConfig(
                arm=arm,
                total_episodes=args.episodes,
                batch_envs=args.batch_envs,
                train_steps_per_episode=args.train_steps,
                sigreg_weight=args.sigreg_weight,
                sadl_weight=args.sadl_weight,
                eval_every_episodes=args.eval_every,
                seed=args.seed,
                device=device,
            ),
        )
        trainer = BabyJEPATrainer(
            cfg,
            dino_encoder=dino_enc,
            minilm_encoder=minilm_enc,
            eval_splits=eval_splits,
            dino_eval_feats=dino_feats,
        )
        res = trainer.run(checkpoint_dir=args.output_dir, verbose=True)
        all_results[arm] = res
        fin = res["final"]
        print(
            f"[{arm}] interactions={fin['interactions']} | "
            f"shape_conf->dec={fin['shape_probe_from_confounded']:.3f} | "
            f"color_conf->dec={fin['color_probe_from_confounded']:.3f} | "
            f"procrustes_shape={fin['procrustes_shape_naming_acc']:.3f} | "
            f"dino_cka={fin.get('dino_cka', 0.0):.3f} | "
            f"erank_Z={fin['erank_Z']:.1f}"
        )

    report_path = os.path.join(args.output_dir, "baby_jepa_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    _save_plots(all_results, args.output_dir)
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
