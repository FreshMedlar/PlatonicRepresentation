"""
Training script for Good-View Text JEPA on Real-World Datasets (TaPaCo + GSM8K).
Executes:
1. Load TaPaCo paraphrase clusters & GSM8K reasoning trajectories
2. Train subword BPE Tokenizer on the combined corpus
3. Stage 1: Train Good-View JEPA Encoder f_theta with SIGReg anti-collapse
4. Stage 2: Train Conditional Text Decoder d_psi with frozen encoder
5. Stage 3: Train Latent State-Transition Predictor P_phi on continuous reasoning paths
6. Run full benchmark suite and save results
"""

import argparse
import json
import os
import sys
import time

# Ensure ROCm flags from ~/.bashrc are set before PyTorch initialization
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("ROCM_PATH", "/opt/rocm")

import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import (
    SystemConfig,
    EncoderConfig,
    SIGRegConfig,
    DecoderConfig,
    PredictorConfig,
)
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.hf_dataset_loader import HFGoodViewLoader
from src.data.parallel_dataset import GoodViewPairDataset
from src.data.sequential_dataset import SequentialReasoningDataset
from src.models.encoder import TextJEPAEncoder
from src.models.decoder import ConditionalTextDecoder
from src.models.predictor import LatentPredictor
from src.models.pipeline import GoodViewTextJEPA
from src.training.train_encoder import train_jepa_encoder
from src.training.train_decoder import train_conditional_decoder
from src.training.train_predictor import train_latent_predictor
from src.evaluation.benchmark import run_full_benchmark


def plot_loss_progression(training_logs: dict, save_path: str):
    """Generates and saves a publication-quality 3-panel loss progression graph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    plt.subplots_adjust(wspace=0.3)

    # Panel 1: Stage 1 JEPA Encoder
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

    # Panel 2: Stage 2 Conditional Text Decoder
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

    # Panel 3: Stage 3 Latent Predictor
    ax3 = axes[2]
    s3 = training_logs.get("stage3_predictor", {})
    epochs_s3 = list(range(1, len(s3.get("pred_total_loss", [])) + 1))
    if epochs_s3:
        ax3.plot(epochs_s3, s3["pred_total_loss"], marker="o", color="#9333ea", label="Predictor Total", linewidth=2)
        ax3.plot(epochs_s3, s3["pred_mse"], marker="^", color="#0284c7", label="State MSE", linewidth=2, linestyle="--")
        ax3.set_title("Stage 3: Latent Predictor (GSM8K)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Epoch", fontsize=10)
        ax3.set_ylabel("Loss", fontsize=10)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(fontsize=9)

    plt.suptitle("Good-View Text JEPA Loss Progression Across Training Stages", fontsize=14, fontweight="bold", y=1.03)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved loss progression plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Good-View Text JEPA on Real Datasets")
    parser.add_argument("--save-dir", type=str, default="checkpoints_real", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per training stage")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of training samples per stage")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--vocab-size", type=int, default=500, help="Tokenizer vocab size")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--output-report", type=str, default="benchmark_report.json", help="Path to save benchmark report")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[*] Training Good-View Text JEPA on Real Datasets using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load Real Datasets
    t0 = time.time()
    hf_loader = HFGoodViewLoader(
        max_clusters=min(5000, max(args.num_samples * 2, 2000)),
        max_trajectories=min(1500, max(args.num_samples, 500)),
    )
    load_time = time.time() - t0
    print(f"[+] Dataset initialization completed in {load_time:.2f}s.")

    # 2. Train Tokenizer
    tok_path = os.path.join(args.save_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        print(f"[+] Loading existing tokenizer from {tok_path}")
        tokenizer = BPETokenizer.load(tok_path)
    else:
        print("[*] Training BPE tokenizer on TaPaCo + GSM8K text corpus...")
        corpus = hf_loader.get_all_texts()
        # Train on a diverse sample of the corpus for speed
        train_corpus = corpus[:10000]
        tokenizer = BPETokenizer()
        tokenizer.train(train_corpus, target_vocab_size=args.vocab_size)
        tokenizer.save(tok_path)
        print(f"[+] Tokenizer trained and saved to {tok_path} (Vocab size: {tokenizer.vocab_size}).")

    vocab_sz = tokenizer.vocab_size
    latent_dim = 64
    d_model = 128

    config = SystemConfig(
        encoder=EncoderConfig(
            vocab_size=vocab_sz,
            max_seq_len=48,
            d_model=d_model,
            num_layers=2,
            num_heads=4,
            d_ff=256,
            latent_dim=latent_dim,
        ),
        sigreg=SIGRegConfig(
            lam=1.0,
            num_projections=32,
            num_integration_steps=17,
            reg_weight=0.05,
        ),
        decoder=DecoderConfig(
            vocab_size=vocab_sz,
            max_seq_len=48,
            d_model=d_model,
            num_layers=2,
            num_heads=4,
            d_ff=256,
            latent_dim=latent_dim,
            num_soft_tokens=4,
            noise_scale=0.05,
            num_styles=4,
            style_dim=16,
        ),
        predictor=PredictorConfig(
            latent_dim=latent_dim,
            max_steps=6,
            d_model=d_model,
            num_layers=2,
            num_heads=4,
            d_ff=256,
            reg_weight=0.05,
        ),
        device=device,
    )

    encoder = TextJEPAEncoder(config.encoder).to(device)
    decoder = ConditionalTextDecoder(config.decoder).to(device)
    predictor = LatentPredictor(config.predictor).to(device)

    # Build PyTorch datasets
    pair_dataset = GoodViewPairDataset(
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        generator=hf_loader,
        max_seq_len=config.encoder.max_seq_len,
        apply_augmentation=True,
    )

    seq_dataset = SequentialReasoningDataset(
        tokenizer=tokenizer,
        num_samples=args.num_samples // 2,
        generator=hf_loader,
        max_steps=config.predictor.max_steps,
        max_seq_len=config.encoder.max_seq_len,
    )

    training_logs = {}

    enc_ckpt_path = os.path.join(args.save_dir, "encoder.pt")
    dec_ckpt_path = os.path.join(args.save_dir, "decoder.pt")
    pred_ckpt_path = os.path.join(args.save_dir, "predictor.pt")

    # 3. Stage 1: Encoder with SIGReg
    if os.path.exists(enc_ckpt_path):
        print(f"\n[+] Loading pre-trained Stage 1 Encoder checkpoint from {enc_ckpt_path}")
        enc_ckpt = torch.load(enc_ckpt_path, map_location=device, weights_only=True)
        encoder.load_state_dict(enc_ckpt["encoder"])
        training_logs["stage1_encoder"] = enc_ckpt.get("history", {})
        print(f"[+] Stage 1 loaded. Final Total Loss: {training_logs['stage1_encoder']['total_loss'][-1]:.4f}")
    else:
        print("\n" + "=" * 60)
        print(" STAGE 1: TRAINING GOOD-VIEW JEPA ENCODER ON TAPACO")
        print("=" * 60)
        t_stage1 = time.time()
        enc_hist = train_jepa_encoder(
            encoder=encoder,
            dataset=pair_dataset,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=args.save_dir,
            verbose=True,
        )
        training_logs["stage1_encoder"] = enc_hist
        print(f"[+] Stage 1 finished in {time.time() - t_stage1:.2f}s. Final Total Loss: {enc_hist['total_loss'][-1]:.4f}")

    # 4. Stage 2: Decoder
    if os.path.exists(dec_ckpt_path):
        print(f"\n[+] Loading pre-trained Stage 2 Decoder checkpoint from {dec_ckpt_path}")
        dec_ckpt = torch.load(dec_ckpt_path, map_location=device, weights_only=True)
        decoder.load_state_dict(dec_ckpt["decoder"])
        training_logs["stage2_decoder"] = dec_ckpt.get("history", {})
        print(f"[+] Stage 2 loaded. Final CE Loss: {training_logs['stage2_decoder']['ce_loss'][-1]:.4f}")
    else:
        print("\n" + "=" * 60)
        print(" STAGE 2: TRAINING CONDITIONAL DECODER ON TAPACO")
        print("=" * 60)
        t_stage2 = time.time()
        dec_hist = train_conditional_decoder(
            decoder=decoder,
            encoder=encoder,
            dataset=pair_dataset,
            tokenizer=tokenizer,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=args.save_dir,
            verbose=True,
        )
        training_logs["stage2_decoder"] = dec_hist
        print(f"[+] Stage 2 finished in {time.time() - t_stage2:.2f}s. Final CE Loss: {dec_hist['ce_loss'][-1]:.4f}")

    # 5. Stage 3: Predictor
    print("\n" + "=" * 60)
    print(" STAGE 3: TRAINING LATENT PREDICTOR ON GSM8K TRAJECTORIES (GPU)")
    print("=" * 60)
    if device.startswith("cuda"):
        print(f"[GPU INFO] Using Device: {torch.cuda.get_device_name(0)}")
        print(f"[GPU INFO] VRAM Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")
    t_stage3 = time.time()
    pred_hist = train_latent_predictor(
        predictor=predictor,
        encoder=encoder,
        dataset=seq_dataset,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_dir,
        verbose=True,
    )
    training_logs["stage3_predictor"] = pred_hist
    print(f"[+] Stage 3 finished in {time.time() - t_stage3:.2f}s. Final Pred Loss: {pred_hist['pred_total_loss'][-1]:.4f}")

    # 6. Full Pipeline Assembly & Benchmarking
    pipeline = GoodViewTextJEPA(
        config=config,
        tokenizer=tokenizer,
        encoder=encoder,
        decoder=decoder,
        predictor=predictor,
    ).to(device)

    # Extract test pairs & trajectories for benchmark
    eval_pairs = []
    for cluster in list(hf_loader.clusters.values())[:300]:
        if len(cluster) >= 2:
            eval_pairs.append((cluster[0], cluster[1]))

    eval_trajectories = [steps for _, steps in hf_loader.trajectories[:150]]

    print("\n" + "=" * 60)
    print(" RUNNING BENCHMARK EVALUATION SUITE")
    print("=" * 60)
    benchmark_results = run_full_benchmark(
        pipeline=pipeline,
        positive_pairs=eval_pairs,
        trajectories=eval_trajectories,
        eval_sts=True,
    )

    # Save complete report
    final_report = {
        "device": device,
        "config": {
            "vocab_size": vocab_sz,
            "latent_dim": latent_dim,
            "d_model": d_model,
            "epochs": args.epochs,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
        },
        "training_history": training_logs,
        "benchmark_metrics": benchmark_results,
    }

    report_path = os.path.join(args.save_dir, args.output_report)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[+] Full benchmark report saved to {report_path}")

    # Generate and save loss progression graph
    plot_path = os.path.join(args.save_dir, "loss_progression.png")
    plot_loss_progression(training_logs, plot_path)

    # Demonstration of Semantic Invariance & Reasoning
    print("\n" + "=" * 60)
    print(" LIVE DEMONSTRATION WITH TRAINED REAL-WORLD SYSTEM")
    print("=" * 60)
    p1 = "I ate the cheese."
    p2 = "I was eating some cheese."
    diff_p = "The airplane flies across the continent."
    sim_same = pipeline.compare_semantic_invariance(p1, p2)
    sim_diff = pipeline.compare_semantic_invariance(p1, diff_p)
    print(f"Prompt A: '{p1}' vs Prompt B: '{p2}'")
    print(f"  -> Cosine Sim (Same meaning): {sim_same['cosine_similarity']:.4f}, L2 Dist: {sim_same['l2_distance']:.4f}")
    print(f"Prompt A: '{p1}' vs Prompt C: '{diff_p}'")
    print(f"  -> Cosine Sim (Different):    {sim_diff['cosine_similarity']:.4f}, L2 Dist: {sim_diff['l2_distance']:.4f}")

    gsm_prompt = "Weng earns $12 an hour for babysitting."
    print(f"\nRollout Reasoning for Prompt: '{gsm_prompt}'")
    rollout = pipeline.run_inference(gsm_prompt, reasoning_steps=2, max_tokens_per_step=16)
    print("  Predicted Latent Steps -> Rendered Tokens:")
    for idx, s in enumerate(rollout["rendered_steps"], 1):
        print(f"    Step {idx}: {s}")


if __name__ == "__main__":
    main()

