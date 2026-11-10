"""
Command-line Interface for Good-View Text JEPA Benchmarks.
Runs the evaluation suite on trained checkpoints or real/synthetic datasets.
"""

import argparse
import os
import sys
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
from src.models.encoder import TextJEPAEncoder
from src.models.decoder import ConditionalTextDecoder
from src.models.predictor import LatentPredictor
from src.models.pipeline import GoodViewTextJEPA
from src.data.view_generator import GoodViewGenerator
from src.evaluation.benchmark import run_full_benchmark


def main():
    parser = argparse.ArgumentParser(description="Benchmark Good-View Text JEPA System")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory with saved models")
    parser.add_argument("--dataset", type=str, choices=["synthetic", "real"], default="real", help="Dataset to evaluate on")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--no-sts", action="store_true", help="Skip online STS-B benchmark")
    parser.add_argument("--max-samples", type=int, default=200, help="Max test samples per task")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[*] Benchmarking Good-View Text JEPA using device: {device}")

    tok_path = os.path.join(args.checkpoint_dir, "tokenizer.json")
    enc_path = os.path.join(args.checkpoint_dir, "encoder.pt")
    dec_path = os.path.join(args.checkpoint_dir, "decoder.pt")
    pred_path = os.path.join(args.checkpoint_dir, "predictor.pt")

    if not os.path.exists(tok_path) or not os.path.exists(enc_path):
        print(f"[-] Checkpoint files not found in {args.checkpoint_dir}. Train the system first.")
        sys.exit(1)

    tokenizer = BPETokenizer.load(tok_path)
    vocab_sz = tokenizer.vocab_size

    # Inspect checkpoint latent dimension
    ckpt_enc = torch.load(enc_path, map_location="cpu", weights_only=True)
    enc_weights = ckpt_enc.get("encoder", ckpt_enc)
    # Projector out features: projector.2.weight shape: (latent_dim, d_model)
    proj_key = [k for k in enc_weights.keys() if "projector" in k and "weight" in k][-1]
    latent_dim = enc_weights[proj_key].shape[0]
    d_model = enc_weights[proj_key].shape[1]

    print(f"[+] Loaded model architecture: vocab_size={vocab_sz}, latent_dim={latent_dim}, d_model={d_model}")

    config = SystemConfig(
        encoder=EncoderConfig(vocab_size=vocab_sz, d_model=d_model, latent_dim=latent_dim, max_seq_len=64),
        sigreg=SIGRegConfig(),
        decoder=DecoderConfig(vocab_size=vocab_sz, d_model=d_model, latent_dim=latent_dim, max_seq_len=64),
        predictor=PredictorConfig(latent_dim=latent_dim, d_model=d_model, max_steps=8),
        device=device,
    )

    encoder = TextJEPAEncoder(config.encoder)
    encoder.load_state_dict(enc_weights)

    decoder = ConditionalTextDecoder(config.decoder)
    if os.path.exists(dec_path):
        ckpt_dec = torch.load(dec_path, map_location="cpu", weights_only=True)
        decoder.load_state_dict(ckpt_dec.get("decoder", ckpt_dec))

    predictor = LatentPredictor(config.predictor)
    if os.path.exists(pred_path):
        ckpt_pred = torch.load(pred_path, map_location="cpu", weights_only=True)
        predictor.load_state_dict(ckpt_pred.get("predictor", ckpt_pred))

    pipeline = GoodViewTextJEPA(
        config=config,
        tokenizer=tokenizer,
        encoder=encoder,
        decoder=decoder,
        predictor=predictor,
    )
    pipeline.to(device)

    # Prepare evaluation data
    if args.dataset == "real":
        from src.data.hf_dataset_loader import HFGoodViewLoader
        loader = HFGoodViewLoader(max_clusters=args.max_samples, max_trajectories=args.max_samples)
        pairs = []
        for cluster in loader.clusters.values():
            if len(cluster) >= 2:
                pairs.append((cluster[0], cluster[1]))
        trajectories = [steps for _, steps in loader.trajectories]
    else:
        gen = GoodViewGenerator()
        pairs = []
        for item in gen.TEMPLATES:
            styles = [s for s in gen.STYLES if s in item]
            if len(styles) >= 2:
                pairs.append((item[styles[0]], item[styles[1]]))
        trajectories = [
            [step["standard"] for step in traj["steps"]] for traj in gen.TRAJECTORY_TEMPLATES
        ]

    # Run Benchmark
    results = run_full_benchmark(
        pipeline=pipeline,
        positive_pairs=pairs,
        trajectories=trajectories,
        eval_sts=not args.no_sts,
    )
    return results


if __name__ == "__main__":
    main()

