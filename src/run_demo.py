"""
Interactive Demonstration of the Good-View Text JEPA System.
Showcases:
1. Stage 1-3 quick self-contained CPU training (or checkpoint loading)
2. Semantic Invariance in Platonic embedding space R^k
3. End-to-end Latent Reasoning Rollout (Encoder -> Latent Predictor -> Conditional Decoder)
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from configs.config import (
    SystemConfig,
    EncoderConfig,
    SIGRegConfig,
    DecoderConfig,
    PredictorConfig,
)
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.view_generator import GoodViewGenerator
from src.data.parallel_dataset import GoodViewPairDataset
from src.data.sequential_dataset import SequentialReasoningDataset
from src.models.encoder import TextJEPAEncoder
from src.models.decoder import ConditionalTextDecoder
from src.models.predictor import LatentPredictor
from src.models.pipeline import GoodViewTextJEPA
from src.training.train_encoder import train_jepa_encoder
from src.training.train_decoder import train_conditional_decoder
from src.training.train_predictor import train_latent_predictor


def build_and_train_system(
    save_dir: str = "checkpoints",
    epochs: int = 5,
    num_samples: int = 256,
    batch_size: int = 16,
    force_train: bool = False,
) -> GoodViewTextJEPA:
    os.makedirs(save_dir, exist_ok=True)
    tok_path = os.path.join(save_dir, "tokenizer.json")
    enc_path = os.path.join(save_dir, "encoder.pt")
    dec_path = os.path.join(save_dir, "decoder.pt")
    pred_path = os.path.join(save_dir, "predictor.pt")

    generator = GoodViewGenerator(seed=42)
    corpus = generator.get_all_texts()

    # 1. Tokenizer
    if os.path.exists(tok_path) and not force_train:
        print(f"[+] Loading tokenizer from {tok_path}")
        tokenizer = BPETokenizer.load(tok_path)
    else:
        print("[*] Training BPE Tokenizer on proposition corpus...")
        tokenizer = BPETokenizer()
        tokenizer.train(corpus, target_vocab_size=250)
        tokenizer.save(tok_path)

    vocab_sz = tokenizer.vocab_size
    latent_dim = 64
    d_model = 128

    config = SystemConfig(
        encoder=EncoderConfig(
            vocab_size=vocab_sz,
            max_seq_len=36,
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
            max_seq_len=36,
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
        device="cpu",
    )

    encoder = TextJEPAEncoder(config.encoder)
    decoder = ConditionalTextDecoder(config.decoder)
    predictor = LatentPredictor(config.predictor)

    pair_dataset = GoodViewPairDataset(
        tokenizer=tokenizer,
        num_samples=num_samples,
        generator=generator,
        max_seq_len=config.encoder.max_seq_len,
    )
    seq_dataset = SequentialReasoningDataset(
        tokenizer=tokenizer,
        num_samples=num_samples,
        generator=generator,
        max_steps=config.predictor.max_steps,
        max_seq_len=config.encoder.max_seq_len,
    )

    # 2. Stage 1: Encoder
    if os.path.exists(enc_path) and not force_train:
        print(f"[+] Loading trained encoder from {enc_path}")
        ckpt = torch.load(enc_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
    else:
        print("\n=== Stage 1: Training Good-View Text JEPA Encoder with SIGReg ===")
        train_jepa_encoder(
            encoder=encoder,
            dataset=pair_dataset,
            config=config,
            epochs=epochs,
            batch_size=batch_size,
            save_path=save_dir,
            verbose=True,
        )

    # 3. Stage 2: Decoder
    if os.path.exists(dec_path) and not force_train:
        print(f"[+] Loading trained decoder from {dec_path}")
        ckpt = torch.load(dec_path, map_location="cpu", weights_only=True)
        decoder.load_state_dict(ckpt["decoder"])
    else:
        print("\n=== Stage 2: Training Conditional Text Decoder ===")
        train_conditional_decoder(
            decoder=decoder,
            encoder=encoder,
            dataset=pair_dataset,
            tokenizer=tokenizer,
            config=config,
            epochs=epochs,
            batch_size=batch_size,
            save_path=save_dir,
            verbose=True,
        )

    # 4. Stage 3: Predictor
    if os.path.exists(pred_path) and not force_train:
        print(f"[+] Loading trained predictor from {pred_path}")
        ckpt = torch.load(pred_path, map_location="cpu", weights_only=True)
        predictor.load_state_dict(ckpt["predictor"])
    else:
        print("\n=== Stage 3: Training Latent State-Transition Predictor ===")
        train_latent_predictor(
            predictor=predictor,
            encoder=encoder,
            dataset=seq_dataset,
            config=config,
            epochs=epochs,
            batch_size=batch_size,
            save_path=save_dir,
            verbose=True,
        )

    pipeline = GoodViewTextJEPA(
        config=config,
        tokenizer=tokenizer,
        encoder=encoder,
        decoder=decoder,
        predictor=predictor,
    )
    return pipeline


def demonstrate_invariance(pipeline: GoodViewTextJEPA):
    print("\n" + "=" * 60)
    print(" DEMO 1: SEMANTIC INVARIANCE IN PLATONIC SPACE R^k")
    print("=" * 60)

    pairs = [
        (
            "Water freezes into solid ice at zero degrees Celsius.",
            "The phase transition of liquid water to solid ice occurs at approximately zero degrees Celsius.",
            "Same concept: Water Freezing (Standard vs Formal)",
        ),
        (
            "Water freezes into solid ice at zero degrees Celsius.",
            "Whenever water hits zero degrees Celsius it turns into solid ice.",
            "Same concept: Water Freezing (Standard vs Casual)",
        ),
        (
            "Water freezes into solid ice at zero degrees Celsius.",
            "Light travels through a vacuum at three hundred thousand kilometers per second.",
            "Different concept: Water Freezing vs Speed of Light",
        ),
    ]

    for t1, t2, desc in pairs:
        metrics = pipeline.compare_semantic_invariance(t1, t2)
        print(f"\n[{desc}]")
        print(f"  Text 1: \"{t1}\"")
        print(f"  Text 2: \"{t2}\"")
        print(f"  -> Cosine Similarity: {metrics['cosine_similarity']:.4f}")
        print(f"  -> L2 Distance:       {metrics['l2_distance']:.4f}")


def demonstrate_latent_reasoning(pipeline: GoodViewTextJEPA, prompt: str, steps: int = 3):
    print("\n" + "=" * 60)
    print(" DEMO 2: LATENT REASONING ROLLOUT (Prompt -> R^k Rollout -> Render)")
    print("=" * 60)
    print(f"Initial Prompt: \"{prompt}\"")
    print(f"Performing {steps} latent thought transitions in R^k...")

    res = pipeline.run_inference(prompt=prompt, reasoning_steps=steps, style_id=0, max_tokens_per_step=20)

    traj = res["latent_trajectory"]
    print(f"Latent trajectory shape: {traj.shape} (T={traj.size(1)}, dim={traj.size(2)})")

    print("\nRendered Semantic Steps:")
    for i, step_str in enumerate(res["rendered_steps"], 1):
        print(f"  [Step {i}] {step_str}")

    print(f"\nFull Output:\n  {res['full_response']}")


def main():
    parser = argparse.ArgumentParser(description="Good-View Text JEPA Demo")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per stage")
    parser.add_argument("--samples", type=int, default=128, help="Number of training samples")
    parser.add_argument("--prompt", type=str, default="Thermal energy is supplied to the liquid container.", help="Prompt for reasoning rollout")
    parser.add_argument("--steps", type=int, default=3, help="Number of latent reasoning steps")
    parser.add_argument("--force-train", action="store_true", help="Force retraining of all stages")
    args = parser.parse_args()

    pipeline = build_and_train_system(
        epochs=args.epochs,
        num_samples=args.samples,
        force_train=args.force_train,
    )

    demonstrate_invariance(pipeline)
    demonstrate_latent_reasoning(pipeline, prompt=args.prompt, steps=args.steps)


if __name__ == "__main__":
    main()
