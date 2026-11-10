"""
Configuration dataclasses for Good-View Text JEPA.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TokenizerConfig:
    vocab_size: int = 1000
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    mask_token: str = "[MASK]"
    special_tokens: List[str] = field(
        default_factory=lambda: ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
    )


@dataclass
class EncoderConfig:
    vocab_size: int = 1000
    max_seq_len: int = 64
    d_model: int = 192
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 512
    latent_dim: int = 128  # Dimension k of Platonic semantic space
    dropout: float = 0.1


@dataclass
class SIGRegConfig:
    """Sketched Isotropic Gaussian Regularization parameters."""
    lam: float = 1.0  # Quadrature bandwidth scale (-4*lam to 4*lam)
    num_projections: int = 64  # Number of 1D Cramér-Wold random projections
    num_integration_steps: int = 17  # Quadrature knots
    reg_weight: float = 0.05  # lambda balancing invariance loss and SIGReg


@dataclass
class DecoderConfig:
    vocab_size: int = 1000
    max_seq_len: int = 64
    d_model: int = 192
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 512
    latent_dim: int = 128
    num_soft_tokens: int = 4  # Number of prefix virtual soft prompt tokens M
    noise_scale: float = 0.05  # sigma for perturbation e_tilde = e + N(0, sigma^2)
    num_styles: int = 4  # Number of style categories (e.g., standard, formal, casual, concise)
    style_dim: int = 32
    dropout: float = 0.1


@dataclass
class PredictorConfig:
    latent_dim: int = 128  # Vector dimension k
    max_steps: int = 16  # Maximum reasoning sequence length T
    d_model: int = 192
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    reg_weight: float = 0.05  # lambda for SIGReg on predicted states


@dataclass
class SystemConfig:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    sigreg: SIGRegConfig = field(default_factory=SIGRegConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    seed: int = 42
    device: str = "cpu"
