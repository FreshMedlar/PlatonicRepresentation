"""
Configuration dataclasses for Baby-JEPA: active, grounded, cross-modal JEPA PoC.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


SHAPES: Tuple[str, ...] = ("sphere", "cube", "cylinder", "cone", "capsule")
COLORS: Tuple[str, ...] = ("red", "green", "blue", "yellow")
TEXTURES: Tuple[str, ...] = ("solid", "striped", "checkered")

ACTIONS: Tuple[str, ...] = (
    "turn_left",
    "turn_right",
    "flip_pitch",
    "orbit_left",
    "orbit_right",
    "orbit_up",
    "orbit_down",
    "shift_light_angle",
    "shift_light_color",
    "zoom_in",
    "zoom_out",
)


@dataclass
class TabletopEnvConfig:
    image_size: int = 64
    num_shapes: int = 4          # default: sphere, cube, cylinder, cone
    num_colors: int = 4          # red, green, blue, yellow
    num_textures: int = 3        # solid, striped, checkered
    episode_length: int = 16
    teacher_word_prob: float = 0.22
    confound_prob: float = 0.85  # initial shape<->color & tinted spotlight confounding
    biased_initial_view: bool = True
    fov_deg: float = 45.0
    curriculum_phase2_shapes: int = 5  # adds capsule in curriculum expansion


@dataclass
class BabyVisionConfig:
    image_size: int = 64
    d_model: int = 256
    latent_dim: int = 128
    dropout: float = 0.0


@dataclass
class BabyPredictorConfig:
    latent_dim: int = 128
    num_actions: int = len(ACTIONS)
    action_dim: int = 32
    d_model: int = 192
    num_layers: int = 3
    num_heads: int = 4
    d_ff: int = 512
    max_steps: int = 16
    ensemble_size: int = 5
    dropout: float = 0.1


@dataclass
class BabyWordConfig:
    vocab_size: int = 1 + len(SHAPES) + len(COLORS) + len(TEXTURES)
    d_word: int = 128
    latent_dim: int = 128


@dataclass
class BabyTrainingConfig:
    arm: str = "active"  # "active", "active_spectral", "random", "passive"
    total_episodes: int = 240
    batch_envs: int = 16
    context_window: int = 8
    lr_vision: float = 3e-4
    lr_predictor: float = 5e-4
    lr_word: float = 3e-4
    grad_clip: float = 1.0
    sigreg_weight: float = 0.05
    sadl_weight: float = 0.01
    cross_modal_weight: float = 0.5
    word_sigreg_weight: float = 0.05
    whitened_loss_weight: float = 0.5
    exploration_epsilon: float = 0.05
    spectral_bonus_weight: float = 0.25
    replay_capacity: int = 1024
    train_steps_per_episode: int = 2
    batch_size: int = 32
    eval_every_episodes: int = 40
    seed: int = 42
    device: str = "cuda"


@dataclass
class BabyJEPAConfig:
    env: TabletopEnvConfig = field(default_factory=TabletopEnvConfig)
    vision: BabyVisionConfig = field(default_factory=BabyVisionConfig)
    predictor: BabyPredictorConfig = field(default_factory=BabyPredictorConfig)
    word: BabyWordConfig = field(default_factory=BabyWordConfig)
    training: BabyTrainingConfig = field(default_factory=BabyTrainingConfig)
