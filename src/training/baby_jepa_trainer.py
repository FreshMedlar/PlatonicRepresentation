"""
Three-Arm Training Loop for Baby-JEPA:
- active: ensemble-disagreement action selection
- active_spectral: ensemble disagreement + underrepresented-eigendirection proxy
- random: uniform random actions over matched interaction budget
- passive: offline replay of fixed low-diversity/confounded trajectories (no closed loop)
"""

from collections import deque
from dataclasses import asdict
import os
from typing import Dict, List, Optional, Tuple
import torch

from configs.baby_config import ACTIONS, BabyJEPAConfig
from src.env.tabletop_world import TabletopWorld
from src.evaluation.baby_benchmark import (
    FrozenDINOEncoder,
    FrozenMiniLMEncoder,
    build_eval_datasets,
    evaluate_baby_jepa,
)
from src.models.baby_jepa import BabyJEPASystem


class TrajectoryReplayBuffer:
    """Stores episode trajectories (images, actions, word_ids, has_word) for replay."""

    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add_batch(
        self,
        images_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        word_ids_seq: torch.Tensor,
        has_word_seq: torch.Tensor,
    ):
        B = images_seq.size(0)
        imgs_cpu = images_seq.detach().cpu()
        acts_cpu = actions_seq.detach().cpu()
        wids_cpu = word_ids_seq.detach().cpu()
        hw_cpu = has_word_seq.detach().cpu()
        for i in range(B):
            self.buffer.append((imgs_cpu[i], acts_cpu[i], wids_cpu[i], hw_cpu[i]))

    def sample(
        self, batch_size: int, device: torch.device, rng: torch.Generator
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(self.buffer)
        idx = torch.randint(0, n, (min(batch_size, n),), generator=rng).tolist()
        imgs = torch.stack([self.buffer[i][0] for i in idx], dim=0).to(device)
        acts = torch.stack([self.buffer[i][1] for i in idx], dim=0).to(device)
        wids = torch.stack([self.buffer[i][2] for i in idx], dim=0).to(device)
        hw = torch.stack([self.buffer[i][3] for i in idx], dim=0).to(device)
        return imgs, acts, wids, hw

    def __len__(self) -> int:
        return len(self.buffer)


def _select_passive_actions(batch_size: int, num_actions: int, device: torch.device, rng: torch.Generator) -> torch.Tensor:
    """
    Low-diversity passive observer policy: mostly small camera jitter / zoom
    (orbit_left=3, orbit_right=4, zoom_in=9, zoom_out=10), rarely turning/flipping object
    or shifting light, modeling a passive trajectory logger without active disambiguation.
    """
    probs = torch.tensor(
        [0.02, 0.02, 0.01, 0.28, 0.28, 0.04, 0.04, 0.01, 0.01, 0.145, 0.145],
        dtype=torch.float32,
    )
    probs = probs[:num_actions]
    probs = probs / probs.sum()
    return torch.multinomial(probs, batch_size, replacement=True, generator=rng).to(device)


@torch.no_grad()
def collect_passive_dataset(
    env: TabletopWorld,
    num_episodes: int,
    episode_length: int,
    seed: int = 123,
) -> TrajectoryReplayBuffer:
    """Pre-collects a fixed offline trajectory buffer for the passive arm."""
    buf = TrajectoryReplayBuffer(capacity=max(num_episodes * env.batch_size, 64))
    rng = torch.Generator(device="cpu").manual_seed(seed)
    num_actions = len(ACTIONS)

    for _ in range(num_episodes):
        obs = env.reset()
        imgs = [obs["images"]]
        wids = [obs["word_id"]]
        hws = [obs["has_word"]]
        acts = []
        for _ in range(episode_length - 1):
            a = _select_passive_actions(env.batch_size, num_actions, env.device, rng)
            acts.append(a)
            obs = env.step(a)
            imgs.append(obs["images"])
            wids.append(obs["word_id"])
            hws.append(obs["has_word"])

        buf.add_batch(
            torch.stack(imgs, dim=1),
            torch.stack(acts, dim=1),
            torch.stack(wids, dim=1),
            torch.stack(hws, dim=1),
        )
    return buf


class BabyJEPATrainer:
    """Trains and evaluates one arm of the Baby-JEPA experiment."""

    def __init__(
        self,
        config: BabyJEPAConfig,
        dino_encoder: Optional[FrozenDINOEncoder] = None,
        minilm_encoder: Optional[FrozenMiniLMEncoder] = None,
        eval_splits: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        dino_eval_feats: Optional[torch.Tensor] = None,
    ):
        self.config = config
        self.device = torch.device(
            config.training.device if torch.cuda.is_available() else "cpu"
        )
        torch.manual_seed(config.training.seed)

        self.system = BabyJEPASystem(config).to(self.device)
        self.env = TabletopWorld(
            config.env,
            batch_size=config.training.batch_envs,
            device=str(self.device),
            seed=config.training.seed,
        )
        self.replay = TrajectoryReplayBuffer(capacity=config.training.replay_capacity)
        self.rng = torch.Generator(device="cpu").manual_seed(config.training.seed + 7)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.system.vision_encoder.parameters(), "lr": config.training.lr_vision},
                {"params": self.system.predictor_ensemble.parameters(), "lr": config.training.lr_predictor},
                {"params": self.system.word_encoder.parameters(), "lr": config.training.lr_word},
                {"params": self.system.cross_modal_head.parameters(), "lr": config.training.lr_word},
            ],
            weight_decay=1e-4,
        )
        total_grad_steps = max(
            1,
            (config.training.total_episodes // config.training.batch_envs)
            * config.training.train_steps_per_episode,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_grad_steps, eta_min=1e-5
        )

        self.eval_splits = eval_splits or build_eval_datasets(
            config.env, device=str(self.device), seed=999
        )
        self.dino_encoder = dino_encoder
        self.minilm_encoder = minilm_encoder
        self.dino_eval_feats = dino_eval_feats
        if self.dino_eval_feats is None and self.dino_encoder is not None:
            with torch.no_grad():
                imgs = self.eval_splits["decorrelated_eval"]["images"]
                feats = [
                    self.dino_encoder(imgs[i : i + 32].to(self.device))
                    for i in range(0, imgs.size(0), 32)
                ]
                self.dino_eval_feats = torch.cat(feats, dim=0)

        self.recent_z: deque = deque(maxlen=256)
        self.action_counts = torch.zeros(len(ACTIONS), dtype=torch.long)

    @torch.no_grad()
    def _rollout_online_episode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rolls out one batched episode under active or random policy."""
        self.system.eval()
        arm = self.config.training.arm
        T = self.config.env.episode_length
        B = self.config.training.batch_envs
        num_actions = len(ACTIONS)

        obs = self.env.reset()
        imgs = [obs["images"]]
        wids = [obs["word_id"]]
        hws = [obs["has_word"]]
        acts = []

        _, _, z0_dyn = self.system.encode_images(obs["images"], return_all=True)
        z_hist = [z0_dyn]  # list of (B, D) in equivariant dynamics space
        a_hist: List[torch.Tensor] = []

        recent_z_tensor = (
            torch.cat(list(self.recent_z), dim=0).to(self.device)
            if len(self.recent_z) > 0
            else None
        )
        spectral_w = (
            self.config.training.spectral_bonus_weight if arm == "active_spectral" else 0.0
        )
        spectral_stats = (
            self.system.predictor_ensemble.compute_spectral_stats(recent_z_tensor)
            if (spectral_w > 0.0 and recent_z_tensor is not None and recent_z_tensor.size(0) >= 8)
            else None
        )

        for t in range(T - 1):
            if arm == "random":
                a = torch.randint(0, num_actions, (B,), generator=self.rng).to(self.device)
            else:
                z_ctx = torch.stack(z_hist[-self.config.training.context_window :], dim=1)
                a_ctx = (
                    torch.stack(a_hist[-(self.config.training.context_window - 1) :], dim=1)
                    if len(a_hist) > 0 and self.config.training.context_window > 1
                    else None
                )
                visited_z = torch.stack(z_hist, dim=1)
                scores, _ = self.system.predictor_ensemble.score_candidate_actions(
                    z_ctx=z_ctx,
                    a_ctx=a_ctx,
                    recent_z=recent_z_tensor,
                    spectral_bonus_weight=spectral_w,
                    spectral_stats=spectral_stats,
                    visited_z=visited_z,
                )
                greedy_a = scores.argmax(dim=-1)
                rand_a = torch.randint(0, num_actions, (B,), generator=self.rng).to(self.device)
                explore = (torch.rand((B,), generator=self.rng) < self.config.training.exploration_epsilon).to(self.device)
                a = torch.where(explore, rand_a, greedy_a)

            for aid in a.cpu().tolist():
                self.action_counts[aid] += 1

            obs = self.env.step(a)
            imgs.append(obs["images"])
            wids.append(obs["word_id"])
            hws.append(obs["has_word"])
            acts.append(a)
            a_hist.append(a)
            _, _, z_next = self.system.encode_images(obs["images"], return_all=True)
            z_hist.append(z_next)

        z_ep = torch.stack(z_hist, dim=1).reshape(-1, self.config.vision.latent_dim).cpu()
        self.recent_z.append(z_ep[:64])

        return (
            torch.stack(imgs, dim=1),
            torch.stack(acts, dim=1),
            torch.stack(wids, dim=1),
            torch.stack(hws, dim=1),
        )

    def _train_step(self, source_buffer: TrajectoryReplayBuffer) -> Dict[str, float]:
        self.system.train()
        imgs, acts, wids, hws = source_buffer.sample(
            self.config.training.batch_size, self.device, self.rng
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.system.compute_training_loss(imgs, acts, wids, hws)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.system.parameters(), self.config.training.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        return metrics

    def save_checkpoint(
        self,
        path: str,
        metrics: Optional[Dict[str, float]] = None,
        action_counts: Optional[Dict[str, int]] = None,
    ):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "arm": self.config.training.arm,
                "config": asdict(self.config),
                "model_state_dict": self.system.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics or {},
                "action_counts": action_counts or {},
                "word_to_id": self.env.word_to_id,
            },
            path,
        )

    def run(self, checkpoint_dir: Optional[str] = None, verbose: bool = False) -> Dict[str, object]:
        arm = self.config.training.arm
        num_batches = max(1, self.config.training.total_episodes // self.config.training.batch_envs)
        eval_interval = max(1, self.config.training.eval_every_episodes // self.config.training.batch_envs)

        passive_buffer = None
        if arm == "passive":
            passive_buffer = collect_passive_dataset(
                self.env,
                num_episodes=num_batches,
                episode_length=self.config.env.episode_length,
                seed=self.config.training.seed + 101,
            )

        history: List[Dict[str, float]] = []
        total_interactions = 0
        best_score = -1.0

        # Step 0 baseline evaluation
        init_eval = evaluate_baby_jepa(
            self.system,
            self.eval_splits,
            dino_encoder=self.dino_encoder,
            minilm_encoder=self.minilm_encoder,
            dino_eval_feats=self.dino_eval_feats,
        )
        init_eval["interactions"] = 0
        init_eval["episodes"] = 0
        history.append(init_eval)

        for batch_idx in range(1, num_batches + 1):
            if arm == "passive":
                src = passive_buffer
            else:
                imgs, acts, wids, hws = self._rollout_online_episode()
                self.replay.add_batch(imgs, acts, wids, hws)
                src = self.replay

            total_interactions += self.config.training.batch_envs * self.config.env.episode_length

            last_train_metrics: Dict[str, float] = {}
            for _ in range(self.config.training.train_steps_per_episode):
                last_train_metrics = self._train_step(src)

            if batch_idx % eval_interval == 0 or batch_idx == num_batches:
                ev = evaluate_baby_jepa(
                    self.system,
                    self.eval_splits,
                    dino_encoder=self.dino_encoder,
                    minilm_encoder=self.minilm_encoder,
                    dino_eval_feats=self.dino_eval_feats,
                )
                ev["interactions"] = total_interactions
                ev["episodes"] = batch_idx * self.config.training.batch_envs
                ev.update({f"train_{k}": v for k, v in last_train_metrics.items()})
                history.append(ev)

                if verbose:
                    print(
                        f"  [{arm}] ep={ev['episodes']:4d} | inter={total_interactions:5d} | "
                        f"shape_c2d={ev['shape_probe_from_confounded']:.3f} | "
                        f"proc_sh={ev['procrustes_shape_naming_acc']:.3f} | "
                        f"cka={ev.get('dino_cka', 0.0):.3f} | fH={ev['f_H']:.3f} | "
                        f"erankZ={ev['erank_Z']:.1f} | wmse={last_train_metrics.get('temporal_wmse', 0.0):.3f}",
                        flush=True,
                    )

                score = ev["procrustes_shape_naming_acc"] + ev.get("dino_cka", 0.0)
                if checkpoint_dir is not None and score > best_score:
                    best_score = score
                    action_hist_now = {
                        ACTIONS[i]: int(self.action_counts[i].item()) for i in range(len(ACTIONS))
                    }
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, f"best_{arm}.pt"),
                        metrics=ev,
                        action_counts=action_hist_now,
                    )

        action_hist = {ACTIONS[i]: int(self.action_counts[i].item()) for i in range(len(ACTIONS))}
        if checkpoint_dir is not None:
            self.save_checkpoint(
                os.path.join(checkpoint_dir, f"checkpoint_{arm}.pt"),
                metrics=history[-1],
                action_counts=action_hist,
            )
        return {
            "arm": arm,
            "history": history,
            "final": history[-1],
            "action_counts": action_hist,
        }
