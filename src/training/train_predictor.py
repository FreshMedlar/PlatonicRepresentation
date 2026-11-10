"""
Stage 3 Training: Latent State-Transition Predictor P_phi.
Trains causal vector-space Transformer to predict next Platonic state vectors.
"""

from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import SystemConfig
from src.data.sequential_dataset import SequentialReasoningDataset
from src.models.encoder import TextJEPAEncoder
from src.models.predictor import LatentPredictor
from src.training.trainer_utils import create_optimizer, get_cosine_schedule_with_warmup, save_checkpoint


def train_latent_predictor(
    predictor: LatentPredictor,
    encoder: TextJEPAEncoder,
    dataset: SequentialReasoningDataset,
    config: SystemConfig,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 5e-4,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Trains latent predictor on sequential Platonic state representations.
    """
    device = torch.device(config.device)
    predictor.to(device)
    encoder.to(device)

    # Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    predictor.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    total_steps = epochs * len(dataloader)
    warmup_steps = max(1, int(0.1 * total_steps))

    optimizer = create_optimizer(predictor, lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = {"pred_total_loss": [], "pred_mse": [], "pred_reg": []}

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Stage 3: Training Latent Predictor")

    for epoch in iterator:
        epoch_tot, epoch_mse, epoch_reg = 0.0, 0.0, 0.0
        steps = 0

        for batch in dataloader:
            step_ids = batch["step_input_ids"].to(device)  # (B, T, L)
            step_masks = batch["step_attention_mask"].to(device)  # (B, T, L)
            step_valid = batch["step_valid_mask"].to(device)  # (B, T)

            B, T, L = step_ids.shape
            D = config.encoder.latent_dim

            # Encode all steps using the frozen JEPA encoder
            flat_ids = step_ids.view(B * T, L)
            flat_masks = step_masks.view(B * T, L)

            with torch.no_grad():
                flat_z = encoder(flat_ids, attention_mask=flat_masks)  # (B*T, D)

            z_seq = flat_z.view(B, T, D)  # (B, T, D)

            optimizer.zero_grad()
            tot_loss, mse_loss, reg_loss = predictor.compute_loss(z_seq, valid_mask=step_valid)

            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_tot += tot_loss.item()
            epoch_mse += mse_loss.item()
            epoch_reg += reg_loss.item()
            steps += 1

        if steps > 0:
            history["pred_total_loss"].append(epoch_tot / steps)
            history["pred_mse"].append(epoch_mse / steps)
            history["pred_reg"].append(epoch_reg / steps)

    if save_path:
        save_checkpoint({"predictor": predictor.state_dict(), "history": history}, save_dir=save_path, filename="predictor.pt")

    return history
