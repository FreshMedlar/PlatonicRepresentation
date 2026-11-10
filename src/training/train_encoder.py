"""
Stage 1 Training: Good-View Text JEPA Encoder f_theta.
Trains purely in embedding space using Invariance MSE Loss + LeJEPA SIGReg.
"""

from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import SystemConfig
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.parallel_dataset import GoodViewPairDataset
from src.data.view_generator import GoodViewGenerator
from src.models.encoder import TextJEPAEncoder
from src.models.sigreg import SIGRegLoss, compute_lejepa_loss
from src.training.trainer_utils import create_optimizer, get_cosine_schedule_with_warmup, save_checkpoint


def train_jepa_encoder(
    encoder: TextJEPAEncoder,
    dataset: GoodViewPairDataset,
    config: SystemConfig,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 5e-4,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Trains the bidirectional encoder with positive good-view pairs and SIGReg.
    """
    device = torch.device(config.device)
    encoder.to(device)
    encoder.train()

    sigreg = SIGRegLoss(
        latent_dim=config.encoder.latent_dim,
        lam=config.sigreg.lam,
        num_projections=config.sigreg.num_projections,
        num_integration_steps=config.sigreg.num_integration_steps,
    ).to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    total_steps = epochs * len(dataloader)
    warmup_steps = max(1, int(0.1 * total_steps))

    optimizer = create_optimizer(encoder, lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = {"total_loss": [], "inv_loss": [], "reg_loss": []}

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Stage 1: Training JEPA Encoder")

    for epoch in iterator:
        epoch_tot, epoch_inv, epoch_reg = 0.0, 0.0, 0.0
        steps = 0

        for batch in dataloader:
            v1_ids = batch["view1_ids"].to(device)
            v1_mask = batch["view1_mask"].to(device)
            v2_ids = batch["view2_ids"].to(device)
            v2_mask = batch["view2_mask"].to(device)

            optimizer.zero_grad()

            # Forward pass both views through encoder
            e1 = encoder(v1_ids, attention_mask=v1_mask)
            e2 = encoder(v2_ids, attention_mask=v2_mask)

            tot_loss, inv_loss, reg_loss = compute_lejepa_loss(
                e1, e2, sigreg, reg_weight=config.sigreg.reg_weight
            )

            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_tot += tot_loss.item()
            epoch_inv += inv_loss.item()
            epoch_reg += reg_loss.item()
            steps += 1

        if steps > 0:
            history["total_loss"].append(epoch_tot / steps)
            history["inv_loss"].append(epoch_inv / steps)
            history["reg_loss"].append(epoch_reg / steps)

    if save_path:
        save_checkpoint({"encoder": encoder.state_dict(), "history": history}, save_dir=save_path, filename="encoder.pt")

    return history
