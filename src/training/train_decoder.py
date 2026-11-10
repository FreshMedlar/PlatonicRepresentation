"""
Stage 2 Training: Conditional Text Decoder d_psi.
Trains causal decoder conditioned on soft prompt prefix projected from state vector e.
"""

from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import SystemConfig
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.parallel_dataset import GoodViewPairDataset
from src.models.encoder import TextJEPAEncoder
from src.models.decoder import ConditionalTextDecoder
from src.training.trainer_utils import create_optimizer, get_cosine_schedule_with_warmup, save_checkpoint


def train_conditional_decoder(
    decoder: ConditionalTextDecoder,
    encoder: TextJEPAEncoder,
    dataset: GoodViewPairDataset,
    tokenizer: BPETokenizer,
    config: SystemConfig,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 5e-4,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Trains conditional decoder while keeping the JEPA encoder frozen.
    """
    device = torch.device(config.device)
    decoder.to(device)
    encoder.to(device)

    # Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    decoder.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    total_steps = epochs * len(dataloader)
    warmup_steps = max(1, int(0.1 * total_steps))

    optimizer = create_optimizer(decoder, lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = {"ce_loss": []}

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Stage 2: Training Conditional Decoder")

    for epoch in iterator:
        epoch_ce = 0.0
        steps = 0

        for batch in dataloader:
            # Reconstruct clean text passages from state vector e
            clean_ids = batch["clean1_ids"].to(device)
            clean_mask = batch["clean1_mask"].to(device)
            style_ids = batch["style1_id"].to(device)

            with torch.no_grad():
                e = encoder(clean_ids, attention_mask=clean_mask)

            # Next-token prediction:
            # Decoder inputs: tokens 0 to L-2
            # Targets: tokens 1 to L-1
            dec_inputs = clean_ids[:, :-1]
            dec_mask = clean_mask[:, :-1]
            targets = clean_ids[:, 1:]

            optimizer.zero_grad()

            logits = decoder(
                e=e,
                input_ids=dec_inputs,
                attention_mask=dec_mask,
                style_ids=style_ids,
                inject_noise=True,
            )

            # Flatten and compute cross-entropy, ignoring padding
            vocab_sz = config.decoder.vocab_size
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_sz),
                targets.reshape(-1),
                ignore_index=tokenizer.pad_id,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_ce += loss.item()
            steps += 1

        if steps > 0:
            history["ce_loss"].append(epoch_ce / steps)

    if save_path:
        save_checkpoint({"decoder": decoder.state_dict(), "history": history}, save_dir=save_path, filename="decoder.pt")

    return history
