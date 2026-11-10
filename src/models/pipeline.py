"""
Full Good-View Text JEPA System Pipeline.
Integrates:
  Encoder f_theta -> Latent Predictor P_phi -> Conditional Decoder d_psi
Providing prompt encoding, latent reasoning rollout, and text rendering.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import SystemConfig
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.models.encoder import TextJEPAEncoder
from src.models.decoder import ConditionalTextDecoder
from src.models.predictor import LatentPredictor


class GoodViewTextJEPA(nn.Module):
    """
    End-to-end Good-View Text JEPA System.
    """

    def __init__(
        self,
        config: SystemConfig,
        tokenizer: BPETokenizer,
        encoder: Optional[TextJEPAEncoder] = None,
        decoder: Optional[ConditionalTextDecoder] = None,
        predictor: Optional[LatentPredictor] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.encoder = encoder or TextJEPAEncoder(config.encoder)
        self.decoder = decoder or ConditionalTextDecoder(config.decoder)
        self.predictor = predictor or LatentPredictor(config.predictor)

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Maps natural language text into invariant Platonic state vector e in R^k.
        """
        self.encoder.eval()
        with torch.no_grad():
            enc = self.tokenizer.encode(
                text,
                add_bos=True,
                add_eos=True,
                max_len=self.config.encoder.max_seq_len,
            )
            input_ids = enc["input_ids"].unsqueeze(0)
            attn_mask = enc["attention_mask"].unsqueeze(0)
            e = self.encoder(input_ids, attention_mask=attn_mask)
        return e  # (1, latent_dim)

    def compare_semantic_invariance(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Compares two texts in the learned JEPA embedding space.
        Returns Euclidean distance and Cosine similarity.
        """
        e1 = self.encode_text(text1)
        e2 = self.encode_text(text2)

        l2_dist = torch.norm(e1 - e2, p=2).item()
        cos_sim = F.cosine_similarity(e1, e2).item()

        return {
            "l2_distance": l2_dist,
            "cosine_similarity": cos_sim,
        }

    def decode_state(
        self,
        z: torch.Tensor,
        style_id: int = 0,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
    ) -> str:
        """
        Renders Platonic state vector z into natural language text tokens.
        """
        self.decoder.eval()
        with torch.no_grad():
            gen_ids = self.decoder.generate(
                z,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                pad_id=self.tokenizer.pad_id,
                style_id=style_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return text

    def run_inference(
        self,
        prompt: str,
        reasoning_steps: int = 3,
        style_id: int = 0,
        max_tokens_per_step: int = 24,
    ) -> Dict[str, Union[str, List[str], torch.Tensor]]:
        """
        Full Prompt-to-Response Pipeline:
        1. Encode prompt -> z_0
        2. Autoregressively rollout future semantic states (z_hat_1, ..., z_hat_m) in R^k
        3. Decode each state vector into human-readable text
        """
        # Step 1: Encode prompt into Platonic semantic space
        z_0 = self.encode_text(prompt)  # (1, latent_dim)

        # Step 2: Rollout reasoning in latent vector space
        z_trajectory = self.predictor.rollout(z_0, num_steps=reasoning_steps)  # (1, 1 + reasoning_steps, latent_dim)

        # Step 3: Render each predicted step using the conditional decoder
        rendered_steps = []
        for step_idx in range(1, 1 + reasoning_steps):
            z_step = z_trajectory[:, step_idx, :]
            step_text = self.decode_state(
                z_step,
                style_id=style_id,
                max_new_tokens=max_tokens_per_step,
            )
            rendered_steps.append(step_text)

        full_response = " ".join(rendered_steps)

        return {
            "prompt": prompt,
            "latent_trajectory": z_trajectory,
            "rendered_steps": rendered_steps,
            "full_response": full_response,
        }
