# a0_loss.py
# AlphaZero policy+value loss with legal-mask-safe policy term.

from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn.functional as F

def az_loss(
    policy_logits: torch.Tensor,   # [B,7]
    value_pred: torch.Tensor,      # [B]
    target_pi: torch.Tensor,       # [B,7] (visit-count distribution; zeros on illegal ok)
    target_z: torch.Tensor,        # [B] in [-1, 0, 1]
    legal_mask: torch.Tensor | None = None,  # [B,7] bool
    policy_coef: float = 1.0,
    value_coef: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Normalize π per row (robust)
    pi_norm = target_pi / target_pi.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # Log-probs from logits (these may be -inf on illegal after masking in the net)
    logp = F.log_softmax(policy_logits, dim=-1)

    if legal_mask is not None:
        m = legal_mask.to(dtype=torch.bool)
        # Zero-out illegal positions BEFORE multiplying to avoid 0 * (-inf) -> NaN
        logp = torch.where(m, logp, torch.zeros_like(logp))
        pi_masked = torch.where(m, pi_norm, torch.zeros_like(pi_norm))
        pi_norm = pi_masked / pi_masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    policy_loss = -(pi_norm * logp).sum(dim=-1).mean()
    value_loss  = F.mse_loss(value_pred.view_as(target_z), target_z)

    loss = policy_coef * policy_loss + value_coef * value_loss
    stats = {
        "loss_total": float(loss.detach().item()),
        "loss_policy": float(policy_loss.detach().item()),
        "loss_value": float(value_loss.detach().item()),
    }
    return loss, stats
