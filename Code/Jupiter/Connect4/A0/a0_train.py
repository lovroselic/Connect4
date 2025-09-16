# a0_train.py  (drop-in update to add diagnostics in AZTrainer.train_step)
from __future__ import annotations
from typing import Dict, Optional
import torch
from torch import nn

try:
    from torch.amp import autocast, GradScaler
    _USE_TORCH_AMP = True
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    _USE_TORCH_AMP = False

from A0.a0_loss import az_loss

class AZTrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        device: torch.device | str = "cpu",
        use_amp: Optional[bool] = None,
        policy_coef: float = 1.0,
        value_coef: float = 1.0,
    ):
        self.model = model
        self.device = torch.device(device)
        self.device_type = "cuda" if self.device.type == "cuda" else "cpu"

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = float(grad_clip)
        self.use_amp = (torch.cuda.is_available() and self.device.type == "cuda") if use_amp is None else bool(use_amp)

        if _USE_TORCH_AMP:
            try:
                self.scaler = GradScaler(device=self.device_type, enabled=self.use_amp)
            except TypeError:
                self.scaler = GradScaler(enabled=self.use_amp)
        else:
            self.scaler = GradScaler(enabled=self.use_amp)

        self.policy_coef = float(policy_coef)
        self.value_coef = float(value_coef)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states     = batch["states"].to(self.device, non_blocking=True)
        legal_mask = batch["legal_mask"].to(self.device, non_blocking=True)
        target_pi  = batch["target_pi"].to(self.device, non_blocking=True)
        target_z   = batch["target_z"].to(self.device, non_blocking=True)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        ctx = autocast(device_type=self.device_type, enabled=self.use_amp) if _USE_TORCH_AMP else autocast(enabled=self.use_amp)
        with ctx:
            logits, v = self.model(states, legal_mask=legal_mask)
            loss, stats = az_loss(
                logits, v, target_pi, target_z, legal_mask=legal_mask,
                policy_coef=self.policy_coef, value_coef=self.value_coef
            )

            # --- live diagnostics ---
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs.clamp_min(1e-8))
            ent_model = (-(probs * log_probs).sum(dim=-1)).mean()

            # normalize target π per-row, mask & renorm like the loss
            pi_norm = target_pi / target_pi.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            pi_norm = torch.where(legal_mask, pi_norm, torch.zeros_like(pi_norm))
            pi_norm = pi_norm / pi_norm.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            ent_target = (-(pi_norm * torch.log(pi_norm.clamp_min(1e-8))).sum(dim=-1)).mean()

            top1_model  = probs.argmax(dim=-1)
            top1_target = pi_norm.argmax(dim=-1)
            top1_match  = (top1_model == top1_target).float().mean()

        # step
        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.grad_clip and self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        self.model.eval()
        stats.update({
            "ent_model": float(ent_model.detach().item()),
            "ent_target": float(ent_target.detach().item()),
            "top1_match": float(top1_match.detach().item())
        })
        return stats
