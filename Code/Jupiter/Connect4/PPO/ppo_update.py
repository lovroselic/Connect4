# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 13:56:55 2025

@author: Uporabnik
"""

# ppo_update.py

import torch
from dataclasses import dataclass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class PPOUpdateCfg:
    epochs: int = 4
    batch_size: int = 256
    clip_range: float = 0.2          # policy clip
    vf_clip_range: float = 0.2       # value clip
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03   # set None to disable early stopping

def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    1 - Var[y_true - y_pred] / Var[y_true]
    """
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    var_y = torch.var(y_true)
    return float(1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8))

def ppo_update(policy, optimizer, buffer, cfg: PPOUpdateCfg = PPOUpdateCfg()):
    """
    Runs multiple epochs of PPO on data from `buffer`.
    Returns a dict of averaged metrics.
    """
    policy.train()

    data = buffer.get_tensors()
    states       = data["states"]         # [T,6,7]
    actions      = data["actions"]        # [T]
    old_logprobs = data["old_logprobs"]   # [T]
    values_old   = data["values"]         # [T]
    advantages   = data["advantages"]     # [T]
    returns      = data["returns"]        # [T]
    legal_masks  = data["legal_masks"]    # [T,7] bool

    T = states.size(0)
    assert T > 0, "Empty buffer."

    # Detach things that should not carry grads
    old_logprobs = old_logprobs.detach()
    values_old   = values_old.detach()
    advantages   = advantages.detach()
    returns      = returns.detach()

    # Accumulators
    total_pi, total_v, total_ent, total_kl, total_clipfrac = 0.0, 0.0, 0.0, 0.0, 0.0
    total_steps = 0

    for epoch in range(cfg.epochs):
        # shuffle indices each epoch
        idx = torch.randperm(T, device=DEVICE)

        for start in range(0, T, cfg.batch_size):
            mb_idx = idx[start:start + cfg.batch_size]

            mb_states       = states[mb_idx]            # [B,6,7]
            mb_actions      = actions[mb_idx]           # [B]
            mb_old_logprobs = old_logprobs[mb_idx]      # [B]
            mb_advantages   = advantages[mb_idx]        # [B]
            mb_returns      = returns[mb_idx]           # [B]
            mb_values_old   = values_old[mb_idx]        # [B]
            mb_legal_masks  = legal_masks[mb_idx]       # [B,7]

            # Policy eval on minibatch
            new_logprobs, entropy, values = policy.evaluate_actions(
                mb_states, mb_actions, legal_action_masks=mb_legal_masks
            )

            # Ratio for clipping
            ratio = torch.exp(new_logprobs - mb_old_logprobs)    # [B]
            # Policy loss (clipped surrogate)
            unclipped = -mb_advantages * ratio
            clipped   = -mb_advantages * torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
            loss_pi   = torch.mean(torch.max(unclipped, clipped))

            # Value loss (with optional clipping)
            if cfg.vf_clip_range is not None:
                v_pred_clipped = mb_values_old + torch.clamp(values - mb_values_old,
                                                            -cfg.vf_clip_range, cfg.vf_clip_range)
                v_loss_unclipped = (values - mb_returns).pow(2)
                v_loss_clipped   = (v_pred_clipped - mb_returns).pow(2)
                loss_v = 0.5 * torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))
            else:
                loss_v = 0.5 * torch.mean((values - mb_returns).pow(2))

            # Entropy bonus (maximize entropy => subtract)
            loss_ent = -torch.mean(entropy)

            loss = loss_pi + cfg.vf_coef * loss_v + cfg.ent_coef * loss_ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            # Metrics
            with torch.no_grad():
                approx_kl = torch.mean(mb_old_logprobs - new_logprobs).abs().item()
                clip_frac = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_range).float()).item()

            bs = mb_idx.numel()
            total_pi       += loss_pi.item() * bs
            total_v        += loss_v.item() * bs
            total_ent      += entropy.mean().item() * bs
            total_kl       += approx_kl * bs
            total_clipfrac += clip_frac * bs
            total_steps    += bs

        # Optional early stop on large policy shift
        if cfg.target_kl is not None and (total_kl / max(1, total_steps)) > cfg.target_kl:
            # print(f"Early stop at epoch {epoch+1} due to KL {total_kl/total_steps:.4f} > {cfg.target_kl}")
            break

    # One more forward to compute EV on full batch (values vs returns)
    with torch.no_grad():
        _, _, values_final = policy.evaluate_actions(states, actions, legal_action_masks=legal_masks)
        ev = explained_variance(values_final, returns)

    return {
        "loss_pi": total_pi / max(1, total_steps),
        "loss_v": total_v / max(1, total_steps),
        "entropy": total_ent / max(1, total_steps),
        "approx_kl": total_kl / max(1, total_steps),
        "clip_frac": total_clipfrac / max(1, total_steps),
        "explained_variance": ev,
        "updates_samples": total_steps,
    }
