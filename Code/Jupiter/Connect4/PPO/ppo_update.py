# PPO/ppo_update.py
from __future__ import annotations

import torch
from dataclasses import dataclass
from torch.nn import functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PPOUpdateCfg:
    epochs: int = 4
    batch_size: int = 256
    clip_range: float = 0.2
    vf_clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    temperature: float = 1.0

    # distillation
    distill_coef: float = 0.0

    # mentor guidance
    mentor_depth: int = 1
    mentor_prob: float = 0.0
    mentor_coef: float = 0.0


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    1 - Var[y - y_hat] / Var[y]
    Use unbiased=False for stability on small-ish batches.
    """
    y_true = y_true.detach().flatten()
    y_pred = y_pred.detach().flatten()
    var_y = torch.var(y_true, unbiased=False)
    return float(1.0 - torch.var(y_true - y_pred, unbiased=False) / (var_y + 1e-8))


def ppo_update(
    policy,
    optimizer,
    buffer,
    cfg: PPOUpdateCfg = PPOUpdateCfg(),
    teacher=None,
    mentor=None,
):
    policy.train()

    data = buffer.get_tensors()
    states       = data["states"]
    actions      = data["actions"]
    old_logprobs = data["old_logprobs"]
    values_old   = data["values"]
    advantages   = data["advantages"]
    returns      = data["returns"]
    legal_masks  = data["legal_masks"]
    ply_indices  = data.get("ply_indices", None)

    T = states.size(0)
    assert T > 0, "Empty buffer."

    # Detach rollouts
    old_logprobs = old_logprobs.detach().flatten()
    values_old   = values_old.detach().flatten()
    advantages   = advantages.detach().flatten()
    returns      = returns.detach().flatten()

    # ---- IMPORTANT: normalize advantages (critical for nonstationary opponents) ----
    adv_mean = advantages.mean()
    adv_std  = advantages.std(unbiased=False)
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    total_pi = total_v = total_ent = 0.0
    total_kl = total_clipfrac = 0.0
    total_distill_kl = 0.0
    total_mentor_ce  = 0.0
    total_steps = 0

    use_distill = (teacher is not None) and (getattr(cfg, "distill_coef", 0.0) > 0.0)
    use_mentor  = (mentor  is not None) and (getattr(cfg, "mentor_coef",  0.0) > 0.0) and (getattr(cfg, "mentor_prob", 0.0) > 0.0)

    stop_early = False
    aux_scale = 1.0

    # A slightly softer early-stop threshold prevents "one minibatch nukes the epoch"
    kl_stop_mult = 1.5  # typical: 1.5x–2.0x

    for epoch in range(cfg.epochs):
        idx = torch.randperm(T, device=states.device)

        for start in range(0, T, cfg.batch_size):
            mb_idx = idx[start:start + cfg.batch_size]

            mb_states       = states[mb_idx]
            mb_actions      = actions[mb_idx]
            mb_old_logprobs = old_logprobs[mb_idx]
            mb_advantages   = advantages[mb_idx]
            mb_returns      = returns[mb_idx]
            mb_values_old   = values_old[mb_idx]
            mb_legal_masks  = legal_masks[mb_idx]
            mb_ply_indices  = ply_indices[mb_idx] if ply_indices is not None else None

            new_logprobs, entropy, values = policy.evaluate_actions(
                mb_states,
                mb_actions,
                legal_action_masks=mb_legal_masks,
                ply_indices=mb_ply_indices,
                temperature=cfg.temperature,
            )

            new_logprobs = new_logprobs.flatten()
            entropy      = entropy.flatten()
            values       = values.flatten()

            # ----- PPO ratios -----
            log_ratio = new_logprobs - mb_old_logprobs
            ratio = torch.exp(log_ratio)

            # ----- KL (use direct for early stop; less jitter) -----
            approx_kl = torch.mean(mb_old_logprobs - new_logprobs).item()

            # ----- aux fadeout when KL high (only matters for distill/mentor) -----
            if cfg.target_kl is None:
                aux_scale = 1.0
            else:
                aux_scale = max(
                    0.0,
                    min(1.0, (cfg.target_kl - approx_kl) / max(1e-12, cfg.target_kl))
                )

            # Policy loss (clipped surrogate)
            unclipped = -mb_advantages * ratio
            clipped   = -mb_advantages * torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
            loss_pi   = torch.mean(torch.max(unclipped, clipped))

            # Value loss (clip) — SmoothL1/Huber
            if cfg.vf_clip_range is not None:
                v_pred_clipped = mb_values_old + torch.clamp(
                    values - mb_values_old,
                    -cfg.vf_clip_range,
                    cfg.vf_clip_range,
                )
                v_loss_unclipped = F.smooth_l1_loss(values,         mb_returns, reduction="none")
                v_loss_clipped   = F.smooth_l1_loss(v_pred_clipped, mb_returns, reduction="none")
                loss_v = torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))
            else:
                loss_v = torch.mean(F.smooth_l1_loss(values, mb_returns, reduction="none"))

            # Entropy bonus (note: we add ent_coef * (-entropy))
            loss_ent = -torch.mean(entropy)

            loss = loss_pi + cfg.vf_coef * loss_v + cfg.ent_coef * loss_ent

            # ---------- distillation ----------
            distill_kl = None
            mentor_ce  = None
            s_logits_for_aux = None

            distill_active = use_distill and (aux_scale > 0.0)
            mentor_active  = use_mentor  and (aux_scale > 0.0)

            if distill_active or mentor_active:
                s_logits_for_aux, _ = policy(mb_states)  # raw logits (B,7)

            if distill_active:
                with torch.no_grad():
                    mb_states_1ch = policy._to_1ch_pov(mb_states)
                    t_logits, _ = teacher(mb_states_1ch)

                lm = mb_legal_masks
                no_legal = (lm.sum(dim=-1) == 0)
                if no_legal.any():
                    lm = lm.clone()
                    lm[no_legal] = True

                illegal = ~lm
                s_logits_masked = s_logits_for_aux.masked_fill(illegal, -1e9)
                t_logits_masked = t_logits.masked_fill(illegal, -1e9)

                s_logp = torch.log_softmax(s_logits_masked, dim=-1)
                t_logp = torch.log_softmax(t_logits_masked, dim=-1)
                t_prob = t_logp.exp()

                distill_kl = (t_prob * (t_logp - s_logp)).sum(dim=-1).mean()
                loss = loss + (cfg.distill_coef * aux_scale) * distill_kl

            # ---------- mentor guidance ----------
            if mentor_active:
                B = mb_states.size(0)
                mask_idx = (torch.rand(B, device=mb_states.device) < cfg.mentor_prob)
                if mask_idx.any():
                    states_sub = mb_states[mask_idx]
                    legal_sub  = mb_legal_masks[mask_idx]

                    mentor_actions = mentor(
                        states_sub,
                        legal_sub,
                        depth=getattr(cfg, "mentor_depth", 1),
                    )

                    s_logits_sub = s_logits_for_aux[mask_idx]
                    illegal_sub  = ~legal_sub
                    s_logits_sub = s_logits_sub.masked_fill(illegal_sub, -1e9)

                    s_logp_sub = torch.log_softmax(s_logits_sub, dim=-1)
                    mentor_ce = F.nll_loss(s_logp_sub, mentor_actions.long(), reduction="mean")
                    loss = loss + (cfg.mentor_coef * aux_scale) * mentor_ce

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                clip_frac = torch.mean(
                    ((ratio < (1.0 - cfg.clip_range)) | (ratio > (1.0 + cfg.clip_range))).float()
                ).item()

            bs = mb_idx.numel()
            total_pi       += loss_pi.item() * bs
            total_v        += loss_v.item() * bs
            total_ent      += entropy.mean().item() * bs
            total_kl       += approx_kl * bs
            total_clipfrac += clip_frac * bs
            if distill_kl is not None:
                total_distill_kl += distill_kl.item() * bs
            if mentor_ce is not None:
                total_mentor_ce  += mentor_ce.item() * bs
            total_steps    += bs

            # Softer stop to avoid "flatline by over-pruning updates"
            if cfg.target_kl is not None and approx_kl > (kl_stop_mult * cfg.target_kl):
                stop_early = True
                break

        if stop_early:
            break

    with torch.no_grad():
        _, _, values_final = policy.evaluate_actions(
            states,
            actions,
            legal_action_masks=legal_masks,
            ply_indices=ply_indices,
            temperature=cfg.temperature,
        )
        ev = explained_variance(values_final, returns)

    return {
        "loss_pi": total_pi / max(1, total_steps),
        "loss_v": total_v / max(1, total_steps),
        "entropy": total_ent / max(1, total_steps),
        "approx_kl": total_kl / max(1, total_steps),
        "clip_frac": total_clipfrac / max(1, total_steps),
        "explained_variance": ev,
        "updates_samples": total_steps,
        "distill_kl": (total_distill_kl / max(1, total_steps)) if use_distill else 0.0,
        "mentor_ce":  (total_mentor_ce  / max(1, total_steps)) if use_mentor  else 0.0,
        "aux_scale_last": float(aux_scale),
        "stopped_early": bool(stop_early),
    }
