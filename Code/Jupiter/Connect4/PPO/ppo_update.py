# ppo_update.py

import torch
from dataclasses import dataclass
from torch.nn import functional as F

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

    # distillation
    distill_coef: float = 0.0

    # mentor guidance
    mentor_depth: int = 1
    mentor_prob: float = 0.0      # 0 => disabled
    mentor_coef: float = 0.0      # 0 => disabled


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    1 - Var[y_true - y_pred] / Var[y_true]
    """
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    var_y = torch.var(y_true)
    return float(1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8))


def ppo_update(
    policy,
    optimizer,
    buffer,
    cfg: PPOUpdateCfg = PPOUpdateCfg(),
    teacher=None,          # optional frozen teacher (e.g. POP ensemble)
    mentor=None,           # optional lookahead mentor
):
    """
    Runs multiple epochs of PPO on data from `buffer`.
    Optionally adds:
      - KL(student || teacher) distillation if teacher is not None and cfg.distill_coef > 0
      - mentor cross-entropy if mentor is not None and mentor_* > 0
    Returns a dict of averaged metrics.
    """
    policy.train()

    data = buffer.get_tensors()
    states       = data["states"]         # [T,...]
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
    total_pi, total_v, total_ent = 0.0, 0.0, 0.0
    total_kl, total_clipfrac     = 0.0, 0.0
    total_distill_kl             = 0.0
    total_mentor_ce              = 0.0
    total_steps                  = 0

    use_distill = (teacher is not None) and (getattr(cfg, "distill_coef", 0.0) > 0.0)
    use_mentor  = (mentor  is not None) and (getattr(cfg, "mentor_coef",  0.0) > 0.0) \
                  and (getattr(cfg, "mentor_prob",  0.0) > 0.0)

    for epoch in range(cfg.epochs):
        # shuffle indices each epoch
        idx = torch.randperm(T, device=DEVICE)

        for start in range(0, T, cfg.batch_size):
            mb_idx = idx[start:start + cfg.batch_size]

            mb_states       = states[mb_idx]            # [B,...]
            mb_actions      = actions[mb_idx]           # [B]
            mb_old_logprobs = old_logprobs[mb_idx]      # [B]
            mb_advantages   = advantages[mb_idx]        # [B]
            mb_returns      = returns[mb_idx]           # [B]
            mb_values_old   = values_old[mb_idx]        # [B]
            mb_legal_masks  = legal_masks[mb_idx]       # [B,7] bool

            # Policy eval on minibatch
            new_logprobs, entropy, values = policy.evaluate_actions(
                mb_states, mb_actions, legal_action_masks=mb_legal_masks
            )

            # Ratio for clipping
            ratio = torch.exp(new_logprobs - mb_old_logprobs)    # [B]

            # Policy loss (clipped surrogate)
            unclipped = -mb_advantages * ratio
            clipped   = -mb_advantages * torch.clamp(
                ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range
            )
            loss_pi   = torch.mean(torch.max(unclipped, clipped))

            # Value loss (with optional clipping)
            if cfg.vf_clip_range is not None:
                v_pred_clipped = mb_values_old + torch.clamp(
                    values - mb_values_old,
                    -cfg.vf_clip_range,
                    cfg.vf_clip_range,
                )
                v_loss_unclipped = (values - mb_returns).pow(2)
                v_loss_clipped   = (v_pred_clipped - mb_returns).pow(2)
                loss_v = 0.5 * torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))
            else:
                loss_v = 0.5 * torch.mean((values - mb_returns).pow(2))

            # Entropy bonus (maximize entropy => subtract)
            loss_ent = -torch.mean(entropy)

            # Base PPO loss
            loss = loss_pi + cfg.vf_coef * loss_v + cfg.ent_coef * loss_ent

            # ---------- distillation from teacher (POP ensemble) ----------
            distill_kl = None
            s_logits_for_aux = None

            if use_distill or use_mentor:
                # we need raw logits for both distill + mentor
                s_logits_for_aux, _ = policy(mb_states)          # [B, 7]

            if use_distill:
                with torch.no_grad():
                    t_logits, _ = teacher(mb_states)     # [B, 7]

                if mb_legal_masks is not None:
                    illegal = ~mb_legal_masks
                    big_neg = -1e9
                    s_logits_masked = s_logits_for_aux.masked_fill(illegal, big_neg)
                    t_logits_masked = t_logits.masked_fill(illegal, big_neg)
                else:
                    s_logits_masked = s_logits_for_aux
                    t_logits_masked = t_logits

                s_logp = torch.log_softmax(s_logits_masked, dim=-1)
                t_logp = torch.log_softmax(t_logits_masked, dim=-1)
                t_prob = t_logp.exp()

                distill_kl = (t_prob * (t_logp - s_logp)).sum(dim=-1).mean()
                loss = loss + cfg.distill_coef * distill_kl

            # ---------- mentor guidance (lookahead) ----------
            mentor_ce = None
            if use_mentor:
                B = mb_states.size(0)
                # sample subset of states for mentor signal
                mask_idx = torch.rand(B, device=mb_states.device) < cfg.mentor_prob

                if mask_idx.any():
                    states_sub = mb_states[mask_idx]
                    legal_sub  = mb_legal_masks[mask_idx]

                    mentor_actions = mentor(
                        states_sub,
                        legal_sub,
                        depth=getattr(cfg, "mentor_depth", 1),
                    )  # LongTensor [B_sub]

                    # logits for same subset
                    s_logits_sub = s_logits_for_aux[mask_idx]

                    illegal_sub = ~legal_sub
                    big_neg = -1e9
                    s_logits_sub = s_logits_sub.masked_fill(illegal_sub, big_neg)

                    s_logp_sub = torch.log_softmax(s_logits_sub, dim=-1)

                    # CE(student, mentor_action)
                    mentor_ce = F.nll_loss(
                        s_logp_sub,
                        mentor_actions.long(),
                        reduction="mean",
                    )

                    loss = loss + cfg.mentor_coef * mentor_ce
            # -------------------------------------------------------------------

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            # Metrics
            with torch.no_grad():
                approx_kl = torch.mean(mb_old_logprobs - new_logprobs).abs().item()
                clip_frac = torch.mean(
                    (torch.abs(ratio - 1.0) > cfg.clip_range).float()
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

        # Optional early stop on large policy shift
        if cfg.target_kl is not None and (total_kl / max(1, total_steps)) > cfg.target_kl:
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
        "distill_kl": (
            total_distill_kl / max(1, total_steps)
            if use_distill else 0.0
        ),
        "mentor_ce": (
            total_mentor_ce / max(1, total_steps)
            if use_mentor else 0.0
        ),
    }
