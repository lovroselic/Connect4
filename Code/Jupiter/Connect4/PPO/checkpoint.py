# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 22:47:15 2025

@author: Uporabnik
"""

# PPO.checkpoint.py

# === Save & Load utilities for PPO ===
import os, time, torch
from PPO.actor_critic import ActorCritic

def save_checkpoint(
    policy,
    optim,
    episode: int,
    lookahead_depth,
    cfg,
    hparams,
    model_path: str,             # e.g. f"{MODEL_DIR}{TRAINING_SESSION}_Connect4 ppo_model_{...} episodes-{...} lookahead-{...}.pt"
    default_model_path: str,     # must be EXACT "Connect4 PPQ model.pt"
):
    # build checkpoint (model + optimizer + a bit of meta)
    ckpt = {
        "model_state": policy.state_dict(),
        "optim_state": optim.state_dict(),
        "episode": int(episode),
        "lookahead_depth": (
            -1 if isinstance(lookahead_depth, str) and lookahead_depth.lower() == "self"
            else (0 if lookahead_depth is None else int(lookahead_depth))
        ),
        "cfg": dict(vars(cfg)) if hasattr(cfg, "__dict__") else None,
        "hparams": dict(vars(hparams)) if hasattr(hparams, "__dict__") else None,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "framework": "ppo",
        "torch_version": torch.__version__,
    }

    # ensure directories exist (if any)
    dir1 = os.path.dirname(model_path)
    if dir1:
        os.makedirs(dir1, exist_ok=True)
    dir2 = os.path.dirname(default_model_path)
    if dir2:
        os.makedirs(dir2, exist_ok=True)

    # save to BOTH exact paths
    torch.save(ckpt, model_path)
    torch.save(ckpt, default_model_path)

    return model_path, default_model_path


def load_checkpoint(load_path: str, device, strict: bool = True):
    """
    Load either a checkpoint dict (recommended) or a raw state_dict.
    Returns (model, optimizer_state_or_None, meta_dict).
    """
    try:
        obj = torch.load(load_path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(load_path, map_location=device)

    model = ActorCritic(action_dim=7).to(device)
    optim_state = None
    meta = {}

    if isinstance(obj, dict) and "model_state" in obj:
        model.load_state_dict(obj["model_state"], strict=strict)
        optim_state = obj.get("optim_state")
        meta = {k: obj.get(k) for k in ("episode","lookahead_depth","cfg","hparams","timestamp","framework","torch_version")}
    else:
        # plain state_dict fallback
        model.load_state_dict(obj, strict=strict)

    model.eval()
    return model, optim_state, meta