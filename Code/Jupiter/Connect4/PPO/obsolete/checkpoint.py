
# PPO.checkpoint.py
# === Save & Load utilities for PPO ===

from __future__ import annotations
import time, torch
from PPO.actor_critic import ActorCritic


def _obj_to_dict_or_none(obj):
    try:
        return dict(vars(obj))
    except Exception:
        return None


def save_checkpoint(
    policy,
    optim,
    episode: int,
    cfg,
    hparams,
    model_path: str,             # e.g. f"{MODEL_DIR}{TRAINING_SESSION}_... .pt"
    default_model_path: str,     # must be EXACT "Connect4 PPQ model.pt"
):
    """
    Save PPO checkpoint to both a versioned path and the fixed default path.
    Uses standard keys: 'model_state_dict' and 'optim_state_dict'.
    """

    ckpt = {
        "model_state_dict": policy.state_dict(),
        "optim_state_dict": optim.state_dict() if optim is not None else None,
        "episode": int(episode),
        "cfg": _obj_to_dict_or_none(cfg),
        "hparams": _obj_to_dict_or_none(hparams),
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "framework": "ppo",
        "torch_version": torch.__version__,
    }

    torch.save(ckpt, model_path)
    if default_model_path is not None: torch.save(ckpt, default_model_path)
    return model_path, default_model_path


def load_checkpoint(load_path: str, device, strict: bool = True):
    """
    Load a checkpoint and return (model, optim_state_or_None, meta_dict).

    Supports:
    - dict with 'model_state_dict'/'optim_state_dict' (preferred)
    - dict with legacy 'model_state'/'optim_state'
    - plain state_dict (no meta)
    """
    try:
        obj = torch.load(load_path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(load_path, map_location=device)

    model = ActorCritic(action_dim=7).to(device)
    optim_state = None
    meta = {}

    if isinstance(obj, dict):
        # Preferred keys
        sd = None
        if "model_state_dict" in obj:
            sd = obj["model_state_dict"]
            optim_state = obj.get("optim_state_dict")
        # Legacy keys
        elif "model_state" in obj:
            sd = obj["model_state"]
            optim_state = obj.get("optim_state")
        # Raw state_dict fallback
        elif all(isinstance(k, str) for k in obj.keys()):
            sd = obj

        if sd is None:
            raise ValueError("Unrecognized checkpoint format: no model state found.")

        model.load_state_dict(sd, strict=strict)

        # meta (best-effort)
        for k in ("episode", "lookahead_depth", "cfg", "hparams", "timestamp", "framework", "torch_version"):
            if k in obj:
                meta[k] = obj[k]

    else:
        # Very old: raw state_dict object
        model.load_state_dict(obj, strict=strict)

    model.eval()
    return model, optim_state, meta
