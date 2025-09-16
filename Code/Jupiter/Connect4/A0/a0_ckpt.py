# A0/a0_ckpt.py
from __future__ import annotations
import os
from typing import Optional, Any, Dict, Tuple
import torch

def save_checkpoint(
    model,
    optimizer,
    scaler,
    cycle: int,
    global_updates: int,
    buffer_len: int,
    path: str,
    extra: Optional[Dict[str, Any]] = None,
    replay_buffer_state: Optional[Dict[str, Any]] = None,
) -> str:
    pkg: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "meta": {
            "cycle": int(cycle),
            "global_updates": int(global_updates),
            "buffer_len": int(buffer_len),
        },
    }
    if extra is not None:
        pkg["extra"] = extra
    if replay_buffer_state is not None:
        pkg["replay_buffer"] = replay_buffer_state

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(pkg, path)
    return path

def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)

def restore_from_checkpoint(
    path: str,
    model,
    optimizer=None,
    scaler=None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    pkg = torch.load(path, map_location=device or "cpu")

    # Model
    model.load_state_dict(pkg["model"], strict=strict)

    # Optimizer (move state tensors to device)
    if optimizer is not None and pkg.get("optimizer") is not None:
        optimizer.load_state_dict(pkg["optimizer"])
        dev = device or next(model.parameters()).device
        for st in optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(dev)

    # Grad scaler
    if scaler is not None and pkg.get("scaler") is not None:
        scaler.load_state_dict(pkg["scaler"])

    meta = pkg.get("meta", {})
    rb_state = pkg.get("replay_buffer", None)
    return meta, rb_state
