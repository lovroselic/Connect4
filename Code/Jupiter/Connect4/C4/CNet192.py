#CNet192.py

# models/cnet192.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNet192(nn.Module):
    """
    CNet192:
      - Conv( in_ch -> 192, k=4, pad=0): 6x7 -> 3x4
      - Optional mid Conv(192 -> 192, k=3, pad=1): 3x4 -> 3x4
      - Conv(192 -> 192, k=2, pad=0): 3x4 -> 2x3
      - FC to 192
      - Policy head: 192 -> 192 -> 7
      - Value head : 192 -> 192 -> 1
    """
    def __init__(self, in_channels: int = 1, use_mid_3x3: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.use_mid_3x3 = bool(use_mid_3x3)

        self.conv1 = nn.Conv2d(self.in_channels, 192, kernel_size=4, padding=0)  # 6x7 -> 3x4
        self.conv_mid = nn.Conv2d(192, 192, kernel_size=3, padding=1) if self.use_mid_3x3 else None
        self.conv2 = nn.Conv2d(192, 192, kernel_size=2, padding=0)  # 3x4 -> 2x3

        # infer flatten size robustly
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, 6, 7)
            z = self._forward_conv(dummy)
            self.flat = int(np.prod(z.shape[1:]))

        self.fc = nn.Linear(self.flat, 192)

        self.policy_fc = nn.Linear(192, 192)
        self.policy_out = nn.Linear(192, 7)

        self.value_fc = nn.Linear(192, 192)
        self.value_out = nn.Linear(192, 1)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        if self.conv_mid is not None:
            x = F.relu(self.conv_mid(x))
        x = F.relu(self.conv2(x))
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        pol = F.relu(self.policy_fc(x))
        pol = self.policy_out(pol)                 # logits (B,7)

        val = F.relu(self.value_fc(x))
        val = self.value_out(val).squeeze(-1)      # (B,)

        return pol, val


def save_cnet192(
    path: str | Path,
    model: CNet192,
    cfg: Optional[Dict[str, Any]] = None,
    **meta: Any,
) -> None:
    """
    Save CNet192 checkpoint with:
      - model_state_dict
      - cfg (merged defaults + user cfg)
      - extra meta (best metric, scores, dataset info, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    default_cfg = {
        "arch": "cnet192",
        "use_mid_3x3": bool(getattr(model, "use_mid_3x3", True)),
        "input_channels": int(getattr(model, "in_channels", 1)),
    }
    user_cfg = dict(cfg or {})
    merged_cfg = {**default_cfg, **user_cfg}

    payload = {
        "model_state_dict": model.state_dict(),
        "cfg": merged_cfg,
        **meta,
    }
    torch.save(payload, path)


def load_cnet192(
    path: str | Path,
    device: torch.device,
    strict: bool = True,
    override_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[CNet192, Dict[str, Any]]:
    """
    Loads a CNet192 checkpoint produced by save_cnet192().

    Returns:
      (model, ckpt_dict)

    Notes:
      - cfg is read from ckpt["cfg"] and can be overridden by override_cfg.
      - all extra metadata saved in the checkpoint remains in ckpt_dict.
    """
    path = Path(path)
    ckpt: Dict[str, Any] = torch.load(path, map_location=device, weights_only=True)

    cfg = dict(ckpt.get("cfg", {}) or {})
    if override_cfg:
        cfg.update(dict(override_cfg))

    # infer architecture params (with safe defaults)
    use_mid_3x3 = bool(cfg.get("use_mid_3x3", True))

    # accept either "input_channels" (new) or legacy "in_channels"
    in_ch = cfg.get("input_channels", cfg.get("in_channels", 1))
    in_ch = int(in_ch)

    model = CNet192(in_channels=in_ch, use_mid_3x3=use_mid_3x3).to(device)

    state = ckpt.get("model_state_dict", None)
    if state is None:
        raise KeyError(f"Checkpoint {path} missing key: 'model_state_dict'")

    model.load_state_dict(state, strict=strict)
    return model, ckpt
