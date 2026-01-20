# kaggle: main.py
# package:
#   tar -czf submit.tar.gz -C submission main.py PPO_14.pt

import os
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- globals (cached across moves) ----------
_DEVICE = torch.device("cpu")
_MODEL: Optional[nn.Module] = None
_CENTER_COL = 3
MODEL_FILE = "PPO_14.pt"


# ---------- CNet192 ----------
class CNet192(nn.Module):
    def __init__(self, in_channels: int = 1, use_mid_3x3: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.use_mid_3x3 = bool(use_mid_3x3)

        self.conv1 = nn.Conv2d(self.in_channels, 192, kernel_size=4, padding=0)  # 6x7 -> 3x4
        self.conv_mid = nn.Conv2d(192, 192, kernel_size=3, padding=1) if self.use_mid_3x3 else None
        self.conv2 = nn.Conv2d(192, 192, kernel_size=2, padding=0)  # 3x4 -> 2x3

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
        pol = self.policy_out(pol)            # (B,7)
        val = F.relu(self.value_fc(x))
        val = self.value_out(val).squeeze(-1) # (B,)
        return pol, val


def _torch_load_any(path: str):
    try:
        return torch.load(path, map_location=_DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=_DEVICE)


def _strip_prefix(sd: dict, prefix: str) -> dict:
    # Keep as-is unless *all* keys share the prefix (avoid silently dropping keys)
    keys = list(sd.keys())
    if keys and all(isinstance(k, str) and k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def _extract_state_and_use_mid(ckpt):
    cfg = {}
    state = None

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        cfg = dict(ckpt.get("cfg", {}) or {})
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        cfg = dict(ckpt.get("cfg", {}) or {})
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        state = ckpt
    else:
        raise RuntimeError("Unrecognized checkpoint format (expected dict with state dict)")

    state = _strip_prefix(state, "module.")
    state = _strip_prefix(state, "net.")

    use_mid = bool(cfg.get("use_mid_3x3", any(k.startswith("conv_mid.") for k in state.keys())))
    return state, use_mid


def _find_model_path() -> str:
    candidates: List[str] = []
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(here, MODEL_FILE))
    except Exception:
        pass
    candidates.append(os.path.join(os.getcwd(), MODEL_FILE))
    candidates.append(os.path.join("/kaggle_simulations/agent", MODEL_FILE))

    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {MODEL_FILE} in: {candidates}")


def _load_model_once() -> nn.Module:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    ckpt_path = _find_model_path()
    ckpt = _torch_load_any(ckpt_path)
    state, use_mid = _extract_state_and_use_mid(ckpt)

    m = CNet192(in_channels=1, use_mid_3x3=use_mid).to(_DEVICE)
    m.load_state_dict(state, strict=True)
    m.eval()

    _MODEL = m
    return _MODEL


# ---------- tiny tactic (ONE): win-now + must-block ----------
def _lowest_empty_row_from_topgrid(grid: np.ndarray, col: int) -> int:
    # grid is (6,7) with row 0 = top
    for r in range(5, -1, -1):
        if grid[r, col] == 0:
            return r
    return -1


def _has_four_from(grid: np.ndarray, row: int, col: int, token: int) -> bool:
    # token is +1 or -1 in POV board
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1

        rr, cc = row + dr, col + dc
        while 0 <= rr < 6 and 0 <= cc < 7 and grid[rr, cc] == token:
            count += 1
            rr += dr
            cc += dc

        rr, cc = row - dr, col - dc
        while 0 <= rr < 6 and 0 <= cc < 7 and grid[rr, cc] == token:
            count += 1
            rr -= dr
            cc -= dc

        if count >= 4:
            return True
    return False


def _is_winning_drop(pov_grid: np.ndarray, col: int, token: int) -> bool:
    r = _lowest_empty_row_from_topgrid(pov_grid, col)
    if r < 0:
        return False
    old = pov_grid[r, col]
    pov_grid[r, col] = token
    win = _has_four_from(pov_grid, r, col, token)
    pov_grid[r, col] = old
    return win


def _argmax_legal_center_tiebreak(vals: np.ndarray, legal: List[int], center: int = 3) -> int:
    best = max(vals[c] for c in legal)
    tied = [c for c in legal if vals[c] == best]
    if len(tied) == 1:
        return int(tied[0])
    return int(sorted(tied, key=lambda c: (abs(c - center), c))[0])


# ---------- Kaggle agent ----------
def agent(obs, config):
    try:
        model = _load_model_once()

        mark = int(obs["mark"]) if isinstance(obs, dict) else int(obs.mark)
        flat = obs["board"] if isinstance(obs, dict) else obs.board
        grid = np.asarray(flat, dtype=np.int8).reshape(6, 7)

        legal = [c for c in range(7) if grid[0, c] == 0]
        if not legal:
            return 0

        stones = int(np.count_nonzero(grid))
        if stones == 0 and 3 in legal:
            return 3  # opening book: center

        # POV scalar: me=+1, opp=-1
        pov = np.zeros((6, 7), dtype=np.int8)
        pov[grid == mark] = 1
        pov[(grid != 0) & (grid != mark)] = -1

        # --- ONE tiny tactic: win-now ---
        for c in legal:
            if _is_winning_drop(pov, c, +1):
                return int(c)

        # --- ONE tiny tactic: must-block (if opponent has a win-in-1) ---
        blocks = [c for c in legal if _is_winning_drop(pov, c, -1)]
        if blocks:
            # if multiple blocks exist, pick the one closest to center (stable, reasonable)
            return int(sorted(blocks, key=lambda c: (abs(c - _CENTER_COL), c))[0])

        # policy argmax with legal mask
        x = torch.from_numpy(pov.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(_DEVICE)
        with torch.inference_mode():
            logits, _ = model(x)
            logits = logits[0].detach().cpu().numpy().astype(np.float32)

        masked = np.full(7, -1e9, dtype=np.float32)
        for c in legal:
            masked[c] = logits[c]

        return int(_argmax_legal_center_tiebreak(masked, legal, center=_CENTER_COL))

    except Exception:
        # never crash validation
        flat = obs["board"] if isinstance(obs, dict) else obs.board
        grid = np.asarray(flat, dtype=np.int8).reshape(6, 7)
        legal = [c for c in range(7) if grid[0, c] == 0]
        if not legal:
            return 0
        if 3 in legal:
            return 3
        return int(legal[0])
