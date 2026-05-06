# kaggle: main.py
# tar -czf submit.tar.gz -C submission main.py PPO_2002.pt


import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_DEVICE = torch.device("cpu")
_MODEL: Optional[nn.Module] = None

_CENTER_COL = 3
_CENTER_ORDER = (3, 4, 2, 5, 1, 6, 0)

MODEL_FILE = "PPO_2002.pt"
#MODEL_FILE = "AZ_003.pt"


# ---------- CNet192 ----------
class CNet192(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 192, kernel_size=4, padding=0)      # 6x7 -> 3x4
        self.conv_mid = nn.Conv2d(192, 192, kernel_size=3, padding=1)           # 3x4 -> 3x4
        self.conv2 = nn.Conv2d(192, 192, kernel_size=2, padding=0)              # 3x4 -> 2x3

        self.fc = nn.Linear(192 * 2 * 3, 192)

        self.policy_fc = nn.Linear(192, 192)
        self.policy_out = nn.Linear(192, 7)

        # Kept because checkpoint loading uses strict=True, do not remove !
        self.value_fc = nn.Linear(192, 192)
        self.value_out = nn.Linear(192, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv_mid(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        pol = F.relu(self.policy_fc(x))
        pol = self.policy_out(pol)            # (B, 7)

        val = F.relu(self.value_fc(x))
        val = self.value_out(val).squeeze(-1) # (B,)

        return pol, val


def _find_model_path():
    # submission runtime (tar extracted here)
    p = f"/kaggle_simulations/agent/{MODEL_FILE}"
    if os.path.exists(p): return p

    raise FileNotFoundError("Model not found in agent dir, or CWD.")


def _load_model_once() -> nn.Module:
    global _MODEL
    if _MODEL is not None: return _MODEL

    ckpt_path = _find_model_path()
    ckpt = torch.load(ckpt_path, map_location=_DEVICE)

    if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt):
        raise RuntimeError("Unexpected checkpoint format: expected dict with 'model_state_dict'")

    model = CNet192(in_channels=1).to(_DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    _MODEL = model
    return _MODEL


# ---------- low-level board helpers ----------
def _lowest_empty_row(grid: np.ndarray, col: int) -> int:
    for r in range(5, -1, -1):
        if grid[r, col] == 0:
            return r
    return -1


def _has_four_from(grid: np.ndarray, row: int, col: int, token: int) -> bool:
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


def _is_winning_drop(pov: np.ndarray, col: int, token: int) -> bool:
    r = _lowest_empty_row(pov, col)
    if r < 0: return False

    old = pov[r, col]
    pov[r, col] = token
    win = _has_four_from(pov, r, col, token)
    pov[r, col] = old
    return win


def _legal_cols_from_pov(pov: np.ndarray):
    return [c for c in range(7) if pov[0, c] == 0]


def _generate_non_losing_moves(pov: np.ndarray):
    """
    Return tactically safe moves for the side to move (+1).

    Logic:
    1) Check whether opponent (-1) already has immediate wins now.
       - If there are 2+, there is no true non-losing move, you are just fucked
       - If there is exactly 1, we are forced to block it.
    2) Otherwise, keep only moves that do not give opponent an immediate
       winning reply after our move.
    """
    legal = _legal_cols_from_pov(pov)
    if not legal: return []

    opp_wins_now = [c for c in _CENTER_ORDER if pov[0, c] == 0 and _is_winning_drop(pov, c, -1)]

    if len(opp_wins_now) >= 2:
        return []

    if len(opp_wins_now) == 1:
        forced = opp_wins_now[0]
        return [forced] if forced in legal else []

    good = []
    for c in _CENTER_ORDER:
        if pov[0, c] != 0:
            continue

        r = _lowest_empty_row(pov, c)
        pov[r, c] = +1

        opp_has_reply_win = False
        for oc in _CENTER_ORDER:
            if pov[0, oc] == 0 and _is_winning_drop(pov, oc, -1):
                opp_has_reply_win = True
                break

        pov[r, c] = 0

        if not opp_has_reply_win:
            good.append(c)

    return good


def _infer_logits(model: nn.Module, pov: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(pov.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        logits, _ = model(x)

    logits = logits[0].detach().cpu().numpy().astype(np.float32)
    logits = np.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    return logits


# ---------- Kaggle agent ----------
def agent(obs, config):
    model = _load_model_once()

    mark = int(obs["mark"]) if isinstance(obs, dict) else int(obs.mark)
    flat = obs["board"] if isinstance(obs, dict) else obs.board
    grid = np.asarray(flat, dtype=np.int8).reshape(6, 7)

    legal = [c for c in range(7) if grid[0, c] == 0]
    if not legal:
        return 0

    stones = int(np.count_nonzero(grid))
    if stones == 0 and _CENTER_COL in legal:
        return _CENTER_COL  # tiny opening book

    # POV scalar board: me (POV) = +1, opp = -1
    pov = np.zeros((6, 7), dtype=np.int8)
    pov[grid == mark] = +1
    pov[(grid != 0) & (grid != mark)] = -1

    # Win now.
    for c in _CENTER_ORDER:
        if c in legal and _is_winning_drop(pov, c, +1):
            return int(c)

    # Prefer tactically safe moves if any exist.
    non_losing = _generate_non_losing_moves(pov)

    # Forced move.
    if len(non_losing) == 1:
        return int(non_losing[0])

    # Candidate set:
    # - prefer non-losing moves if any exist
    # - otherwise fall back to all legal moves in already-lost positions
    candidates = non_losing if non_losing else [c for c in _CENTER_ORDER if c in legal]

    # Policy inference.
    logits = _infer_logits(model, pov)

    # Final selection:
    # PPO logits first, center-distance and column index as deterministic tie-breaks.
    ordered = sorted(
        candidates,
        key=lambda c: (
            float(logits[c]),
            -abs(c - _CENTER_COL),
            -c,
        ),
        reverse=True,
    )

    return int(ordered[0])