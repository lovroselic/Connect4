# kaggle: main.py
# tar -czf submit.tar.gz -C submission main.py PPO_14.pt
# tar -czf submit.tar.gz -C submission main.py PPO_408.pt
# tar -czf submit.tar.gz -C submission main.py PPO_404.pt

import os
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_DEVICE = torch.device("cpu")
_MODEL: Optional[nn.Module] = None
_CENTER_COL = 3
#MODEL_FILE = "PPO_404.pt"
#submission name PPO_14_DT
MODEL_FILE = "PPO_14.pt"


# ---------- CNet192 (mid always ON) ----------
class CNet192(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 192, kernel_size=4, padding=0)  # 6x7 -> 3x4
        self.conv_mid = nn.Conv2d(192, 192, kernel_size=3, padding=1)      # 3x4 -> 3x4
        self.conv2 = nn.Conv2d(192, 192, kernel_size=2, padding=0)         # 3x4 -> 2x3

        self.fc = nn.Linear(192 * 2 * 3, 192)

        self.policy_fc = nn.Linear(192, 192)
        self.policy_out = nn.Linear(192, 7)

        self.value_fc = nn.Linear(192, 192)
        self.value_out = nn.Linear(192, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv_mid(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        pol = F.relu(self.policy_fc(x))
        pol = self.policy_out(pol)            # (B,7)

        val = F.relu(self.value_fc(x))
        val = self.value_out(val).squeeze(-1) # (B,)

        return pol, val


def _find_model_path():

    # submission runtime (tar extracted here)
    p = f"/kaggle_simulations/agent/{MODEL_FILE}"
    if os.path.exists(p):
        return p

    # fallback: current working directory
    p = MODEL_FILE
    if os.path.exists(p):
        return p

    raise FileNotFoundError("Model not found in agent dir, or CWD.")


def _load_model_once() -> nn.Module:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    ckpt_path = _find_model_path()
    ckpt = torch.load(ckpt_path, map_location=_DEVICE)


    if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt):
        raise RuntimeError("Unexpected checkpoint format: expected dict with 'model_state_dict'")

    m = CNet192(in_channels=1).to(_DEVICE)
    m.load_state_dict(ckpt["model_state_dict"], strict=True)
    m.eval()

    _MODEL = m
    return _MODEL


# ---------- tiny tactics ----------
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
    if r < 0:
        return False
    old = pov[r, col]
    pov[r, col] = token
    win = _has_four_from(pov, r, col, token)
    pov[r, col] = old
    return win

# def _argmax_center_tiebreak(vals: np.ndarray, legal: List[int]) -> int:
#     best_c = legal[0]
#     best_v = float(vals[best_c])
#     best_d = abs(best_c - _CENTER_COL)
#     for c in legal[1:]:
#         v = float(vals[c])
#         if v > best_v:
#             best_v = v
#             best_c = c
#             best_d = abs(c - _CENTER_COL)
#         elif v == best_v:
#             d = abs(c - _CENTER_COL)
#             if d < best_d or (d == best_d and c < best_c):
#                 best_c = c
#                 best_d = d
#     return int(best_c)

def _would_handover_win(pov: np.ndarray, col: int) -> bool:
    """
    True if playing 'col' (for +1) gives opponent (-1) any win-in-1 immediately after.
    Uses in-place move + revert on the POV board.
    """
    r = _lowest_empty_row(pov, col)
    if r < 0:
        return True  # illegal treated as bad

    # play our move
    pov[r, col] = +1

    # check if opponent now has a winning drop anywhere
    for oc in range(7):
        if pov[0, oc] == 0 and _is_winning_drop(pov, oc, -1):
            pov[r, col] = 0
            return True

    pov[r, col] = 0
    return False

def _would_allow_double_threat(pov: np.ndarray, col: int) -> bool:
    """
    True if after we play 'col' (+1), opponent (-1) has >=2 different winning drops.
    (Fork / double-threat next move)
    """
    r = _lowest_empty_row(pov, col)
    if r < 0:
        return True

    pov[r, col] = +1

    cnt = 0
    for oc in range(7):
        if pov[0, oc] == 0 and _is_winning_drop(pov, oc, -1):
            cnt += 1
            if cnt >= 2:
                pov[r, col] = 0
                return True

    pov[r, col] = 0
    return False


# ---------- Kaggle agent ----------
def agent(obs, config):
    
    model = _load_model_once()

    mark = int(obs["mark"]) if isinstance(obs, dict) else int(obs.mark)
    flat = obs["board"] if isinstance(obs, dict) else obs.board
    grid = np.asarray(flat, dtype=np.int8).reshape(6, 7)

    legal = [c for c in range(7) if grid[0, c] == 0]
    if not legal: return 0

    stones = int(np.count_nonzero(grid))
    if stones == 0 and _CENTER_COL in legal:
        return _CENTER_COL  # opening book

    # POV scalar board: me=+1, opp=-1
    pov = np.zeros((6, 7), dtype=np.int8)
    pov[grid == mark] = 1
    pov[(grid != 0) & (grid != mark)] = -1

    # win-now
    for c in legal:
        if _is_winning_drop(pov, c, +1):
            return int(c)

    # must-block (any opp win-in-1)
    blocks = [c for c in legal if _is_winning_drop(pov, c, -1)]
    if blocks:
        return int(sorted(blocks, key=lambda c: (abs(c - _CENTER_COL), c))[0])

    # policy argmax (masked)
    x = torch.from_numpy(pov.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    
    with torch.no_grad():  # portable replacement for inference_mode
        logits, _ = model(x)

    logits = logits[0].detach().cpu().numpy().astype(np.float32)
    logits = np.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    # --- handover filter & DT guard
    # order legal moves by logits desc
    ordered = sorted(
        legal,
        key=lambda c: (-float(logits[c]), abs(c - _CENTER_COL), c)
    )
    
    first_non_handover = None
    
    for c in ordered:
        if _would_handover_win(pov, c): continue
        if first_non_handover is None: first_non_handover = c  # best non-handover so far
        if not _would_allow_double_threat(pov, c): return int(c)
    
    if first_non_handover is not None: return int(first_non_handover)
    return int(ordered[0])

    


