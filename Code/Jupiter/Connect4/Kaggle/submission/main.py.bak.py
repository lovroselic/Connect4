# kaggle: main.py
# tar -czf submit.tar.gz -C submission main.py PPO_14.pt
# tar -czf submit.tar.gz -C submission main.py PPO_408.pt
# tar -czf submit.tar.gz -C submission main.py PPO_404.pt
# tar -czf submit.tar.gz -C submission main.py PPO_735.pt

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

# MODEL_FILE = "PPO_404.pt"
# submission name PPO_704
MODEL_FILE = "PPO_735.pt"

# -------------------------------------------------------------------------
# Inference / tactical wrapper knobs
# -------------------------------------------------------------------------

# Symmetry-aware inference:
# Evaluate both the original POV board and its horizontal mirror, then flip the
# mirrored logits back and average them. This is the policy-agent analogue of
# "symmetry-aware TT" in search engines.
_USE_MIRROR_TTA = True

# Threat-aware tie-breaker:
# We still let the PPO logits drive the final choice, but when several moves are
# close / legal / tactically safe, we prefer moves that create immediate threats.
_USE_THREAT_TIEBREAK = True


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
    if r < 0:
        return False

    old = pov[r, col]
    pov[r, col] = token
    win = _has_four_from(pov, r, col, token)
    pov[r, col] = old
    return win


def _legal_cols_from_pov(pov: np.ndarray):
    return [c for c in range(7) if pov[0, c] == 0]


def _count_immediate_wins(pov: np.ndarray, token: int) -> int:
    """
    Count how many columns are immediate winning drops for 'token' on the current
    POV board. We scan in center-first order because that is also our preferred
    move geometry everywhere else.
    """
    cnt = 0
    for c in _CENTER_ORDER:
        if pov[0, c] == 0 and _is_winning_drop(pov, c, token):
            cnt += 1
    return cnt


def _generate_non_losing_moves(pov: np.ndarray):
    """
    Return the set of tactically safe moves for the side to move (+1).

    This is the PPO-agent equivalent of the non-losing generator used in the
    bitboard search agent.

    Logic:
    1) Detect opponent immediate wins *in the current position*.
       - If opponent has 2+ immediate wins already, the position is effectively
         lost; there is no true non-losing move, so return [].
       - If opponent has exactly 1 immediate win, we are forced to block it, so
         return only that blocking column.
    2) Otherwise, keep only moves that do NOT hand the opponent any immediate
       winning reply after our move.

    This replaces the old "handover" and "double-threat" guard family with one
    cleaner tactical filter.
    """
    legal = _legal_cols_from_pov(pov)
    if not legal:
        return []

    opp_wins_now = [c for c in _CENTER_ORDER if pov[0, c] == 0 and _is_winning_drop(pov, c, -1)]

    if len(opp_wins_now) >= 2:
        # Already busted: opponent has multiple direct wins now.
        return []

    if len(opp_wins_now) == 1:
        # Forced block.
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


def _threat_tiebreak_tuple(pov: np.ndarray, col: int):
    """
    Build a tactical tie-break tuple *after* playing move 'col' for us (+1).

    This is not a full board evaluation. It is only a ranking signal used when
    the PPO logits need tactical help among already-acceptable candidates.

    Returned tuple fields, from most important to least:
    - fork_flag:      1 if our move creates >=2 immediate wins next turn
    - my_threats:     how many immediate wins we create for our next move
    - neg_opp_threats: negative opponent immediate-win count after our move
                       (larger is better, so 0 beats -1)
    - neg_center_dist: prefer center as final geometric tie-break
    - neg_col:        stable deterministic tie-break (smaller column wins)

    In sorted(reverse=True) order, this means:
    stronger forcing moves > more threats > fewer opponent threats > center > left.
    """
    r = _lowest_empty_row(pov, col)
    if r < 0:
        return (-1, -1, -99, -99, -99)

    pov[r, col] = +1
    my_threats = _count_immediate_wins(pov, +1)
    opp_threats = _count_immediate_wins(pov, -1)
    pov[r, col] = 0

    fork_flag = 1 if my_threats >= 2 else 0
    neg_opp_threats = -opp_threats
    neg_center_dist = -abs(col - _CENTER_COL)
    neg_col = -col

    return (fork_flag, my_threats, neg_opp_threats, neg_center_dist, neg_col)


def _infer_logits(model: nn.Module, pov: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(pov.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        logits, _ = model(x)
    logits = logits[0].detach().cpu().numpy().astype(np.float32)
    logits = np.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    return logits


def _infer_logits_symmetry_aware(model: nn.Module, pov: np.ndarray) -> np.ndarray:
    """
    Symmetry-aware inference:
    - run the model on the original POV board
    - run it on the horizontally mirrored POV board
    - flip mirrored logits back
    - average

    This reduces arbitrary left/right asymmetry in a horizontally symmetric game.
    It is the policy wrapper analogue of symmetry-aware TT reuse in the search agent.
    """
    logits = _infer_logits(model, pov)

    if not _USE_MIRROR_TTA:
        return logits

    pov_m = pov[:, ::-1].copy()
    logits_m = _infer_logits(model, pov_m)[::-1].copy()

    return 0.5 * (logits + logits_m)


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

    # POV scalar board: me=+1, opp=-1
    pov = np.zeros((6, 7), dtype=np.int8)
    pov[grid == mark] = +1
    pov[(grid != 0) & (grid != mark)] = -1

    # ------------------------------------------------------------------
    # 1) Tactical overrides before consulting the model
    # ------------------------------------------------------------------

    # Win now.
    for c in _CENTER_ORDER:
        if c in legal and _is_winning_drop(pov, c, +1):
            return int(c)

    # Generate non-losing moves. This automatically handles:
    # - forced single blocks
    # - rejection of moves that hand opponent an immediate win
    non_losing = _generate_non_losing_moves(pov)

    # If exactly one non-losing move exists, it is forced.
    if len(non_losing) == 1:
        return int(non_losing[0])

    # Candidate set:
    # - prefer non-losing moves if any exist
    # - otherwise fall back to all legal moves in already-lost positions
    candidates = non_losing if non_losing else [c for c in _CENTER_ORDER if c in legal]

    # ------------------------------------------------------------------
    # 2) Policy inference with symmetry-aware averaging
    # ------------------------------------------------------------------
    logits = _infer_logits_symmetry_aware(model, pov)

    # ------------------------------------------------------------------
    # 3) Final move selection
    #
    # Primary driver:
    #   PPO policy logits
    #
    # Secondary tie-break:
    #   tactical threat tuple after the move
    #
    # This keeps the neural policy in charge while giving it cleaner tactical
    # footing among safe moves.
    # ------------------------------------------------------------------
    if _USE_THREAT_TIEBREAK:
        ordered = sorted(
            candidates,
            key=lambda c: (
                float(logits[c]),
                *_threat_tiebreak_tuple(pov, c),
            ),
            reverse=True,
        )
    else:
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