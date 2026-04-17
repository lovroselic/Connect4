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

# Mirror test-time augmentation:
# Evaluate original board and mirrored board, flip mirrored logits back, average.
_USE_MIRROR_TTA = True

# Threat-aware tie-breaker layered on top of policy logits.
_USE_THREAT_TIEBREAK = True

# Tiny hybrid lookahead:
# - shortlist this many root candidates from policy ordering
# - for each, inspect this many opponent replies
_USE_VALUE_LOOKAHEAD = True
_LOOKAHEAD_TOP_K = 3
_OPP_REPLY_TOP_K = 2

# Large tactical sentinel so immediate tactical truths dominate value-head noise.
_TACTICAL_WIN_SCORE = 1_000_000.0


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


def _play_move_copy(pov: np.ndarray, col: int, token: int):
    """
    Return a copied board after dropping 'token' in 'col', plus the landing row.
    If illegal, returns (None, -1).
    """
    r = _lowest_empty_row(pov, col)
    if r < 0:
        return None, -1
    nxt = pov.copy()
    nxt[r, col] = token
    return nxt, r


def _legal_cols_from_pov(pov: np.ndarray):
    return [c for c in range(7) if pov[0, c] == 0]


def _count_immediate_wins(pov: np.ndarray, token: int) -> int:
    """
    Count how many immediate winning drops exist for 'token' in the current
    scalar POV board.
    """
    cnt = 0
    for c in _CENTER_ORDER:
        if pov[0, c] == 0 and _is_winning_drop(pov, c, token):
            cnt += 1
    return cnt


def _generate_non_losing_moves(pov: np.ndarray):
    """
    Return tactically safe moves for the side to move (+1).

    Logic:
    1) If the opponent (-1) already has 2+ immediate wins in the current
       position, there is no true non-losing move -> return [].
    2) If the opponent has exactly 1 immediate win, we must block it -> return
       only that forced blocking move.
    3) Otherwise, keep only moves that do NOT hand the opponent any immediate
       winning reply after our move.

    This replaces the older pile of overlapping handover/double-threat guards
    with one cleaner tactical filter.
    """
    legal = _legal_cols_from_pov(pov)
    if not legal:
        return []

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

        nxt, _ = _play_move_copy(pov, c, +1)

        opp_has_reply_win = False
        for oc in _CENTER_ORDER:
            if nxt[0, oc] == 0 and _is_winning_drop(nxt, oc, -1):
                opp_has_reply_win = True
                break

        if not opp_has_reply_win:
            good.append(c)

    return good


def _candidate_set_with_tactics(pov: np.ndarray):
    """
    Candidate moves for the side to move (+1), ordered by tactical priority:

    1) winning moves now
    2) forced / non-losing moves
    3) all legal moves if already lost

    This helper is used both at root and inside the tiny lookahead reply model.
    """
    legal = [c for c in _CENTER_ORDER if pov[0, c] == 0]
    if not legal:
        return []

    wins = [c for c in legal if _is_winning_drop(pov, c, +1)]
    if wins:
        return wins

    non_losing = _generate_non_losing_moves(pov)
    if non_losing:
        return non_losing

    return legal


def _threat_tiebreak_tuple(pov: np.ndarray, col: int):
    """
    Tactical tie-break tuple after playing 'col' for the side to move (+1).

    This is NOT a full evaluation, only a move-ranking signal.

    Returned fields, in descending importance:
    - fork_flag:        1 if move creates >=2 immediate wins next turn
    - my_threats:       number of our immediate wins after the move
    - neg_opp_threats:  fewer opponent threats is better
    - neg_center_dist:  prefer center
    - neg_col:          deterministic final tie-break

    Used with reverse=True, so bigger tuple is better.
    """
    nxt, r = _play_move_copy(pov, col, +1)
    if nxt is None or r < 0:
        return (-1, -1, -99, -99, -99)

    my_threats = _count_immediate_wins(nxt, +1)
    opp_threats = _count_immediate_wins(nxt, -1)

    fork_flag = 1 if my_threats >= 2 else 0
    neg_opp_threats = -opp_threats
    neg_center_dist = -abs(col - _CENTER_COL)
    neg_col = -col

    return (fork_flag, my_threats, neg_opp_threats, neg_center_dist, neg_col)


def _infer_logits_and_value(model: nn.Module, pov: np.ndarray):
    x = torch.from_numpy(pov.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        logits, value = model(x)

    logits = logits[0].detach().cpu().numpy().astype(np.float32)
    logits = np.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

    value = float(value[0].detach().cpu().item())
    if not np.isfinite(value):
        value = 0.0

    return logits, value


def _infer_logits_value_symmetry_aware(model: nn.Module, pov: np.ndarray):
    """
    Symmetry-aware inference:
    - evaluate original POV board
    - evaluate horizontally mirrored POV board
    - flip mirrored logits back
    - average both logits and values

    This is the policy/value analogue of symmetry-aware TT reuse in a searcher.
    """
    logits, value = _infer_logits_and_value(model, pov)

    if not _USE_MIRROR_TTA:
        return logits, value

    pov_m = pov[:, ::-1].copy()
    logits_m, value_m = _infer_logits_and_value(model, pov_m)

    logits_m = logits_m[::-1].copy()
    logits = 0.5 * (logits + logits_m)
    value = 0.5 * (value + value_m)

    return logits, value


def _order_policy_candidates(candidates, logits: np.ndarray, pov: np.ndarray):
    """
    Order candidate moves with PPO logits as the main signal and threat-based
    tie-breaking as a tactical assistant.
    """
    if _USE_THREAT_TIEBREAK:
        return sorted(
            candidates,
            key=lambda c: (
                float(logits[c]),
                *_threat_tiebreak_tuple(pov, c),
            ),
            reverse=True,
        )

    return sorted(
        candidates,
        key=lambda c: (
            float(logits[c]),
            -abs(c - _CENTER_COL),
            -c,
        ),
        reverse=True,
    )


def _score_root_move_with_tiny_lookahead(model: nn.Module, pov: np.ndarray, root_col: int) -> float:
    """
    Tiny 1-ply hybrid lookahead for one root move.

    Flow:
    1) play our candidate move
    2) if that wins immediately, return huge positive
    3) let opponent choose among a very small shortlist of plausible replies:
       - their tactical candidate set
       - ordered by their own PPO policy + threat tie-break
       - restricted to top _OPP_REPLY_TOP_K replies
    4) for each reply, score resulting position from OUR POV using the value head
    5) return the worst reply score (min over opponent replies)

    This is deliberately tiny:
    - no recursive search
    - no TT
    - just enough lookahead to stop obviously optimistic policy picks
    """
    after_my, row_my = _play_move_copy(pov, root_col, +1)
    if after_my is None:
        return -_TACTICAL_WIN_SCORE

    if _has_four_from(after_my, row_my, root_col, +1):
        return _TACTICAL_WIN_SCORE

    if np.all(after_my[0] != 0):
        return 0.0  # draw after our move

    # Opponent to move POV is just sign-flipped.
    opp_pov = -after_my

    opp_candidates = _candidate_set_with_tactics(opp_pov)
    if not opp_candidates:
        return 0.0

    opp_logits, _ = _infer_logits_value_symmetry_aware(model, opp_pov)
    opp_ordered = _order_policy_candidates(opp_candidates, opp_logits, opp_pov)
    opp_ordered = opp_ordered[:min(_OPP_REPLY_TOP_K, len(opp_ordered))]

    worst_reply_value = float("inf")

    for oc in opp_ordered:
        after_opp, row_opp = _play_move_copy(opp_pov, oc, +1)
        if after_opp is None:
            continue

        # Opponent wins immediately after our root move -> disastrous root move.
        if _has_four_from(after_opp, row_opp, oc, +1):
            reply_value = -_TACTICAL_WIN_SCORE
        else:
            # Convert back to our POV: now it is our turn again.
            our_pov_next = -after_opp

            if np.all(our_pov_next[0] != 0):
                reply_value = 0.0  # draw

            else:
                # If we have an immediate tactical win after their reply, prefer it
                # strongly rather than asking the value head to guess.
                our_win_now = False
                for cc in _CENTER_ORDER:
                    if our_pov_next[0, cc] == 0 and _is_winning_drop(our_pov_next, cc, +1):
                        our_win_now = True
                        break

                if our_win_now:
                    reply_value = _TACTICAL_WIN_SCORE
                else:
                    _, reply_value = _infer_logits_value_symmetry_aware(model, our_pov_next)

        if reply_value < worst_reply_value:
            worst_reply_value = reply_value

    if worst_reply_value == float("inf"):
        return 0.0

    return float(worst_reply_value)


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
    # 1) Fast tactical layer
    # ------------------------------------------------------------------

    # Win now.
    for c in _CENTER_ORDER:
        if c in legal and _is_winning_drop(pov, c, +1):
            return int(c)

    # Candidate set with tactical filtering.
    candidates = _candidate_set_with_tactics(pov)
    if not candidates:
        candidates = [c for c in _CENTER_ORDER if c in legal]

    # If there is exactly one tactically acceptable move, just play it.
    if len(candidates) == 1:
        return int(candidates[0])

    # ------------------------------------------------------------------
    # 2) Root policy inference
    # ------------------------------------------------------------------
    root_logits, _ = _infer_logits_value_symmetry_aware(model, pov)
    ordered = _order_policy_candidates(candidates, root_logits, pov)

    # ------------------------------------------------------------------
    # 3) Tiny 1-ply hybrid lookahead for top root candidates
    # ------------------------------------------------------------------
    if _USE_VALUE_LOOKAHEAD and len(ordered) > 1:
        shortlist = ordered[:min(_LOOKAHEAD_TOP_K, len(ordered))]

        best_col = shortlist[0]
        best_key = None

        for c in shortlist:
            lookahead_value = _score_root_move_with_tiny_lookahead(model, pov, c)

            # Final ranking:
            # 1) tiny worst-reply value
            # 2) original root policy logit
            # 3) tactical tie-break tuple
            #
            # This keeps PPO in charge of candidate generation, but lets the
            # value head veto overly optimistic moves once we peek one reply ahead.
            key = (
                float(lookahead_value),
                float(root_logits[c]),
                *_threat_tiebreak_tuple(pov, c),
            )

            if best_key is None or key > best_key:
                best_key = key
                best_col = c

        return int(best_col)

    # Fallback: pure policy+tactics ordering
    return int(ordered[0])