# c4_move_data_generator.py
"""
Connect-4 move-data generator (Excel-first).

Produces per-move rows (one row per action taken) that can be used for:
- seeding replay buffers
- supervised / distillation pretraining (store state BEFORE the move + action)
- analytics (reward distributions, ply distributions, etc.)

Key design choices:
- Fast legality checks use bitboard mask & TOP_MASK[c]
- Optional "perfect" vs "noisy" move selection
- Optional "teacher policy probs" stored as 7 columns (uniform among best-scored legal moves)

Expected external API (from your project):
- C4.connect4_env.Connect4Env with:
    - reset() -> obs
    - step(action) -> (obs, reward, done) or (obs, reward, done, info)
    - board: numpy array shape (6,7), values in {0,1,2} or {0,1,-1}, top row index 0
    - current_player: int in {1,2} (or possibly {1,-1}; we normalize in lookahead)
- C4.numba_connect4_lookahead.Connect4Lookahead (your Numba bitboard lookahead)
"""

from __future__ import annotations

from typing import Dict, List, Iterable, Any, Optional, Tuple
from pathlib import Path
import random

import numpy as np
import pandas as pd

from C4.connect4_env import Connect4Env
from C4.fast_connect4_lookahead import Connect4Lookahead


# --------------------------- Row / DF helpers ---------------------------

def _is_random_spec(x: Any) -> bool:
    if isinstance(x, str):
        s = x.strip().upper()
        return s == "R" or s == "RANDOM" or s.startswith("R_")
    return False


def _legal_cols_from_board_toprow(board: np.ndarray) -> np.ndarray:
    """
    Legal columns = columns where top cell is empty (0).
    Works with boards in {0,1,2} or {0,1,-1} as long as empty is 0.
    """
    top = board[0]
    return np.flatnonzero(top == 0)


def _uniform_legal_probs(legal_cols: np.ndarray, n_actions: int = 7) -> np.ndarray:
    probs = np.zeros(n_actions, dtype=np.float64)
    if legal_cols is None or len(legal_cols) == 0:
        return probs
    probs[legal_cols] = 1.0 / float(len(legal_cols))
    return probs


def choose_move_mixed(
    env: Connect4Env,
    la: Connect4Lookahead,
    spec: int | str,
    noise_p: float,
    force_loss_p: float,
    rng: random.Random,
    teacher_mode: str = "scores",
    return_policy_probs: bool = False,
) -> Tuple[int, Optional[np.ndarray]]:
    """
    spec:
      - int depth => lookahead teacher (optionally noisy)
      - "R"/"Random" => uniform random legal move

    Returns:
      action, teacher_probs (optional)
    """
    # --- Random agent ---
    if _is_random_spec(spec):
        legal_cols = _legal_cols_from_board_toprow(env.board)
        if len(legal_cols) == 0:
            # defensive fallback
            a = 3
            probs = None
        else:
            a = int(rng.choice(legal_cols))
            probs = _uniform_legal_probs(legal_cols) if return_policy_probs else None
        return a, probs

    # --- Lookahead agent (your existing logic) ---
    depth = int(spec)
    a, probs = choose_move_noisy(
        env, la, depth=depth,
        noise_p=noise_p, force_loss_p=force_loss_p,
        rng=rng, teacher_mode=teacher_mode,
    )
    if not return_policy_probs:
        probs = None
    return int(a), probs


def _board_to_cells(board: np.ndarray) -> Dict[str, int]:
    """
    Flatten board into dict columns named "r-c" for r in [0..5], c in [0..6],
    where r=0 is the TOP row (same as board[0, :]).
    """
    rows, cols = board.shape
    out: Dict[str, int] = {}
    for r in range(rows):
        for c in range(cols):
            out[f"{r}-{c}"] = int(board[r, c])
    return out


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of row dicts into a stable, column-ordered DataFrame.

    We preserve your original column order:
      label, reward, game, ply, <optional extras>, 0-0..5-6
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)

    # Board columns in your legacy order
    board_cols = [f"{r}-{c}" for r in range(6) for c in range(7)]

    head_cols = [c for c in ["label", "reward", "game", "ply"] if c in df.columns]
    # keep any extra (action/player/done/probs/scores/etc) but not board, in a stable order
    extra_cols = [c for c in df.columns if c not in set(head_cols + board_cols)]
    df = df[head_cols + sorted(extra_cols) + [c for c in board_cols if c in df.columns]]

    return df


def upsert_excel(df_new: pd.DataFrame, xlsx_path: str | Path, sheet_name: str = "Sheet1") -> pd.DataFrame:
    """
    Append df_new into xlsx_path/sheet_name, drop exact duplicates, and rewrite the sheet.

    This is intentionally simple:
    - duplicates are removed by full-row equality
    - the sheet is rewritten (replace), so it stays clean
    """
    xlsx_path = Path(xlsx_path)
    if df_new is None or len(df_new) == 0:
        return df_new

    if xlsx_path.exists():
        try:
            df_old = pd.read_excel(xlsx_path, sheet_name=sheet_name)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_all = df_new.copy()
        df_all = df_all.drop_duplicates().reset_index(drop=True)

        with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            df_all.to_excel(w, sheet_name=sheet_name, index=False)
        return df_all
    else:
        df_new = df_new.drop_duplicates().reset_index(drop=True)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="w") as w:
            df_new.to_excel(w, sheet_name=sheet_name, index=False)
        return df_new


# --------------------------- Teacher policy helpers ---------------------------

def _uniform_among_best(scores: np.ndarray) -> np.ndarray:
    """
    Given scores[7] where illegal columns are -inf,
    return probs[7] uniform over the best-scored legal actions.
    """
    probs = np.zeros_like(scores, dtype=np.float64)
    best = np.nanmax(scores)
    if not np.isfinite(best):
        return probs
    idx = np.flatnonzero(scores == best)
    if idx.size == 0:
        return probs
    probs[idx] = 1.0 / float(idx.size)
    return probs


def _sample_from_probs(probs: np.ndarray, rng: random.Random) -> int:
    """
    Sample action from probs[7] using Python RNG (deterministic if seeded).
    probs need not sum to 1 exactly.
    """
    p = probs.astype(np.float64, copy=False)
    s = float(p.sum())
    if s <= 0.0:
        return 0
    r = rng.random() * s
    acc = 0.0
    for a in range(p.shape[0]):
        acc += float(p[a])
        if r <= acc:
            return int(a)
    return int(p.shape[0] - 1)


# --------------------------- Noise / safety helpers ---------------------------

def _safe_unsafe_alternatives(
    la: Connect4Lookahead,
    board: np.ndarray,
    player: int,
    exclude: Optional[Iterable[int]] = None,
) -> Tuple[List[int], List[int]]:
    """
    Split legal actions into:
      - safe: does NOT give opponent an immediate win next ply
      - unsafe: does give opponent an immediate win next ply

    Uses the Numba core bitboard routines (fast).
    """
    exclude_set = set(exclude) if exclude is not None else set()

    p1, p2, mask = la._parse_board_bitboards(board)
    me_mark = la._p(player)
    pos = p1 if me_mark == 1 else p2

    N = la._N
    safe: List[int] = []
    unsafe: List[int] = []

    for c in la._CENTER_ORDER:
        if c in exclude_set:
            continue
        if (mask & la.TOP_MASK[c]) != 0:
            continue

        mv = int(la._play_bit_py(int(mask), int(c)))
        nm = int(mask | mv)
        opp_pos = nm ^ (pos | mv)  # position for the next player under Pons convention

        # If opponent has ANY immediate winning reply, this is unsafe
        opp_wins = int(N["count_immediate_wins_bits"](
            np.uint64(opp_pos), np.uint64(nm),
            N["CENTER_ORDER"], N["TOP_MASK"], N["BOTTOM_MASK"], N["COL_MASK"],
            np.int32(la.STRIDE)
        ))
        if opp_wins > 0:
            unsafe.append(int(c))
        else:
            safe.append(int(c))

    return safe, unsafe


def choose_move_noisy(
    env: Connect4Env,
    la: Connect4Lookahead,
    depth: int,
    noise_p: float,
    force_loss_p: float,
    rng: random.Random,
    teacher_mode: str = "scores",
) -> Tuple[int, Optional[np.ndarray]]:
    """
    Noisy teacher move:
      - compute teacher policy (uniform among best) and sample from it
      - with probability noise_p, sample from "safe alternatives" excluding all best moves
      - with probability force_loss_p (evaluated only if we are in noise), sample from unsafe moves
        (useful to seed 'punishment' experiences)

    Returns:
      action, teacher_probs (or None if teacher_mode != "scores")
    """
    teacher_probs = None

    if teacher_mode == "scores":
        scores = la.n_step_action_scores(env.board, env.current_player, depth=depth)
        teacher_probs = _uniform_among_best(scores)
        best_actions = set(np.flatnonzero(teacher_probs > 0.0).tolist())
        best_action = _sample_from_probs(teacher_probs, rng)
    else:
        # fast, deterministic best (center-first ordering)
        best_action = int(la.n_step_lookahead(env.board, env.current_player, depth=depth, tie_break="random"))
        best_actions = {best_action}

    # inject noise?
    if noise_p > 0.0 and rng.random() < float(noise_p):
        safe, unsafe = _safe_unsafe_alternatives(la, env.board, env.current_player, exclude=best_actions)

        if force_loss_p > 0.0 and rng.random() < float(force_loss_p) and len(unsafe) > 0:
            return int(rng.choice(unsafe)), teacher_probs

        if len(safe) > 0:
            return int(rng.choice(safe)), teacher_probs
        # if no safe alternatives exist, fall back to best
        return int(best_action), teacher_probs

    return int(best_action), teacher_probs


# --------------------------- Main generation ---------------------------

def _play_one_game_rows(
    lookA: int | str,
    lookB: int | str,
    label: str,
    game_index: int,
    seed: int = 666,
    CFG: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Play ONE game and return a list of ROW dicts, one per MOVE *we choose to record*.

    IMPORTANT (supervised safety):
      - If the current mover is Random ("R"/"Random"/"R_*"), we still PLAY the move
        to diversify positions, but we DO NOT EMIT a row for that move.
        (Random is poison for supervised labels.)

    lookA/lookB can be:
      - int depth (lookahead)
      - "R" / "Random" (uniform random legal move)

    CFG supports:
      - noiseA, noiseB: float, probability of deviating from teacher best on each move (lookahead only)
      - forceLoss: float, probability to pick an unsafe move when deviating (lookahead only)
      - teacher_mode: "scores" (uniform among best) or "fast" (deterministic)
      - state_mode: "after" (default) or "before"
      - store_action: bool (default True)
      - store_player: bool (default True)
      - store_done: bool (default False)
      - store_policy_probs: bool (default False)
          If True (and move is recorded):
            - lookahead + teacher_mode="scores": p0..p6 = uniform among best
            - (Random mover rows are never stored, so probs never stored for Random)
    """
    CFG = CFG or {}
    rng = random.Random(seed)

    noiseA = float(CFG.get("noiseA", 0.0))
    noiseB = float(CFG.get("noiseB", 0.0))
    forceLoss = float(CFG.get("forceLoss", 0.0))

    teacher_mode = str(CFG.get("teacher_mode", "scores")).lower()
    state_mode = str(CFG.get("state_mode", "after")).lower()

    store_action = bool(CFG.get("store_action", True))
    store_player = bool(CFG.get("store_player", True))
    store_done = bool(CFG.get("store_done", False))
    store_probs = bool(CFG.get("store_policy_probs", False))

    env = Connect4Env()
    la = Connect4Lookahead()

    # diversity overrides (your existing knobs)
    #la.C4_CENTER_BONUS = 7.5           # 5-10+
    #la.C4_DEFENSIVE = 1.20             # 1 +-
    #la.C4_FLOATING_NEAR = 0.70         # 0.6-0.8
    #la.C4_FLOATING_FAR = 0.30          # 0.1-0.3
    #env.CENTER_REWARD = 0.01           # 0.01 - 0.2 
    #env.THREAT3_VALUE = 12             # 10 - 20
    #env.STEP_PENALTY = 0.5               # 0 - 2

    _ = env.reset()

    rows: List[Dict[str, Any]] = []
    ply = 0

    while True:
        ply += 1
        player = int(env.current_player)

        spec = (lookA if player == 1 else lookB)
        noise_p = noiseA if player == 1 else noiseB

        is_random_turn = _is_random_spec(spec)

        # Only copy "before" state if we're going to record and we need it
        board_before = None
        if (not is_random_turn) and (state_mode == "before"):
            board_before = np.array(env.board, copy=True)

        # Choose action. If Random turn, we do not request policy probs at all.
        action, teacher_probs = choose_move_mixed(
            env,
            la,
            spec=spec,
            noise_p=noise_p,
            force_loss_p=forceLoss,
            rng=rng,
            teacher_mode=teacher_mode,
            return_policy_probs=(store_probs and (not is_random_turn)),
        )

        step_out = env.step(int(action))
        if isinstance(step_out, (tuple, list)):
            if len(step_out) >= 3:
                reward, done = step_out[1], step_out[2]
            else:
                reward, done = 0.0, False
        else:
            reward, done = 0.0, False

        # Supervised safety: NEVER emit a row when Random was the mover.
        if not is_random_turn:
            if state_mode == "after":
                board_store = np.array(env.board, copy=True)
            else:
                board_store = board_before  # already copied

            row: Dict[str, Any] = {
                "label": str(label),
                "reward": float(reward),
                "game": int(game_index),
                "ply": int(ply),
                **_board_to_cells(board_store),
            }

            if store_player:
                row["player"] = int(player)
            if store_action:
                row["action"] = int(action)
            if store_done:
                row["done"] = bool(done)

            if store_probs and teacher_probs is not None:
                for a in range(7):
                    row[f"p{a}"] = float(teacher_probs[a])

            rows.append(row)

        if bool(done):
            break

    return rows



def generate_dataset(
    PLAYS: Dict[str, Dict[str, Any]],
    seed: int = 666,
    CFG: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate a dataset from a PLAYS dict like:
      PLAYS = {
        "L5L7":     {"A": 5,   "B": 7,   "games": 10},
        "L7_vs_R":  {"A": 7,   "B": "R", "games": 50},
        "R_vs_L7":  {"A": "R", "B": 7,   "games": 50},
      }

    Returns:
      list of row dicts (moves across all games/labels)
    """
    CFG = CFG or {}

    all_rows: List[Dict[str, Any]] = []
    base_seed = int(seed)

    for label, spec in PLAYS.items():
        lookA = spec.get("A", 1)
        lookB = spec.get("B", 1)
        games = int(spec.get("games", 1))

        # normalize A/B: keep "R" as-is, cast others to int
        if not _is_random_spec(lookA):
            lookA = int(lookA)
        if not _is_random_spec(lookB):
            lookB = int(lookB)

        for g in range(games):
            s = base_seed + (hash(label) & 0xFFFF) + g * 9973
            rows = _play_one_game_rows(
                lookA=lookA,
                lookB=lookB,
                label=str(label),
                game_index=g,
                seed=s,
                CFG=CFG,
            )
            all_rows.extend(rows)

    return all_rows
