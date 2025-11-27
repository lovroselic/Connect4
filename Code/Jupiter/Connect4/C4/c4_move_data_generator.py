# c4_move_data_generator.py
from __future__ import annotations

from typing import Dict, List, Iterable, Any
import random
import numpy as np
import pandas as pd

from C4.connect4_env import Connect4Env  
from C4.fast_connect4_lookahead import Connect4Lookahead  


def _play_one_game_rows(lookA: int, lookB: int, label: str, game_index: int, seed: int = 666, CFG: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Play ONE game; return a list of ROWS, one per MOVE, each row containing:
      {"label": label, "reward": immediate_step_reward, "game": game_index, "ply": ply(1-based), "0-0": ..., ..., "5-6": ...}
    'reward' is now the immediate mover-centric reward returned by env.step(action).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = Connect4Env()
    la  = Connect4Lookahead()
    _ = env.reset()

    done = False
    ply = 1
    rows: List[Dict[str, float]] = []

    CFG = CFG or {}  # <— important if caller passes None
    noiseA = CFG.get("noiseA", 0.0)
    noiseB = CFG.get("noiseB", 0.0)
    FL     = CFG.get("forceLoss", 0.0)

    while not done:
        player = env.current_player
        depth  = lookA if player == 1 else lookB
        p_mist = noiseA if player == 1 else noiseB

        if p_mist > 0:
            action = choose_move_noisy(env, la, player, depth, p_mistake=p_mist, FL=FL)
        else:
            action = la.n_step_lookahead(env.board, player, depth=depth)

        _, reward, done = env.step(action)

        row_dict: Dict[str, float] = {"label": label, "reward": float(reward), "game": int(game_index), "ply": int(ply)}
        for r in range(env.ROWS):
            for c in range(env.COLS):
                row_dict[f"{r}-{c}"] = int(env.board[r, c])

        rows.append(row_dict)
        ply += 1

    return rows




def generate_dataset(plays: Dict[str, Dict[str, int]], seed: int = 666, CFG: Dict[str, Any] = None) -> List[Dict[str, float]]:
    """
    Given a PLAYS dict like {"L3L1":{"A":3,"B":1,"games":N}, "L2":{"A":2,"B":2,"games":M}},
    return a FLAT list of PER-MOVE rows across all requested games.
    """
    all_rows: List[Dict[str, float]] = []
    rng = random.Random(seed) if seed is not None else random
    for label, cfg in plays.items():
        depthA = int(cfg["A"])
        depthB = int(cfg["B"])
        n      = int(cfg.get("games", 1))
        for g in range(n):
            s = None if seed is None else rng.randint(0, 2**31 - 1)
            all_rows.extend(_play_one_game_rows(depthA, depthB, label, game_index=g, seed=s, CFG=CFG))
    return all_rows


def records_to_dataframe(records: Iterable[Dict[str, float]]) -> pd.DataFrame:
    """
    Convert list of per-move rows to a DataFrame with stable columns:
      ["label","reward","game","ply","0-0","0-1",...,"5-6"]
    """
    # Pull sizes from env to avoid hard-coding 6x7
    env = Connect4Env()
    col_order = ["label", "reward", "game", "ply"] + [f"{r}-{c}" for r in range(env.ROWS) for c in range(env.COLS)]
    df = pd.DataFrame(records)
    for col in col_order:
        if col not in df.columns:
            if col in {"label"}:
                df[col] = ""
            elif col in {"reward"}:
                df[col] = 0.0
            elif col in {"game", "ply"}:
                df[col] = 0
            else:
                df[col] = 0.0
    df = df[col_order]
    return df


def upsert_excel(df_new: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Append to Excel (if exists), drop exact duplicates across all columns, rewrite.
    """
    try:
        existing = pd.read_excel(path)
        df_all = pd.concat([existing, df_new], ignore_index=True)
    except Exception:
        df_all = df_new.copy()
    df_all = df_all.drop_duplicates().reset_index(drop=True)
    df_all.to_excel(path, index=False)
    return df_all


def _safe_alternatives(env, la: Connect4Lookahead, player: int, FL: float):
    """
    Bitboard-aware "safe" moves:
      - legal if column top isn't occupied
      - SAFE if, after our move, opponent has zero immediate winning replies
      - If there are UNSAFE moves and random() <= FL, return a single random unsafe move
        (to intentionally seed some losing lines).
    """
    # Current position as bitboards
    p1, p2, mask = la._parse_board_bitboards(env.board)
    me_mark = la._p(player)               # +1 for player 1, -1 for player 2 / -1
    pos = p1 if me_mark == 1 else p2

    safe, unsafe = [], []

    # Short references to Numba-side arrays
    TOP_MASK    = la._N["TOP_MASK"]
    BOTTOM_MASK = la._N["BOTTOM_MASK"]
    COL_MASK    = la._N["COL_MASK"]
    CENTER_ORD  = la._N["CENTER_ORDER"]
    stride_i    = np.int32(la.STRIDE)

    for c in range(la.COLS):
        # legal move?
        if (mask & la.TOP_MASK[c]) != 0:
            continue

        # Make move in bitboard space
        mv = la._play_bit_py(mask, c)
        nm = mask | mv
        me_after = pos | mv
        opp_pos = nm ^ me_after  # opponent's stones after our move

        # Count opponent's immediate wins (center-first order for stability)
        opp_imm = la._N["count_immediate_wins_bits"](
            np.uint64(opp_pos), np.uint64(nm),
            CENTER_ORD, TOP_MASK, BOTTOM_MASK, COL_MASK, stride_i
        )

        if int(opp_imm) == 0:
            safe.append(c)
        else:
            unsafe.append(c)

    if unsafe and random.random() <= FL:
        return [random.choice(unsafe)]
    return safe

def choose_move_noisy(env, la: Connect4Lookahead, player: int, depth: int, p_mistake=0.15, prefer_offcenter=True, FL=0.0):
    # Clean best via lookahead
    best = la.n_step_lookahead(env.board, player, depth=depth)
    if random.random() >= p_mistake:
        return best

    # Safe alternatives (excluding the best)
    alts = [c for c in _safe_alternatives(env, la, player, FL) if c != best]
    if not alts:
        return best

    if prefer_offcenter:
        center = la.CENTER_COL
        alts.sort(key=lambda c: abs(center - c), reverse=True)
        alts = alts[:min(2, len(alts))]  # choose among the 1–2 most off-center

    return random.choice(alts)

