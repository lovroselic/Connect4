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
    _ = env.reset()  # state not used directly; board snapshot taken from env.board

    done = False
    ply = 1  # 1-based as before
    rows: List[Dict[str, float]] = []
    
    noiseA = CFG.get("noiseA", 0.0)
    noiseB = CFG.get("noiseB", 0.0)
    FL = CFG.get("forceLoss", 0.0)

    while not done:
        player = env.current_player
        depth  = lookA if player == 1 else lookB
        p_mist = noiseA if player == 1 else noiseB
        
        if p_mist > 0:
            action = choose_move_noisy(env, la, player, depth, p_mistake=p_mist, FL=FL)
        # normal action
        else: action = la.n_step_lookahead(env.board, player, depth=depth)

        _, reward, done = env.step(action)  # immediate mover-centric reward

        row_dict: Dict[str, float] = {
            "label": label,
            "reward": float(reward),   # <-- immediate step reward (was cumulative)
            "game": int(game_index),
            "ply": int(ply),
        }
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


def _safe_alternatives(env, la, player, FL):

    p = -1 if player in (-1, 2) else 1
    opp = -p

    la._load_from_numpy(env.board) # Load the top-based NumPy board ONCE into lookahead's internal state

    safe, unsafe = [], []

    for c in la._legal_moves():
        la._make_move(c, p)

        opp_immediate = la._count_immediate_wins(opp)
        if opp_immediate == 0: safe.append(c)
        else: unsafe.append(c)

        la._unmake_move(c)

    if unsafe and random.random() <= FL: return [random.choice(unsafe)]
    return safe


def choose_move_noisy(env, la, player, depth, p_mistake=0.15, prefer_offcenter=True, FL = 0):
    # clean best
    best = la.n_step_lookahead(env.board, player, depth=depth)
    if random.random() >= p_mistake: return best

    # safe alternatives (excluding best)
    alts = [c for c in _safe_alternatives(env, la, player, FL) if c != best]
    if not alts: return best  # fall back if no safe alternative

    if prefer_offcenter:
        # bias away from center to de-bias the dataset
        center = la.CENTER_COL
        alts.sort(key=lambda c: abs(center - c), reverse=True)
        alts = alts[:min(2, len(alts))]  # pick among the farthest 1–2
    return random.choice(alts)
