# DQN.seed_vs_teacher.py
from __future__ import annotations
from typing import Dict, List, Optional
import random
import numpy as np
from C4.connect4_env import Connect4Env
from C4.connect4_lookahead import Connect4Lookahead
from DQN.teacher_policy import GreedyDQNPolicy
from tqdm import tqdm


ROWS, COLS = 6, 7

def _planes4(board: np.ndarray) -> np.ndarray:
    p_pos = (board == 1).astype(np.float32)
    p_neg = (board == -1).astype(np.float32)
    row_plane = np.tile(np.linspace(-1, 1, ROWS, dtype=np.float32)[:, None], (1, COLS))
    col_plane = np.tile(np.linspace(-1, 1, COLS, dtype=np.float32)[None, :], (ROWS, 1))
    return np.stack([p_pos, p_neg, row_plane, col_plane], axis=0)

def _flatten(board: np.ndarray) -> Dict[str, int]:
    return {f"{r}-{c}": int(board[r, c]) for r in range(ROWS) for c in range(COLS)}


def play_series_teacher_random(
    teacher_path: str,
    games_per_side: int = 200,
    seed: Optional[int] = 666,
    show_progress: bool = True,
) -> List[Dict[str, float]]:
    """
    Build per-move rows for Teacher(+1) vs Random and Random(+1) vs Teacher.
    Rows match  schema: label, reward, game, ply, 0-0..5-6.
    """
    
    label_prefix = teacher_path.split(".")[0]
    rng = random.Random(seed) if seed is not None else random
    env = Connect4Env() 
    la = Connect4Lookahead() 
    T = GreedyDQNPolicy(teacher_path)

    rows: List[Dict[str, float]] = []

    it1 = range(games_per_side)
    it2 = range(games_per_side)
    if show_progress:
        it1 = tqdm(it1, desc="Teacher(+1) vs Random", leave=False)
        it2 = tqdm(it2, desc="Random(+1) vs Teacher", leave=False)

    # Teacher starts (+1)
    for g in it1:
        env.reset(); done = False; ply = 0
        while not done:
            ply += 1
            player = env.current_player
            legal  = env.available_actions()
            a = (T(env.board.copy(), legal, player, la)
                 if player == +1 else rng.choice(legal))
            _, r, done = env.step(a)
            rec = {"label": f"{label_prefix}LKT_vs_RND", "reward": float(r), "game": int(g), "ply": int(ply)}
            rec.update(_flatten(env.board)); rows.append(rec)

    # Teacher plays second (-1)
    for g in it2:
        env.reset(); done = False; ply = 0
        while not done:
            ply += 1
            player = env.current_player
            legal  = env.available_actions()
            a = (rng.choice(legal)
                 if player == +1 else T(env.board.copy(), legal, player, la))
            _, r, done = env.step(a)
            rec = {"label": f"{label_prefix}RND_vs_LKT", "reward": float(r), "game": int(g), "ply": int(ply)}
            rec.update(_flatten(env.board)); rows.append(rec)

    return rows

