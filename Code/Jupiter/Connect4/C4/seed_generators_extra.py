# seed_generators_extra.py
from __future__ import annotations
from typing import  List, Dict, Tuple, Optional
import random, numpy as np
from tqdm import tqdm

from C4.connect4_env import Connect4Env
from C4.connect4_lookahead import Connect4Lookahead

ROWS, COLS = 6, 7
BOARD_COLS = [f"{r}-{c}" for r in range(ROWS) for c in range(COLS)]

# ---- policies ----
def policy_random(board: np.ndarray, legal: List[int], player: int, la: Connect4Lookahead) -> int:
    return random.choice(legal)

def make_policy_lookahead(depth: int):
    def _pick(board: np.ndarray, legal: List[int], player: int, la: Connect4Lookahead) -> int:
        return la.n_step_lookahead(board, player, depth=depth)
    return _pick

# ---- utils ----
def _flatten_board(board: np.ndarray) -> Dict[str, int]:
    return {f"{r}-{c}": int(board[r, c]) for r in range(ROWS) for c in range(COLS)}

def _legal(board: np.ndarray) -> List[int]:
    return [c for c in range(COLS) if board[0, c] == 0]

def _drop(board: np.ndarray, col: int, player: int) -> np.ndarray:
    b = board.copy()
    for r in range(ROWS-1, -1, -1):
        if b[r, col] == 0:
            b[r, col] = player
            return b
    raise ValueError("Column full")

# ---- core simulator -> per-move rows ----
def play_game_rows_with_policies(
    label: str, game_index: int,
    policy_first, policy_second,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    env = Connect4Env(); la = Connect4Lookahead(); env.reset()

    rows, done, ply = [], False, 0
    while not done:
        ply += 1
        player = env.current_player
        legal  = env.available_actions()
        policy = policy_first if player == +1 else policy_second
        a = policy(env.board.copy(), legal, player, la)
        _, r, done = env.step(a)
        rec = {"label": label, "reward": float(r), "game": int(game_index), "ply": int(ply)}
        rec.update(_flatten_board(env.board))
        rows.append(rec)
    return rows

# ---- Lx vs Random (both directions) ----
def generate_lx_vs_random_records(
    depth: int,
    games_per_side: int,
    seed: Optional[int] = None,
    label_prefix: str = "",
    show_progress: bool = True,
) -> List[Dict[str, float]]:

    rows: List[Dict[str, float]] = []
    rng = random.Random(seed) if seed is not None else random
    pol_L, pol_R = make_policy_lookahead(depth), policy_random

    it1 = range(games_per_side)
    it2 = range(games_per_side)
    if show_progress:
        it1 = tqdm(it1, desc=f"L{depth} vs RND (+1 starts)", leave=False)
        it2 = tqdm(it2, desc=f"RND vs L{depth} (+1 random)", leave=False)

    for g in it1:
        s = None if seed is None else rng.randint(0, 2**31-1)
        label = f"{label_prefix}L{depth}_vs_RND"
        rows.extend(play_game_rows_with_policies(label, g, pol_L, pol_R, seed=s))

    for g in it2:
        s = None if seed is None else rng.randint(0, 2**31-1)
        label = f"{label_prefix}RND_vs_L{depth}"
        rows.extend(play_game_rows_with_policies(label, g, pol_R, pol_L, seed=s))

    return rows

def generate_lx_vs_random_batch(
    depths: Tuple[int, ...] = (1, 2),
    games_per_side: int = 200,
    seed: Optional[int] = None,
    label_prefix: str = "",
    show_progress: bool = True,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    depth_iter = depths
    if show_progress:
        depth_iter = tqdm(depths, desc="Depths", leave=False)
    for d in depth_iter:
        rows.extend(generate_lx_vs_random_records(
            d, games_per_side, seed=seed, label_prefix=label_prefix, show_progress=show_progress
        ))
    return rows

# ---- synthetic must-block vs violation pairs ----
def _count_immediate_wins(board: np.ndarray, player: int, la: Connect4Lookahead) -> List[int]:
    return list(la.count_immediate_wins(board, player))

def find_must_block_and_unsafe(board: np.ndarray, player: int, la: Connect4Lookahead):
    legal = _legal(board)
    if not legal: return None, []
    opp = -player
    unsafe, safe = [], []
    for a in legal:
        after = _drop(board, a, player)
        if len(_count_immediate_wins(after, opp, la)) > 0: unsafe.append(a)
        else: safe.append(a)
    must_block = None
    if len(_count_immediate_wins(board, opp, la)) > 0 and safe:
        safe.sort(key=lambda c: abs(3-c)); must_block = safe[0]
    return must_block, unsafe

def synth_violation_pairs(
    label_base: str,
    n_pairs: int,
    max_random_plies: Tuple[int,int] = (3, 20),
    rng_seed: Optional[int] = None,
    show_progress: bool = True,
) -> List[Dict[str, float]]:

    if rng_seed is not None:
        random.seed(rng_seed); np.random.seed(rng_seed)

    env = Connect4Env(); la = Connect4Lookahead()
    rows: List[Dict[str, float]] = []
    made, gid = 0, 0
    pbar = tqdm(total=n_pairs, desc=f"Synth pairs ({label_base})", leave=False) if show_progress else None

    while made < n_pairs:
        env.reset()
        for _ in range(random.randint(*max_random_plies)):
            legal = env.available_actions()
            if not legal or env.done: break
            env.step(random.choice(legal))
        if env.done: continue

        board, player = env.board.copy(), env.current_player
        must, unsafe = find_must_block_and_unsafe(board, player, la)
        if must is None or not unsafe: continue

        # good
        env2 = Connect4Env(); env2.board = board.copy(); env2.current_player = player
        _, r_good, _ = env2.step(must)
        rec_good = {"label": f"{label_base}_GOOD", "game": gid, "ply": 1, "reward": float(r_good)}
        rec_good.update(_flatten_board(env2.board)); rows.append(rec_good)

        # bad
        unsafe.sort(key=lambda c: abs(3-c))
        bad = unsafe[0]
        env3 = Connect4Env(); env3.board = board.copy(); env3.current_player = player
        _, r_bad, _ = env3.step(bad)
        rec_bad = {"label": f"{label_base}_BAD", "game": gid, "ply": 1, "reward": float(r_bad)}
        rec_bad.update(_flatten_board(env3.board)); rows.append(rec_bad)

        gid += 1; made += 1
        if pbar: pbar.update(1)

    if pbar: pbar.close()
    return rows
