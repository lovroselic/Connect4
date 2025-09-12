# c4_seed_util.py
"""
Seed a replay buffer from final-board snapshots stored in a DataFrame.

Relies on:
  • NStepBuffer  (imported from your DQN module)
  • Connect4Env, Connect4Lookahead
"""

from typing import Optional, List, Tuple
import numpy as np
import time
from tqdm import tqdm

from C4.connect4_env import Connect4Env
from C4.connect4_lookahead import Connect4Lookahead
from DQN.nstep_buffer import NStepBuffer

ROWS, COLS = 6, 7
BOARD_COLS = [f"{r}-{c}" for r in range(ROWS) for c in range(COLS)]

def _row_to_board(row) -> np.ndarray:
    vals = [int(row[f"{r}-{c}"]) for r in range(ROWS) for c in range(COLS)]
    return np.array(vals, dtype=int).reshape(ROWS, COLS)

def _stacks_from_final(final_board: np.ndarray) -> Tuple[List[List[int]], List[int], int]:
    stacks, heights, total = [], [], 0
    for c in range(COLS):
        col = []
        for r in range(ROWS - 1, -1, -1):
            v = int(final_board[r, c])
            if v == 0:
                break
            col.append(v)
        stacks.append(col)
        heights.append(len(col))
        total += len(col)
    return stacks, heights, total

def _reconstruct_moves(final_board: np.ndarray,
                       winner: int,
                       look: Connect4Lookahead,
                       prefer_center: bool = True,
                       max_nodes: int = 250_000,
                       max_seconds: float | None = 2.0) -> list[int] | None:
    ROWS, COLS = final_board.shape
    stacks, heights, total = _stacks_from_final(final_board)

    p1 = int((final_board == 1).sum())
    p2 = int((final_board == -1).sum())
    if not (p1 == p2 or p1 == p2 + 1):
        return None
    last_player_expected = 1 if (total % 2 == 1) else -1
    if winner in (1, -1) and winner != last_player_expected:
        return None

    is_draw_final = (winner == 0 and total == ROWS * COLS)
    order = sorted(range(COLS), key=lambda c: abs(c - 3)) if prefer_center else list(range(COLS))

    cur = np.zeros_like(final_board, dtype=int)
    used = [0] * COLS

    def has_four_from(r: int, c: int, player: int) -> bool:
        DIRS = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        for dr, dc in DIRS:
            cnt = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and cur[rr, cc] == player:
                cnt += 1
                rr += dr; cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and cur[rr, cc] == player:
                cnt += 1
                rr -= dr; cc -= dc
            if cnt >= 4:
                return True
        return False

    def place(c: int, player: int) -> tuple[int, int]:
        r = ROWS - 1 - used[c]
        cur[r, c] = player
        used[c] += 1
        return r, c

    def unplace(c: int, r: int):
        used[c] -= 1
        cur[r, c] = 0

    path = []
    nodes = 0
    t0 = time.time()

    def dfs(t: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            return False
        if max_seconds is not None and (time.time() - t0) > max_seconds:
            return False
        if t == total:
            return np.array_equal(cur, final_board)

        player = 1 if (t % 2 == 0) else -1
        last = (t == total - 1)
        cands = []

        for c in order:
            if used[c] >= heights[c]:
                continue
            if stacks[c][used[c]] != player:
                continue
            if cur[0, c] != 0:
                continue
            cands.append(c)

        if not cands:
            return False

        for c in cands:
            r, _ = place(c, player)
            win_now = has_four_from(r, c, player)

            if is_draw_final:
                if win_now:
                    unplace(c, r)
                    continue
            else:
                if win_now and not last:
                    unplace(c, r)
                    continue
                if last:
                    if player != winner:
                        unplace(c, r)
                        continue
                    if not win_now:
                        if not np.array_equal(cur, final_board):
                            unplace(c, r)
                            continue

            path.append(c)
            if dfs(t + 1):
                return True
            path.pop()
            unplace(c, r)

        return False

    return path if dfs(0) else None

def seed_from_dataframe(
    nbuf,
    df,
    agent,
    n_step: int = 3,
    gamma: Optional[float] = None,
    verbose: bool = True,
    reward_scale: Optional[float] = None,  # <-- NEW: explicit override
    prefer_center: bool = True,            # keep default behavior
) -> NStepBuffer:
    if gamma is None:
        gamma = float(agent.gamma)

    # Determine reward scaling: prefer explicit arg -> agent.reward_scale -> 1.0
    rs = float(reward_scale if reward_scale is not None else getattr(agent, "reward_scale", 1.0))

    look = Connect4Lookahead()

    games_added = 0
    steps_added = 0
    failed = 0
    failed_labels = []

    # sanity stats
    raw_min, raw_max = float("inf"), float("-inf")
    sca_min, sca_max = float("inf"), float("-inf")

    for label, row in tqdm(df.iterrows(), total=len(df), desc="Seeding"):
        final_board = _row_to_board(row)
        winner = int(row["winner"])
        game_label = row.get("label", label)

        seq = _reconstruct_moves(final_board, winner, look, prefer_center=prefer_center)
        if seq is None:
            failed += 1
            failed_labels.append(game_label)
            continue

        env = Connect4Env()
        env.reset()
        nbuf.reset()

        for a in seq:
            # Pre-move snapshot & mover (matches training loop semantics)
            mover = int(env.current_player)        # +1 or -1, player about to move
            s = env.get_state()                    # (4, 6, 7): [+1, -1, row, col]

            # Apply move -> raw reward from env
            s_next, r_raw, done = env.step(int(a))

            # --- SCALE REWARD HERE ---
            r_step = float(r_raw) * rs
            raw_min = min(raw_min, float(r_raw)); raw_max = max(raw_max, float(r_raw))
            sca_min = min(sca_min, r_step);       sca_max = max(sca_max, r_step)

            # Store exactly like training (immediate reward, correct mover)
            nbuf.append(s, int(a), r_step, s_next.copy(), bool(done), player=mover)

            if done:
                break

        nbuf.flush()
        games_added += 1
        steps_added += len(seq)

    if verbose:
        print(f"[seed] games={games_added}, steps={steps_added}, failed={failed}")
        print(f"[seed] reward_scale={rs:.6f} | raw_min..max={raw_min:.3f}..{raw_max:.3f} | scaled_min..max={sca_min:.3f}..{sca_max:.3f}")
        if failed_labels:
            print("  Failed game labels:", failed_labels[:10], "..." if len(failed_labels) > 10 else "")

    nbuf.reset()
    return nbuf

