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

def seed_from_dataframe(df, agent, n_step: int = 3, gamma: Optional[float] = None, verbose: bool = True) -> NStepBuffer:
    if gamma is None:
        gamma = float(agent.gamma)

    nbuf = NStepBuffer(n=n_step, gamma=gamma, memory=agent.memory)
    look = Connect4Lookahead()

    games_added = 0
    steps_added = 0
    failed = 0
    failed_labels = []

    row_plane = np.tile(np.linspace(-1, 1, ROWS)[:, None], (1, COLS))
    col_plane = np.tile(np.linspace(-1, 1, COLS)[None, :], (ROWS, 1))

    for label, row in tqdm(df.iterrows(), total=len(df), desc="Seeding"):
        final_board = _row_to_board(row)
        winner = int(row["winner"])
        game_label = row.get("label", label)

        seq = _reconstruct_moves(final_board, winner, look)
        if seq is None:
            failed += 1
            failed_labels.append(game_label)
            continue

        env = Connect4Env()
        env.reset()
        nbuf.reset()

        for a in seq:
            s = env.get_state()
            s_next, r, done = env.step(a)
            mover = -env.current_player

            if s.ndim == 2:
                s = np.stack([(s == 1).astype(int), (s == -1).astype(int)])
            if s_next.ndim == 2:
                s_next = np.stack([(s_next == 1).astype(int), (s_next == -1).astype(int)])

            s_aug = np.concatenate([s, row_plane[None], col_plane[None]], axis=0)
            s_next_aug = np.concatenate([s_next, row_plane[None], col_plane[None]], axis=0)

            nbuf.append(s_aug, a, float(r), s_next_aug.copy(), bool(done), player=int(mover))
            if done:
                break

        nbuf.flush()
        games_added += 1
        steps_added += len(seq)

    if verbose:
        print(f"[seed] games={games_added}, steps={steps_added}, failed={failed}")
        if failed_labels:
            print("  Failed game labels:", failed_labels)

    nbuf.reset()
    return nbuf
