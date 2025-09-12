# -*- coding: utf-8 -*-
# PPO/ppo_bc_seed_adapters.py
"""
Adapters that convert your per-move Connect-4 DATA (one row per move)
into samples for PPO behavior cloning:
  dict(board: (6,7) np.int8 in {-1,0,+1}, player: +1/-1, action: int 0..6)

Expected DATA columns (same as your c4_seed_from_moves.py):
  - label, reward, game, ply
  - "r-c" per cell (e.g., "0-0" .. "5-6") with {-1,0,+1}
"""

from __future__ import annotations
from typing import Iterator, Tuple, List
import numpy as np
import pandas as pd

ROWS, COLS = 6, 7
BOARD_COLS = [f"{r}-{c}" for r in range(ROWS) for c in range(COLS)]

def _row_to_board(row) -> np.ndarray:
    vals = [int(row[f"{r}-{c}"]) for r in range(ROWS) for c in range(COLS)]
    return np.array(vals, dtype=np.int8).reshape(ROWS, COLS)

def _diff_action(prev_board: np.ndarray, next_board: np.ndarray) -> int | None:
    """
    Infer action (column) as the column where exactly ONE additional non-zero token
    appears compared to prev_board. Return None if not uniquely determined.
    """
    d = next_board - prev_board
    changed_cols = np.where(np.any(d != 0, axis=0))[0]
    if changed_cols.size != 1:
        return None
    c = int(changed_cols[0])
    col_prev = prev_board[:, c]
    col_next = next_board[:, c]
    nz_prev = int(np.count_nonzero(col_prev))
    nz_next = int(np.count_nonzero(col_next))
    if nz_next != nz_prev + 1:
        return None
    return c

def iter_moves_for_bc(DATA: pd.DataFrame, verbose: bool = True) -> Iterator[dict]:
    """
    Yield dict(board, player, action) for behavior cloning.

    Notes:
    - Uses the same de-dup + grouping logic as your DQN seeder.
    - 'player' is mover at this ply: +1 for odd plies, -1 for even plies.
    - Skips games/plies where the action cannot be uniquely inferred.
    """
    required = {"label", "reward", "game", "ply"} | set(BOARD_COLS)
    missing = [c for c in required if c not in DATA.columns]
    if missing:
        raise ValueError(f"DATA missing required columns: {missing}")

    subset_cols = ["label", "game", "ply"] + BOARD_COLS
    df_sorted = (
        DATA.sort_values(by=["label", "game", "ply"])
            .drop_duplicates(subset=subset_cols, keep="last")
            .reset_index(drop=True)
    )

    games = 0
    yielded = 0
    skipped: List[Tuple[str, int]] = []

    for (label, game_idx), gdf in df_sorted.groupby(["label", "game"]):
        games += 1
        prev_board = np.zeros((ROWS, COLS), dtype=np.int8)
 

        for _, row in gdf.iterrows():
            ply = int(row["ply"])
            mover = 1 if (ply % 2 == 1) else -1
            cur_board = _row_to_board(row)

            # tolerate duplicates within a game
            if np.array_equal(cur_board, prev_board):
                continue

            a = _diff_action(prev_board, cur_board)
            if a is None:
         
                if verbose:
                    skipped.append((str(label), int(game_idx)))
                break

            # Emit sample at the *pre-move* state: board==prev_board, player==mover, action==a
            yield {"board": prev_board.copy(), "player": int(mover), "action": int(a)}
            yielded += 1

            prev_board = cur_board  # advance

        # (optional) we could log only once per game
        # if not valid_game and verbose: ...

    if verbose:
        print(f"[BC-adapter] games={games}, samples_emitted={yielded}, skipped_games={len(skipped)}")
        if skipped[:10]:
            print("  skipped (first 10):", skipped[:10])
