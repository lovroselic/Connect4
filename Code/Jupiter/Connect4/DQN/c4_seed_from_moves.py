# c4_seed_from_moves.py
"""
Seed a replay buffer from a *per-move* Connect-4 dataset (one row per move).

Expected DataFrame columns:
  - "label": str
  - "reward": float      # IMMEDIATE mover-centric step reward at this move (NOT cumulative)
  - "game": int          # 0-based game index within label
  - "ply": int           # 1-based move index within the game
  - "r-c" for each cell (e.g., "0-0", ..., "5-6") with values in {+1, -1, 0}

How transitions are built:
  For each game (label, game) sorted by ply ascending:
    s_board      = previous board (zeros for ply==1)
    s_next_board = current row board
    action       = the column whose stack increased by exactly one token
    r_step       = row["reward"] (immediate mover-centric reward)
    done         = (this is the last move row for the game)

States are encoded as 4-channel tensors:
  [ plane(+1), plane(-1), row_encoding, col_encoding ]
and 'player' is set to the mover token (+1 on odd ply, -1 on even).

Usage:
    from c4_seed_from_moves import seed_from_dataframe_moves

    agent.memory.begin_seeding()
    nstep_buf = seed_from_dataframe_moves(nbuf, DATA, agent, n_step=3, gamma=agent.gamma, verbose=True)
    agent.memory.end_seeding()
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from DQN.nstep_buffer import NStepBuffer

ROWS, COLS = 6, 7
BOARD_COLS = [f"{r}-{c}" for r in range(ROWS) for c in range(COLS)]


def _row_to_board(row) -> np.ndarray:
    vals = [int(row[f"{r}-{c}"]) for r in range(ROWS) for c in range(COLS)]
    return np.array(vals, dtype=int).reshape(ROWS, COLS)


def _board_to_planes(board: np.ndarray) -> np.ndarray:
    """Return (2, ROWS, COLS): [+1 plane, -1 plane] as int {0,1}."""
    p_pos = (board == 1).astype(np.int32)
    p_neg = (board == -1).astype(np.int32)
    return np.stack([p_pos, p_neg], axis=0)


def _add_positional_planes(twoplanes: np.ndarray) -> np.ndarray:
    """Concatenate row/col encoding planes to get (4, ROWS, COLS)."""
    row_plane = np.tile(np.linspace(-1, 1, ROWS, dtype=np.float32)[:, None], (1, COLS))
    col_plane = np.tile(np.linspace(-1, 1, COLS, dtype=np.float32)[None, :], (ROWS, 1))
    return np.concatenate([twoplanes.astype(np.float32), row_plane[None], col_plane[None]], axis=0)


def _diff_action(prev_board: np.ndarray, next_board: np.ndarray) -> Optional[int]:
    """
    Infer action (column) as the column where exactly ONE additional non-zero
    token appears compared to prev_board. Return None if not uniquely determined.
    """
    d = next_board - prev_board
    changed_cols = np.where(np.any(d != 0, axis=0))[0]
    if changed_cols.size != 1: return None
    c = int(changed_cols[0])
    col_prev = prev_board[:, c]
    col_next = next_board[:, c]
    nz_prev = int(np.count_nonzero(col_prev))
    nz_next = int(np.count_nonzero(col_next))
    if nz_next != nz_prev + 1: return None
    return c


def seed_from_dataframe_moves(
    nbuf: NStepBuffer,
    df: pd.DataFrame,
    agent,
    n_step: int = 3,
    gamma: Optional[float] = None,
    verbose: bool = True,
    reward_scale: Optional[float] = None,  # <-- NEW: optional override
) -> NStepBuffer:
    """
    Seed agent.memory with transitions derived from a per-move dataset.
    Rewards in `df["reward"]` are *raw/unscaled*; they will be
    multiplied by `reward_scale` (default: agent.reward_scale).
    Returns the NStepBuffer used for seeding (already flushed/reset per game).
    """
    if gamma is None: gamma = float(agent.gamma)

    # Determine reward scaling (prefer explicit arg, else agent.reward_scale, else 1.0)
    rs = float(reward_scale if reward_scale is not None else getattr(agent, "reward_scale", 1.0))

    required = {"label", "reward", "game", "ply"} | set(BOARD_COLS)
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"DataFrame missing required columns: {missing}")

    # Sort and deduplicate on the board snapshot (ignore reward) to avoid
    # "cannot infer unique action" caused by identical boards with different rewards.
    subset_cols = ["label", "game", "ply"] + BOARD_COLS
    df_sorted = (
        df.sort_values(by=["label", "game", "ply"])
          .drop_duplicates(subset=subset_cols, keep="last")
          .reset_index(drop=True)
    )

    games_added = 0
    steps_added = 0
    failed_games: List[Tuple[str, int]] = []

    # simple sanity stats
    raw_min, raw_max = float("inf"), float("-inf")
    sca_min, sca_max = float("inf"), float("-inf")

    for (label, game_idx), gdf in tqdm(df_sorted.groupby(["label", "game"]), desc="Seeding", disable=False):
        prev_board = np.zeros((ROWS, COLS), dtype=np.int32)
        max_ply = int(gdf["ply"].max())

        nbuf.reset()
        valid_game = True

        for _, row in gdf.iterrows():
            ply = int(row["ply"])                 # 1-based
            mover = 1 if (ply % 2 == 1) else -1   # +1 on odd plies, -1 on even

            cur_board = _row_to_board(row)

            # Tolerate stray duplicate snapshots inside a group
            if np.array_equal(cur_board, prev_board): continue

            a = _diff_action(prev_board, cur_board)
            if a is None:
                valid_game = False
                break

            # --- SCALE REWARD HERE ---
            r_raw = float(row["reward"])       # raw in DATA
            r_step = r_raw * rs                # scaled for training
            raw_min = min(raw_min, r_raw); raw_max = max(raw_max, r_raw)
            sca_min = min(sca_min, r_step); sca_max = max(sca_max, r_step)

            s      = _add_positional_planes(_board_to_planes(prev_board))
            s_next = _add_positional_planes(_board_to_planes(cur_board))
            done = (ply == max_ply)

            nbuf.append(s, int(a), r_step, s_next, bool(done), player=int(mover))
            steps_added += 1

            prev_board = cur_board  # advance

            if done: break

        if valid_game:
            nbuf.flush()
            games_added += 1
        else: failed_games.append((str(label), int(game_idx)))

    if verbose:
        print(f"[seed] games_added={games_added}, steps_added={steps_added}, failed_games={len(failed_games)}")
        print(f"[seed] reward_scale={rs:.6f} | raw_min..max={raw_min:.3f}..{raw_max:.3f} | scaled_min...max={sca_min:.3f}...{sca_max:.3f}")
        if failed_games: print("  Failed:", failed_games[:10], ("..." if len(failed_games) > 10 else ""))

    nbuf.reset()
    return nbuf

