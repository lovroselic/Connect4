# PPO/ppo_bc_seed_adapters.py
"""
Adapters that convert your per-move Connect-4 DATA (one row per move)
into samples for PPO behavior cloning:
  dict(board: (6,7) np.int8 in {-1,0,+1}, player: +1/-1, action: int 0..6)

Supports two styles of DATA:

1) Full logs (both players, every ply):
   - no 'winner'/'player' columns required
   - uses ply parity (odd=+1, even=-1) and simple board diff

2) "Happy" winner-only logs (your BC_clean):
   - must have columns 'player' and 'winner'
   - for each (label,game): all rows have same winner, and player==winner
   - we reconstruct opponent plies between winner snapshots.

Expected base columns (both styles):
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
    (Used for full logs only.)
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

    Modes
    -----
    - Full logs mode:
        Both players present, one row per actual ply.
        'player'/'winner' columns not required.
        We use ply parity (odd=+1, even=-1) and a simple one-move diff.

    - Winner-only "happy" mode:
        For a given (label, game), all rows have the same 'winner',
        and 'player' == 'winner' for all rows.
        These rows are snapshots AFTER each winner move; opponent plies
        in between are missing. We reconstruct the pre-move state by:
          * First winner move:
                - if only one disc -> winner started; pre-board is empty.
                - if two discs    -> winner is second; pre-board has only
                                     the opponent's first disc.
          * Subsequent winner moves:
                - between consecutive winner snapshots there are two new
                  discs (one winner, one opponent); we identify them from
                  the signs and reconstruct "prev_winner + opp_move".
    """
    required_base = {"label", "reward", "game", "ply"} | set(BOARD_COLS)
    missing = [c for c in required_base if c not in DATA.columns]
    if missing:
        raise ValueError(f"DATA missing required columns: {missing}")

    has_player = "player" in DATA.columns
    has_winner = "winner" in DATA.columns

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
        gdf = gdf.sort_values("ply")

        # ---------- detect winner-only "happy" mode ----------
        use_happy = False
        winner_sign = None

        if has_player and has_winner:
            # one constant winner per game, and all rows belong to that winner
            uniq_winners = gdf["winner"].unique()
            uniq_players = gdf["player"].unique()
            if (
                len(uniq_winners) == 1
                and len(uniq_players) == 1
                and int(uniq_players[0]) == int(uniq_winners[0])
            ):
                winner_sign = int(uniq_winners[0])  # should be +1 or -1
                if winner_sign in (-1, 1):
                    use_happy = True

        # ============================================================
        # 1) HAPPY / WINNER-ONLY MODE
        # ============================================================
        if use_happy:
            # Board marks are assumed in {-1,0,+1}
            me = winner_sign
            opp = -winner_sign

            prev_winner_board = np.zeros((ROWS, COLS), dtype=np.int8)
            first_move_done = False
            game_ok = True

            for _, row in gdf.iterrows():
                cur_board = _row_to_board(row)

                if not first_move_done:
                    # First winner move: compare to empty board
                    changed = cur_board != 0
                    rs, cs = np.where(changed)
                    n_changes = rs.size

                    if n_changes not in (1, 2):
                        if verbose:
                            skipped.append((str(label), int(game_idx)))
                        game_ok = False
                        break

                    coords = list(zip(rs.tolist(), cs.tolist()))
                    # Cells containing winner / opponent stones
                    win_cells = [(r, c) for (r, c) in coords if cur_board[r, c] == me]
                    opp_cells = [(r, c) for (r, c) in coords if cur_board[r, c] == opp]

                    # Must have exactly one winner stone
                    if len(win_cells) != 1:
                        if verbose:
                            skipped.append((str(label), int(game_idx)))
                        game_ok = False
                        break

                    wr, wc = win_cells[0]

                    # Pre-move board:
                    #   - if winner started: only that stone appears -> pre is empty
                    #   - if opponent started: we also add their first stone
                    pre_board = np.zeros_like(cur_board, dtype=np.int8)
                    if len(opp_cells) == 1:
                        orow, ocol = opp_cells[0]
                        pre_board[orow, ocol] = opp

                    yield {
                        "board": pre_board.copy(),
                        "player": int(me),
                        "action": int(wc),
                    }
                    yielded += 1

                    prev_winner_board = cur_board
                    first_move_done = True
                    continue

                # Subsequent winner moves: prev_winner_board -> cur_board
                prev = prev_winner_board
                diff_mask = cur_board != prev
                rs, cs = np.where(diff_mask)
                n_changes = rs.size

                # We expect exactly two new stones (one opp, one me)
                if n_changes != 2:
                    if verbose:
                        skipped.append((str(label), int(game_idx)))
                    game_ok = False
                    break

                coords = list(zip(rs.tolist(), cs.tolist()))
                win_cells = [
                    (r, c)
                    for (r, c) in coords
                    if cur_board[r, c] == me and prev[r, c] == 0
                ]
                opp_cells = [
                    (r, c)
                    for (r, c) in coords
                    if cur_board[r, c] == opp and prev[r, c] == 0
                ]

                if len(win_cells) != 1 or len(opp_cells) != 1:
                    if verbose:
                        skipped.append((str(label), int(game_idx)))
                    game_ok = False
                    break

                wr, wc = win_cells[0]
                orow, ocol = opp_cells[0]

                # Build pre-move board = previous winner board + opponent's new stone
                pre_board = prev.copy()
                if pre_board[orow, ocol] != 0:
                    # This should never happen; inconsistent log
                    if verbose:
                        skipped.append((str(label), int(game_idx)))
                    game_ok = False
                    break
                pre_board[orow, ocol] = opp

                yield {
                    "board": pre_board.copy(),
                    "player": int(me),
                    "action": int(wc),
                }
                yielded += 1

                prev_winner_board = cur_board

            # done with this game
            if not game_ok:
                continue
            else:
                continue  # go to next game

        # ============================================================
        # 2) FULL LOGS MODE (OR NO WINNER INFO)
        # ============================================================
        prev_board = np.zeros((ROWS, COLS), dtype=np.int8)

        for _, row in gdf.iterrows():
            ply = int(row["ply"])
            mover = 1 if (ply % 2 == 1) else -1
            cur_board = _row_to_board(row)

            # tolerate exact duplicates
            if np.array_equal(cur_board, prev_board):
                continue

            a = _diff_action(prev_board, cur_board)
            if a is None:
                if verbose:
                    skipped.append((str(label), int(game_idx)))
                break

            yield {
                "board": prev_board.copy(),
                "player": int(mover),
                "action": int(a),
            }
            yielded += 1
            prev_board = cur_board

    if verbose:
        print(
            f"[BC-adapter] games={games}, samples_emitted={yielded}, "
            f"skipped_games={len(skipped)}"
        )
        if skipped[:10]:
            print("  skipped (first 10):", skipped[:10])
