# A0/a0_bench.py
from __future__ import annotations
import random
import numpy as np
import torch
from typing import Dict, List, Optional
from tqdm import tqdm

from A0.a0_opponents import Opponent, build_suite
from A0.a0_logger_xlsx import ExcelLogger

def _play_match(
    env,
    mcts,
    opponent: Opponent,
    games: int,
    eval_sims: int,
    seed: Optional[int] = None
) -> Dict[str, float]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    old_sims = mcts.sims
    mcts.sims = eval_sims

    # Global MCTS stats
    W = L = D = 0
    # Split by side that MCTS controlled
    W_P1 = L_P1 = D_P1 = 0   # MCTS was +1
    W_P2 = L_P2 = D_P2 = 0   # MCTS was -1

    for g in tqdm(range(games), desc=f"Final eval vs {opponent.name}", leave=False):
        board = env.reset()

        # Alternate starting player; whichever side starts this game is +1
        to_play = +1 if (g % 2 == 0) else -1

        # MCTS controls whichever side is +1 when the game starts
        # (so MCTS alternates: +1, then -1, then +1, ...)
        mcts_side = +1 if to_play == +1 else -1
        mcts_is_p1 = (mcts_side == +1)

        while True:
            if to_play == +1:
                if mcts_side == +1:
                    pi, _ = mcts.run(board, to_play=+1, temperature=0.0, add_noise=False)
                    action = int(np.argmax(pi))
                else:
                    action = opponent.decide(env, board, +1)
            else:
                if mcts_side == -1:
                    pi, _ = mcts.run(board, to_play=-1, temperature=0.0, add_noise=False)
                    action = int(np.argmax(pi))
                else:
                    action = opponent.decide(env, board, -1)

            board = env.next_board(board, action, to_play)
            to_play = -to_play

            if env.is_terminal(board):
                w = env.check_winner(board)  # +1, -1, or 0 (draw)

                if w == mcts_side:
                    W += 1
                    if mcts_is_p1: W_P1 += 1
                    else:          W_P2 += 1
                elif w == -mcts_side:
                    L += 1
                    if mcts_is_p1: L_P1 += 1
                    else:          L_P2 += 1
                else:
                    D += 1
                    if mcts_is_p1: D_P1 += 1
                    else:          D_P2 += 1
                break

    mcts.sims = old_sims

    total = max(1, W + L + D)
    return {
        "games": W + L + D,
        "sims": eval_sims,
        "W": W, "L": L, "D": D,
        "win_rate": W / total,
        "loss_rate": L / total,
        "draw_rate": D / total,
        "W_as_P1": W_P1, "L_as_P1": L_P1, "D_as_P1": D_P1,
        "W_as_P2": W_P2, "L_as_P2": L_P2, "D_as_P2": D_P2,
    }

def evaluate_suite(
    env,
    mcts,
    opponent_ids: List[str],
    games_each: int,
    eval_sims: int,
    out_xlsx: Optional[str] = None,
    sheet: str = "final_eval",
    seed: Optional[int] = 123,
) -> List[Dict[str, float]]:
    opponents = build_suite(opponent_ids)
    results: List[Dict[str, float]] = []

    xl = None
    if out_xlsx:
        xl = ExcelLogger(
            out_xlsx,
            sheet_name=sheet,
            header=[
                "opponent", "games", "sims", "W", "L", "D",
                "win_rate", "loss_rate", "draw_rate",
                "W_as_P1", "L_as_P1", "D_as_P1", "W_as_P2", "L_as_P2", "D_as_P2",
            ],
        )

    for opp in opponents:
        r = _play_match(env, mcts, opp, games_each, eval_sims, seed=seed)
        row = {"opponent": getattr(opp, "name", opp.__class__.__name__), **r}
        results.append(row)
        if xl:
            xl.log(row, [
                "opponent", "games", "sims", "W", "L", "D",
                "win_rate", "loss_rate", "draw_rate",
                "W_as_P1", "L_as_P1", "D_as_P1", "W_as_P2", "L_as_P2", "D_as_P2",
            ])

    if xl:
        xl.close()
    return results
