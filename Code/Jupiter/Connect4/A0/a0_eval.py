# a0_eval.py  (update to allow eval_sims override)
from __future__ import annotations
import random
import numpy as np

def _random_move(env, board):
    legal = env.legal_moves(board)
    cols = np.where(legal)[0]
    return int(random.choice(cols)) if cols.size else 0

def _winning_move(env, board, player):
    legal = env.legal_moves(board)
    for c in range(7):
        if not legal[c]: continue
        nb = env.next_board(board, c, player)
        if env.check_winner(nb) == player:
            return c
    return None

def _l1_blocker(env, board, player):
    legal = env.legal_moves(board)
    w = _winning_move(env, board, player)
    if w is not None: return w
    b = _winning_move(env, board, -player)
    if b is not None: return b
    for c in [3,2,4,1,5,0,6]:
        if legal[c]: return c
    return _random_move(env, board)

def _play_match(env, mcts, opponent_fn, games: int = 50, eval_sims: int | None = None) -> dict:
    old_sims = mcts.sims
    if eval_sims is not None:
        mcts.sims = eval_sims
    wins = losses = draws = 0
    for _ in range(games):
        board = env.reset()
        to_play = +1 if random.random() < 0.5 else -1
        while True:
            if to_play == +1:
                pi, _ = mcts.run(board, to_play=+1, temperature=0.0, add_noise=False)
                action = int(np.argmax(pi))
            else:
                action = opponent_fn(env, board, -1) if opponent_fn is _l1_blocker else opponent_fn(env, board)
            board = env.next_board(board, action, to_play)
            to_play = -to_play
            if env.is_terminal(board):
                w = env.check_winner(board)
                if w == +1: wins += 1
                elif w == -1: losses += 1
                else: draws += 1
                break
    mcts.sims = old_sims
    total = wins + losses + draws
    return {"games": total,
            "win_rate": wins / total if total else 0.0,
            "loss_rate": losses / total if total else 0.0,
            "draw_rate": draws / total if total else 0.0,
            "W": wins, "L": losses, "D": draws}

def eval_vs_random(env, mcts, games=50, eval_sims: int | None = None):
    return _play_match(env, mcts, _random_move, games, eval_sims)

def eval_vs_l1(env, mcts, games=50, eval_sims: int | None = None):
    return _play_match(env, mcts, _l1_blocker, games, eval_sims)
