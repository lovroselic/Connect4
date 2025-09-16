# a0_selfplay.py
# Self-play episode generator for AlphaZero-style Connect-4.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class TrajectoryStep:
    """
    A single self-play position:
      - state:      np.float32 [4,6,7]  (player-relative channels)
      - pi:         np.float32 [7]      (visit-count policy)
      - z:          float               (filled post-game: +1/-1/0 from that state's player POV)
      - legal_mask: np.bool_   [7]      (legal moves at this position)
      - player:     int                 (+1 or -1) player to move at this state
    """
    state: np.ndarray
    pi: np.ndarray
    z: float
    legal_mask: np.ndarray
    player: int

def _temperature_for_ply(ply: int, t_switch: int = 12, tau: float = 1.0, tau_final: float = 0.0) -> float:
    """AlphaZero-style schedule: use τ early for exploration, then 0 for determinism."""
    return float(tau if ply <= t_switch else tau_final)

def play_game(env, mcts, to_play: int = +1, t_switch: int = 12, tau: float = 1.0,
              tau_final: float = 0.0, max_moves: int = 42) -> Tuple[List[TrajectoryStep], int]:
    """
    Generate one self-play game using MCTS.
    - env:  EnvAdapter (must provide legal_moves, next_board, check_winner, is_terminal, encode_state)
    - mcts: your MCTS object exposing run(board, to_play, temperature, add_noise) -> (pi, root)
    Returns: (steps, winner) where winner ∈ {-1,0,+1}
    """
    board = env.reset()
    steps: List[TrajectoryStep] = []
    ply = 1

    while True:
        legal = env.legal_moves(board)                    # bool[7]
        # Safety stop
        if ply > max_moves:
            winner = env.check_winner(board)
            winner = 0 if winner not in (-1, +1) else int(winner)
            break

        # Encode state from current player's POV
        x = env.encode_state(board, to_play)              # torch [1,4,6,7]
        state_np = x.detach().cpu().numpy().squeeze(0).astype(np.float32)  # [4,6,7]

        # Run MCTS; add Dirichlet noise only at root (ply==1)
        temperature = _temperature_for_ply(ply, t_switch, tau, tau_final)
        pi, _ = mcts.run(board, to_play, temperature=temperature, add_noise=(ply == 1))
        pi = pi.astype(np.float32)
        pi[~legal] = 0.0
        s = float(pi.sum())
        if s <= 0:
            # If numeric issues, fall back to uniform over legal
            idx = np.where(legal)[0]
            pi = np.zeros(7, dtype=np.float32)
            if idx.size:
                pi[idx] = 1.0 / idx.size

        # Sample (or argmax if τ=0 -> pi likely one-hot already)
        action = int(np.random.choice(7, p=(pi / (pi.sum() + 1e-8))))

        # Record step (z is filled after game ends)
        steps.append(TrajectoryStep(
            state=state_np,
            pi=pi,
            z=0.0,                          # placeholder, filled post-game
            legal_mask=legal.astype(np.bool_),
            player=int(to_play),
        ))

        # Apply move
        board = env.next_board(board, action, to_play)
        to_play = -to_play
        ply += 1

        # Terminal?
        if env.is_terminal(board):
            winner = int(env.check_winner(board))
            if winner not in (-1, 0, +1):
                winner = 0
            break

    # Set outcomes z for each recorded state from that state's player's POV
    if winner == 0:
        z_final = 0.0
        for st in steps:
            st.z = z_final
    else:
        for st in steps:
            st.z = +1.0 if winner == st.player else -1.0

    return steps, winner
