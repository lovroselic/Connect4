"""
connect4_board_display.py

Lightweight board visualization helpers for Connect-4 policy evaluation.
No DQN/PPO legacy imports.

Board convention:
- numpy array (6,7) with values {-1, 0, +1}
- row 0 is top, row 5 is bottom

Usage:
  from connect4_board_display import display_final_boards_PPO
  display_final_boards_PPO(policy, ["Random", "Leftmost", "Lookahead-1"], lookahead=la)
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from ppo_agent_eval import play_single_game_ppo


class Connect4_BoardDisplayer:
    def __init__(self):
        self.rows = 6
        self.cols = 7

    def _draw_board(self, ax, board: np.ndarray, title: str = "") -> None:
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(self.rows - 0.5, -0.5)
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_title(title)

        # grid + discs
        for r in range(self.rows):
            for c in range(self.cols):
                v = int(board[r, c])
                if v == 1:
                    face = "red"
                elif v == -1:
                    face = "gold"
                else:
                    face = "white"
                ax.add_patch(Circle((c, r), 0.42, facecolor=face, edgecolor="black", linewidth=1))

        ax.grid(True, linestyle="--", alpha=0.25)

    def show_board(self, board: np.ndarray, title: str = "") -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        self._draw_board(ax, board, title=title)
        plt.show()


def display_final_boards_PPO(
    policy,
    opponent_labels: Iterable[str],
    lookahead=None,
    seed: int = 666,
    policy_deterministic: bool = True,
    policy_temperature: float = 1.0,
) -> None:
    """
    Play one game vs each opponent in `opponent_labels` and display the final boards.
    """
    labels = list(opponent_labels)
    if not labels:
        return

    disp = Connect4_BoardDisplayer()

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for i, lab in enumerate(labels):
        res, board = play_single_game_ppo(
            policy,
            opponent_label=lab,
            seed=seed + i,
            lookahead=lookahead,
            policy_deterministic=policy_deterministic,
            policy_temperature=policy_temperature,
        )
        outcome = "WIN" if res == 1 else ("DRAW" if res == 0 else "LOSS")
        disp._draw_board(axes[i], board, title=f"{lab} | {outcome}")

    plt.tight_layout()
    plt.show()
