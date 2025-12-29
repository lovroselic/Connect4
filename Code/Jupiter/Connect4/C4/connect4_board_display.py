# C4/connect4_board_display.py
"""
connect4_board_display.py

Board visualization helpers for the "new line" models (supervised CNet192 + PPO),
built to work with PPO.ppo_agent_eval (no legacy imports).

Key behaviors:
- For each opponent label, we play TWO games:
    1) policy starts (policy plays as +1 internally)
    2) opponent starts (policy plays as -1 internally)
- Display is POLICY-CENTRIC:
    - policy discs are always drawn as O (gold)  -> board value -1 in display-space
    - opponent discs are always drawn as X (red) -> board value +1 in display-space
  (This makes comparisons much easier.)

Board convention (engine-space):
- numpy array (6,7), values in {-1, 0, +1}
- row 0 is top, row 5 is bottom

Usage:
  from C4.connect4_board_display import display_final_boards_PPO
  from C4.fast_connect4_lookahead import Connect4Lookahead

  la = Connect4Lookahead()
  display_final_boards_PPO(policy, ["Random", "Leftmost", "Lookahead-1"], lookahead=la)
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from PPO.ppo_agent_eval import play_one_game, make_opponent


# ----------------------------- display helpers -----------------------------

class Connect4_BoardDisplayer:
    def __init__(self):
        self.rows = 6
        self.cols = 7

    def _draw_board(self, ax, board_policy_view: np.ndarray, title: str = "") -> None:
        """
        board_policy_view:
          - policy == -1 (gold O)
          - opponent == +1 (red X)
          - empty == 0
        """
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(self.rows - 0.5, -0.5)
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_title(title)

        for r in range(self.rows):
            for c in range(self.cols):
                v = int(board_policy_view[r, c])

                if v == -1:
                    face = "gold"   # policy (O)
                    mark = "O"
                elif v == 1:
                    face = "red"    # opponent (X)
                    mark = "X"
                else:
                    face = "white"
                    mark = ""

                ax.add_patch(Circle((c, r), 0.42, facecolor=face, edgecolor="black", linewidth=1))

                if mark:
                    ax.text(
                        c, r, mark,
                        ha="center", va="center",
                        fontsize=18, fontweight="bold",
                        color="black",
                    )

        ax.grid(True, linestyle="--", alpha=0.25)

    def show_board(self, board_policy_view: np.ndarray, title: str = "") -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        self._draw_board(ax, board_policy_view, title=title)
        plt.show()


def _to_policy_view(board_engine: np.ndarray, policy_player: int) -> np.ndarray:
    """
    Convert engine-space board into policy-centric display-space:

    We want policy discs to be -1 always.
      - if policy_player == +1: policy discs are +1 => flip signs
      - if policy_player == -1: policy discs are -1 => keep as-is
    """
    b = board_engine
    if policy_player == 1:
        return (-b).astype(np.int8, copy=False)
    return b.astype(np.int8, copy=False)


def _policy_pov_result(winner: int, policy_player: int) -> int:
    """Return result from policy POV: +1 win, 0 draw, -1 loss."""
    if winner == 0:
        return 0
    return 1 if winner == policy_player else -1


def _outcome_str(res: int) -> str:
    return "WIN" if res == 1 else ("DRAW" if res == 0 else "LOSS")


# ----------------------------- public API -----------------------------

def play_one_game_for_display(
    policy,
    opponent_label: str,
    *,
    lookahead=None,
    device=None,
    seed: int = 666,
    policy_player: int = 1,
    policy_deterministic: bool = True,
    policy_temperature: float = 1.0,
) -> Tuple[int, np.ndarray]:
    """
    Plays one game and returns:
      (result_from_policy_pov, final_board_policy_view)
    """
    if device is None:
        device = next(policy.parameters()).device

    rng = np.random.default_rng(seed)
    opp = make_opponent(opponent_label, lookahead=lookahead, rng=rng)

    gr = play_one_game(
        policy=policy,
        opponent=opp,
        device=device,
        seed=seed,
        policy_player=int(policy_player),
        policy_deterministic=bool(policy_deterministic),
        policy_temperature=float(policy_temperature),
    )

    res = _policy_pov_result(gr.winner, policy_player=int(policy_player))
    board_view = _to_policy_view(gr.final_board, policy_player=int(policy_player))
    return res, board_view


def display_final_boards_PPO(
    policy,
    opponent_labels: Iterable[str],
    *,
    lookahead=None,
    seed: int = 666,
    n_pairs: int = 1,
    device=None,
    policy_deterministic: bool = True,
    policy_temperature: float = 1.0,
) -> None:
    """
    For each opponent label, play TWO games per pair:
      - policy starts once (policy_player=+1)
      - opponent starts once (policy_player=-1)

    POLICY is always drawn as O (gold) in the plots.
    """
    labels = [str(x) for x in opponent_labels]
    if not labels:
        return

    if device is None:
        device = next(policy.parameters()).device

    disp = Connect4_BoardDisplayer()

    rows = len(labels) * max(1, int(n_pairs))
    cols = 2  # (policy starts, policy second)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    if rows == 1:
        axes = np.array([axes])  # shape (1,2)

    r = 0
    for pi in range(max(1, int(n_pairs))):
        for lab in labels:
            # deterministic seed spacing, but not identical across matchups
            s0 = seed + (pi * 10_000) + (r * 101) + 0
            s1 = seed + (pi * 10_000) + (r * 101) + 1

            # Game A: policy starts (policy_player=+1 in engine)
            resA, boardA = play_one_game_for_display(
                policy, lab,
                lookahead=lookahead,
                device=device,
                seed=s0,
                policy_player=1,
                policy_deterministic=policy_deterministic,
                policy_temperature=policy_temperature,
            )

            # Game B: opponent starts (policy_player=-1 in engine)
            resB, boardB = play_one_game_for_display(
                policy, lab,
                lookahead=lookahead,
                device=device,
                seed=s1,
                policy_player=-1,
                policy_deterministic=policy_deterministic,
                policy_temperature=policy_temperature,
            )

            tA = f"{lab} | policy starts | {_outcome_str(resA)}"
            tB = f"{lab} | policy second | {_outcome_str(resB)}"

            disp._draw_board(axes[r, 0], boardA, title=tA)
            disp._draw_board(axes[r, 1], boardB, title=tB)

            r += 1

    plt.tight_layout()
    plt.show()
