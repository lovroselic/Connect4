# C4/connect4_board_display.py
"""
connect4_board_display.py

Board visualization helpers for the "new line" models.

Supports:
- PPO / supervised 1-channel policies (CNet192 family) via PPO.ppo_agent_eval
- DQN agents (q_net + board_to_state) via local bitboard runner

Key behaviors:
- For each opponent label, we play TWO games per pair:
    1) policy/agent starts (plays as +1 internally)
    2) opponent starts (policy/agent plays as -1 internally)
- Display is POLICY/AGENT-CENTRIC:
    - policy/agent discs are always drawn as O (gold)
    - opponent discs are always drawn as X (red)

Board convention (engine-space for display runner):
- numpy array (6,7), values in {-1, 0, +1}
- row 0 is top, row 5 is bottom

Usage:
  from C4.connect4_board_display import display_final_boards_PPO, display_final_boards_DQN
  from C4.fast_connect4_lookahead import Connect4Lookahead

  la = Connect4Lookahead()

  display_final_boards_PPO(policy, ["Random", "Center", "Lookahead-7"], lookahead=la)
  display_final_boards_DQN(agent,  ["Leftmost","Center","Random","Lookahead-1","Lookahead-7"], lookahead=la)
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, List, Any, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch

from PPO.ppo_agent_eval import play_one_game as play_one_game_ppo
from PPO.ppo_agent_eval import make_opponent as make_opponent_eval

from C4.connect4_env import (
    ROWS, COLS, STRIDE,
    TOP_MASK, FULL_MASK,
    _bb_has_won,
    UINT,
)

# ----------------------------- display helpers -----------------------------

class Connect4_BoardDisplayer:
    def __init__(self):
        self.rows = 6
        self.cols = 7

    def _draw_board(self, ax, board_policy_view: np.ndarray, title: str = "") -> None:
        """
        board_policy_view:
          - policy/agent == -1 (gold O)
          - opponent    == +1 (red X)
          - empty       == 0
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
                    face = "gold"   # policy/agent (O)
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

    We want policy/agent discs to be -1 always.
      - if policy_player == +1: policy discs are +1 => flip signs
      - if policy_player == -1: policy discs are -1 => keep as-is
    """
    b = np.asarray(board_engine, dtype=np.int8)
    if int(policy_player) == 1:
        return (-b).astype(np.int8, copy=False)
    return b.astype(np.int8, copy=False)


def _policy_pov_result(winner: int, policy_player: int) -> int:
    """Return result from policy POV: +1 win, 0 draw, -1 loss."""
    if int(winner) == 0:
        return 0
    return 1 if int(winner) == int(policy_player) else -1


def _outcome_str(res: int) -> str:
    return "WIN" if res == 1 else ("DRAW" if res == 0 else "LOSS")


def _as_uint64(x: np.uint64 | int) -> np.uint64:
    return UINT(int(x))


def _legal_cols_from_mask(mask: np.uint64) -> List[int]:
    m = _as_uint64(mask)
    out: List[int] = []
    for c in range(COLS):
        if (m & TOP_MASK[c]) == 0:
            out.append(c)
    return out


def _apply_action_inplace(
    board: np.ndarray,
    heights: np.ndarray,
    pos1: np.uint64,
    pos2: np.uint64,
    mask: np.uint64,
    col: int,
    player: int,
) -> Tuple[np.uint64, np.uint64, np.uint64]:
    c = int(col)
    h = int(heights[c])
    bit = _as_uint64(1) << _as_uint64(c * STRIDE + h)
    mask = mask | bit
    if player == 1:
        pos1 = pos1 | bit
    else:
        pos2 = pos2 | bit
    heights[c] = h + 1
    board[ROWS - 1 - h, c] = np.int8(player)
    return pos1, pos2, mask


# =============================
# PPO DISPLAY (your version)
# =============================

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
    opp = make_opponent_eval(opponent_label, lookahead=lookahead, rng=rng)

    gr = play_one_game_ppo(
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
        axes = np.array([axes])

    r = 0
    for pi in range(max(1, int(n_pairs))):
        for lab in labels:
            s0 = seed + (pi * 10_000) + (r * 101) + 0
            s1 = seed + (pi * 10_000) + (r * 101) + 1

            resA, boardA = play_one_game_for_display(
                policy, lab,
                lookahead=lookahead,
                device=device,
                seed=s0,
                policy_player=1,
                policy_deterministic=policy_deterministic,
                policy_temperature=policy_temperature,
            )

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


# =============================
# DQN DISPLAY (new)
# =============================

def _infer_device_from_agent(agent) -> torch.device:
    q_net = getattr(agent, "q_net", None)
    if q_net is None:
        raise ValueError("agent must have .q_net")
    for p in q_net.parameters():
        return p.device
    return torch.device("cpu")


def _state_tensor_from_board(agent, board_pm1: np.ndarray, player_pm1: int, device: torch.device) -> torch.Tensor:
    """
    Robust adapter for agent.board_to_state.

    Try:
      1) board in {-1,0,+1}, player in {+1,-1}
      2) board in {0,1,2}, player in {1,2}

    Returns (1,C,6,7) float32 tensor on device.
    """
    fn = getattr(agent, "board_to_state", None)
    if fn is None:
        raise ValueError("agent must have .board_to_state(board, player)")

    b = np.asarray(board_pm1, dtype=np.int8)

    # attempt #1: +/-1 convention
    try:
        st = fn(b, int(player_pm1))
    except Exception:
        st = None

    # attempt #2: 0/1/2 convention
    if st is None:
        b12 = np.zeros((ROWS, COLS), dtype=np.int8)
        b12[b == 1] = 1
        b12[b == -1] = 2
        p12 = 1 if int(player_pm1) == 1 else 2
        st = fn(b12, int(p12))

    if isinstance(st, torch.Tensor):
        x = st
    else:
        x = torch.from_numpy(np.asarray(st))

    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"board_to_state must yield (C,6,7) or (1,C,6,7), got {tuple(x.shape)}")

    return x.to(device=device, dtype=torch.float32)


@torch.no_grad()
def _dqn_greedy_action(
    agent,
    board_pm1: np.ndarray,
    player_pm1: int,
    mask: np.uint64,
    device: torch.device,
) -> int:
    q_net = getattr(agent, "q_net", None)
    if q_net is None:
        raise ValueError("agent must have .q_net")

    legal = _legal_cols_from_mask(mask)
    if not legal:
        return 0

    x = _state_tensor_from_board(agent, board_pm1, player_pm1, device=device)

    was_training = bool(getattr(q_net, "training", False))
    q_net.eval()
    out = q_net(x)
    if was_training:
        q_net.train(True)

    if isinstance(out, (tuple, list)):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise TypeError("q_net output must be a Tensor (or tuple/list with first Tensor).")

    q = out.detach().cpu().numpy().astype(np.float64, copy=False)
    if q.ndim == 2:
        q = q[0]
    if q.shape[0] != COLS:
        raise ValueError(f"q_net must output length {COLS}, got {q.shape}")

    q_masked = q.copy()
    illegal = set(range(COLS)) - set(legal)
    for c in illegal:
        q_masked[c] = -1e18

    # center tiebreak
    best = np.max([q_masked[c] for c in legal])
    tied = [c for c in legal if abs(float(q_masked[c]) - float(best)) <= 1e-12]
    if len(tied) == 1:
        return int(tied[0])
    return int(sorted(tied, key=lambda c: (abs(c - 3), c))[0])


def _play_one_game_dqn_bitboard(
    agent,
    opponent_label: str,
    *,
    lookahead=None,
    device: torch.device,
    seed: int,
    agent_player: int,
) -> Tuple[int, np.ndarray]:
    """
    Plays one game using bitboards for speed, returns:
      (winner in {+1,-1,0}, final_board (6,7) int8 in {-1,0,+1})
    """
    rng = np.random.default_rng(int(seed))
    opp = make_opponent_eval(opponent_label, lookahead=lookahead, rng=rng)

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    heights = np.zeros(COLS, dtype=np.int8)

    pos1 = UINT(0)
    pos2 = UINT(0)
    mask = UINT(0)

    player = 1

    for _ in range(42):
        if player == int(agent_player):
            a = _dqn_greedy_action(agent, board, player, mask, device=device)
        else:
            a = int(opp(board, player, mask))

        # illegal => mover loses instantly
        if (a < 0) or (a >= COLS) or ((mask & TOP_MASK[int(a)]) != 0):
            return (-player), board

        pos1, pos2, mask = _apply_action_inplace(board, heights, pos1, pos2, mask, int(a), player)

        me_bb = pos1 if player == 1 else pos2
        if _bb_has_won(me_bb, STRIDE):
            return player, board

        if mask == FULL_MASK:
            return 0, board

        player = -player

    return 0, board


def play_one_game_for_display_DQN(
    agent,
    opponent_label: str,
    *,
    lookahead=None,
    device: Optional[torch.device] = None,
    seed: int = 666,
    agent_player: int = 1,
) -> Tuple[int, np.ndarray]:
    """
    Plays one game and returns:
      (result_from_agent_pov, final_board_policy_view)
    """
    if device is None:
        device = _infer_device_from_agent(agent)

    winner, board_engine = _play_one_game_dqn_bitboard(
        agent,
        opponent_label,
        lookahead=lookahead,
        device=device,
        seed=int(seed),
        agent_player=int(agent_player),
    )

    res = _policy_pov_result(winner, policy_player=int(agent_player))
    board_view = _to_policy_view(board_engine, policy_player=int(agent_player))
    return res, board_view


def display_final_boards_DQN(
    agent,
    opponent_labels: Iterable[str],
    *,
    lookahead=None,
    seed: int = 666,
    n_pairs: int = 1,
    device: Optional[torch.device] = None,
) -> None:
    """
    For each opponent label, play TWO games per pair:
      - agent starts once (agent_player=+1)
      - opponent starts once (agent_player=-1)

    AGENT is always drawn as O (gold) in the plots.
    """
    labels = [str(x) for x in opponent_labels]
    if not labels:
        return

    if device is None:
        device = _infer_device_from_agent(agent)

    disp = Connect4_BoardDisplayer()

    rows = len(labels) * max(1, int(n_pairs))
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    if rows == 1:
        axes = np.array([axes])

    r = 0
    for pi in range(max(1, int(n_pairs))):
        for lab in labels:
            s0 = seed + (pi * 10_000) + (r * 101) + 0
            s1 = seed + (pi * 10_000) + (r * 101) + 1

            resA, boardA = play_one_game_for_display_DQN(
                agent, lab,
                lookahead=lookahead,
                device=device,
                seed=s0,
                agent_player=1,
            )
            resB, boardB = play_one_game_for_display_DQN(
                agent, lab,
                lookahead=lookahead,
                device=device,
                seed=s1,
                agent_player=-1,
            )

            tA = f"{lab} | agent starts | {_outcome_str(resA)}"
            tB = f"{lab} | agent second | {_outcome_str(resB)}"

            disp._draw_board(axes[r, 0], boardA, title=tA)
            disp._draw_board(axes[r, 1], boardB, title=tB)

            r += 1

    plt.tight_layout()
    plt.show()


__all__ = [
    "Connect4_BoardDisplayer",
    "display_final_boards_PPO",
    "display_final_boards_DQN",
    "play_one_game_for_display",
    "play_one_game_for_display_DQN",
]