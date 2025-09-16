# az_env.py
# Minimal, fast Connect-4 environment adapter for AlphaZero-style MCTS.
# Board convention: numpy array shape (6, 7) with values in {-1, 0, +1}.
# +1 and -1 are player stones; 0 is empty.
# Rows increase downward (row 0 is the TOP). A "drop" fills from the bottom.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
import torch

BOARD_H, BOARD_W = 6, 7

@dataclass
class StepResult:
    board: np.ndarray  # next board (copied)
    winner: int        # -1, 0, +1
    done: bool

def _lowest_empty_row(board: np.ndarray, col: int) -> int:
    # Find the lowest empty (value==0) cell in column 'col' (bottom is BOARD_H-1).
    col_vals = board[:, col]
    for r in range(BOARD_H - 1, -1, -1):
        if col_vals[r] == 0:
            return r
    return -1  # full

def _check_winner(board: np.ndarray) -> int:
    H, W = BOARD_H, BOARD_W
    b = board

    # Horizontal
    for r in range(H):
        row = b[r]
        for c in range(W - 3):
            v = row[c]
            if v != 0 and row[c+1] == v and row[c+2] == v and row[c+3] == v:
                return int(v)

    # Vertical
    for r in range(H - 3):
        for c in range(W):
            v = b[r, c]
            if v != 0 and b[r+1, c] == v and b[r+2, c] == v and b[r+3, c] == v:
                return int(v)

    # Diagonal down-right
    for r in range(H - 3):
        for c in range(W - 3):
            v = b[r, c]
            if v != 0 and b[r+1, c+1] == v and b[r+2, c+2] == v and b[r+3, c+3] == v:
                return int(v)

    # Diagonal down-left
    for r in range(H - 3):
        for c in range(3, W):
            v = b[r, c]
            if v != 0 and b[r+1, c-1] == v and b[r+2, c-2] == v and b[r+3, c-3] == v:
                return int(v)

    return 0

def _board_full(board: np.ndarray) -> bool:
    return not (board == 0).any()

class EnvAdapter:
    """
    A small, self-contained adapter exposing exactly what MCTS needs:
      - legal_moves(board) -> np.ndarray(bool, shape [7])
      - next_board(board, action, player) -> np.ndarray(int8, shape [6,7])  (copy; no mutation)
      - check_winner(board) -> {-1,0,+1}
      - is_terminal(board) -> bool
      - encode_state(board, to_play) -> torch.FloatTensor[1,4,6,7]

    Also includes:
      - reset() -> empty board
      - pretty(board) -> human-friendly string for debugging
      - mirror(board) / color_swap(board) for data augmentation

    You can inject your own 4-channel encoder via __init__(state_encoder=...).
    """
    def __init__(self, state_encoder: Optional[Callable[[np.ndarray, int], torch.Tensor]] = None):
        self.state_encoder = state_encoder

    # ---- Core API for MCTS ----
    def reset(self) -> np.ndarray:
        return np.zeros((BOARD_H, BOARD_W), dtype=np.int8)

    def legal_moves(self, board: np.ndarray) -> np.ndarray:
        # A move is legal if the TOP cell in that column is empty (row 0).
        top_row = board[0, :]
        return top_row == 0  # shape (7,) bool

    def next_board(self, board: np.ndarray, action: int, player: int) -> np.ndarray:
        assert 0 <= action < BOARD_W, f"action out of range: {action}"
        assert player in (-1, +1), f"player must be -1 or +1, got {player}"
        r = _lowest_empty_row(board, action)
        if r < 0:
            raise ValueError(f"Column {action} is full")
        nb = board.copy()
        nb[r, action] = player
        return nb

    def check_winner(self, board: np.ndarray) -> int:
        return _check_winner(board)

    def is_terminal(self, board: np.ndarray) -> bool:
        return _check_winner(board) != 0 or _board_full(board)

    def encode_state(self, board: np.ndarray, to_play: int) -> torch.Tensor:
        """
        Default 4-channel encoder:
          - current player stones (1.0 where board == to_play)
          - opponent stones (1.0 where board == -to_play)
          - row encoding  (normalized 0..1 from top (0) to bottom (1))
          - col encoding  (normalized 0..1 from left (0) to right (1))
        Returns: torch.FloatTensor of shape [1, 4, 6, 7]
        """
        if self.state_encoder is not None:
            return self.state_encoder(board, to_play)

        b = board
        cur = (b == to_play).astype(np.float32)
        opp = (b == -to_play).astype(np.float32)

        # Positional encodings
        rows = np.linspace(0.0, 1.0, BOARD_H, dtype=np.float32).reshape(BOARD_H, 1)
        cols = np.linspace(0.0, 1.0, BOARD_W, dtype=np.float32).reshape(1, BOARD_W)
        row_plane = np.repeat(rows, BOARD_W, axis=1)
        col_plane = np.repeat(cols, BOARD_H, axis=0)

        x = np.stack([cur, opp, row_plane, col_plane], axis=0)  # [4,6,7]
        return torch.from_numpy(x).unsqueeze(0)  # [1,4,6,7]

    # ---- Utilities ----
    def pretty(self, board: np.ndarray) -> str:
        # Map: +1 -> 'X', -1 -> 'O', 0 -> '.'
        m = {1: 'X', -1: 'O', 0: '.'}
        lines = []
        for r in range(BOARD_H):
            line = ''.join(m[int(v)] for v in board[r, :])
            lines.append(line)
        footer = '0123456'
        return '\n'.join(lines + [footer])

    @staticmethod
    def mirror(board: np.ndarray) -> np.ndarray:
        # Horizontal flip (mirror over vertical axis)
        return np.ascontiguousarray(board[:, ::-1])

    @staticmethod
    def color_swap(board: np.ndarray) -> np.ndarray:
        # Multiply by -1 swaps players
        return (board * -1).astype(board.dtype)
