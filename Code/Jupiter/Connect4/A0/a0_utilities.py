# a0_utilities.py
# Utilities for AlphaZero-style Connect-4.

from __future__ import annotations
import numpy as np
import torch

BOARD_H, BOARD_W = 6, 7

def state_encoder(board: np.ndarray, to_play: int) -> torch.Tensor:
    """Player-relative 4-channel encoder -> [1,4,6,7]."""
    assert board.shape == (BOARD_H, BOARD_W), f"Expected {(BOARD_H, BOARD_W)}, got {board.shape}"
    cur = (board == to_play).astype(np.float32)
    opp = (board == -to_play).astype(np.float32)

    rows = np.linspace(0.0, 1.0, BOARD_H, dtype=np.float32).reshape(BOARD_H, 1)
    cols = np.linspace(0.0, 1.0, BOARD_W, dtype=np.float32).reshape(1, BOARD_W)
    row_plane = np.repeat(rows, BOARD_W, axis=1)
    col_plane = np.repeat(cols, BOARD_H, axis=0)

    x = np.stack([cur, opp, row_plane, col_plane], axis=0)   # [4,6,7]
    return torch.from_numpy(x).unsqueeze(0)                   # [1,4,6,7]

def legal_mask_from_board(board: np.ndarray) -> torch.Tensor:
    """Top cell empty => legal. Returns BoolTensor [7]."""
    return torch.from_numpy(board[0, :] == 0)

def mask_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """
    Replace illegal action logits with −inf (safe for softmax/log_softmax).
    logits: [B,7], legal_mask: [B,7] or [7]
    """
    if legal_mask.dim() == 1:
        legal_mask = legal_mask.unsqueeze(0)
    legal_mask = legal_mask.to(device=logits.device, dtype=torch.bool)
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    return torch.where(legal_mask, logits, neg_inf)
