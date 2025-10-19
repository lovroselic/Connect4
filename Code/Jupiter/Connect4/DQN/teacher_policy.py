# DQN.teacher_policy.py
from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch

from C4.fast_connect4_lookahead import Connect4Lookahead  # only for type hints
from DQN.dqn_model import DQN

ROWS, COLS = 6, 7

def _planes4(board: np.ndarray) -> np.ndarray:
    """Build 4 channels: [+1 plane, -1 plane, row_code, col_code] with [-1,1] encodings."""
    p_pos = (board == 1).astype(np.float32)
    p_neg = (board == -1).astype(np.float32)
    row_plane = np.tile(np.linspace(-1, 1, ROWS, dtype=np.float32)[:, None], (1, COLS))
    col_plane = np.tile(np.linspace(-1, 1, COLS, dtype=np.float32)[None, :], (ROWS, 1))
    return np.stack([p_pos, p_neg, row_plane, col_plane], axis=0)  # (4,6,7)

class GreedyDQNPolicy:
    """Greedy action from a saved DQN checkpoint. Masks illegal moves."""
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.net = DQN().to(self.device).eval()
        sd = torch.load(weights_path, map_location=self.device, weights_only=True)
        state = sd["model_state_dict"] if isinstance(sd, dict) and "model_state_dict" in sd else sd
        self.net.load_state_dict(state, strict=False)

    @torch.no_grad()
    def __call__(self, board: np.ndarray, legal: List[int], player: int,
                 la: Optional[Connect4Lookahead] = None) -> int:
        # Orient board to +1 perspective
        b = board if player == 1 else -board
        x = torch.tensor(_planes4(b)[None], dtype=torch.float32, device=self.device)  # (1,4,6,7)
        q = self.net(x).squeeze(0).cpu().numpy()  # (7,)

        if not legal:  # safety fallback on terminal/full states
            return int(np.argmax(q))

        mask = np.full(COLS, -1e9, dtype=np.float32)
        mask[legal] = 0.0
        return int(np.argmax(q + mask))
