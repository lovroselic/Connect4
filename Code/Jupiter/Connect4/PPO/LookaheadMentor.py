# PPO/LookaheadMentor.py
import numpy as np
import random
import torch
from C4.fast_connect4_lookahead import Connect4Lookahead


class LookaheadMentor:
    """
    Wraps bitboard lookahead so PPO can ask:
        'Given this 1-channel POV state, what would Lk do?'

    Expected state convention (single-channel POV):
      - states: [B, 1, 6, 7] float32
      - values: +1 for current player discs, -1 for opponent discs, 0 empty
      - (i.e. state = board * player_to_move)

    Returns:
      - actions: [B] int64 (columns 0..6)
    """

    def __init__(self, depth: int, device: torch.device):
        self.depth = int(depth)
        self.device = device
        self.engine = Connect4Lookahead()

    @staticmethod
    def _decode_state_to_board(state_1ch: np.ndarray) -> np.ndarray:
        """
        state_1ch: [1, 6, 7] or [6, 7] float32 (POV).
        Rebuild a board int8 in {-1, 0, +1} (same POV), so we can call lookahead
        with player=+1 (current POV player).
        """
        x = np.asarray(state_1ch)

        if x.ndim == 3:
            # [1,6,7] -> [6,7]
            x = x[0]
        elif x.ndim != 2:
            raise ValueError(f"Expected state shape [6,7] or [1,6,7], got {x.shape}")

        board = np.zeros_like(x, dtype=np.int8)

        # thresholds tolerate float noise / logits-ish rounding
        board[x > 0.5] = 1
        board[x < -0.5] = -1

        return board

    def __call__(
        self,
        states: torch.Tensor,       # [B, 1, 6, 7] (or [B, 6, 7] tolerated)
        legal_masks: torch.Tensor,  # [B, 7] bool (or 0/1)
        depth: int | None = None,
    ) -> torch.Tensor:
        depth = int(self.depth if depth is None else depth)

        states_np = states.detach().cpu().numpy()
        legal_np  = legal_masks.detach().cpu().numpy()

        actions = []
        for s, legal_row in zip(states_np, legal_np):
            # tolerate [6,7] per-sample too
            board = self._decode_state_to_board(s)

            # legal_row can be bool or 0/1
            legal_cols = np.flatnonzero(legal_row > 0.5)
            if legal_cols.size == 0:
                actions.append(3)  # ultra-defensive fallback
                continue

            # In POV space, "current player" is always +1
            a_star = int(self.engine.n_step_lookahead(board, player=+1, depth=depth))

            # safety: ensure mentor move is legal under current mask
            if a_star not in legal_cols:
                a_star = int(random.choice(legal_cols.tolist()))

            actions.append(a_star)

        return torch.tensor(actions, device=self.device, dtype=torch.long)
