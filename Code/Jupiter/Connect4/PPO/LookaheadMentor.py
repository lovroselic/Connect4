#PPO.LookaheadMentor.py
import numpy as np
import random
import torch
from C4.fast_connect4_lookahead import Connect4Lookahead


class LookaheadMentor:
    """
    Wraps bitboard lookahead so PPO can ask:
        'Given this 2-channel state, what would Lk do?'
    Returns a batch of best actions (columns 0–6).
    """

    def __init__(self, depth: int, device: torch.device):
        self.depth = depth
        self.device = device
        self.engine = Connect4Lookahead()

    @staticmethod
    def _decode_state_to_board(state_2ch: np.ndarray) -> np.ndarray:
        """
        state_2ch: [2, 6, 7] float32, agent-centric:
          channel 0 -> agent discs
          channel 1 -> opponent discs
        We rebuild a board {0,1,2} where player 1 = current agent, 2 = opponent.
        """
        agent_plane = state_2ch[0]
        opp_plane   = state_2ch[1]

        board = np.zeros_like(agent_plane, dtype=np.int8)
        board[agent_plane > 0.5] = 1
        board[opp_plane   > 0.5] = 2
        return board

    def __call__(
        self,
        states: torch.Tensor,      # [B, 2, 6, 7]
        legal_masks: torch.Tensor, # [B, 7] bool
        depth: int | None = None,
    ) -> torch.Tensor:
        depth = depth or self.depth

        states_np = states.detach().cpu().numpy()
        legal_np  = legal_masks.detach().cpu().numpy()

        actions = []
        for s, legal_row in zip(states_np, legal_np):
            board = self._decode_state_to_board(s)
            legal_cols = np.nonzero(legal_row)[0]
            if len(legal_cols) == 0:
                # super defensive fallback; should not really happen
                actions.append(3)
                continue

            # ask bitboard mentor
            # adjust method name if your class uses something else
            a_star = self.engine.n_step_lookahead(board, player=1, depth=depth)

            # safety: ensure mentor move is legal under current mask
            if a_star not in legal_cols:
                a_star = int(random.choice(legal_cols))

            actions.append(int(a_star))

        return torch.tensor(actions, device=self.device, dtype=torch.long)
