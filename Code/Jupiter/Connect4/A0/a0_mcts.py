# a0_mcts.py
# Compact AlphaZero-style MCTS with PUCT for 6x7 Connect-4.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math, numpy as np, torch
from torch import nn

@dataclass
class Node:
    to_play: int                          # player to move at this node (+1 or -1)
    P: Optional[np.ndarray] = None        # prior over actions [7]
    N: Optional[np.ndarray] = None        # visit counts [7]
    W: Optional[np.ndarray] = None        # total value [7]
    Q: Optional[np.ndarray] = None        # mean value [7]
    children: Optional[Dict[int, "Node"]] = None
    is_expanded: bool = False

    def __post_init__(self):
        if self.N is None:
            self.N = np.zeros(7, dtype=np.int32)
        if self.W is None:
            self.W = np.zeros(7, dtype=np.float32)
        if self.Q is None:
            self.Q = np.zeros(7, dtype=np.float32)
        if self.children is None:
            self.children = {}

def _puct(Q: np.ndarray, P: np.ndarray, N_sum: int, N_a: int, c_puct: float) -> float:
    return Q + c_puct * P * math.sqrt(max(1, N_sum)) / (1 + N_a)

def _add_dirichlet(P: np.ndarray, legal: np.ndarray, eps: float = 0.25, alpha: float = 0.3) -> np.ndarray:
    idx = np.where(legal)[0]
    if len(idx) == 0:
        return P
    noise = np.random.dirichlet([alpha] * len(idx))
    Pn = P.copy()
    Pn[idx] = (1 - eps) * P[idx] + eps * noise
    return Pn

class MCTS:
    """
    Minimal AlphaZero MCTS using:
      - Env API: legal_moves(board)->bool[7], next_board(board,a,player)->board,
                 check_winner(board)->{-1,0,+1}, is_terminal(board)->bool,
                 encode_state(board,to_play)->torch.FloatTensor[1,4,6,7]
      - Net API: forward(x, legal_mask=None)->(policy_logits[B,7], value[B] in [-1,1])
    """
    def __init__(self, net: nn.Module, env, c_puct: float = 2.0, sims: int = 400, device: str | torch.device = "cpu"):
           self.env = env
           self.c_puct = float(c_puct)
           self.sims = int(sims)
           self.device = torch.device(device)
           self.net = net.to(self.device).eval()   # <<< ensure model lives on the same device

    @torch.no_grad()
    def _nn_prior_value(self, board: np.ndarray, to_play: int) -> Tuple[np.ndarray, float]:
        legal = self.env.legal_moves(board)                       # bool[7]
        x = self.env.encode_state(board, to_play).float().to(self.device)   # [1,4,6,7]
        mask = torch.as_tensor(legal, dtype=torch.bool, device=self.device).unsqueeze(0) # [1,7]
        logits, v = self.net(x, legal_mask=mask)
        p = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()          # [7]
        p[~legal] = 0.0
        s = p.sum()
        if s <= 0:
            # uniform over legal if numeric underflow
            legal_idx = np.where(legal)[0]
            p[:] = 0.0
            if len(legal_idx) > 0:
                p[legal_idx] = 1.0 / len(legal_idx)
        else:
            p /= s
        return p, float(v.item())

    def run(self, board: np.ndarray, to_play: int, temperature: float = 1.0, add_noise: bool = True) -> Tuple[np.ndarray, Node]:
        root = Node(to_play=to_play)
        P_root, _ = self._nn_prior_value(board, to_play)
        legal_root = self.env.legal_moves(board)
        if add_noise:
            P_root = _add_dirichlet(P_root, legal_root)
        root.P = P_root
        root.is_expanded = True

        for _ in range(self.sims):
            self._simulate(root, board, to_play)

        visits = root.N.astype(np.float32)
        visits[~legal_root] = 0.0
        if temperature <= 0.0:
            pi = np.zeros_like(visits, dtype=np.float32)
            legal_idx = np.where(legal_root)[0]
            best = legal_idx[np.argmax(visits[legal_idx])] if len(legal_idx) else 0
            pi[best] = 1.0
        else:
            vtemp = visits ** (1.0 / temperature)
            s = vtemp.sum()
            pi = vtemp / s if s > 0 else np.ones_like(vtemp) / max(1, vtemp.size)
        return pi, root

    def _simulate(self, node: Node, board: np.ndarray, to_play: int) -> None:
        path = []
        cur_node = node
        cur_board = board
        cur_player = to_play

        # Selection
        while cur_node.is_expanded:
            legal = self.env.legal_moves(cur_board)
            N_sum = int(cur_node.N.sum())
            ucb = np.full(7, -1e9, dtype=np.float32)
            for a in range(7):
                if legal[a]:
                    ucb[a] = _puct(cur_node.Q[a], cur_node.P[a], N_sum, cur_node.N[a], self.c_puct)
            a = int(np.argmax(ucb))
            path.append((cur_node, a))
            next_board = self.env.next_board(cur_board, a, cur_player)
            next_player = -cur_player
            if a not in cur_node.children:
                cur_board = next_board
                cur_player = next_player
                break
            cur_node = cur_node.children[a]
            cur_board = next_board
            cur_player = next_player

        # Expansion / Evaluation
        if self.env.is_terminal(cur_board):
            winner = self.env.check_winner(cur_board)
            # Value from the perspective of player to move at this leaf (cur_player)
            if winner == 0:
                v = 0.0
            elif winner == cur_player:
                v = +1.0
            else:
                v = -1.0
        else:
            child = Node(to_play=cur_player)
            P, v = self._nn_prior_value(cur_board, cur_player)
            child.P = P
            child.is_expanded = True
            # Attach child to last edge in path
            if path:
                parent, act = path[-1]
                parent.children[act] = child
            cur_node = child  # continue for clarity

        # Backup
        # v is from the perspective of player-to-move at the leaf.
        for parent, act in reversed(path):
            parent.N[act] += 1
            parent.W[act] += v
            parent.Q[act] = parent.W[act] / parent.N[act]
            v = -v
