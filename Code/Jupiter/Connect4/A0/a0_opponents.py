# A0/a0_opponents.py
from __future__ import annotations
import re, random
import numpy as np
from typing import List

from C4.connect4_lookahead import Connect4Lookahead

class Opponent:
    id: str = "base"
    name: str = "Base"
    def decide(self, env, board: np.ndarray, player: int) -> int:
        raise NotImplementedError

class LookaheadOpponent(Opponent):
    """
    One class covers: "random", "L1", "L2"... "L7"
    - "random": uniform legal move
    - "L1": win → block → center → random (uses lookahead helpers)
    - "Lk" (k>=2): k-ply lookahead via Connect4Lookahead.n_step_lookahead(depth=k)
    """
    def __init__(self, level: str):
        self.id = level
        self.lk = Connect4Lookahead()
        self.level = level.strip().lower()
        self.name = f"{level.upper()} (Lookahead)"

        m = re.fullmatch(r"l(\d+)", self.level)
        self.k = int(m.group(1)) if m else None  # None for "random"

    def _legal_cols(self, env, board):
        legal = env.legal_moves(board)
        return [int(c) for c in np.where(legal)[0]]

    def decide(self, env, board: np.ndarray, player: int) -> int:
        # ensure 2D int board
        b = np.asarray(board, dtype=np.int8)

        # RANDOM
        if self.level == "random":
            cols = self._legal_cols(env, b)
            return random.choice(cols) if cols else 0

        # L1: immediate win → block opponent immediate win → center → fallback
        if self.k is None or self.k == 1:
            legal = env.legal_moves(b)

            # win now
            for c in range(7):
                if not legal[c]: continue
                if self.lk.check_win(env.next_board(b, c, player), player):
                    return c

            # block opp win
            for c in range(7):
                if not legal[c]: continue
                if self.lk.check_win(env.next_board(b, c, -player), -player):
                    return c

            # center preference
            for c in [3,2,4,1,5,0,6]:
                if legal[c]: return c

            cols = self._legal_cols(env, b)
            return random.choice(cols) if cols else 0

        # Lk (k >= 2): k-ply lookahead (lower k for speed if needed)
        depth = max(2, self.k)
        return int(self.lk.n_step_lookahead(b, player, depth=depth))

# ----- helpers -----
def build_suite(ids: List[str]) -> List[Opponent]:
    """ids examples: ['random','L1','L2','L5']"""
    return [LookaheadOpponent(i) for i in ids]

def list_opponents() -> List[str]:
    return ["random", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
