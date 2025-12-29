# C4.connect4_env_sparse.py
# Connect-4 env with sparse rewards, bitboard + optional numba acceleration

import numpy as np
from C4.fast_connect4_lookahead import Connect4Lookahead

# Optional: Numba-accelerated bitboard win check
try:
    from numba import njit, uint64, int32
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

ROWS, COLS = 6, 7
STRIDE = ROWS + 1  # Pascal Pons layout stride (7)
CENTER_COL = 3     # kept for compatibility; not used in rewards here

# ---- Bitboard masks (module-level, built once) ----
UINT = np.uint64

COL_MASK    = np.zeros(COLS, dtype=UINT)
TOP_MASK    = np.zeros(COLS, dtype=UINT)
BOTTOM_MASK = np.zeros(COLS, dtype=UINT)
FULL_MASK   = UINT(0)

for c in range(COLS):
    col_bits = UINT(0)
    for r in range(ROWS):
        col_bits |= UINT(1) << UINT(c * STRIDE + r)
    COL_MASK[c]    = col_bits
    BOTTOM_MASK[c] = UINT(1) << UINT(c * STRIDE + 0)
    TOP_MASK[c]    = UINT(1) << UINT(c * STRIDE + (ROWS - 1))
    FULL_MASK     |= col_bits

# ---- Numba win check ----
if _HAVE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _bb_has_won(bb: uint64, stride_i: int32) -> bool:
        # vertical
        m = bb & (bb >> uint64(1))
        if (m & (m >> uint64(2))) != uint64(0):
            return True
        # horizontal
        m = bb & (bb >> uint64(stride_i))
        if (m & (m >> uint64(2 * stride_i))) != uint64(0):
            return True
        # diag up-right
        m = bb & (bb >> uint64(stride_i + 1))
        if (m & (m >> uint64(2 * (stride_i + 1)))) != uint64(0):
            return True
        # diag up-left
        m = bb & (bb >> uint64(stride_i - 1))
        if (m & (m >> uint64(2 * (stride_i - 1)))) != uint64(0):
            return True
        return False
else:
    def _bb_has_won(bb: np.uint64, stride_i: int) -> bool:
        # vertical
        m = bb & (bb >> 1)
        if (m & (m >> 2)) != 0:
            return True
        # horizontal
        m = bb & (bb >> stride_i)
        if (m & (m >> (2 * stride_i))) != 0:
            return True
        # diag up-right
        m = bb & (bb >> (stride_i + 1))
        if (m & (m >> (2 * (stride_i + 1)))) != 0:
            return True
        # diag up-left
        m = bb & (bb >> (stride_i - 1))
        if (m & (m >> (2 * (stride_i - 1)))) != 0:
            return True
        return False


class Connect4EnvSparse:
    """
    Mover-centric Connect-4 environment with sparse rewards.

    - Observation: (4, 6, 7) planes:
        [0] agent pieces (perspective player)
        [1] opponent pieces
        [2] row coordinate plane in [-1, 1]
        [3] column coordinate plane in [-1, 1]

    - Reward (ALWAYS from the perspective of the player who just moved):
        Non-terminal step: 0.0
        Win:   +1.0
        Draw:   0.0
        Loss:  -1.0   (should not occur for mover-centric win detection,
                       kept for robustness)

    Illegal moves raise ValueError (expected to be prevented by action masking).
    """

    ROWS = ROWS
    COLS = COLS

    # Sparse reward constants
    WIN_REWARD   = 1.0
    DRAW_REWARD  = 0.0
    LOSS_REWARD  = -1.0
    STEP_REWARD  = 0.0  # all non-terminal moves

    # Precomputed coordinate planes (float32) reused for every state
    _ROW_PLANE = np.tile(
        np.linspace(-1, 1, ROWS, dtype=np.float32)[:, None],
        (1, COLS)
    )
    _COL_PLANE = np.tile(
        np.linspace(-1, 1, COLS, dtype=np.float32)[None, :],
        (ROWS, 1)
    )

    def __init__(self):
        # Kept for compatibility; not used in sparse reward calculation
        self.lookahead = Connect4Lookahead()
        self.reset()

    # ---------------------- Core API ----------------------

    def reset(self):
        # 2D board (top-first) kept for compatibility with agents/state builder
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = 1  # mover; {+1, -1}
        self.done = False
        self.winner = None
        self.ply = 0

        # bitboards + per-column heights for speed
        self._pos1 = UINT(0)
        self._pos2 = UINT(0)
        self._mask = UINT(0)
        self._heights = np.zeros(self.COLS, dtype=np.int8)  # stones per column [0..6]

        return self.get_state(perspective=self.current_player)

    def get_state(self, perspective=None) -> np.ndarray:
        """
        Return mover-centric state as (4, 6, 7) float32 tensor.
        """
        if perspective is None:
            perspective = self.current_player
        b = self.board
        p = perspective

        agent_plane = (b == p).astype(np.float32, copy=False)
        opp_plane   = (b == -p).astype(np.float32, copy=False)

        return np.stack(
            [agent_plane, opp_plane, self._ROW_PLANE, self._COL_PLANE],
            axis=0
        )

    def available_actions(self):
        """
        Return list of legal columns (0..6) where a move can be played.
        """
        return [c for c in range(self.COLS) if self._heights[c] < self.ROWS]

    def step(self, action):
        """
        Apply a move for current_player in column `action`.

        Returns:
            next_state (mover-centric, for next player if game continues),
            reward (float),
            done (bool)
        """
        if self.done:
            # Repeat terminal state with zero reward if stepped after done
            return self.get_state(perspective=self.current_player), 0.0, True

        c = int(action)
        # Illegal move -> explicit error (should be prevented by masking)
        if c < 0 or c >= self.COLS or self._heights[c] >= self.ROWS:
            raise ValueError(f"ILLEGAL MOVE DETECTED in column {c}!")

        # ---- Apply move: update bitboards, heights, and 2D board ----
        r_bot = int(self._heights[c])  # bottom-based row [0..5]
        bit   = UINT(1) << UINT(c * STRIDE + r_bot)

        self._mask |= bit
        if self.current_player == 1:
            self._pos1 |= bit
        else:
            self._pos2 |= bit
        self._heights[c] = r_bot + 1

        # 2D board (top-first) row index:
        placed_row = self.ROWS - 1 - r_bot
        self.board[placed_row, c] = self.current_player

        # ---- Terminal check (bitboard fast) ----
        me_bb = self._pos1 if self.current_player == 1 else self._pos2

        if _bb_has_won(me_bb, STRIDE):
            self.done = True
            self.winner = int(self.current_player)
        elif self._mask == FULL_MASK:
            self.done = True
            self.winner = 0  # draw
        else:
            self.done = False
            self.winner = None

        # ---- Reward logic (sparse) ----
        if self.done:
            # reward is always from perspective of the mover who just played
            if self.winner == 0:
                reward = float(self.DRAW_REWARD)
            elif self.winner == self.current_player:
                reward = float(self.WIN_REWARD)
            else:
                # In this env design this case should not happen, but keep it:
                reward = float(self.LOSS_REWARD)
            # Do NOT switch current_player on terminal
        else:
            # Non-terminal move: zero reward
            reward = float(self.STEP_REWARD)
            # Switch to the next mover
            self.current_player *= -1
            self.ply += 1

        return self.get_state(perspective=self.current_player), reward, self.done

    # Kept for compatibility, but uses bitboards
    def check_game_over(self):
        """
        Utility for external callers: check game-over status and winner.

        Returns:
            done (bool),
            winner (1, -1, 0, or None)
        """
        if _bb_has_won(self._pos1, STRIDE):
            return True, 1
        if _bb_has_won(self._pos2, STRIDE):
            return True, -1
        if self._mask == FULL_MASK:
            return True, 0
        return False, None
