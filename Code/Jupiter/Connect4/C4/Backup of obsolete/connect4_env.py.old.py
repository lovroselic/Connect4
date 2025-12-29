# connect4_env.py  
# bitboard + numba accelerated
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
CENTER_COL = 3

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
        m = bb & (bb >> uint64(1))                     # vertical
        if (m & (m >> uint64(2))) != uint64(0): return True
        m = bb & (bb >> uint64(stride_i))              # horizontal
        if (m & (m >> uint64(2 * stride_i))) != uint64(0): return True
        m = bb & (bb >> uint64(stride_i + 1))          # diag up-right
        if (m & (m >> uint64(2 * (stride_i + 1)))) != uint64(0): return True
        m = bb & (bb >> uint64(stride_i - 1))          # diag up-left
        if (m & (m >> uint64(2 * (stride_i - 1)))) != uint64(0): return True
        return False
else:
    def _bb_has_won(bb: np.uint64, stride_i: int) -> bool:
        m = bb & (bb >> 1)
        if (m & (m >> 2)) != 0: return True
        m = bb & (bb >> stride_i)
        if (m & (m >> (2 * stride_i))) != 0: return True
        m = bb & (bb >> (stride_i + 1))
        if (m & (m >> (2 * (stride_i + 1)))) != 0: return True
        m = bb & (bb >> (stride_i - 1))
        if (m & (m >> (2 * (stride_i - 1)))) != 0: return True
        return False


class Connect4Env:
    # ENV is mover centric
    # reward is for the mover, from mower's perspective
    ROWS = ROWS
    COLS = COLS

    # reward constants (unchanged)
    THREAT2_VALUE = 10
    THREAT3_VALUE = 20
    BLOCK2_VALUE = 12
    BLOCK3_VALUE = 25
    MAX_REWARD = 80
    WIN_REWARD = 100
    DRAW_REWARD = 15
    LOSS_PENALTY = -100
    CENTER_REWARD = 1.0
    CENTER_REWARD_BOTTOM = 50
    FORK_BONUS = 50
    BLOCK_FORK_BONUS = 55
    OPP_IMMEDIATE_PENALTY = 70
    STEP_PENALTY = 1
    CENTER_WEIGHTS = [0.25, 0, 0.5, 1.0, 0.5, 0, 0.25]
    #CENTER_WEIGHTS = [0.85, 0.75, 0.90, 1.0, 0.90, 0.75, 0.85]
    OPENING_DECAY_STEPS = 5 #6

    # Precomputed coordinate planes (float32) reused for every state
    _ROW_PLANE = np.tile(np.linspace(-1, 1, ROWS, dtype=np.float32)[:, None], (1, COLS))
    _COL_PLANE = np.tile(np.linspace(-1, 1, COLS, dtype=np.float32)[None, :], (ROWS, 1))
    _CENTER_WEIGHTS_ARR = np.asarray(CENTER_WEIGHTS, dtype=np.float32)

    def __init__(self):
        self.lookahead = Connect4Lookahead()  # your (now fast) bitboard lookahead
        self.reset()

    # ---------------------- Core API ----------------------

    def reset(self):
        # 2D board (top-first) kept for compatibility with agent/state builder
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.ply = 0

        # bitboards + per-column heights for speed
        self._pos1 = UINT(0)
        self._pos2 = UINT(0)
        self._mask = UINT(0)
        self._heights = np.zeros(self.COLS, dtype=np.int8)  # number of stones in each column [0..6]

        return self.get_state(perspective=self.current_player)

    def get_state(self, perspective=None) -> np.ndarray:
        """Return (4,6,7) mover-centric planes. Uses precomputed row/col planes."""
        if perspective is None: perspective = self.current_player
        b = self.board
        p = perspective

        agent_plane = (b == p).astype(np.float32, copy=False)
        opp_plane   = (b == -p).astype(np.float32, copy=False)
        # row/col planes are constants
        return np.stack([agent_plane, opp_plane, self._ROW_PLANE, self._COL_PLANE], axis=0)

    def available_actions(self):
        # Using heights avoids reading top row
        # return columns where height < ROWS
        return [c for c in range(self.COLS) if self._heights[c] < self.ROWS]

    def step(self, action):
        if self.done: return self.get_state(perspective=self.current_player), 0.0, True

        c = int(action)
        if c < 0 or c >= self.COLS or self._heights[c] >= self.ROWS: raise ValueError("ILLEGAL MOVE DETECTED!")

        # Snapshot before move (for block-delta computation)
        board_before = self.board.copy()
        opponent_before = -self.current_player

        # ---- Apply move: update bitboards, heights, and 2D board ----
        r_bot = int(self._heights[c])                         # bottom-based row [0..5]
        bit   = UINT(1) << UINT(c * STRIDE + r_bot)
        self._mask |= bit
        if self.current_player == 1: self._pos1 |= bit
        else: self._pos2 |= bit
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
            self.winner = 0
        else:
            self.done = False
            self.winner = None

        # ---- Rewards (unchanged logic) ----
        if not self.done:
            # Fast counts via lookahead helpers (your class; now fast)
            threat2_count = self.lookahead.count_pure(self.board, self.current_player, 2)
            threat3_count = self.lookahead.count_pure(self.board, self.current_player, 3)
            block2_count  = self.lookahead.count_pure_block_delta(board_before, self.board, opponent_before, 2)
            block3_count  = self.lookahead.count_pure_block_delta(board_before, self.board, opponent_before, 3)

            # Center reward (+ opening special for bottom center at ply 0)
            center_reward = float(self.CENTER_REWARD) * float(self._CENTER_WEIGHTS_ARR[c])

            if c == CENTER_COL and r_bot == 0:  # exactly bottom-center
                opening_decay = float(np.exp(-self.ply / self.OPENING_DECAY_STEPS))
                bonus = self.CENTER_REWARD_BOTTOM * (2.0 if self.ply == 0 else opening_decay)
                center_reward += bonus

            # Fork / anti-fork shaping
            signals = self.lookahead.compute_fork_signals(board_before, self.board, self.current_player)
            fork_bonus = self.FORK_BONUS if signals["my_after"] >= 2 else 0
            blocked_fork = (signals["opp_before"] >= 2) and (signals["opp_after"] < signals["opp_before"])
            block_fork_bonus = self.BLOCK_FORK_BONUS if blocked_fork else 0

            opp_immediate_after = signals["opp_after"]
            immediate_loss_penalty = self.OPP_IMMEDIATE_PENALTY * opp_immediate_after

            threat_reward = (self.THREAT2_VALUE * threat2_count) + (self.THREAT3_VALUE * threat3_count)
            block_reward  = (self.BLOCK2_VALUE * block2_count) + (self.BLOCK3_VALUE * block3_count)

            reward = (threat_reward + block_reward + fork_bonus + block_fork_bonus + center_reward - immediate_loss_penalty)

            reward = float(np.clip(reward, -self.MAX_REWARD, self.MAX_REWARD))
            reward -= self.STEP_PENALTY

            # Switch turns (keep as in your original)
            self.current_player *= -1
            self.ply += 1

        else:
            # Terminal rewards (same semantics as before)
            if self.winner == self.current_player: reward = float(self.WIN_REWARD)
            elif self.winner == 0: reward = float(self.DRAW_REWARD)
            else: reward = float(self.LOSS_PENALTY)

        return self.get_state(perspective=self.current_player), reward, self.done

    # Kept for compatibility, but now uses bitboards
    def check_game_over(self):
        if _bb_has_won(self._pos1, STRIDE): return True, 1
        if _bb_has_won(self._pos2, STRIDE): return True, -1
        if self._mask == FULL_MASK:         return True, 0
        return False, None
