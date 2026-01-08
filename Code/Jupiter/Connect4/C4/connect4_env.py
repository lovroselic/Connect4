# C4/connect4_env.py
# Bitboard + (optional) numba accelerated Connect4 environment.
#
# ENV is mover-centric:
# - state returned is from the *current_player* POV
# - reward is attributed to the mover (the one who just played), from mover POV
#
# This version mirrors (where applicable) the "Allis-ish" heuristic concepts added to
# C4/fast_connect4_lookahead.py:
# - Vertical threats weighted more (gravity makes them more forcing)
# - Parity shaping: prefer agent parity, avoid unlocking opponent parity
# - Tempo / zugzwang proxy: reduce opponent safe moves
# - Threat-space: reward moves that create future immediate-win options
#

import numpy as np
from C4.fast_connect4_lookahead import Connect4Lookahead

from numba import njit, uint64, int32
_HAVE_NUMBA = True

ROWS, COLS = 6, 7
STRIDE = ROWS + 1
CENTER_COL = 3

UINT = np.uint64

COL_MASK = np.zeros(COLS, dtype=UINT)
TOP_MASK = np.zeros(COLS, dtype=UINT)
BOTTOM_MASK = np.zeros(COLS, dtype=UINT)
FULL_MASK = UINT(0)

for c in range(COLS):
    col_bits = UINT(0)
    for r in range(ROWS):
        col_bits |= UINT(1) << UINT(c * STRIDE + r)
    COL_MASK[c] = col_bits
    BOTTOM_MASK[c] = UINT(1) << UINT(c * STRIDE + 0)
    TOP_MASK[c] = UINT(1) << UINT(c * STRIDE + (ROWS - 1))
    FULL_MASK |= col_bits

# Precompute vertical 4-in-a-row window masks (21 windows total)
_VERT_WIN_MASKS = []
for c in range(COLS):
    for r0 in range(ROWS - 4 + 1):
        m = 0
        for k in range(4):
            m |= 1 << (c * STRIDE + (r0 + k))
        _VERT_WIN_MASKS.append(m)

if _HAVE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _bb_has_won(bb: uint64, stride_i: int32) -> bool:
        m = bb & (bb >> uint64(1))
        if (m & (m >> uint64(2))) != uint64(0):
            return True
        m = bb & (bb >> uint64(stride_i))
        if (m & (m >> uint64(2 * stride_i))) != uint64(0):
            return True
        m = bb & (bb >> uint64(stride_i + 1))
        if (m & (m >> uint64(2 * (stride_i + 1)))) != uint64(0):
            return True
        m = bb & (bb >> uint64(stride_i - 1))
        if (m & (m >> uint64(2 * (stride_i - 1)))) != uint64(0):
            return True
        return False
else:
    def _bb_has_won(bb: np.uint64, stride_i: int) -> bool:
        m = bb & (bb >> 1)
        if (m & (m >> 2)) != 0:
            return True
        m = bb & (bb >> stride_i)
        if (m & (m >> (2 * stride_i))) != 0:
            return True
        m = bb & (bb >> (stride_i + 1))
        if (m & (m >> (2 * (stride_i + 1)))) != 0:
            return True
        m = bb & (bb >> (stride_i - 1))
        if (m & (m >> (2 * (stride_i - 1)))) != 0:
            return True
        return False


class Connect4Env:
    ROWS = ROWS
    COLS = COLS

    THREAT2_VALUE = 5.0
    THREAT3_VALUE = 10.0
    BLOCK2_VALUE = 5.0
    BLOCK3_VALUE = 20.0

    VERT_THREAT_MUL_2 = 1.1
    VERT_THREAT_MUL_3 = 1.5
    VERT_BLOCK_MUL_2 = 1.5
    VERT_BLOCK_MUL_3 = 1.25

    WIN_REWARD = 10000.0
    MAX_REWARD = WIN_REWARD * 0.35
   
    DRAW_REWARD = 100.0
    LOSS_PENALTY = -WIN_REWARD

    CENTER_REWARD = 0.1
    CENTER_REWARD_BOTTOM = 1000
    CENTER_WEIGHTS = [1.0] * 7
    OPENING_DECAY_STEPS = 7

    FORK_BONUS = 25.0
    BLOCK_FORK_BONUS = 30.0
    OPP_IMMEDIATE_PENALTY = 1000.0
    STEP_PENALTY = 1.0

    PARITY_MOVE_BONUS = 0.20            #0.25
    PARITY_UNLOCK_PENALTY = 0.10        #0.20
    TEMPO_SQUEEZE_W = 0.15
    THREATSPACE_W = 0.5   #1.25

    _CENTER_WEIGHTS_ARR = np.asarray(CENTER_WEIGHTS, dtype=np.float32)

    def __init__(self):
        self.lookahead = Connect4Lookahead()
        self._N = self.lookahead._N
        self.reset()

    def reset(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.ply = 0

        self._pos1 = UINT(0)
        self._pos2 = UINT(0)
        self._mask = UINT(0)
        self._heights = np.zeros(self.COLS, dtype=np.int8)

        return self.get_state(perspective=self.current_player)

    def get_state(self, perspective=None) -> np.ndarray:
        if perspective is None:
            perspective = self.current_player
        return (self.board.astype(np.float32) * float(perspective))[None, :, :]

    def available_actions(self):
        return [c for c in range(self.COLS) if self._heights[c] < self.ROWS]

    @staticmethod
    def _bit_at(c: int, r_bot: int) -> UINT:
        return UINT(1) << UINT(c * STRIDE + r_bot)

    @staticmethod
    def _role_first_from_center(mask: UINT, pos1: UINT, pos2: UINT) -> tuple[int, bool]:
        b_d1 = UINT(1) << UINT(CENTER_COL * STRIDE + 0)
        if (mask & b_d1) == UINT(0):
            return 1, False
        if (pos1 & b_d1) != UINT(0):
            return 1, True
        return -1, True

    def _count_pure_all(self, me_bb: int, opp_bb: int, n: int) -> int:
        WIN_MASKS = self._N["WIN_MASKS"]
        cnt = 0
        me = int(me_bb)
        opp = int(opp_bb)
        for i in range(WIN_MASKS.shape[0]):
            w = int(WIN_MASKS[i])
            mp = w & me
            mo = w & opp
            if mo == 0 and (mp.bit_count() == n):
                cnt += 1
        return cnt

    def _count_pure_vertical(self, me_bb: int, opp_bb: int, n: int) -> int:
        cnt = 0
        me = int(me_bb)
        opp = int(opp_bb)
        for w in _VERT_WIN_MASKS:
            mp = w & me
            mo = w & opp
            if mo == 0 and (mp.bit_count() == n):
                cnt += 1
        return cnt

    def _count_immediate_wins_bits(self, pos_bb: int, mask_bb: int) -> int:
        return int(
            self._N["count_immediate_wins_bits"](
                np.uint64(pos_bb),
                np.uint64(mask_bb),
                self._N["CENTER_ORDER"],
                self._N["TOP_MASK"],
                self._N["BOTTOM_MASK"],
                self._N["COL_MASK"],
                np.int32(STRIDE),
            )
        )

    def step(self, action):
        if self.done:
            return self.get_state(perspective=self.current_player), 0.0, True

        c = int(action)
        if c < 0 or c >= self.COLS or self._heights[c] >= self.ROWS:
            raise ValueError("ILLEGAL MOVE DETECTED!")

        mask_before = int(self._mask)
        pos1_before = int(self._pos1)
        pos2_before = int(self._pos2)
        mover = int(self.current_player)

        mover_bb_before = pos1_before if mover == 1 else pos2_before
        opp_bb_before = pos2_before if mover == 1 else pos1_before

        opp_immediate_before = self._count_immediate_wins_bits(opp_bb_before, mask_before)

        r_bot = int(self._heights[c])
        bit = self._bit_at(c, r_bot)

        self._mask |= bit
        if self.current_player == 1:
            self._pos1 |= bit
        else:
            self._pos2 |= bit
        self._heights[c] = r_bot + 1

        placed_row = self.ROWS - 1 - r_bot
        self.board[placed_row, c] = self.current_player

        me_bb_after_u = self._pos1 if self.current_player == 1 else self._pos2
        if _bb_has_won(me_bb_after_u, STRIDE):
            self.done = True
            self.winner = int(self.current_player)
        elif self._mask == FULL_MASK:
            self.done = True
            self.winner = 0
        else:
            self.done = False
            self.winner = None

        if not self.done:
            mask_after = int(self._mask)
            pos1_after = int(self._pos1)
            pos2_after = int(self._pos2)

            mover_bb_after = pos1_after if mover == 1 else pos2_after
            opp_bb_after = pos2_after if mover == 1 else pos1_after

            threat2_total = self._count_pure_all(mover_bb_after, opp_bb_after, 2)
            threat3_total = self._count_pure_all(mover_bb_after, opp_bb_after, 3)

            threat2_vert = self._count_pure_vertical(mover_bb_after, opp_bb_after, 2)
            threat3_vert = self._count_pure_vertical(mover_bb_after, opp_bb_after, 3)

            threat2_other = max(0, threat2_total - threat2_vert)
            threat3_other = max(0, threat3_total - threat3_vert)

            opp2_before_total = self._count_pure_all(opp_bb_before, mover_bb_before, 2)
            opp3_before_total = self._count_pure_all(opp_bb_before, mover_bb_before, 3)
            opp2_after_total = self._count_pure_all(opp_bb_after, mover_bb_after, 2)
            opp3_after_total = self._count_pure_all(opp_bb_after, mover_bb_after, 3)

            block2_total = max(0, opp2_before_total - opp2_after_total)
            block3_total = max(0, opp3_before_total - opp3_after_total)

            opp2_before_vert = self._count_pure_vertical(opp_bb_before, mover_bb_before, 2)
            opp3_before_vert = self._count_pure_vertical(opp_bb_before, mover_bb_before, 3)
            opp2_after_vert = self._count_pure_vertical(opp_bb_after, mover_bb_after, 2)
            opp3_after_vert = self._count_pure_vertical(opp_bb_after, mover_bb_after, 3)

            block2_vert = max(0, opp2_before_vert - opp2_after_vert)
            block3_vert = max(0, opp3_before_vert - opp3_after_vert)

            block2_other = max(0, block2_total - block2_vert)
            block3_other = max(0, block3_total - block3_vert)

            threat_reward = (
                self.THREAT2_VALUE * float(threat2_other)
                + self.THREAT3_VALUE * float(threat3_other)
                + (self.THREAT2_VALUE * self.VERT_THREAT_MUL_2) * float(threat2_vert)
                + (self.THREAT3_VALUE * self.VERT_THREAT_MUL_3) * float(threat3_vert)
            )

            block_reward = (
                self.BLOCK2_VALUE * float(block2_other)
                + self.BLOCK3_VALUE * float(block3_other)
                + (self.BLOCK2_VALUE * self.VERT_BLOCK_MUL_2) * float(block2_vert)
                + (self.BLOCK3_VALUE * self.VERT_BLOCK_MUL_3) * float(block3_vert)
            )

            center_reward = float(self.CENTER_REWARD) * float(self._CENTER_WEIGHTS_ARR[c])
            if c == CENTER_COL and r_bot == 0:
                opening_decay = float(np.exp(-self.ply / self.OPENING_DECAY_STEPS))
                bonus = self.CENTER_REWARD_BOTTOM * (2.0 if self.ply == 0 else opening_decay)
                center_reward += bonus

            my_immediate_after = self._count_immediate_wins_bits(mover_bb_after, mask_after)
            opp_immediate_after = self._count_immediate_wins_bits(opp_bb_after, mask_after)

            fork_bonus = self.FORK_BONUS if my_immediate_after >= 2 else 0.0
            blocked_fork = (opp_immediate_before >= 2) and (opp_immediate_after < opp_immediate_before)
            block_fork_bonus = self.BLOCK_FORK_BONUS if blocked_fork else 0.0

            immediate_loss_penalty = float(self.OPP_IMMEDIATE_PENALTY) * float(opp_immediate_after)

            parity_reward = 0.0
            role_first_mark, parity_enabled = self._role_first_from_center(self._mask, self._pos1, self._pos2)
            if parity_enabled:
                mover_is_first_role = (mover == int(role_first_mark))
                prefer_par = 0 if mover_is_first_role else 1
                opp_prefer_par = 1 if mover_is_first_role else 0

                if (r_bot & 1) == prefer_par:
                    parity_reward += float(self.PARITY_MOVE_BONUS)
                else:
                    parity_reward -= float(self.PARITY_MOVE_BONUS)

                if r_bot + 1 < self.ROWS:
                    if ((r_bot + 1) & 1) == opp_prefer_par:
                        parity_reward -= float(self.PARITY_UNLOCK_PENALTY)

            tempo_reward = 0.0
            if self.TEMPO_SQUEEZE_W != 0.0:
                opp_safe = int(
                    self._N["count_safe_moves"](
                        np.uint64(opp_bb_after),
                        np.uint64(mask_after),
                        self._N["CENTER_ORDER"],
                        self._N["TOP_MASK"],
                        self._N["BOTTOM_MASK"],
                        self._N["COL_MASK"],
                        np.int32(STRIDE),
                    )
                )
                tempo_reward += float(self.TEMPO_SQUEEZE_W) * float(max(0, 7 - opp_safe))

            threatspace_reward = float(self.THREATSPACE_W) * float(my_immediate_after) if self.THREATSPACE_W != 0.0 else 0.0

            reward = (
                threat_reward
                + block_reward
                + float(fork_bonus)
                + float(block_fork_bonus)
                + float(center_reward)
                + float(parity_reward)
                + float(tempo_reward)
                + float(threatspace_reward)
                - float(immediate_loss_penalty)
            )

            reward = float(np.clip(reward, -self.MAX_REWARD, self.MAX_REWARD))
            reward -= float(self.STEP_PENALTY)

            self.current_player *= -1
            self.ply += 1

        else:
            if self.winner == self.current_player:
                reward = float(self.WIN_REWARD)
            elif self.winner == 0:
                reward = float(self.DRAW_REWARD)
            else:
                reward = float(self.LOSS_PENALTY)

        return self.get_state(perspective=self.current_player), float(reward), bool(self.done)

    def check_game_over(self):
        if _bb_has_won(self._pos1, STRIDE):
            return True, 1
        if _bb_has_won(self._pos2, STRIDE):
            return True, -1
        if self._mask == FULL_MASK:
            return True, 0
        return False, None