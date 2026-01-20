# C4/connect4_env.py
# Bitboard + Numba-accelerated Connect4 environment.
#
# ENV is mover-centric:
# - state returned is from the *current_player* POV (current_player in {+1, -1})
# - reward is attributed to the mover (the one who just played), from mover POV
#
# Mirrors (where applicable) the "Allis-ish" heuristic concepts used in your lookahead:
# - Floating-aware shaping: discount "pure threats" that need support drops (gravity reality check)
#   IMPORTANT FIX: floating distance uses max(dh) across empty cells in a window (NOT sum(dh)).
# - Parity shaping: prefer mover parity, avoid unlocking opponent parity (enabled only if D1 is occupied)
# - Tempo squeeze: reduce opponent safe moves
# - Threat-space: reward moves that increase immediate-win count (fork-ish pressure)
#

import numpy as np
from numba import njit, uint64, int32, int16

from C4.fast_connect4_lookahead import Connect4Lookahead

ROWS, COLS = 6, 7
STRIDE = ROWS + 1
CENTER_COL = 3
K = 4

UINT = np.uint64

# ------------------------------ bitboard masks -----------------------------
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


@njit(cache=True, fastmath=True)
def _popcount64(x: uint64) -> int32:
    x = x - ((x >> uint64(1)) & uint64(0x5555555555555555))
    x = (x & uint64(0x3333333333333333)) + ((x >> uint64(2)) & uint64(0x3333333333333333))
    x = (x + (x >> uint64(4))) & uint64(0x0F0F0F0F0F0F0F0F)
    return int32((x * uint64(0x0101010101010101)) >> uint64(56))


def _bit_at_py(c: int, r_bot: int) -> int:
    return 1 << (c * STRIDE + r_bot)


# -------------------------- precompute ALL windows --------------------------
# kind: 0=horiz, 1=vert, 2=diag up-right, 3=diag up-left
_WIN_MASKS = []
_WIN_B = []
_WIN_C = []
_WIN_R = []
_WIN_KIND = []

def _add_window(cells, kind: int):
    m = 0
    bs = [0, 0, 0, 0]
    cs = [0, 0, 0, 0]
    rs = [0, 0, 0, 0]
    for i, (c, r) in enumerate(cells):
        b = _bit_at_py(c, r)
        m |= b
        bs[i] = b
        cs[i] = c
        rs[i] = r
    _WIN_MASKS.append(m)
    _WIN_B.append(bs)
    _WIN_C.append(cs)
    _WIN_R.append(rs)
    _WIN_KIND.append(kind)

# horiz
for r in range(ROWS):
    for c in range(COLS - K + 1):
        _add_window([(c + i, r) for i in range(K)], kind=0)
# vert
for c in range(COLS):
    for r in range(ROWS - K + 1):
        _add_window([(c, r + i) for i in range(K)], kind=1)
# diag up-right
for r in range(ROWS - K + 1):
    for c in range(COLS - K + 1):
        _add_window([(c + i, r + i) for i in range(K)], kind=2)
# diag up-left
for r in range(ROWS - K + 1):
    for c in range(K - 1, COLS):
        _add_window([(c - i, r + i) for i in range(K)], kind=3)

WIN_MASKS = np.asarray(_WIN_MASKS, dtype=np.uint64)
WIN_B = np.asarray(_WIN_B, dtype=np.uint64)      # (W,4)
WIN_C = np.asarray(_WIN_C, dtype=np.int16)       # (W,4)
WIN_R = np.asarray(_WIN_R, dtype=np.int16)       # (W,4)
WIN_KIND = np.asarray(_WIN_KIND, dtype=np.int16) # (W,)


@njit(cache=True, fastmath=True)
def _sum_pure_weighted(
    me: uint64,
    opp: uint64,
    mask: uint64,
    heights: np.ndarray,      # int8[7], bottom-based heights
    n: int32,                 # 2 or 3
    WIN_MASKS_: np.ndarray,   # uint64[W]
    WIN_B_: np.ndarray,       # uint64[W,4]
    WIN_C_: np.ndarray,       # int16[W,4]
    WIN_R_: np.ndarray,       # int16[W,4]
    WIN_KIND_: np.ndarray,    # int16[W]
    FLOATING_NEAR: float,
    FLOATING_FAR: float,
    VERT_MUL: float,
) -> float:
    """
    Floating-aware pure-window "count".

    Floating FIX (the one you wanted):
    - Compute support distance per empty cell: dh = rr - heights[col]
    - Use need_max = max(dh) across empty cells in the window
      (NOT sum(dh), which over-penalizes diagonals with multiple independently-supported empties)

    Contribution per qualifying window:
      mul = 1.0 if need_max==0
          = FLOATING_NEAR if need_max==1
          = FLOATING_FAR if need_max>=2
      if vertical: mul *= VERT_MUL
    """
    total = 0.0
    W = WIN_MASKS_.shape[0]
    for i in range(W):
        w = WIN_MASKS_[i]

        # pure for me: opponent has none
        if (w & opp) != uint64(0):
            continue

        mp = w & me
        if _popcount64(mp) != n:
            continue

        need_max = int16(0)
        for k in range(4):
            b = WIN_B_[i, k]
            if (mask & b) == uint64(0):
                cc = WIN_C_[i, k]
                rr = WIN_R_[i, k]
                dh = rr - int16(heights[int(cc)])
                if dh > need_max:
                    need_max = dh

        mul = 1.0
        if need_max == 1:
            mul = FLOATING_NEAR
        elif need_max >= 2:
            mul = FLOATING_FAR

        if WIN_KIND_[i] == 1:
            mul *= VERT_MUL

        total += mul

    return total


class Connect4Env:
    ROWS = ROWS
    COLS = COLS

    # ---------------- knobs aligned to LA grid ----------------
    FLOATING_NEAR = 0.25
    FLOATING_FAR = 0.125
    VERT_MUL = 0.875

    PARITY_MOVE_BONUS = 0.99       # 0.50~ LA: PARITY_MOVE_W
    PARITY_UNLOCK_PENALTY = 0.49   # 0.25~ LA: PARITY_UNLOCK_W
    TEMPO_SQUEEZE_W = 4          # 75 ~ LA: TEMPO_W (ENV proxy differs, but knob intent matches)
    THREATSPACE_W =  8.75            # ~ LA: THREATSPACE_W

    # ---------------- shaping magnitudes  ----------------
    THREAT2_VALUE = 10.0
    THREAT3_VALUE = 50.0
    BLOCK2_VALUE = 15.0
    BLOCK3_VALUE = 75.0

    WIN_REWARD = 10000.0
    MAX_REWARD = WIN_REWARD * 0.35
    DRAW_REWARD = 100.0
    LOSS_PENALTY = -WIN_REWARD

    CENTER_REWARD = 1 #0.1
    CENTER_REWARD_BOTTOM = 1000     
    CENTER_WEIGHTS = [0, 0, 0, 1.0, 0, 0, 0]
    OPENING_DECAY_STEPS = 7         # KEEP

    _CENTER_WEIGHTS_ARR = np.asarray(CENTER_WEIGHTS, dtype=np.float32)

    FORK_BONUS = 150
    BLOCK_FORK_BONUS = 200
    OPP_IMMEDIATE_PENALTY = 1000.0
    STEP_PENALTY = 0.1

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

        self._pos1 = UINT(0)  # player +1 stones
        self._pos2 = UINT(0)  # player -1 stones
        self._mask = UINT(0)
        self._heights = np.zeros(self.COLS, dtype=np.int8)  # bottom-based heights

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
        """
        Parity shaping is enabled only if D1 (center-bottom) is occupied.
        "First role" = the owner of D1 (player +1 if pos1 owns it, else player -1).
        """
        b_d1 = UINT(1) << UINT(CENTER_COL * STRIDE + 0)
        if (mask & b_d1) == UINT(0):
            return 1, False
        if (pos1 & b_d1) != UINT(0):
            return 1, True
        return -1, True

    def _count_immediate_wins_bits(self, pos_bb: int, mask_bb: int) -> int:
        # Provided by your Connect4Lookahead numba helpers
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

        mover = int(self.current_player)

        # snapshot before move (for block deltas + floating)
        mask_before = np.uint64(self._mask)
        pos1_before = np.uint64(self._pos1)
        pos2_before = np.uint64(self._pos2)
        heights_before = self._heights.copy()

        mover_bb_before = pos1_before if mover == 1 else pos2_before
        opp_bb_before = pos2_before if mover == 1 else pos1_before

        opp_immediate_before = self._count_immediate_wins_bits(int(opp_bb_before), int(mask_before))

        # apply move
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

        # terminal?
        me_bb_after_u = self._pos1 if self.current_player == 1 else self._pos2
        if _bb_has_won(np.uint64(me_bb_after_u), np.int32(STRIDE)):
            self.done = True
            self.winner = int(self.current_player)
        elif self._mask == FULL_MASK:
            self.done = True
            self.winner = 0
        else:
            self.done = False
            self.winner = None

        if not self.done:
            mask_after = np.uint64(self._mask)
            pos1_after = np.uint64(self._pos1)
            pos2_after = np.uint64(self._pos2)
            heights_after = self._heights

            mover_bb_after = pos1_after if mover == 1 else pos2_after
            opp_bb_after = pos2_after if mover == 1 else pos1_after

            # ---------- Floating-aware threat shaping ----------
            threat2 = _sum_pure_weighted(
                np.uint64(mover_bb_after),
                np.uint64(opp_bb_after),
                np.uint64(mask_after),
                heights_after,
                np.int32(2),
                WIN_MASKS, WIN_B, WIN_C, WIN_R, WIN_KIND,
                float(self.FLOATING_NEAR),
                float(self.FLOATING_FAR),
                float(self.VERT_MUL),
            )
            threat3 = _sum_pure_weighted(
                np.uint64(mover_bb_after),
                np.uint64(opp_bb_after),
                np.uint64(mask_after),
                heights_after,
                np.int32(3),
                WIN_MASKS, WIN_B, WIN_C, WIN_R, WIN_KIND,
                float(self.FLOATING_NEAR),
                float(self.FLOATING_FAR),
                float(self.VERT_MUL),
            )

            # Opponent threats (before/after) -> "block" deltas (floating-aware)
            opp2_before = _sum_pure_weighted(
                np.uint64(opp_bb_before),
                np.uint64(mover_bb_before),
                np.uint64(mask_before),
                heights_before,
                np.int32(2),
                WIN_MASKS, WIN_B, WIN_C, WIN_R, WIN_KIND,
                float(self.FLOATING_NEAR),
                float(self.FLOATING_FAR),
                float(self.VERT_MUL),
            )
            opp3_before = _sum_pure_weighted(
                np.uint64(opp_bb_before),
                np.uint64(mover_bb_before),
                np.uint64(mask_before),
                heights_before,
                np.int32(3),
                WIN_MASKS, WIN_B, WIN_C, WIN_R, WIN_KIND,
                float(self.FLOATING_NEAR),
                float(self.FLOATING_FAR),
                float(self.VERT_MUL),
            )
            opp2_after = _sum_pure_weighted(
                np.uint64(opp_bb_after),
                np.uint64(mover_bb_after),
                np.uint64(mask_after),
                heights_after,
                np.int32(2),
                WIN_MASKS, WIN_B, WIN_C, WIN_R, WIN_KIND,
                float(self.FLOATING_NEAR),
                float(self.FLOATING_FAR),
                float(self.VERT_MUL),
            )
            opp3_after = _sum_pure_weighted(
                np.uint64(opp_bb_after),
                np.uint64(mover_bb_after),
                np.uint64(mask_after),
                heights_after,
                np.int32(3),
                WIN_MASKS, WIN_B, WIN_C, WIN_R, WIN_KIND,
                float(self.FLOATING_NEAR),
                float(self.FLOATING_FAR),
                float(self.VERT_MUL),
            )

            block2 = float(max(0.0, float(opp2_before - opp2_after)))
            block3 = float(max(0.0, float(opp3_before - opp3_after)))

            threat_reward = (self.THREAT2_VALUE * float(threat2)) + (self.THREAT3_VALUE * float(threat3))
            block_reward = (self.BLOCK2_VALUE * float(block2)) + (self.BLOCK3_VALUE * float(block3))

            # ---------- Center shaping (KEEP: center-bottom anchor + decay) ----------
            center_reward = float(self.CENTER_REWARD) * float(self._CENTER_WEIGHTS_ARR[c])
            
            if c == CENTER_COL and r_bot == 0:
                if self.ply == 0:
                    center_reward += self.CENTER_REWARD_BOTTOM * 2.0
                elif self.ply <= 2:
                    opening_decay = float(np.exp(-self.ply / self.OPENING_DECAY_STEPS))
                    center_reward += self.CENTER_REWARD_BOTTOM * opening_decay
                # else: no special CENTER_REWARD_BOTTOM bonus


            # ---------- Immediate wins / fork-ish ----------
            my_immediate_after = self._count_immediate_wins_bits(int(mover_bb_after), int(mask_after))
            opp_immediate_after = self._count_immediate_wins_bits(int(opp_bb_after), int(mask_after))

            fork_bonus = self.FORK_BONUS if my_immediate_after >= 2 else 0.0
            blocked_fork = (opp_immediate_before >= 2) and (opp_immediate_after < opp_immediate_before)
            block_fork_bonus = self.BLOCK_FORK_BONUS if blocked_fork else 0.0

            # Big negative if you allow opponent wins-in-1
            new_opp_immediates = max(0, opp_immediate_after - opp_immediate_before)
            immediate_loss_penalty = self.OPP_IMMEDIATE_PENALTY * new_opp_immediates

            # ---------- Parity shaping (only if D1 is occupied) ----------
            parity_reward = 0.0
            role_first_mark, parity_enabled = self._role_first_from_center(self._mask, self._pos1, self._pos2)
            if parity_enabled:
                mover_is_first_role = (mover == int(role_first_mark))
                prefer_par     = 0 if mover_is_first_role else 1
                opp_prefer_par = 1 if mover_is_first_role else 0


                if (r_bot & 1) == prefer_par:
                    parity_reward += float(self.PARITY_MOVE_BONUS)
                else:
                    parity_reward -= float(self.PARITY_MOVE_BONUS)

                # Avoid unlocking opponent preferred parity at r+1
                if r_bot + 1 < self.ROWS:
                    if ((r_bot + 1) & 1) == opp_prefer_par:
                        parity_reward -= float(self.PARITY_UNLOCK_PENALTY)

            # ---------- Tempo squeeze (proxy: reduce opponent safe moves) ----------
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

            # ---------- Threat-space ----------
            threatspace_reward = (
                float(self.THREATSPACE_W) * float(my_immediate_after)
                if self.THREATSPACE_W != 0.0
                else 0.0
            )

            reward = (
                float(threat_reward)
                + float(block_reward)
                + float(fork_bonus)
                + float(block_fork_bonus)
                + float(center_reward)
                + float(parity_reward)
                + float(tempo_reward)
                + float(threatspace_reward)
                - float(immediate_loss_penalty)
            )

            
            reward -= float(self.STEP_PENALTY)
            reward = float(np.clip(reward, -self.MAX_REWARD, self.MAX_REWARD))

            # switch player
            self.current_player *= -1
            self.ply += 1

        else:
            # terminal: reward from mover POV (mover == current_player at this moment)
            if self.winner == self.current_player:
                reward = float(self.WIN_REWARD)
            elif self.winner == 0:
                reward = float(self.DRAW_REWARD)
            else:
                reward = float(self.LOSS_PENALTY)

        return self.get_state(perspective=self.current_player), float(reward), bool(self.done)

    def check_game_over(self):
        if _bb_has_won(np.uint64(self._pos1), np.int32(STRIDE)):
            return True, 1
        if _bb_has_won(np.uint64(self._pos2), np.int32(STRIDE)):
            return True, -1
        if self._mask == FULL_MASK:
            return True, 0
        return False, None
