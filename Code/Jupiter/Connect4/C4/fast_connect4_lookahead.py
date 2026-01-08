# C4/fast_connect4_lookahead.py
# Drop-in Numba-accelerated version of your bitboard lookahead class.
# Public API:
#   - get_heuristic, is_terminal, minimax, n_step_lookahead
#   - has_four / check_win
#   - count_immediate_wins, compute_fork_signals
#   - count_pure, count_pure_block_delta
#
# Internals:
#   - Pascal Pons bitboard layout: bit = c*(ROWS+1) + r (rows are bottom-based)
#   - Numba-compiled evaluation + search (negamax + alpha-beta + tiny TT)
#   - Center-first ordering + killer/history ordering
#   - Gravity-aware heuristic (FLOATING_NEAR/FAR) + DEFENSIVE factor
#
# Improvements:
#   Opening book (depth-independent, now deterministic by default for optimization stability):
#     a) P1: play center bottom (D1)
#     b) P2: if P1 played D1, play adjacent C1 (col 2) if available else E1 (col 4); else play D1 if available
#     c) If P1 then plays D2, P2 responds with D3 (if available)
#
#   Parity preference (odd/even rows):
#     - First player (the one owning center-bottom) prefers ODD rows (1,3,5 from bottom => r%2==0)
#     - Second player prefers EVEN rows (2,4,6 from bottom => r%2==1)
#     - If center-bottom is NOT occupied, parity bonus is disabled.
#
# Added "core Allis-ish" concepts (cheap, search-friendly):
#   - Vertical threat bias: vertical windows are worth more (gravity makes them nastier).
#   - Tempo / zugzwang proxy: "safe-move" mobility term squeezes opponent options.
#   - Root-only parity move/unlock: don't just "like my parity", actively avoid gifting theirs.
#   - Root-only threat-space: prefer moves that create more immediate wins (next-turn threats).
#   - Root-only win-in-2 forcing: detect true 2-ply forced wins before deeper search.
#
# Patch notes (this version):
#   1) Corrected FLOATING model in evaluate(): per-empty multiplicative penalty
#      using FN for dh==1 and FF for dh>=2
#   2) Added win-in-2 forcing pre-pass in root_select_fixed().

from typing import List, Tuple, Dict, Optional
import numpy as np
from numba import njit


# ============================ Heuristic Defaults ============================

C4_WIN = 100000.0
C4_IMMEDIATE_W = C4_WIN
C4_FORK_W = C4_WIN

# DEFENSIVE > 1 means opponent threats hurt more than your threats help.
# Too high and the agent becomes a professional coward (blocks ghosts, loses initiative).
C4_DEFENSIVE = 1.4  # 1.4

# Floating penalties: discourage patterns that require "miracle levitation".
# Too high and it refuses to build diagonals like diagonals are a moral failing.
C4_FLOATING_NEAR = 0.225  # 0.225
C4_FLOATING_FAR = 0.09    # 0.09

# Center bonus: steady positional bias.
# Too high and it plays center even when it's on fire.
C4_CENTER_BONUS = 7.0  # 10

# Board-global parity preference (only when center-bottom is occupied).
# Too high and it starts playing "parity cosplay" instead of winning.
C4_PARITY_BONUS = 1.0

# Vertical windows are inherently more forcing (gravity + zugzwang structure).
# Too high and it becomes a "stacking simulator" and forgets diagonals exist.
C4_VERT_MUL = 1.1

# Extra bump for a vertical-3 whose winning square is playable now.
# Too high and it overchases vertical 3s even when tactically unsound.
C4_VERT_3_READY_BONUS = 0  # 2500.0

# Tempo / zugzwang proxy: reward having more "safe" moves and/or reducing opponent safe moves.
C4_TEMPO_W = 0.25  # 2.0

# Root-only parity move preference (what row you land on).
C4_PARITY_MOVE_W = 0.2

# Root-only parity unlock penalty (what row you unlock above the placed disc).
C4_PARITY_UNLOCK_W = 0.1

# Root-only threat-space: prefer moves that increase your immediate-win count (if you moved again).
C4_THREATSPACE_W = 0.5  # 0.6

# Default eval weights (immutable); copied per instance into a dict.
C4_DEFAULT_WEIGHTS_ITEMS = (
    (2, 1.0),
    (3, 1000.0),
    (4, C4_WIN),
)

# Mate scale multipliers (single source)
C4_SOFT_MATE_MULT = 100.0
C4_MATE_SCORE_MULT = 1000.0


# ============================ Numba Core & Cache ============================

_NUMBA_C4_CLASS_CACHE = {}  # built once, reused by all instances


def _ensure_numba_cache():
    """Build & JIT-compile once; stash in _NUMBA_C4_CLASS_CACHE."""
    global _NUMBA_C4_CLASS_CACHE
    if _NUMBA_C4_CLASS_CACHE:
        return _NUMBA_C4_CLASS_CACHE

    # ---------- constants ----------
    ROWS, COLS, K = 6, 7, 4
    STRIDE = ROWS + 1
    UINT = np.uint64

    CENTER_COL = 3
    CENTER_ORDER = np.array([3, 4, 2, 5, 1, 6, 0], dtype=np.int8)

    # NOTE:
    # We keep defaults here only for caching/reporting; all tunable knobs are passed
    # as runtime parameters into Numba functions.
    CENTER_BONUS = float(C4_CENTER_BONUS)
    DEFENSIVE = float(C4_DEFENSIVE)
    FLOATING_NEAR = float(C4_FLOATING_NEAR)
    FLOATING_FAR = float(C4_FLOATING_FAR)
    VERT_MUL = float(C4_VERT_MUL)
    VERT_3_READY_BONUS = float(C4_VERT_3_READY_BONUS)
    TEMPO_W = float(C4_TEMPO_W)

    # Bitboard columns & masks
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
    CENTER_MASK = COL_MASK[CENTER_COL]

    # Row parity masks (bottom-based rows):
    # bottom row r=0 is "row 1" (odd), so odd rows are r%2==0.
    ODD_MASK = UINT(0)
    EVEN_MASK = UINT(0)
    for c in range(COLS):
        for r in range(ROWS):
            b = UINT(1) << UINT(c * STRIDE + r)
            if (r & 1) == 0:
                ODD_MASK |= b
            else:
                EVEN_MASK |= b

    def _bit_at(c, r):
        return UINT(1) << UINT(c * STRIDE + r)

    # Windows (69 of length 4)
    # WIN_KIND: 0=horiz, 1=vert, 2=diag up-right, 3=diag up-left
    _win_bits, _win_cols, _win_rows, _WIN_MASKS, _WIN_KIND = [], [], [], [], []

    def _add_window(cells, kind: int):
        mask = UINT(0)
        bs, cs, rs = [], [], []
        for rr, cc in cells:
            b = _bit_at(cc, rr)
            mask |= b
            bs.append(b)
            cs.append(cc)
            rs.append(rr)
        _WIN_MASKS.append(mask)
        _win_bits.append(bs)
        _win_cols.append(cs)
        _win_rows.append(rs)
        _WIN_KIND.append(kind)

    # horiz
    for r in range(ROWS):
        for c in range(COLS - K + 1):
            _add_window([(r, c + i) for i in range(K)], kind=0)
    # vert
    for c in range(COLS):
        for r in range(ROWS - K + 1):
            _add_window([(r + i, c) for i in range(K)], kind=1)
    # diag up-right
    for r in range(ROWS - K + 1):
        for c in range(COLS - K + 1):
            _add_window([(r + i, c + i) for i in range(K)], kind=2)
    # diag up-left
    for r in range(ROWS - K + 1):
        for c in range(K - 1, COLS):
            _add_window([(r + i, c - i) for i in range(K)], kind=3)

    WIN_MASKS = np.array(_WIN_MASKS, dtype=UINT)
    WIN_B = np.array(_win_bits, dtype=UINT)        # (W,4)
    WIN_C = np.array(_win_cols, dtype=np.int8)     # (W,4)
    WIN_R = np.array(_win_rows, dtype=np.int8)     # (W,4)
    WIN_KIND = np.array(_WIN_KIND, dtype=np.int8)  # (W,)

    # ---------- Numba core ----------

    @njit(cache=True, fastmath=True)
    def popcount64(x: UINT) -> np.int32:
        x = x - ((x >> 1) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + ((x >> 2) & np.uint64(0x3333333333333333))
        x = (x + (x >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return np.int32((x * np.uint64(0x0101010101010101)) >> np.uint64(56))

    @njit(cache=True, fastmath=True)
    def has_won(bb: UINT, stride_i: np.int32) -> bool:
        m = bb & (bb >> UINT(1))  # vertical
        if (m & (m >> UINT(2))) != UINT(0):
            return True
        m = bb & (bb >> UINT(stride_i))  # horizontal
        if (m & (m >> UINT(2 * stride_i))) != UINT(0):
            return True
        m = bb & (bb >> UINT(stride_i + 1))  # diag up-right
        if (m & (m >> UINT(2 * (stride_i + 1)))) != UINT(0):
            return True
        m = bb & (bb >> UINT(stride_i - 1))  # diag up-left
        if (m & (m >> UINT(2 * (stride_i - 1)))) != UINT(0):
            return True
        return False

    @njit(cache=True, fastmath=True)
    def can_play(mask_: UINT, c: np.int32, TOP_MASK_: np.ndarray) -> bool:
        return (mask_ & TOP_MASK_[c]) == UINT(0)

    @njit(cache=True, fastmath=True)
    def play_bit(mask_: UINT, c: np.int32, BOTTOM_MASK_: np.ndarray, COL_MASK_: np.ndarray) -> UINT:
        return (mask_ + BOTTOM_MASK_[c]) & COL_MASK_[c]

    @njit(cache=True, fastmath=True)
    def is_winning_move(
        pos_: UINT,
        mask_: UINT,
        c: np.int32,
        BOTTOM_MASK_: np.ndarray,
        COL_MASK_: np.ndarray,
        stride_i: np.int32,
    ) -> bool:
        mv = play_bit(mask_, c, BOTTOM_MASK_, COL_MASK_)
        return has_won(pos_ | mv, stride_i)

    @njit(cache=True, fastmath=True)
    def count_immediate_wins_bits(
        pos_: UINT,
        mask_: UINT,
        CENTER_ORDER_: np.ndarray,
        TOP_MASK_: np.ndarray,
        BOTTOM_MASK_: np.ndarray,
        COL_MASK_: np.ndarray,
        stride_i: np.int32,
    ) -> np.int32:
        cnt = 0
        for i in range(CENTER_ORDER_.shape[0]):
            c = np.int32(CENTER_ORDER_[i])
            if can_play(mask_, c, TOP_MASK_) and is_winning_move(pos_, mask_, c, BOTTOM_MASK_, COL_MASK_, stride_i):
                cnt += 1
        return cnt

    @njit(cache=True, fastmath=True)
    def is_immediate_blunder(
        pos_: UINT,
        mask_: UINT,
        c: np.int32,
        CENTER_ORDER_: np.ndarray,
        TOP_MASK_: np.ndarray,
        BOTTOM_MASK_: np.ndarray,
        COL_MASK_: np.ndarray,
        stride_i: np.int32,
    ) -> bool:
        mv = play_bit(mask_, c, BOTTOM_MASK_, COL_MASK_)
        nm = mask_ | mv
        opp_pos = nm ^ (pos_ | mv)
        for i in range(CENTER_ORDER_.shape[0]):
            cc = np.int32(CENTER_ORDER_[i])
            if can_play(nm, cc, TOP_MASK_) and is_winning_move(opp_pos, nm, cc, BOTTOM_MASK_, COL_MASK_, stride_i):
                return True
        return False

    @njit(cache=True, fastmath=True)
    def count_safe_moves(
        pos_: UINT,
        mask_: UINT,
        CENTER_ORDER_: np.ndarray,
        TOP_MASK_: np.ndarray,
        BOTTOM_MASK_: np.ndarray,
        COL_MASK_: np.ndarray,
        stride_i: np.int32,
    ) -> np.int32:
        # "Safe" = legal and does not hand opponent a win-in-1 immediately.
        s = 0
        for i in range(CENTER_ORDER_.shape[0]):
            c = np.int32(CENTER_ORDER_[i])
            if can_play(mask_, c, TOP_MASK_) and (
                not is_immediate_blunder(pos_, mask_, c, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, stride_i)
            ):
                s += 1
        return s

    @njit(cache=True, fastmath=True)
    def evaluate(
        pos_: UINT,
        mask_: UINT,
        COL_MASK_: np.ndarray,
        WIN_MASKS_: np.ndarray,
        WIN_KIND_: np.ndarray,
        WIN_B_: np.ndarray,
        WIN_C_: np.ndarray,
        WIN_R_: np.ndarray,
        WARR_: np.ndarray,
        DEF_: float,
        FN_: float,
        FF_: float,
        CENTER_MASK_: UINT,
        ODD_MASK_: UINT,
        EVEN_MASK_: UINT,
        parity_bonus_: float,
        parity_enabled_: np.int8,
        root_pos_is_first_: np.int8,
        ply_: np.int16,
        immediate_w_: float,
        fork_w_: float,
        center_bonus_: float,
        vert_mul_: float,
        vert3_ready_bonus_: float,
        tempo_w_: float,
        CENTER_ORDER_: np.ndarray,
        TOP_MASK_: np.ndarray,
        BOTTOM_MASK_: np.ndarray,
        COL_MASK2_: np.ndarray,
        stride_i: np.int32,
        rows_i: np.int32,
        cols_i: np.int32,
    ) -> float:
        opp = mask_ ^ pos_

        # heights (for floating penalties + playability)
        H = np.empty(cols_i, dtype=np.int16)
        for c in range(cols_i):
            H[c] = popcount64(mask_ & COL_MASK_[c])

        score = 0.0
        W = WIN_MASKS_.shape[0]
        for idx in range(W):
            wmask = WIN_MASKS_[idx]
            mo = wmask & opp
            mp = wmask & pos_
            if mo != UINT(0) and mp != UINT(0):
                continue

            p = popcount64(mp)
            o = popcount64(mo)
            if (p + o) < 2:
                continue

            # Corrected floating/gravity realism:
            # - per empty cell: dh==1 -> *FN_, dh>=2 -> *FF_
            # - multiple empties compound multiplicatively
            mul = 1.0
            ready_vertical3 = False

            if p == 0 or o == 0:
                # only compute floating for pure windows (one side only)
                for k2 in range(4):
                    b = WIN_B_[idx, k2]
                    if (mask_ & b) == UINT(0):
                        cc = np.int32(WIN_C_[idx, k2])
                        rr = np.int32(WIN_R_[idx, k2])
                        dh = rr - np.int32(H[cc])
                        if dh == 1:
                            mul *= FN_
                        elif dh >= 2:
                            mul *= FF_
                        else:
                            # dh==0 => playable now
                            if WIN_KIND_[idx] == np.int8(1) and p == 3 and o == 0:
                                ready_vertical3 = True

            # Vertical windows get a multiplier (gravity makes them more forcing).
            if WIN_KIND_[idx] == np.int8(1):
                mul *= vert_mul_

            if o == 0:
                score += mul * (WARR_[p] if p <= 4 else 0.0)
                if ready_vertical3:
                    score += vert3_ready_bonus_
            elif p == 0:
                score -= DEF_ * mul * (WARR_[o] if o <= 4 else 0.0)

        # immediate wins / forks
        my_imm = count_immediate_wins_bits(pos_, mask_, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK2_, stride_i)
        opp_imm = count_immediate_wins_bits(opp, mask_, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK2_, stride_i)
        score += immediate_w_ * (my_imm - DEF_ * opp_imm)

        if my_imm >= 2:
            score += fork_w_ * (my_imm - 1)
        if opp_imm >= 2:
            score -= DEF_ * (fork_w_ * (opp_imm - 1))

        # center occupancy
        score += center_bonus_ * (popcount64(pos_ & CENTER_MASK_) - popcount64(opp & CENTER_MASK_))

        # parity preference (from POV of pos_)
        if parity_enabled_ != np.int8(0):
            # pos_ is always "player to move" stones in this negamax convention.
            is_root_turn = (ply_ & np.int16(1)) == np.int16(0)
            pos_is_first = (root_pos_is_first_ == np.int8(1)) if is_root_turn else (root_pos_is_first_ != np.int8(1))

            if pos_is_first:
                # first-role prefers odd rows; second-role prefers even
                score += parity_bonus_ * (float(popcount64(pos_ & ODD_MASK_)) - DEF_ * float(popcount64(opp & EVEN_MASK_)))
            else:
                # second-role prefers even; first-role prefers odd
                score += parity_bonus_ * (float(popcount64(pos_ & EVEN_MASK_)) - DEF_ * float(popcount64(opp & ODD_MASK_)))

        # tempo / zugzwang proxy: safe-move mobility
        if tempo_w_ != 0.0:
            my_safe = count_safe_moves(pos_, mask_, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK2_, stride_i)
            opp_safe2 = count_safe_moves(opp, mask_, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK2_, stride_i)
            score += tempo_w_ * (float(my_safe) - DEF_ * float(opp_safe2))

        return score

    @njit(cache=True, fastmath=True)
    def order_moves(
        mask_: UINT,
        ply: np.int32,
        killers_: np.ndarray,
        history_: np.ndarray,
        CENTER_ORDER_: np.ndarray,
        TOP_MASK_: np.ndarray,
    ) -> (np.ndarray, np.int32):
        moves = np.empty(7, dtype=np.int8)
        scores = np.empty(7, dtype=np.int32)
        m = 0
        k1, k2 = killers_[ply, 0], killers_[ply, 1]
        for i in range(CENTER_ORDER_.shape[0]):
            c = np.int32(CENTER_ORDER_[i])
            if can_play(mask_, c, TOP_MASK_):
                s = 0
                if c == k1:
                    s += 1_000_000
                elif c == k2:
                    s += 500_000
                s += history_[c]
                moves[m] = np.int8(c)
                scores[m] = s
                m += 1

        # insertion sort (desc)
        for i in range(1, m):
            km, ks = moves[i], scores[i]
            j = i - 1
            while j >= 0 and scores[j] < ks:
                moves[j + 1] = moves[j]
                scores[j + 1] = scores[j]
                j -= 1
            moves[j + 1] = km
            scores[j + 1] = ks
        return moves, m

    @njit(cache=True, fastmath=True)
    def tt_hash_local(pos_: UINT, mask_: UINT, size_mask_i: np.int32) -> np.int32:
        h = np.uint64(pos_) ^ (np.uint64(mask_) * np.uint64(0x9E3779B97F4A7C15))
        h ^= (h >> np.uint64(7))
        return np.int32(h & np.uint64(size_mask_i))

    @njit(cache=True, fastmath=True)
    def tt_lookup(
        pos_: UINT,
        mask_: UINT,
        depth: np.int16,
        alpha: float,
        beta: float,
        TT_pos_: np.ndarray,
        TT_mask_: np.ndarray,
        TT_depth_: np.ndarray,
        TT_flag_: np.ndarray,
        TT_val_: np.ndarray,
        TT_move_: np.ndarray,
    ) -> (bool, float, np.int8, float, float):
        size_mask = np.int32(TT_pos_.shape[0] - 1)
        idx = tt_hash_local(pos_, mask_, size_mask)
        for _ in range(32):
            if TT_depth_[idx] == -1:
                return False, 0.0, np.int8(-1), alpha, beta
            if TT_pos_[idx] == pos_ and TT_mask_[idx] == mask_:
                d = TT_depth_[idx]
                flag = TT_flag_[idx]
                val = TT_val_[idx]
                mv = TT_move_[idx]
                if d >= depth:
                    # EXACT=0, LOWER=1, UPPER=2
                    if flag == 0:
                        return True, val, mv, alpha, beta
                    if flag == 1 and val > alpha:
                        alpha = val
                    elif flag == 2 and val < beta:
                        beta = val
                    if alpha >= beta:
                        return True, val, mv, alpha, beta
                return False, val, mv, alpha, beta
            idx = (idx + 1) & size_mask
        return False, 0.0, np.int8(-1), alpha, beta

    @njit(cache=True, fastmath=True)
    def tt_store(
        pos_: UINT,
        mask_: UINT,
        depth: np.int16,
        val: float,
        alpha0: float,
        beta: float,
        best_mv: np.int8,
        TT_pos_: np.ndarray,
        TT_mask_: np.ndarray,
        TT_depth_: np.ndarray,
        TT_flag_: np.ndarray,
        TT_val_: np.ndarray,
        TT_move_: np.ndarray,
    ) -> None:
        # EXACT=0, LOWER=1, UPPER=2
        flag = 0
        if val <= alpha0:
            flag = 2
        elif val >= beta:
            flag = 1

        size_mask = np.int32(TT_pos_.shape[0] - 1)
        idx = tt_hash_local(pos_, mask_, size_mask)
        victim = -1
        for _ in range(32):
            if TT_depth_[idx] == -1:
                victim = idx
                break
            if TT_pos_[idx] == pos_ and TT_mask_[idx] == mask_:
                victim = idx
                break
            if victim == -1 or TT_depth_[idx] < TT_depth_[victim]:
                victim = idx
            idx = (idx + 1) & size_mask

        TT_pos_[victim] = pos_
        TT_mask_[victim] = mask_
        TT_depth_[victim] = depth
        TT_flag_[victim] = flag
        TT_val_[victim] = val
        TT_move_[victim] = best_mv

    @njit(cache=True)
    def should_stop(node_counter: np.ndarray, max_nodes: np.int64) -> bool:
        node_counter[0] += 1
        return node_counter[0] >= max_nodes

    @njit(cache=True, fastmath=True)
    def negamax(
        pos_: UINT,
        mask_: UINT,
        depth: np.int16,
        alpha: float,
        beta: float,
        ply: np.int16,
        root_pos_is_first_: np.int8,
        parity_enabled_: np.int8,
        node_counter: np.ndarray,
        max_nodes: np.int64,
        MATE_SCORE_i: float,
        COL_MASK_: np.ndarray,
        WIN_MASKS_: np.ndarray,
        WIN_KIND_: np.ndarray,
        WIN_B_: np.ndarray,
        WIN_C_: np.ndarray,
        WIN_R_: np.ndarray,
        WARR_: np.ndarray,
        DEF_: float,
        FN_: float,
        FF_: float,
        CENTER_MASK_: UINT,
        ODD_MASK_: UINT,
        EVEN_MASK_: UINT,
        parity_bonus_: float,
        immediate_w_: float,
        fork_w_: float,
        center_bonus_: float,
        vert_mul_: float,
        vert3_ready_bonus_: float,
        tempo_w_: float,
        CENTER_ORDER_: np.ndarray,
        TOP_MASK_: np.ndarray,
        BOTTOM_MASK_: np.ndarray,
        stride_i: np.int32,
        rows_i: np.int32,
        cols_i: np.int32,
        killers_: np.ndarray,
        history_: np.ndarray,
        FULL_MASK_: UINT,
        TT_pos_: np.ndarray,
        TT_mask_: np.ndarray,
        TT_depth_: np.ndarray,
        TT_flag_: np.ndarray,
        TT_val_: np.ndarray,
        TT_move_: np.ndarray,
    ) -> (float, np.int8):

        if should_stop(node_counter, max_nodes):
            return (
                evaluate(
                    pos_,
                    mask_,
                    COL_MASK_,
                    WIN_MASKS_,
                    WIN_KIND_,
                    WIN_B_,
                    WIN_C_,
                    WIN_R_,
                    WARR_,
                    DEF_,
                    FN_,
                    FF_,
                    CENTER_MASK_,
                    ODD_MASK_,
                    EVEN_MASK_,
                    parity_bonus_,
                    parity_enabled_,
                    root_pos_is_first_,
                    ply,
                    immediate_w_,
                    fork_w_,
                    center_bonus_,
                    vert_mul_,
                    vert3_ready_bonus_,
                    tempo_w_,
                    CENTER_ORDER_,
                    TOP_MASK_,
                    BOTTOM_MASK_,
                    COL_MASK_,
                    stride_i,
                    rows_i,
                    cols_i,
                ),
                np.int8(-1),
            )

        alpha0 = alpha
        hit, val_tt, mv_tt, alpha, beta = tt_lookup(
            pos_, mask_, depth, alpha, beta, TT_pos_, TT_mask_, TT_depth_, TT_flag_, TT_val_, TT_move_
        )
        if hit:
            return val_tt, mv_tt

        # win-in-1
        for i in range(CENTER_ORDER_.shape[0]):
            c = np.int32(CENTER_ORDER_[i])
            if can_play(mask_, c, TOP_MASK_) and is_winning_move(pos_, mask_, c, BOTTOM_MASK_, COL_MASK_, stride_i):
                return MATE_SCORE_i - float(ply), np.int8(c)

        if mask_ == FULL_MASK_:
            return 0.0, np.int8(-1)

        if depth == 0:
            return (
                evaluate(
                    pos_,
                    mask_,
                    COL_MASK_,
                    WIN_MASKS_,
                    WIN_KIND_,
                    WIN_B_,
                    WIN_C_,
                    WIN_R_,
                    WARR_,
                    DEF_,
                    FN_,
                    FF_,
                    CENTER_MASK_,
                    ODD_MASK_,
                    EVEN_MASK_,
                    parity_bonus_,
                    parity_enabled_,
                    root_pos_is_first_,
                    ply,
                    immediate_w_,
                    fork_w_,
                    center_bonus_,
                    vert_mul_,
                    vert3_ready_bonus_,
                    tempo_w_,
                    CENTER_ORDER_,
                    TOP_MASK_,
                    BOTTOM_MASK_,
                    COL_MASK_,
                    stride_i,
                    rows_i,
                    cols_i,
                ),
                np.int8(-1),
            )

        best_val = -1e100
        best_col = np.int8(-1)

        moves, m = order_moves(mask_, ply, killers_, history_, CENTER_ORDER_, TOP_MASK_)

        # prune immediate handovers if possible
        safe = np.empty(7, dtype=np.int8)
        s = 0
        for j in range(m):
            c = np.int32(moves[j])
            if not is_immediate_blunder(pos_, mask_, c, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, stride_i):
                safe[s] = np.int8(c)
                s += 1
        use_moves, use_len = (safe, s) if s > 0 else (moves, m)

        for j in range(use_len):
            c = np.int32(use_moves[j])
            mv = play_bit(mask_, c, BOTTOM_MASK_, COL_MASK_)
            nm = mask_ | mv
            next_pos = nm ^ (pos_ | mv)

            child_val, _ = negamax(
                next_pos,
                nm,
                np.int16(depth - 1),
                -beta,
                -alpha,
                np.int16(ply + 1),
                root_pos_is_first_,
                parity_enabled_,
                node_counter,
                max_nodes,
                MATE_SCORE_i,
                COL_MASK_,
                WIN_MASKS_,
                WIN_KIND_,
                WIN_B_,
                WIN_C_,
                WIN_R_,
                WARR_,
                DEF_,
                FN_,
                FF_,
                CENTER_MASK_,
                ODD_MASK_,
                EVEN_MASK_,
                parity_bonus_,
                immediate_w_,
                fork_w_,
                center_bonus_,
                vert_mul_,
                vert3_ready_bonus_,
                tempo_w_,
                CENTER_ORDER_,
                TOP_MASK_,
                BOTTOM_MASK_,
                stride_i,
                rows_i,
                cols_i,
                killers_,
                history_,
                FULL_MASK_,
                TT_pos_,
                TT_mask_,
                TT_depth_,
                TT_flag_,
                TT_val_,
                TT_move_,
            )
            val = -child_val

            if val > best_val:
                best_val = val
                best_col = np.int8(c)
            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                if killers_[ply, 0] != c:
                    killers_[ply, 1] = killers_[ply, 0]
                    killers_[ply, 0] = np.int8(c)
                history_[c] += int(depth) * int(depth)
                break

        tt_store(
            pos_, mask_, depth, best_val, alpha0, beta, best_col,
            TT_pos_, TT_mask_, TT_depth_, TT_flag_, TT_val_, TT_move_
        )
        return best_val, best_col

    @njit(cache=True, fastmath=True)
    def root_select_fixed(
        pos_: UINT,
        mask_: UINT,
        depth: np.int16,
        root_pos_is_first_: np.int8,
        parity_enabled_: np.int8,
        COL_MASK_: np.ndarray,
        WIN_MASKS_: np.ndarray,
        WIN_KIND_: np.ndarray,
        WIN_B_: np.ndarray,
        WIN_C_: np.ndarray,
        WIN_R_: np.ndarray,
        WARR_: np.ndarray,
        DEF_: float,
        FN_: float,
        FF_: float,
        CENTER_MASK_: UINT,
        ODD_MASK_: UINT,
        EVEN_MASK_: UINT,
        parity_bonus_: float,
        immediate_w_: float,
        fork_w_: float,
        center_bonus_: float,
        vert_mul_: float,
        vert3_ready_bonus_: float,
        tempo_w_: float,
        parity_move_w_: float,
        parity_unlock_w_: float,
        threatspace_w_: float,
        CENTER_ORDER_: np.ndarray,
        TOP_MASK_: np.ndarray,
        BOTTOM_MASK_: np.ndarray,
        stride_i: np.int32,
        rows_i: np.int32,
        cols_i: np.int32,
        FULL_MASK_: UINT,
        MATE_SCORE_i: float,
        killers_: np.ndarray,
        history_: np.ndarray,
        TT_pos_: np.ndarray,
        TT_mask_: np.ndarray,
        TT_depth_: np.ndarray,
        TT_flag_: np.ndarray,
        TT_val_: np.ndarray,
        TT_move_: np.ndarray,
    ) -> np.int32:

        # legal moves
        legal = np.empty(7, dtype=np.int8)
        L = 0
        for i in range(CENTER_ORDER_.shape[0]):
            c = np.int32(CENTER_ORDER_[i])
            if can_play(mask_, c, TOP_MASK_):
                legal[L] = np.int8(c)
                L += 1
        if L == 0:
            return np.int32(0)

        # immediate win
        for i in range(L):
            c = np.int32(legal[i])
            if is_winning_move(pos_, mask_, c, BOTTOM_MASK_, COL_MASK_, stride_i):
                return np.int32(c)

        # single must-block
        opp_pos = mask_ ^ pos_
        block_col, block_count = -1, 0
        for i in range(L):
            c = np.int32(legal[i])
            if is_winning_move(opp_pos, mask_, c, BOTTOM_MASK_, COL_MASK_, stride_i):
                block_col = c
                block_count += 1
                if block_count > 1:
                    break
        if block_count == 1:
            return np.int32(block_col)

        # avoid obvious handovers if possible
        safe = np.empty(7, dtype=np.int8)
        S = 0
        for i in range(L):
            c = np.int32(legal[i])
            if not is_immediate_blunder(pos_, mask_, c, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, stride_i):
                safe[S] = np.int8(c)
                S += 1
        if S > 0:
            legal, L = safe, S

        # -------------------------------------------------------------
        # Root tactical pre-pass: WIN-IN-2 / forcing threat detection
        #
        # True 2-ply forced win:
        #   After my move c, for every opponent reply oc, I have a win-in-1.
        #
        # This catches classic forks AND "hanging" forcing threats that can
        # slip through noisy eval or shallow-ish depths.
        # -------------------------------------------------------------
        for i in range(L):
            c = np.int32(legal[i])
            mv = play_bit(mask_, c, BOTTOM_MASK_, COL_MASK_)
            nm = mask_ | mv
            my_after = pos_ | mv
            opp_after = nm ^ my_after

            # If opponent has any immediate win now, it's not a clean win-in-2.
            if count_immediate_wins_bits(opp_after, nm, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, stride_i) != 0:
                continue

            forced = True
            any_reply = False

            for j in range(CENTER_ORDER_.shape[0]):
                oc = np.int32(CENTER_ORDER_[j])
                if not can_play(nm, oc, TOP_MASK_):
                    continue
                any_reply = True

                mv2 = play_bit(nm, oc, BOTTOM_MASK_, COL_MASK_)
                nm2 = nm | mv2

                # Do I have a win-in-1 after this reply?
                win1 = False
                for k in range(CENTER_ORDER_.shape[0]):
                    cc = np.int32(CENTER_ORDER_[k])
                    if can_play(nm2, cc, TOP_MASK_) and is_winning_move(my_after, nm2, cc, BOTTOM_MASK_, COL_MASK_, stride_i):
                        win1 = True
                        break

                if not win1:
                    forced = False
                    break

            if any_reply and forced:
                return np.int32(c)

        best_move = np.int32(legal[0])
        best_val = -1e100

        node_counter = np.zeros(1, dtype=np.int64)
        max_nodes = np.int64(9_000_000_000_000)

        # Root-only helpers: compute column heights once
        H = np.empty(cols_i, dtype=np.int16)
        for c0 in range(cols_i):
            H[c0] = popcount64(mask_ & COL_MASK_[c0])

        # Root role parity preferences
        # first-role prefers odd rows (r%2==0), second-role prefers even (r%2==1)
        root_is_first = (root_pos_is_first_ == np.int8(1))
        pref_parity_root = np.int16(0) if root_is_first else np.int16(1)
        pref_parity_opp = np.int16(1) if root_is_first else np.int16(0)

        for i in range(L):
            c = np.int32(legal[i])

            # Root-only small biases (do not replace search, just nudge it).
            bias = 0.0

            # Determine landing row for this move at root.
            r = np.int16(H[c])  # next free row in this column (0-based)
            if parity_enabled_ != np.int8(0):
                # move parity preference
                if (r & np.int16(1)) == pref_parity_root:
                    bias += parity_move_w_
                else:
                    bias -= parity_move_w_

                # parity unlock: playing at r unlocks r+1 (if within board)
                r2 = np.int16(r + 1)
                if r2 < np.int16(rows_i):
                    if (r2 & np.int16(1)) == pref_parity_opp:
                        bias -= parity_unlock_w_

            mv = play_bit(mask_, c, BOTTOM_MASK_, COL_MASK_)
            nm = mask_ | mv
            my_after = pos_ | mv
            opp_after = nm ^ my_after

            # Root-only threat-space approximation:
            if threatspace_w_ != 0.0:
                my_threats = count_immediate_wins_bits(my_after, nm, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, stride_i)
                bias += threatspace_w_ * float(my_threats)

                opp_threats = count_immediate_wins_bits(opp_after, nm, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, stride_i)
                bias -= DEF_ * threatspace_w_ * 0.25 * float(opp_threats)

            next_pos = opp_after  # opponent to move in negamax convention

            v, _ = negamax(
                next_pos,
                nm,
                np.int16(depth - 1),
                -MATE_SCORE_i,
                MATE_SCORE_i,
                np.int16(1),
                root_pos_is_first_,
                parity_enabled_,
                node_counter,
                max_nodes,
                MATE_SCORE_i,
                COL_MASK_,
                WIN_MASKS_,
                WIN_KIND_,
                WIN_B_,
                WIN_C_,
                WIN_R_,
                WARR_,
                DEF_,
                FN_,
                FF_,
                CENTER_MASK_,
                ODD_MASK_,
                EVEN_MASK_,
                parity_bonus_,
                immediate_w_,
                fork_w_,
                center_bonus_,
                vert_mul_,
                vert3_ready_bonus_,
                tempo_w_,
                CENTER_ORDER_,
                TOP_MASK_,
                BOTTOM_MASK_,
                stride_i,
                rows_i,
                cols_i,
                killers_,
                history_,
                FULL_MASK_,
                TT_pos_,
                TT_mask_,
                TT_depth_,
                TT_flag_,
                TT_val_,
                TT_move_,
            )

            val = -v + bias
            if val > best_val:
                best_val = val
                best_move = np.int32(c)

        return best_move

    _NUMBA_C4_CLASS_CACHE = dict(
        ROWS=ROWS,
        COLS=COLS,
        STRIDE=STRIDE,
        UINT=UINT,
        CENTER_ORDER=CENTER_ORDER,
        CENTER_MASK=CENTER_MASK,
        COL_MASK=COL_MASK,
        TOP_MASK=TOP_MASK,
        BOTTOM_MASK=BOTTOM_MASK,
        FULL_MASK=FULL_MASK,
        ODD_MASK=ODD_MASK,
        EVEN_MASK=EVEN_MASK,
        WIN_MASKS=WIN_MASKS,
        WIN_KIND=WIN_KIND,
        WIN_B=WIN_B,
        WIN_C=WIN_C,
        WIN_R=WIN_R,
        # defaults (not binding)
        DEFENSIVE=DEFENSIVE,
        FLOATING_NEAR=FLOATING_NEAR,
        FLOATING_FAR=FLOATING_FAR,
        CENTER_BONUS=CENTER_BONUS,
        VERT_MUL=VERT_MUL,
        VERT_3_READY_BONUS=VERT_3_READY_BONUS,
        TEMPO_W=TEMPO_W,
        # functions
        has_won=has_won,
        count_immediate_wins_bits=count_immediate_wins_bits,
        count_safe_moves=count_safe_moves,
        evaluate=evaluate,
        is_winning_move=is_winning_move,
        is_immediate_blunder=is_immediate_blunder,
        order_moves=order_moves,
        negamax=negamax,
        root_select_fixed=root_select_fixed,
    )
    return _NUMBA_C4_CLASS_CACHE


# ================================ The Class =================================

class Connect4Lookahead:
    ROWS = 6
    COLS = 7
    K = 4
    STRIDE = ROWS + 1
    CENTER_COL = 3
    OPENING_BOOK = True

    # Heuristic knobs (instance-overridable; notebook can mutate per variant)
    immediate_w = C4_IMMEDIATE_W
    fork_w = C4_FORK_W
    DEFENSIVE = C4_DEFENSIVE
    FLOATING_NEAR = C4_FLOATING_NEAR
    FLOATING_FAR = C4_FLOATING_FAR
    CENTER_BONUS = C4_CENTER_BONUS
    PARITY_BONUS = C4_PARITY_BONUS

    # New knobs
    VERT_MUL = C4_VERT_MUL
    VERT_3_READY_BONUS = C4_VERT_3_READY_BONUS
    TEMPO_W = C4_TEMPO_W
    PARITY_MOVE_W = C4_PARITY_MOVE_W
    PARITY_UNLOCK_W = C4_PARITY_UNLOCK_W
    THREATSPACE_W = C4_THREATSPACE_W

    # Opening book determinism (important for weight sweeps / round-robin stability)
    # For stochasticity, set instance field OPENING_RANDOM = True.
    OPENING_RANDOM = True  # False

    _CENTER_ORDER = [3, 4, 2, 5, 1, 6, 0]

    _PRECOMP_DONE = False
    COL_MASK: List[int] = [0] * COLS
    TOP_MASK: List[int] = [0] * COLS
    BOTTOM_MASK: List[int] = [0] * COLS
    FULL_MASK: int = 0
    CENTER_MASK: int = 0
    WIN_MASKS: List[int] = []
    WIN_CELLS: List[List[Tuple[int, int, int]]] = []

    def __init__(self, weights=None):
        if weights is None:
            self.weights: Dict[int, float] = dict(C4_DEFAULT_WEIGHTS_ITEMS)
        else:
            self.weights = dict(weights)

        self.SOFT_MATE = float(C4_SOFT_MATE_MULT) * float(self.weights[4])
        self.MATE_SCORE = float(C4_MATE_SCORE_MULT) * float(self.weights[4])

        if not Connect4Lookahead._PRECOMP_DONE:
            self._build_precomp()

        self._N = _ensure_numba_cache()

    # ============================== Public API ===============================

    def get_heuristic(self, board, player) -> float:
        p1, p2, mask = self._parse_board_bitboards(board)
        me_mark = self._p(player)
        me = p1 if me_mark == 1 else p2

        # weights -> WARR
        WARR = np.zeros(self.K + 1, dtype=np.float64)
        for k in (2, 3, 4):
            WARR[k] = float(self.weights.get(k, 0.0))

        # parity role anchor: center-bottom owner, else disabled
        role_first_mark, parity_enabled = self._role_first_from_center(mask, p1, p2)
        root_pos_is_first = np.int8(1 if (parity_enabled and me_mark == role_first_mark) else 0)

        return float(
            self._N["evaluate"](
                np.uint64(me),
                np.uint64(mask),
                self._N["COL_MASK"],
                self._N["WIN_MASKS"],
                self._N["WIN_KIND"],
                self._N["WIN_B"],
                self._N["WIN_C"],
                self._N["WIN_R"],
                WARR,
                float(self.DEFENSIVE),
                float(self.FLOATING_NEAR),
                float(self.FLOATING_FAR),
                np.uint64(self._N["CENTER_MASK"]),
                np.uint64(self._N["ODD_MASK"]),
                np.uint64(self._N["EVEN_MASK"]),
                float(self.PARITY_BONUS),
                np.int8(1 if parity_enabled else 0),
                root_pos_is_first,
                np.int16(0),
                float(self.immediate_w),
                float(self.fork_w),
                float(self.CENTER_BONUS),
                float(self.VERT_MUL),
                float(self.VERT_3_READY_BONUS),
                float(self.TEMPO_W),
                self._N["CENTER_ORDER"],
                self._N["TOP_MASK"],
                self._N["BOTTOM_MASK"],
                self._N["COL_MASK"],
                np.int32(self.STRIDE),
                np.int32(self.ROWS),
                np.int32(self.COLS),
            )
        )

    def is_terminal(self, board) -> bool:
        p1, p2, mask = self._parse_board_bitboards(board)
        hw = self._N["has_won"]
        return bool(
            hw(np.uint64(p1), np.int32(self.STRIDE))
            or hw(np.uint64(p2), np.int32(self.STRIDE))
            or (np.uint64(mask) == self._N["FULL_MASK"])
        )

    def minimax(self, board, depth, maximizing, player, alpha, beta) -> float:
        p1, p2, mask = self._parse_board_bitboards(board)
        root_pov = self._p(player)
        to_move = root_pov if maximizing else -root_pov
        pos = p1 if to_move == 1 else p2

        WARR = np.zeros(self.K + 1, dtype=np.float64)
        for k in (2, 3, 4):
            WARR[k] = float(self.weights.get(k, 0.0))

        killers = np.full((64, 2), -1, dtype=np.int8)
        history = np.zeros(self.COLS, dtype=np.int32)
        node_counter = np.zeros(1, dtype=np.int64)
        max_nodes = np.int64(9_000_000_000_000)

        TT_SIZE = 1 << 16
        TT_pos = np.zeros(TT_SIZE, dtype=np.uint64)
        TT_mask = np.zeros(TT_SIZE, dtype=np.uint64)
        TT_depth = np.full(TT_SIZE, -1, dtype=np.int16)
        TT_flag = np.zeros(TT_SIZE, dtype=np.int8)
        TT_val = np.zeros(TT_SIZE, dtype=np.float64)
        TT_move = np.full(TT_SIZE, -1, dtype=np.int8)

        role_first_mark, parity_enabled = self._role_first_from_center(mask, p1, p2)
        root_pos_is_first = np.int8(1 if (parity_enabled and to_move == role_first_mark) else 0)

        ply0 = np.int16(0)

        v, _ = self._N["negamax"](
            np.uint64(pos),
            np.uint64(mask),
            np.int16(depth),
            float(alpha),
            float(beta),
            ply0,
            root_pos_is_first,
            np.int8(1 if parity_enabled else 0),
            node_counter,
            max_nodes,
            float(self.MATE_SCORE),
            self._N["COL_MASK"],
            self._N["WIN_MASKS"],
            self._N["WIN_KIND"],
            self._N["WIN_B"],
            self._N["WIN_C"],
            self._N["WIN_R"],
            WARR,
            float(self.DEFENSIVE),
            float(self.FLOATING_NEAR),
            float(self.FLOATING_FAR),
            np.uint64(self._N["CENTER_MASK"]),
            np.uint64(self._N["ODD_MASK"]),
            np.uint64(self._N["EVEN_MASK"]),
            float(self.PARITY_BONUS),
            float(self.immediate_w),
            float(self.fork_w),
            float(self.CENTER_BONUS),
            float(self.VERT_MUL),
            float(self.VERT_3_READY_BONUS),
            float(self.TEMPO_W),
            self._N["CENTER_ORDER"],
            self._N["TOP_MASK"],
            self._N["BOTTOM_MASK"],
            np.int32(self.STRIDE),
            np.int32(self.ROWS),
            np.int32(self.COLS),
            killers,
            history,
            np.uint64(self._N["FULL_MASK"]),
            TT_pos,
            TT_mask,
            TT_depth,
            TT_flag,
            TT_val,
            TT_move,
        )
        val = float(v)
        return val if to_move == root_pov else -val

    def n_step_lookahead(self, board, player, depth=3) -> int:

        def _pick_side(cols=(2, 4)):
            legal = [c for c in cols if self._can_play_py(mask, c)]
            if not legal:
                return None
            if len(legal) == 1:
                return int(legal[0])
            if bool(getattr(self, "OPENING_RANDOM", False)):
                return int(np.random.default_rng().choice(np.array(legal, dtype=np.int8)))
            return int(legal[0])  # deterministic: prefer 2 then 4

        p1, p2, mask = self._parse_board_bitboards(board)
        stones = int(np.count_nonzero(board))
        me_mark = self._p(player)

        # ---------- opening book (depth-independent) ----------
        if stones == 0 and me_mark == 1:
            return self.CENTER_COL

        if stones == 1 and me_mark == -1:
            b_d1 = 1 << (self.CENTER_COL * self.STRIDE + 0)
            if mask & b_d1:
                choices = [2, 4]  # C1/E1
                if bool(getattr(self, "OPENING_RANDOM", False)):
                    return int(np.random.default_rng().choice(np.array(choices, dtype=np.int8)))
                return int(choices[0])  # deterministic: C1

        if stones == 2 and me_mark == 1:
            b_d1 = 1 << (self.CENTER_COL * self.STRIDE + 0)
            if mask & b_d1:
                b_d2 = 1 << (self.CENTER_COL * self.STRIDE + 1)

                b_a1 = 1 << (0 * self.STRIDE + 0)
                b_b1 = 1 << (1 * self.STRIDE + 0)
                b_c1 = 1 << (2 * self.STRIDE + 0)
                b_e1 = 1 << (4 * self.STRIDE + 0)
                b_f1 = 1 << (5 * self.STRIDE + 0)
                b_g1 = 1 << (6 * self.STRIDE + 0)

                # If P2 stacked center: 1.d1 d2 -> P1 plays d3 (center again)
                if mask & b_d2:
                    if self._can_play_py(mask, self.CENTER_COL):
                        return self.CENTER_COL

                # If P2 replied C1/E1: 1.d1 c1 -> 2.f1 ; 1.d1 e1 -> 2.b1
                if mask & b_c1:
                    return 5  # F1
                if mask & b_e1:
                    return 1  # B1

                # Corner replies: vs A1 -> E1, vs G1 -> C1
                if mask & b_a1:
                    return 4  # E1
                if mask & b_g1:
                    return 2  # C1

                # B1/F1 replies: mirror
                if mask & b_b1:
                    return 5  # F1
                if mask & b_f1:
                    return 1  # B1

                # Fallback: take center (D2) if legal
                if self._can_play_py(mask, self.CENTER_COL):
                    return self.CENTER_COL

        if stones == 3 and me_mark == -1:
            # If the stacked-center line happened: D1, D2, D3 are filled -> P2 plays D4
            b_d1 = 1 << (self.CENTER_COL * self.STRIDE + 0)
            b_d2 = 1 << (self.CENTER_COL * self.STRIDE + 1)
            b_d3 = 1 << (self.CENTER_COL * self.STRIDE + 2)
            if (mask & b_d1) and (mask & b_d2) and (mask & b_d3):
                if self._can_play_py(mask, self.CENTER_COL):
                    return self.CENTER_COL

        # ---------------------------------------------------------------------------

        me = p1 if me_mark == 1 else p2

        WARR = np.zeros(self.K + 1, dtype=np.float64)
        for k in (2, 3, 4):
            WARR[k] = float(self.weights.get(k, 0.0))

        killers = np.full((64, 2), -1, dtype=np.int8)
        history = np.zeros(self.COLS, dtype=np.int32)

        TT_SIZE = 1 << 16
        TT_pos = np.zeros(TT_SIZE, dtype=np.uint64)
        TT_mask = np.zeros(TT_SIZE, dtype=np.uint64)
        TT_depth = np.full(TT_SIZE, -1, dtype=np.int16)
        TT_flag = np.zeros(TT_SIZE, dtype=np.int8)
        TT_val = np.zeros(TT_SIZE, dtype=np.float64)
        TT_move = np.full(TT_SIZE, -1, dtype=np.int8)

        role_first_mark, parity_enabled = self._role_first_from_center(mask, p1, p2)
        root_pos_is_first = np.int8(1 if (parity_enabled and me_mark == role_first_mark) else 0)

        mv = self._N["root_select_fixed"](
            np.uint64(me),
            np.uint64(mask),
            np.int16(depth),
            root_pos_is_first,
            np.int8(1 if parity_enabled else 0),
            self._N["COL_MASK"],
            self._N["WIN_MASKS"],
            self._N["WIN_KIND"],
            self._N["WIN_B"],
            self._N["WIN_C"],
            self._N["WIN_R"],
            WARR,
            float(self.DEFENSIVE),
            float(self.FLOATING_NEAR),
            float(self.FLOATING_FAR),
            np.uint64(self._N["CENTER_MASK"]),
            np.uint64(self._N["ODD_MASK"]),
            np.uint64(self._N["EVEN_MASK"]),
            float(self.PARITY_BONUS),
            float(self.immediate_w),
            float(self.fork_w),
            float(self.CENTER_BONUS),
            float(self.VERT_MUL),
            float(self.VERT_3_READY_BONUS),
            float(self.TEMPO_W),
            float(self.PARITY_MOVE_W),
            float(self.PARITY_UNLOCK_W),
            float(self.THREATSPACE_W),
            self._N["CENTER_ORDER"],
            self._N["TOP_MASK"],
            self._N["BOTTOM_MASK"],
            np.int32(self.STRIDE),
            np.int32(self.ROWS),
            np.int32(self.COLS),
            np.uint64(self._N["FULL_MASK"]),
            float(self.MATE_SCORE),
            killers,
            history,
            TT_pos,
            TT_mask,
            TT_depth,
            TT_flag,
            TT_val,
            TT_move,
        )
        return int(mv)

    def has_four(self, board, player: int) -> bool:
        p1, p2, _ = self._parse_board_bitboards(board)
        bb = p1 if self._p(player) == 1 else p2
        return bool(self._N["has_won"](np.uint64(bb), np.int32(self.STRIDE)))

    check_win = has_four

    def count_immediate_wins(self, board, player: int) -> List[int]:
        p1, p2, mask = self._parse_board_bitboards(board)
        me = p1 if self._p(player) == 1 else p2

        wins = []
        for c in self._CENTER_ORDER:
            if (mask & self.TOP_MASK[c]) == 0 and self._is_winning_move_py(me, mask, c):
                wins.append(c)
        return wins

    def compute_fork_signals(self, board_before, board_after, mover: int) -> Dict[str, int]:
        # Kept for compatibility with older code, but your env no longer needs it.
        mover = self._p(mover)
        opp = -mover
        my_after = len(self.count_immediate_wins(board_after, mover))
        opp_before = len(self.count_immediate_wins(board_before, opp))
        opp_after = len(self.count_immediate_wins(board_after, opp))
        return {"my_after": my_after, "opp_before": opp_before, "opp_after": opp_after}

    # ---------------- Analytics helpers preserved ----------------

    def count_pure(self, board, player, n: int) -> int:
        p1, p2, _ = self._parse_board_bitboards(board)
        me = p1 if self._p(player) == 1 else p2
        opp = p2 if self._p(player) == 1 else p1
        cnt = 0
        for wmask in self.WIN_MASKS:
            mp = wmask & me
            mo = wmask & opp
            if mo == 0 and mp.bit_count() == n:
                cnt += 1
        return cnt

    def count_pure_block_delta(self, before_board, after_board, player, n: int) -> int:
        before = self.count_pure(before_board, player, n)
        after = self.count_pure(after_board, player, n)
        return max(0, before - after)

    # ============================== Internals ===============================

    @classmethod
    def _build_precomp(cls) -> None:
        cls.WIN_MASKS.clear()
        cls.WIN_CELLS.clear()

        full = 0
        for c in range(cls.COLS):
            col_bits = 0
            for r in range(cls.ROWS):
                col_bits |= 1 << (c * cls.STRIDE + r)
            cls.COL_MASK[c] = col_bits
            cls.BOTTOM_MASK[c] = 1 << (c * cls.STRIDE + 0)
            cls.TOP_MASK[c] = 1 << (c * cls.STRIDE + (cls.ROWS - 1))
            full |= col_bits
        cls.FULL_MASK = full
        cls.CENTER_MASK = cls.COL_MASK[cls.CENTER_COL]

        def bit_at(c, r):
            return 1 << (c * cls.STRIDE + r)

        # horizontal
        for r in range(cls.ROWS):
            for c in range(cls.COLS - cls.K + 1):
                cells = [(r, c + i) for i in range(cls.K)]
                mask, triples = 0, []
                for rr, cc in cells:
                    b = bit_at(cc, rr)
                    mask |= b
                    triples.append((b, cc, rr))
                cls.WIN_MASKS.append(mask)
                cls.WIN_CELLS.append(triples)

        # vertical
        for c in range(cls.COLS):
            for r in range(cls.ROWS - cls.K + 1):
                cells = [(r + i, c) for i in range(cls.K)]
                mask, triples = 0, []
                for rr, cc in cells:
                    b = bit_at(cc, rr)
                    mask |= b
                    triples.append((b, cc, rr))
                cls.WIN_MASKS.append(mask)
                cls.WIN_CELLS.append(triples)

        # diag up-right
        for r in range(cls.ROWS - cls.K + 1):
            for c in range(cls.COLS - cls.K + 1):
                cells = [(r + i, c + i) for i in range(cls.K)]
                mask, triples = 0, []
                for rr, cc in cells:
                    b = bit_at(cc, rr)
                    mask |= b
                    triples.append((b, cc, rr))
                cls.WIN_MASKS.append(mask)
                cls.WIN_CELLS.append(triples)

        # diag up-left
        for r in range(cls.ROWS - cls.K + 1):
            for c in range(cls.K - 1, cls.COLS):
                cells = [(r + i, c - i) for i in range(cls.K)]
                mask, triples = 0, []
                for rr, cc in cells:
                    b = bit_at(cc, rr)
                    mask |= b
                    triples.append((b, cc, rr))
                cls.WIN_MASKS.append(mask)
                cls.WIN_CELLS.append(triples)

        cls._PRECOMP_DONE = True

    @staticmethod
    def _p(p: int) -> int:
        # Map {1,2} or {1,-1} to {+1,-1}
        return -1 if p == 2 else int(p)

    @classmethod
    def _can_play_py(cls, mask: int, c: int) -> bool:
        return (mask & cls.TOP_MASK[c]) == 0

    @classmethod
    def _play_bit_py(cls, mask: int, c: int) -> int:
        return (mask + cls.BOTTOM_MASK[c]) & cls.COL_MASK[c]

    @classmethod
    def _has_won_py(cls, bb: int) -> bool:
        m = bb & (bb >> 1)
        if m & (m >> 2):
            return True
        m = bb & (bb >> cls.STRIDE)
        if m & (m >> (2 * cls.STRIDE)):
            return True
        m = bb & (bb >> (cls.STRIDE + 1))
        if m & (m >> (2 * (cls.STRIDE + 1))):
            return True
        m = bb & (bb >> (cls.STRIDE - 1))
        if m & (m >> (2 * (cls.STRIDE - 1))):
            return True
        return False

    @classmethod
    def _is_winning_move_py(cls, pos: int, mask: int, c: int) -> bool:
        mv = cls._play_bit_py(mask, c)
        return cls._has_won_py(pos | mv)

    def _parse_board_bitboards(self, board) -> Tuple[int, int, int]:
        """
        Accepts numpy-like 2D board in {0,1,2} or {0,1,-1}, top-based rows.
        Returns (p1_bb, p2_bb, mask) as Python ints.
        """
        p1 = 0
        p2 = 0
        mask = 0
        for r_top in range(self.ROWS):
            r = self.ROWS - 1 - r_top  # flip to bottom-based
            for c in range(self.COLS):
                v = int(board[r_top, c])
                if v == 0:
                    continue
                b = 1 << (c * self.STRIDE + r)
                mask |= b
                if v == 1:
                    p1 |= b
                else:
                    p2 |= b
        return p1, p2, mask

    @classmethod
    def _role_first_from_center(cls, mask: int, p1: int, p2: int) -> Tuple[int, bool]:
        """
        Determine 'first player role' from center-bottom owner (D1).
        Returns (role_first_mark, enabled).
          - role_first_mark is +1 or -1
          - enabled is False if center-bottom is empty
        """
        b_d1 = 1 << (cls.CENTER_COL * cls.STRIDE + 0)
        if (mask & b_d1) == 0:
            return 1, False  # disabled; mark irrelevant
        if (p1 & b_d1) != 0:
            return 1, True
        return -1, True

    # ---------------- scoring helpers preserved ----------------

    def n_step_action_scores(self, board, player, depth=1) -> np.ndarray:
        p1, p2, mask = self._parse_board_bitboards(board)
        me_mark = self._p(player)
        me = p1 if me_mark == 1 else p2

        WARR = np.zeros(self.K + 1, dtype=np.float64)
        for k in (2, 3, 4):
            WARR[k] = float(self.weights.get(k, 0.0))

        scores = np.full(self.COLS, -np.inf, dtype=np.float64)
        killers = np.full((64, 2), -1, dtype=np.int8)
        history = np.zeros(self.COLS, dtype=np.int32)

        role_first_mark, parity_enabled = self._role_first_from_center(mask, p1, p2)
        root_pos_is_first = np.int8(1 if (parity_enabled and me_mark == role_first_mark) else 0)
        parity_enabled_i8 = np.int8(1 if parity_enabled else 0)

        for c in self._CENTER_ORDER:
            if (mask & self.TOP_MASK[c]) != 0:
                continue

            if self._is_winning_move_py(me, mask, c):
                scores[c] = self.MATE_SCORE
                continue

            mv = self._play_bit_py(mask, c)
            nm = mask | mv
            next_pos = nm ^ (me | mv)

            TT_SIZE = 1 << 16
            TT_pos = np.zeros(TT_SIZE, dtype=np.uint64)
            TT_mask = np.zeros(TT_SIZE, dtype=np.uint64)
            TT_depth = np.full(TT_SIZE, -1, dtype=np.int16)
            TT_flag = np.zeros(TT_SIZE, dtype=np.int8)
            TT_val = np.zeros(TT_SIZE, dtype=np.float64)
            TT_move = np.full(TT_SIZE, -1, dtype=np.int8)

            node_counter = np.zeros(1, dtype=np.int64)
            max_nodes = np.int64(9_000_000_000_000)

            v, _ = self._N["negamax"](
                np.uint64(next_pos),
                np.uint64(nm),
                np.int16(depth - 1),
                -float(self.MATE_SCORE),
                float(self.MATE_SCORE),
                np.int16(1),
                root_pos_is_first,
                parity_enabled_i8,
                node_counter,
                max_nodes,
                float(self.MATE_SCORE),
                self._N["COL_MASK"],
                self._N["WIN_MASKS"],
                self._N["WIN_KIND"],
                self._N["WIN_B"],
                self._N["WIN_C"],
                self._N["WIN_R"],
                WARR,
                float(self.DEFENSIVE),
                float(self.FLOATING_NEAR),
                float(self.FLOATING_FAR),
                np.uint64(self._N["CENTER_MASK"]),
                np.uint64(self._N["ODD_MASK"]),
                np.uint64(self._N["EVEN_MASK"]),
                float(self.PARITY_BONUS),
                float(self.immediate_w),
                float(self.fork_w),
                float(self.CENTER_BONUS),
                float(self.VERT_MUL),
                float(self.VERT_3_READY_BONUS),
                float(self.TEMPO_W),
                self._N["CENTER_ORDER"],
                self._N["TOP_MASK"],
                self._N["BOTTOM_MASK"],
                np.int32(self.STRIDE),
                np.int32(self.ROWS),
                np.int32(self.COLS),
                killers,
                history,
                np.uint64(self._N["FULL_MASK"]),
                TT_pos,
                TT_mask,
                TT_depth,
                TT_flag,
                TT_val,
                TT_move,
            )
            scores[c] = -float(v)

        return scores

    def policy_scores_delta(self, board, player, depth=1) -> np.ndarray:
        base = float(self.get_heuristic(board, player))
        sc = self.n_step_action_scores(board, player, depth=depth)
        return sc - base

    def _choose_uniform_among_best(
        self,
        scores: np.ndarray,
        legal_cols: List[int],
        rng=None,
        atol: float = 1e-9,
        rtol: float = 1e-9,
    ) -> int:
        if rng is None:
            rng = np.random.default_rng()
        if not legal_cols:
            return 0
        best = max(float(scores[c]) for c in legal_cols)
        best_cols = [c for c in legal_cols if np.isclose(scores[c], best, atol=atol, rtol=rtol)]
        if not best_cols:
            best_cols = legal_cols
        return int(rng.choice(best_cols))

    # ---------------- Baselines (convenience) ----------------

    def legal_actions(self, board=None, mask=None) -> List[int]:
        if mask is None:
            if board is None:
                raise ValueError("legal_actions() requires either board=... or mask=...")
            _, _, mask = self._parse_board_bitboards(board)
        return [c for c in range(self.COLS) if (mask & self.TOP_MASK[c]) == 0]

    def random_action(self, board, rng: Optional[object] = None) -> int:
        legal = self.legal_actions(board)
        if not legal:
            return 0
        if rng is None:
            rng = np.random.default_rng()
        try:
            return int(rng.choice(legal))
        except Exception:
            return int(np.random.default_rng().choice(legal))

    def leftmost_action(self, board) -> int:
        legal = self.legal_actions(board)
        return int(min(legal)) if legal else 0

    def baseline_action(self, board, kind: str = "random", rng: Optional[object] = None) -> int:
        k = (kind or "").lower()
        if k in ("random", "rnd"):
            return self.random_action(board, rng=rng)
        if k in ("leftmost", "left", "lm"):
            return self.leftmost_action(board)
        if k in ("center", "centre"):
            _, _, mask = self._parse_board_bitboards(board)
            for c in self._CENTER_ORDER:
                if (mask & self.TOP_MASK[c]) == 0:
                    return int(c)
            return 0
        raise ValueError(f"Unknown baseline kind={kind!r}")

    def baseline_policy_probs(self, board, kind: str = "random") -> np.ndarray:
        probs = np.zeros(self.COLS, dtype=np.float32)
        _, _, mask = self._parse_board_bitboards(board)
        k = (kind or "").lower()

        if k in ("random", "rnd"):
            legal = 0
            for c in range(self.COLS):
                if (mask & self.TOP_MASK[c]) == 0:
                    legal += 1
            if legal == 0:
                probs[0] = 1.0
                return probs
            p = 1.0 / float(legal)
            for c in range(self.COLS):
                if (mask & self.TOP_MASK[c]) == 0:
                    probs[c] = p
            return probs

        if k in ("leftmost", "left", "lm"):
            for c in range(self.COLS):
                if (mask & self.TOP_MASK[c]) == 0:
                    probs[c] = 1.0
                    return probs
            probs[0] = 1.0
            return probs

        if k in ("center", "centre"):
            for c in self._CENTER_ORDER:
                if (mask & self.TOP_MASK[c]) == 0:
                    probs[c] = 1.0
                    return probs
            probs[0] = 1.0
            return probs

        raise ValueError(f"Unknown baseline kind={kind!r}")

    def sample_action_from_probs(self, probs: np.ndarray, rng: Optional[object] = None) -> int:
        if rng is None:
            rng = np.random.default_rng()
        probs = np.asarray(probs, dtype=np.float64)
        s = probs.sum()
        if s <= 0:
            return 0
        probs = probs / s
        try:
            return int(rng.choice(self.COLS, p=probs))
        except Exception:
            return int(np.random.default_rng().choice(self.COLS, p=probs))

    def _baseline_selfcheck(self, board, kind="random"):
        a = self.baseline_action(board, kind=kind)
        p = self.baseline_policy_probs(board, kind=kind)
        assert p[a] > 0.0, (kind, a, p)
