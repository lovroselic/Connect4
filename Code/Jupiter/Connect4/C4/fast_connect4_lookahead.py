# replacing: # fast_connect4_lookahead.py
# numba_connect4_lookahead.py

# Drop-in Numba-accelerated version of your bitboard lookahead class.
# Public API preserved:
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

from typing import List, Tuple, Dict
import numpy as np
from numba import njit


# ============================ Numba Core & Cache ============================

_NUMBA_C4_CLASS_CACHE = {}  # built once, reused by all instances


def _ensure_numba_cache():
    """Build & JIT-compile once; stash in _NUMBA_C4_CLASS_CACHE."""
    global _NUMBA_C4_CLASS_CACHE
    if _NUMBA_C4_CLASS_CACHE: return _NUMBA_C4_CLASS_CACHE

    # ---------- constants ----------
    ROWS, COLS, K = 6, 7, 4
    STRIDE = ROWS + 1
    UINT = np.uint64

    CENTER_COL = 3
    CENTER_ORDER = np.array([3, 4, 2, 5, 1, 6, 0], dtype=np.int8)

    DEFENSIVE = 1.5
    FLOATING_NEAR = 0.75
    FLOATING_FAR  = 0.50
    CENTER_BONUS  = 6.0

    # Bitboard columns & masks
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
    CENTER_MASK = COL_MASK[CENTER_COL]

    def _bit_at(c, r): return UINT(1) << UINT(c * STRIDE + r)

    # Windows (69 of length 4)
    _win_bits, _win_cols, _win_rows, _WIN_MASKS = [], [], [], []
    def _add_window(cells):
        mask = UINT(0); bs, cs, rs = [], [], []
        for rr, cc in cells:
            b = _bit_at(cc, rr); mask |= b
            bs.append(b); cs.append(cc); rs.append(rr)
        _WIN_MASKS.append(mask); _win_bits.append(bs); _win_cols.append(cs); _win_rows.append(rs)

    # horiz
    for r in range(ROWS):
        for c in range(COLS - K + 1): _add_window([(r, c+i) for i in range(K)])
    # vert
    for c in range(COLS):
        for r in range(ROWS - K + 1): _add_window([(r+i, c) for i in range(K)])
    # diag up-right
    for r in range(ROWS - K + 1):
        for c in range(COLS - K + 1): _add_window([(r+i, c+i) for i in range(K)])
    # diag up-left
    for r in range(ROWS - K + 1):
        for c in range(K - 1, COLS):  _add_window([(r+i, c-i) for i in range(K)])

    WIN_MASKS = np.array(_WIN_MASKS, dtype=UINT)
    WIN_B = np.array(_win_bits, dtype=UINT)        # (W,4)
    WIN_C = np.array(_win_cols, dtype=np.int8)     # (W,4)
    WIN_R = np.array(_win_rows, dtype=np.int8)     # (W,4)

    # ---------- Numba core (no objmode, all arrays passed as args) ----------
    @njit(cache=True, fastmath=True)
    def popcount64(x: UINT) -> np.int32:
        cnt = 0
        while x != UINT(0):
            x &= (x - UINT(1))
            cnt += 1
        return cnt

    @njit(cache=True, fastmath=True)
    def has_won(bb: UINT, stride_i: np.int32) -> bool:
        m = bb & (bb >> UINT(1))                 # vertical
        if (m & (m >> UINT(2))) != UINT(0): return True
        m = bb & (bb >> UINT(stride_i))          # horizontal
        if (m & (m >> UINT(2 * stride_i))) != UINT(0): return True
        m = bb & (bb >> UINT(stride_i + 1))      # diag up-right
        if (m & (m >> UINT(2 * (stride_i + 1)))) != UINT(0): return True
        m = bb & (bb >> UINT(stride_i - 1))      # diag up-left
        if (m & (m >> UINT(2 * (stride_i - 1)))) != UINT(0): return True
        return False

    @njit(cache=True, fastmath=True)
    def can_play(mask_: UINT, c: np.int32, TOP_MASK_: np.ndarray) -> bool:
        return (mask_ & TOP_MASK_[c]) == UINT(0)

    @njit(cache=True, fastmath=True)
    def play_bit(mask_: UINT, c: np.int32, BOTTOM_MASK_: np.ndarray, COL_MASK_: np.ndarray) -> UINT:
        return (mask_ + BOTTOM_MASK_[c]) & COL_MASK_[c]

    @njit(cache=True, fastmath=True)
    def is_winning_move(pos_: UINT, mask_: UINT, c: np.int32,
                        BOTTOM_MASK_: np.ndarray, COL_MASK_: np.ndarray, stride_i: np.int32) -> bool:
        mv = play_bit(mask_, c, BOTTOM_MASK_, COL_MASK_)
        return has_won(pos_ | mv, stride_i)

    @njit(cache=True, fastmath=True)
    def count_immediate_wins_bits(pos_: UINT, mask_: UINT, CENTER_ORDER_: np.ndarray,
                                  TOP_MASK_: np.ndarray, BOTTOM_MASK_: np.ndarray,
                                  COL_MASK_: np.ndarray, stride_i: np.int32) -> np.int32:
        cnt = 0
        for i in range(CENTER_ORDER_.shape[0]):
            c = np.int32(CENTER_ORDER_[i])
            if can_play(mask_, c, TOP_MASK_) and is_winning_move(pos_, mask_, c, BOTTOM_MASK_, COL_MASK_, stride_i):
                cnt += 1
        return cnt

    @njit(cache=True, fastmath=True)
    def evaluate(pos_: UINT, mask_: UINT, COL_MASK_: np.ndarray, WIN_MASKS_: np.ndarray, WIN_B_: np.ndarray,
                 WIN_C_: np.ndarray, WIN_R_: np.ndarray, WARR_: np.ndarray, DEF_: float, FN_: float, FF_: float,
                 CENTER_MASK_: UINT, immediate_w_: float, fork_w_: float, center_bonus_: float,
                 CENTER_ORDER_: np.ndarray, TOP_MASK_: np.ndarray, BOTTOM_MASK_: np.ndarray,
                 COL_MASK2_: np.ndarray, stride_i: np.int32, rows_i: np.int32, cols_i: np.int32) -> float:
        opp = mask_ ^ pos_
        # heights
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

            need = 0
            if p == 0 or o == 0:
                for k2 in range(4):
                    b = WIN_B_[idx, k2]
                    if (mask_ & b) == UINT(0):
                        cc = np.int32(WIN_C_[idx, k2])
                        rr = np.int32(WIN_R_[idx, k2])
                        dh = rr - np.int32(H[cc])
                        if dh > 0:
                            need += dh

            mul = 1.0 if need == 0 else (FN_ if need == 1 else FF_)
            if o == 0:
                score += mul * (WARR_[p] if p <= 4 else 0.0)
            elif p == 0:
                score -= DEF_ * mul * (WARR_[o] if o <= 4 else 0.0)

        my_imm  = count_immediate_wins_bits(pos_,  mask_, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK2_, stride_i)
        opp_imm = count_immediate_wins_bits(opp,   mask_, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK2_, stride_i)
        score += immediate_w_ * (my_imm - DEF_ * opp_imm)
        if my_imm >= 2:  score += fork_w_ * (my_imm - 1)
        if opp_imm >= 2: score -= DEF_ * (fork_w_ * (opp_imm - 1))
        score += center_bonus_ * (popcount64(pos_ & CENTER_MASK_) - popcount64(opp & CENTER_MASK_))
        return score

    @njit(cache=True, fastmath=True)
    def is_immediate_blunder(pos_: UINT, mask_: UINT, c: np.int32, CENTER_ORDER_: np.ndarray, TOP_MASK_: np.ndarray,
                             BOTTOM_MASK_: np.ndarray, COL_MASK_: np.ndarray, stride_i: np.int32) -> bool:
        mv = play_bit(mask_, c, BOTTOM_MASK_, COL_MASK_)
        nm = mask_ | mv
        opp_pos = nm ^ (pos_ | mv)
        for i in range(CENTER_ORDER_.shape[0]):
            cc = np.int32(CENTER_ORDER_[i])
            if can_play(nm, cc, TOP_MASK_) and is_winning_move(opp_pos, nm, cc, BOTTOM_MASK_, COL_MASK_, stride_i):
                return True
        return False

    @njit(cache=True, fastmath=True)
    def order_moves(mask_: UINT, ply: np.int32, killers_: np.ndarray, history_: np.ndarray,
                    CENTER_ORDER_: np.ndarray, TOP_MASK_: np.ndarray) -> (np.ndarray, np.int32):
        moves = np.empty(7, dtype=np.int8)
        scores = np.empty(7, dtype=np.int32)
        m = 0
        k1, k2 = killers_[ply, 0], killers_[ply, 1]
        for i in range(CENTER_ORDER_.shape[0]):
            c = np.int32(CENTER_ORDER_[i])
            if can_play(mask_, c, TOP_MASK_):
                s = 0
                if c == k1: s += 1_000_000
                elif c == k2: s += 500_000
                s += history_[c]
                moves[m] = np.int8(c)
                scores[m] = s
                m += 1
        # insertion sort (desc)
        for i in range(1, m):
            km, ks = moves[i], scores[i]
            j = i - 1
            while j >= 0 and scores[j] < ks:
                moves[j+1] = moves[j]
                scores[j+1] = scores[j]
                j -= 1
            moves[j+1] = km
            scores[j+1] = ks
        return moves, m

    @njit(cache=True, fastmath=True)
    def tt_hash_local(pos_: UINT, mask_: UINT, size_mask_i: np.int32) -> np.int32:
        h = np.uint64(pos_) ^ (np.uint64(mask_) * np.uint64(0x9E3779B97F4A7C15))
        h ^= (h >> np.uint64(7))
        return np.int32(h & np.uint64(size_mask_i))

    @njit(cache=True, fastmath=True)
    def tt_lookup(pos_: UINT, mask_: UINT, depth: np.int16, alpha: float, beta: float,
                  TT_pos_: np.ndarray, TT_mask_: np.ndarray, TT_depth_: np.ndarray, TT_flag_: np.ndarray,
                  TT_val_: np.ndarray, TT_move_: np.ndarray) -> (bool, float, np.int8, float, float):
        size_mask = np.int32(TT_pos_.shape[0] - 1)
        idx = tt_hash_local(pos_, mask_, size_mask)
        for _ in range(32):
            if TT_depth_[idx] == -1:
                return False, 0.0, np.int8(-1), alpha, beta
            if TT_pos_[idx] == pos_ and TT_mask_[idx] == mask_:
                d = TT_depth_[idx]; flag = TT_flag_[idx]; val = TT_val_[idx]; mv = TT_move_[idx]
                if d >= depth:
                    # EXACT=0, LOWER=1, UPPER=2
                    if flag == 0:
                        return True, val, mv, alpha, beta
                    if flag == 1 and val > alpha: alpha = val
                    elif flag == 2 and val < beta: beta = val
                    if alpha >= beta: return True, val, mv, alpha, beta
                return False, val, mv, alpha, beta
            idx = (idx + 1) & size_mask
        return False, 0.0, np.int8(-1), alpha, beta

    @njit(cache=True, fastmath=True)
    def tt_store(pos_: UINT, mask_: UINT, depth: np.int16, val: float, alpha0: float, beta: float, best_mv: np.int8,
                 TT_pos_: np.ndarray, TT_mask_: np.ndarray, TT_depth_: np.ndarray, TT_flag_: np.ndarray,
                 TT_val_: np.ndarray, TT_move_: np.ndarray) -> None:
        # EXACT=0, LOWER=1, UPPER=2
        flag = 0
        if val <= alpha0: flag = 2
        elif val >= beta: flag = 1
        size_mask = np.int32(TT_pos_.shape[0] - 1)
        idx = tt_hash_local(pos_, mask_, size_mask)
        victim = -1
        for _ in range(32):
            if TT_depth_[idx] == -1: victim = idx; break
            if TT_pos_[idx] == pos_ and TT_mask_[idx] == mask_: victim = idx; break
            if victim == -1 or TT_depth_[idx] < TT_depth_[victim]: victim = idx
            idx = (idx + 1) & size_mask
        TT_pos_[victim]   = pos_
        TT_mask_[victim]  = mask_
        TT_depth_[victim] = depth
        TT_flag_[victim]  = flag
        TT_val_[victim]   = val
        TT_move_[victim]  = best_mv

    @njit(cache=True)
    def should_stop(node_counter: np.ndarray, max_nodes: np.int64) -> bool:
        node_counter[0] += 1
        return node_counter[0] >= max_nodes

    @njit(cache=True, fastmath=True)
    def negamax(pos_: UINT, mask_: UINT, depth: np.int16, alpha: float, beta: float, ply: np.int16,
                node_counter: np.ndarray, max_nodes: np.int64, MATE_SCORE_i: float,
                COL_MASK_: np.ndarray, WIN_MASKS_: np.ndarray, WIN_B_: np.ndarray, WIN_C_: np.ndarray, WIN_R_: np.ndarray,
                WARR_: np.ndarray, DEF_: float, FN_: float, FF_: float, CENTER_MASK_: UINT,
                immediate_w_: float, fork_w_: float, center_bonus_: float, CENTER_ORDER_: np.ndarray,
                TOP_MASK_: np.ndarray, BOTTOM_MASK_: np.ndarray, stride_i: np.int32, rows_i: np.int32, cols_i: np.int32,
                killers_: np.ndarray, history_: np.ndarray, FULL_MASK_: UINT,
                TT_pos_: np.ndarray, TT_mask_: np.ndarray, TT_depth_: np.ndarray, TT_flag_: np.ndarray,
                TT_val_: np.ndarray, TT_move_: np.ndarray) -> (float, np.int8):

        if should_stop(node_counter, max_nodes):
            return evaluate(pos_, mask_, COL_MASK_, WIN_MASKS_, WIN_B_, WIN_C_, WIN_R_, WARR_,
                            DEF_, FN_, FF_, CENTER_MASK_, immediate_w_, fork_w_, center_bonus_,
                            CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, stride_i, rows_i, cols_i), np.int8(-1)

        alpha0 = alpha
        hit, val_tt, mv_tt, alpha, beta = tt_lookup(pos_, mask_, depth, alpha, beta,
                                                    TT_pos_, TT_mask_, TT_depth_, TT_flag_, TT_val_, TT_move_)
        if hit: return val_tt, mv_tt

        # win-in-1
        for i in range(CENTER_ORDER_.shape[0]):
            c = np.int32(CENTER_ORDER_[i])
            if can_play(mask_, c, TOP_MASK_) and is_winning_move(pos_, mask_, c, BOTTOM_MASK_, COL_MASK_, stride_i):
                return MATE_SCORE_i - float(ply), np.int8(c)

        if mask_ == FULL_MASK_: return 0.0, np.int8(-1)
        if depth == 0:
            return evaluate(pos_, mask_, COL_MASK_, WIN_MASKS_, WIN_B_, WIN_C_, WIN_R_, WARR_,
                            DEF_, FN_, FF_, CENTER_MASK_, immediate_w_, fork_w_, center_bonus_,
                            CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, stride_i, rows_i, cols_i), np.int8(-1)

        best_val = -1e100
        best_col = np.int8(-1)

        moves, m = order_moves(mask_, ply, killers_, history_, CENTER_ORDER_, TOP_MASK_)
        # prune immediate handovers if possible
        safe = np.empty(7, dtype=np.int8); s = 0
        for j in range(m):
            c = np.int32(moves[j])
            if not is_immediate_blunder(pos_, mask_, c, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, stride_i):
                safe[s] = np.int8(c); s += 1
        use_moves, use_len = (safe, s) if s > 0 else (moves, m)

        for j in range(use_len):
            c = np.int32(use_moves[j])
            mv = play_bit(mask_, c, BOTTOM_MASK_, COL_MASK_)
            nm = mask_ | mv
            next_pos = nm ^ (pos_ | mv)

            child_val, _ = negamax(next_pos, nm, np.int16(depth - 1), -beta, -alpha, np.int16(ply + 1),
                                   node_counter, max_nodes, MATE_SCORE_i,
                                   COL_MASK_, WIN_MASKS_, WIN_B_, WIN_C_, WIN_R_, WARR_,
                                   DEF_, FLOATING_NEAR, FLOATING_FAR, CENTER_MASK_,
                                   immediate_w_, fork_w_, center_bonus_,
                                   CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, stride_i, rows_i, cols_i,
                                   killers_, history_, FULL_MASK_,
                                   TT_pos_, TT_mask_, TT_depth_, TT_flag_, TT_val_, TT_move_)
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

        tt_store(pos_, mask_, depth, best_val, alpha0, beta, best_col,
                 TT_pos_, TT_mask_, TT_depth_, TT_flag_, TT_val_, TT_move_)
        return best_val, best_col

    @njit(cache=True, fastmath=True)
    def root_select_fixed(pos_: UINT, mask_: UINT, depth: np.int16,
                          COL_MASK_: np.ndarray, WIN_MASKS_: np.ndarray, WIN_B_: np.ndarray, WIN_C_: np.ndarray, WIN_R_: np.ndarray,
                          WARR_: np.ndarray, DEF_: float, FN_: float, FF_: float, CENTER_MASK_: UINT,
                          immediate_w_: float, fork_w_: float, center_bonus_: float,
                          CENTER_ORDER_: np.ndarray, TOP_MASK_: np.ndarray, BOTTOM_MASK_: np.ndarray,
                          stride_i: np.int32, rows_i: np.int32, cols_i: np.int32, FULL_MASK_: UINT,
                          MATE_SCORE_i: float,
                          killers_: np.ndarray, history_: np.ndarray,
                          TT_pos_: np.ndarray, TT_mask_: np.ndarray, TT_depth_: np.ndarray, TT_flag_: np.ndarray,
                          TT_val_: np.ndarray, TT_move_: np.ndarray) -> np.int32:

        # legal moves
        legal = np.empty(7, dtype=np.int8); L = 0
        for i in range(CENTER_ORDER_.shape[0]):
            c = np.int32(CENTER_ORDER_[i])
            if can_play(mask_, c, TOP_MASK_):
                legal[L] = np.int8(c); L += 1
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
                block_col = c; block_count += 1
                if block_count > 1: break
        if block_count == 1:
            return np.int32(block_col)

        # avoid obvious handovers if possible
        safe = np.empty(7, dtype=np.int8); S = 0
        for i in range(L):
            c = np.int32(legal[i])
            if not is_immediate_blunder(pos_, mask_, c, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, stride_i):
                safe[S] = np.int8(c); S += 1
        if S > 0:
            legal, L = safe, S

        best_move = np.int32(legal[0])
        best_val = evaluate(pos_, mask_, COL_MASK_, WIN_MASKS_, WIN_B_, WIN_C_, WIN_R_, WARR_,
                            DEF_, FN_, FF_, CENTER_MASK_, immediate_w_, fork_w_, center_bonus_,
                            CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, stride_i, rows_i, cols_i)

        # generous budget for exact depth
        node_counter = np.zeros(1, dtype=np.int64)
        max_nodes = np.int64(9_000_000_000_000)  # effectively "no limit" in practice

        for i in range(L):
            c = np.int32(legal[i])
            mv = play_bit(mask_, c, BOTTOM_MASK_, COL_MASK_)
            nm = mask_ | mv
            next_pos = nm ^ (pos_ | mv)

            v, _ = negamax(next_pos, nm, np.int16(depth - 1), -MATE_SCORE_i, MATE_SCORE_i, np.int16(1),
                           node_counter, max_nodes, MATE_SCORE_i,
                           COL_MASK_, WIN_MASKS_, WIN_B_, WIN_C_, WIN_R_, WARR_,
                           DEF_, FLOATING_NEAR, FLOATING_FAR, CENTER_MASK_,
                           immediate_w_, fork_w_, center_bonus_,
                           CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, stride_i, rows_i, cols_i,
                           killers_, history_, FULL_MASK_,
                           TT_pos_, TT_mask_, TT_depth_, TT_flag_, TT_val_, TT_move_)

            val = -v
            if val > best_val:
                best_val = val
                best_move = np.int32(c)

        return best_move

    _NUMBA_C4_CLASS_CACHE = dict(
        ROWS=ROWS, COLS=COLS, STRIDE=STRIDE, UINT=UINT,
        CENTER_ORDER=CENTER_ORDER, CENTER_MASK=CENTER_MASK,
        COL_MASK=COL_MASK, TOP_MASK=TOP_MASK, BOTTOM_MASK=BOTTOM_MASK, FULL_MASK=FULL_MASK,
        WIN_MASKS=WIN_MASKS, WIN_B=WIN_B, WIN_C=WIN_C, WIN_R=WIN_R,
        DEFENSIVE=DEFENSIVE, FLOATING_NEAR=FLOATING_NEAR, FLOATING_FAR=FLOATING_FAR, CENTER_BONUS=CENTER_BONUS,
        # jitted fns
        has_won=has_won,
        count_immediate_wins_bits=count_immediate_wins_bits,
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
    K    = 4
    STRIDE = ROWS + 1
    CENTER_COL = 3

    # Heuristic knobs (kept from your original class)
    immediate_w   = 250.0
    fork_w        = 150.0
    DEFENSIVE     = 1.5
    FLOATING_NEAR = 0.75
    FLOATING_FAR  = 0.50
    CENTER_BONUS  = 6.0

    # Center-first order (stable)
    _CENTER_ORDER = [3, 4, 2, 5, 1, 6, 0]

    # Python-side (for analytics helpers)
    _PRECOMP_DONE = False
    COL_MASK: List[int]    = [0]*COLS
    TOP_MASK: List[int]    = [0]*COLS
    BOTTOM_MASK: List[int] = [0]*COLS
    FULL_MASK: int         = 0
    CENTER_MASK: int       = 0
    WIN_MASKS: List[int]   = []
    WIN_CELLS: List[List[Tuple[int, int, int]]] = []

    def __init__(self, weights=None):
        self.weights: Dict[int, float] = weights if weights else {2: 10.0, 3: 200.0, 4: 1000.0}
        # mate scale (kept compatible with your old code)
        self.SOFT_MATE  = 100.0 * float(self.weights[4])
        self.MATE_SCORE = float(self.weights[4]) * 1000.0

        # Build Python-side precomp (for helpers) once
        if not Connect4Lookahead._PRECOMP_DONE:
            self._build_precomp()

        # Ensure Numba core is ready
        self._N = _ensure_numba_cache()

    # ============================== Public API ===============================

    def get_heuristic(self, board, player) -> float:
        p1, p2, mask = self._parse_board_bitboards(board)
        me = p1 if self._p(player) == 1 else p2
        WARR = np.zeros(self.K + 1, dtype=np.float64)
        for k in (2, 3, 4):
            WARR[k] = float(self.weights.get(k, 0.0))
        return float(self._N["evaluate"](
            np.uint64(me), np.uint64(mask),
            self._N["COL_MASK"], self._N["WIN_MASKS"], self._N["WIN_B"], self._N["WIN_C"], self._N["WIN_R"],
            WARR,
            float(self.DEFENSIVE), float(self.FLOATING_NEAR), float(self.FLOATING_FAR),
            np.uint64(self._N["CENTER_MASK"]),
            float(self.immediate_w), float(self.fork_w), float(self.CENTER_BONUS),
            self._N["CENTER_ORDER"], self._N["TOP_MASK"], self._N["BOTTOM_MASK"], self._N["COL_MASK"],
            np.int32(self.STRIDE), np.int32(self.ROWS), np.int32(self.COLS)
        ))

    def is_terminal(self, board) -> bool:
        p1, p2, mask = self._parse_board_bitboards(board)
        hw = self._N["has_won"]
        return bool(hw(np.uint64(p1), np.int32(self.STRIDE)) or
                    hw(np.uint64(p2), np.int32(self.STRIDE)) or
                    (np.uint64(mask) == self._N["FULL_MASK"]))

    def minimax(self, board, depth, maximizing, player, alpha, beta) -> float:
        """
        Alpha-beta value in `player` POV (same semantics as before).
        """
        p1, p2, mask = self._parse_board_bitboards(board)
        root_pov = self._p(player)
        to_move  = root_pov if maximizing else -root_pov
        pos = p1 if to_move == 1 else p2

        WARR = np.zeros(self.K + 1, dtype=np.float64)
        for k in (2, 3, 4):
            WARR[k] = float(self.weights.get(k, 0.0))

        # scratch
        killers = np.full((64, 2), -1, dtype=np.int8)
        history = np.zeros(self.COLS, dtype=np.int32)
        node_counter = np.zeros(1, dtype=np.int64)
        max_nodes = np.int64(9_000_000_000_000)

        # TT
        TT_SIZE = 1 << 16
        TT_pos  = np.zeros(TT_SIZE, dtype=np.uint64)
        TT_mask = np.zeros(TT_SIZE, dtype=np.uint64)
        TT_depth= np.full(TT_SIZE, -1, dtype=np.int16)
        TT_flag = np.zeros(TT_SIZE, dtype=np.int8)
        TT_val  = np.zeros(TT_SIZE, dtype=np.float64)
        TT_move = np.full(TT_SIZE, -1, dtype=np.int8)

        v, _ = self._N["negamax"](
            np.uint64(pos), np.uint64(mask), np.int16(depth),
            float(alpha), float(beta), np.int16(0),
            node_counter, max_nodes, float(self.MATE_SCORE),
            self._N["COL_MASK"], self._N["WIN_MASKS"], self._N["WIN_B"], self._N["WIN_C"], self._N["WIN_R"],
            WARR,
            float(self.DEFENSIVE), float(self.FLOATING_NEAR), float(self.FLOATING_FAR),
            np.uint64(self._N["CENTER_MASK"]),
            float(self.immediate_w), float(self.fork_w), float(self.CENTER_BONUS),
            self._N["CENTER_ORDER"], self._N["TOP_MASK"], self._N["BOTTOM_MASK"],
            np.int32(self.STRIDE), np.int32(self.ROWS), np.int32(self.COLS),
            killers, history, np.uint64(self._N["FULL_MASK"]),
            TT_pos, TT_mask, TT_depth, TT_flag, TT_val, TT_move
        )
        val = float(v)
        return val if to_move == root_pov else -val

    def n_step_lookahead(self, board, player, depth=3) -> int:
        """
        Root move selection with exact fixed `depth` (no timeouts).
        - Immediate win
        - Single must-block
        - Avoid immediate handovers if possible
        - Negamax(depth-1), center-first ordering
        """
        p1, p2, mask = self._parse_board_bitboards(board)
        me_mark = self._p(player)
        me  = p1 if me_mark == 1 else p2

        # weights -> WARR
        WARR = np.zeros(self.K + 1, dtype=np.float64)
        for k in (2, 3, 4):
            WARR[k] = float(self.weights.get(k, 0.0))

        # per-call scratch
        killers = np.full((64, 2), -1, dtype=np.int8)
        history = np.zeros(self.COLS, dtype=np.int32)

        # TT per move
        TT_SIZE = 1 << 16
        TT_pos  = np.zeros(TT_SIZE, dtype=np.uint64)
        TT_mask = np.zeros(TT_SIZE, dtype=np.uint64)
        TT_depth= np.full(TT_SIZE, -1, dtype=np.int16)
        TT_flag = np.zeros(TT_SIZE, dtype=np.int8)
        TT_val  = np.zeros(TT_SIZE, dtype=np.float64)
        TT_move = np.full(TT_SIZE, -1, dtype=np.int8)

        mv = self._N["root_select_fixed"](
            np.uint64(me), np.uint64(mask), np.int16(depth),
            self._N["COL_MASK"], self._N["WIN_MASKS"], self._N["WIN_B"], self._N["WIN_C"], self._N["WIN_R"],
            WARR,
            float(self.DEFENSIVE), float(self.FLOATING_NEAR), float(self.FLOATING_FAR),
            np.uint64(self._N["CENTER_MASK"]),
            float(self.immediate_w), float(self.fork_w), float(self.CENTER_BONUS),
            self._N["CENTER_ORDER"], self._N["TOP_MASK"], self._N["BOTTOM_MASK"],
            np.int32(self.STRIDE), np.int32(self.ROWS), np.int32(self.COLS),
            np.uint64(self._N["FULL_MASK"]),
            float(self.MATE_SCORE),
            killers, history,
            TT_pos, TT_mask, TT_depth, TT_flag, TT_val, TT_move
        )
        return int(mv)

    def has_four(self, board, player: int) -> bool:
        p1, p2, _ = self._parse_board_bitboards(board)
        bb = p1 if self._p(player) == 1 else p2
        return bool(self._N["has_won"](np.uint64(bb), np.int32(self.STRIDE)))

    # alias
    check_win = has_four

    def count_immediate_wins(self, board, player: int) -> List[int]:
        p1, p2, mask = self._parse_board_bitboards(board)
        me = p1 if self._p(player) == 1 else p2
        # We want the actual columns (center-first)
        wins = []
        for c in self._CENTER_ORDER:
            if (mask & self.TOP_MASK[c]) == 0 and self._is_winning_move_py(me, mask, c):
                wins.append(c)
        return wins

    def compute_fork_signals(self, board_before, board_after, mover: int) -> Dict[str, int]:
        mover = self._p(mover)
        opp   = -mover
        my_after   = len(self.count_immediate_wins(board_after, mover))
        opp_before = len(self.count_immediate_wins(board_before, opp))
        opp_after  = len(self.count_immediate_wins(board_after,  opp))
        return {"my_after": my_after, "opp_before": opp_before, "opp_after": opp_after}

    # ---------------- Analytics helpers preserved ----------------

    def count_pure(self, board, player, n: int) -> int:
        """
        Count 4-length windows containing exactly `n` of `player` and no opponent stones.
        (Python-side; not performance critical for training loop.)
        """
        p1, p2, _ = self._parse_board_bitboards(board)
        me  = p1 if self._p(player) == 1 else p2
        opp = p2 if self._p(player) == 1 else p1
        cnt = 0
        for wmask in self.WIN_MASKS:
            mp = wmask & me
            mo = wmask & opp
            if mo == 0 and mp.bit_count() == n:
                cnt += 1
        return cnt

    def count_pure_block_delta(self, before_board, after_board, player, n: int) -> int:
        """
        Positive number of `player`’s n-in-a-row *pure* windows removed by the move.
        """
        before = self.count_pure(before_board, player, n)
        after  = self.count_pure(after_board,  player, n)
        return max(0, before - after)

    # ============================== Internals ===============================

    @classmethod
    def _build_precomp(cls) -> None:
        # Reset (in case of hot-reload)
        cls.WIN_MASKS.clear()
        cls.WIN_CELLS.clear()

        # Columns
        full = 0
        for c in range(cls.COLS):
            col_bits = 0
            for r in range(cls.ROWS):
                col_bits |= 1 << (c * cls.STRIDE + r)
            cls.COL_MASK[c]    = col_bits
            cls.BOTTOM_MASK[c] = 1 << (c * cls.STRIDE + 0)
            cls.TOP_MASK[c]    = 1 << (c * cls.STRIDE + (cls.ROWS - 1))
            full |= col_bits
        cls.FULL_MASK  = full
        cls.CENTER_MASK = cls.COL_MASK[cls.CENTER_COL]

        def bit_at(c, r): return 1 << (c * cls.STRIDE + r)

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
        # vertical
        m = bb & (bb >> 1)
        if m & (m >> 2): return True
        # horizontal
        m = bb & (bb >> cls.STRIDE)
        if m & (m >> (2 * cls.STRIDE)): return True
        # diag up-right
        m = bb & (bb >> (cls.STRIDE + 1))
        if m & (m >> (2 * (cls.STRIDE + 1))): return True
        # diag up-left
        m = bb & (bb >> (cls.STRIDE - 1))
        if m & (m >> (2 * (cls.STRIDE - 1))): return True
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
                    # treat {2 or -1} as the other side
                    p2 |= b
        return p1, p2, mask
