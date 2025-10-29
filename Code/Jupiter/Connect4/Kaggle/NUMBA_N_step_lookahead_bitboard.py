# NUMBA_N_step_lookahead_bitboard.py

# N_step_lookahead_bitboard.py — FULL NUMBA + ASPIRATION WINDOWS
# - Negamax + alpha-beta entirely inside Numba (nopython)
# - Open-addressed transposition table (uint64 keys for (pos,mask))
# - Killer moves + history heuristic in Numba
# - Immediate-loss pruning (handover avoidance) in Numba
# - Gravity-aware heuristic (FLOATING_NEAR / FLOATING_FAR) in Numba
# - Iterative deepening with aspiration windows at root
# - Soft time guard via objmode perf_counter checks (amortized)

def N_step_lookahead_bitboard(obs, config):
    import numpy as np
    import time
    from numba import njit, objmode

    # ------------------------------ config ---------------------------------
    N_STEPS  = 9
    DEADLINE = time.perf_counter() + 7.6  # Kaggle hard cap ~8s
    DEBUG    = False
    # ------------------------------ constants ------------------------------
    ROWS, COLS, K = 6, 7, 4
    STRIDE = ROWS + 1
    CENTER_ORDER = np.array([3, 4, 2, 5, 1, 6, 0], dtype=np.int8)
    CENTER_COL = 3
    MARK = int(obs.mark)

    # Heuristic weights
    weights_dict = {2: 10.0, 3: 1000.0, 4: 100000.0}
    MATE_SCORE   = float(weights_dict[4] * 1000.0)
    immediate_w  = float(weights_dict[4])
    fork_w       = float(weights_dict[3] * 2)
    DEFENSIVE    = 1.5
    FLOATING_NEAR = 0.75
    FLOATING_FAR  = 0.50
    CENTER_BONUS  = 6.0

    UINT = np.uint64

    # --------------------------- bitboard tables ---------------------------
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

    def bit_at(c, r):
        return UINT(1) << UINT(c * STRIDE + r)

    # --------------------------- windows (NumPy) ---------------------------
    win_bits = []
    win_cols = []
    win_rows = []
    WIN_MASKS = []

    def add_window(cells):
        mask = UINT(0)
        bs, cs, rs = [], [], []
        for rr, cc in cells:
            b = bit_at(cc, rr)
            mask |= b
            bs.append(b); cs.append(cc); rs.append(rr)
        WIN_MASKS.append(mask)
        win_bits.append(bs); win_cols.append(cs); win_rows.append(rs)

    # horizontal
    for r in range(ROWS):
        for c in range(COLS - K + 1):
            add_window([(r, c + i) for i in range(K)])
    # vertical
    for c in range(COLS):
        for r in range(ROWS - K + 1):
            add_window([(r + i, c) for i in range(K)])
    # diag up-right
    for r in range(ROWS - K + 1):
        for c in range(COLS - K + 1):
            add_window([(r + i, c + i) for i in range(K)])
    # diag up-left
    for r in range(ROWS - K + 1):
        for c in range(K - 1, COLS):
            add_window([(r + i, c - i) for i in range(K)])

    WIN_MASKS = np.array(WIN_MASKS, dtype=UINT)
    WIN_B = np.array(win_bits, dtype=UINT)        # (W,4)
    WIN_C = np.array(win_cols, dtype=np.int8)     # (W,4)
    WIN_R = np.array(win_rows, dtype=np.int8)     # (W,4)

    # Pattern weights
    WARR = np.zeros(K + 1, dtype=np.float64)
    for i in range(2, min(K, 4) + 1):
        WARR[i] = float(weights_dict.get(i, 0.0))

    # -------------------------- build pos/mask -----------------------------
    pos1 = UINT(0); pos2 = UINT(0); mask = UINT(0)
    board = obs.board
    for r_top in range(ROWS):
        r = ROWS - 1 - r_top
        base = r_top * COLS
        for c in range(COLS):
            v = board[base + c]
            if v:
                b = bit_at(c, r)
                mask |= b
                if v == 1: pos1 |= b
                else:       pos2 |= b
    pos = pos1 if MARK == 1 else pos2

    # ----------------------------- Numba JIT -------------------------------
    EXACT, LOWER, UPPER = 0, 1, 2
    TT_SIZE = np.int32(1 << 16)  # 65536 entries

    @njit(cache=True, fastmath=True)
    def popcount64(x: UINT) -> np.int32:
        cnt = 0
        while x != UINT(0):
            x &= (x - UINT(1))
            cnt += 1
        return cnt

    @njit(cache=True, fastmath=True)
    def has_won(bb: UINT, STRIDE_i: np.int32) -> bool:
        m = bb & (bb >> UINT(1))
        if (m & (m >> UINT(2))) != UINT(0): return True
        m = bb & (bb >> UINT(STRIDE_i))
        if (m & (m >> UINT(2 * STRIDE_i))) != UINT(0): return True
        m = bb & (bb >> UINT(STRIDE_i + 1))
        if (m & (m >> UINT(2 * (STRIDE_i + 1)))) != UINT(0): return True
        m = bb & (bb >> UINT(STRIDE_i - 1))
        if (m & (m >> UINT(2 * (STRIDE_i - 1)))) != UINT(0): return True
        return False

    @njit(cache=True, fastmath=True)
    def can_play(mask_: UINT, c: np.int32, TOP_MASK_: np.ndarray) -> bool:
        return (mask_ & TOP_MASK_[c]) == UINT(0)

    @njit(cache=True, fastmath=True)
    def play_bit(mask_: UINT, c: np.int32, BOTTOM_MASK_: np.ndarray, COL_MASK_: np.ndarray) -> UINT:
        return (mask_ + BOTTOM_MASK_[c]) & COL_MASK_[c]

    @njit(cache=True, fastmath=True)
    def is_winning_move(pos_: UINT, mask_: UINT, c: np.int32,
                        BOTTOM_MASK_: np.ndarray, COL_MASK_: np.ndarray, STRIDE_i: np.int32) -> bool:
        mv = play_bit(mask_, c, BOTTOM_MASK_, COL_MASK_)
        return has_won(pos_ | mv, STRIDE_i)

    @njit(cache=True, fastmath=True)
    def count_immediate_wins(pos_: UINT, mask_: UINT,
                             CENTER_ORDER_: np.ndarray, TOP_MASK_: np.ndarray,
                             BOTTOM_MASK_: np.ndarray, COL_MASK_: np.ndarray,
                             STRIDE_i: np.int32) -> np.int32:
        cnt = 0
        for i in range(CENTER_ORDER_.shape[0]):
            c = np.int32(CENTER_ORDER_[i])
            if can_play(mask_, c, TOP_MASK_) and is_winning_move(pos_, mask_, c, BOTTOM_MASK_, COL_MASK_, STRIDE_i):
                cnt += 1
        return cnt

    @njit(cache=True, fastmath=True)
    def evaluate(pos_: UINT, mask_: UINT,
                 COL_MASK_: np.ndarray, WIN_MASKS_: np.ndarray, WIN_B_: np.ndarray,
                 WIN_C_: np.ndarray, WIN_R_: np.ndarray, WARR_: np.ndarray,
                 DEF_: float, FN_: float, FF_: float, CENTER_MASK_: UINT,
                 immediate_w_: float, fork_w_: float, center_bonus_: float,
                 CENTER_ORDER_: np.ndarray, TOP_MASK_: np.ndarray,
                 BOTTOM_MASK_: np.ndarray, COL_MASK2_: np.ndarray, STRIDE_i: np.int32,
                 ROWS_i: np.int32, COLS_i: np.int32) -> float:
        opp = mask_ ^ pos_
        # heights
        H = np.empty(COLS_i, dtype=np.int16)
        for c in range(COLS_i):
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
                wv = WARR_[p] if p <= 4 else 0.0
                score += mul * wv
            elif p == 0:
                wv = WARR_[o] if o <= 4 else 0.0
                score -= DEF_ * mul * wv

        my_imm  = count_immediate_wins(pos_,  mask_, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK2_, STRIDE_i)
        opp_imm = count_immediate_wins(opp,   mask_, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK2_, STRIDE_i)
        score += immediate_w_ * (my_imm - DEF_ * opp_imm)
        if my_imm >= 2:
            score += fork_w_ * (my_imm - 1)
        if opp_imm >= 2:
            score -= DEF_ * (fork_w_ * (opp_imm - 1))

        score += center_bonus_ * (popcount64(pos_ & CENTER_MASK_) - popcount64(opp & CENTER_MASK_))
        return score

    @njit(cache=True, fastmath=True)
    def is_immediate_blunder(pos_: UINT, mask_: UINT, c: np.int32,
                             CENTER_ORDER_: np.ndarray, TOP_MASK_: np.ndarray,
                             BOTTOM_MASK_: np.ndarray, COL_MASK_: np.ndarray, STRIDE_i: np.int32) -> bool:
        mv = play_bit(mask_, c, BOTTOM_MASK_, COL_MASK_)
        nm = mask_ | mv
        opp_pos = nm ^ (pos_ | mv)
        for i in range(CENTER_ORDER_.shape[0]):
            cc = np.int32(CENTER_ORDER_[i])
            if can_play(nm, cc, TOP_MASK_) and is_winning_move(opp_pos, nm, cc, BOTTOM_MASK_, COL_MASK_, STRIDE_i):
                return True
        return False

    # --------- move ordering (killers/history) with insertion sort ----------
    @njit(cache=True, fastmath=True)
    def order_moves(mask_: UINT, ply: np.int32,
                    killers_: np.ndarray, history_: np.ndarray,
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

        # insertion sort (descending)
        for i in range(1, m):
            key_m = moves[i]; key_s = scores[i]
            j = i - 1
            while j >= 0 and scores[j] < key_s:
                moves[j + 1] = moves[j]
                scores[j + 1] = scores[j]
                j -= 1
            moves[j + 1] = key_m
            scores[j + 1] = key_s
        return moves, m

    # -------------------------- TT (open addressing) ------------------------
    TT_pos  = np.zeros(TT_SIZE, dtype=UINT)
    TT_mask = np.zeros(TT_SIZE, dtype=UINT)
    TT_depth= np.full(TT_SIZE, -1, dtype=np.int16)
    TT_flag = np.zeros(TT_SIZE, dtype=np.int8)
    TT_val  = np.zeros(TT_SIZE, dtype=np.float64)
    TT_move = np.full(TT_SIZE, -1, dtype=np.int8)

    @njit(cache=True, fastmath=True)
    def tt_hash(pos_: UINT, mask_: UINT) -> np.int32:
        h = np.uint64(pos_) ^ (np.uint64(mask_) * np.uint64(0x9E3779B97F4A7C15))
        h ^= (h >> np.uint64(7))
        return np.int32(h & np.uint64(TT_SIZE - 1))

    @njit(cache=True, fastmath=True)
    def tt_lookup(pos_: UINT, mask_: UINT, depth: np.int16,
                  alpha: float, beta: float) -> (bool, float, np.int8, float, float):
        idx = tt_hash(pos_, mask_)
        for _ in range(32):
            if TT_depth[idx] == -1:
                return False, 0.0, np.int8(-1), alpha, beta
            if TT_pos[idx] == pos_ and TT_mask[idx] == mask_:
                d = TT_depth[idx]; flag = TT_flag[idx]; val = TT_val[idx]; mv = TT_move[idx]
                if d >= depth:
                    if flag == EXACT:
                        return True, val, mv, alpha, beta
                    if flag == LOWER and val > alpha:
                        alpha = val
                    elif flag == UPPER and val < beta:
                        beta = val
                    if alpha >= beta:
                        return True, val, mv, alpha, beta
                return False, val, mv, alpha, beta
            idx = (idx + 1) & (TT_SIZE - 1)
        return False, 0.0, np.int8(-1), alpha, beta

    @njit(cache=True, fastmath=True)
    def tt_store(pos_: UINT, mask_: UINT, depth: np.int16, val: float,
                 alpha0: float, beta: float, best_mv: np.int8) -> None:
        flag = EXACT
        if val <= alpha0:
            flag = UPPER
        elif val >= beta:
            flag = LOWER
        idx = tt_hash(pos_, mask_)
        victim = -1
        for _ in range(32):
            if TT_depth[idx] == -1:
                victim = idx; break
            if TT_pos[idx] == pos_ and TT_mask[idx] == mask_:
                victim = idx; break
            if victim == -1 or TT_depth[idx] < TT_depth[victim]:
                victim = idx
            idx = (idx + 1) & (TT_SIZE - 1)
        TT_pos[victim]   = pos_
        TT_mask[victim]  = mask_
        TT_depth[victim] = depth
        TT_flag[victim]  = flag
        TT_val[victim]   = val
        TT_move[victim]  = best_mv

    # --------------------------- time guard helper --------------------------
    TIME_CHECK_MASK = np.int64(0x7FF)  # every ~2048 nodes

    @njit(cache=True)
    def should_stop(node_counter: np.ndarray, deadline: float) -> bool:
        node_counter[0] += 1
        if (node_counter[0] & TIME_CHECK_MASK) != 0:
            return False
        with objmode(now='f8'):
            now = time.perf_counter()
        return now > deadline

    # ------------------------------ negamax --------------------------------
    @njit(cache=True, fastmath=True)
    def negamax(pos_: UINT, mask_: UINT, depth: np.int16, alpha: float, beta: float, ply: np.int16,
                node_counter: np.ndarray, MATE_SCORE_i: float,
                COL_MASK_: np.ndarray, WIN_MASKS_: np.ndarray, WIN_B_: np.ndarray,
                WIN_C_: np.ndarray, WIN_R_: np.ndarray, WARR_: np.ndarray,
                DEF_: float, FN_: float, FF_: float, CENTER_MASK_: UINT,
                immediate_w_: float, fork_w_: float, center_bonus_: float,
                CENTER_ORDER_: np.ndarray, TOP_MASK_: np.ndarray,
                BOTTOM_MASK_: np.ndarray, STRIDE_i: np.int32, ROWS_i: np.int32, COLS_i: np.int32,
                killers_: np.ndarray, history_: np.ndarray, FULL_MASK_: UINT,
                deadline: float) -> (float, np.int8):

        if should_stop(node_counter, deadline):
            return evaluate(pos_, mask_, COL_MASK_, WIN_MASKS_, WIN_B_, WIN_C_, WIN_R_, WARR_,
                            DEF_, FN_, FF_, CENTER_MASK_, immediate_w_, fork_w_, center_bonus_,
                            CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, STRIDE_i, ROWS_i, COLS_i), np.int8(-1)

        # TT probe
        alpha0 = alpha
        hit, val_tt, mv_tt, alpha, beta = tt_lookup(pos_, mask_, depth, alpha, beta)
        if hit:
            return val_tt, mv_tt

        # tactical win-in-1
        for i in range(CENTER_ORDER_.shape[0]):
            c = np.int32(CENTER_ORDER_[i])
            if can_play(mask_, c, TOP_MASK_) and is_winning_move(pos_, mask_, c, BOTTOM_MASK_, COL_MASK_, STRIDE_i):
                return MATE_SCORE_i - float(ply), np.int8(c)

        # draw?
        if mask_ == FULL_MASK_:
            return 0.0, np.int8(-1)

        if depth == 0:
            return evaluate(pos_, mask_, COL_MASK_, WIN_MASKS_, WIN_B_, WIN_C_, WIN_R_, WARR_,
                            DEF_, FN_, FF_, CENTER_MASK_, immediate_w_, fork_w_, center_bonus_,
                            CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, STRIDE_i, ROWS_i, COLS_i), np.int8(-1)

        best_val = -1e100
        best_col = np.int8(-1)

        moves, m = order_moves(mask_, ply, killers_, history_, CENTER_ORDER_, TOP_MASK_)
        # immediate blunder pruning (unless all moves are blunders)
        safe = np.empty(7, dtype=np.int8)
        s = 0
        for j in range(m):
            c = np.int32(moves[j])
            if not is_immediate_blunder(pos_, mask_, c, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, STRIDE_i):
                safe[s] = np.int8(c); s += 1
        use_moves = moves; use_len = m
        if s > 0:
            use_moves = safe; use_len = s

        for j in range(use_len):
            c = np.int32(use_moves[j])
            mv = play_bit(mask_, c, BOTTOM_MASK_, COL_MASK_)
            nm = mask_ | mv
            next_pos = nm ^ (pos_ | mv)

            child_val, _ = negamax(next_pos, nm, depth - 1, -beta, -alpha, ply + 1,
                                   node_counter, MATE_SCORE_i,
                                   COL_MASK_, WIN_MASKS_, WIN_B_, WIN_C_, WIN_R_, WARR_,
                                   DEF_, FN_, FF_, CENTER_MASK_,
                                   immediate_w_, fork_w_, center_bonus_,
                                   CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, STRIDE_i, ROWS_i, COLS_i,
                                   killers_, history_, FULL_MASK_, deadline)
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

        tt_store(pos_, mask_, depth, best_val, alpha0, beta, best_col)
        return best_val, best_col

    # --------------------------- root selection -----------------------------
    @njit(cache=True, fastmath=True)
    def root_select(pos_: UINT, mask_: UINT, N_STEPS_i: np.int16, deadline: float,
                    COL_MASK_: np.ndarray, WIN_MASKS_: np.ndarray, WIN_B_: np.ndarray,
                    WIN_C_: np.ndarray, WIN_R_: np.ndarray, WARR_: np.ndarray,
                    DEF_: float, FN_: float, FF_: float, CENTER_MASK_: UINT,
                    immediate_w_: float, fork_w_: float, center_bonus_: float,
                    CENTER_ORDER_: np.ndarray, TOP_MASK_: np.ndarray,
                    BOTTOM_MASK_: np.ndarray, STRIDE_i: np.int32, ROWS_i: np.int32, COLS_i: np.int32,
                    FULL_MASK_: UINT, MATE_SCORE_i: float) -> np.int32:

        killers_ = np.full((64, 2), -1, dtype=np.int8)  # safe margin on ply
        history_ = np.zeros(COLS_i, dtype=np.int32)
        node_counter = np.zeros(1, dtype=np.int64)

        # legal moves
        legal = np.empty(7, dtype=np.int8); L = 0
        for i in range(CENTER_ORDER_.shape[0]):
            c = np.int32(CENTER_ORDER_[i])
            if can_play(mask_, c, TOP_MASK_):
                legal[L] = np.int8(c); L += 1
        if L == 0:
            return np.int32(0)

        # Immediate win for us?
        for i in range(L):
            c = np.int32(legal[i])
            if is_winning_move(pos_, mask_, c, BOTTOM_MASK_, COL_MASK_, STRIDE_i):
                return c

        # Single must-block
        opp_pos_now = mask_ ^ pos_
        block_col = -1; block_count = 0
        for i in range(L):
            c = np.int32(legal[i])
            if is_winning_move(opp_pos_now, mask_, c, BOTTOM_MASK_, COL_MASK_, STRIDE_i):
                block_col = c; block_count += 1
                if block_count > 1: break
        if block_count == 1:
            return block_col

        # Avoid obvious handovers if possible
        safe = np.empty(7, dtype=np.int8); S = 0
        for i in range(L):
            c = np.int32(legal[i])
            if not is_immediate_blunder(pos_, mask_, c, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, STRIDE_i):
                safe[S] = np.int8(c); S += 1
        if S > 0:
            legal = safe; L = S

        best_move = np.int32(legal[0])
        # Initial guess for aspiration windows: static eval
        best_val = evaluate(pos_, mask_, COL_MASK_, WIN_MASKS_, WIN_B_, WIN_C_, WIN_R_, WARR_,
                            DEF_, FN_, FF_, CENTER_MASK_, immediate_w_, fork_w_, center_bonus_,
                            CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK_, STRIDE_i, ROWS_i, COLS_i)

        # window size tuned to your scale (immediate_w == 1e5); widen if needed
        ASP_INIT = float(10.0 * immediate_w_)   # e.g., 1,000,000.0
        for d in range(1, int(N_STEPS_i) + 1):
            # time guard between iterations
            with objmode(now='f8'):
                now = time.perf_counter()
            if now > deadline:
                break

            alpha = max(-MATE_SCORE_i, best_val - ASP_INIT)
            beta  = min( MATE_SCORE_i, best_val + ASP_INIT)

            v, mv = negamax(pos_, mask_, np.int16(d), alpha, beta, np.int16(0),
                            node_counter, MATE_SCORE_i,
                            COL_MASK_, WIN_MASKS_, WIN_B_, WIN_C_, WIN_R_, WARR_,
                            DEF_, FLOATING_NEAR, FLOATING_FAR, CENTER_MASK_,
                            immediate_w_, fork_w_, center_bonus_,
                            CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, STRIDE_i, ROWS_i, COLS_i,
                            killers_, history_, FULL_MASK_, deadline)

            # Re-search on fail-low/high
            if mv < 0 or v <= alpha:
                v, mv = negamax(pos_, mask_, np.int16(d), -MATE_SCORE_i, beta, np.int16(0),
                                node_counter, MATE_SCORE_i,
                                COL_MASK_, WIN_MASKS_, WIN_B_, WIN_C_, WIN_R_, WARR_,
                                DEF_, FLOATING_NEAR, FLOATING_FAR, CENTER_MASK_,
                                immediate_w_, fork_w_, center_bonus_,
                                CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, STRIDE_i, ROWS_i, COLS_i,
                                killers_, history_, FULL_MASK_, deadline)
            elif v >= beta:
                v, mv = negamax(pos_, mask_, np.int16(d), alpha,  MATE_SCORE_i, np.int16(0),
                                node_counter, MATE_SCORE_i,
                                COL_MASK_, WIN_MASKS_, WIN_B_, WIN_C_, WIN_R_, WARR_,
                                DEF_, FLOATING_NEAR, FLOATING_FAR, CENTER_MASK_,
                                immediate_w_, fork_w_, center_bonus_,
                                CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, STRIDE_i, ROWS_i, COLS_i,
                                killers_, history_, FULL_MASK_, deadline)

            if mv >= 0:
                best_move = np.int32(mv)
                best_val  = v  # refine window center

        return best_move

    # ----------------------------- call root --------------------------------
    best = root_select(
        np.uint64(pos), np.uint64(mask), np.int16(N_STEPS), float(DEADLINE),
        COL_MASK, WIN_MASKS, WIN_B, WIN_C, WIN_R, WARR,
        DEFENSIVE, FLOATING_NEAR, FLOATING_FAR, np.uint64(CENTER_MASK),
        immediate_w, fork_w, CENTER_BONUS,
        CENTER_ORDER, TOP_MASK, BOTTOM_MASK, np.int32(STRIDE), np.int32(ROWS), np.int32(COLS),
        np.uint64(FULL_MASK), float(MATE_SCORE)
    )

    if DEBUG: print("[bitboard-numba-asp] pick:", int(best))

    return int(best)
