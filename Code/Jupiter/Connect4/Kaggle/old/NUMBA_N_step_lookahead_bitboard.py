# NUMBA_N_step_lookahead_bitboard.py

def NUMBA_N_step_lookahead_bitboard(obs, config):
    import numpy as np, time
    from numba import njit

    #MAX_DEPTH = 12
    MAX_DEPTH = 9   #best so far, better than 11?? or just luck
    #MAX_DEPTH = 11 
    #MAX_DEPTH = 13 # let's see, curiousity never hurts, except cats

    # cache compiled core & precomputed tables across calls
    _CACHE = globals().setdefault("_NUMBA_C4_CACHE", {})

    if not _CACHE:
        # ---------- constants ----------
        ROWS, COLS, K = 6, 7, 4
        STRIDE = ROWS + 1
        UINT = np.uint64

        CENTER_COL = 3
        CENTER_ORDER = np.array([3, 4, 2, 5, 1, 6, 0], dtype=np.int8)

        # Heuristic weights (your scale)
        WARR = np.zeros(K + 1, dtype=np.float64)
        WARR[2] = 10.0
        WARR[3] = 1000.0
        WARR[4] = 100000.0
        MATE_SCORE = float(WARR[4] * 1000.0)
        IMMEDIATE_W = float(WARR[4])
        FORK_W = float(WARR[3] * 2)
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
        # @njit(cache=True, fastmath=True)
        # def popcount64(x: UINT) -> np.int32:
        #     cnt = 0
        #     while x != UINT(0):
        #         x &= (x - UINT(1))
        #         cnt += 1
        #     return cnt
        
        @njit(cache=True, fastmath=True)
        def popcount64(x: UINT) -> np.int32:
            x = x - ((x >> 1) & 0x5555555555555555)
            x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
            x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
            return np.int32((x * 0x0101010101010101) >> 56)

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
        def count_immediate_wins(pos_: UINT, mask_: UINT, CENTER_ORDER_: np.ndarray,
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

            my_imm  = count_immediate_wins(pos_,  mask_, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK2_, stride_i)
            opp_imm = count_immediate_wins(opp,   mask_, CENTER_ORDER_, TOP_MASK_, BOTTOM_MASK_, COL_MASK2_, stride_i)
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

                child_val, _ = negamax(next_pos, nm, depth - 1, -beta, -alpha, ply + 1,
                                       node_counter, max_nodes, MATE_SCORE_i,
                                       COL_MASK_, WIN_MASKS_, WIN_B_, WIN_C_, WIN_R_, WARR_,
                                       DEF_, FLOATING_NEAR, FLOATING_FAR, CENTER_MASK_,
                                       IMMEDIATE_W, FORK_W, CENTER_BONUS,
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

        # stash in cache
        _CACHE.update(dict(
            ROWS=ROWS, COLS=COLS, STRIDE=STRIDE, UINT=UINT,
            CENTER_ORDER=CENTER_ORDER, CENTER_MASK=CENTER_MASK,
            COL_MASK=COL_MASK, TOP_MASK=TOP_MASK, BOTTOM_MASK=BOTTOM_MASK, FULL_MASK=FULL_MASK,
            WIN_MASKS=WIN_MASKS, WIN_B=WIN_B, WIN_C=WIN_C, WIN_R=WIN_R, WARR=WARR,
            MATE_SCORE=MATE_SCORE, IMMEDIATE_W=IMMEDIATE_W, FORK_W=FORK_W,
            DEFENSIVE=DEFENSIVE, FLOATING_NEAR=FLOATING_NEAR, FLOATING_FAR=FLOATING_FAR,
            CENTER_BONUS=CENTER_BONUS,
            negamax=negamax, evaluate=evaluate, is_winning_move=is_winning_move,
            is_immediate_blunder=is_immediate_blunder, order_moves=order_moves
        ))

    # ---------- unpack cache ----------
    ROWS = _CACHE["ROWS"]; COLS = _CACHE["COLS"]; STRIDE = _CACHE["STRIDE"]
    UINT = _CACHE["UINT"]
    CENTER_ORDER = _CACHE["CENTER_ORDER"]; CENTER_MASK = _CACHE["CENTER_MASK"]
    COL_MASK = _CACHE["COL_MASK"]; TOP_MASK = _CACHE["TOP_MASK"]; BOTTOM_MASK = _CACHE["BOTTOM_MASK"]; FULL_MASK = _CACHE["FULL_MASK"]
    WIN_MASKS = _CACHE["WIN_MASKS"]; WIN_B = _CACHE["WIN_B"]; WIN_C = _CACHE["WIN_C"]; WIN_R = _CACHE["WIN_R"]; WARR = _CACHE["WARR"]
    MATE_SCORE = _CACHE["MATE_SCORE"]; IMMEDIATE_W = _CACHE["IMMEDIATE_W"]; FORK_W = _CACHE["FORK_W"]
    DEFENSIVE = _CACHE["DEFENSIVE"]; FLOATING_NEAR = _CACHE["FLOATING_NEAR"]; FLOATING_FAR = _CACHE["FLOATING_FAR"]; CENTER_BONUS = _CACHE["CENTER_BONUS"]
    negamax = _CACHE["negamax"]; evaluate = _CACHE["evaluate"]
    is_winning_move = _CACHE["is_winning_move"]; is_immediate_blunder = _CACHE["is_immediate_blunder"]

    # ---------- build pos/mask from Kaggle obs ----------
    MARK = int(obs.mark)
    pos1 = np.uint64(0); pos2 = np.uint64(0); mask = np.uint64(0)
    board = obs.board  # len 42, top-first
    for r_top in range(ROWS):
        r = ROWS - 1 - r_top
        base = r_top * COLS
        for c in range(COLS):
            v = board[base + c]
            if v:
                b = np.uint64(1) << np.uint64(c * STRIDE + r)
                mask |= b
                if v == 1: pos1 |= b
                else:       pos2 |= b
    pos = pos1 if MARK == 1 else pos2

    # ---------- root logic: iterative deepening + aspiration ----------
    DEADLINE = time.perf_counter() + 7.6


    # per-move scratch
    killers = np.full((64, 2), -1, dtype=np.int8)  # 42 plies max, 64 for headroom
    history = np.zeros(COLS, dtype=np.int32)
    node_counter = np.zeros(1, dtype=np.int64)

    # TT per move
    TT_SIZE = 1 << 16
    TT_pos  = np.zeros(TT_SIZE, dtype=np.uint64)
    TT_mask = np.zeros(TT_SIZE, dtype=np.uint64)
    TT_depth= np.full(TT_SIZE, -1, dtype=np.int16)
    TT_flag = np.zeros(TT_SIZE, dtype=np.int8)
    TT_val  = np.zeros(TT_SIZE, dtype=np.float64)
    TT_move = np.full(TT_SIZE, -1, dtype=np.int8)

    # legal moves
    legal = [c for c in CENTER_ORDER.tolist() if (mask & TOP_MASK[int(c)]) == 0]
    if not legal:
        return 0

    # immediate win
    for c in legal:
        ci = int(c)
        if is_winning_move(pos, mask, ci, BOTTOM_MASK, COL_MASK, STRIDE):
            return ci

    # single must-block
    opp_pos = mask ^ pos
    block_col, block_count = -1, 0
    for c in legal:
        ci = int(c)
        if is_winning_move(opp_pos, mask, ci, BOTTOM_MASK, COL_MASK, STRIDE):
            block_col = ci; block_count += 1
            if block_count > 1: break
    if block_count == 1:
        return block_col

    # avoid obvious handovers if possible
    safe = [int(c) for c in legal if not is_immediate_blunder(pos, mask, int(c), CENTER_ORDER, TOP_MASK, BOTTOM_MASK, COL_MASK, STRIDE)]
    if safe:
        legal = safe

    best_move = int(legal[0])
    # initial eval as aspiration center
    best_val = float(evaluate(pos, mask, COL_MASK, WIN_MASKS, WIN_B, WIN_C, WIN_R, WARR,
                              float(DEFENSIVE), float(FLOATING_NEAR), float(FLOATING_FAR), np.uint64(CENTER_MASK),
                              float(IMMEDIATE_W), float(FORK_W), float(CENTER_BONUS),
                              CENTER_ORDER, TOP_MASK, BOTTOM_MASK, COL_MASK, np.int32(STRIDE), np.int32(ROWS), np.int32(COLS)))

    ASP_INIT = 10.0 * IMMEDIATE_W  # e.g., 1e6

    def depth_to_budget(d: int) -> np.int64:
        base = 250_000
        growth = 1.8 ** d
        return np.int64(min(8_000_000, int(base * growth)))

    for d in range(1, MAX_DEPTH + 1):
        if time.perf_counter() > DEADLINE:
            break

        alpha = max(-MATE_SCORE, best_val - ASP_INIT)
        beta  = min( MATE_SCORE, best_val + ASP_INIT)

        node_counter[0] = 0
        max_nodes = depth_to_budget(d)

        v, mv = negamax(np.uint64(pos), np.uint64(mask), np.int16(d), float(alpha), float(beta), np.int16(0),
                        node_counter, max_nodes, float(MATE_SCORE),
                        COL_MASK, WIN_MASKS, WIN_B, WIN_C, WIN_R, WARR,
                        float(DEFENSIVE), float(FLOATING_NEAR), float(FLOATING_FAR), np.uint64(CENTER_MASK),
                        float(IMMEDIATE_W), float(FORK_W), float(CENTER_BONUS),
                        CENTER_ORDER, TOP_MASK, BOTTOM_MASK, np.int32(STRIDE), np.int32(ROWS), np.int32(COLS),
                        killers, history, np.uint64(FULL_MASK),
                        TT_pos, TT_mask, TT_depth, TT_flag, TT_val, TT_move)

        # re-search on fail-low/high
        if mv < 0 or v <= alpha:
            node_counter[0] = 0
            v, mv = negamax(np.uint64(pos), np.uint64(mask), np.int16(d), -MATE_SCORE, float(beta), np.int16(0),
                            node_counter, max_nodes, float(MATE_SCORE),
                            COL_MASK, WIN_MASKS, WIN_B, WIN_C, WIN_R, WARR,
                            float(DEFENSIVE), float(FLOATING_NEAR), float(FLOATING_FAR), np.uint64(CENTER_MASK),
                            float(IMMEDIATE_W), float(FORK_W), float(CENTER_BONUS),
                            CENTER_ORDER, TOP_MASK, BOTTOM_MASK, np.int32(STRIDE), np.int32(ROWS), np.int32(COLS),
                            killers, history, np.uint64(FULL_MASK),
                            TT_pos, TT_mask, TT_depth, TT_flag, TT_val, TT_move)
        elif v >= beta:
            node_counter[0] = 0
            v, mv = negamax(np.uint64(pos), np.uint64(mask), np.int16(d), float(alpha), MATE_SCORE, np.int16(0),
                            node_counter, max_nodes, float(MATE_SCORE),
                            COL_MASK, WIN_MASKS, WIN_B, WIN_C, WIN_R, WARR,
                            float(DEFENSIVE), float(FLOATING_NEAR), float(FLOATING_FAR), np.uint64(CENTER_MASK),
                            float(IMMEDIATE_W), float(FORK_W), float(CENTER_BONUS),
                            CENTER_ORDER, TOP_MASK, BOTTOM_MASK, np.int32(STRIDE), np.int32(ROWS), np.int32(COLS),
                            killers, history, np.uint64(FULL_MASK),
                            TT_pos, TT_mask, TT_depth, TT_flag, TT_val, TT_move)

        if mv >= 0:
            best_move = int(mv)
            best_val  = float(v)

    return int(best_move)
