# NUMBA_N_step_lookahead_bitboard_fixed.py

def NUMBA_N_step_lookahead_bitboard_optimized(obs, config):
    import numpy as np, time
    from numba import njit

    #MAX_DEPTH = 9  #
    MAX_DEPTH = 11  #
    
    # cache compiled core & precomputed tables across calls
    _CACHE = globals().setdefault("_NUMBA_C4_CACHE_FIXED", {})

    if not _CACHE:
        # ---------- constants ----------
        ROWS, COLS, K = 6, 7, 4
        STRIDE = ROWS + 1
        UINT = np.uint64

        CENTER_COL = 3
        CENTER_ORDER = np.array([3, 4, 2, 5, 1, 6, 0], dtype=np.int8)

        # Conservative heuristic weights (proven values)
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

        # Bitboard columns & masks (same as original)
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

        def _bit_at(c, r): return UINT(1) << UINT(c * STRIDE + r)

        # Windows (69 of length 4) - same as original
        _win_bits, _win_cols, _win_rows, _WIN_MASKS = [], [], [], []
        def _add_window(cells):
            mask = UINT(0); bs, cs, rs = [], [], []
            for rr, cc in cells:
                b = _bit_at(cc, rr); mask |= b
                bs.append(b); cs.append(cc); rs.append(rr)
            _WIN_MASKS.append(mask); _win_bits.append(bs); _win_cols.append(cs); _win_rows.append(rs)

        for r in range(ROWS):
            for c in range(COLS - K + 1): _add_window([(r, c+i) for i in range(K)])
        for c in range(COLS):
            for r in range(ROWS - K + 1): _add_window([(r+i, c) for i in range(K)])
        for r in range(ROWS - K + 1):
            for c in range(COLS - K + 1): _add_window([(r+i, c+i) for i in range(K)])
        for r in range(ROWS - K + 1):
            for c in range(K - 1, COLS):  _add_window([(r+i, c-i) for i in range(K)])

        WIN_MASKS = np.array(_WIN_MASKS, dtype=UINT)
        WIN_B = np.array(_win_bits, dtype=UINT)
        WIN_C = np.array(_win_cols, dtype=np.int8)
        WIN_R = np.array(_win_rows, dtype=np.int8)

        # ---------- Conservative Optimizations ----------
        
        # Keep the optimized popcount - this is purely performance
        @njit(cache=True, fastmath=True)
        def popcount64(x: UINT) -> np.int32:
            x = x - ((x >> 1) & 0x5555555555555555)
            x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
            x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
            return np.int32((x * 0x0101010101010101) >> 56)

        # Keep all the core game logic identical to original
        @njit(cache=True, fastmath=True)
        def has_won(bb: UINT, stride_i: np.int32) -> bool:
            m = bb & (bb >> UINT(1))
            if (m & (m >> UINT(2))) != UINT(0): return True
            m = bb & (bb >> UINT(stride_i))
            if (m & (m >> UINT(2 * stride_i))) != UINT(0): return True
            m = bb & (bb >> UINT(stride_i + 1))
            if (m & (m >> UINT(2 * (stride_i + 1)))) != UINT(0): return True
            m = bb & (bb >> UINT(stride_i - 1))
            if (m & (m >> UINT(2 * (stride_i - 1)))) != UINT(0): return True
            return False

        @njit(cache=True, fastmath=True)
        def can_play(mask: UINT, c: np.int32, TOP_MASK: np.ndarray) -> bool:
            return (mask & TOP_MASK[c]) == UINT(0)

        @njit(cache=True, fastmath=True)
        def play_bit(mask: UINT, c: np.int32, BOTTOM_MASK: np.ndarray, COL_MASK: np.ndarray) -> UINT:
            return (mask + BOTTOM_MASK[c]) & COL_MASK[c]

        @njit(cache=True, fastmath=True)
        def is_winning_move(pos: UINT, mask: UINT, c: np.int32,
                            BOTTOM_MASK: np.ndarray, COL_MASK: np.ndarray, stride_i: np.int32) -> bool:
            mv = play_bit(mask, c, BOTTOM_MASK, COL_MASK)
            return has_won(pos | mv, stride_i)

        @njit(cache=True, fastmath=True)
        def count_immediate_wins(pos: UINT, mask: UINT, CENTER_ORDER: np.ndarray,
                                 TOP_MASK: np.ndarray, BOTTOM_MASK: np.ndarray,
                                 COL_MASK: np.ndarray, stride_i: np.int32) -> np.int32:
            cnt = 0
            for i in range(CENTER_ORDER.shape[0]):
                c = np.int32(CENTER_ORDER[i])
                if can_play(mask, c, TOP_MASK) and is_winning_move(pos, mask, c, BOTTOM_MASK, COL_MASK, stride_i):
                    cnt += 1
            return cnt

        # CORRECTED evaluation - keep COL_MASK2 parameter!
        @njit(cache=True, fastmath=True)
        def evaluate(pos: UINT, mask: UINT, COL_MASK: np.ndarray, WIN_MASKS: np.ndarray, WIN_B: np.ndarray,
                     WIN_C: np.ndarray, WIN_R: np.ndarray, WARR: np.ndarray, DEF: float, FN: float, FF: float,
                     CENTER_MASK: UINT, immediate_w: float, fork_w: float, center_bonus: float,
                     CENTER_ORDER: np.ndarray, TOP_MASK: np.ndarray, BOTTOM_MASK: np.ndarray,
                     COL_MASK2: np.ndarray, stride_i: np.int32, rows_i: np.int32, cols_i: np.int32) -> float:
            opp = mask ^ pos
            H = np.empty(cols_i, dtype=np.int16)
            for c in range(cols_i):
                H[c] = popcount64(mask & COL_MASK[c])

            score = 0.0
            W = WIN_MASKS.shape[0]
            for idx in range(W):
                wmask = WIN_MASKS[idx]
                mo = wmask & opp
                mp = wmask & pos
                if mo != UINT(0) and mp != UINT(0):
                    continue
                p = popcount64(mp)
                o = popcount64(mo)
                if (p + o) < 2:
                    continue

                need = 0
                if p == 0 or o == 0:
                    for k2 in range(4):
                        b = WIN_B[idx, k2]
                        if (mask & b) == UINT(0):
                            cc = np.int32(WIN_C[idx, k2])
                            rr = np.int32(WIN_R[idx, k2])
                            dh = rr - np.int32(H[cc])
                            if dh > 0:
                                need += dh

                mul = 1.0 if need == 0 else (FN if need == 1 else FF)
                if o == 0:
                    score += mul * (WARR[p] if p <= 4 else 0.0)
                elif p == 0:
                    score -= DEF * mul * (WARR[o] if o <= 4 else 0.0)

            # Keep only the proven evaluation components
            my_imm  = count_immediate_wins(pos,  mask, CENTER_ORDER, TOP_MASK, BOTTOM_MASK, COL_MASK2, stride_i)
            opp_imm = count_immediate_wins(opp,   mask, CENTER_ORDER, TOP_MASK, BOTTOM_MASK, COL_MASK2, stride_i)
            score += immediate_w * (my_imm - DEF * opp_imm)
            if my_imm >= 2:  score += fork_w * (my_imm - 1)
            if opp_imm >= 2: score -= DEF * (fork_w * (opp_imm - 1))
            score += center_bonus * (popcount64(pos & CENTER_MASK) - popcount64(opp & CENTER_MASK))
            return score

        @njit(cache=True, fastmath=True)
        def is_immediate_blunder(pos: UINT, mask: UINT, c: np.int32, CENTER_ORDER: np.ndarray, TOP_MASK: np.ndarray,
                                 BOTTOM_MASK: np.ndarray, COL_MASK: np.ndarray, stride_i: np.int32) -> bool:
            mv = play_bit(mask, c, BOTTOM_MASK, COL_MASK)
            nm = mask | mv
            opp_pos = nm ^ (pos | mv)
            for i in range(CENTER_ORDER.shape[0]):
                cc = np.int32(CENTER_ORDER[i])
                if can_play(nm, cc, TOP_MASK) and is_winning_move(opp_pos, nm, cc, BOTTOM_MASK, COL_MASK, stride_i):
                    return True
            return False

        # SIMPLIFIED move ordering - remove threat detection, keep killers + history
        @njit(cache=True, fastmath=True)
        def order_moves(mask: UINT, ply: np.int32, killers: np.ndarray, history: np.ndarray,
                        CENTER_ORDER: np.ndarray, TOP_MASK: np.ndarray) -> (np.ndarray, np.int32):
            moves = np.empty(7, dtype=np.int8)
            scores = np.empty(7, dtype=np.int32)
            m = 0
            k1, k2 = killers[ply, 0], killers[ply, 1]
            for i in range(CENTER_ORDER.shape[0]):
                c = np.int32(CENTER_ORDER[i])
                if can_play(mask, c, TOP_MASK):
                    s = 0
                    if c == k1: s += 1_000_000
                    elif c == k2: s += 500_000
                    s += history[c]
                    moves[m] = np.int8(c)
                    scores[m] = s
                    m += 1
            
            # Simple insertion sort (no threat detection)
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

        # Keep basic TT without age complexity
        @njit(cache=True, fastmath=True)
        def tt_hash_local(pos: UINT, mask: UINT, size_mask_i: np.int32) -> np.int32:
            h = np.uint64(pos) ^ (np.uint64(mask) * np.uint64(0x9E3779B97F4A7C15))
            h ^= (h >> np.uint64(7))
            return np.int32(h & np.uint64(size_mask_i))

        @njit(cache=True, fastmath=True)
        def tt_lookup(pos: UINT, mask: UINT, depth: np.int16, alpha: float, beta: float,
                      TT_pos: np.ndarray, TT_mask: np.ndarray, TT_depth: np.ndarray, TT_flag: np.ndarray,
                      TT_val: np.ndarray, TT_move: np.ndarray) -> (bool, float, np.int8, float, float):
            size_mask = np.int32(TT_pos.shape[0] - 1)
            idx = tt_hash_local(pos, mask, size_mask)
            for _ in range(32):
                if TT_depth[idx] == -1:
                    return False, 0.0, np.int8(-1), alpha, beta
                if TT_pos[idx] == pos and TT_mask[idx] == mask:
                    d = TT_depth[idx]; flag = TT_flag[idx]; val = TT_val[idx]; mv = TT_move[idx]
                    if d >= depth:
                        if flag == 0:
                            return True, val, mv, alpha, beta
                        if flag == 1 and val > alpha: alpha = val
                        elif flag == 2 and val < beta: beta = val
                        if alpha >= beta: return True, val, mv, alpha, beta
                    return False, val, mv, alpha, beta
                idx = (idx + 1) & size_mask
            return False, 0.0, np.int8(-1), alpha, beta

        @njit(cache=True, fastmath=True)
        def tt_store(pos: UINT, mask: UINT, depth: np.int16, val: float, alpha0: float, beta: float, best_mv: np.int8,
                     TT_pos: np.ndarray, TT_mask: np.ndarray, TT_depth: np.ndarray, TT_flag: np.ndarray,
                     TT_val: np.ndarray, TT_move: np.ndarray) -> None:
            flag = 0
            if val <= alpha0: flag = 2
            elif val >= beta: flag = 1
            size_mask = np.int32(TT_pos.shape[0] - 1)
            idx = tt_hash_local(pos, mask, size_mask)
            victim = -1
            for _ in range(32):
                if TT_depth[idx] == -1: victim = idx; break
                if TT_pos[idx] == pos and TT_mask[idx] == mask: victim = idx; break
                if victim == -1 or TT_depth[idx] < TT_depth[victim]: victim = idx
                idx = (idx + 1) & size_mask
            if victim >= 0:
                TT_pos[victim] = pos
                TT_mask[victim] = mask
                TT_depth[victim] = depth
                TT_flag[victim] = flag
                TT_val[victim] = val
                TT_move[victim] = best_mv

        @njit(cache=True)
        def should_stop(node_counter: np.ndarray, max_nodes: np.int64) -> bool:
            node_counter[0] += 1
            return node_counter[0] >= max_nodes

        # REMOVE all aggressive pruning - keep search complete
        @njit(cache=True, fastmath=True)
        def negamax(pos: UINT, mask: UINT, depth: np.int16, alpha: float, beta: float, ply: np.int16,
                    node_counter: np.ndarray, max_nodes: np.int64, MATE_SCORE: float,
                    COL_MASK: np.ndarray, WIN_MASKS: np.ndarray, WIN_B: np.ndarray, WIN_C: np.ndarray, WIN_R: np.ndarray,
                    WARR: np.ndarray, DEF: float, FN: float, FF: float, CENTER_MASK: UINT,
                    immediate_w: float, fork_w: float, center_bonus: float, CENTER_ORDER: np.ndarray,
                    TOP_MASK: np.ndarray, BOTTOM_MASK: np.ndarray, stride_i: np.int32, rows_i: np.int32, cols_i: np.int32,
                    killers: np.ndarray, history: np.ndarray, FULL_MASK: UINT,
                    TT_pos: np.ndarray, TT_mask: np.ndarray, TT_depth: np.ndarray, TT_flag: np.ndarray,
                    TT_val: np.ndarray, TT_move: np.ndarray) -> (float, np.int8):

            if should_stop(node_counter, max_nodes):
                return evaluate(pos, mask, COL_MASK, WIN_MASKS, WIN_B, WIN_C, WIN_R, WARR,
                                DEF, FN, FF, CENTER_MASK, immediate_w, fork_w, center_bonus,
                                CENTER_ORDER, TOP_MASK, BOTTOM_MASK, COL_MASK, stride_i, rows_i, cols_i), np.int8(-1)

            alpha0 = alpha
            hit, val_tt, mv_tt, alpha, beta = tt_lookup(pos, mask, depth, alpha, beta,
                                                        TT_pos, TT_mask, TT_depth, TT_flag, TT_val, TT_move)
            if hit: return val_tt, mv_tt

            # win-in-1
            for i in range(CENTER_ORDER.shape[0]):
                c = np.int32(CENTER_ORDER[i])
                if can_play(mask, c, TOP_MASK) and is_winning_move(pos, mask, c, BOTTOM_MASK, COL_MASK, stride_i):
                    return MATE_SCORE - float(ply), np.int8(c)

            if mask == FULL_MASK: return 0.0, np.int8(-1)
            
            # NO early pruning - search everything
            if depth == 0:
                return evaluate(pos, mask, COL_MASK, WIN_MASKS, WIN_B, WIN_C, WIN_R, WARR,
                                DEF, FN, FF, CENTER_MASK, immediate_w, fork_w, center_bonus,
                                CENTER_ORDER, TOP_MASK, BOTTOM_MASK, COL_MASK, stride_i, rows_i, cols_i), np.int8(-1)

            best_val = -1e100
            best_col = np.int8(-1)

            moves, m = order_moves(mask, ply, killers, history, CENTER_ORDER, TOP_MASK)
            # prune immediate handovers if possible
            safe = np.empty(7, dtype=np.int8); s = 0
            for j in range(m):
                c = np.int32(moves[j])
                if not is_immediate_blunder(pos, mask, c, CENTER_ORDER, TOP_MASK, BOTTOM_MASK, COL_MASK, stride_i):
                    safe[s] = np.int8(c); s += 1
            use_moves, use_len = (safe, s) if s > 0 else (moves, m)

            for j in range(use_len):
                c = np.int32(use_moves[j])
                mv = play_bit(mask, c, BOTTOM_MASK, COL_MASK)
                nm = mask | mv
                next_pos = nm ^ (pos | mv)

                child_val, _ = negamax(next_pos, nm, depth - 1, -beta, -alpha, ply + 1,
                                       node_counter, max_nodes, MATE_SCORE,
                                       COL_MASK, WIN_MASKS, WIN_B, WIN_C, WIN_R, WARR,
                                       DEF, FLOATING_NEAR, FLOATING_FAR, CENTER_MASK,
                                       IMMEDIATE_W, FORK_W, CENTER_BONUS,
                                       CENTER_ORDER, TOP_MASK, BOTTOM_MASK, stride_i, rows_i, cols_i,
                                       killers, history, FULL_MASK,
                                       TT_pos, TT_mask, TT_depth, TT_flag, TT_val, TT_move)
                val = -child_val

                if val > best_val:
                    best_val = val
                    best_col = np.int8(c)
                if best_val > alpha:
                    alpha = best_val
                if alpha >= beta:
                    if killers[ply, 0] != c:
                        killers[ply, 1] = killers[ply, 0]
                        killers[ply, 0] = np.int8(c)
                    history[c] += int(depth) * int(depth)
                    break

            tt_store(pos, mask, depth, best_val, alpha0, beta, best_col,
                     TT_pos, TT_mask, TT_depth, TT_flag, TT_val, TT_move)
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
            is_immediate_blunder=is_immediate_blunder, order_moves=order_moves,
            popcount64=popcount64
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
    order_moves = _CACHE["order_moves"]; popcount64 = _CACHE["popcount64"]

    # ---------- build pos/mask from Kaggle obs ----------
    MARK = int(obs.mark)
    pos1 = np.uint64(0); pos2 = np.uint64(0); mask = np.uint64(0)
    board = obs.board
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

    # ---------- root logic: identical to original ----------
    DEADLINE = time.perf_counter() + 7.6

    killers = np.full((64, 2), -1, dtype=np.int8)
    history = np.zeros(COLS, dtype=np.int32)
    node_counter = np.zeros(1, dtype=np.int64)

    TT_SIZE = 1 << 16
    TT_pos  = np.zeros(TT_SIZE, dtype=np.uint64)
    TT_mask = np.zeros(TT_SIZE, dtype=np.uint64)
    TT_depth= np.full(TT_SIZE, -1, dtype=np.int16)
    TT_flag = np.zeros(TT_SIZE, dtype=np.int8)
    TT_val  = np.zeros(TT_SIZE, dtype=np.float64)
    TT_move = np.full(TT_SIZE, -1, dtype=np.int8)

    legal = [c for c in CENTER_ORDER.tolist() if (mask & TOP_MASK[int(c)]) == 0]
    if not legal:
        return 0

    for c in legal:
        ci = int(c)
        if is_winning_move(pos, mask, ci, BOTTOM_MASK, COL_MASK, STRIDE):
            return ci

    opp_pos = mask ^ pos
    block_col, block_count = -1, 0
    for c in legal:
        ci = int(c)
        if is_winning_move(opp_pos, mask, ci, BOTTOM_MASK, COL_MASK, STRIDE):
            block_col = ci; block_count += 1
            if block_count > 1: break
    if block_count == 1:
        return block_col

    safe = [int(c) for c in legal if not is_immediate_blunder(pos, mask, int(c), CENTER_ORDER, TOP_MASK, BOTTOM_MASK, COL_MASK, STRIDE)]
    if safe:
        legal = safe

    best_move = int(legal[0])
    # FIXED evaluation call - proper argument count (22 arguments)
    best_val = float(evaluate(
        pos, mask, 
        COL_MASK, WIN_MASKS, WIN_B, WIN_C, WIN_R, WARR,
        float(DEFENSIVE), float(FLOATING_NEAR), float(FLOATING_FAR), 
        np.uint64(CENTER_MASK),
        float(IMMEDIATE_W), float(FORK_W), float(CENTER_BONUS),
        CENTER_ORDER, TOP_MASK, BOTTOM_MASK, 
        COL_MASK,  # This is the COL_MASK2 parameter
        np.int32(STRIDE), np.int32(ROWS), np.int32(COLS)
    ))

    ASP_INIT = 10.0 * IMMEDIATE_W

    def depth_to_budget(d: int) -> np.int64:
        base = 250_000  # Back to original budget
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