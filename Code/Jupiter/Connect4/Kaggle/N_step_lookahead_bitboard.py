# N_step_lookahead_bitboard.py

# - Bitboard alpha–beta (negamax) + iterative deepening
# - Transposition table + killer/history move ordering
# - Immediate-loss pruning (skip obvious handovers)
# - Gravity-aware heuristic multipliers (FLOATING_NEAR / FLOATING_FAR)

# Time guard:
# - Uses a soft deadline  and iterative deepening; always returns a move in time.
# http://blog.gamesolver.org/solving-connect-four/06-bitboard/

def N_step_lookahead_bitboard(obs, config):
    import time

    # ------------------------------ config ---------------------------------
    #N_STEPS = 7
    N_STEPS = 9
    # Time budget: keep margin under Kaggle's ~8s hard cap
    DEADLINE = time.perf_counter() + 7.6
    DEBUG = False
    # ------------------------------ config ---------------------------------
    ROWS, COLS, K = 6, 7, 4
    CENTER_COL = 3
    CENTER_ORDER = [3, 4, 2, 5, 1, 6, 0]
    K = 4
    MARK  = int(obs.mark)

    # Heuristic weights 
    weights_dict = {2: 10.0, 3: 1000.0, 4: 100000.0}
    MATE_SCORE   = weights_dict[4] * 1000
    immediate_w  = weights_dict[4]
    fork_w       = weights_dict[3] * 2
    DEFENSIVE    = 1.5
    FLOATING_NEAR = 0.75   # needs exactly 1 filler to become supported
    FLOATING_FAR  = 0.50   # needs 2+ fillers, still counts but less
    CENTER_BONUS = 6.0

    

    # --------------------------- bitboard layout ---------------------------
    # Mapping (Pascal Pons style): bit index = c*STRIDE + r, bottom-based rows.
    STRIDE = ROWS + 1
    FULL_MASK = 0
    BOTTOM_MASK = [0] * COLS
    TOP_MASK    = [0] * COLS
    COL_MASK    = [0] * COLS

    for c in range(COLS):
        col_bits = 0
        for r in range(ROWS):
            col_bits |= 1 << (c * STRIDE + r)
        COL_MASK[c]    = col_bits
        BOTTOM_MASK[c] = 1 << (c * STRIDE + 0)
        TOP_MASK[c]    = 1 << (c * STRIDE + (ROWS - 1))
        FULL_MASK     |= col_bits

    CENTER_MASK = COL_MASK[CENTER_COL]

    def bit_at(c, r):
        return 1 << (c * STRIDE + r)

    def can_play(mask, c):
        return (mask & TOP_MASK[c]) == 0

    def play_bit(mask, c):
        # next single-bit in col c where a stone would land
        return (mask + BOTTOM_MASK[c]) & COL_MASK[c]

    # 4-in-a-row test via bit tricks (C-fast .bit_count and shifts)
    def has_won(bb):
        # vertical
        m = bb & (bb >> 1)
        if m & (m >> 2): return True
        
        # horizontal
        m = bb & (bb >> STRIDE)
        if m & (m >> (2 * STRIDE)): return True
        
        # diag up-right
        m = bb & (bb >> (STRIDE + 1))
        if m & (m >> (2 * (STRIDE + 1))): return True
        
        # diag up-left
        m = bb & (bb >> (STRIDE - 1))
        if m & (m >> (2 * (STRIDE - 1))): return True
        return False

    def is_winning_move(pos, mask, c):
        mv = play_bit(mask, c)
        return has_won(pos | mv)

    # --------------------------- precompute windows ------------------------
    # Build all K-length masks and (bit,c,r) triples per window for gravity math.
    WIN_MASKS = []
    WIN_CELLS = []  # parallel to WIN_MASKS: list[ list[(bit, c, r)] ]

    def add_window(cells):
        m = 0
        triple = []
        for (r, c) in cells:
            b = bit_at(c, r)
            m |= b
            triple.append((b, c, r))
        WIN_MASKS.append(m)
        WIN_CELLS.append(triple)

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

    # Pattern weights array WARR (index: count in window)
    WARR = [0.0] * (K + 1)
    for i in range(2, min(K, 4) + 1):
        WARR[i] = float(weights_dict.get(i, 0.0))

    # -------------------------- build pos/mask -----------------------------
    # Convert Kaggle's top-based 0/1/2 grid to bitboards (bottom-based).
    pos1 = 0
    pos2 = 0
    mask = 0
    board = obs.board  # length ROWS*COLS, top row first
    for r_top in range(ROWS):
        r = ROWS - 1 - r_top
        for c in range(COLS):
            v = board[r_top * COLS + c]
            if v:
                b = bit_at(c, r)
                mask |= b
                if v == 1: pos1 |= b
                else: pos2 |= b

    pos = pos1 if MARK == 1 else pos2  # stones of side-to-move (negamax POV)

    # ------------------------- helper: heights(H) --------------------------
    # Heights computed by popcount per column (cheap on tiny 64-bit-ish ints).
    def make_heights(mask_):
        # H[c] = number of occupied cells in column c (i.e., next free row index)
        return [ (mask_ & COL_MASK[c]).bit_count() for c in range(COLS) ]

    # ---------------------- immediate win counters -------------------------
    def count_immediate_wins(pos_, mask_):
        cnt = 0
        for c in CENTER_ORDER:
            if can_play(mask_, c) and is_winning_move(pos_, mask_, c):
                cnt += 1
        return cnt

    # ----------------------------- evaluation ------------------------------
    # Gravity-aware leaf eval using your multipliers and defensive factor.
    def evaluate(pos_, mask_):
        me  = pos_
        opp = mask_ ^ pos_
        H   = make_heights(mask_)  # gravity base

        score = 0.0

        for idx, wmask in enumerate(WIN_MASKS):
            mo = wmask & opp
            mp = wmask & me
            if mo and mp: continue  # window blocked by both sides

            p = mp.bit_count()
            o = mo.bit_count()
            if (p + o) < 2: continue  # ignore very dilute windows

            # Gravity need: how many fillers required for the empty cells in this window?
            # We approximate: need += (r - H[c]) for empties that are above current height.
            need = 0
            if p == 0 or o == 0:
                for (b, c, r) in WIN_CELLS[idx]:
                    if (mask_ & b) == 0:
                        dh = r - H[c]
                        if dh > 0:
                            need += dh

            mul = 1.0 if need == 0 else (FLOATING_NEAR if need == 1 else FLOATING_FAR)

            if o == 0: score += mul * WARR[p if p <= K else 0]
            elif p == 0: score -= DEFENSIVE * mul * WARR[o if o <= K else 0]

        # Immediate tactical signals (win-in-1 and forks)
        my_imm  = count_immediate_wins(me,  mask_)
        opp_imm = count_immediate_wins(opp, mask_)
        score += immediate_w * (my_imm - DEFENSIVE * opp_imm)
        if my_imm >= 2: score += fork_w * (my_imm - 1)
        if opp_imm >= 2: score -= DEFENSIVE * fork_w * (opp_imm - 1)

        # Small center preference
        score += CENTER_BONUS * ((me & CENTER_MASK).bit_count() - (opp & CENTER_MASK).bit_count())
        return score

    # ----------------- TT, killers, history & ordering ---------------------
    TT = {}  # key: (pos, mask) -> (depth, flag, value, best_col)
    EXACT, LOWER, UPPER = 0, 1, 2
    killers = [[-1, -1] for _ in range(N_STEPS + 2)]
    history = [0] * COLS

    def tt_lookup(key, depth, alpha, beta):
        e = TT.get(key)
        if not e:
            return None, None, False
        d, flag, val, mv = e
        if d >= depth:
            if flag == EXACT:
                return val, mv, True
            if flag == LOWER and val > alpha:
                alpha = val
            elif flag == UPPER and val < beta:
                beta = val
            if alpha >= beta:
                return val, mv, True
        return None, mv, False

    def tt_store(key, depth, val, alpha0, beta, best_mv):
        flag = EXACT
        if val <= alpha0:
            flag = UPPER
        elif val >= beta:
            flag = LOWER
        TT[key] = (depth, flag, val, best_mv)

    def ordered_moves(mask_, depth):
        cand = [c for c in CENTER_ORDER if can_play(mask_, c)]
        k1, k2 = killers[depth]
        def key(c):
            return (0 if c == k1 or c == k2 else 1, -history[c])
        cand.sort(key=key)
        return cand

    def is_immediate_blunder(pos_, mask_, c):
        mv = play_bit(mask_, c)
        nm = mask_ | mv
        opp_pos = nm ^ (pos_ | mv)
        for cc in CENTER_ORDER:
            if can_play(nm, cc) and is_winning_move(opp_pos, nm, cc):
                return True
        return False

    # --------------------------- search core -------------------------------
    # Negamax with alpha–beta; returns (value, best_col).
    # Time check is amortized using a counter to minimize overhead.
    node_counter = 0
    TIME_CHECK_MASK = 0x7FF  # check ~ every 2048 nodes

    def negamax(pos_, mask_, depth, alpha, beta, ply):
        nonlocal node_counter
        node_counter += 1
        if (node_counter & TIME_CHECK_MASK) == 0 and time.perf_counter() > DEADLINE:
            return evaluate(pos_, mask_), -1

        key = (pos_, mask_)
        alpha0 = alpha
        val_tt, mv_tt, hit = tt_lookup(key, depth, alpha, beta)
        if hit: return val_tt, mv_tt

        # Quick win-in-1 for side to move
        for c in CENTER_ORDER:
            if can_play(mask_, c) and is_winning_move(pos_, mask_, c):
                return MATE_SCORE - ply, c

        # Draw?
        if mask_ == FULL_MASK: return 0.0, -1

        if depth == 0: return evaluate(pos_, mask_), -1

        best_val = -1e100
        best_col = -1

        # Generate & prune immediate blunders (unless all moves are blunders)
        moves = ordered_moves(mask_, ply)
        safe = [c for c in moves if not is_immediate_blunder(pos_, mask_, c)]
        use_moves = safe if safe else moves

        for c in use_moves:
            mv = play_bit(mask_, c)
            nm = mask_ | mv
            next_pos = nm ^ (pos_ | mv)

            child_val, _ = negamax(next_pos, nm, depth - 1, -beta, -alpha, ply + 1)
            val = -child_val

            if val > best_val:
                best_val = val
                best_col = c
            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                # killer & history bookkeeping
                k = killers[ply]
                if c != k[0]:
                    k[1] = k[0]
                    k[0] = c
                history[c] += depth * depth
                break

        tt_store(key, depth, best_val, alpha0, beta, best_col)
        return best_val, best_col

    # ------------------------- root tactical picks -------------------------
    legal = [c for c in CENTER_ORDER if can_play(mask, c)]
    
    # Immediate win for us?
    for c in legal:
        if is_winning_move(pos, mask, c): return c
    
    # If opponent has a win-in-one *right now*, we must play *that* column to block.
    opp_pos_now = mask ^ pos
    opp_win_cols = [c for c in CENTER_ORDER if can_play(mask, c) and is_winning_move(opp_pos_now, mask, c)]
    if len(opp_win_cols) == 1: return opp_win_cols[0]
    # If len >= 2 there is no single-block; fall through to search (may still pick the least-bad).
    
    # avoid obvious blunders at root (if at least one safe move exists)
    safe_legal = [c for c in legal if not is_immediate_blunder(pos, mask, c)]
    if safe_legal: legal = safe_legal


    # -------------------------- iterative deepening ------------------------
    best_move = legal[0] if legal else 0
    best_val  = None
    for depth in range(1, N_STEPS + 1):
        if time.perf_counter() > DEADLINE: break
        v, mv = negamax(pos, mask, depth, -MATE_SCORE, MATE_SCORE, 0)
        if mv is not None and mv >= 0: best_move, best_val = mv, v
            
    if DEBUG: print("DEBUG, time", time.perf_counter(), depth, "best val", best_val)

    return best_move
