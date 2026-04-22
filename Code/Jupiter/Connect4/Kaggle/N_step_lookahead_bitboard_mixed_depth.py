# N_step_lookahead_bitboard_mixed_depth.py
# Pure Python bitboard alpha-beta (negamax) + iterative deepening
# With global precompute cache + opening book
# + center-bottom-anchored parity + "core Allis-ish" goodies:
#   - Vertical threat bias (vertical windows worth more)
#   - Tempo / zugzwang proxy (safe-move mobility)
#   - Root-only parity move/unlock nudges
#   - Root-only threat-space nudge (immediate-win count after move)
#
# Configurable search depth by starting role:
#   - bb_first_depth  (default 7)
#   - bb_second_depth (default 6)
#
# No Numba.

def N_step_lookahead_bitboard(obs, config):

    import time
    import random

    def cfg_get(cfg, key, default):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def bit_at(c, r):
        return 1 << (c * STRIDE + r)

    # ------------------------------ config ---------------------------------

    DEFAULT_FIRST_DEPTH = 7
    DEFAULT_SECOND_DEPTH = 6

    DEBUG = False
    OPENING_BOOK = True
    WIN2_CHECK = True
    DOUBLE_THREAT_GUARD = True   # after my move, opponent has >=2 win-in-1 moves
    FORK_REPLY_GUARD    = True   # after my move, opponent has a reply that creates >=2 win-in-1 threats

    # ----------------------------------------------------------------------

    # --------- global cache (precompute once across calls in same process) -
    _CACHE = globals().setdefault("_PY_C4_BIT_CACHE", {})
    if not _CACHE:
        ROWS, COLS, K = 6, 7, 4
        STRIDE = ROWS + 1

        CENTER_COL = 3
        CENTER_ORDER = (3, 4, 2, 5, 1, 6, 0)

        # submission label: bbLA-role (w2-10) (D1.55 FN0.25 FF0.125 C3 PB0.75 V0.8 T75 PM0.5 PU0.25 TS9)

        # Heuristic weights
        W2, W3, W4 = 10.0, 1000.0, 100000.0
        WARR = (0.0, 0.0, W2, W3, W4)

        MATE_SCORE    = W4 * 1000.0
        IMMEDIATE_W   = W4
        FORK_W        = W4
        DEFENSIVE     = 1.55
        FLOATING_NEAR = 0.50
        FLOATING_FAR  = 0.25
        CENTER_BONUS  = 6.0
        PARITY_BONUS  = 0.75
        VERT_MUL = 0.25
        VERT_3_READY_BONUS = 0.0
        TEMPO_W = 75
        PARITY_MOVE_W = 0.5
        PARITY_UNLOCK_W = 0.10
        THREATSPACE_W = 10

        # Column masks
        COL_MASK = [0] * COLS
        TOP_MASK = [0] * COLS
        BOTTOM_MASK = [0] * COLS
        FULL_MASK = 0

        for c in range(COLS):
            col_bits = 0
            base = c * STRIDE
            for r in range(ROWS):
                col_bits |= 1 << (base + r)
            COL_MASK[c] = col_bits
            BOTTOM_MASK[c] = 1 << (base + 0)
            TOP_MASK[c] = 1 << (base + (ROWS - 1))
            FULL_MASK |= col_bits

        CENTER_MASK = COL_MASK[CENTER_COL]

        # Row parity masks (bottom-based rows):
        # bottom row r=0 is "row 1" (odd), so odd rows are r%2==0.
        ODD_MASK = 0
        EVEN_MASK = 0
        for c in range(COLS):
            base = c * STRIDE
            for r in range(ROWS):
                b = 1 << (base + r)
                if (r & 1) == 0:
                    ODD_MASK |= b
                else:
                    EVEN_MASK |= b

        # Precompute all windows (69)
        # WIN_KIND: 0=horiz, 1=vert, 2=diag up-right, 3=diag up-left
        WIN_MASKS = []
        WIN_BITS  = []
        WIN_C     = []
        WIN_R     = []
        WIN_KIND  = []

        def add_window(cells, kind):
            m = 0
            bs = [0, 0, 0, 0]
            cs = [0, 0, 0, 0]
            rs = [0, 0, 0, 0]
            for i, (r, c) in enumerate(cells):
                b = bit_at(c, r)
                m |= b
                bs[i] = b
                cs[i] = c
                rs[i] = r
            WIN_MASKS.append(m)
            WIN_BITS.append(tuple(bs))
            WIN_C.append(tuple(cs))
            WIN_R.append(tuple(rs))
            WIN_KIND.append(kind)

        # horiz
        for r in range(ROWS):
            for c in range(COLS - K + 1):
                add_window([(r, c + i) for i in range(K)], kind=0)
        # vert
        for c in range(COLS):
            for r in range(ROWS - K + 1):
                add_window([(r + i, c) for i in range(K)], kind=1)
        # diag up-right
        for r in range(ROWS - K + 1):
            for c in range(COLS - K + 1):
                add_window([(r + i, c + i) for i in range(K)], kind=2)
        # diag up-left
        for r in range(ROWS - K + 1):
            for c in range(K - 1, COLS):
                add_window([(r + i, c - i) for i in range(K)], kind=3)

        _CACHE.update(dict(
            ROWS=ROWS, COLS=COLS, K=K, STRIDE=STRIDE,
            CENTER_COL=CENTER_COL, CENTER_ORDER=CENTER_ORDER,
            COL_MASK=COL_MASK, TOP_MASK=TOP_MASK, BOTTOM_MASK=BOTTOM_MASK,
            FULL_MASK=FULL_MASK, CENTER_MASK=CENTER_MASK,
            ODD_MASK=ODD_MASK, EVEN_MASK=EVEN_MASK,
            PARITY_BONUS=PARITY_BONUS,
            WIN_MASKS=tuple(WIN_MASKS),
            WIN_BITS=tuple(WIN_BITS),
            WIN_C=tuple(WIN_C),
            WIN_R=tuple(WIN_R),
            WIN_KIND=tuple(WIN_KIND),
            WARR=WARR,
            MATE_SCORE=MATE_SCORE,
            IMMEDIATE_W=IMMEDIATE_W,
            FORK_W=FORK_W,
            DEFENSIVE=DEFENSIVE,
            FLOATING_NEAR=FLOATING_NEAR,
            FLOATING_FAR=FLOATING_FAR,
            CENTER_BONUS=CENTER_BONUS,
            VERT_MUL=VERT_MUL,
            VERT_3_READY_BONUS=VERT_3_READY_BONUS,
            TEMPO_W=TEMPO_W,
            PARITY_MOVE_W=PARITY_MOVE_W,
            PARITY_UNLOCK_W=PARITY_UNLOCK_W,
            THREATSPACE_W=THREATSPACE_W,
        ))

    # -------------------------- unpack cache -------------------------------
    ROWS = _CACHE["ROWS"]; COLS = _CACHE["COLS"]; STRIDE = _CACHE["STRIDE"]
    CENTER_COL = _CACHE["CENTER_COL"]; CENTER_ORDER = _CACHE["CENTER_ORDER"]
    COL_MASK = _CACHE["COL_MASK"]; TOP_MASK = _CACHE["TOP_MASK"]; BOTTOM_MASK = _CACHE["BOTTOM_MASK"]
    FULL_MASK = _CACHE["FULL_MASK"]; CENTER_MASK = _CACHE["CENTER_MASK"]
    ODD_MASK = _CACHE["ODD_MASK"]; EVEN_MASK = _CACHE["EVEN_MASK"]; PARITY_BONUS = _CACHE["PARITY_BONUS"]
    WIN_MASKS = _CACHE["WIN_MASKS"]; WIN_BITS = _CACHE["WIN_BITS"]; WIN_C = _CACHE["WIN_C"]; WIN_R = _CACHE["WIN_R"]
    WIN_KIND = _CACHE["WIN_KIND"]
    WARR = _CACHE["WARR"]
    MATE_SCORE = _CACHE["MATE_SCORE"]; IMMEDIATE_W = _CACHE["IMMEDIATE_W"]; FORK_W = _CACHE["FORK_W"]
    DEFENSIVE = _CACHE["DEFENSIVE"]; FLOATING_NEAR = _CACHE["FLOATING_NEAR"]; FLOATING_FAR = _CACHE["FLOATING_FAR"]
    CENTER_BONUS = _CACHE["CENTER_BONUS"]
    VERT_MUL = _CACHE["VERT_MUL"]
    VERT_3_READY_BONUS = _CACHE["VERT_3_READY_BONUS"]
    TEMPO_W = _CACHE["TEMPO_W"]
    PARITY_MOVE_W = _CACHE["PARITY_MOVE_W"]
    PARITY_UNLOCK_W = _CACHE["PARITY_UNLOCK_W"]
    THREATSPACE_W = _CACHE["THREATSPACE_W"]

    # -------------------------- time budget --------------------------------
    DEADLINE = time.perf_counter() + 2 - 0.05

    # -------------------------- obs access ---------------------------------
    MARK = int(obs["mark"]) if isinstance(obs, dict) else int(obs.mark)
    board = obs["board"] if isinstance(obs, dict) else obs.board

    # Configurable depth by starting role:
    # MARK == 1 -> agent started first
    # MARK == 2 -> agent started second
    N_STEPS_FIRST = max(1, int(cfg_get(config, "bb_first_depth", DEFAULT_FIRST_DEPTH)))
    N_STEPS_SECOND = max(1, int(cfg_get(config, "bb_second_depth", DEFAULT_SECOND_DEPTH)))
    N_STEPS = N_STEPS_FIRST if MARK == 1 else N_STEPS_SECOND

    # -------------------------- bitboard build -----------------------------
    pos1 = 0
    pos2 = 0
    mask = 0

    for r_top in range(ROWS):
        r = ROWS - 1 - r_top
        base = r_top * COLS
        for c in range(COLS):
            v = board[base + c]
            if v:
                b = bit_at(c, r)
                mask |= b
                if v == 1:
                    pos1 |= b
                else:
                    pos2 |= b

    stones = mask.bit_count()
    pos = pos1 if MARK == 1 else pos2
    opp_pos_now = mask ^ pos

    # -------------------------- parity role anchor --------------------------
    # Enabled only if center-bottom (D1) is occupied. "First role" = owner of D1.
    b_d1 = 1 << (CENTER_COL * STRIDE + 0)
    parity_enabled = (mask & b_d1) != 0
    if parity_enabled:
        role_first_mark = 1 if (pos1 & b_d1) != 0 else 2
        root_pos_is_first = (MARK == role_first_mark)
    else:
        root_pos_is_first = False

    # ---------------- depth-based FLOATING_NEAR schedule ----------
    FLOATING_NEAR_BY_DEPTH = (
        (8,  FLOATING_NEAR),        # depth 1..8
        (10, FLOATING_NEAR / 2.0),  # depth 9..10
        (99, FLOATING_NEAR / 8.0),  # depth 11+
    )

    def fn_for_root_depth(root_depth):
        d = int(root_depth)
        for bucket_id, (max_d, fnv) in enumerate(FLOATING_NEAR_BY_DEPTH):
            if d <= int(max_d):
                return float(fnv), int(bucket_id)
        return float(FLOATING_NEAR_BY_DEPTH[-1][1]), int(len(FLOATING_NEAR_BY_DEPTH) - 1)

    FN_EFF, FN_BUCKET = fn_for_root_depth(1)
    # ----------------------------------------------------------------------

    # -------------------------- opening book -------------------------------
    if OPENING_BOOK:
        b_a1 = 1 << (0 * STRIDE + 0)
        b_b1 = 1 << (1 * STRIDE + 0)
        b_c1 = 1 << (2 * STRIDE + 0)
        b_d2 = 1 << (CENTER_COL * STRIDE + 1)
        b_d3 = 1 << (CENTER_COL * STRIDE + 2)
        b_e1 = 1 << (4 * STRIDE + 0)
        b_f1 = 1 << (5 * STRIDE + 0)
        b_g1 = 1 << (6 * STRIDE + 0)

        if stones == 0 and MARK == 1:
            return CENTER_COL

        if stones == 1 and MARK == 2:
            if mask & b_d1:
                return random.choice([2, 4])

        if stones == 2 and MARK == 1:
            if mask & b_d1:
                if mask & b_d2:
                    if (mask & TOP_MASK[CENTER_COL]) == 0:
                        return CENTER_COL

                if mask & b_c1:
                    return 5
                if mask & b_e1:
                    return 1

                if mask & b_a1:
                    return 4
                if mask & b_g1:
                    return 2

                if mask & b_b1:
                    return 5
                if mask & b_f1:
                    return 1

                if (mask & TOP_MASK[CENTER_COL]) == 0:
                    return CENTER_COL

        if stones == 3 and MARK == 2:
            if (mask & b_d1) and (mask & b_d2) and (mask & b_d3):
                if (mask & TOP_MASK[CENTER_COL]) == 0:
                    return CENTER_COL

    # -------------------------- bitboard helpers ---------------------------
    def can_play(mask_, c):
        return (mask_ & TOP_MASK[c]) == 0

    def play_bit(mask_, c):
        return (mask_ + BOTTOM_MASK[c]) & COL_MASK[c]

    def has_won(bb):
        m = bb & (bb >> 1)
        if (m & (m >> 2)) != 0:
            return True
        m = bb & (bb >> STRIDE)
        if (m & (m >> (2 * STRIDE))) != 0:
            return True
        m = bb & (bb >> (STRIDE + 1))
        if (m & (m >> (2 * (STRIDE + 1)))) != 0:
            return True
        m = bb & (bb >> (STRIDE - 1))
        if (m & (m >> (2 * (STRIDE - 1)))) != 0:
            return True
        return False

    def is_winning_move(pos_, mask_, c):
        mv = play_bit(mask_, c)
        return has_won(pos_ | mv)

    def heights(mask_):
        return [(mask_ & COL_MASK[c]).bit_count() for c in range(COLS)]

    def count_immediate_wins(pos_, mask_):
        cnt = 0
        for c in CENTER_ORDER:
            if can_play(mask_, c) and is_winning_move(pos_, mask_, c):
                cnt += 1
        return cnt

    def has_any_immediate_win(pos_, mask_):
        for c in CENTER_ORDER:
            if can_play(mask_, c) and is_winning_move(pos_, mask_, c):
                return True
        return False

    def has_two_immediate_wins(pos_, mask_):
        cnt = 0
        for c in CENTER_ORDER:
            if can_play(mask_, c) and is_winning_move(pos_, mask_, c):
                cnt += 1
                if cnt >= 2:
                    return True
        return False

    def opp_immediate_wins_after_my_move(pos_, mask_, c):
        mv = play_bit(mask_, c)
        nm = mask_ | mv
        my_after = pos_ | mv
        if has_won(my_after):
            return 0
        opp_after = nm ^ my_after
        return count_immediate_wins(opp_after, nm)

    def opp_can_reply_create_double_threat(pos_, mask_, c):
        mv = play_bit(mask_, c)
        nm = mask_ | mv
        my_after = pos_ | mv
        if has_won(my_after):
            return False

        opp_pos = nm ^ my_after

        for oc in CENTER_ORDER:
            if not can_play(nm, oc):
                continue

            mv2 = play_bit(nm, oc)
            nm2 = nm | mv2
            opp_after = opp_pos | mv2

            if has_won(opp_after):
                return True

            if has_any_immediate_win(my_after, nm2):
                continue

            if has_two_immediate_wins(opp_after, nm2):
                return True

        return False

    def is_immediate_blunder(pos_, mask_, c):
        mv = play_bit(mask_, c)
        nm = mask_ | mv
        opp_pos = nm ^ (pos_ | mv)
        for cc in CENTER_ORDER:
            if can_play(nm, cc) and is_winning_move(opp_pos, nm, cc):
                return True
        return False

    def count_safe_moves(pos_, mask_):
        s = 0
        for c in CENTER_ORDER:
            if can_play(mask_, c) and (not is_immediate_blunder(pos_, mask_, c)):
                s += 1
        return s

    def hands_over_win_in_1(pos_, mask_, c):
        mv = play_bit(mask_, c)
        nm = mask_ | mv
        my_after  = pos_ | mv
        opp_after = nm ^ my_after
        for cc in CENTER_ORDER:
            if can_play(nm, cc) and is_winning_move(opp_after, nm, cc):
                return True
        return False

    def is_forced_win_in_2(pos_, mask_, c):
        mv = play_bit(mask_, c)
        nm = mask_ | mv
        my_after = pos_ | mv
        opp_after = nm ^ my_after

        my_imm = count_immediate_wins(my_after, nm)
        if my_imm < 2:
            replies = 0
            for oc in CENTER_ORDER:
                if can_play(nm, oc):
                    replies += 1
                    if replies >= 2:
                        return False

        if count_immediate_wins(opp_after, nm) != 0:
            return False

        any_reply = False
        for oc in CENTER_ORDER:
            if not can_play(nm, oc):
                continue
            any_reply = True

            mv2 = play_bit(nm, oc)
            nm2 = nm | mv2

            win1 = False
            for cc in CENTER_ORDER:
                if can_play(nm2, cc) and is_winning_move(my_after, nm2, cc):
                    win1 = True
                    break

            if not win1:
                return False

        return any_reply

    # ----------------------------- evaluation ------------------------------
    def evaluate(pos_, mask_, ply):
        me = pos_
        opp = mask_ ^ pos_
        H = heights(mask_)

        score = 0.0
        for idx, wmask in enumerate(WIN_MASKS):
            mp = wmask & me
            mo = wmask & opp
            if mp and mo:
                continue

            p = mp.bit_count()
            o = mo.bit_count()
            if (p + o) < 2:
                continue

            mul = 1.0
            ready_vertical3 = False

            if p == 0 or o == 0:
                bits = WIN_BITS[idx]
                cols = WIN_C[idx]
                rows = WIN_R[idx]
                kind = WIN_KIND[idx]
                for k in range(4):
                    b = bits[k]
                    if (mask_ & b) == 0:
                        cc = cols[k]
                        rr = rows[k]
                        dh = rr - H[cc]
                        if dh == 1:
                            mul *= FN_EFF
                        elif dh >= 2:
                            mul *= FLOATING_FAR
                        else:
                            if kind == 1 and p == 3 and o == 0:
                                ready_vertical3 = True

            if WIN_KIND[idx] == 1:
                mul *= VERT_MUL

            if o == 0:
                score += mul * WARR[p]
                if ready_vertical3:
                    score += VERT_3_READY_BONUS
            elif p == 0:
                score -= DEFENSIVE * mul * WARR[o]

        my_imm = count_immediate_wins(me, mask_)
        opp_imm = count_immediate_wins(opp, mask_)
        score += IMMEDIATE_W * (my_imm - DEFENSIVE * opp_imm)

        if my_imm >= 2:
            score += FORK_W * (my_imm - 1)
        if opp_imm >= 2:
            score -= DEFENSIVE * FORK_W * (opp_imm - 1)

        score += CENTER_BONUS * ((me & CENTER_MASK).bit_count() - (opp & CENTER_MASK).bit_count())

        if parity_enabled:
            pos_is_first = root_pos_is_first if (ply & 1) == 0 else (not root_pos_is_first)
            if pos_is_first:
                score += PARITY_BONUS * (
                    (me & ODD_MASK).bit_count() - DEFENSIVE * (opp & EVEN_MASK).bit_count()
                )
            else:
                score += PARITY_BONUS * (
                    (me & EVEN_MASK).bit_count() - DEFENSIVE * (opp & ODD_MASK).bit_count()
                )

        if TEMPO_W:
            my_safe = count_safe_moves(me, mask_)
            opp_safe = count_safe_moves(opp, mask_)
            score += TEMPO_W * (float(my_safe) - DEFENSIVE * float(opp_safe))

        return score

    # ------------------------- root tactical picks -------------------------
    legal = [c for c in CENTER_ORDER if can_play(mask, c)]
    if not legal:
        return 0

    for c in legal:
        if is_winning_move(pos, mask, c):
            return c

    block = []
    for c in legal:
        if is_winning_move(opp_pos_now, mask, c):
            block.append(c)
            if len(block) > 1:
                break
    if len(block) == 1:
        return block[0]

    safe_legal = [c for c in legal if not is_immediate_blunder(pos, mask, c)]
    if safe_legal:
        legal = safe_legal

    non_handover = [c for c in legal if not hands_over_win_in_1(pos, mask, c)]
    if non_handover:
        legal = non_handover

    if DOUBLE_THREAT_GUARD:
        guarded = [c for c in legal if opp_immediate_wins_after_my_move(pos, mask, c) < 2]
        if guarded:
            legal = guarded

    if FORK_REPLY_GUARD:
        guarded = [c for c in legal if not opp_can_reply_create_double_threat(pos, mask, c)]
        if guarded:
            legal = guarded

    if WIN2_CHECK:
        for c in legal:
            if is_forced_win_in_2(pos, mask, c):
                return c

    # ----------------- TT, killers, history & ordering ---------------------
    TT = {}
    EXACT, LOWER, UPPER = 0, 1, 2
    killers = [[-1, -1] for _ in range(64)]
    history_tbl = [0] * COLS

    def tt_key(pos_, mask_):
        return (pos_ << 66) | (mask_ << 2) | FN_BUCKET

    def tt_lookup(key, depth, alpha, beta):
        e = TT.get(key)
        if not e:
            return None, -1, False, alpha, beta
        d, flag, val, mv = e
        if d >= depth:
            if flag == EXACT:
                return val, mv, True, alpha, beta
            if flag == LOWER and val > alpha:
                alpha = val
            elif flag == UPPER and val < beta:
                beta = val
            if alpha >= beta:
                return val, mv, True, alpha, beta
        return None, mv, False, alpha, beta

    def tt_store(key, depth, val, alpha0, beta, best_mv):
        flag = EXACT
        if val <= alpha0:
            flag = UPPER
        elif val >= beta:
            flag = LOWER
        TT[key] = (depth, flag, val, best_mv)

    def ordered_moves(mask_, ply, tt_mv=-1):
        k1, k2 = killers[ply]
        cand = [c for c in CENTER_ORDER if can_play(mask_, c)]

        out = []
        if tt_mv != -1 and tt_mv in cand:
            out.append(tt_mv)
            cand.remove(tt_mv)

        if k1 != -1 and k1 in cand:
            out.append(k1)
            cand.remove(k1)
        if k2 != -1 and k2 in cand:
            out.append(k2)
            cand.remove(k2)

        cand.sort(key=lambda c: history_tbl[c], reverse=True)
        out.extend(cand)
        return out

    # --------------------------- search core -------------------------------
    node_counter = [0]
    TIME_CHECK_MASK = 0x3FF

    if parity_enabled:
        root_is_first = bool(root_pos_is_first)
        pref_parity_root = 0 if root_is_first else 1
        pref_parity_opp  = 1 if root_is_first else 0
    else:
        pref_parity_root = 0
        pref_parity_opp = 1

    def negamax(pos_, mask_, depth, alpha, beta, ply):
        node_counter[0] += 1
        if (node_counter[0] & TIME_CHECK_MASK) == 0 and time.perf_counter() > DEADLINE:
            return evaluate(pos_, mask_, ply), -1

        key = tt_key(pos_, mask_)
        alpha0 = alpha
        val_tt, mv_tt, hit, alpha, beta = tt_lookup(key, depth, alpha, beta)
        if hit:
            return val_tt, mv_tt

        for c in CENTER_ORDER:
            if can_play(mask_, c) and is_winning_move(pos_, mask_, c):
                return MATE_SCORE - ply, c

        if mask_ == FULL_MASK:
            return 0.0, -1

        if depth == 0:
            return evaluate(pos_, mask_, ply), -1

        best_val = -1e100
        best_col = -1

        moves = ordered_moves(mask_, ply, mv_tt)

        safe = [c for c in moves if not is_immediate_blunder(pos_, mask_, c)]
        use_moves = safe if safe else moves

        if DOUBLE_THREAT_GUARD:
            guarded = [c for c in use_moves if opp_immediate_wins_after_my_move(pos_, mask_, c) < 2]
            if guarded:
                use_moves = guarded

        if FORK_REPLY_GUARD:
            guarded = [c for c in use_moves if not opp_can_reply_create_double_threat(pos_, mask_, c)]
            if guarded:
                use_moves = guarded

        if ply == 0 and parity_enabled:
            H0 = heights(mask_)
        else:
            H0 = None

        for c in use_moves:
            mv = play_bit(mask_, c)
            nm = mask_ | mv
            my_after = pos_ | mv
            opp_after = nm ^ my_after
            next_pos = opp_after

            bias = 0.0
            if ply == 0:
                if parity_enabled and H0 is not None:
                    r = H0[c]
                    if (r & 1) == pref_parity_root:
                        bias += PARITY_MOVE_W
                    else:
                        bias -= PARITY_MOVE_W

                    r2 = r + 1
                    if r2 < ROWS and ((r2 & 1) == pref_parity_opp):
                        bias -= PARITY_UNLOCK_W

                if THREATSPACE_W:
                    my_threats = count_immediate_wins(my_after, nm)
                    bias += THREATSPACE_W * float(my_threats)

                    opp_threats = count_immediate_wins(opp_after, nm)
                    bias -= DEFENSIVE * THREATSPACE_W * 0.25 * float(opp_threats)

            child_val, _ = negamax(next_pos, nm, depth - 1, -beta, -alpha, ply + 1)
            val = -child_val + bias

            if val > best_val:
                best_val = val
                best_col = c

            if best_val > alpha:
                alpha = best_val

            if alpha >= beta:
                k = killers[ply]
                if c != k[0]:
                    k[1] = k[0]
                    k[0] = c
                history_tbl[c] += depth * depth
                break

        tt_store(key, depth, best_val, alpha0, beta, best_col)
        return best_val, best_col

    # -------------------------- iterative deepening ------------------------
    best_move = legal[0]
    best_val = evaluate(pos, mask, 0)
    ASP_INIT = 10.0 * IMMEDIATE_W

    depth = 0
    for depth in range(1, N_STEPS + 1):
        FN_EFF, FN_BUCKET = fn_for_root_depth(depth)

        node_counter[0] = 0
        alpha = max(-MATE_SCORE, best_val - ASP_INIT)
        beta  = min(MATE_SCORE, best_val + ASP_INIT)

        v, mv = negamax(pos, mask, depth, alpha, beta, 0)

        if mv < 0 or v <= alpha:
            node_counter[0] = 0
            v, mv = negamax(pos, mask, depth, -MATE_SCORE, beta, 0)
        elif v >= beta:
            node_counter[0] = 0
            v, mv = negamax(pos, mask, depth, alpha, MATE_SCORE, 0)

        if mv >= 0:
            best_move = mv
            best_val = v

        if time.perf_counter() >= DEADLINE:
            if DEBUG:
                print("----- OVER TIME: stopping early -----")
            break

    if DEBUG:
        print("DEBUG depth_end best_move", best_move, "best_val", best_val)
        print("MARK", MARK, "N_STEPS", N_STEPS, "first_depth", N_STEPS_FIRST, "second_depth", N_STEPS_SECOND)
        tr = DEADLINE - time.perf_counter()
        print("Depth reached", depth, "time remaining", tr, "OVER", tr < 0)
        print("FN_EFF", FN_EFF, "FN_BUCKET", FN_BUCKET)

    return int(best_move)