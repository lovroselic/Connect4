# kaggle: main.py
# Hybrid2 ConnectX submission:
#   - shared opening book first
#   - CNet192 policy model handles opening / middlegame
#   - bitboard lookahead takes over late, when branching is lower
#
# Package:
#   copy main_hybrid2.py submission\main.py
#   tar -czf submit.tar.gz -C submission main.py PPO_2004.pt

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_DEVICE = torch.device("cpu")
_MODEL: Optional[nn.Module] = None

_ROWS = 6
_COLS = 7
_CENTER_COL = 3
_CENTER_ORDER = (3, 4, 2, 5, 1, 6, 0)

# Main model for opening / middlegame mode.
# MODEL_FILE = "PPO_926.pt"
MODEL_FILE = "PPO_2004.pt"

# Hybrid2 switch.
# This is total stones already on the board BEFORE our move, not agent-turn count.
# Examples:
#   20 = earlier LA takeover
#   24 = conservative late takeover
#   28 = very late endgame-only takeover
LA_TAKES_OVER_AT_STONES = 18

# Late LA target depth. Iterative deepening still respects the time deadline, so
# values >7 are safe to test late; unfinished depths simply do not overwrite the
# last completed best move. Try 7, 8, 9, 10 as ablations.
LATE_LOOKAHEAD_STEPS = 7

# Preserve the old LA opening-book random side reply: after opponent opens center,
# reply either C or E. Set False for deterministic C/E mirroring.
RANDOMIZE_SECOND_PLAYER_BOOK_REPLY = True


# N_step_lookahead_bitboard.py
# Pure Python bitboard alpha-beta (negamax) + iterative deepening
# With global precompute cache + opening book
# + center-bottom-anchored parity + "core Allis-ish" goodies:
#   - Vertical threat bias (vertical windows worth more)
#   - Tempo / zugzwang proxy (safe-move mobility)
#   - Root-only parity move/unlock nudges
#   - Root-only threat-space nudge (immediate-win count after move)
#
# https://www.kaggleusercontent.com/episodes/#.json
# No Numba.

def N_step_lookahead_bitboard(obs, config):

    import time
    import random

    def bit_at(c, r):
        return 1 << (c * STRIDE + r)

    # ------------------------------ config ---------------------------------

    N_STEPS = int(globals().get("LATE_LOOKAHEAD_STEPS", 7))
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

        # submission label: bbLA-7 (w2-10) (D1.55 FN0.25 FF0.125 C3 PB0.75 V0.8 T75 PM0.5 PU0.25 TS9)

        # Heuristic weights
        W2, W3, W4 = 10.0, 1000.0, 100000.0
        WARR = (0.0, 0.0, W2, W3, W4)

        MATE_SCORE    = W4 * 1000.0
        IMMEDIATE_W   = W4
        FORK_W        = W4
        
        DEFENSIVE     = 1.55
        FLOATING_NEAR = 0.25
        FLOATING_FAR  = 0.125
        CENTER_BONUS  = 4.0
        PARITY_BONUS  = 0.75
        VERT_MUL = 0.8
        VERT_3_READY_BONUS = 0.0
        TEMPO_W = 72.5
        PARITY_MOVE_W = 1.75
        PARITY_UNLOCK_W = 0.25
        THREATSPACE_W = 9

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

    # ---------------- depth-based FLOATING_NEAR schedule ----------
    # Root search depth -> FN value (NOT integer division, never use // here).
    FLOATING_NEAR_BY_DEPTH = (
        (8,  FLOATING_NEAR),        # depth 1..8: classic FN
        (10, FLOATING_NEAR / 2.0),  # depth 9..10: light FN
        (99, FLOATING_NEAR / 8.0),  # depth 11+: very light FN
    )

    def fn_for_root_depth(root_depth):
        """Return (FN_eff, bucket_id) for the given root search depth."""
        d = int(root_depth)
        for bucket_id, (max_d, fnv) in enumerate(FLOATING_NEAR_BY_DEPTH):
            if d <= int(max_d):
                return float(fnv), int(bucket_id)
        # fallback
        return float(FLOATING_NEAR_BY_DEPTH[-1][1]), int(len(FLOATING_NEAR_BY_DEPTH) - 1)

    FN_EFF, FN_BUCKET = fn_for_root_depth(1)  # default before first evaluate/aspiration
    # ----------------------------------------------------------------------

    # -------------------------- time budget --------------------------------
    DEADLINE = time.perf_counter() + 2 - 0.05

    # -------------------------- obs access ---------------------------------
    MARK = int(obs["mark"]) if isinstance(obs, dict) else int(obs.mark)
    board = obs["board"] if isinstance(obs, dict) else obs.board

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

    # Early-exit variants (faster in the fork-guard loops)
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
        """After I play column c, how many immediate wins does opponent have on their turn?"""
        mv = play_bit(mask_, c)
        nm = mask_ | mv
        my_after = pos_ | mv
        if has_won(my_after):
            return 0
        opp_after = nm ^ my_after
        return count_immediate_wins(opp_after, nm)

    def opp_can_reply_create_double_threat(pos_, mask_, c):
        """
        2-ply tactical guard:
        After I play c, does opponent have a reply oc such that after oc,
        opponent has >=2 immediate wins on the next turn (fork),
        AND I do not have an immediate win right after oc?
        """
        mv = play_bit(mask_, c)
        nm = mask_ | mv
        my_after = pos_ | mv
        if has_won(my_after):
            return False

        opp_pos = nm ^ my_after  # opponent stones after my move (before their reply)

        for oc in CENTER_ORDER:
            if not can_play(nm, oc):
                continue

            mv2 = play_bit(nm, oc)
            nm2 = nm | mv2
            opp_after = opp_pos | mv2

            # If their reply wins immediately, then c is losing anyway.
            if has_won(opp_after):
                return True

            # If we can win immediately now, their fork attempt is irrelevant (they won't choose it).
            if has_any_immediate_win(my_after, nm2):
                continue

            # Fork: opponent threatens two different win-in-1 moves next.
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

    # true win-in-2 forcing check (root tactical pre-pass) ---
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
                            mul *= FN_EFF      # depth-based FN (root-depth controlled)
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

    # win-in-1
    for c in legal:
        if is_winning_move(pos, mask, c):
            return c

    # single must-block
    block = []
    for c in legal:
        if is_winning_move(opp_pos_now, mask, c):
            block.append(c)
            if len(block) > 1:
                break
    if len(block) == 1:
        return block[0]

    # avoid obvious handovers if possible
    safe_legal = [c for c in legal if not is_immediate_blunder(pos, mask, c)]
    if safe_legal:
        legal = safe_legal

    non_handover = [c for c in legal if not hands_over_win_in_1(pos, mask, c)]
    if non_handover:
        legal = non_handover

    # Root guards (cheap + helps before search even starts)
    if DOUBLE_THREAT_GUARD:
        guarded = [c for c in legal if opp_immediate_wins_after_my_move(pos, mask, c) < 2]
        if guarded:
            legal = guarded

    if FORK_REPLY_GUARD:
        guarded = [c for c in legal if not opp_can_reply_create_double_threat(pos, mask, c)]
        if guarded:
            legal = guarded

    # win-in-2 forcing (root tactical pre-pass)
    if WIN2_CHECK:
        for c in legal:
            if is_forced_win_in_2(pos, mask, c):
                return c

    # ----------------- TT, killers, history & ordering ---------------------
    TT = {}
    EXACT, LOWER, UPPER = 0, 1, 2
    killers = [[-1, -1] for _ in range(64)]
    history_tbl = [0] * COLS

    # TT key includes FN_BUCKET to prevent mixing TT across FN regimes.
    # This matters because FN changes at depth 9 and 11 in schedule.
    def tt_key(pos_, mask_):
        # 49-bit-ish boards, this keeps everything separated:
        # key = (pos << 66) | (mask << 2) | bucket_id
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
            out.append(k1); cand.remove(k1)
        if k2 != -1 and k2 in cand:
            out.append(k2); cand.remove(k2)

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

        key = tt_key(pos_, mask_)  # bucket-aware TT key
        alpha0 = alpha
        val_tt, mv_tt, hit, alpha, beta = tt_lookup(key, depth, alpha, beta)
        if hit:
            return val_tt, mv_tt

        # win-in-1
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

        # Hard tactical filtering (keep at all plies; it prunes losing lines fast)
        if DOUBLE_THREAT_GUARD:
            guarded = [c for c in use_moves if opp_immediate_wins_after_my_move(pos_, mask_, c) < 2]
            if guarded:
                use_moves = guarded

        if FORK_REPLY_GUARD:
            guarded = [c for c in use_moves if not opp_can_reply_create_double_threat(pos_, mask_, c)]
            if guarded:
                use_moves = guarded

        # Root-only: precompute heights once (for parity move/unlock)
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

            # Root-only nudges
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
    best_val = evaluate(pos, mask, 0)  # uses FN_EFF (currently depth=1 FN)
    ASP_INIT = 10.0 * IMMEDIATE_W

    depth = 0
    for depth in range(1, N_STEPS + 1):
        # update FN_EFF and FN_BUCKET for this root search depth
        FN_EFF, FN_BUCKET = fn_for_root_depth(depth)

        node_counter[0] = 0
        alpha = max(-MATE_SCORE, best_val - ASP_INIT)
        beta  = min( MATE_SCORE, best_val + ASP_INIT)

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
        tr = DEADLINE - time.perf_counter()
        print("Depth reached", depth, "time remaining", tr, "OVER", tr < 0)
        print("FN_EFF", FN_EFF, "FN_BUCKET", FN_BUCKET)

    return int(best_move)



# ---------- CNet192 ----------
class CNet192(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 192, kernel_size=4, padding=0)      # 6x7 -> 3x4
        self.conv_mid = nn.Conv2d(192, 192, kernel_size=3, padding=1)           # 3x4 -> 3x4
        self.conv2 = nn.Conv2d(192, 192, kernel_size=2, padding=0)              # 3x4 -> 2x3

        self.fc = nn.Linear(192 * 2 * 3, 192)

        self.policy_fc = nn.Linear(192, 192)
        self.policy_out = nn.Linear(192, 7)

        # Kept because checkpoint loading uses strict=True, do not remove!
        self.value_fc = nn.Linear(192, 192)
        self.value_out = nn.Linear(192, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv_mid(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        pol = F.relu(self.policy_fc(x))
        pol = self.policy_out(pol)            # (B, 7)

        val = F.relu(self.value_fc(x))
        val = self.value_out(val).squeeze(-1) # (B,)

        return pol, val


def _find_model_path() -> str:
    # Kaggle submission runtime: tar is extracted here.
    p = f"/kaggle_simulations/agent/{MODEL_FILE}"
    if os.path.exists(p):
        return p

    # Local smoke tests / notebook runs.
    p = os.path.join(os.getcwd(), MODEL_FILE)
    if os.path.exists(p):
        return p

    p = MODEL_FILE
    if os.path.exists(p):
        return p

    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")


def _load_model_once() -> nn.Module:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    ckpt_path = _find_model_path()
    ckpt = torch.load(ckpt_path, map_location=_DEVICE)

    if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt):
        raise RuntimeError("Unexpected checkpoint format: expected dict with 'model_state_dict'")

    model = CNet192(in_channels=1).to(_DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    _MODEL = model
    return _MODEL


# ---------- low-level board helpers ----------
def _get_obs_mark_and_grid(obs) -> Tuple[int, np.ndarray]:
    mark = int(obs["mark"]) if isinstance(obs, dict) else int(obs.mark)
    flat = obs["board"] if isinstance(obs, dict) else obs.board
    grid = np.asarray(flat, dtype=np.int8).reshape(_ROWS, _COLS)
    return mark, grid


def _legal_cols_from_grid(grid: np.ndarray):
    return [c for c in range(_COLS) if grid[0, c] == 0]


def _lowest_empty_row(grid: np.ndarray, col: int) -> int:
    for r in range(_ROWS - 1, -1, -1):
        if grid[r, col] == 0:
            return r
    return -1


def _has_four_from(grid: np.ndarray, row: int, col: int, token: int) -> bool:
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1

        rr, cc = row + dr, col + dc
        while 0 <= rr < _ROWS and 0 <= cc < _COLS and grid[rr, cc] == token:
            count += 1
            rr += dr
            cc += dc

        rr, cc = row - dr, col - dc
        while 0 <= rr < _ROWS and 0 <= cc < _COLS and grid[rr, cc] == token:
            count += 1
            rr -= dr
            cc -= dc

        if count >= 4:
            return True
    return False


def _is_winning_drop(pov: np.ndarray, col: int, token: int) -> bool:
    r = _lowest_empty_row(pov, col)
    if r < 0:
        return False

    old = pov[r, col]
    pov[r, col] = token
    win = _has_four_from(pov, r, col, token)
    pov[r, col] = old
    return win


def _legal_cols_from_pov(pov: np.ndarray):
    return [c for c in range(_COLS) if pov[0, c] == 0]


def _generate_non_losing_moves(pov: np.ndarray):
    """
    Return tactically safe moves for the side to move (+1).

    Logic:
    1) Check whether opponent (-1) already has immediate wins now.
       - If there are 2+, there is no true non-losing move.
       - If there is exactly 1, we are forced to block it.
    2) Otherwise, keep only moves that do not give opponent an immediate
       winning reply after our move.
    """
    legal = _legal_cols_from_pov(pov)
    if not legal:
        return []

    opp_wins_now = [
        c for c in _CENTER_ORDER
        if pov[0, c] == 0 and _is_winning_drop(pov, c, -1)
    ]

    if len(opp_wins_now) >= 2:
        return []

    if len(opp_wins_now) == 1:
        forced = opp_wins_now[0]
        return [forced] if forced in legal else []

    good = []
    for c in _CENTER_ORDER:
        if pov[0, c] != 0:
            continue

        r = _lowest_empty_row(pov, c)
        pov[r, c] = +1

        opp_has_reply_win = False
        for oc in _CENTER_ORDER:
            if pov[0, oc] == 0 and _is_winning_drop(pov, oc, -1):
                opp_has_reply_win = True
                break

        pov[r, c] = 0

        if not opp_has_reply_win:
            good.append(c)

    return good


def _infer_logits(model: nn.Module, pov: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(pov.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        logits, _ = model(x)

    logits = logits[0].detach().cpu().numpy().astype(np.float32)
    logits = np.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    return logits


# ---------- shared opening book ----------
def _opening_book_move(grid: np.ndarray, mark: int) -> Optional[int]:
    """
    Shared early book, ported from the bitboard agent to top-row-first grid space.
    This runs before the hybrid dispatch, so the book is kept for both sides.
    """
    legal = _legal_cols_from_grid(grid)
    if not legal:
        return None

    stones = int(np.count_nonzero(grid))

    # Board coordinates here: row 5 is bottom, row 4 is second from bottom, etc.
    a1 = grid[5, 0] != 0
    b1 = grid[5, 1] != 0
    c1 = grid[5, 2] != 0
    d1 = grid[5, 3] != 0
    e1 = grid[5, 4] != 0
    f1 = grid[5, 5] != 0
    g1 = grid[5, 6] != 0
    d2 = grid[4, 3] != 0
    d3 = grid[3, 3] != 0

    if stones == 0 and mark == 1 and _CENTER_COL in legal:
        return _CENTER_COL

    if stones == 1 and mark == 2:
        if d1:
            choices = [c for c in (2, 4) if c in legal]
            if choices:
                if RANDOMIZE_SECOND_PLAYER_BOOK_REPLY:
                    return int(random.choice(choices))
                return int(choices[0])

    if stones == 2 and mark == 1:
        if d1:
            if d2 and _CENTER_COL in legal:
                return _CENTER_COL

            if c1 and 5 in legal:
                return 5
            if e1 and 1 in legal:
                return 1

            if a1 and 4 in legal:
                return 4
            if g1 and 2 in legal:
                return 2

            if b1 and 5 in legal:
                return 5
            if f1 and 1 in legal:
                return 1

            if _CENTER_COL in legal:
                return _CENTER_COL

    if stones == 3 and mark == 2:
        if d1 and d2 and d3 and _CENTER_COL in legal:
            return _CENTER_COL

    return None


# ---------- RL/model path ----------
def _rl_agent(obs, config, mark: Optional[int] = None, grid: Optional[np.ndarray] = None) -> int:
    model = _load_model_once()

    if mark is None or grid is None:
        mark, grid = _get_obs_mark_and_grid(obs)

    legal = _legal_cols_from_grid(grid)
    if not legal:
        return 0

    # POV scalar board: me (POV) = +1, opp = -1
    pov = np.zeros((_ROWS, _COLS), dtype=np.int8)
    pov[grid == mark] = +1
    pov[(grid != 0) & (grid != mark)] = -1

    # Win now.
    for c in _CENTER_ORDER:
        if c in legal and _is_winning_drop(pov, c, +1):
            return int(c)

    # Prefer tactically safe moves if any exist.
    non_losing = _generate_non_losing_moves(pov)

    # Forced move.
    if len(non_losing) == 1:
        return int(non_losing[0])

    # Candidate set:
    # - prefer non-losing moves if any exist
    # - otherwise fall back to all legal moves in already-lost positions
    candidates = non_losing if non_losing else [c for c in _CENTER_ORDER if c in legal]

    # Policy inference.
    logits = _infer_logits(model, pov)

    # Final selection: logits first, center-distance and column index as deterministic tie-breaks.
    ordered = sorted(
        candidates,
        key=lambda c: (
            float(logits[c]),
            -abs(c - _CENTER_COL),
            -c,
        ),
        reverse=True,
    )

    return int(ordered[0])


# ---------- Kaggle agent ----------
def agent(obs, config):
    mark, grid = _get_obs_mark_and_grid(obs)

    legal = _legal_cols_from_grid(grid)
    if not legal:
        return 0

    stones = int(np.count_nonzero(grid))

    # 1) Shared opening book exactly before dispatch.
    # This keeps the old LA early book while allowing PPO/AZ to own the
    # opening/middlegame after book exhaustion.
    book_move = _opening_book_move(grid, mark)
    if book_move is not None and book_move in legal:
        return int(book_move)

    # 2) PPO/AZ owns early and middlegame.
    # 3) LA takes over late, where reduced branching makes deeper search more
    # reliable and less likely to overfit a shallow tactical horizon.
    if stones >= int(LA_TAKES_OVER_AT_STONES):
        return int(N_step_lookahead_bitboard(obs, config))

    return int(_rl_agent(obs, config, mark=mark, grid=grid))
