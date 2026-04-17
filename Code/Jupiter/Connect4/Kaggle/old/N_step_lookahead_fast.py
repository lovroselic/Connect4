# N_step_lookahead_fast.py

def N_step_lookahead_fast(obs, config):
    """
    ConnectX agent with fast in-place alpha–beta minimax lookahead and gravity-aware heuristic.
    Converted from my Connect-4 game in JavaScript: https://www.laughingskull.org/Games/Connect4/Connect4.php
    Source: https://github.com/lovroselic/Connect4
    """
    import numpy as np
    
    # Search depth 
    N_STEPS = 7

    weights_dict = {2: 10.0, 3: 1000.0, 4: 100000.0}

    # Heuristic weights 
    MATE_SCORE = 1e12
    immediate_w = weights_dict[4]
    fork_w = weights_dict[3] * 2
    DEFENSIVE = 1.5
    FLOATING_NEAR = 0.75   # needs exactly 1 filler to become supported
    FLOATING_FAR  = 0.50   # needs 2+ fillers, still counts but less
    
    
    ###########################################################################
    
    # ----------------------------- primitives -------------------------------
    def legal_moves():
        return [c for c in CENTER_ORDER if H[c] < ROWS]

    def make_move(col, player):
        r = H[col]
        B[r][col] = player
        H[col] = r + 1
        return r

    def unmake_move(col):
        H[col] -= 1
        r = H[col]
        B[r][col] = 0

    def line_len(r, c, dr, dc, player):
        n = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and B[rr][cc] == player:
            n += 1; rr += dr; cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and B[rr][cc] == player:
            n += 1; rr -= dr; cc -= dc
        return n

    def is_win_from(r, c, player):
        return (
            line_len(r, c, 1, 0, player) >= K or   # vertical
            line_len(r, c, 0, 1, player) >= K or   # horizontal
            line_len(r, c, 1, 1, player) >= K or   # diag /
            line_len(r, c, 1, -1, player) >= K     # diag \
        )

    def someone_has_k():
        for window in W:
            z = 0; s = 0
            for (r, c) in window:
                v = B[r][c]
                if v == 0:
                    z += 1
                s += v
            if z == 0 and (s == K or s == -K):
                return True
        return False

    def count_immediate_wins(player):
        cnt = 0
        for c in range(COLS):
            if H[c] >= ROWS:
                continue
            r = make_move(c, player)
            win = is_win_from(r, c, player)
            unmake_move(c)
            if win:
                cnt += 1
        return cnt

    # -------------------------- evaluation (leaf) ---------------------------
    def evaluate_leaf(pov):
        score = 0.0
        # pattern scoring with gravity multipliers
        for window in W:
            p = 0; o = 0; need = 0
            for (r, c) in window:
                v = B[r][c]
                if v == 0:
                    # how many tokens must be dropped in this column before this cell is playable?
                    if r > 0 and B[r - 1][c] == 0:
                        need += (r - H[c])
                elif v == pov:
                    p += 1
                else:
                    o += 1
            if (p + o) < 2: continue
                    
                    
            mul = 1.0 if need == 0 else (FLOATING_NEAR if need == 1 else FLOATING_FAR)
            if o == 0:
                score += mul * WARR[p if p <= K else 0]
            elif p == 0:
                score -= mul * DEFENSIVE * WARR[o if o <= K else 0]

        # immediate wins & fork signals
        my_imm  = count_immediate_wins(pov)
        opp_imm = count_immediate_wins(-pov)
        score += immediate_w * (my_imm - DEFENSIVE * opp_imm)
        if my_imm >= 2:
            score += fork_w * (my_imm - 1)
        if opp_imm >= 2:
            score -= DEFENSIVE * (fork_w * (opp_imm - 1))
        return score

    # ----------------------------- alpha–beta -------------------------------
    def ab_minimax(depth, to_move, alpha, beta):
        # Terminal/leaf → evaluate in ROOT POV
        if depth == 0 or someone_has_k(): return evaluate_leaf(ROOT)

        lm = legal_moves()
        if not lm: return evaluate_leaf(ROOT)

        # Early win-in-1 cut for side to move
        for c in lm:
            r = make_move(c, to_move)
            win = is_win_from(r, c, to_move)
            unmake_move(c)
            
            if win: 
                return MATE_SCORE if to_move == ROOT else -MATE_SCORE

        maximizing = (to_move == ROOT)
        if maximizing:
            value = -float("inf")
            for col in lm:  # center-ordered
                r = make_move(col, to_move)
                if is_win_from(r, col, to_move):
                    child = MATE_SCORE if to_move == ROOT else -MATE_SCORE
                else:
                    child = ab_minimax(depth - 1, -to_move, alpha, beta)
                unmake_move(col)

                if child > value: value = child
                if value > alpha: alpha = value
                if alpha >= beta: break
            return value
        else:
            value = float("inf")
            for col in lm:
                r = make_move(col, to_move)
                if is_win_from(r, col, to_move):
                    child = MATE_SCORE if to_move == ROOT else -MATE_SCORE
                else:
                    child = ab_minimax(depth - 1, -to_move, alpha, beta)
                unmake_move(col)

                if child < value: value = child
                if value < beta:  beta = value
                if alpha >= beta: break
            return value
    
    ###########################################################################

   

    # ------------------------------- config ---------------------------------
    
    ROWS, COLS, K = 6, 7, 4
    CENTER_COL = 3
    CENTER_ORDER = [3, 4, 2, 5, 1, 6, 0]

    # Pattern weights for counts in a 4-window; unused indices default to 0
    
    WARR = [0.0] * (K + 1)
    for i in range(2, min(K, 4) + 1):
        WARR[i] = float(weights_dict.get(i, 0.0))

    # ------------------------------ state -----------------------------------
    # Bottom-based board: row 0 is the bottom, heights track next free row per column.
    B = [[0] * COLS for _ in range(ROWS)]
    H = [0] * COLS

    # Precompute all K-length windows (horiz, vert, diag) in bottom-based coords.
    W = []
    
    # horizontal
    for r in range(ROWS):
        for c in range(COLS - K + 1):
            W.append([(r, c + i) for i in range(K)])
    # vertical
    for c in range(COLS):
        for r in range(ROWS - K + 1):
            W.append([(r + i, c) for i in range(K)])
    # diag up-right
    for r in range(ROWS - K + 1):
        for c in range(COLS - K + 1):
            W.append([(r + i, c + i) for i in range(K)])
    # diag up-left
    for r in range(ROWS - K + 1):
        for c in range(K - 1, COLS):
            W.append([(r + i, c - i) for i in range(K)])

    # -------------------------- load observation ----------------------------
    # Convert flat top-based board (0,1,2) to bottom-based lists with 2 -> -1.
    # my js game does not have Kaggle competition compatible data sctructure
    
    grid = np.asarray(obs.board, dtype=int).reshape(ROWS, COLS)
    for r_np in range(ROWS):
        inv_r = ROWS - 1 - r_np  # flip vertically
        for c in range(COLS):
            v = int(grid[r_np, c])
            if v == 2:
                v = -1
            if v != 0:
                B[inv_r][c] = v

    for c in range(COLS):
        h = 0
        while h < ROWS and B[h][c] != 0:
            h += 1
        H[c] = h

    # Root player's POV: 1 for mark==1, -1 for mark==2
    ROOT = 1 if int(obs.mark) == 1 else -1

    # ------------------------------ Main ----------------------------------
    
    legal = legal_moves()

    # Immediate winning move now? 
    for c in legal:
        r = make_move(c, ROOT)
        won = is_win_from(r, c, ROOT)
        unmake_move(c)
        if won: return c
    
    for c in legal:
        r = make_move(c, -ROOT)
        won = is_win_from(r, c, -ROOT)
        unmake_move(c)
        if won: return c

    # Lookahead
    best_score = -float("inf")
    best_cols = []
    for col in legal:
        r = make_move(col, ROOT)
        
        if is_win_from(r, col, ROOT):
            score = MATE_SCORE
        else:
            score = ab_minimax(N_STEPS - 1, -ROOT, -float("inf"), float("inf"))
        
        unmake_move(col)

        if score > best_score:
            best_score = score
            best_cols = [col]
        elif score == best_score:
            best_cols.append(col)

    
    best_cols.sort(key=lambda x: abs(CENTER_COL - x))
    return best_cols[0]
