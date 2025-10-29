# replacing: # fast_connect4_lookahead.py
# bb_connect4_lookahead.py

# Bitboard-accelerated Connect-4 lookahead with the same public API as before.
# - Class name kept: Connect4Lookahead
# - Public methods preserved: get_heuristic, is_terminal, minimax, n_step_lookahead,
#   has_four/check_win, count_immediate_wins, compute_fork_signals,
#   count_pure, count_pure_block_delta
#
# Internals:
# - Pascal Pons bitboard layout: bit index = c*(ROWS+1) + r, bottom-based rows.
# - Negamax + alpha-beta with a tiny TT (exact/lower/upper) and center-first ordering.
# - Gravity-aware heuristic with FLOATING_NEAR/FAR multipliers and DEFENSIVE factor.
#
# NOTE: Defaults match your previous class (weights, fork/immediate multipliers, etc.)
#       so behavior stays aligned; only the internals are bitboard-based for speed.

import random
from typing import List, Tuple, Dict, Optional


class Connect4Lookahead:
    ROWS = 6
    COLS = 7
    K    = 4
    CENTER_COL = 3

    # Heuristic knobs (kept from your original class)
    immediate_w  = 250.0
    fork_w       = 150.0
    DEFENSIVE    = 1.5
    FLOATING_NEAR = 0.75   # needs exactly 1 filler to be supported
    FLOATING_FAR  = 0.50   # needs 2+ fillers

    # Center-first order (stable with your previous implementation)
    _CENTER_ORDER = [3, 4, 2, 5, 1, 6, 0]

    # ----- Precomputed static bitboard tables (built once) -----
    _PRECOMP_DONE = False
    STRIDE: int = ROWS + 1
    COL_MASK: List[int] = [0]*COLS
    TOP_MASK: List[int] = [0]*COLS
    BOTTOM_MASK: List[int] = [0]*COLS
    FULL_MASK: int = 0
    CENTER_MASK: int = 0
    # Windows for heuristic (masks + (bit,c,r) triples to compute gravity needs)
    WIN_MASKS: List[int] = []
    WIN_CELLS: List[List[Tuple[int, int, int]]] = []

    def __init__(self, weights=None):
        # Keep your previous defaults for continuity
        self.weights: Dict[int, float] = weights if weights else {2: 10.0, 3: 200.0, 4: 1000.0}
        # Soft mate scale kept compatible with your old code
        self.SOFT_MATE = 100.0 * float(self.weights[4])
        # A (higher) internal mate for negamax bound framing & win-in-N; still consistent
        self.MATE_SCORE = float(self.weights[4]) * 1000.0

        # Tiny transposition table (can be disabled if you want byte-for-byte determinism)
        self._use_tt = True
        self._TT: Dict[Tuple[int, int], Tuple[int, int, float, int]] = {}
        self._EXACT, self._LOWER, self._UPPER = 0, 1, 2

        if not Connect4Lookahead._PRECOMP_DONE:
            self._build_precomp()

    # ============================== Public API ===============================

    def get_heuristic(self, board, player) -> float:
        """Heuristic evaluation in the POV of `player` (1 or 2 / or -1)."""
        p1, p2, mask = self._parse_board_bitboards(board)
        me = p1 if self._p(player) == 1 else p2
        return self._evaluate(me, mask)

    def is_terminal(self, board) -> bool:
        """Terminal if any side has 4 or board is full."""
        p1, p2, mask = self._parse_board_bitboards(board)
        return self._has_won(p1) or self._has_won(p2) or (mask == self.FULL_MASK)

    def minimax(self, board, depth, maximizing, player, alpha, beta) -> float:
        """
        Alpha-beta value in `player` POV (same semantics you had).
        If `maximizing` is True, the side to move is `player`, otherwise the opponent starts.
        """
        p1, p2, mask = self._parse_board_bitboards(board)
        root = self._p(player)
        to_move = root if maximizing else -root
        pos = p1 if to_move == 1 else p2

        # Negamax returns value in side-to-move POV
        val = self._negamax(pos, mask, depth, float(alpha), float(beta), 0)[0]
        return val if to_move == root else -val

    def n_step_lookahead(self, board, player, depth=3) -> int:
        """
        Root move selection (keeps your behavior):
        - Immediate win now
        - Must-block if opponent has a single win-in-1
        - Otherwise, search with negamax(depth-1)
        - Center-first tie-breaking, keep top-K (<=3) near-center and pick random
        """
        p1, p2, mask = self._parse_board_bitboards(board)
        me_mark = self._p(player)
        me  = p1 if me_mark == 1 else p2
        opp = p2 if me_mark == 1 else p1

        legal = [c for c in self._CENTER_ORDER if self._can_play(mask, c)]
        if not legal:
            raise ValueError("n_step_lookahead: No valid moves available.")

        # Immediate win?
        for c in legal:
            if self._is_winning_move(me, mask, c):
                return c

        # Must-block single win-in-1
        opp_wins = [c for c in self._CENTER_ORDER if self._can_play(mask, c) and self._is_winning_move(opp, mask, c)]
        if len(opp_wins) == 1:
            return opp_wins[0]

        # Evaluate each legal move by negamax (root POV)
        best_score = -float("inf")
        best_moves: List[int] = []

        for c in legal:
            mv = self._play_bit(mask, c)
            nm = mask | mv
            # Switch side-to-move (Pascal Pons trick): next_pos = nm ^ (me | mv)
            next_pos = nm ^ (me | mv)

            # If our move already wins, grant soft mate (kept from your logic)
            if self._has_won(me | mv):
                score = self.SOFT_MATE
            else:
                child_val, _ = self._negamax(next_pos, nm, depth - 1, -self.MATE_SCORE, self.MATE_SCORE, 1)
                score = -child_val

            if score > best_score:
                best_score = score
                best_moves = [c]
            elif score == best_score:
                best_moves.append(c)

        # same diversity rule as before: sort by distance to center, keep top K, then random
        best_moves.sort(key=lambda x: abs(self.CENTER_COL - x))
        K = min(3, len(best_moves))
        return random.choice(best_moves[:K])

    def has_four(self, board, player: int) -> bool:
        p1, p2, _ = self._parse_board_bitboards(board)
        bb = p1 if self._p(player) == 1 else p2
        return self._has_won(bb)

    # alias
    check_win = has_four

    def count_immediate_wins(self, board, player: int) -> List[int]:
        p1, p2, mask = self._parse_board_bitboards(board)
        me = p1 if self._p(player) == 1 else p2
        wins = []
        for c in self._CENTER_ORDER:
            if self._can_play(mask, c) and self._is_winning_move(me, mask, c):
                wins.append(c)
        return wins

    def compute_fork_signals(self, board_before, board_after, mover: int) -> Dict[str, int]:
        mover = self._p(mover)
        opp   = -mover
        my_after  = len(self.count_immediate_wins(board_after, mover))
        opp_before = len(self.count_immediate_wins(board_before, opp))
        opp_after  = len(self.count_immediate_wins(board_after,  opp))
        return {"my_after": my_after, "opp_before": opp_before, "opp_after": opp_after}

    # ---------------- Analytics helpers preserved ----------------

    def count_pure(self, board, player, n: int) -> int:
        """
        Count 4-length windows containing exactly `n` of `player` and no opponent stones.
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

    # -------- Bitboard core --------

    @classmethod
    def _build_precomp(cls) -> None:
        # Columns
        full = 0
        for c in range(cls.COLS):
            col_bits = 0
            for r in range(cls.ROWS):
                col_bits |= 1 << (c * cls.STRIDE + r)
            cls.COL_MASK[c] = col_bits
            cls.BOTTOM_MASK[c] = 1 << (c * cls.STRIDE + 0)
            cls.TOP_MASK[c]    = 1 << (c * cls.STRIDE + (cls.ROWS - 1))
            full |= col_bits
        cls.FULL_MASK = full
        cls.CENTER_MASK = cls.COL_MASK[cls.CENTER_COL]

        # Windows: masks + (bit,c,r) triplets for gravity accounting
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
    def _can_play(cls, mask: int, c: int) -> bool:
        return (mask & cls.TOP_MASK[c]) == 0

    @classmethod
    def _play_bit(cls, mask: int, c: int) -> int:
        # Next single-bit where a stone would land in column c
        return (mask + cls.BOTTOM_MASK[c]) & cls.COL_MASK[c]

    @classmethod
    def _has_won(cls, bb: int) -> bool:
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
    def _is_winning_move(cls, pos: int, mask: int, c: int) -> bool:
        mv = cls._play_bit(mask, c)
        return cls._has_won(pos | mv)

    # -------- Evaluation (gravity-aware, defensive) --------

    def _evaluate(self, pos: int, mask: int) -> float:
        """
        Evaluate leaf in POV of side whose stones are in `pos`.
        Uses gravity multipliers and immediate win/fork signals.
        """
        # Opponent stones from mask ^ pos
        opp = mask ^ pos

        # Heights per column by popcount
        H = [(mask & self.COL_MASK[c]).bit_count() for c in range(self.COLS)]

        # Pattern weights array WARR (index by count)
        WARR = [0.0] * (self.K + 1)
        for i in (2, 3, 4):
            WARR[i] = float(self.weights.get(i, 0.0))

        score = 0.0
        for idx, wmask in enumerate(self.WIN_MASKS):
            mo = wmask & opp
            mp = wmask & pos
            if mo and mp:
                continue  # blocked window

            p = mp.bit_count()
            o = mo.bit_count()
            if (p + o) < 2:
                continue  # ignore very dilute windows

            # Gravity need = total fillers required under empty cells in the window
            need = 0
            if p == 0 or o == 0:
                for (b, c, r) in self.WIN_CELLS[idx]:
                    if (mask & b) == 0:
                        dh = r - H[c]
                        if dh > 0:
                            need += dh

            mul = 1.0 if need == 0 else (self.FLOATING_NEAR if need == 1 else self.FLOATING_FAR)
            if o == 0:
                score += mul * WARR[p]
            elif p == 0:
                score -= self.DEFENSIVE * mul * WARR[o]

        # Tactical signals: win-in-1 and forks
        my_imm  = self._count_immediate_wins_bits(pos, mask)
        opp_imm = self._count_immediate_wins_bits(opp, mask)
        score += self.immediate_w * (my_imm - self.DEFENSIVE * opp_imm)
        if my_imm >= 2:
            score += self.fork_w * (my_imm - 1)
        if opp_imm >= 2:
            score -= self.DEFENSIVE * (self.fork_w * (opp_imm - 1))

        # Mild center bias (helps move ordering/value stability)
        score += 6.0 * ((pos & self.CENTER_MASK).bit_count() - (opp & self.CENTER_MASK).bit_count())
        return score

    def _count_immediate_wins_bits(self, pos: int, mask: int) -> int:
        cnt = 0
        for c in self._CENTER_ORDER:
            if self._can_play(mask, c) and self._is_winning_move(pos, mask, c):
                cnt += 1
        return cnt

    # -------- Negamax + alpha-beta + tiny TT --------

    def _tt_lookup(self, key, depth, alpha, beta) -> Tuple[Optional[float], Optional[int], bool, float, float]:
        if not self._use_tt:
            return None, None, False, alpha, beta
        e = self._TT.get(key)
        if not e:
            return None, None, False, alpha, beta
        d, flag, val, mv = e
        if d >= depth:
            if flag == self._EXACT:
                return val, mv, True, alpha, beta
            if flag == self._LOWER and val > alpha:
                alpha = val
            elif flag == self._UPPER and val < beta:
                beta = val
            if alpha >= beta:
                return val, mv, True, alpha, beta
        return None, mv, False, alpha, beta

    def _tt_store(self, key, depth, val, alpha0, beta, best_mv):
        if not self._use_tt:
            return
        flag = self._EXACT
        if val <= alpha0:
            flag = self._UPPER
        elif val >= beta:
            flag = self._LOWER
        self._TT[key] = (depth, flag, val, best_mv)

    def _ordered_moves(self, mask: int) -> List[int]:
        return [c for c in self._CENTER_ORDER if self._can_play(mask, c)]

    def _negamax(self, pos: int, mask: int, depth: int, alpha: float, beta: float, ply: int) -> Tuple[float, int]:
        """
        Negamax search with alpha-beta. Returns (value, best_col) in POV of side to move (pos).
        """
        # TT probe
        key = (pos, mask)
        alpha0 = alpha
        val_tt, mv_tt, hit, alpha, beta = self._tt_lookup(key, depth, alpha, beta)
        if hit:
            return val_tt, mv_tt if mv_tt is not None else -1

        # Quick tactical: win in one
        for c in self._CENTER_ORDER:
            if self._can_play(mask, c) and self._is_winning_move(pos, mask, c):
                return self.MATE_SCORE - ply, c

        # Draw/full?
        if mask == self.FULL_MASK:
            return 0.0, -1

        if depth == 0:
            return self._evaluate(pos, mask), -1

        best_val = -1e100
        best_col = -1

        for c in self._ordered_moves(mask):
            mv = self._play_bit(mask, c)
            nm = mask | mv
            next_pos = nm ^ (pos | mv)

            child_val, _ = self._negamax(next_pos, nm, depth - 1, -beta, -alpha, ply + 1)
            val = -child_val

            if val > best_val:
                best_val = val
                best_col = c
            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                break

        self._tt_store(key, depth, best_val, alpha0, beta, best_col)
        return best_val, best_col

    # -------- Board parsing --------

    def _parse_board_bitboards(self, board) -> Tuple[int, int, int]:
        """
        Accepts numpy-like 2D board in {0,1,2} or {0,1,-1}, top-based rows.
        Returns (p1_bb, p2_bb, mask).
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
