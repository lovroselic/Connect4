# fast_connect4_lookahead.py


import random
import numpy as np
from typing import List, Tuple


class Connect4Lookahead:
    ROWS = 6
    COLS = 7
    CENTER_COL = 3

    immediate_w = 250
    fork_w = 150
    DEFENSIVE = 1.5
    FLOATING_FACTOR = 0.25

    def __init__(self, weights=None):
        self.weights = weights if weights else {2: 10, 3: 100, 4: 1000}

        # CHANGE: internal fast state (bottom-based rows)
        self._board: List[List[int]] = [[0] * self.COLS for _ in range(self.ROWS)]
        self._heights: List[int] = [0] * self.COLS
        self._root_player: int = 1

        # CHANGE: precompute windows; center-first order
        self._windows: List[List[Tuple[int, int]]] = self._precompute_windows()
        self._CENTER_ORDER = [3, 4, 2, 5, 1, 6, 0]

    # ------------------------ original helpers (kept) ------------------------

    def get_legal_moves(self, board):
        order = [3, 4, 2, 5, 1, 6, 0]
        return [c for c in order if board[0][c] == 0]

    def drop_piece(self, board, col, player):
        new_board = board.copy()
        for row in reversed(range(self.ROWS)):
            if new_board[row][col] == 0:
                new_board[row][col] = player
                break
        return new_board

    def board_to_patterns(self, board, players):
        patterns = []
        # Horizontal
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                window = board[r, c:c+4]
                if any(val in players for val in window):
                    patterns.append(window)
        # Vertical
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                window = board[r:r+4, c]
                if any(val in players for val in window):
                    patterns.append(window)
        # Diagonal \
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                window = np.array([board[r+i][c+i] for i in range(4)])
                if any(val in players for val in window):
                    patterns.append(window)
        # Diagonal /
        for r in range(3, self.ROWS):
            for c in range(self.COLS - 3):
                window = np.array([board[r-i][c+i] for i in range(4)])
                if any(val in players for val in window):
                    patterns.append(window)
        return patterns

    def count_windows(self, patterns, count, player):
        return sum(
            1
            for w in patterns
            if np.count_nonzero(w == player) == count
            and np.count_nonzero(w == 0) == 4 - count
        )

    # ---------------------------- fast public API ----------------------------

    def get_heuristic(self, board, player):
        self._load_from_numpy(board)
        player = -1 if player == 2 else int(player)
        self._root_player = player
        return self._evaluate_leaf(self._root_player)

    def is_terminal(self, board):
        self._load_from_numpy(board)
        for window in self._windows:
            s = 0; z = 0
            for (r, c) in window:
                v = self._board[r][c]
                s += v; z += (v == 0)
            if z == 0 and (s == 4 or s == -4): return True
        return all(self._heights[c] >= self.ROWS for c in range(self.COLS))

    def minimax(self, board, depth, maximizing, player, alpha, beta):
        self._load_from_numpy(board)
        player = -1 if player == 2 else int(player)
        self._root_player = player

        to_move = player if maximizing else -player
        return self._ab_minimax(depth, to_move, float(alpha), float(beta))  # returns in root POV

    def n_step_lookahead(self, board, player, depth=3):
        self._load_from_numpy(board)
        player = -1 if player == 2 else int(player)
        self._root_player = player

        legal = self._legal_moves()
        if not legal: raise ValueError("n_step_lookahead: No valid moves available.")

        # Immediate winning move now? Take center-most.
        wins_now = []
        for c in legal:
            r = self._make_move(c, player)
            win = self._is_win_from(r, c, player)
            self._unmake_move(c)
            if win:
                wins_now.append(c)
        if wins_now:
            wins_now.sort(key=lambda x: abs(self.CENTER_COL - x))
            return wins_now[0]

        best_score = -float("inf")
        best_moves = []

        for col in self._ordered_moves(legal):
            r = self._make_move(col, player)
            if self._is_win_from(r, col, player):
                score = self._evaluate_leaf(self._root_player)
            else:
                score = self._ab_minimax(depth - 1, -player, -float("inf"), float("inf"))
            self._unmake_move(col)

            if score > best_score:
                best_score = score
                best_moves = [col]
            elif score == best_score:
                best_moves.append(col)

        best_moves.sort(key=lambda x: abs(self.CENTER_COL - x))
        K = min(3, len(best_moves))
        return random.choice(best_moves[:K])

    def has_four(self, board, player: int) -> bool:
        self._load_from_numpy(board)
        player = -1 if player == 2 else int(player)
        for window in self._windows:
            for (r, c) in window:
                if self._board[r][c] != player:
                    break
            else:
                return True
        return False

    # alias
    check_win = has_four

    def count_immediate_wins(self, board, player: int):
        self._load_from_numpy(board)
        player = -1 if player == 2 else int(player)
        wins = []
        for c in range(self.COLS):
            if self._heights[c] >= self.ROWS:
                continue
            r = self._make_move(c, player)
            win = self._is_win_from(r, c, player)
            self._unmake_move(c)
            if win:
                wins.append(c)
        return wins

    def compute_fork_signals(self, board_before, board_after, mover: int):
        mover = -1 if mover == 2 else int(mover)
        opp = -mover
        my_after = len(self.count_immediate_wins(board_after, mover))
        opp_before = len(self.count_immediate_wins(board_before, opp))
        opp_after = len(self.count_immediate_wins(board_after, opp))
        return {"my_after": my_after, "opp_before": opp_before, "opp_after": opp_after}

    # ----------------------- optimized alpha–beta core -----------------------

    def _ab_minimax(self, depth: int, to_move: int, alpha: float, beta: float) -> float:

        # Terminal/Leaf → evaluate in root POV 
        if depth == 0 or self._someone_has_four():
            return self._evaluate_leaf(self._root_player)

        legal = self._legal_moves()
        if not legal:
            return self._evaluate_leaf(self._root_player)

        maximizing = (to_move == self._root_player)

        # Early mate-in-1 cut: if the side-to-move can win immediately,
        # return the child's real heuristic in root POV.
        for c in legal:
            r = self._make_move(c, to_move)
            win = self._is_win_from(r, c, to_move)
            if win:
                score = self._evaluate_leaf(self._root_player)
                self._unmake_move(c)
                return score
            self._unmake_move(c)

        if maximizing:
            value = -float("inf")
            for col in self._ordered_moves(legal):
                r = self._make_move(col, to_move)
                if self._is_win_from(r, col, to_move):
                    child_val = self._evaluate_leaf(self._root_player)
                else:
                    child_val = self._ab_minimax(depth - 1, -to_move, alpha, beta)
                self._unmake_move(col)
                if child_val > value:
                    value = child_val
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break
            return value
        else:
            value = float("inf")
            for col in self._ordered_moves(legal):
                r = self._make_move(col, to_move)
                if self._is_win_from(r, col, to_move):
                    child_val = self._evaluate_leaf(self._root_player)
                else:
                    child_val = self._ab_minimax(depth - 1, -to_move, alpha, beta)
                self._unmake_move(col)
                if child_val < value:
                    value = child_val
                if value < beta:
                    beta = value
                if alpha >= beta:
                    break
            return value

    # --------------------------- fast evaluation ----------------------------
    

    def _evaluate_leaf(self, pov: int) -> float:
         # gravity-aware with soft discount for floating windows
         B = self._board
         DEF = self.DEFENSIVE
         FLOAT = self.FLOATING_FACTOR  # multiplier for windows with any unsupported empty
         w = (0.0, 0.0, float(self.weights[2]), float(self.weights[3]), float(self.weights[4]))
     
         score = 0.0
         for window in self._windows:
             p = o = 0
             supported = True
             for (r, c) in window:
                 v = B[r][c]
                 if v == 0:
                     if r > 0 and B[r - 1][c] == 0:
                         supported = False
                 elif v == pov:
                     p += 1
                 else:
                     o += 1
     
             mul = 1.0 if supported else FLOAT
             if o == 0: score += mul * w[p]
             elif p == 0: score -= mul * DEF * w[o]
     
         my_imm  = self._count_immediate_wins(pov)
         opp_imm = self._count_immediate_wins(-pov)
         score += self.immediate_w * (my_imm - DEF * opp_imm)
         if my_imm >= 2: score += self.fork_w * (my_imm - 1)
         if opp_imm >= 2: score -= DEF * (self.fork_w * (opp_imm - 1))
         return score

    # ----------------------- in-place board ops/checks -----------------------

    def _ordered_moves(self, legal: List[int]) -> List[int]:
        idx = {c: i for i, c in enumerate(self._CENTER_ORDER)}
        return sorted(legal, key=lambda c: idx[c])

    def _legal_moves(self) -> List[int]:
        return [c for c in self._CENTER_ORDER if self._heights[c] < self.ROWS]

    def _make_move(self, col: int, player: int) -> int:
        r = self._heights[col]
        self._board[r][col] = player
        self._heights[col] += 1
        return r

    def _unmake_move(self, col: int) -> None:
        self._heights[col] -= 1
        r = self._heights[col]
        self._board[r][col] = 0

    def _is_win_from(self, r: int, c: int, player: int) -> bool:
        return (
            self._line_len(r, c, 1, 0, player) >= 4 or
            self._line_len(r, c, 0, 1, player) >= 4 or
            self._line_len(r, c, 1, 1, player) >= 4 or
            self._line_len(r, c, 1, -1, player) >= 4
        )

    def _line_len(self, r: int, c: int, dr: int, dc: int, player: int) -> int:
        n = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < self.ROWS and 0 <= cc < self.COLS and self._board[rr][cc] == player:
            n += 1; rr += dr; cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < self.ROWS and 0 <= cc < self.COLS and self._board[rr][cc] == player:
            n += 1; rr -= dr; cc -= dc
        return n

    def _count_immediate_wins(self, player: int) -> int:
        cnt = 0
        for c in range(self.COLS):
            if self._heights[c] >= self.ROWS:
                continue
            r = self._make_move(c, player)
            win = self._is_win_from(r, c, player)
            self._unmake_move(c)
            if win:
                cnt += 1
        return cnt

    def _someone_has_four(self) -> bool:
        for window in self._windows:
            s = 0; z = 0
            for (r, c) in window:
                v = self._board[r][c]
                s += v; z += (v == 0)
            if z == 0 and (s == 4 or s == -4):
                return True
        return False

    # ------------------------------- setup -----------------------------------

    def _precompute_windows(self) -> List[List[Tuple[int, int]]]:
        W = []
        # horizontal
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                W.append([(r, c + i) for i in range(4)])
        # vertical
        for c in range(self.COLS):
            for r in range(self.ROWS - 3):
                W.append([(r + i, c) for i in range(4)])
        # diag up-right
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                W.append([(r + i, c + i) for i in range(4)])
        # diag up-left
        for r in range(self.ROWS - 3):
            for c in range(3, self.COLS):
                W.append([(r + i, c - i) for i in range(4)])
        return W

    def _load_from_numpy(self, board) -> None:
        """
        Convert top-based NumPy board -> bottom-based lists once.
        Accepts {0, 1, -1} or {0, 1, 2} (2→-1).
        """
        # reset
        for r in range(self.ROWS):
            row = self._board[r]
            for c in range(self.COLS):
                row[c] = 0
        for c in range(self.COLS):
            self._heights[c] = 0

        # map & flip
        for r_np in range(self.ROWS):
            for c in range(self.COLS):
                v = int(board[r_np, c])
                if v == 2: v = -1
                if v != 0:
                    r = self.ROWS - 1 - r_np
                    self._board[r][c] = v

        # heights
        for c in range(self.COLS):
            h = 0
            while h < self.ROWS and self._board[h][c] != 0:
                h += 1
            self._heights[c] = h
