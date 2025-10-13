# connect4_lookahead.py
import numpy as np
import random

class Connect4Lookahead:
    ROWS = 6
    COLS = 7
    CENTER_COL = 3
    immediate_w = 250
    fork_w = 150

    def __init__(self, weights=None):
        # Default heuristic weights for 2-in-a-row, 3-in-a-row, 4-in-a-row
        self.weights = weights if weights else {2: 10, 3: 100, 4: 1000}

    def get_legal_moves(self, board):
        order = [3,4,2,5,1,6,0]
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
        return sum(1 for w in patterns if np.count_nonzero(w == player) == count and np.count_nonzero(w == 0) == 4 - count)

    def get_heuristic(self, board, player):
        opponent = -player
        patterns = self.board_to_patterns(board, [player, opponent])
        score = 0
        for n in [2, 3, 4]:
            my_count = self.count_windows(patterns, n, player)
            opp_count = self.count_windows(patterns, n, opponent)
            score += self.weights[n] * (my_count - 1.5 * opp_count)
            
        # immediate wins / forks (computed only at leaf)
        my_imm  = len(self.count_immediate_wins(board, player))
        opp_imm = len(self.count_immediate_wins(board, opponent))
        
        score += self.immediate_w * (my_imm - 1.5 * opp_imm)
        
        # forks: >=2 immediate wins → bonus/penalty (scale with (count-1))
        if my_imm >= 2:
            score += self.fork_w * (my_imm - 1)
        if opp_imm >= 2:
            score -= 1.5 * self.fork_w * (opp_imm - 1)
        
        return score

    def is_terminal(self, board):
        patterns = self.board_to_patterns(board, [1, -1])
        return (self.count_windows(patterns, 4, 1) > 0 or
                self.count_windows(patterns, 4, -1) > 0 or
                np.all(board != 0))

    def minimax(self, board, depth, maximizing, player, alpha, beta):
        if depth == 0 or self.is_terminal(board): return self.get_heuristic(board, player)
        valid_moves = self.get_legal_moves(board)
        
        if maximizing:
            value = -np.inf
            for col in valid_moves:
                child = self.drop_piece(board, col, player)
                value = max(value, self.minimax(child, depth - 1, False, player, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = np.inf
            opponent = -player
            for col in valid_moves:
                child = self.drop_piece(board, col, opponent)
                value = min(value, self.minimax(child, depth - 1, True, player, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def n_step_lookahead(self, board, player, depth=3):
        valid_moves = self.get_legal_moves(board)
        
        if not valid_moves:
            raise ValueError("n_step_lookahead: No valid moves available — the board may be full or corrupted.")
    
        scores = {}
        for move in valid_moves:
            new_board = self.drop_piece(board, move, player)
            score = self.minimax(new_board, depth - 1, False, player, -np.inf, np.inf)
            scores[move] = score
    
        if not scores:
            raise ValueError("n_step_lookahead: Failed to compute scores — no moves were evaluated.")
    
        max_score = max(scores.values())
        best_moves = [m for m, s in scores.items() if s == max_score]
        
        if not best_moves:
            raise ValueError("n_step_lookahead: No best moves found despite valid scores — unexpected logic error.")
    
        best_moves.sort(key=lambda x: abs(self.CENTER_COL - x))  # prefer center
        K = min(3, len(best_moves))
        
        return random.choice(best_moves[:K])

    
    def has_four(self, board, player: int) -> bool:
        """True if 'player' has a 4-in-a-row on 'board'."""
        patterns = self.board_to_patterns(board, [player])
        return self.count_windows(patterns, 4, player) > 0
    
    #alias
    check_win = has_four
    
    def count_immediate_wins(self, board, player: int):
        """
        Return list of columns that produce an immediate win for 'player'
        (i.e., drop once and you win).
        """
        wins = []
        for col in self.get_legal_moves(board):
            b2 = self.drop_piece(board, col, player)
            if self.has_four(b2, player):
                wins.append(col)
        return wins
    
    def compute_fork_signals(self, board_before, board_after, mover: int):
        """
        For the move just played by 'mover':
          - my_after: # of mover's immediate wins after the move
          - opp_before: # of opponent's immediate wins before the move
          - opp_after:  # of opponent's immediate wins after the move
        """
        opp = -mover
        my_after   = len(self.count_immediate_wins(board_after, mover))
        opp_before = len(self.count_immediate_wins(board_before, opp))
        opp_after  = len(self.count_immediate_wins(board_after,  opp))
        return {"my_after": my_after, "opp_before": opp_before, "opp_after": opp_after}
    
