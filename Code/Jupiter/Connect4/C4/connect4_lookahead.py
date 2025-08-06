# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 14:26:37 2025

@author: Uporabnik
"""

import numpy as np

class Connect4Lookahead:
    ROWS = 6
    COLS = 7
    CENTER_COL = 3

    def __init__(self, weights=None):
        # Default heuristic weights for 2-in-a-row, 3-in-a-row, 4-in-a-row
        self.weights = weights if weights else {2: 10, 3: 100, 4: 1000}

    def get_legal_moves(self, board):
        return [c for c in range(self.COLS) if board[0][c] == 0]

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
        return score

    def is_terminal(self, board):
        patterns = self.board_to_patterns(board, [1, -1])
        return (self.count_windows(patterns, 4, 1) > 0 or
                self.count_windows(patterns, 4, -1) > 0 or
                np.all(board != 0))

    def minimax(self, board, depth, maximizing, player, alpha, beta):
        if depth == 0 or self.is_terminal(board):
            return self.get_heuristic(board, player)

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
        scores = {}
        for move in valid_moves:
            new_board = self.drop_piece(board, move, player)
            score = self.minimax(new_board, depth - 1, False, player, -np.inf, np.inf)
            scores[move] = score

        max_score = max(scores.values())
        best_moves = [m for m, s in scores.items() if s == max_score]
        best_moves.sort(key=lambda x: abs(self.CENTER_COL - x))  # prefer center
        return best_moves[0]
