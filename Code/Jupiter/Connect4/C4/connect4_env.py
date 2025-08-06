# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 14:01:51 2025

@author: Uporabnik
"""

import numpy as np

class Connect4Env:
    ROWS = 6
    COLS = 7

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return self.board.copy()

    def available_actions(self):
        return [c for c in range(self.COLS) if self.board[0][c] == 0]

    def step(self, action):
        if self.done or self.board[0][action] != 0:
            return self.get_state(), -10, True  # Illegal move penalty
    
        # Drop piece in the selected column
        for row in reversed(range(self.ROWS)):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                break
    
        reward = 0
        self.done, winner = self.check_game_over()
        self.winner = winner  # ✅ Store winner in env
    
        if self.done:
            if winner == self.current_player:
                reward = 1  # Win
            elif winner == 0:
                reward = 0.5  # Draw
            else:
                reward = -1  # Loss
    
        self.current_player *= -1  # Switch turns
        return self.get_state(), reward, self.done

    def check_game_over(self):
        # Horizontal
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                line = self.board[r, c:c+4]
                if abs(np.sum(line)) == 4:
                    return True, int(np.sign(np.sum(line)))

        # Vertical
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                line = self.board[r:r+4, c]
                if abs(np.sum(line)) == 4:
                    return True, int(np.sign(np.sum(line)))

        # Diagonal /
        for r in range(3, self.ROWS):
            for c in range(self.COLS - 3):
                line = [self.board[r-i][c+i] for i in range(4)]
                if abs(np.sum(line)) == 4:
                    return True, int(np.sign(np.sum(line)))

        # Diagonal \
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                line = [self.board[r+i][c+i] for i in range(4)]
                if abs(np.sum(line)) == 4:
                    return True, int(np.sign(np.sum(line)))

        # Draw
        if not np.any(self.board == 0):
            return True, 0

        return False, None
