# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:10:53 2025

@author: Uporabnik
"""

# board_display.py
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

class Connect4_BoardDisplayer:
    @staticmethod
    def display_board(board, title="Final Board — Agent: O, Opponent: X"):
        """
        Displays the given Connect-4 board using matplotlib table.

        Args:
            board (np.ndarray): 2D array of shape (6, 7), values in {1, -1, 0}
            title (str): Title for the plot
        """
        if not isinstance(board, np.ndarray) or board.shape != (6, 7):
            raise ValueError("Board must be a 6x7 numpy array")

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title(title)
        ax.axis('off')

        # Build symbol table
        board_display = np.full_like(board, '', dtype=object)
        for r in range(6):
            for c in range(7):
                if board[r, c] == 1:
                    board_display[r, c] = 'O'
                elif board[r, c] == -1:
                    board_display[r, c] = 'X'

        # Color mapping
        cell_colors = [
            ['#e0ffe0' if cell == 'O' else '#ffe0e0' if cell == 'X' else '#f0f0f0' for cell in row]
            for row in board_display
        ]

        # Create table
        table = ax.table(
            cellText=board_display,
            cellLoc='center',
            loc='center',
            cellColours=cell_colors
        )
        table.scale(1, 1.5)
        table.auto_set_font_size(False)
        table.set_fontsize(14)

        display(fig)
        plt.close(fig)
