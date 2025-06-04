# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:38:09 2022

@author: lovro
"""
import numpy as np
COLS = 7
ROWS = 6
INROW = 4


def boardToPatterns(grid):
    pats = boardDiagonals(grid)
    pats.extend(boardHorizontals(grid))
    pats.extend(boardHorizontals(grid.T))
    pats = list(filter(lambda x: x.count(0) <= 2, pats))
    return pats


def boardDiagonals(grid):
    diags = []
    for col in range(COLS - (INROW - 1)):
        for row in range(ROWS - (INROW - 1)):
            w = []
            for i in range(INROW):
                w.append(grid[row+i][col+i])
            diags.append(w)
            for row in range(INROW - 1, ROWS):
                w = []
                for i in range(INROW):
                    w.append(grid[row-i][col+i])
            diags.append(w)
    return diags


def boardHorizontals(grid):
    pats = []
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1] - (INROW - 1)):
            pats.append(list(grid[row, col:col+INROW]))
    return pats


def count_windows_in_pattern(patterns, num, piece):
    return sum([window.count(piece) == num and window.count(0) == INROW - num for window in patterns])

board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2,
         1, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 1, 2, 0, 2, 0]
grid = np.asarray(board).reshape(6, 7)
patterns = boardToPatterns(grid)

check = count_windows_in_pattern(patterns, 2, 1)
