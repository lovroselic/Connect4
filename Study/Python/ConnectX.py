# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:38:09 2022

@author: lovro
"""
import numpy as np

def N_step_lookahead(obs, config):
    import numpy as np
    N_STEPS = 3

    ########################################################

    def score_move(grid, col, mark, config, nsteps):
        next_grid = drop_piece(grid, col, mark, config)
        score = minimax(next_grid, nsteps-1, False, mark, config, -np.Inf, np.Inf)
        return score

    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    def get_heuristic(grid, mark, config):
        A = 1e6
        B = 1e2
        C = 1
        D = -1
        E = -1e2
        F = -1e6
        num_twos = count_windows(grid, 2, mark, config)
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_twos_opp = count_windows(grid, 2, mark % 2 + 1, config)
        num_threes_opp = count_windows(grid, 3, mark % 2 + 1, config)
        num_fours_opp = count_windows(grid, 4, mark % 2 + 1, config)
        score = A*num_fours + B*num_threes + C*num_twos + D*num_twos_opp + E*num_threes_opp +F *num_fours_opp
        return score

    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow - num_discs)

    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col + config.inarow])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row + config.inarow, col])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row + config.inarow), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        return num_windows
    

    def is_terminal_window(window, config):
        return window.count(1) == config.inarow or window.count(2) == config.inarow

    def is_terminal_node(grid, config):
        # Check for draw 
        if list(grid[0, :]).count(0) == 0:
            return True
        # Check for win: horizontal, vertical, or diagonal
        # horizontal 
        for row in range(config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[row, col:col + config.inarow])
                if is_terminal_window(window, config):
                    return True
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row + config.inarow, col])
                if is_terminal_window(window, config):
                    return True
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
                if is_terminal_window(window, config):
                    return True
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
                if is_terminal_window(window, config):
                    return True
        return False
    
    ########################################################
    
    # Minimax, ab pruning
    def minimax(node, depth, maximizingPlayer, mark, config, A, B):
        if depth == 0 or is_terminal_node(node, config):
            return get_heuristic(node, mark, config)
        
        valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
        
        if maximizingPlayer:
            value = -np.Inf
            for col in valid_moves:
                child = drop_piece(node, col, mark, config)
                value = max(value, minimax(child, depth-1, False, mark, config, A, B))
                if value >= B:
                    break
                A = max(A, value)
            return value
        else:
            value = np.Inf
            for col in valid_moves:
                child = drop_piece(node, col, mark%2+1, config)
                value = min(value, minimax(child, depth-1, True, mark, config, A, B))
                if value <= A:
                    break
                B = min(B, value)
            return value
        
    def innermost(arr):
        mid = (config.columns - 1) / 2
        distance = [-abs(c-mid) for c in arr]
        return arr[np.argmax(distance)]

    ########################################################

    order = [config.columns//2 - i//2 - 1 if i%2 else config.columns//2 + i//2 for i in range(config.columns)]
    valid_moves = [c for c in order if obs.board[c] == 0]
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    return innermost(max_cols)

board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 1, 2, 0, 2, 0]
grid = np.asarray(board).reshape(6, 7)
H = str(grid)

V = str(grid.transpose())
diagonal_str = ""
for col in range(0,4):
    for row in range(0,3):    
        for i in range(4):
            diagonal_str+=str(grid[row+i][col+i])+" "
        diagonal_str+=","
    for row in range(3,6):    
        for i in range(4):
            diagonal_str+=str(grid[row-i][col+i])+" "
        diagonal_str+=","
