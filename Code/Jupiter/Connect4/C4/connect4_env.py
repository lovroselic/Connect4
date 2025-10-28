# connect4_env.py
import numpy as np
from C4.fast_connect4_lookahead import Connect4Lookahead

class Connect4Env:
    # ENV is mover centric
    # reward is for the mover, from mower's perspective
    ROWS = 6
    COLS = 7
    
    #reward constants
    THREAT2_VALUE = 10  
    THREAT3_VALUE = 20   
    BLOCK2_VALUE = 12   
    BLOCK3_VALUE = 25    
    MAX_REWARD = 80 
    WIN_REWARD = 100
    DRAW_REWARD = 15
    LOSS_PENALTY = -100
    CENTER_REWARD = 1.0
    CENTER_REWARD_BOTTOM = 50 #18
    FORK_BONUS = 50  #18
    BLOCK_FORK_BONUS = 55 #22
    OPP_IMMEDIATE_PENALTY = 70 
    STEP_PENALTY = 1
    CENTER_WEIGHTS = [0.25, 0, 0.5, 1.0, 0.5, 0, 0.25]
    OPENING_DECAY_STEPS = 6  
    

    def __init__(self):
        self.reset()
        self.lookahead = Connect4Lookahead()
        
    def reset(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.ply = 0
        return self.get_state(perspective=self.current_player) 

    def get_state(self, perspective=None) -> np.ndarray:                          #mover's perspective
        if perspective is None: perspective = self.current_player
        board = self.board
        player = perspective
    
        agent_plane = (board == player).astype(np.float32)
        opp_plane = (board == -player).astype(np.float32)
        row_plane = np.tile(np.linspace(-1, 1, self.ROWS)[:, None], (1, self.COLS))
        col_plane = np.tile(np.linspace(-1, 1, self.COLS)[None, :], (self.ROWS, 1))
    
        return np.stack([agent_plane, opp_plane, row_plane, col_plane])

    def available_actions(self):
        return [c for c in range(self.COLS) if self.board[0][c] == 0]

    def step(self, action):
        if self.done: return self.get_state(perspective=self.current_player), 0.0, True
        if self.board[0][action] != 0: raise ValueError("ILLEGAL MOVE DETECTED!")

        # Save board state BEFORE move for threat detection
        board_before = self.board.copy()
        opponent_before = -self.current_player
        
        # Drop piece in selected column
        for row in reversed(range(self.ROWS)):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                placed_row = row
                break
        

        reward = 0
        threat2_count = 0
        threat3_count = 0
        block2_count = 0
        block3_count = 0
        
        # Check game status
        self.done, self.winner = self.check_game_over()
        
        if not self.done:
            
            # Fast counts via precomputed windows
            threat2_count = self.lookahead.count_pure(self.board, self.current_player, 2)
            threat3_count = self.lookahead.count_pure(self.board, self.current_player, 3)
            block2_count  = self.lookahead.count_pure_block_delta(board_before, self.board, opponent_before, 2)
            block3_count  = self.lookahead.count_pure_block_delta(board_before, self.board, opponent_before, 3)
            
            center_reward = self.CENTER_REWARD * self.CENTER_WEIGHTS[action]
            
            # Extra if it is **bottom-center** 
            if action == 3 and placed_row == self.ROWS - 1:
                opening_decay = np.exp(-self.ply / self.OPENING_DECAY_STEPS)
                bonus = self.CENTER_REWARD_BOTTOM * (2.0 if self.ply == 0 else opening_decay)
                center_reward += bonus
                
            # --- Fork / anti-fork shaping using Lookahead ---
            signals = self.lookahead.compute_fork_signals(board_before, self.board, self.current_player)
            fork_bonus = self.FORK_BONUS if signals["my_after"] >= 2 else 0
            blocked_fork = (signals["opp_before"] >= 2) and (signals["opp_after"] < signals["opp_before"])
            block_fork_bonus = self.BLOCK_FORK_BONUS if blocked_fork else 0
            
            # do NOT leave opponent a 1-move win (can be >1 if you blunder badly)
            opp_immediate_after = signals["opp_after"]
            immediate_loss_penalty = self.OPP_IMMEDIATE_PENALTY * opp_immediate_after

            # CALCULATE SCALED REWARDS
            threat_reward = (self.THREAT2_VALUE * threat2_count) + (self.THREAT3_VALUE * threat3_count)
            block_reward  = (self.BLOCK2_VALUE * block2_count) + (self.BLOCK3_VALUE * block3_count)
            
            reward = (
                threat_reward
                + block_reward
                + fork_bonus
                + block_fork_bonus
                + center_reward
                - immediate_loss_penalty        
            )
            
            reward = np.clip(reward, -self.MAX_REWARD, self.MAX_REWARD)
            reward -= self.STEP_PENALTY
            
            self.current_player *= -1   # Switch turns, KEEP! even thogh it 0s probably obsolete;
            self.ply += 1

        else:  # Terminal rewards
            if self.winner == self.current_player:  reward = self.WIN_REWARD
            elif self.winner == 0:                  reward = self.DRAW_REWARD
            else:                                   reward = self.LOSS_PENALTY
                
            
        return self.get_state(perspective=self.current_player), reward, self.done

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
