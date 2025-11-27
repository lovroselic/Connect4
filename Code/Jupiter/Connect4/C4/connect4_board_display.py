# board_display.py
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from DQN.eval_utilities import play_single_game
from PPO.ppo_agent_eval import play_single_game_ppo

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


def display_final_boards(agent, env, device, Lookahead, opponents):
    for label in opponents:
        print(f"\n🎯 Opponent: {label}")

        for game_index in (0, 1):  # 0: agent first, 1: opponent first
            outcome, _ = play_single_game(agent, env, device, Lookahead, label, game_index)

            title = (
                f"{label} — {'Agent First' if game_index == 0 else 'Opponent First'} — "
                f"{'Win' if outcome == 1 else 'Loss' if outcome == -1 else 'Draw'}"
            )
            
            Connect4_BoardDisplayer.display_board(env.board, title)
            
def display_final_boards_PPO(policy, opponents, seed: int = 666):
    """
    Show two final boards per opponent label: agent-first (seed) and opponent-first (seed+1).
    Uses play_single_game_ppo() and Connect4_BoardDisplayer.display_board(...).
    """
    for label in opponents:
        print(f"\n🎯 Opponent: {label}")
        for who_first, s in (("Agent first", seed), ("Opponent first", seed + 1)):
            outcome, final_board = play_single_game_ppo(policy, opponent_label=label, seed=s)
            title = f"{label} — {who_first} — {'Win' if outcome==1.0 else 'Loss' if outcome==-1.0 else 'Draw'}"
            Connect4_BoardDisplayer.display_board(final_board, title=title)



