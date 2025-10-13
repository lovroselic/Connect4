#opponent_action.py
import random
from C4.fast_connect4_lookahead import Connect4Lookahead
from DQN.training_phases_config import TRAINING_PHASES

Lookahead = Connect4Lookahead()

def get_opponent_action(env, agent, episode, next_state, player, depth, frozen_opp=None, phase=None):
        
    valid_actions = env.available_actions()
    if len(valid_actions) == 0:
        print(f"❌ [get_opponent_action] no valid actions {valid_actions}, step:  {env.ply}")
        raise ValueError("[get_opponent_action] no valid actions")
    
    if frozen_opp is not None:
        return frozen_opp.act(next_state, valid_actions, player=-1, depth=depth)

    board = agent.decode_board_from_state(next_state, player) # decode board from player's perpective

    # --- schedule ---
    phase_cfg = TRAINING_PHASES.get(phase, {})
    opponent_weights = phase_cfg.get("opponent", [1, 0, 0, 0, 0, 0, 0, 0])
    choice = random.choices(range(8), weights=opponent_weights)[0]

    if choice == 0:
        return random.choice(valid_actions)
    else:
        return Lookahead.n_step_lookahead(board, player=-1, depth=choice)


