#opponent_action.py
import random
from C4.connect4_lookahead import Connect4Lookahead
from DQN.training_phases_config import TRAINING_PHASES

Lookahead = Connect4Lookahead()

def get_opponent_action(env, agent, episode, next_state, player, depth, frozen_opp=None, phase=None):
    AGENT = 1
    
    valid_actions = env.available_actions()
    if not valid_actions:
        return 0
    
    board = agent.decode_board_from_state(next_state, AGENT) # decode board from agent's perpective

    if frozen_opp is not None:
        return frozen_opp.act(next_state, valid_actions, player=-1, depth=depth)


    # --- schedule ---
    phase_cfg = TRAINING_PHASES.get(phase, {})
    opponent_weights = phase_cfg.get("opponent", [1, 0, 0, 0, 0, 0, 0, 0])
    choice = random.choices(range(8), weights=opponent_weights)[0]

    if choice == 0:
        return random.choice(valid_actions)
    else:
        return Lookahead.n_step_lookahead(board, player=-1, depth=choice)


