#opponent_action.py
import random
from C4.fast_connect4_lookahead import Connect4Lookahead
from DQN.training_phases_config import TRAINING_PHASES

Lookahead = Connect4Lookahead()

def get_opponent_action(env, agent, episode, next_state, depth, frozen_opp=None, phase=None):
    agent._opp_act_total += 1
        
    valid_actions = env.available_actions()
    if len(valid_actions) == 0: raise ValueError(f"❌ [get_opponent_action] no valid actions {valid_actions}, step:  {env.ply}")
    
    if frozen_opp is not None: return frozen_opp.act(next_state, valid_actions, player=-1, depth=depth)

    board = agent.decode_board_from_state(next_state, player=+1) # decode board from agents's perpective!

    # --- schedule ---
    phase_cfg = TRAINING_PHASES.get(phase, {})
    opponent_weights = phase_cfg.get("opponent", [1, 0, 0, 0, 0, 0, 0, 0])
    choice = random.choices(range(8), weights=opponent_weights)[0]

    if choice == 0:
        agent._bump_opp("rand")
        return random.choice(valid_actions)
    else:
        agent._bump_opp(f"L{choice}")
        return Lookahead.n_step_lookahead(board, player=-1, depth=choice)


