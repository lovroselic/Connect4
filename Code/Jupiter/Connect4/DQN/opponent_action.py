# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 15:15:37 2025

@author: Lovro
"""
import random
from C4.connect4_lookahead import Connect4Lookahead
from DQN.training_phases_config import TRAINING_PHASES

Lookahead = Connect4Lookahead()

def get_opponent_action(env, agent, episode, next_state, player, depth):
    valid_actions = env.available_actions()

    if episode < TRAINING_PHASES["Random1"]["length"]:
        opp_action = random.choice(valid_actions)

    elif episode < TRAINING_PHASES["Random2"]["length"]:
        opp_action = random.choice(valid_actions)

    elif episode < TRAINING_PHASES["Mixed_RR1"]["length"]:
        if random.random() < 0.75:
            opp_action = random.choice(valid_actions)
        else:
            opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=1)

    elif episode < TRAINING_PHASES["Mixed_R1"]["length"]:
        if random.random() < 0.5:
            opp_action = random.choice(valid_actions)
        else:
            opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=1)

    elif episode < TRAINING_PHASES["Mixed_R11"]["length"]:
        if random.random() < 0.3:
            opp_action = random.choice(valid_actions)
        else:
            opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=1)

    elif episode < TRAINING_PHASES["SelfPlay_Begin"]["length"]:
        opp_action = agent.act_with_curriculum(next_state, valid_actions, player=-1, depth=depth)

    ####################################



    elif episode < TRAINING_PHASES["Fixed1"]["length"]:
        opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=1)

    elif episode < TRAINING_PHASES["Fixed1-Reprise"]["length"]:
        opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=1)

    elif episode < TRAINING_PHASES["SelfPlay_F1"]["length"]:
        opp_action = agent.act_with_curriculum(next_state, valid_actions, player=-1, depth=depth)

    # ######################################

    elif episode < TRAINING_PHASES["Mixed_R12"]["length"]:
        choice = random.randint(0,2)
        if choice == 0:
            opp_action = random.choice(valid_actions)
        elif choice == 1:
            opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=1)
        else:
            opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=2)

    elif episode < TRAINING_PHASES["Mixed_R1122"]["length"]:
        choice = random.randint(0,4)
        if choice == 0:
            opp_action = random.choice(valid_actions)
        elif choice <= 2:
            opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=1)
        else:
            opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=2)

    elif episode < TRAINING_PHASES["Mixed12"]["length"]:
        shallow_depth = random.randint(1, 2)
        opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=shallow_depth)

    elif episode < TRAINING_PHASES["SelfPlay_L1L2"]["length"]:
        opp_action = agent.act_with_curriculum(next_state, valid_actions, player=-1, depth=depth)

    # ######################################

    elif episode < TRAINING_PHASES["Mixed123"]["length"]:
        depth = random.choice([1,2,3])
        opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=depth)

    elif episode < TRAINING_PHASES["Fixed2"]["length"]:
        opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=2)

    elif episode < TRAINING_PHASES["Fixed2_Reprise"]["length"]:
        opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=2)

    elif episode < TRAINING_PHASES["SelfPlay_L2"]["length"]:
        opp_action = agent.act_with_curriculum(next_state, valid_actions, player=-1, depth=depth)

    # ######################################

    # elif episode < TRAINING_PHASES["Variable23"]["length"]:
    #     variable_depth = random.randint(2, 3)
    #     opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=variable_depth)

    # elif episode < TRAINING_PHASES["Variable3"]["length"]:
    #     variable_depth = random.randint(1, 3)
    #     opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=variable_depth)

    # elif episode < TRAINING_PHASES["SelfPlay_L3"]["length"]:
    #     opp_action = agent.act_with_curriculum(next_state, valid_actions, player=-1, depth=depth)

    ######################################

    # elif episode < TRAINING_PHASES["Fixed3"]["length"]:
    #     opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=3)

    # elif episode < TRAINING_PHASES["Variable34"]["length"]:
    #     variable_depth = random.randint(3, 4)
    #     opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=variable_depth)

    # elif episode < TRAINING_PHASES["Fixed4"]["length"]:
    #     opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=4)

    # elif episode < TRAINING_PHASES["Variable35"]["length"]:
    #     variable_depth = random.randint(3, 5)
    #     opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=variable_depth)

    # elif episode < TRAINING_PHASES["SelfPlay_L4"]["length"]:
    #     opp_action = agent.act(next_state, valid_actions, player=-1, depth=depth)

    ######################################

    # elif episode < TRAINING_PHASES["Full"]["length"]:
    #     opp_action = Lookahead.n_step_lookahead(next_state, player=-1, depth=depth)

    else:
        opp_action = agent.act_with_curriculum(next_state, valid_actions, player=-1, depth=depth)

    return opp_action
