#DQN.loop_helpers.py

from DQN.opponent_action import get_opponent_action
from DQN.dqn_utilities import record_first_move, evaluate_final_result, map_final_result_to_reward

def opponent_move_step(env, agent, episode, state, *, phase, frozen_opp, lookahead_depth,nstep_buf, ply_idx, opening_kpis, openings, env_steps):
    """Single opponent turn: pre-step upkeep, choose action (frozen or scripted), env.step, N-step append, first-move bookkeeping."""
    env.current_player = -1
    env_steps += 1
    
    agent.maybe_update_target(env_steps)

    opp_action = get_opponent_action(env, agent, episode, state, depth=lookahead_depth, frozen_opp=frozen_opp,phase=phase)
    next_state, r_opp, _ = env.step(opp_action)
    nstep_buf.append(state, opp_action, r_opp, next_state, env.done, player=-1)

    # First-move stats (opponent started)
    if ply_idx == 0:
        record_first_move(opening_kpis, opp_action, who="opp")
        openings.on_first_move(int(opp_action), is_agent=False)

    return next_state, env.done, ply_idx + 1, env_steps

def agent_move_step(env, agent, state, *, ply_idx, lookahead_depth, strategy_weights, nstep_buf, opening_kpis, openings, env_steps):
    """Single agent turn: pre-step upkeep, choose action, env.step, N-step append, first-move bookkeeping."""
    env.current_player = +1
    env_steps += 1
    agent.maybe_update_target(env_steps)

    valid_actions = env.available_actions()
    if not valid_actions: raise ValueError(f"No valid actions in agent turn, done={env.done}, step={env.ply}")

    action = agent.act(state, valid_actions, depth=lookahead_depth, strategy_weights=strategy_weights, ply=ply_idx)
    next_state, r_agent, _ = env.step(action)

    # N-step memory
    nstep_buf.append(state, action, r_agent, next_state, env.done, player=+1)

    # First-move stats (agent started)
    if ply_idx == 0:
        record_first_move(opening_kpis, action, who="agent")
        openings.on_first_move(int(action), is_agent=True)

    return next_state, env.done, ply_idx + 1, env_steps


def maybe_opponent_opening(env, agent, episode, state, *, phase, frozen_opp, lookahead_depth,nstep_buf, ply_idx, opening_kpis, openings, env_steps):
    """If opponent starts (odd episodes), play exactly one opponent turn."""
    if episode % 2 == 1:
        
        return opponent_move_step(
            env, agent, episode, state,
            phase=phase, frozen_opp=frozen_opp, lookahead_depth=lookahead_depth,
            nstep_buf=nstep_buf, ply_idx=ply_idx, opening_kpis=opening_kpis,
            openings=openings, env_steps=env_steps
        )
    
    return state, ply_idx, env_steps

def finalize_if_done(env):
    """Returns (is_done, final_result, total_reward)."""
    if env.done:
        final_result = evaluate_final_result(env, agent_player=+1)
        total_reward = map_final_result_to_reward(final_result)
        return True, final_result, total_reward
    return False, None, 0.0