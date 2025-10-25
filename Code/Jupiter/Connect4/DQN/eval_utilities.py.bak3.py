# DQN/eval_utilities.py

import time
import random
import numpy as np
import os
import pandas as pd
import re
from tqdm.auto import tqdm

# ----------------------------- helpers -----------------------------

def _valid_actions_of(env):
    return env.available_actions()

def _mask_argmax(q, valid_actions):
    q = np.asarray(q, dtype=np.float32)
    mask = np.full_like(q, -1e9)
    for a in valid_actions:
        mask[a] = 0.0
    a = int(np.argmax(q + mask))
    return a if a in valid_actions else random.choice(valid_actions)

def _parse_lookahead_depth(label: str, default: int = 2) -> int:
    if label.startswith("Lookahead"):
        m = re.search(r"(\d+)", label)
        if m: return int(m.group(1))
    return default

# ----------------------------- core eval -----------------------------

def _agent_q_action(agent, state_abs, valid_actions):
    """
    Pure Q argmax for the +1 agent on mover-first planes.
    state_abs is absolute planes from env.get_state(): [ +1 plane, -1 plane, row, col ].
    Our agent is always the +1 color, so mover-first == [+1, -1, row, col] on agent turns.
    """
    # orient to mover-first planes (agent is +1)
    oriented = agent._agent_planes(state_abs, +1)       # (4,6,7) – this is just state_abs reordered; safe
    q = agent._q_values_np(oriented)                    # (7,)
    return _mask_argmax(q, valid_actions)

def _opp_action(env, state_abs, Lookahead, label, depth, rng):
    valid = env.available_actions()
    if label == "Random": return rng.choice(valid)
    # opponent plays -1 on the absolute board
    # decode board from +1 perspective (absolute): +1 are agent stones, -1 are opponent stones
    board = state_abs[0] - state_abs[1]
    a = Lookahead.n_step_lookahead(board, player=-1, depth=depth)
    return a if a in valid else rng.choice(valid)

def play_single_game(agent, env, device, Lookahead, opponent_label, game_index):
    """
    Agent is ALWAYS +1 (same convention as training).
    We alternate who STARTS by giving the starter exactly one move before the loop.
    Then we explicitly set current_player=+1 for the agent move and current_player=-1 for the opponent move
    every turn. We do not rely on env's internal flip for control flow.
    """
    AGENT_COLOR, OPP_COLOR = +1, -1
    depth = _parse_lookahead_depth(opponent_label, default=2)
    rng = random.Random()

    # --- disable all stochastic/bias knobs for eval ---
    pc_backup, tol_backup = agent.prefer_center, agent.tie_tol
    sym_backup = agent.use_symmetry_policy
    eps_bak, epsmin_bak = agent.epsilon, agent.epsilon_min

    agent.prefer_center = False
    agent.tie_tol = 0.0
    agent.use_symmetry_policy = False
    agent.epsilon = 0.0
    agent.epsilon_min = 0.0

    # --- fresh game ---
    state = env.reset()  # absolute planes (+1, -1, row, col)
    done = False

    # --- alternate who starts ---
    agent_starts = (game_index % 2 == 0)
    if not agent_starts:
        # Opponent makes exactly one opening move
        env.current_player = OPP_COLOR
        a_opp = _opp_action(env, state, Lookahead, opponent_label, depth, rng)
        state, _, done = env.step(a_opp)

    # --- main loop: agent, then opp, repeat ---
    while not done:
        # Agent move (explicitly set mover)
        env.current_player = AGENT_COLOR
        a_agent = _agent_q_action(agent, state, env.available_actions())
        state, _, done = env.step(a_agent)
        if done: break

        # Opponent move
        env.current_player = OPP_COLOR
        a_opp = _opp_action(env, state, Lookahead, opponent_label, depth, rng)
        state, _, done = env.step(a_opp)
        if done: break

    # outcome (from agent's perspective)
    if env.winner == AGENT_COLOR: outcome = 1.0
    elif env.winner == OPP_COLOR: outcome = -1.0
    else: outcome = 0.5

    # --- restore agent knobs ---
    agent.prefer_center, agent.tie_tol = pc_backup, tol_backup
    agent.use_symmetry_policy = sym_backup
    agent.epsilon, agent.epsilon_min = eps_bak, epsmin_bak

    return outcome, state

def evaluate_agent_model(agent, env, evaluation_opponents, device, Lookahead):
    """
    Evaluates agent.model vs opponents.
    Uses explicit mover setting each turn for correctness.
    """
    results = {}

    # Backup train/eval modes
    model_mode = agent.model.training
    target_mode = agent.target_model.training
    agent.model.eval()
    agent.target_model.eval()

    for label, num_games in evaluation_opponents.items():
        wins = losses = draws = 0
        with tqdm(total=num_games, desc=f"Opponent: {label}", position=1, leave=False) as pbar:
            for game_index in range(num_games):
                outcome, _ = play_single_game(agent, env, device, Lookahead, label, game_index)
                if outcome == 1.0: wins += 1
                elif outcome == -1.0: losses += 1
                else: draws += 1
                pbar.update(1)
            pbar.clear()

        results[label] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": round(wins / num_games, 3),
            "loss_rate": round(losses / num_games, 3),
            "draw_rate": round(draws / num_games, 3),
        }

    # Restore modes
    agent.model.train(model_mode)
    agent.target_model.train(target_mode)
    return results

def log_phase_evaluation(agent, env, phase_name, episode, device, Lookahead,
                         evaluation_opponents, excel_path):
    if phase_name is None:
        return
    print(f"📊 Running evaluation after phase: {phase_name} (ep {episode})")
    start_time = time.time()

    results = evaluate_agent_model(agent, env, evaluation_opponents, device, Lookahead)

    flat = {
        "TRAINING_SESSION": f"{phase_name}-EP-{episode}",
        "TIME [h]": round((time.time() - start_time) / 3600, 6),
        "EPISODES": episode,
    }
    for label, metrics in results.items():
        flat[label] = metrics["win_rate"]

    df_new = pd.DataFrame([flat])
    if os.path.exists(excel_path):
        try:
            df_existing = pd.read_excel(excel_path)
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_all = df_new
    else:
        df_all = df_new

    df_all.to_excel(excel_path, index=False)
    print(f"✅ Evaluation results saved to: {excel_path}")

# ----------------------------- optional sanity check -----------------------------

def sanity_check_random_vs_random(env, n_games=200):
    """
    Pure environment sanity: +1 random vs −1 random with strict alternation.
    Expected overall score ~0.5±noise (first-move advantage cancels across alternation).
    """
    wins = losses = draws = 0
    rng = random.Random()
    for g in range(n_games):
        state = env.reset()
        done = False
        agent_starts = (g % 2 == 0)

        if not agent_starts:  # opp opens
            env.current_player = -1
            a = rng.choice(env.available_actions())
            state, _, done = env.step(a)

        while not done:
            env.current_player = +1
            a = rng.choice(env.available_actions())
            state, _, done = env.step(a)
            if done: break

            env.current_player = -1
            a = rng.choice(env.available_actions())
            state, _, done = env.step(a)

        if env.winner == +1: wins += 1
        elif env.winner == -1: losses += 1
        else: draws += 1

    total = float(n_games)
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins/total,
        "loss_rate": losses/total,
        "draw_rate": draws/total,
        "score_rate": (wins + 0.5*draws)/total
    }
