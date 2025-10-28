# DQN/eval_utilities.py

import time
import random
import numpy as np
from tqdm.auto import tqdm
import os
import pandas as pd
import re


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


def play_single_game(agent, env, device, Lookahead, opponent_label, game_index, debug = False, debug_depth = 1):
    
    # backup + force “pure” argmax behavior
    pc_backup, tol_backup = agent.prefer_center, agent.tie_tol
    sym_backup = agent.use_symmetry_policy
    agent.prefer_center = False
    agent.tie_tol = 0.0
    agent.use_symmetry_policy = False
    
    AGENT_SIDE, OPPONENT_SIDE = +1, -1
    depth = _parse_lookahead_depth(opponent_label, default=2)
    state = env.reset()  # absolute planes ( +1 plane, -1 plane, row, col )
    agent_starts = (game_index % 2 == 0)
    done = False

    rng = random.Random()

    while not done:
        valid_actions = env.available_actions()
        side = env.current_player
        board = agent.decode_board_from_state(state, player=AGENT_SIDE)

        if side == AGENT_SIDE:
            if debug:
                if debug_depth == 0:
                    action = rng.choice(valid_actions)
                else:
                    action = Lookahead.n_step_lookahead(board, player=AGENT_SIDE, depth=debug_depth)
            else: 
                oriented = agent._agent_planes(state, AGENT_SIDE)         # (4,6,7)
                q = agent._q_values_np(oriented)                          # (7,)
                action = _mask_argmax(q, valid_actions)
        else:
            if opponent_label == "Random":
                action = rng.choice(valid_actions)
            else:
                action = Lookahead.n_step_lookahead(board, player=OPPONENT_SIDE, depth=depth)
                if action not in valid_actions:
                    action = rng.choice(valid_actions)

        state, _, done = env.step(action)

    # restore
    agent.prefer_center, agent.tie_tol = pc_backup, tol_backup
    agent.use_symmetry_policy = sym_backup

    if env.winner == AGENT_SIDE:   outcome = 1
    elif env.winner == OPPONENT_SIDE: outcome = -1
    else: outcome = 0.5
    return outcome, state

def evaluate_agent_model(agent, env, evaluation_opponents, device, Lookahead, debug = False, debug_depth = 1):
    """
    Evaluates agent.model vs various scripted opponents.
    Each game is played from fresh env.reset(), alternating starting player.
    """
    
    if debug: print(f"DEBUG run, depth {debug_depth}")
    
    results = {}

    # Backup agent state
    model_mode = agent.model.training
    target_mode = agent.target_model.training
    _eps, _epsmin = agent.epsilon, agent.epsilon_min
    pc_backup, tol_backup = agent.prefer_center, agent.tie_tol
    
    agent.prefer_center = False         # uniform among ties
    agent.tie_tol = 0.0                 # fewer ties; true argmax
    agent.model.eval()
    agent.target_model.eval()
    agent.epsilon = 0.0
    agent.epsilon_min = 0.0


    for label, num_games in evaluation_opponents.items():
        wins = losses = draws = 0

        with tqdm(total=num_games, desc=f"Opponent: {label}", position=1, leave=False) as pbar:
            for game_index in range(num_games):
                outcome, _ = play_single_game(agent, env, device, Lookahead, label, game_index, debug = None, debug_depth = 1)
                if outcome == 1: wins += 1
                elif outcome == -1: losses += 1
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
      

    # Restore agent state
    agent.model.train(model_mode)
    agent.target_model.train(target_mode)
    agent.epsilon = _eps
    agent.epsilon_min = _epsmin
    agent.prefer_center, agent.tie_tol = pc_backup, tol_backup

    return results

def log_phase_evaluation(agent, env, phase_name, episode, device, Lookahead,
                         evaluation_opponents, excel_path):
    """
    Evaluates the agent after a training phase, logs results to Excel.
    
    Params:
    - agent: trained DQNAgent
    - env: Connect4Env (resettable)
    - phase_name: string, name of the training phase
    - episode: total episode count reached
    - device: torch.device
    - Lookahead: Connect4Lookahead instance
    - evaluation_opponents: dict of {"label": num_games}
    - excel_path: output Excel file to append to
    """

    if phase_name is None: return
    print(f"📊 Running evaluation after phase: {phase_name} (ep {episode})")
    start_time = time.time()

    # --- Evaluate ---
    results = evaluate_agent_model(agent, env, evaluation_opponents, device, Lookahead)

    # --- Flatten results ---
    flat = {
        "TRAINING_SESSION": f"{phase_name}-EP-{episode}",
        "TIME [h]": round((time.time() - start_time) / 3600, 6),
        "EPISODES": episode,
    }
    for label, metrics in results.items():
        flat[label] = metrics["win_rate"]

    # --- Append to Excel ---
    df_new = pd.DataFrame([flat])

    if os.path.exists(excel_path):
        df_existing = pd.read_excel(excel_path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_excel(excel_path, index=False)
    print(f"✅ Evaluation results saved to: {excel_path}")


