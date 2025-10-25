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


def play_single_game(agent, env, device, Lookahead, opponent_label, game_index,
                     debug: bool = False, debug_depth: int = 1):
    """
    Play one game: the agent alternates colors by game index.
      - even game_index: agent plays +1
      - odd  game_index: agent plays -1

    If debug=True:
      - debug_depth == 0  -> agent plays RANDOM
      - debug_depth > 0   -> agent plays Lookahead(depth=debug_depth)
    """
    # Decide which color the agent plays THIS game
    agent_color = +1 if (game_index % 2 == 0) else -1
    opp_color   = -agent_color

    depth = _parse_lookahead_depth(opponent_label, default=2)

    # Fresh env; set the starting side
    state = env.reset()             # absolute (+1 plane, -1 plane, row, col)
    env.current_player = agent_color
    done = False

    rng = random.Random()

    while not done:
        valid_actions = env.available_actions()
        side = env.current_player

        # Build an absolute numeric board once (for lookahead)
        # Decode using +1 perspective to get the true numeric board
        board_abs = agent.decode_board_from_state(state, player=+1)

        if side == agent_color:
            # ---- Agent move ----
            if debug:
                if debug_depth == 0:
                    action = rng.choice(valid_actions)
                else:
                    action = Lookahead.n_step_lookahead(board_abs, player=agent_color, depth=debug_depth)
                    if action not in valid_actions:
                        action = rng.choice(valid_actions)
            else:
                # Orient planes so channel 0 == agent, channel 1 == opp
                oriented = agent._agent_planes(state, agent_color)     # (4,6,7)
                q = agent._q_values_np(oriented)                       # (7,)
                action = _mask_argmax(q, valid_actions)
        else:
            # ---- Opponent move ----
            if opponent_label == "Random":
                action = rng.choice(valid_actions)
            else:
                action = Lookahead.n_step_lookahead(board_abs, player=side, depth=depth)
                if action not in valid_actions:
                    action = rng.choice(valid_actions)

        state, _, done = env.step(action)

    # Outcome from the AGENT's color for THIS game
    if env.winner == agent_color:
        outcome = 1
    elif env.winner == opp_color:
        outcome = -1
    else:
        outcome = 0.5  # draw or None fallback

    return outcome, state


def evaluate_agent_model(agent, env, evaluation_opponents, device, Lookahead,
                         debug: bool = False, debug_depth: int = 1):
    """
    Evaluates agent.model vs scripted opponents.
    Alternates agent color each game; computes outcome from the agent's perspective.
    """
    if debug:
        print(f"DEBUG run (agent debug policy), depth={debug_depth}")

    results = {}

    # Backup agent state/knobs and force pure argmax w/o tie bias
    model_mode = agent.model.training
    target_mode = agent.target_model.training
    _eps, _epsmin = agent.epsilon, agent.epsilon_min
    pc_backup, tol_backup = agent.prefer_center, agent.tie_tol
    sym_backup = agent.use_symmetry_policy

    agent.model.eval(); agent.target_model.eval()
    agent.epsilon = 0.0; agent.epsilon_min = 0.0
    agent.prefer_center = False
    agent.tie_tol = 0.0
    agent.use_symmetry_policy = False

    for label, num_games in evaluation_opponents.items():
        wins = losses = draws = 0

        with tqdm(total=num_games, desc=f"Opponent: {label}", position=1, leave=False) as pbar:
            for game_index in range(num_games):
                outcome, _ = play_single_game(
                    agent, env, device, Lookahead, label, game_index,
                    debug=debug, debug_depth=debug_depth
                )
                if outcome == 1:   wins += 1
                elif outcome == -1: losses += 1
                else:               draws += 1
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

    # Restore agent state/knobs
    agent.model.train(model_mode)
    agent.target_model.train(target_mode)
    agent.epsilon = _eps
    agent.epsilon_min = _epsmin
    agent.prefer_center, agent.tie_tol = pc_backup, tol_backup
    agent.use_symmetry_policy = sym_backup

    return results


def log_phase_evaluation(agent, env, phase_name, episode, device, Lookahead,
                         evaluation_opponents, excel_path):
    """
    Evaluates the agent after a training phase, logs results to Excel.
    """
    if phase_name is None: return
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
        df_existing = pd.read_excel(excel_path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_excel(excel_path, index=False)
    print(f"✅ Evaluation results saved to: {excel_path}")
