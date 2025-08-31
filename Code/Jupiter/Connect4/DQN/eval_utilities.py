# DQN/eval_utilities.py

import time
import random
import numpy as np
import torch
from tqdm import tqdm
import os
import pandas as pd

def _valid_actions_of(env):
    return env.available_actions() if hasattr(env, "available_actions") else env.valid_actions()

def _mask_argmax(q, valid_actions):
    q = np.asarray(q, dtype=np.float32)
    mask = np.full_like(q, -1e9)
    for a in valid_actions:
        mask[a] = 0.0
    a = int(np.argmax(q + mask))
    return a if a in valid_actions else random.choice(valid_actions)

def build_input(agent, state, player, device):
    """
    Decodes 4-channel state to 2D board, then reconstructs 4-channel state for inference.
    Returns (1, 4, 6, 7) tensor on device.
    """
    board = agent.decode_board_from_state(state, player)
    reencoded = agent.board_to_state(board, player)
    x = torch.tensor(reencoded, dtype=torch.float32, device=device).unsqueeze(0)
    return x

def evaluate_agent_model(agent, env, evaluation_opponents, device, Lookahead):
    """
    Evaluates agent.model vs various scripted opponents.
    Each game is played from fresh env.reset(), alternating starting player.
    """
    results = {}

    # Backup agent state
    model_mode = agent.model.training
    target_mode = agent.target_model.training
    _eps, _epsmin = agent.epsilon, agent.epsilon_min

    agent.model.eval()
    agent.target_model.eval()
    agent.epsilon = 0.0
    agent.epsilon_min = 0.0

    start_time = time.time()

    for label, num_games in evaluation_opponents.items():
        wins = losses = draws = 0
        depth = int(label.split("-")[1]) if label.startswith("Lookahead") else None

        with tqdm(total=num_games, desc=f"Opponent: {label}") as pbar:
            for game_index in range(num_games):
                state = env.reset()
                env.current_player = 1 if (game_index % 2 == 0) else -1
                done = False

                while not done:
                    valid_actions = _valid_actions_of(env)
                    player = env.current_player

                    if player == 1:
                        with torch.no_grad():
                            x = build_input(agent, state, player=1, device=device)
                            q = agent.model(x).squeeze(0).cpu().numpy()  # shape: (7,)
                        action = _mask_argmax(q, valid_actions)
                    else:
                        if label == "Random":
                            action = random.choice(valid_actions)
                        else:
                            board = agent.decode_board_from_state(state, player=-1)
                            action = Lookahead.n_step_lookahead(board, player=-1, depth=depth)
                            if action not in valid_actions:
                                action = random.choice(valid_actions)

                    state, _, done = env.step(action)

                # Outcome tally from agent's perspective
                if env.winner == 1:
                    wins += 1
                elif env.winner == -1:
                    losses += 1
                else:
                    draws += 1

                pbar.update(1)

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

    elapsed = time.time() - start_time
    print(f"✅ Evaluation completed in {elapsed/60:.1f} minutes")

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

