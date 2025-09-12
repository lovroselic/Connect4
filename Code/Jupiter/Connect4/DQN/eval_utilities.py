# DQN/eval_utilities.py

import time
import random
import numpy as np
import torch
from tqdm import tqdm
import os
import pandas as pd

def _valid_actions_of(env):
    return env.available_actions()

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

def play_single_game(agent, env, device, Lookahead, opponent_label, game_index):
    """
    Plays a single game between the agent and the scripted opponent.
    Alternates starting player based on game_index (even = agent starts).
    Always returns board where agent is +1 and opponent is -1.
    
    Returns:
        outcome: 1 (win), -1 (loss), 0.5 (draw) from agent's perspective
        final_state: the final state at game end
    """
    # Always treat agent as +1, opponent as -1
    AGENT_SIDE = 1
    OPPONENT_SIDE = -1

    # Parse opponent depth
    depth = int(opponent_label.split("-")[1]) if opponent_label.startswith("Lookahead") else None

    # Alternate who starts
    state = env.reset()
    env.current_player = AGENT_SIDE if (game_index % 2 == 0) else OPPONENT_SIDE
    done = False

    while not done:
        valid_actions = _valid_actions_of(env)
        current = env.current_player

        if current == AGENT_SIDE:
            with torch.no_grad():
                x = build_input(agent, state, player=AGENT_SIDE, device=device)
                q = agent.model(x).squeeze(0).cpu().numpy()
            action = _mask_argmax(q, valid_actions)

        else:
            if opponent_label == "Random":
                action = random.choice(valid_actions)
            else:
                # Always decode from agent's POV (AGENT_SIDE = +1)
                board = agent.decode_board_from_state(state, player=AGENT_SIDE)
                action = Lookahead.n_step_lookahead(board, player=OPPONENT_SIDE, depth=depth)

                # Safety: fallback to valid if needed
                if action not in valid_actions:
                    action = random.choice(valid_actions)

        state, _, done = env.step(action)

    # Outcome from agent’s POV (agent is always +1)
    if env.winner == AGENT_SIDE:
        outcome = 1
    elif env.winner == OPPONENT_SIDE:
        outcome = -1
    elif env.winner == 0:
        outcome = 0.5
    else:
        outcome = 0.5

    return outcome, state



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

        with tqdm(total=num_games, desc=f"Opponent: {label}") as pbar:
            for game_index in range(num_games):
                outcome, _ = play_single_game(agent, env, device, Lookahead, label, game_index)
                if outcome == 1:
                    wins += 1
                elif outcome == -1:
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

