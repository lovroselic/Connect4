# PPO/ppo_eval_utilities.py
# Visual finals + scripted-opponent evaluation for PPO ActorCritic

import os, time, random,  torch
from tqdm import tqdm
import pandas as pd

from C4.connect4_env import Connect4Env
from C4.connect4_board_display import Connect4_BoardDisplayer
from C4.connect4_lookahead import Connect4Lookahead
from PPO.actor_critic import ActorCritic
from PPO.ppo_utilities import encode_two_channel_agent_centric


# ----------------------------- Single game vs scripted opponent -----------------------------



@torch.inference_mode()
def ppo_play_single_game(policy: ActorCritic,
                         env: Connect4Env,
                         opponent_label: str,
                         game_index: int,
                         lookahead_impl: Connect4Lookahead | None = None,
                         temperature: float = 0.0):
    """
    Plays one game: PPO policy (agent = +1) vs scripted opponent (Random or Lookahead-k).
    Alternates starter: even game_index -> agent starts, odd -> opponent starts.
    Returns (outcome, final_board).
    """
    look = lookahead_impl or Connect4Lookahead()

    # >>> FIX: start a fresh game and THEN set who starts
    env.reset()
    env.current_player = +1 if (game_index % 2 == 0) else -1

    while not env.done:
        legal = env.available_actions()

        if env.current_player == +1:
            enc = encode_two_channel_agent_centric(env.board, +1)
            action, _, _, _ = policy.act(enc, legal, temperature=temperature)
        else:
            if opponent_label == "Random":
                action = random.choice(legal)
            else:
                depth = int(opponent_label.split("-")[1]) if opponent_label.startswith("Lookahead") else None
                try:
                    a = look.n_step_lookahead(env.board.copy(), player=-1, depth=depth) if depth else random.choice(legal)
                    action = a if a in legal else random.choice(legal)
                except Exception:
                    action = random.choice(legal)

        _, _, _ = env.step(action)  # env.done updates internally

    # Outcome from agent POV (+1)
    if env.winner == +1:
        return 1, env.board.copy()
    if env.winner == -1:
        return -1, env.board.copy()
    return 0.5, env.board.copy()



# ----------------------------- Show two finals per opponent -----------------------------

def display_final_boards_ppo(policy, env, opponents, lookahead_impl=None, temperature=0.0):
    for label in opponents:
        print(f"\n🎯 Opponent: {label}")
        for game_index in [0, 1]:
            # use the same env object but the play fn resets it each time
            outcome, final_board = ppo_play_single_game(
                policy=policy, env=env, opponent_label=label,
                game_index=game_index, lookahead_impl=lookahead_impl,
                temperature=temperature
            )
            result = "Win" if outcome == 1 else ("Loss" if outcome == -1 else "Draw")
            title = f"{label} — {'Agent First' if game_index == 0 else 'Opponent First'} — {result}"
            Connect4_BoardDisplayer.display_board(final_board, title)



# ----------------------------- Batch evaluation vs scripted opponents -----------------------------

@torch.inference_mode()
def evaluate_policy_model(policy: ActorCritic,
                          env: Connect4Env,
                          evaluation_opponents: dict[str, int],
                          lookahead_impl: Connect4Lookahead | None = None,
                          temperature: float = 0.0):
    """
    Evaluate PPO policy vs scripted opponents. Alternates starter each game.
    Returns dict like DQN's evaluate_agent_model.
    """
    look = lookahead_impl or Connect4Lookahead()
    results = {}

    start_time = time.time()
    for label, num_games in evaluation_opponents.items():
        wins = losses = draws = 0
        with tqdm(total=num_games, desc=f"Opponent: {label}") as pbar:
            for gi in range(num_games):
                outcome, _ = ppo_play_single_game(
                    policy=policy, env=env,
                    opponent_label=label, game_index=gi,
                    lookahead_impl=look, temperature=temperature
                )
                if outcome == 1: wins += 1
                elif outcome == -1: losses += 1
                else: draws += 1
                pbar.update(1)

        results[label] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": round(wins / num_games, 3),
            "loss_rate": round(losses / num_games, 3),
            "draw_rate": round(draws / num_games, 3),
        }

    elapsed = time.time() - start_time
    print(f"✅ PPO evaluation completed in {elapsed/60:.1f} minutes")
    return results


# ----------------------------- Phase logging to Excel (PPO) -----------------------------

def log_phase_evaluation_ppo(policy: ActorCritic,
                             env: Connect4Env,
                             phase_name: str,
                             episode: int,
                             evaluation_opponents: dict[str, int],
                             excel_path: str,
                             lookahead_impl: Connect4Lookahead | None = None,
                             temperature: float = 0.0):
    """
    Mirror of DQN logger: run eval and append win rates to an Excel file.
    """
    if not phase_name:
        return

    print(f"📊 PPO eval after phase: {phase_name} (ep {episode})")
    t0 = time.time()
    results = evaluate_policy_model(
        policy=policy,
        env=env,
        evaluation_opponents=evaluation_opponents,
        lookahead_impl=lookahead_impl,
        temperature=temperature,
    )

    flat = {
        "TRAINING_SESSION": f"{phase_name}-EP-{episode}",
        "TIME [h]": round((time.time() - t0) / 3600, 6),
        "EPISODES": episode,
    }
    for label, metrics in results.items():
        flat[label] = metrics["win_rate"]

    df_new = pd.DataFrame([flat])
    if os.path.exists(excel_path):
        try:
            df_old = pd.read_excel(excel_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_all = df_new
    else:
        df_all = df_new

    df_all.to_excel(excel_path, index=False)
    print(f"✅ Saved PPO eval row to: {excel_path}")
