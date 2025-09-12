#PPO/ppo_eval.py
"""
PPO evaluation helper that mirrors DQN's evaluate_agent_model() output format.
Agent is always +1; starts alternate. Opponent: Random or Lookahead-k.
"""

import random
import numpy as np
import torch
from C4.connect4_env import Connect4Env
from C4.connect4_lookahead import Connect4Lookahead
from PPO.ppo_utilities import encode_two_channel_agent_centric

@torch.inference_mode()
def evaluate_actor_critic_model(
    agent,                                  # ActorCritic
    env: Connect4Env,
    evaluation_opponents: dict,             # {"Random": 201, "Lookahead-1": 101, ...}
    Lookahead=Connect4Lookahead,
    temperature: float = 0.0,               # greedy by default
):
    results = {}
    look = Lookahead()

    for label, num_games in evaluation_opponents.items():
        wins = losses = draws = 0
        depth = int(label.split("-")[1]) if label.startswith("Lookahead") else None

        for game_index in range(num_games):
            state = env.reset()
            env.current_player = 1 if (game_index % 2 == 0) else -1
            done = False

            while not done:
                if env.current_player == 1:
                    # ---- Agent (always +1) ----
                    legal = env.available_actions()
                    if not legal:
                        draws += 1
                        break

                    s2 = encode_two_channel_agent_centric(state, +1)
                    action, _, _, _ = agent.act(
                        s2.astype(np.float32), legal_actions=legal, temperature=temperature
                    )

                else:
                    # ---- Opponent (-1) ----
                    legal = env.available_actions()
                    if not legal:
                        draws += 1
                        break

                    if label == "Random":
                        action = random.choice(legal)
                    else:
                        a = look.n_step_lookahead(env.board.copy(), player=-1, depth=depth)
                        action = a if a in legal else random.choice(legal)

                state, _, done = env.step(action)

            # tally
            if env.winner == 1:
                wins += 1
            elif env.winner == -1:
                losses += 1
            else:
                draws += 1

        results[label] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": round(wins / num_games, 3),
            "loss_rate": round(losses / num_games, 3),
            "draw_rate": round(draws / num_games, 3),
        }

    return results
