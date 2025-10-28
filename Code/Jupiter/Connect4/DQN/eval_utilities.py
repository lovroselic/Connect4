# DQN/eval_utilities.py

import time
import random
import numpy as np
import os
import pandas as pd
import re
from tqdm.auto import tqdm


# ----------------------------- helpers -----------------------------

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


# ----------------------------- core moves -----------------------------

def _agent_q_action(agent, state_abs, valid_actions):
    """
    Pure argmax(Q) among legal actions for the +1 agent.
    state_abs: (4,6,7) = [+1, -1, row, col] from env.get_state().
    """
    # state_abs is already mover-first for +1; no reorientation needed
    q = agent._q_values_np(state_abs)           # (7,)
    # use the agent’s legal-argmax with eval-time knobs disabled upstream
    a = agent._argmax_legal_with_tiebreak(q, valid_actions)
    # safety (helps catch NaNs or masking mistakes in debugging)
    #assert a in valid_actions, f"Illegal action {a} chosen; valid={valid_actions}"
    return a

def _opp_action(env, Lookahead, label, depth, rng):
    """
    Opponent plays directly on env.board (no plane re-orientation).
    """
    valid = env.available_actions()
    if label == "Random": return rng.choice(valid)
    a = Lookahead.n_step_lookahead(env.board, player=-1, depth=depth)
    return a if a in valid else rng.choice(valid)

# ----------------------------- single game -----------------------------

def play_single_game(agent, env, device, Lookahead, opponent_label, game_index,
                     debug: bool = False, debug_depth: int = 1):
    """
    Agent is ALWAYS +1. We alternate who starts by giving the starter exactly one move
    before the main loop. We explicitly set env.current_player each turn (don’t rely on
    env’s flip for control flow).
    If debug=True:
        - debug_depth == 0 -> agent plays Random
        - debug_depth > 0  -> agent plays Lookahead(depth=debug_depth)
      Opponent behavior is still controlled by `opponent_label`.
    """
    AGENT, OPP = +1, -1
    depth = _parse_lookahead_depth(opponent_label, default=2)
    rng = random.Random()

    # --- backup & force “pure policy” eval for the agent ---
    pc_bak, tol_bak = agent.prefer_center, agent.tie_tol
    sym_bak = agent.use_symmetry_policy
    eps_bak, epsmin_bak = agent.epsilon, agent.epsilon_min

    agent.prefer_center = False
    agent.tie_tol = 0.0
    agent.use_symmetry_policy = False
    agent.epsilon = 0.0
    agent.epsilon_min = 0.0

    # --- fresh game ---
    state = env.reset()  # absolute planes (+1, -1, row, col)
    done = False

    # --- alternate who starts (even -> agent starts) ---
    if (game_index % 2) != 0:
        env.current_player = OPP
        a_opp = _opp_action(env, Lookahead, opponent_label, depth, rng)
        state, _, done = env.step(a_opp)

    # --- main loop: agent, then opponent ---
    while not done:
        # Agent move
        env.current_player = AGENT
        if debug:
            if debug_depth == 0: a_agent = rng.choice(env.available_actions())
            else:
                a_agent = Lookahead.n_step_lookahead(env.board, player=+1, depth=debug_depth)
                if a_agent not in env.available_actions(): raise ValueError("ILLEGAL MOVE DETECTED!")

        else: a_agent = _agent_q_action(agent, state, env.available_actions())

        state, _, done = env.step(a_agent)
        if done: break

        # Opponent move
        env.current_player = OPP
        a_opp = _opp_action(env, Lookahead, opponent_label, depth, rng)
        state, _, done = env.step(a_opp)
        if done: break

    # --- outcome from agent’s POV ---
    if env.winner == AGENT:     outcome = 1.0
    elif env.winner == OPP:     outcome = -1.0
    else:                       outcome = 0.5

    # --- restore agent knobs ---
    agent.prefer_center, agent.tie_tol = pc_bak, tol_bak
    agent.use_symmetry_policy = sym_bak
    agent.epsilon, agent.epsilon_min = eps_bak, epsmin_bak

    return outcome, state


# ----------------------------- batch eval -----------------------------

def evaluate_agent_model(agent, env, evaluation_opponents, device, Lookahead,
                         debug: bool = False, debug_depth: int = 1):
    """
    Evaluates agent.model vs opponents. Pass debug=True to make the AGENT
    play Random (depth=0) or Lookahead (depth>0) for sanity checks.
    """
    results = {}
    
    if debug: print(f"DEBUG depth {debug_depth}")

    # Backup train/eval modes
    model_mode = agent.model.training
    target_mode = agent.target_model.training
    agent.model.eval()
    agent.target_model.eval()

    for label, num_games in evaluation_opponents.items():
        wins = losses = draws = 0
        with tqdm(total=num_games, desc=f"Opponent: {label}", position=1, leave=False) as pbar:
            for game_index in range(num_games):
                outcome, _ = play_single_game(
                    agent, env, device, Lookahead, label, game_index,
                    debug=debug, debug_depth=debug_depth
                )
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


# ----------------------------- logging -----------------------------

def log_phase_evaluation(agent, env, phase_name, episode, device, Lookahead,
                         evaluation_opponents, excel_path,
                         debug: bool = False, debug_depth: int = 1):
    if phase_name is None:
        return
    print(f"📊 Running evaluation after phase: {phase_name} (ep {episode})")
    start_time = time.time()

    results = evaluate_agent_model(
        agent, env, evaluation_opponents, device, Lookahead,
        debug=debug, debug_depth=debug_depth
    )

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


# ----------------------------- sanity check (optional) -----------------------------

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

        if not agent_starts:  # opponent opens
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
            if done: break

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
        "score_rate": (wins + 0.5*draws)/total,
    }
