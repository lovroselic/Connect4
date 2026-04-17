# DQN/eval_utilities.py
"""
Evaluation utilities for 1-channel POV DQN agents.

Assumptions:
- Env exposes env.board (6x7) with values in {0,+1,-1} OR {0,1,2}.
- Env exposes env.current_player as +1/-1 OR 1/2.
- Agent expects 1ch POV state: float32 (1,6,7) where mover's stones are +1, opponent -1.
- Agent provides:
    - act(state_1ch, legal_actions, epsilon_override=None) -> int
  (If not, we fall back to greedy_model_action on agent.q_net / agent.model.)
"""

from __future__ import annotations

import os
import re
import time
import random
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from DQN.dqn_loop_helpers_1ch import (
    ensure_1ch_state,
    legal_actions_from_state,
    greedy_model_action,
    extract_step_out,
)


# ----------------------------- helpers -----------------------------

def _parse_lookahead_depth(label: str, default: int = 2) -> int:
    # supports "Lookahead-7", "Lookahead7", "L7", etc.
    m = re.search(r"(\d+)", str(label))
    return int(m.group(1)) if m else int(default)


def _normalize_player(v: Any) -> int:
    # normalize winner/current_player to +1/-1 when possible
    try:
        v = int(v)
    except Exception:
        return 0
    if v == 2:
        return -1
    return v


def _board_to_pm1(board: np.ndarray) -> np.ndarray:
    # if board is {0,1,2} -> {0,+1,-1}
    b = np.asarray(board, dtype=np.int8)
    if b.max(initial=0) >= 2:
        b = np.where(b == 2, -1, b).astype(np.int8, copy=False)
    return b


def _state_from_env(env: Any, obs_fallback: Any) -> np.ndarray:
    """
    Return 1ch POV state (1,6,7) float32 from env if possible, else from obs.
    """
    if hasattr(env, "board") and hasattr(env, "current_player"):
        b = _board_to_pm1(env.board)
        p = _normalize_player(getattr(env, "current_player", 1))
        if p == 0:
            p = 1
        s = (b * p).astype(np.float32, copy=False)[None, :, :]
        return s
    return ensure_1ch_state(obs_fallback)


def _available_actions(env: Any, state_1ch: np.ndarray) -> list[int]:
    if hasattr(env, "available_actions"):
        try:
            return list(map(int, env.available_actions()))
        except Exception:
            pass
    # fallback: infer from top row of POV state
    return list(map(int, legal_actions_from_state(state_1ch)))


def _get_lookahead_obj(Lookahead: Any) -> Any:
    """
    Accept either:
      - an instance with .n_step_lookahead(...)
      - a class Connect4Lookahead (callable) -> instance
    """
    if Lookahead is None:
        return None
    if hasattr(Lookahead, "n_step_lookahead"):
        # could be instance OR class; class method call will TypeError later, we handle there too
        return Lookahead
    if callable(Lookahead):
        obj = Lookahead()
        if hasattr(obj, "n_step_lookahead"):
            return obj
    return Lookahead


def _opp_action(env: Any, lookahead_obj: Any, label: str, depth: int, rng: random.Random) -> int:
    state = _state_from_env(env, None)
    valid = _available_actions(env, state)

    if label == "Random":
        return int(rng.choice(valid))

    if lookahead_obj is None:
        raise ValueError("Lookahead opponent requested but Lookahead is None")

    # Most of your lookahead code is instance-method based.
    try:
        a = lookahead_obj.n_step_lookahead(_board_to_pm1(env.board), player=-1, depth=depth)
    except TypeError:
        # probably passed a class instead of an instance
        a = lookahead_obj().n_step_lookahead(_board_to_pm1(env.board), player=-1, depth=depth)

    a = int(a)
    return a if a in valid else int(rng.choice(valid))


def _agent_action(agent: Any, state_1ch: np.ndarray, valid: list[int], device, *,
                  debug: bool, debug_depth: int, lookahead_obj: Any, rng: random.Random) -> int:
    if debug:
        if int(debug_depth) <= 0:
            return int(rng.choice(valid))
        if lookahead_obj is None:
            raise ValueError("debug_depth>0 requested but Lookahead is None")
        # In POV state, mover is always +1
        b = np.asarray(state_1ch[0], dtype=np.int8)
        try:
            a = lookahead_obj.n_step_lookahead(b, player=+1, depth=int(debug_depth))
        except TypeError:
            a = lookahead_obj().n_step_lookahead(b, player=+1, depth=int(debug_depth))
        a = int(a)
        return a if a in valid else int(rng.choice(valid))

    # Normal eval: greedy (epsilon 0)
    if hasattr(agent, "act"):
        return int(agent.act(state_1ch, valid, epsilon_override=0.0))

    # fallback: greedy from model
    model = getattr(agent, "q_net", None) or getattr(agent, "model", None)
    if model is None:
        raise AttributeError("Agent has neither act() nor q_net/model for greedy fallback")
    return int(greedy_model_action(model, state_1ch, valid, device=device))


# ----------------------------- single game -----------------------------

def play_single_game(
    agent,
    env,
    device,
    Lookahead,
    opponent_label,
    game_index,
    debug: bool = False,
    debug_depth: int = 1,
):
    """
    Agent is ALWAYS +1. We alternate who starts by giving the starter exactly one move
    before the main loop. We set env.current_player explicitly each turn.
    """
    AGENT, OPP = +1, -1
    depth = _parse_lookahead_depth(opponent_label, default=2)
    rng = random.Random()

    lookahead_obj = _get_lookahead_obj(Lookahead)

    obs = env.reset()
    done = False

    # --- alternate who starts (odd game -> opponent opens) ---
    if (game_index % 2) != 0:
        if hasattr(env, "current_player"):
            env.current_player = OPP
        a_opp = _opp_action(env, lookahead_obj, opponent_label, depth, rng)
        obs, _r, done, _info = extract_step_out(env.step(a_opp))

    # --- main loop: agent, then opponent ---
    while not done:
        # Agent move
        if hasattr(env, "current_player"):
            env.current_player = AGENT
        state = _state_from_env(env, obs)
        valid = _available_actions(env, state)

        a_agent = _agent_action(
            agent, state, valid, device,
            debug=debug, debug_depth=debug_depth,
            lookahead_obj=lookahead_obj, rng=rng
        )

        obs, _r, done, _info = extract_step_out(env.step(a_agent))
        if done:
            break

        # Opponent move
        if hasattr(env, "current_player"):
            env.current_player = OPP
        a_opp = _opp_action(env, lookahead_obj, opponent_label, depth, rng)
        obs, _r, done, _info = extract_step_out(env.step(a_opp))

    # --- outcome from agent POV ---
    if hasattr(env, "winner"):
        w = _normalize_player(env.winner)
        if w == AGENT:
            outcome = 1.0
        elif w == OPP:
            outcome = -1.0
        else:
            outcome = 0.5
    else:
        # fallback: if env doesn't expose winner, treat as draw
        outcome = 0.5

    return outcome, obs


# ----------------------------- batch eval -----------------------------

def evaluate_agent_model(
    agent,
    env,
    evaluation_opponents,
    device,
    Lookahead,
    debug: bool = False,
    debug_depth: int = 1,
):
    """
    Evaluates agent (greedy) vs evaluation_opponents dict: {label: num_games}.
    Keeps API used by dqn_utilities.update_benchmark_winrates().
    """
    results = {}

    # Put nets in eval during evaluation
    model = getattr(agent, "q_net", None) or getattr(agent, "model", None)
    target = getattr(agent, "target_net", None) or getattr(agent, "target_model", None)

    model_mode = model.training if hasattr(model, "training") else None
    tgt_mode = target.training if (target is not None and hasattr(target, "training")) else None
    if hasattr(model, "eval"):
        model.eval()
    if target is not None and hasattr(target, "eval"):
        target.eval()

    for label, num_games in evaluation_opponents.items():
        wins = losses = draws = 0
        with tqdm(total=int(num_games), desc=f"Opponent: {label}", position=1, leave=False) as pbar:
            for game_index in range(int(num_games)):
                outcome, _ = play_single_game(
                    agent, env, device, Lookahead, label, game_index,
                    debug=debug, debug_depth=debug_depth
                )
                if outcome == 1.0:
                    wins += 1
                elif outcome == -1.0:
                    losses += 1
                else:
                    draws += 1
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
    if model_mode is not None and hasattr(model, "train"):
        model.train(model_mode)
    if target is not None and tgt_mode is not None and hasattr(target, "train"):
        target.train(tgt_mode)

    return results


# ----------------------------- logging -----------------------------

def log_phase_evaluation(
    agent,
    env,
    phase_name,
    episode,
    device,
    Lookahead,
    evaluation_opponents,
    excel_path,
    debug: bool = False,
    debug_depth: int = 1,
):
    if phase_name is None:
        return

    print(f"Running evaluation after phase: {phase_name} (ep {episode})")
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
    print(f"Evaluation results saved to: {excel_path}")


# ----------------------------- sanity check (optional) -----------------------------

def sanity_check_random_vs_random(env, n_games=200):
    wins = losses = draws = 0
    rng = random.Random()

    for g in range(int(n_games)):
        env.reset()
        done = False
        agent_starts = (g % 2 == 0)

        if not agent_starts:
            if hasattr(env, "current_player"):
                env.current_player = -1
            a = rng.choice(_available_actions(env, _state_from_env(env, None)))
            _obs, _r, done, _info = extract_step_out(env.step(a))

        while not done:
            if hasattr(env, "current_player"):
                env.current_player = +1
            a = rng.choice(_available_actions(env, _state_from_env(env, None)))
            _obs, _r, done, _info = extract_step_out(env.step(a))
            if done:
                break

            if hasattr(env, "current_player"):
                env.current_player = -1
            a = rng.choice(_available_actions(env, _state_from_env(env, None)))
            _obs, _r, done, _info = extract_step_out(env.step(a))

        w = _normalize_player(getattr(env, "winner", 0))
        if w == +1:
            wins += 1
        elif w == -1:
            losses += 1
        else:
            draws += 1

    total = float(n_games)
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
        "score_rate": (wins + 0.5 * draws) / total,
    }
