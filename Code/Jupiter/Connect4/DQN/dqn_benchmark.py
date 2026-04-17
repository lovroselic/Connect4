"""dqn_benchmark.py

Benchmarking helpers for 1-channel DQN/CNet192-style models.

This is deliberately small and predictable:
- evaluates a model vs a set of opponents (Random / Lookahead depths / SP / POP)
- returns win/loss/draw + win_rate
- computes a depth-weighted global score using the same function as your PPO tooling
  (exponential weights base^depth).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from DQN.dqn_loop_helpers_1ch import (
    ensure_1ch_state,
    legal_actions_from_state,
    greedy_model_action,
    biased_random_choice,
    extract_step_out,
    select_opponent_actor,
)

from DQN.dqn_utilities import global_score_from_results


def _play_game_model_vs_opponent(
    *,
    env: Any,
    model: torch.nn.Module,
    opponent_act: Callable,
    device: torch.device,
    rng: np.random.Generator,
    model_starts: bool,
    opening_noise_k: int = 0,
) -> float:
    """Returns result from model perspective: 1 win, 0.5 draw, 0 loss."""

    obs = env.reset()
    if hasattr(env, "board") and hasattr(env, "current_player"):
        b = np.asarray(env.board, dtype=np.int8)
        p = int(getattr(env, "current_player", 1))
        state = (b * p).astype(np.float32)[None, :, :]
    else:
        state = ensure_1ch_state(obs)

    ply = 0
    while True:
        legal = legal_actions_from_state(state)

        model_to_move = (ply % 2 == 0) == bool(model_starts)

        if ply < opening_noise_k:
            a = biased_random_choice(legal, rng=rng, bias="center")
        else:
            if model_to_move:
                a = greedy_model_action(model, state, legal, device=device)
            else:
                a = int(opponent_act(state, legal, ply))

        obs2, r, done, _ = extract_step_out(env.step(a))
        if hasattr(env, "board") and hasattr(env, "current_player"):
            b2 = np.asarray(env.board, dtype=np.int8)
            p2 = int(getattr(env, "current_player", 1))
            state = (b2 * p2).astype(np.float32)[None, :, :]
        else:
            state = ensure_1ch_state(obs2)
        ply += 1

        if done:
            if r == 0:
                return 0.5

            mover_is_model = (ply - 1) % 2 == 0 == bool(model_starts)
            if r > 0:
                return 1.0 if mover_is_model else 0.0
            else:
                return 0.0 if mover_is_model else 1.0


def eval_vs_one(
    *,
    model: torch.nn.Module,
    opponent_key: str,
    env_factory: Callable[[], Any],
    device: torch.device,
    n_games: int = 200,
    seed: int = 0,
    opening_noise_k: int = 0,
    sp_q_net: Optional[torch.nn.Module] = None,
    pop_ensemble: Optional[torch.nn.Module] = None,
    lookahead: Optional[Any] = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(int(seed))

    opp_act, _tag = select_opponent_actor(
        opponent_key,
        device=device,
        rng=rng,
        sp_q_net=sp_q_net,
        pop_ensemble=pop_ensemble,
        lookahead=lookahead,
    )

    wins = losses = draws = 0
    for i in range(int(n_games)):
        # alternate starts
        model_starts = bool((i % 2) == 0)
        r = _play_game_model_vs_opponent(
            env=env_factory(),
            model=model,
            opponent_act=opp_act,
            device=device,
            rng=rng,
            model_starts=model_starts,
            opening_noise_k=int(opening_noise_k),
        )

        if r > 0.75:
            wins += 1
        elif r < 0.25:
            losses += 1
        else:
            draws += 1

    n = float(n_games)
    win_rate = wins / n
    return {
        "n": float(n_games),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "win_rate": float(win_rate),
        "score_rate": float((wins + 0.5 * draws) / n),
    }


def benchmark_suite(
    *,
    model: torch.nn.Module,
    opponents: Sequence[str],
    env_factory: Callable[[], Any],
    device: torch.device,
    n_games: int = 200,
    seed: int = 0,
    opening_noise_k: int = 0,
    sp_q_net: Optional[torch.nn.Module] = None,
    pop_ensemble: Optional[torch.nn.Module] = None,
    lookahead: Optional[Any] = None,
    global_base: float = 1.4,
) -> Dict[str, Any]:
    results: Dict[str, Dict[str, float]] = {}

    for k in opponents:
        label = "Random" if k.upper() == "R" else (f"Lookahead-{k[1:]}" if k.upper().startswith("L") else k.upper())
        results[label] = eval_vs_one(
            model=model,
            opponent_key=str(k),
            env_factory=env_factory,
            device=device,
            n_games=int(n_games),
            seed=int(seed),
            opening_noise_k=int(opening_noise_k),
            sp_q_net=sp_q_net,
            pop_ensemble=pop_ensemble,
            lookahead=lookahead,
        )

    global_score = float(global_score_from_results(results, base=float(global_base)))
    return {
        "by_opponent": results,
        "global_score": global_score,
    }


__all__ = ["benchmark_suite", "eval_vs_one"]
