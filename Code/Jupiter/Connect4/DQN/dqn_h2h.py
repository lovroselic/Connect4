"""dqn_h2h.py

Head-to-head evaluation for DQN-style (score-only) models.

Plays Connect-4 games between two frozen models A and B.
Models can be:
  - CNet192 (returns (logits, value))
  - DQN q-net (returns logits/Q)
  - an ensemble (returns logits/Q)

All actions are greedy with a center tie-break.

The env is passed as a factory so each game is clean.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from DQN.dqn_loop_helpers_1ch import (
    ensure_1ch_state,
    legal_actions_from_state,
    biased_random_choice,
    greedy_model_action,
    extract_step_out,
)


def _opening_noise_move(state, legal, rng, bias: str = "center") -> int:
    return biased_random_choice(legal, rng=rng, bias=bias)


def play_game_greedy(
    *,
    env: Any,
    modelA: torch.nn.Module,
    modelB: torch.nn.Module,
    device: torch.device,
    A_starts: bool,
    opening_noise_k: int = 0,
    opening_bias: str = "center",
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Returns result from A perspective: 1 win, 0.5 draw, 0 loss."""

    if rng is None:
        rng = np.random.default_rng(0)

    obs = env.reset()
    if hasattr(env, "board") and hasattr(env, "current_player"):
        b = np.asarray(env.board, dtype=np.int8)
        p = int(getattr(env, "current_player", 1))
        state = (b * p).astype(np.float32)[None, :, :]
    else:
        state = ensure_1ch_state(obs)

    ply = 0
    # Who acts now? If A_starts, then on ply even -> A else B. If not, swap.
    while True:
        legal = legal_actions_from_state(state)

        if ply < int(opening_noise_k):
            a = _opening_noise_move(state, legal, rng=rng, bias=opening_bias)
        else:
            a_is_A = (ply % 2 == 0) == bool(A_starts)
            model = modelA if a_is_A else modelB
            a = greedy_model_action(model, state, legal, device=device)

        obs2, r, done, _info = extract_step_out(env.step(a))
        if hasattr(env, "board") and hasattr(env, "current_player"):
            b2 = np.asarray(env.board, dtype=np.int8)
            p2 = int(getattr(env, "current_player", 1))
            state = (b2 * p2).astype(np.float32)[None, :, :]
        else:
            state = ensure_1ch_state(obs2)
        ply += 1

        if done:
            # reward is for mover. Determine whether mover was A.
            mover_is_A = (ply - 1) % 2 == 0 == bool(A_starts)
            if r == 0:
                return 0.5

            # mover got positive reward => mover won
            if r > 0:
                return 1.0 if mover_is_A else 0.0
            else:
                # mover lost
                return 0.0 if mover_is_A else 1.0


def head_to_head_models(
    modelA: torch.nn.Module,
    modelB: torch.nn.Module,
    *,
    env_factory: Callable[[], Any],
    n_games: int = 200,
    device: torch.device,
    opening_noise_k: int = 0,
    opening_bias: str = "center",
    seed: int = 0,
    paired_openings: bool = True,
) -> Dict[str, float]:
    """Play A vs B. Returns dict with A_score_rate, A_win_rate, A_loss_rate, draw_rate."""

    rng = np.random.default_rng(int(seed))

    A_wins = 0
    A_losses = 0
    draws = 0

    # paired openings: use same rng sub-seed for two games with swapped starts
    i = 0
    while i < int(n_games):
        if paired_openings and (i + 1) < int(n_games):
            sub_seed = int(rng.integers(0, 2**31 - 1))
            g1 = np.random.default_rng(sub_seed)
            g2 = np.random.default_rng(sub_seed)

            r1 = play_game_greedy(
                env=env_factory(),
                modelA=modelA,
                modelB=modelB,
                device=device,
                A_starts=True,
                opening_noise_k=opening_noise_k,
                opening_bias=opening_bias,
                rng=g1,
            )
            r2 = play_game_greedy(
                env=env_factory(),
                modelA=modelA,
                modelB=modelB,
                device=device,
                A_starts=False,
                opening_noise_k=opening_noise_k,
                opening_bias=opening_bias,
                rng=g2,
            )

            for r in (r1, r2):
                if r > 0.75:
                    A_wins += 1
                elif r < 0.25:
                    A_losses += 1
                else:
                    draws += 1

            i += 2
        else:
            A_starts = bool((i % 2) == 0)
            r = play_game_greedy(
                env=env_factory(),
                modelA=modelA,
                modelB=modelB,
                device=device,
                A_starts=A_starts,
                opening_noise_k=opening_noise_k,
                opening_bias=opening_bias,
                rng=rng,
            )

            if r > 0.75:
                A_wins += 1
            elif r < 0.25:
                A_losses += 1
            else:
                draws += 1

            i += 1

    n = float(n_games)
    A_score_rate = (A_wins + 0.5 * draws) / n
    return {
        "A_score_rate": float(A_score_rate),
        "A_win_rate": float(A_wins / n),
        "A_loss_rate": float(A_losses / n),
        "draw_rate": float(draws / n),
        "n_games": float(n_games),
    }


__all__ = ["head_to_head_models"]
