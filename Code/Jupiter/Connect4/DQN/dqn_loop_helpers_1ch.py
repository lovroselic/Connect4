#DQN.dqn_loop_helpers_1ch
"""dqn_loop_helpers_1ch.py

Notebook-friendly helpers for a 1-channel Connect-4 DQN loop.

Key goals:
- Keep the main loop readable.
- Be strict about state shapes.
- Support the PPO-style opponent sampler keys: R, L1/L3/..., SP, POP.

Assumptions about env (minimal):
  env.reset() -> obs
  env.step(action) -> (obs2, reward, done) or (obs2, reward, done, info)
  env.done (bool) optional (we derive from step return)
  env.board (6,7) optional (used for lookahead)

Obs can be:
  - (6,7) or (1,6,7) with values in {-1,0,+1} from player-to-move POV

If your env returns a different encoding, adapt `ensure_1ch_state()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


COLS = 7
CENTER_COL = 3


# ----------------------------- state helpers -----------------------------

def ensure_1ch_state(obs: Any) -> np.ndarray:
    """Return float32 state shaped (1,6,7)."""
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()

    s = np.asarray(obs)

    # allow dict obs style
    if isinstance(obs, dict):
        if "state" in obs:
            s = np.asarray(obs["state"])
        elif "obs" in obs:
            s = np.asarray(obs["obs"])
        else:
            raise ValueError(f"Unsupported dict obs keys: {list(obs.keys())}")

    if s.ndim == 2 and s.shape == (6, 7):
        s = s[None, :, :]

    if s.ndim != 3 or s.shape != (1, 6, 7):
        raise ValueError(f"Expected (6,7) or (1,6,7), got {tuple(s.shape)}")

    return s.astype(np.float32, copy=False)


def legal_actions_from_state(s1: np.ndarray) -> List[int]:
    s1 = ensure_1ch_state(s1)
    top = s1[0, 0, :]
    return [c for c in range(COLS) if top[c] == 0]


def extract_step_out(step_out: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    if isinstance(step_out, (tuple, list)):
        if len(step_out) == 3:
            obs2, r, done = step_out
            info = {}
        elif len(step_out) == 4:
            obs2, r, done, info = step_out
            info = {} if info is None else dict(info)
        else:
            raise ValueError(f"env.step returned {len(step_out)} values, expected 3 or 4")
    else:
        raise TypeError("env.step must return tuple/list")

    return obs2, float(r), bool(done), info


# --------------------------- action selection ---------------------------

def argmax_center_tiebreak(scores: Union[np.ndarray, torch.Tensor], legal: Sequence[int]) -> int:
    legal = [int(a) for a in legal]
    if not legal:
        raise ValueError("No legal actions")

    if isinstance(scores, torch.Tensor):
        v = scores.detach().float().cpu().numpy()
    else:
        v = np.asarray(scores, dtype=np.float32)

    m = float(max(v[a] for a in legal))
    tied = [a for a in legal if abs(float(v[a]) - m) <= 1e-8]
    if len(tied) == 1:
        return int(tied[0])

    tied.sort(key=lambda a: (abs(a - CENTER_COL), a))
    return int(tied[0])


@torch.inference_mode()
def greedy_model_action(model: torch.nn.Module, state: np.ndarray, legal: Sequence[int], device: torch.device) -> int:
    """Greedy action for any model that returns (B,7) or (B,7),(B,...)"""
    if len(legal) == 1:
        return int(legal[0])

    s = ensure_1ch_state(state)
    x = torch.from_numpy(s).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,1,6,7)

    out = model(x)
    scores = out[0] if isinstance(out, (tuple, list)) else out
    if scores.dim() != 2 or scores.size(-1) != 7:
        raise ValueError(f"Model must return (B,7) scores, got {tuple(scores.shape)}")

    return argmax_center_tiebreak(scores[0], legal)


def biased_random_choice(legal: Sequence[int], rng: np.random.Generator, bias: str = "center") -> int:
    legal = [int(a) for a in legal]
    if len(legal) == 1:
        return int(legal[0])

    if bias == "none":
        return int(rng.choice(legal))

    # center-biased: weights fall off with distance
    d = np.array([abs(a - CENTER_COL) for a in legal], dtype=np.float32)
    w = 1.0 / (1.0 + d)
    w = w / w.sum()
    return int(rng.choice(legal, p=w))


# ------------------------- opponent dispatch ----------------------------

@dataclass
class OpponentSpec:
    key: str                # canonical key: "R", "LEFT", "CENTER", "SP", "POP", "L3"
    mode: Union[str, int]   # "random", "leftmost", "center", "self", depth


def _clean_opponent_key(key: str) -> str:
    return str(key).strip().upper().replace("_", "-").replace(" ", "")


def _parse_lookahead_depth(k: str) -> Optional[int]:
    """
    Accepts:
      L1, L3, L13
      LA1, LA-1
      LOOKAHEAD1, LOOKAHEAD-1
    """
    prefixes = ("LOOKAHEAD-", "LOOKAHEAD", "LA-", "LA", "L")

    for prefix in prefixes:
        if k.startswith(prefix):
            tail = k[len(prefix):]
            if tail.isdigit():
                return int(tail)

    return None


def key_to_opponent_mode(key: str) -> OpponentSpec:
    k = _clean_opponent_key(key)

    if k in ("R", "RAND", "RANDOM"):
        return OpponentSpec(key="R", mode="random")

    if k in ("LEFT", "LEFTMOST"):
        return OpponentSpec(key="LEFT", mode="leftmost")

    if k in ("C", "CENTER", "CENTRE"):
        return OpponentSpec(key="CENTER", mode="center")

    if k in ("SP", "SELF", "SELFPLAY", "SELF-PLAY"):
        return OpponentSpec(key="SP", mode="self")

    if k in ("POP", "ENSEMBLE", "POPENSEMBLE", "POP-ENSEMBLE"):
        return OpponentSpec(key="POP", mode="self")

    depth = _parse_lookahead_depth(k)
    if depth is not None:
        return OpponentSpec(key=f"L{depth}", mode=depth)

    raise ValueError(f"Unknown opponent key: {key}")


def select_opponent_actor(
    sampled_key: str,
    *,
    device: torch.device,
    rng: np.random.Generator,
    sp_q_net: Optional[torch.nn.Module] = None,
    pop_ensemble: Optional[torch.nn.Module] = None,
    lookahead: Optional[Any] = None,
) -> Tuple[Callable[[np.ndarray, Sequence[int], Optional[int]], int], str]:
    """Return (act_fn, tag).

    act_fn signature: (state_1ch, legal_actions, ply_idx) -> action

    Supported keys:
    - R / Random
    - LEFT / Leftmost
    - C / Center
    - SP
    - POP
    - Lk / LA-k / Lookahead-k
    """

    spec = key_to_opponent_mode(sampled_key)

    if spec.mode == "random":
        def _act(s, legal, ply_idx=None):
            return biased_random_choice(legal, rng=rng, bias="center")
        return _act, "R"

    if spec.mode == "leftmost":
        def _act(s, legal, ply_idx=None):
            legal = [int(a) for a in legal]
            if not legal:
                raise ValueError("No legal actions")
            return int(min(legal))
        return _act, "LEFT"

    if spec.mode == "center":
        def _act(s, legal, ply_idx=None):
            legal = [int(a) for a in legal]
            if not legal:
                raise ValueError("No legal actions")

            for c in (3, 4, 2, 5, 1, 6, 0):
                if c in legal:
                    return int(c)

            return int(legal[0])
        return _act, "CENTER"

    if spec.key == "SP":
        if sp_q_net is None:
            raise ValueError("SP requested but sp_q_net is None")

        def _act(s, legal, ply_idx=None):
            return greedy_model_action(sp_q_net, s, legal, device=device)

        return _act, "SP"

    if spec.key == "POP":
        if pop_ensemble is None:
            raise ValueError("POP requested but pop_ensemble is None")

        def _act(s, legal, ply_idx=None):
            return greedy_model_action(pop_ensemble, s, legal, device=device)

        return _act, "POP"

    # Lookahead
    depth = int(spec.mode)

    if lookahead is None:
        raise ValueError(f"{sampled_key} requested but lookahead is None")

    def _act(s, legal, ply_idx=None):
        legal = [int(a) for a in legal]

        if not legal:
            raise ValueError("No legal actions")

        # s is 1ch POV: shape (1,6,7) or (6,7),
        # with "to-move" stones positive, opponent stones negative.
        # Connect4Lookahead expects board {0,1,2} and player in {1,2}.
        if hasattr(lookahead, "n_step_lookahead"):
            arr = np.asarray(s)

            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]

            board012 = np.zeros((6, 7), dtype=np.int8)
            board012[arr > 0] = 1
            board012[arr < 0] = 2

            mv = int(lookahead.n_step_lookahead(board012, player=1, depth=depth))

            if mv in legal:
                return mv

            # Safety fallback if encoding/orientation mismatch ever appears.
            for c in (3, 4, 2, 5, 1, 6, 0):
                if c in legal:
                    return int(c)

            return int(legal[0])

        if hasattr(lookahead, "act"):
            mv = int(lookahead.act(
                state=s,
                legal_actions=list(legal),
                depth=depth,
                ply_idx=ply_idx,
            ))
            return mv if mv in legal else int(legal[0])

        if hasattr(lookahead, "best_action"):
            mv = int(lookahead.best_action(
                state=s,
                legal_actions=list(legal),
                depth=depth,
            ))
            return mv if mv in legal else int(legal[0])

        if hasattr(lookahead, "best_move"):
            mv = int(lookahead.best_move(
                state=s,
                legal_actions=list(legal),
                depth=depth,
            ))
            return mv if mv in legal else int(legal[0])

        raise AttributeError(
            "Lookahead object needs one of: "
            "n_step_lookahead(board, player, depth), "
            "act(state, legal_actions, depth, ply_idx), "
            "best_action(...), best_move(...)"
        )

    return _act, f"L{depth}"

# ----------------------------- episode play -----------------------------

def play_episode_dqn(
    *,
    env: Any,
    agent: Any,  # DQNAgent-like
    opponent_act: Callable[[np.ndarray, Sequence[int], Optional[int]], int],
    opponent_tag: str,
    store_opponent_moves: bool = True,
    epsilon_override: Optional[float] = None,
    max_ply: int = 42,
    alt_start: bool = True,
    episode_idx0: int = 0,
    rng: Optional[np.random.Generator] = None,
    opening_kpis: Optional[Dict[str, Any]] = None,
    openings: Optional[Any] = None,
) -> Tuple[float, Optional[float], int]:
    """Run one self-play episode.

    Returns: (total_reward, final_result, plies)
      final_result: 1.0 win, -1.0 loss, 0.5 draw (when detectable), else None

    Notes:
    - Stores transitions for BOTH players by default (store_opponent_moves=True).
    - Expects agent.remember_1step(state, action, reward, next_state, done).
    """

    if rng is None:
        rng = np.random.default_rng(episode_idx0)

    obs = env.reset()
    if hasattr(env, "board") and hasattr(env, "current_player"):
        b = np.asarray(env.board, dtype=np.int8)
        p = int(getattr(env, "current_player", 1))
        state = (b * p).astype(np.float32)[None, :, :]
    else:
        state = ensure_1ch_state(obs)

    ply = 0
    total_reward = 0.0
    final_result: Optional[float] = None

    # alternating start: opponent opens on odd episodes (0-based)
    opponent_starts = bool(alt_start and ((episode_idx0 % 2) == 1))

    def _log_opening(col: int, is_agent: bool, ply_idx: int):
        # Only track first move stats
        if openings is None:
            return
        if ply_idx != 0:
            return
        try:
            openings.on_first_move(int(col), bool(is_agent))
        except Exception:
            pass

    # optional opponent opening
    if opponent_starts:
        legal = legal_actions_from_state(state)
        a = opponent_act(state, legal, ply)
        obs2, r, done, _info = extract_step_out(env.step(a))
        if hasattr(env, "board") and hasattr(env, "current_player"):
            b2 = np.asarray(env.board, dtype=np.int8)
            p2 = int(getattr(env, "current_player", 1))
            s2 = (b2 * p2).astype(np.float32)[None, :, :]
        else:
            s2 = ensure_1ch_state(obs2)

        if store_opponent_moves:
            agent.remember_1step(state, a, r, s2, done)

        _log_opening(a, is_agent=False, ply_idx=ply)
        ply += 1
        total_reward += r
        state = s2

        if done:
            # reward is for the mover (opponent), so from current state POV it's terminal
            # final_result is from *agent* perspective, so flip sign
            final_result = float(-1.0 if r > 0 else (1.0 if r < 0 else 0.5))
            return total_reward, final_result, ply

    # main loop
    while ply < max_ply:
        # --- agent move ---
        legal = legal_actions_from_state(state)
        a = agent.act(state, legal, epsilon_override=epsilon_override)
        obs2, r, done, _info = extract_step_out(env.step(a))
        if hasattr(env, "board") and hasattr(env, "current_player"):
            b2 = np.asarray(env.board, dtype=np.int8)
            p2 = int(getattr(env, "current_player", 1))
            s2 = (b2 * p2).astype(np.float32)[None, :, :]
        else:
            s2 = ensure_1ch_state(obs2)
        agent.remember_1step(state, a, r, s2, done)
        _log_opening(a, is_agent=True, ply_idx=ply)

        ply += 1
        total_reward += r
        state = s2

        if done:
            final_result = float(1.0 if r > 0 else (-1.0 if r < 0 else 0.5))
            break

        # --- opponent move ---
        legal = legal_actions_from_state(state)
        ao = opponent_act(state, legal, ply)
        obs2, r, done, _info = extract_step_out(env.step(ao))
        if hasattr(env, "board") and hasattr(env, "current_player"):
            b2 = np.asarray(env.board, dtype=np.int8)
            p2 = int(getattr(env, "current_player", 1))
            s2 = (b2 * p2).astype(np.float32)[None, :, :]
        else:
            s2 = ensure_1ch_state(obs2)

        if store_opponent_moves:
            agent.remember_1step(state, ao, r, s2, done)

        ply += 1
        total_reward += r
        state = s2

        if done:
            # reward is for opponent, so flip for agent
            final_result = float(-1.0 if r > 0 else (1.0 if r < 0 else 0.5))
            break

    return total_reward, final_result, ply


def track_result(final_result: Optional[float], win_history: List[int]) -> Tuple[int, int, int]:
    """Update win_history with 1 win / 0 draw / -1 loss for plotting."""
    wins = losses = draws = 0
    if final_result is None:
        return 0, 0, 0

    if final_result > 0.75:
        wins = 1
        win_history.append(1)
    elif final_result < -0.75:
        losses = 1
        win_history.append(-1)
    else:
        draws = 1
        win_history.append(0)

    return wins, losses, draws
