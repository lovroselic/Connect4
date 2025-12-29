# PPO/ppo_agent_eval.py
"""\
Unified evaluation helpers for "new line" Connect-4 policies (CNet192 only).

This module intentionally drops ALL legacy ActorCritic / PPO.checkpoint compatibility.
It assumes your policy is a plain torch.nn.Module that takes a single-channel POV board
and outputs logits for 7 columns.

Board convention:
- numpy array shape (6,7), dtype int8
- values: 0 empty, +1 player1, -1 player2
- row 0 is the top row, row 5 is the bottom

Model convention (STRICT):
- single-channel POV input: x = (board * player_to_move) with shape (B,1,6,7)
- policy output:
    - logits: (B,7)
    - or (logits, value) where value is (B,) or (B,1)

Opponents supported:
- Random / Leftmost / Center baselines
- Lookahead-N via Connect4Lookahead (bitboard/Numba), if provided

Also includes:
- Excel logging helpers (append a row with win_rates per opponent)
- Exploitation histogram (greedy action frequency on random sampled states)
- Empty-board action distribution (softmax probs)
- Plot helpers (matplotlib figures)
- Head-to-head (H2H) helpers with shared openings + paired games
- Round-robin matrix + metascores helpers

Typical usage:

  from C4.eval_oppo_dict import EVAL_CFG
  from C4.fast_connect4_lookahead import Connect4Lookahead
  from PPO.ppo_agent_eval import (
      evaluate_and_log_to_excel,
      head_to_head_models, plot_winrate_bar,
      round_robin_matrix, metascores_from_matrix,
  )

  la = Connect4Lookahead()

  suite_df, row_df = evaluate_and_log_to_excel(
      policy=policy,
      opponents_cfg=EVAL_CFG,
      excel_path="EVAL_PPO_results.xlsx",
      run_tag=TRAINING_SESSION,
      device=DEVICE,
      lookahead=la,
      seed=SEED,
      episodes=meta.get("episode"),
  )

  res = head_to_head_models(policyA, policyB, n_games=400, opening_noise_k={0:0.7,1:0.2,2:0.1})
  plot_winrate_bar(res)

  rr = round_robin_matrix({"A": policyA, "B": policyB, "C": policyC}, n_games=200)
  ms = metascores_from_matrix(rr)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# Keep evaluation consistent with your env bitboard logic
from C4.connect4_env import (
    ROWS, COLS, STRIDE,
    TOP_MASK, FULL_MASK,
    _bb_has_won,
    UINT,
)

NEG_INF = -1e9

OpponentFn = Callable[[np.ndarray, int, np.uint64], int]


# ----------------------------- internal helpers -----------------------------

def _require_board(board: np.ndarray) -> None:
    if not isinstance(board, np.ndarray):
        raise TypeError(f"board must be np.ndarray, got {type(board)}")
    if board.shape != (ROWS, COLS):
        raise ValueError(f"board must have shape ({ROWS},{COLS}), got {board.shape}")
    if board.dtype != np.int8:
        # we keep it strict because this module is performance-sensitive
        raise TypeError(f"board dtype must be int8, got {board.dtype}")


def _model_device(m: torch.nn.Module) -> torch.device:
    for p in m.parameters():
        return p.device
    return torch.device("cpu")

def _assert_cnet192_1ch(m: torch.nn.Module, name: str = "model") -> None:
    # strict-ish: if it looks like CNet192, enforce 1ch
    conv1 = getattr(m, "conv1", None)
    if conv1 is None:
        return
    in_ch = getattr(conv1, "in_channels", None)
    if in_ch is None:
        try:
            in_ch = int(conv1.weight.shape[1])
        except Exception:
            return
    if int(in_ch) != 1:
        raise ValueError(f"{name} conv1 expects {in_ch} channels, but this eval module is 1-channel POV only.")
        
def _as_uint64(x: np.uint64 | int) -> np.uint64:
    return UINT(int(x))


# ----------------------------- core game utils (bitboard-backed) -----------------------------

@dataclass
class GameResult:
    winner: int              # +1, -1, or 0 (draw)
    plies: int
    final_board: np.ndarray  # (6,7) int8


def _legal_mask_to_cols(mask: np.uint64) -> List[int]:
    out: List[int] = []
    for c in range(COLS):
        if (mask & TOP_MASK[c]) == 0:
            out.append(c)
    return out


def _apply_action_inplace(
    board: np.ndarray,
    heights: np.ndarray,
    pos1: np.uint64,
    pos2: np.uint64,
    mask: np.uint64,
    col: int,
    player: int,
) -> Tuple[np.uint64, np.uint64, np.uint64]:
    """Apply a move in-place on board/heights and return updated (pos1,pos2,mask)."""
    _require_board(board)

    c = int(col)
    if c < 0 or c >= COLS:
        raise ValueError(f"Illegal move: col={c}")

    h = int(heights[c])
    if h >= ROWS:
        raise ValueError(f"Illegal move: column {c} is full")

    bit = _as_uint64(1) << _as_uint64(c * STRIDE + h)
    mask = mask | bit
    if player == 1:
        pos1 = pos1 | bit
    elif player == -1:
        pos2 = pos2 | bit
    else:
        raise ValueError(f"player must be +1 or -1, got {player}")

    heights[c] = h + 1
    board[ROWS - 1 - h, c] = np.int8(player)
    return pos1, pos2, mask


def _sample_random_state(
    rng: np.random.Generator,
    max_random_plies: int = 12,
) -> Tuple[np.ndarray, np.uint64, int]:
    """Sample (board, mask, player_to_move) by playing random legal moves from empty."""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    heights = np.zeros(COLS, dtype=np.int8)
    pos1 = UINT(0)
    pos2 = UINT(0)
    mask = UINT(0)

    player = 1
    n = int(rng.integers(0, int(max_random_plies) + 1))

    for _ in range(n):
        if mask == FULL_MASK:
            break
        if _bb_has_won(pos1, STRIDE) or _bb_has_won(pos2, STRIDE):
            break

        legal = _legal_mask_to_cols(mask)
        if not legal:
            break

        a = int(rng.choice(legal))
        pos1, pos2, mask = _apply_action_inplace(board, heights, pos1, pos2, mask, a, player)

        me_bb = pos1 if player == 1 else pos2
        if _bb_has_won(me_bb, STRIDE) or (mask == FULL_MASK):
            player = -player
            break

        player = -player

    return board, mask, player


# ----------------------------- model action utils -----------------------------

@torch.no_grad()
def board_to_pov_tensor(board: np.ndarray, player: int, device: torch.device) -> torch.Tensor:
    """(6,7) int8 -> (1,1,6,7) float32 POV tensor."""
    _require_board(board)
    if player not in (+1, -1):
        raise ValueError(f"player must be +1/-1, got {player}")
    x = (board.astype(np.float32) * float(player))[None, None, :, :]
    return torch.from_numpy(x).to(device)


@torch.no_grad()
def masked_logits(
    model: torch.nn.Module,
    board: np.ndarray,
    player: int,
    device: torch.device,
    mask: Optional[np.uint64] = None,
) -> torch.Tensor:
    """Return (7,) logits with illegal columns masked to a very negative number."""
    x = board_to_pov_tensor(board, player, device)
    out = model(x)

    logits = out[0] if isinstance(out, (tuple, list)) else out

    if not isinstance(logits, torch.Tensor):
        raise TypeError("Model output must be a Tensor or (Tensor, value)")

    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != COLS:
        raise ValueError(f"Model logits must have shape (1,{COLS}), got {tuple(logits.shape)}")

    logits = logits[0].clone()  # (7,)

    # mask illegal
    if mask is None:
        # derive legality from board top row
        for c in range(COLS):
            if board[0, c] != 0:
                logits[c] = -1e9
    else:
        mask = _as_uint64(mask)
        for c in range(COLS):
            if (mask & TOP_MASK[c]) != 0:
                logits[c] = -1e9

    return logits


def _argmax_legal_center_tiebreak(
    logits: torch.Tensor,
    legal: List[int],
    center: int = 3,
    tol: float = 1e-12,
) -> int:
    """Deterministic argmax with tie-break preferring the center column."""
    if not legal:
        raise ValueError("No legal actions")

    # logits is length COLS
    vals = logits.detach().cpu().numpy().astype(np.float64, copy=False)
    best = np.max([vals[c] for c in legal])

    tied = [c for c in legal if abs(vals[c] - best) <= tol]
    if len(tied) == 1:
        return int(tied[0])

    # prefer closeness to center, then smallest index
    return int(sorted(tied, key=lambda c: (abs(c - center), c))[0])


@torch.no_grad()
def policy_action(
    model: torch.nn.Module,
    board: np.ndarray,
    player: int,
    device: torch.device,
    deterministic: bool = True,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    mask: Optional[np.uint64] = None,
) -> int:
    """Pick an action for `player` from the model, with legality masking."""
    if rng is None:
        rng = np.random.default_rng()

    logits = masked_logits(model, board, player, device, mask=mask)

    legal = _legal_mask_to_cols(_as_uint64(mask) if mask is not None else UINT(0)) if mask is not None else [c for c in range(COLS) if board[0, c] == 0]
    if not legal:
        return 0

    if deterministic or temperature <= 0:
        return _argmax_legal_center_tiebreak(logits, legal)

    probs = torch.softmax(logits / float(temperature), dim=0).detach().cpu().numpy()
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    s = float(probs.sum())
    if s <= 0:
        return int(rng.choice(legal))

    probs /= s

    # sample only from legal (renormalize)
    p_legal = np.array([probs[c] for c in legal], dtype=np.float64)
    ss = float(p_legal.sum())
    if ss <= 0:
        return int(rng.choice(legal))
    p_legal /= ss
    return int(rng.choice(np.array(legal, dtype=np.int64), p=p_legal))


@torch.no_grad()
def policy_probs_on_board(
    model: torch.nn.Module,
    board: np.ndarray,
    player: int,
    device: torch.device,
    temperature: float = 1.0,
    mask: Optional[np.uint64] = None,
) -> np.ndarray:
    """Softmax probabilities over actions, with illegal actions masked out. Returns probs[7]."""
    logits = masked_logits(model, board, player, device, mask=mask)
    t = float(temperature) if temperature and temperature > 0 else 1.0
    probs = torch.softmax(logits / t, dim=0).detach().cpu().numpy().astype(np.float64)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    # safety: re-zero illegal explicitly
    if mask is None:
        for c in range(COLS):
            if board[0, c] != 0:
                probs[c] = 0.0
    else:
        mask = _as_uint64(mask)
        for c in range(COLS):
            if (mask & TOP_MASK[c]) != 0:
                probs[c] = 0.0

    s = float(probs.sum())
    if s <= 0:
        return np.zeros(COLS, dtype=np.float64)
    return (probs / s).astype(np.float64)


# ----------------------------- opponent factory -----------------------------

def make_opponent(
    label: str,
    lookahead: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
) -> OpponentFn:
    """Create an opponent function from a label."""
    if rng is None:
        rng = np.random.default_rng()

    s = (label or "").strip().lower()

    has_la_baselines = (lookahead is not None) and hasattr(lookahead, "baseline_action")

    # --- baselines ---
    if s in ("random", "rnd"):
        if has_la_baselines:
            def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
                return int(lookahead.baseline_action(board, kind="random", rng=rng))
            return _op
        def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
            legal = _legal_mask_to_cols(_as_uint64(mask))
            return int(rng.choice(legal)) if legal else 0
        return _op

    if s in ("leftmost", "left", "lm"):
        if has_la_baselines:
            def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
                return int(lookahead.baseline_action(board, kind="leftmost", rng=rng))
            return _op
        def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
            mask = _as_uint64(mask)
            for c in range(COLS):
                if (mask & TOP_MASK[c]) == 0:
                    return int(c)
            return 0
        return _op

    if s in ("center", "centre"):
        if has_la_baselines:
            def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
                return int(lookahead.baseline_action(board, kind="center", rng=rng))
            return _op
        order = (3, 4, 2, 5, 1, 6, 0)
        def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
            mask = _as_uint64(mask)
            for c in order:
                if (mask & TOP_MASK[c]) == 0:
                    return int(c)
            return 0
        return _op

    # --- lookahead ---
    if s.startswith("lookahead"):
        if lookahead is None:
            raise ValueError("Lookahead opponent requested but lookahead=None. Pass your Connect4Lookahead instance.")
        depth = int("".join(ch for ch in s if ch.isdigit()) or "3")

        def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
            # NOTE: lookahead implementation is assumed to handle legality internally.
            return int(lookahead.n_step_lookahead(board, player, depth=depth))
        return _op

    raise ValueError(f"Unknown opponent label: {label!r}")


# ----------------------------- evaluation (policy vs opponent) -----------------------------

def play_one_game(
    policy: torch.nn.Module,
    opponent: OpponentFn,
    device: torch.device,
    seed: int = 666,
    policy_player: int = 1,  # +1 => policy plays first, -1 => policy plays second
    policy_deterministic: bool = True,
    policy_temperature: float = 1.0,
    max_plies: int = 42,
) -> GameResult:
    """Play one game policy vs opponent using bitboards for speed."""
    if policy_player not in (+1, -1):
        raise ValueError(f"policy_player must be +1/-1, got {policy_player}")

    rng = np.random.default_rng(int(seed))

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    heights = np.zeros(COLS, dtype=np.int8)

    pos1 = UINT(0)
    pos2 = UINT(0)
    mask = UINT(0)

    player = 1
    plies = 0

    was_training = bool(getattr(policy, "training", False))
    policy.eval()

    for _ in range(int(max_plies)):
        if player == policy_player:
            a = policy_action(
                policy, board, player,
                device=device,
                deterministic=policy_deterministic,
                temperature=policy_temperature,
                rng=rng,
                mask=mask,
            )
        else:
            a = int(opponent(board, player, mask))

        # illegal => immediate loss for mover
        if (a < 0) or (a >= COLS) or ((mask & TOP_MASK[a]) != 0):
            if was_training:
                policy.train(True)
            return GameResult(winner=-player, plies=plies, final_board=board)

        pos1, pos2, mask = _apply_action_inplace(board, heights, pos1, pos2, mask, a, player)
        plies += 1

        me_bb = pos1 if player == 1 else pos2
        if _bb_has_won(me_bb, STRIDE):
            if was_training:
                policy.train(True)
            return GameResult(winner=player, plies=plies, final_board=board)

        if mask == FULL_MASK:
            if was_training:
                policy.train(True)
            return GameResult(winner=0, plies=plies, final_board=board)

        player = -player

    if was_training:
        policy.train(True)
    return GameResult(winner=0, plies=plies, final_board=board)


def evaluate_vs_opponent(
    policy: torch.nn.Module,
    opponent_label: str,
    device: torch.device,
    lookahead: Optional[Any] = None,
    n_games: int = 200,
    seed: int = 666,
    swap_sides: bool = True,
    policy_deterministic: bool = True,
    policy_temperature: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate policy vs a given opponent label."""
    n_games = int(n_games)
    if n_games <= 0:
        raise ValueError("n_games must be > 0")

    if swap_sides and (n_games % 2 == 1):
        n_games += 1  # keep balanced starts

    rng = np.random.default_rng(int(seed))
    opp = make_opponent(opponent_label, lookahead=lookahead, rng=rng)

    wins = losses = draws = 0
    total_plies = 0

    was_training = bool(getattr(policy, "training", False))
    policy.eval()

    for i in range(n_games):
        policy_player = 1 if (not swap_sides or (i % 2 == 0)) else -1

        gr = play_one_game(
            policy, opp, device=device,
            seed=int(rng.integers(0, 2**31 - 1)),
            policy_player=policy_player,
            policy_deterministic=policy_deterministic,
            policy_temperature=policy_temperature,
        )

        total_plies += gr.plies

        if gr.winner == 0:
            draws += 1
        elif gr.winner == policy_player:
            wins += 1
        else:
            losses += 1

    if was_training:
        policy.train(True)

    score = (wins + 0.5 * draws) / float(n_games)

    return {
        "opponent": str(opponent_label),
        "games": int(n_games),
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "win_rate": float(wins / n_games),
        "loss_rate": float(losses / n_games),
        "draw_rate": float(draws / n_games),
        "score": float(score),
        "avg_plies": float(total_plies / n_games),
    }


def evaluate_suite(
    policy: torch.nn.Module,
    opponents: Union[Dict[str, int], Iterable[str]],
    device: torch.device,
    lookahead: Optional[Any] = None,
    n_games_default: int = 200,
    seed: int = 666,
    swap_sides: bool = True,
    policy_deterministic: bool = True,
    policy_temperature: float = 1.0,
    progress: bool = True,
) -> pd.DataFrame:
    """Evaluate policy against a suite of opponents."""
    items: List[Tuple[str, int]]
    if isinstance(opponents, dict):
        items = [(str(k), int(v)) for k, v in opponents.items()]
    else:
        items = [(str(o), int(n_games_default)) for o in opponents]

    rows: List[Dict[str, Any]] = []

    if progress:
        pbar = tqdm(items, desc="Eval", leave=True)
        for label, ng in pbar:
            pbar.set_description(f"Eval vs {label} ({ng})")
            rows.append(
                evaluate_vs_opponent(
                    policy,
                    opponent_label=label,
                    device=device,
                    lookahead=lookahead,
                    n_games=int(ng),
                    seed=seed,
                    swap_sides=swap_sides,
                    policy_deterministic=policy_deterministic,
                    policy_temperature=policy_temperature,
                )
            )
    else:
        for label, ng in items:
            rows.append(
                evaluate_vs_opponent(
                    policy,
                    opponent_label=label,
                    device=device,
                    lookahead=lookahead,
                    n_games=int(ng),
                    seed=seed,
                    swap_sides=swap_sides,
                    policy_deterministic=policy_deterministic,
                    policy_temperature=policy_temperature,
                )
            )

    return pd.DataFrame(rows)


# ----------------------------- Excel logging helpers -----------------------------

def suite_df_to_row(
    run_tag: str,
    suite_df: pd.DataFrame,
    elapsed_h: float,
    episodes: Optional[int] = None,
) -> pd.DataFrame:
    """Convert evaluate_suite() DataFrame into a single-row DataFrame with win_rates."""
    row: Dict[str, Any] = {
        "TRAINING_SESSION": str(run_tag),
        "TIME [h]": round(float(elapsed_h), 6),
        "EPISODES": episodes,
    }

    for _, r in suite_df.iterrows():
        row[str(r["opponent"])] = float(r["win_rate"])

    return pd.DataFrame([row]).set_index("TRAINING_SESSION")


def append_eval_row_to_excel(df_row: pd.DataFrame, excel_path: str) -> None:
    """Append df_row (single row with TRAINING_SESSION index) to an Excel file."""
    excel_path = str(excel_path)

    if os.path.exists(excel_path):
        old = pd.read_excel(excel_path)
        if not old.empty and not old.isnull().all().all():
            new = pd.concat([old, df_row.reset_index()], ignore_index=True)
        else:
            new = df_row.reset_index()
    else:
        new = df_row.reset_index()

    new.to_excel(excel_path, index=False)


def evaluate_and_log_to_excel(
    policy: torch.nn.Module,
    opponents_cfg: Union[Dict[str, int], Iterable[str]],
    excel_path: str,
    run_tag: str,
    device: torch.device,
    lookahead: Optional[Any] = None,
    seed: int = 666,
    swap_sides: bool = True,
    policy_deterministic: bool = True,
    policy_temperature: float = 1.0,
    episodes: Optional[int] = None,
    progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One-call convenience: evaluate suite + append win-rates row into Excel."""
    t0 = time.time()

    suite_df = evaluate_suite(
        policy=policy,
        opponents=opponents_cfg,
        device=device,
        lookahead=lookahead,
        seed=seed,
        swap_sides=swap_sides,
        policy_deterministic=policy_deterministic,
        policy_temperature=policy_temperature,
        progress=progress,
    )

    elapsed_h = (time.time() - t0) / 3600.0
    row_df = suite_df_to_row(run_tag=run_tag, suite_df=suite_df, elapsed_h=elapsed_h, episodes=episodes)
    append_eval_row_to_excel(row_df, excel_path)

    return suite_df, row_df


# ----------------------------- exploitation histogram + plots -----------------------------

@torch.inference_mode()
def exploitation_histogram(
    policy: torch.nn.Module,
    device: torch.device,
    n_states: int = 256,
    rng_seed: int = 0,
    max_random_plies: int = 12,
    deterministic: bool = True,
    temperature: float = 1.0,
) -> np.ndarray:
    """Counts greedy (or sampled) action choices on random sampled states. Returns counts[7]."""
    counts = np.zeros(COLS, dtype=np.int64)
    rng = np.random.default_rng(int(rng_seed))

    was_training = bool(getattr(policy, "training", False))
    policy.eval()

    for _ in range(int(n_states)):
        board, mask, player = _sample_random_state(rng, max_random_plies=int(max_random_plies))
        a = policy_action(
            policy, board, player,
            device=device,
            deterministic=deterministic,
            temperature=temperature,
            rng=rng,
            mask=mask,
        )
        if 0 <= a < COLS:
            counts[a] += 1

    if was_training:
        policy.train(True)

    return counts


@torch.inference_mode()
def empty_board_action_distribution(
    policy: torch.nn.Module,
    device: torch.device,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Returns greedy action + softmax probs on empty board (player=+1)."""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    mask = UINT(0)
    player = 1

    was_training = bool(getattr(policy, "training", False))
    policy.eval()

    probs = policy_probs_on_board(policy, board, player, device, temperature=float(temperature), mask=mask)

    legal = list(range(COLS))
    greedy = _argmax_legal_center_tiebreak(torch.from_numpy(probs.astype(np.float64)), legal) if probs.sum() > 0 else 0

    if was_training:
        policy.train(True)

    return {"greedy": int(greedy), "probs": probs.astype(np.float64)}


def plot_exploitation_histogram(
    counts: np.ndarray,
    model_name: str = "policy",
):
    """Returns a matplotlib Figure."""
    import matplotlib.pyplot as plt

    counts = np.asarray(counts, dtype=np.int64)
    fig = plt.figure(figsize=(7, 3.5))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(COLS), counts)
    ax.set_title(f"Exploitation histogram, {model_name}")
    ax.set_xlabel("action (column)")
    ax.set_ylabel("count")
    ax.set_xticks(np.arange(COLS))
    fig.tight_layout()
    return fig


def plot_empty_board_action_distribution(
    policy: torch.nn.Module,
    device: torch.device,
    model_name: str = "policy",
    temperature: float = 1.0,
):
    """Bar plot of action probabilities on the empty board. Returns a matplotlib Figure."""
    import matplotlib.pyplot as plt

    info = empty_board_action_distribution(policy, device=device, temperature=float(temperature))
    probs = np.asarray(info["probs"], dtype=np.float64)
    greedy = int(info["greedy"])

    fig = plt.figure(figsize=(7, 3.5))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(COLS), probs)
    ax.set_title(f"Empty-board action probs, {model_name} (greedy={greedy})")
    ax.set_xlabel("action (column)")
    ax.set_ylabel("probability")
    ax.set_xticks(np.arange(COLS))
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    return fig


# ----------------------------- summary print + bar plot from suite_df -----------------------------

def print_eval_summary_from_suite_df(suite_df: pd.DataFrame) -> None:
    """Pretty-print a per-opponent summary from evaluate_suite() DataFrame."""
    print("\n📊 Evaluation Summary:")
    for _, r in suite_df.iterrows():
        label = str(r["opponent"])
        w = int(r.get("wins", 0))
        l = int(r.get("losses", 0))
        d = int(r.get("draws", 0))
        wr = float(r.get("win_rate", 0.0)) * 100.0
        lr = float(r.get("loss_rate", 0.0)) * 100.0
        dr = float(r.get("draw_rate", 0.0)) * 100.0
        print(f"{label}: {w}W / {l}L / {d}D → Win: {wr:.1f}%, Loss: {lr:.1f}%, Draw: {dr:.1f}%")


def plot_eval_bar_summary_from_suite_df(
    suite_df: pd.DataFrame,
    model_name: str = "policy",
    save_path: Optional[str] = None,
    rotation: int = 15,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 150,
):
    """Grouped bar plot: Win/Loss/Draw percentages per opponent."""
    import matplotlib.pyplot as plt

    labels = [str(x) for x in suite_df["opponent"].tolist()]
    win_rates = (suite_df["win_rate"].astype(float) * 100.0).tolist()
    loss_rates = (suite_df["loss_rate"].astype(float) * 100.0).tolist()
    draw_rates = (suite_df["draw_rate"].astype(float) * 100.0).tolist()

    x = np.arange(len(labels), dtype=np.float64)
    bar_width = 0.25

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.bar(x, win_rates, width=bar_width, label="Win %")
    ax.bar(x + bar_width, loss_rates, width=bar_width, label="Loss %")
    ax.bar(x + 2 * bar_width, draw_rates, width=bar_width, label="Draw %")

    ax.set_xlabel("Opponent")
    ax.set_ylabel("Percentage")
    ax.set_title(f"{model_name} vs opponents")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(labels, rotation=rotation, ha="right")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=int(dpi))
        print(f"📊 Plot saved to {save_path}")

    return fig


# ----------------------------- compatibility wrapper (board display expects this) -----------------------------

def play_single_game_ppo(
    policy: torch.nn.Module,
    opponent_label: str = "Lookahead-1",
    seed: int = 666,
    lookahead: Optional[Any] = None,
    device: Optional[torch.device] = None,
    policy_deterministic: bool = True,
    policy_temperature: float = 1.0,
) -> Tuple[int, np.ndarray]:
    """Returns (result, final_board) where result is +1 win, 0 draw, -1 loss (policy as player1)."""
    if device is None:
        device = _model_device(policy)

    rng = np.random.default_rng(int(seed))
    opp = make_opponent(opponent_label, lookahead=lookahead, rng=rng)

    gr = play_one_game(
        policy, opp, device=device,
        seed=int(seed),
        policy_player=1,
        policy_deterministic=policy_deterministic,
        policy_temperature=policy_temperature,
    )

    if gr.winner == 0:
        r = 0
    elif gr.winner == 1:
        r = 1
    else:
        r = -1

    return int(r), gr.final_board


# ----------------------------- H2H utilities (shared openings + paired games) -----------------------------

def mean_ci(scores: List[float], z: float = 1.96) -> Tuple[float, float]:
    x = np.asarray(scores, dtype=np.float64)
    n = int(x.size)
    if n <= 0:
        return (0.0, 1.0)
    m = float(x.mean())
    if n <= 1:
        return (max(0.0, m), min(1.0, m))
    s = float(x.std(ddof=1))
    half = float(z) * s / (n ** 0.5)
    return (max(0.0, m - half), min(1.0, m + half))


def _resolve_opening_k(rng: random.Random, opening_noise_k: Union[int, Dict[int, float]]) -> int:
    """
    opening_noise_k can be:
      - int: fixed k
      - dict[int->float]: categorical distribution over k, e.g. {0:0.70, 1:0.20, 2:0.10}
    """
    if isinstance(opening_noise_k, dict):
        items = [(int(k), float(p)) for k, p in opening_noise_k.items() if float(p) > 0.0]
        if not items:
            return 0
        ks, ps = zip(*items)
        s = float(sum(ps))
        psn = [p / s for p in ps]
        r = rng.random()
        c = 0.0
        for k, p in zip(ks, psn):
            c += p
            if r <= c:
                return int(max(0, k))
        return int(max(0, ks[-1]))
    return int(max(0, int(opening_noise_k)))


def _sample_opening_action(
    rng: random.Random,
    legal: List[int],
    bias: str = "uniform",
    center: int = 3,
) -> int:
    if not legal:
        raise ValueError("No legal actions for opening move.")

    if bias == "uniform":
        return int(rng.choice(legal)) if hasattr(rng, "choice") else int(legal[int(rng.random() * len(legal))])

    if bias == "center":
        # dist 0:1.00, dist 1:0.55, dist 2:0.25, dist 3:0.10, else:0.05
        wmap = {0: 1.00, 1: 0.55, 2: 0.25, 3: 0.10}
        weights = [wmap.get(abs(int(c) - center), 0.05) for c in legal]
        s = float(sum(weights))
        if s <= 0.0:
            return int(legal[int(rng.random() * len(legal))])
        r = rng.random() * s
        acc = 0.0
        for c, w in zip(legal, weights):
            acc += float(w)
            if r <= acc:
                return int(c)
        return int(legal[-1])

    raise ValueError(f"Unknown opening bias: {bias!r}")


def _generate_opening_moves(
    opening_noise_k: Union[int, Dict[int, float]],
    seed: int,
    opening_bias: str = "uniform",
) -> List[int]:
    """
    Generate a legal opening sequence from empty board using bitboard rules.
    Independent of A/B.
    """
    rng = random.Random(int(seed))

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    heights = np.zeros(COLS, dtype=np.int8)
    pos1 = UINT(0)
    pos2 = UINT(0)
    mask = UINT(0)

    player = 1
    moves: List[int] = []
    k = _resolve_opening_k(rng, opening_noise_k)

    for _ in range(int(k)):
        if mask == FULL_MASK:
            break
        if _bb_has_won(pos1, STRIDE) or _bb_has_won(pos2, STRIDE):
            break

        legal = _legal_mask_to_cols(mask)
        if not legal:
            break

        a = _sample_opening_action(rng, legal, bias=opening_bias)
        moves.append(int(a))

        pos1, pos2, mask = _apply_action_inplace(board, heights, pos1, pos2, mask, int(a), player)

        me_bb = pos1 if player == 1 else pos2
        if _bb_has_won(me_bb, STRIDE) or (mask == FULL_MASK):
            player = -player
            break

        player = -player

    return moves


@torch.no_grad()
def play_game_h2h(
    A: torch.nn.Module,
    B: torch.nn.Module,
    device: torch.device,
    A_color: int = +1,
    opening_noise_k: Union[int, Dict[int, float]] = 0,
    seed: int = 0,
    opening_moves: Optional[List[int]] = None,
    opening_bias: str = "uniform",
    deterministic: bool = True,
    temperature: float = 1.0,
    max_plies: int = 42,
) -> float:
    """
    One H2H game, greedy by default.

    Returns:
      1.0 if A wins, 0.5 draw, 0.0 if A loses
    """
    _assert_cnet192_1ch(A, "A")
    _assert_cnet192_1ch(B, "B")

    rng = np.random.default_rng(int(seed))
    rng_open = random.Random(int(seed))

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    heights = np.zeros(COLS, dtype=np.int8)
    pos1 = UINT(0)
    pos2 = UINT(0)
    mask = UINT(0)

    player = 1
    plies = 0

    # Apply opening (shared or sampled)
    if opening_moves is None:
        k = _resolve_opening_k(rng_open, opening_noise_k)
        for _ in range(int(k)):
            if mask == FULL_MASK:
                break
            legal = _legal_mask_to_cols(mask)
            if not legal:
                break
            a = _sample_opening_action(rng_open, legal, bias=opening_bias)

            pos1, pos2, mask = _apply_action_inplace(board, heights, pos1, pos2, mask, int(a), player)
            plies += 1

            me_bb = pos1 if player == 1 else pos2
            if _bb_has_won(me_bb, STRIDE):
                winner = player
                return 1.0 if winner == int(A_color) else 0.0
            if mask == FULL_MASK:
                return 0.5

            player = -player
    else:
        for a0 in opening_moves:
            if mask == FULL_MASK:
                break
            legal = _legal_mask_to_cols(mask)
            if not legal:
                break

            a = int(a0)
            if a not in legal:
                # safety fallback (shouldn't happen)
                a = int(rng.choice(legal))

            pos1, pos2, mask = _apply_action_inplace(board, heights, pos1, pos2, mask, a, player)
            plies += 1

            me_bb = pos1 if player == 1 else pos2
            if _bb_has_won(me_bb, STRIDE):
                winner = player
                return 1.0 if winner == int(A_color) else 0.0
            if mask == FULL_MASK:
                return 0.5

            player = -player

    # Main play
    for _ in range(int(max_plies - plies)):
        if mask == FULL_MASK:
            return 0.5

        legal = _legal_mask_to_cols(mask)
        if not legal:
            return 0.5

        model = A if player == int(A_color) else B
        a = policy_action(
            model,
            board,
            player,
            device=device,
            deterministic=deterministic,
            temperature=temperature,
            rng=rng,
            mask=mask,
        )

        # illegal -> immediate loss for mover
        if (a < 0) or (a >= COLS) or ((mask & TOP_MASK[int(a)]) != 0):
            winner = -player
            return 1.0 if winner == int(A_color) else 0.0

        pos1, pos2, mask = _apply_action_inplace(board, heights, pos1, pos2, mask, int(a), player)

        me_bb = pos1 if player == 1 else pos2
        if _bb_has_won(me_bb, STRIDE):
            winner = player
            return 1.0 if winner == int(A_color) else 0.0

        if mask == FULL_MASK:
            return 0.5

        player = -player

    return 0.5


@torch.inference_mode()
def head_to_head_models(
    A: torch.nn.Module,
    B: torch.nn.Module,
    n_games: int = 200,
    A_label: str = "A",
    B_label: str = "B",
    device: Optional[torch.device] = None,
    opening_noise_k: Union[int, Dict[int, float]] = 0,
    seed: int = 123,
    progress: bool = True,
    paired_openings: bool = True,
    opening_bias: str = "uniform",
) -> Dict[str, Any]:
    """Head-to-head match between two 1-channel POV models (CNet192 family)."""
    if device is None:
        device = _model_device(A)

    _assert_cnet192_1ch(A, "A")
    _assert_cnet192_1ch(B, "B")

    rng = random.Random(int(seed))

    wasA = bool(getattr(A, "training", False))
    wasB = bool(getattr(B, "training", False))
    A.eval()
    B.eval()

    timeline: List[float] = []
    wins = draws = losses = 0
    total = int(n_games)
    if total <= 0:
        raise ValueError("n_games must be > 0")

    if not paired_openings:
        colors = ([+1, -1] * ((total + 1) // 2))[:total]
        iterable = tqdm(colors, total=total, desc=f"{A_label} vs {B_label}") if progress else colors

        for A_color in iterable:
            res = play_game_h2h(
                A, B,
                device=device,
                A_color=int(A_color),
                opening_noise_k=opening_noise_k,
                seed=rng.randint(0, 10**9),
                opening_moves=None,
                opening_bias=opening_bias,
                deterministic=True,
                temperature=1.0,
            )
            timeline.append(float(res))
            if res == 1.0:
                wins += 1
            elif res == 0.5:
                draws += 1
            else:
                losses += 1

            if progress:
                g = len(timeline)
                score = (wins + 0.5 * draws) / g
                iterable.set_postfix(W=wins, D=draws, L=losses, score=f"{score:.3f}")

    else:
        n_pairs = total // 2
        extra = total % 2

        pbar = tqdm(total=total, desc=f"{A_label} vs {B_label}") if progress else None

        def _update_postfix():
            if pbar is None:
                return
            g = len(timeline)
            score = (wins + 0.5 * draws) / g if g else 0.0
            pbar.set_postfix(W=wins, D=draws, L=losses, score=f"{score:.3f}")

        for _ in range(int(n_pairs)):
            opening_seed = rng.randint(0, 10**9)
            opening_moves = _generate_opening_moves(opening_noise_k, seed=opening_seed, opening_bias=opening_bias)

            for A_color in (+1, -1):
                res = play_game_h2h(
                    A, B,
                    device=device,
                    A_color=int(A_color),
                    opening_noise_k=0,          # opening controlled by opening_moves
                    seed=rng.randint(0, 10**9),
                    opening_moves=opening_moves,
                    opening_bias=opening_bias,
                    deterministic=True,
                    temperature=1.0,
                )
                timeline.append(float(res))
                if res == 1.0:
                    wins += 1
                elif res == 0.5:
                    draws += 1
                else:
                    losses += 1

                if pbar is not None:
                    pbar.update(1)
                    _update_postfix()

        if extra:
            opening_seed = rng.randint(0, 10**9)
            opening_moves = _generate_opening_moves(opening_noise_k, seed=opening_seed, opening_bias=opening_bias)
            res = play_game_h2h(
                A, B,
                device=device,
                A_color=+1,
                opening_noise_k=0,
                seed=rng.randint(0, 10**9),
                opening_moves=opening_moves,
                opening_bias=opening_bias,
                deterministic=True,
                temperature=1.0,
            )
            timeline.append(float(res))
            if res == 1.0:
                wins += 1
            elif res == 0.5:
                draws += 1
            else:
                losses += 1

            if pbar is not None:
                pbar.update(1)
                _update_postfix()

        if pbar is not None:
            pbar.close()

    n = len(timeline)
    score = (wins + 0.5 * draws) / n if n else 0.0
    ci95 = mean_ci(timeline, z=1.96)

    if wasA:
        A.train(True)
    if wasB:
        B.train(True)

    return {
        "A_path": str(A_label),
        "B_path": str(B_label),
        "games": int(n),
        "A_wins": int(wins),
        "draws": int(draws),
        "A_losses": int(losses),
        "A_score_rate": float(score),
        "A_score_CI95": tuple(ci95),
        "paired_openings": bool(paired_openings),
        "opening_bias": str(opening_bias),
        "opening_noise_k": opening_noise_k,
    }


def plot_winrate_bar(res: Dict[str, Any], title: Optional[str] = None, show_score_bars: bool = True):
    """Quick W/D/L bar + optional overall score bar. Returns matplotlib Figures."""
    import matplotlib.pyplot as plt

    n = int(res["games"])
    Aw = int(res["A_wins"])
    Ad = int(res["draws"])
    Al = int(res["A_losses"])
    A_score = float(res["A_score_rate"])
    B_score = 1.0 - A_score

    pw, pd, pl = (Aw / n if n else 0.0), (Ad / n if n else 0.0), (Al / n if n else 0.0)

    a_name = str(res.get("A_path", "A"))
    b_name = str(res.get("B_path", "B"))
    title = title or f"{a_name} vs {b_name}"

    fig1, ax = plt.subplots(figsize=(6.2, 1.6))
    left = 0.0
    ax.barh([0], [pw], left=left, label="Win")
    left += pw
    ax.barh([0], [pd], left=left, label="Draw")
    left += pd
    ax.barh([0], [pl], left=left, label="Loss")

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Proportion")
    ax.set_title(f"{title} — A Win/Draw/Loss")

    if n > 0:
        ax.text(pw / 2 if pw > 0 else 0.01, 0, f"W {Aw}", ha="center", va="center", fontsize=10)
        ax.text(pw + (pd / 2 if pd > 0 else 0.01), 0, f"D {Ad}", ha="center", va="center", fontsize=10)
        ax.text(pw + pd + (pl / 2 if pl > 0 else 0.01), 0, f"L {Al}", ha="center", va="center", fontsize=10)

    ax.legend(ncols=3, loc="lower right", frameon=False)
    plt.tight_layout()

    fig2 = None
    if show_score_bars:
        fig2, ax2 = plt.subplots(figsize=(4.8, 3.2))
        ax2.bar([a_name, b_name], [A_score, B_score])
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Score rate")
        ax2.set_title(f"{title} — overall score")
        for i, v in enumerate([A_score, B_score]):
            ax2.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        ax2.grid(axis="y", ls="--", alpha=0.3)
        plt.tight_layout()

    return fig1, fig2


# ----------------------------- round-robin + metascores -----------------------------

@torch.inference_mode()
def round_robin_matrix(
    models: Union[Dict[str, torch.nn.Module], List[Tuple[str, torch.nn.Module]]],
    n_games: int = 200,
    device: Optional[torch.device] = None,
    opening_noise_k: Union[int, Dict[int, float]] = 0,
    seed: int = 123,
    progress: bool = True,
    paired_openings: bool = True,
    opening_bias: str = "uniform",
    with_meta_column: bool = False,
    sort_by_meta: bool = False,
    return_details: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
    """
    Round-robin head-to-head matrix for 1-channel POV models (CNet192 family).

    Returns:
      rr: square DataFrame (N x N), rr.loc[A,B] = A score rate vs B in [0,1]
          diagonal is set to 0.5 by convention.

    If with_meta_column=True:
      adds rr["meta_score"] (row-average vs others; diagonal ignored).

    If sort_by_meta=True (requires with_meta_column=True):
      sorts rows/cols by meta_score descending.

    If return_details=True:
      returns (rr, details) where details are head_to_head_models() dicts per pair.
    """
    # normalize model list
    if isinstance(models, dict):
        items = [(str(k), v) for k, v in models.items()]
    else:
        items = [(str(k), v) for (k, v) in list(models)]

    names = [k for (k, _) in items]
    mods  = [m for (_, m) in items]

    if len(names) < 2:
        raise ValueError("round_robin_matrix needs at least 2 models")
    if int(n_games) <= 0:
        raise ValueError("n_games must be > 0")

    # device: prefer first model
    if device is None:
        device = _model_device(mods[0])

    # strict 1ch sanity
    for nm, m in zip(names, mods):
        _assert_cnet192_1ch(m, nm)

    rr = pd.DataFrame(np.nan, index=names, columns=names, dtype=np.float64)

    # fill diagonal to 0.5 by convention (prevents NaN “selfplay” confusion)
    np.fill_diagonal(rr.values, 0.5)

    details: List[Dict[str, Any]] = []

    n = len(names)
    total_pairs = n * (n - 1) // 2

    pbar = tqdm(total=total_pairs, desc="Round-robin", leave=True) if progress else None
    base_rng = random.Random(int(seed))

    for i in range(n):
        for j in range(i + 1, n):
            A_name, B_name = names[i], names[j]
            A, B = mods[i], mods[j]

            pair_seed = base_rng.randint(0, 10**9)

            res = head_to_head_models(
                A, B,
                n_games=int(n_games),
                A_label=A_name,
                B_label=B_name,
                device=device,
                opening_noise_k=opening_noise_k,
                seed=int(pair_seed),
                progress=False,  # avoid nested tqdm
                paired_openings=paired_openings,
                opening_bias=opening_bias,
            )

            details.append(res)

            a_score = float(res["A_score_rate"])
            rr.loc[A_name, B_name] = a_score
            rr.loc[B_name, A_name] = 1.0 - a_score

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"{A_name} vs {B_name}: {a_score:.3f}")

    if pbar is not None:
        pbar.close()

    if with_meta_column:
        meta = metascores_from_matrix(rr)  # Series aligned to rr.index
        rr = rr.copy()
        rr["meta_score"] = meta.reindex(rr.index).astype(np.float64)

        if sort_by_meta:
            rr = rr.sort_values("meta_score", ascending=False)
            order = list(rr.index)
            rr = rr.loc[order, order + ["meta_score"]]

    if return_details:
        return rr, details
    return rr

def metascores_from_matrix(rr: pd.DataFrame) -> pd.Series:
    """
    Compute metascore per row: average score vs all others.

    Robust behavior:
      - Works even if rr has extra columns (e.g. 'meta_score').
      - Uses only the square NxN intersection: names = rr.index ∩ rr.columns.
      - Ignores diagonal (self-play) even if it is 0.5.
    Returns:
      pd.Series indexed by model name (same order as rr.index, where applicable).
    """
    if not isinstance(rr, pd.DataFrame):
        raise TypeError("rr must be a pandas DataFrame")

    names = rr.index.intersection(rr.columns)
    if len(names) == 0:
        raise ValueError("No overlap between rr.index and rr.columns; cannot compute metascores.")

    M = rr.loc[names, names].to_numpy(dtype=np.float64, copy=True)

    # ignore diagonal no matter what value is there
    np.fill_diagonal(M, np.nan)

    # mean over opponents (skip NaNs)
    meta = np.nanmean(M, axis=1)

    # If N==1, nanmean gives NaN; define it as 0.5 (neutral)
    meta = np.nan_to_num(meta, nan=0.5)

    return pd.Series(meta, index=names, name="meta_score")


def meta_ranking_from_matrix(rr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metascores from a round-robin matrix and return a ranking DF:
      index: model
      col: meta_score
    Sorted descending.
    """
    ms = metascores_from_matrix(rr)  # Series
    ranking = ms.to_frame()          # DataFrame with column 'meta_score'
    ranking.index.name = "model"
    ranking = ranking.sort_values("meta_score", ascending=False)
    return ranking



def plot_meta_heatmap(
    meta: Union[Dict[str, float], pd.Series, pd.DataFrame],
    title: str = "Meta-score heatmap",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    annotate: bool = True,
    fmt: str = "{:.3f}",
    sort_desc: bool = True,
):
    """
    Plot metascore vector as a 1xN heatmap using matplotlib (no seaborn).
    Accepts:
      - dict{name->score}
      - pd.Series (index=name)
      - pd.DataFrame with column 'meta_score'
    Returns matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if isinstance(meta, dict):
        s = pd.Series(meta, dtype=np.float64)
    elif isinstance(meta, pd.Series):
        s = meta.astype(np.float64)
    elif isinstance(meta, pd.DataFrame):
        if "meta_score" not in meta.columns:
            raise ValueError("meta DataFrame must contain column 'meta_score'")
        s = meta["meta_score"].astype(np.float64)
    else:
        raise TypeError("meta must be dict, pd.Series, or pd.DataFrame")

    s = s.dropna()
    if sort_desc:
        s = s.sort_values(ascending=False)

    names = [str(x) for x in s.index.tolist()]
    vals = s.to_numpy(dtype=np.float64)[None, :]  # (1,N)

    n = len(names)
    if figsize is None:
        figsize = (max(6.0, 0.45 * n + 2.0), 2.4)

    fig = plt.figure(figsize=figsize, dpi=int(dpi))
    ax = fig.add_subplot(111)

    im = ax.imshow(vals, aspect="auto")  # let matplotlib pick the colormap
    ax.set_title(title)
    ax.set_yticks([0])
    ax.set_yticklabels(["meta"])
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(names, rotation=35, ha="right")

    if annotate:
        for j in range(n):
            ax.text(j, 0, fmt.format(float(vals[0, j])), ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    fig.tight_layout()
    return fig


def plot_meta_heatmap_from_matrix(
    rr: pd.DataFrame,
    title: str = "Meta-score heatmap",
    **kwargs,
):
    """Convenience: compute metascores from rr and plot."""
    ranking = meta_ranking_from_matrix(rr)
    return plot_meta_heatmap(ranking, title=title, **kwargs)

def matches_df_from_details(details: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert head_to_head_models() detail dicts into a tidy matches dataframe.
    Expects items like:
      {A_path, B_path, games, A_wins, draws, A_losses, A_score_rate, A_score_CI95, ...}
    """
    rows: List[Dict[str, Any]] = []
    for d in details:
        ci = d.get("A_score_CI95", (np.nan, np.nan))
        try:
            ci_lo, ci_hi = float(ci[0]), float(ci[1])
        except Exception:
            ci_lo, ci_hi = np.nan, np.nan

        rows.append({
            "A": str(d.get("A_path", "")),
            "B": str(d.get("B_path", "")),
            "games": int(d.get("games", 0)),
            "A_wins": int(d.get("A_wins", 0)),
            "draws": int(d.get("draws", 0)),
            "A_losses": int(d.get("A_losses", 0)),
            "A_score_rate": float(d.get("A_score_rate", np.nan)),
            "CI95_lo": ci_lo,
            "CI95_hi": ci_hi,
            "paired_openings": bool(d.get("paired_openings", False)),
            "opening_bias": str(d.get("opening_bias", "")),
            "opening_noise_k": d.get("opening_noise_k", None),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["A", "B"]).reset_index(drop=True)
    return df


def export_round_robin_bundle_to_excel(
    score_df: pd.DataFrame,
    details: Optional[List[Dict[str, Any]]] = None,
    out_dir: str = "Logs/PPO_matrix/",
    matrix_filename: str = "matrix_scores.xlsx",
    matches_filename: str = "pairwise_matches.xlsx",
    ranking_filename: str = "meta_ranking.xlsx",
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    score_export = score_df.copy()
    if "meta_score" in score_export.columns:
        score_export = score_export.drop(columns=["meta_score"])

    # ensure diagonal is 0.5 (export convention)
    names = score_export.index.intersection(score_export.columns)
    if len(names) > 0:
        tmp = score_export.loc[names, names].to_numpy(dtype=np.float64, copy=False)
        np.fill_diagonal(tmp, 0.5)

    ranking = meta_ranking_from_matrix(score_export)

    if details is None:
        matches_df = pd.DataFrame(columns=[
            "A","B","games","A_wins","draws","A_losses","A_score_rate","CI95_lo","CI95_hi",
            "paired_openings","opening_bias","opening_noise_k"
        ])
    else:
        matches_df = matches_df_from_details(details)

    score_export.to_excel(os.path.join(out_dir, matrix_filename))
    matches_df.to_excel(os.path.join(out_dir, matches_filename), index=False)
    ranking.to_excel(os.path.join(out_dir, ranking_filename))

    print("Saved matrix / matches / ranking to", out_dir)
    return matches_df, ranking

def plot_rr_heatmap(
    rr: pd.DataFrame,
    title: str = "Round-robin score heatmap",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    annotate: bool = True,
    fmt: str = "{:.3f}",
    rotation: int = 35,
    show_diagonal: bool = True,
):
    """
    Heatmap of the full NxN score matrix:
      rr.loc[A,B] = A score vs B in [0,1]
    Ignores extra columns like 'meta_score' by using index∩columns.
    """
    import matplotlib.pyplot as plt

    names = rr.index.intersection(rr.columns)
    if len(names) == 0:
        raise ValueError("rr has no index∩columns overlap to plot")

    M = rr.loc[names, names].to_numpy(dtype=np.float64, copy=True)

    # If someone kept NaN diagonal but you want 0.5, enforce it.
    if show_diagonal:
        np.fill_diagonal(M, 0.5)
    else:
        np.fill_diagonal(M, np.nan)

    n = len(names)
    if figsize is None:
        figsize = (max(6.5, 0.55 * n + 2.5), max(5.0, 0.45 * n + 2.0))

    fig = plt.figure(figsize=figsize, dpi=int(dpi))
    ax = fig.add_subplot(111)

    im = ax.imshow(M, aspect="auto", vmin=0.0, vmax=1.0)  # score is always [0,1]
    ax.set_title(title)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([str(x) for x in names], rotation=rotation, ha="right")
    ax.set_yticklabels([str(x) for x in names])

    # light grid
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.4, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for i in range(n):
            for j in range(n):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, fmt.format(float(v)), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    return fig


__all__ = [
    # core
    "GameResult",
    "make_opponent",
    "play_one_game",
    "evaluate_vs_opponent",
    "evaluate_suite",
    # excel
    "suite_df_to_row",
    "append_eval_row_to_excel",
    "evaluate_and_log_to_excel",
    # misc eval
    "exploitation_histogram",
    "empty_board_action_distribution",
    "plot_exploitation_histogram",
    "plot_empty_board_action_distribution",
    "print_eval_summary_from_suite_df",
    "plot_eval_bar_summary_from_suite_df",
    "play_single_game_ppo",
    # h2h
    "mean_ci",
    "head_to_head_models",
    "play_game_h2h",
    "plot_winrate_bar",
    "round_robin_matrix",
    "metascores_from_matrix",
    "meta_ranking_from_matrix",
    "plot_meta_heatmap",
    "plot_meta_heatmap_from_matrix",
    "matches_df_from_details",
    "export_round_robin_bundle_to_excel",
    "plot_rr_heatmap",
    
    
]
