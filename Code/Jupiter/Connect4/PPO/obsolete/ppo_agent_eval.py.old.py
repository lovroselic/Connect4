# PPO/ppo_agent_eval.py
"""
Unified evaluation helpers for "new line" models:
- supervised CNet192 checkpoints
- PPO checkpoints (same network, different training)

Board convention:
- numpy array shape (6,7), dtype int8
- values: 0 empty, +1 player1, -1 player2
- row 0 is the top row, row 5 is the bottom

Model convention:
- single-channel POV input: x = (board * player_to_move) with shape (B,1,6,7)
- policy output: logits (B,7)
- value output: (B,) or (B,1) (ignored by eval, but supported)

Opponents supported:
- Random / Leftmost / Center baselines
- Lookahead-N via Connect4Lookahead (Numba-bitboard), if provided

Also includes:
- Excel logging helpers (append row with win_rates per opponent)
- Exploitation histogram (greedy action frequency on random sampled states)
- Empty-board action distribution (greedy + softmax probs)
- Plot helpers (matplotlib figures)
- Bar plot summary from suite_df
- Pretty-print summary from suite_df

Designed to be imported from BOTH:
- standalone evaluation notebooks
- PPO training loops (online evaluation)

Typical notebook usage (NO helper defs in cells):

  from C4.eval_oppo_dict import EVALUATION_OPPONENTS, EVAL_CFG
  from C4.fast_connect4_lookahead import Connect4Lookahead
  from PPO.ppo_agent_eval import (
      evaluate_and_log_to_excel,
      exploitation_histogram, plot_exploitation_histogram,
      plot_empty_board_action_distribution,
      print_eval_summary_from_suite_df,
      plot_eval_bar_summary_from_suite_df,
  )

  la = Connect4Lookahead()
  suite_df, row_df = evaluate_and_log_to_excel(
      policy=policy,
      opponents_cfg=EVAL_CFG,   # or EVALUATION_OPPONENTS
      excel_path="EVAL_PPO_results.xlsx",
      run_tag=TRAINING_SESSION,
      device=DEVICE,
      lookahead=la,
      seed=SEED,
      episodes=meta.get("episode"),
  )

  print_eval_summary_from_suite_df(suite_df)

  fig_bar = plot_eval_bar_summary_from_suite_df(suite_df, model_name=model_name, save_path="eval_bar.png")
  plt.show(fig_bar)

  counts = exploitation_histogram(policy, device=DEVICE, n_states=256, rng_seed=0)
  fig_h = plot_exploitation_histogram(counts, model_name=model_name)
  plt.show(fig_h)

  fig_e = plot_empty_board_action_distribution(policy, device=DEVICE, model_name=model_name)
  plt.show(fig_e)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# Keep evaluation consistent with your env bitboard logic
from C4.connect4_env import (
    ROWS, COLS, STRIDE,
    TOP_MASK, FULL_MASK,
    _bb_has_won,  # module-level function (Numba if available)
    UINT,
)

OpponentFn = Callable[[np.ndarray, int, np.uint64], int]


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
    """Apply move in-place on board/heights and return updated (pos1,pos2,mask)."""
    c = int(col)
    if c < 0 or c >= COLS:
        raise ValueError(f"Illegal move: col={c}")

    h = int(heights[c])
    if h >= ROWS:
        raise ValueError(f"Illegal move: column {c} is full")

    bit = UINT(1) << UINT(c * STRIDE + h)
    mask = mask | bit
    if player == 1:
        pos1 = pos1 | bit
    else:
        pos2 = pos2 | bit

    heights[c] = h + 1
    board[ROWS - 1 - h, c] = np.int8(player)
    return pos1, pos2, mask


def _sample_random_state(
    rng: np.random.Generator,
    max_random_plies: int = 12,
) -> Tuple[np.ndarray, np.uint64, int]:
    """
    Sample a (board, mask, player_to_move) by playing random legal moves from empty.

    Returns:
      board: (6,7) int8 in {-1,0,1}
      mask : occupancy bitboard (uint64)
      player_to_move: +1 or -1
    """
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    heights = np.zeros(COLS, dtype=np.int8)
    pos1 = UINT(0)
    pos2 = UINT(0)
    mask = UINT(0)

    player = 1
    n = int(rng.integers(0, max_random_plies + 1))

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
    """
    Return (7,) logits with illegal columns masked to a very negative number.
    Supports models returning:
      - logits
      - (logits, value)
    """
    x = board_to_pov_tensor(board, player, device)
    out = model(x)

    logits = out[0] if isinstance(out, (tuple, list)) else out
    logits = logits[0].clone()  # (7,)

    if mask is None:
        for c in range(COLS):
            if board[0, c] != 0:
                logits[c] = -1e9
    else:
        for c in range(COLS):
            if (mask & TOP_MASK[c]) != 0:
                logits[c] = -1e9

    return logits


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

    if deterministic or temperature <= 0:
        return int(torch.argmax(logits).item())

    probs = torch.softmax(logits / float(temperature), dim=0).cpu().numpy()
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    s = float(probs.sum())
    if s <= 0:
        legal = _legal_mask_to_cols(mask) if mask is not None else [c for c in range(COLS) if board[0, c] == 0]
        return int(rng.choice(legal)) if legal else 0
    probs /= s
    return int(rng.choice(COLS, p=probs))


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
    if temperature <= 0:
        temperature = 1.0
    probs = torch.softmax(logits / float(temperature), dim=0).detach().cpu().numpy()
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
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
    """
    Create an opponent function from a label.

    IMPORTANT:
      If lookahead is provided AND has baseline_action(), we reuse it for:
        Random / Leftmost / Center
      Otherwise we fall back to mask-based baselines here.
    """
    if rng is None:
        rng = np.random.default_rng()

    s = (label or "").strip().lower()

    has_la_baselines = (lookahead is not None) and hasattr(lookahead, "baseline_action")

    # --- baselines (prefer lookahead.baseline_action for single source of truth) ---
    if s in ("random", "rnd"):
        if has_la_baselines:
            def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
                return int(lookahead.baseline_action(board, kind="random", rng=rng))
            return _op
        def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
            legal = _legal_mask_to_cols(mask)
            return int(rng.choice(legal)) if legal else 0
        return _op

    if s in ("leftmost", "left", "lm"):
        if has_la_baselines:
            def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
                return int(lookahead.baseline_action(board, kind="leftmost", rng=rng))
            return _op
        def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
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
            return int(lookahead.n_step_lookahead(board, player, depth=depth))
        return _op

    raise ValueError(f"Unknown opponent label: {label!r}")


# ----------------------------- evaluation -----------------------------

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
    rng = np.random.default_rng(seed)

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    heights = np.zeros(COLS, dtype=np.int8)

    pos1 = UINT(0)
    pos2 = UINT(0)
    mask = UINT(0)

    player = 1
    plies = 0

    was_training = bool(getattr(policy, "training", False))
    policy.eval()

    for _ in range(max_plies):
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
    if swap_sides and (n_games % 2 == 1):
        n_games += 1  # keep balanced starts

    rng = np.random.default_rng(seed)
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
    """
    Evaluate policy against a suite of opponents.

    - If `opponents` is a dict {label: n_games}, uses per-opponent counts.
    - If it's an iterable of labels, uses n_games_default.
    - progress=True shows ONE tqdm bar and updates description to the real opponent label.
    """
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
    """
    Convert evaluate_suite() DataFrame into a single-row DataFrame:
      index: TRAINING_SESSION
      columns: opponent labels -> win_rate
      plus TIME [h], EPISODES
    """
    row: Dict[str, Any] = {
        "TRAINING_SESSION": str(run_tag),
        "TIME [h]": round(float(elapsed_h), 6),
        "EPISODES": episodes,
    }

    for _, r in suite_df.iterrows():
        row[str(r["opponent"])] = float(r["win_rate"])

    return pd.DataFrame([row]).set_index("TRAINING_SESSION")


def append_eval_row_to_excel(df_row: pd.DataFrame, excel_path: str) -> None:
    """
    Append df_row (single row with TRAINING_SESSION index) to an Excel file.
    Creates the file if missing.
    """
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
    """
    One-call convenience:
      - runs evaluate_suite()
      - builds a single "row" df with win_rates per opponent
      - appends to excel

    Returns:
      (suite_df, row_df)
    """
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


# ----------------------------- Exploitation histogram + plots -----------------------------

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
    """
    Counts greedy (or sampled) action choices on random sampled states.
    Returns counts[7] as int64.
    """
    counts = np.zeros(COLS, dtype=np.int64)
    rng = np.random.default_rng(rng_seed)

    was_training = bool(getattr(policy, "training", False))
    policy.eval()

    for _ in range(int(n_states)):
        board, mask, player = _sample_random_state(rng, max_random_plies=max_random_plies)
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

    probs = policy_probs_on_board(policy, board, player, device, temperature=temperature, mask=mask)
    greedy = int(np.argmax(probs)) if probs.sum() > 0 else 0

    if was_training:
        policy.train(True)

    return {"greedy": greedy, "probs": probs.astype(np.float64)}


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

    info = empty_board_action_distribution(policy, device=device, temperature=temperature)
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
    """
    Grouped bar plot: Win/Loss/Draw percentages per opponent.
    Returns a matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    labels = [str(x) for x in suite_df["opponent"].tolist()]
    win_rates  = (suite_df["win_rate"].astype(float)  * 100.0).tolist()
    loss_rates = (suite_df["loss_rate"].astype(float) * 100.0).tolist()
    draw_rates = (suite_df["draw_rate"].astype(float) * 100.0).tolist()

    x = np.arange(len(labels), dtype=np.float64)
    bar_width = 0.25

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.bar(x, win_rates, width=bar_width, label="Win %")
    ax.bar(x + bar_width, loss_rates, width=bar_width, label="Loss %")
    ax.bar(x + 2 * bar_width, draw_rates, width=bar_width, label="Draw %")

    ax.set_xlabel("Opponent Type")
    ax.set_ylabel("Percentage")
    ax.set_title(f"{model_name} Performance vs Various Opponents")
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
    """
    Returns (result, final_board) where:
      result = +1 win, 0 draw, -1 loss (from policy POV as player1)
    """
    if device is None:
        device = next(policy.parameters()).device

    rng = np.random.default_rng(seed)
    opp = make_opponent(opponent_label, lookahead=lookahead, rng=rng)

    gr = play_one_game(
        policy, opp, device=device,
        seed=seed,
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

    return r, gr.final_board


__all__ = [
    "GameResult",
    "make_opponent",
    "play_one_game",
    "evaluate_vs_opponent",
    "evaluate_suite",
    "suite_df_to_row",
    "append_eval_row_to_excel",
    "evaluate_and_log_to_excel",
    "exploitation_histogram",
    "empty_board_action_distribution",
    "plot_exploitation_histogram",
    "plot_empty_board_action_distribution",
    "print_eval_summary_from_suite_df",
    "plot_eval_bar_summary_from_suite_df",
    "play_single_game_ppo",
]
