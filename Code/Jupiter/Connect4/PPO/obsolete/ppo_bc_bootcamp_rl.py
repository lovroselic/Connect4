# PPO/ppo_bc_bootcamp_rl.py
# -*- coding: utf-8 -*-
"""
Offline 'BC boot camp RL' for PPO using replay games with rewards.

We take complete L1..L3 games (standard move-log format),
rebuild transitions (state_before_move, action, reward, done) and run
**PPO updates** on those transitions.

Key points
----------
- Uses the same per-step reward signal as training (the 'reward' column).
- States are encoded with the same 2ch agent-centric encoding:
      encode_two_channel_agent_centric(board_before, player)
  so every move is seen from the mover's perspective ("agent").
- We push everything through PPOBuffer + ppo_update, so:
    * rewards are stacked over the whole game via GAE
    * good sequences get positive advantage, bad ones negative
    * noisy games do not get cloned equally.
- Optional H-flip + color-swap augmentations are supported via PPOBuffer.

Expected DATA format (per row = one move)
-----------------------------------------
Columns:
    'label'   : arbitrary label / tag
    'reward'  : immediate mover-centric reward from env.step(action)
    'game'    : game index (int) **within this file/run**
    'ply'     : 1-based ply index within game
    '0-0', ..., '5-6' : board snapshot AFTER the move (6x7 ints {-1,0,+1})

Each game is reconstructed as:
    state_before_move = board BEFORE this row's move
    action            = column where disc was added between prev_board and curr_board
    reward            = row['reward']   (for the mover, as in Connect4Env)
    done              = True only for last ply in that game

IMPORTANT
---------
- Each DataFrame you pass should come from a single generator run, i.e.
  'game' must not restart from 0 inside the same DataFrame.
  If you want to use multiple runs, read each into its own DataFrame and
  pass them as a list: data_frames=[df_clean, df_noisy, ...].
- If a game contains an impossible board transition (more than 1 changed
  cell between successive plies), we raise ValueError with game / ply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Dict, Any, Iterable, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from PPO.actor_critic import ActorCritic
from PPO.ppo_update import ppo_update, PPOUpdateCfg
from PPO.ppo_buffer import PPOBuffer, PPOHyperParams
from PPO.ppo_utilities import encode_two_channel_agent_centric

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOARD_ROWS = 6
BOARD_COLS = 7
BOARD_CELL_COLS = [f"{r}-{c}" for r in range(BOARD_ROWS) for c in range(BOARD_COLS)]


# =====================================================================
# Boot camp RL config
# =====================================================================

@dataclass
class BCBootcampRLConfig:
    """
    Hyperparameters for reward-based offline boot camp (PPO style).

    passes          : how many full passes over the entire dataset
    batch_size_ppo  : minibatch size inside ppo_update
    steps_per_update: how many *environment steps* we accumulate before
                      each PPO update (buffer capacity is 4x this)
    reward_scale    : must match your PPO training REWARD_SCALE
    gamma           : must match PPO training gamma
    gae_lambda      : must match PPO training lambda
    augment_hflip   : horizontally flip columns as extra samples
    augment_colorswap: if True, also generate role-swapped variants
    verbose         : print progress & simple stats
    """
    passes: int = 1
    batch_size_ppo: int = 256
    steps_per_update: int = 2048

    reward_scale: float = 0.005
    gamma: float = 0.99
    gae_lambda: float = 0.95

    augment_hflip: bool = True
    augment_colorswap: bool = True

    verbose: bool = True


# =====================================================================
# Helpers: DataFrame -> transitions
# =====================================================================

def _extract_board_from_row(row: pd.Series) -> np.ndarray:
    """
    Take a single BC row and return a (6,7) int8 board
    in the same orientation as in Connect4Env and the log.
    """
    b = row[BOARD_CELL_COLS].to_numpy(dtype=np.int8).reshape(BOARD_ROWS, BOARD_COLS)
    return b


def _iter_bc_transitions(df: pd.DataFrame) -> Iterable[Dict[str, Any]]:
    """
    Yield per-move transitions from a BC spreadsheet.

    - The raw data can contain many rows with the same (game, ply) and identical
      boards (different labels). We collapse those to one canonical row.

    - Each DataFrame is assumed to come from a single generator run where 'game'
      is unique per game and does not restart.

    For each game we infer:
        * the column of the move (action) from the single changed cell
        * the player (+1 / -1) from the sign of the delta in that cell
        * the immediate reward from the 'reward' column (mover-centric)
        * done=True only for the last ply in that game.

    The very first ply of each game is treated as a move from the empty board.
    """
    if not {"game", "ply"}.issubset(df.columns):
        raise ValueError("BC data frame must contain 'game' and 'ply' columns")

    # Canonicalise: one row per (game, ply)
    df_canon = (
        df.sort_values(["game", "ply"])
          .groupby(["game", "ply"], as_index=False)
          .first()
    )

    for game_id, gdf in df_canon.groupby("game"):
        gdf = gdf.sort_values("ply").reset_index(drop=True)
        if gdf.empty:
            continue

        boards: list[np.ndarray] = []
        plies: list[int] = []
        rewards: list[float] = []

        for _, row in gdf.iterrows():
            boards.append(_extract_board_from_row(row))
            plies.append(int(row["ply"]))
            rewards.append(float(row["reward"]))

        prev_board = np.zeros_like(boards[0], dtype=np.int8)
        n = len(boards)

        for idx, (board, ply, rew) in enumerate(zip(boards, plies, rewards)):
            diff = board - prev_board
            changed = np.argwhere(diff != 0)

            if changed.shape[0] != 1:
                raise ValueError(
                    f"Game {game_id}, ply {ply}: expected exactly 1 changed cell, "
                    f"got {changed.shape[0]}"
                )

            r, c = changed[0]
            delta = int(diff[r, c])
            if delta not in (-1, 1):
                raise ValueError(
                    f"Game {game_id}, ply {ply}: delta at changed cell is {delta}, "
                    "expected ±1"
                )

            player = delta          # sign of the stone that just appeared
            action = int(c)         # column index 0..6
            done = (idx == n - 1)   # last ply in this game

            yield {
                "game":         int(game_id),
                "ply":          ply,
                "player":       player,
                "action":       action,
                "board_before": prev_board.copy(),
                "board_after":  board.copy(),
                "reward":       rew,
                "done":         done,
            }

            prev_board = board


def _iter_all_transitions(dfs: Sequence[pd.DataFrame]) -> Iterable[Dict[str, Any]]:
    """Chain transitions from multiple DataFrames."""
    for df in dfs:
        yield from _iter_bc_transitions(df)


# =====================================================================
# Core: fill PPOBuffer from offline games & run PPO updates
# =====================================================================

def _legal_mask_from_board(board: np.ndarray) -> np.ndarray:
    """
    Compute legal moves as a bool[7] mask from a (6,7) board.

    Dataset/env convention: row 5 is bottom, row 0 is top.
    Column is illegal if the *top* cell (row 0) is occupied.
    """
    assert board.shape == (6, 7)
    return (board[0, :] == 0).astype(bool)


def run_bc_bootcamp_rl(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    data_frames: Sequence[pd.DataFrame],
    ppo_cfg: PPOUpdateCfg,
    bc_cfg: BCBootcampRLConfig = BCBootcampRLConfig(),
) -> Dict[str, List[float]]:
    """
    Reward-based offline boot camp *using PPO*.

    We replay full games from the logs, treat every move from the mover's
    perspective (encode_two_channel_agent_centric(board_before, player)),
    and run PPO updates with GAE over those trajectories.

    This uses rewards exactly like online training:
      - rewards are stacked over time via gamma / lambda
      - both sides are trained symmetrically (whoever moves is "the agent")
      - noisy games with low/negative return get low/negative advantage.

    Returns
    -------
    metrics_history : dict
        Per-update metrics with an extra key "pass_idx" telling
        which BC pass each update belongs to.
    """
    if not data_frames:
        raise ValueError("run_bc_bootcamp_rl: data_frames list is empty.")

    policy.to(DEVICE)
    policy.train()

    # match PPO training hyperparams, but override batch_size for boot camp
    ppo_cfg = PPOUpdateCfg(
        epochs        = ppo_cfg.epochs,
        batch_size    = bc_cfg.batch_size_ppo,
        clip_range    = ppo_cfg.clip_range,
        vf_clip_range = ppo_cfg.vf_clip_range,
        ent_coef      = ppo_cfg.ent_coef,
        vf_coef       = ppo_cfg.vf_coef,
        max_grad_norm = ppo_cfg.max_grad_norm,
        target_kl     = ppo_cfg.target_kl,
        distill_coef  = ppo_cfg.distill_coef,
        mentor_depth  = getattr(ppo_cfg, "mentor_depth", 1),
        mentor_prob   = getattr(ppo_cfg, "mentor_prob", 0.0),
        mentor_coef   = getattr(ppo_cfg, "mentor_coef", 0.0),
    )

    hparams = PPOHyperParams(
        gamma        = bc_cfg.gamma,
        gae_lambda   = bc_cfg.gae_lambda,
        normalize_adv= True,
    )

    buffer = PPOBuffer(
        capacity     = bc_cfg.steps_per_update * 4,  # headroom for augs
        action_dim   = 7,
        hparams      = hparams,
        reward_scale = bc_cfg.reward_scale,
    )

    metrics_history: Dict[str, List[float]] = {
        "pass_idx": [],
        "loss_pi": [],
        "loss_v": [],
        "entropy": [],
        "approx_kl": [],
        "clip_frac": [],
        "explained_variance": [],
        "distill_kl": [],
        "mentor_ce": [],
        "updates_samples": [],
    }

    def log_metrics(m: Dict[str, Any], pass_idx: int):
        metrics_history["pass_idx"].append(pass_idx)
        metrics_history["loss_pi"].append(float(m.get("loss_pi", 0.0)))
        metrics_history["loss_v"].append(float(m.get("loss_v", 0.0)))
        metrics_history["entropy"].append(float(m.get("entropy", 0.0)))
        metrics_history["approx_kl"].append(float(m.get("approx_kl", 0.0)))
        metrics_history["clip_frac"].append(float(m.get("clip_frac", 0.0)))
        metrics_history["explained_variance"].append(
            float(m.get("explained_variance", 0.0))
        )
        metrics_history["distill_kl"].append(float(m.get("distill_kl", 0.0)))
        metrics_history["mentor_ce"].append(float(m.get("mentor_ce", 0.0)))
        metrics_history["updates_samples"].append(float(m.get("updates_samples", 0.0)))

    total_updates = 0

    for pass_idx in tqdm(
        range(1, bc_cfg.passes + 1),
        desc="BC boot camp RL passes",
        disable=not bc_cfg.verbose,
    ):
        if bc_cfg.verbose:
            print(f"\n[BC boot camp RL] Pass {pass_idx}/{bc_cfg.passes}")

        buffer.clear()

        for trans in _iter_all_transitions(data_frames):
            board_before = trans["board_before"]   # (6,7) int8, BEFORE move
            player       = int(trans["player"])    # +1 / -1 (mover)
            action       = int(trans["action"])    # 0..6
            reward       = float(trans["reward"])  # mover-centric reward
            done         = bool(trans["done"])

            # encode from mover's perspective into 2-channel agent-centric state
            state_2ch  = encode_two_channel_agent_centric(board_before, player)  # (2,6,7)
            legal_mask = _legal_mask_from_board(board_before)                    # (7,)

            state_t = torch.from_numpy(state_2ch).unsqueeze(0).to(DEVICE).float()
            act_t   = torch.tensor([action], dtype=torch.long, device=DEVICE)
            mask_t  = torch.from_numpy(legal_mask).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logp, entropy, value = policy.evaluate_actions(
                    state_t, act_t, legal_action_masks=mask_t
                )

            logp  = logp.squeeze(0)
            value = value.squeeze(0)

            # Write to PPOBuffer (with augs if enabled)
            if bc_cfg.augment_hflip or bc_cfg.augment_colorswap:
                buffer.add_augmented(
                    state_2ch,
                    action,
                    logp,
                    value,
                    reward,
                    done,
                    legal_mask,
                    hflip=bc_cfg.augment_hflip,
                    colorswap=bc_cfg.augment_colorswap,
                    include_original=True,
                    recompute_fn=None,  # reuse logp/value for augs (OK for boot camp)
                )
            else:
                buffer.add(
                    state_2ch,
                    action,
                    logp,
                    value,
                    reward,
                    done,
                    legal_mask,
                )

            # PPO update when buffer is full
            if len(buffer) >= bc_cfg.steps_per_update:
                buffer.compute_gae(last_value=0.0, last_done=True)
                metrics = ppo_update(
                    policy=policy,
                    optimizer=optimizer,
                    buffer=buffer,
                    cfg=ppo_cfg,
                    teacher=None,
                    mentor=None,
                )
                log_metrics(metrics, pass_idx)
                total_updates += 1
                buffer.clear()

        # flush leftover transitions for this pass (if any)
        if len(buffer) > 0:
            buffer.compute_gae(last_value=0.0, last_done=True)
            metrics = ppo_update(
                policy=policy,
                optimizer=optimizer,
                buffer=buffer,
                cfg=ppo_cfg,
                teacher=None,
                mentor=None,
            )
            log_metrics(metrics, pass_idx)
            total_updates += 1
            buffer.clear()

        if bc_cfg.verbose:
            print(f"[BC boot camp RL] Pass {pass_idx} finished; "
                  f"PPO updates so far: {total_updates}")

    return metrics_history

# =====================================================================
# Simple per-pass summary plot
# =====================================================================

def plot_bc_bootcamp_rl(
    metrics: Dict[str, List[float]],
    title: str = "BC boot camp (RL), per-pass summaries",
):
    """
    Summarize PPO-style BC boot camp metrics per pass.

    Expects metrics from run_bc_bootcamp_rl() with keys:
      - "pass_idx"        : list[int], one per PPO update
      - "loss_pi"         : list[float]
      - "loss_v"          : list[float]
      - "entropy"         : list[float]
      - "clip_frac"       : list[float]
      (other keys are ignored here)

    For each pass, we average over all PPO updates in that pass and plot:
      - loss_v (per-pass mean)
      - loss_pi (per-pass mean)
      - entropy (per-pass mean)
      - clip_frac (per-pass mean)
    """
    if "pass_idx" not in metrics or len(metrics["pass_idx"]) == 0:
        raise ValueError("metrics['pass_idx'] is empty; nothing to plot.")

    pass_idx = np.asarray(metrics["pass_idx"], dtype=int)
    unique_passes = np.unique(pass_idx)

    def mean_per_pass(key: str) -> np.ndarray:
        vals = np.asarray(metrics[key], dtype=float)
        return np.array(
            [vals[pass_idx == p].mean() for p in unique_passes],
            dtype=float,
        )

    loss_v_mean  = mean_per_pass("loss_v")
    loss_pi_mean = mean_per_pass("loss_pi")
    ent_mean     = mean_per_pass("entropy")
    clip_mean    = mean_per_pass("clip_frac")

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    fig.suptitle(title)

    # loss_v
    ax = axes[0, 0]
    ax.plot(unique_passes, loss_v_mean, marker="o")
    ax.set_xlabel("BC pass")
    ax.set_ylabel("loss_v")
    ax.set_title("Value loss (per-pass mean)")
    ax.grid(True, linestyle="--", alpha=0.3)

    # loss_pi
    ax = axes[0, 1]
    ax.plot(unique_passes, loss_pi_mean, marker="o")
    ax.set_xlabel("BC pass")
    ax.set_ylabel("loss_pi")
    ax.set_title("Policy loss (per-pass mean)")
    ax.grid(True, linestyle="--", alpha=0.3)

    # entropy
    ax = axes[1, 0]
    ax.plot(unique_passes, ent_mean, marker="o")
    ax.set_xlabel("BC pass")
    ax.set_ylabel("entropy")
    ax.set_title("Policy entropy (per-pass mean)")
    ax.grid(True, linestyle="--", alpha=0.3)

    # clip_frac
    ax = axes[1, 1]
    ax.plot(unique_passes, clip_mean, marker="o")
    ax.set_xlabel("BC pass")
    ax.set_ylabel("clip_frac")
    ax.set_title("Clip fraction (per-pass mean)")
    ax.grid(True, linestyle="--", alpha=0.3)

    return fig