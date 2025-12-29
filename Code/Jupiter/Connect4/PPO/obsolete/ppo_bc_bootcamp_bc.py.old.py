# PPO/ppo_bc_bootcamp_bc.py
# -*- coding: utf-8 -*-
"""
Behaviour-cloning 'BC boot camp' for PPO.

We take complete L1..L3 games (in your standard move-log format),
rebuild transitions (state_before_move, action, reward) and run
supervised policy updates on those transitions.

Differences vs the RL boot camp
-------------------------------
- No PPO buffer, no clipping, no value loss, no GAE.
- We only update the policy head by maximising log pi(a|s) on the
  dataset, optionally re-weighted by the per-move reward.
- Optional symmetry augmentation:
    * horizontal flip        (board[:, ::-1], action -> 6 - action)
    * colour swap / role swap (board -> -board, player -> -player)

Expected DATA format (per row = one move)
-----------------------------------------
Columns:
    'label'   : arbitrary label / tag (ignored here)
    'reward'  : immediate mover-centric reward from env.step(action)
    'game'    : game index (int, monotonically non-decreasing)
    'ply'     : 1-based ply index within game
    '0-0', ..., '5-6' : board snapshot AFTER the move (6x7 ints {-1,0,+1})

We reconstruct transitions as:
    board_before   = board BEFORE this row's move
    action         = column where disc was added between prev_board and curr_board
    player         = sign of the delta (+1/-1) in that cell
    reward         = row['reward'] (for the mover)

We *do not* re-run the env; we trust that 'reward' was computed by
Connect4Env when generating the data.

Any inconsistency (e.g. more than 1 changed cell between plys) raises
a ValueError with a precise message – we do NOT silently skip games.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Dict, Any, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from PPO.actor_critic import ActorCritic
from PPO.ppo_utilities import encode_two_channel_agent_centric

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOARD_ROWS = 6
BOARD_COLS = 7
BOARD_CELL_COLS = [f"{r}-{c}" for r in range(BOARD_ROWS) for c in range(BOARD_COLS)]


# =====================================================================
# Config
# =====================================================================

@dataclass
class BCBootcampConfig:
    """
    Hyperparameters for behaviour-cloning boot camp.

    passes           : how many passes over the entire dataset
    batch_size       : SGD minibatch size
    shuffle_each_pass: shuffle dataset indices before every pass
    lr_scale         : multiply current optimizer LR by this factor
                       for the duration of boot camp (restored after)
    use_reward_weights: if True, weight samples by (clipped) reward
    reward_alpha     : strength of reward weighting in [0, +inf)
                        w = 1 + reward_alpha * (clip(r)/reward_clip)
    reward_clip      : symmetric clip for rewards before weighting
    augment_hflip    : add horizontally-flipped copies of each sample
    augment_colorswap: add colour-swapped (role-swapped) copies
    verbose          : print progress, enable tqdm
    """
    passes: int = 1
    batch_size: int = 2048
    shuffle_each_pass: bool = True

    lr_scale: float = 0.3

    use_reward_weights: bool = True
    reward_alpha: float = 0.5
    reward_clip: float = 100.0

    augment_hflip: bool = True
    augment_colorswap: bool = False  # keep conservative by default

    verbose: bool = True


# =====================================================================
# Helpers: DataFrame -> transitions
# =====================================================================

def _extract_board_from_row(row: pd.Series) -> np.ndarray:
    """
    Take a single BC row and return a (6,7) int8 board.

    Board convention matches Connect4Env.board:
      - shape (6,7)
      - row 0 is TOP, row 5 is BOTTOM
    """
    b = row[BOARD_CELL_COLS].to_numpy(dtype=np.int8).reshape(BOARD_ROWS, BOARD_COLS)
    return b


def _iter_bc_transitions(df: pd.DataFrame) -> Iterable[Dict[str, Any]]:
    """
    Yield per-move transitions from a BC spreadsheet for *one run*.

    We assume:
      - 'game' and 'ply' columns exist
      - within this DataFrame 'game' is non-decreasing
      - each game has plys 1..N with no gaps

    For each (game, ply) we infer:
        * board_before (6x7 int8)
        * player (+1 / -1)
        * action (0..6)
        * reward (float)
    """
    required = {"game", "ply", "reward"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"BC data frame is missing required columns: {missing}")

    # 1) Canonicalise: one row per (game, ply)
    df_canon = (
        df.sort_values(["game", "ply"])
          .groupby(["game", "ply"], as_index=False)
          .first()
    )

    # 2) Group by game id from the sheet
    for game_id, gdf in df_canon.groupby("game"):
        gdf = gdf.sort_values("ply").reset_index(drop=True)
        if gdf.empty:
            continue

        plies = gdf["ply"].astype(int).to_numpy()
        # sanity check: plys should be 1..N with no gaps
        expected = np.arange(1, len(plies) + 1, dtype=int)
        if not np.array_equal(plies, expected):
            raise ValueError(
                f"Game {game_id}: expected ply sequence {expected.tolist()}, "
                f"found {plies.tolist()} – game is 'dirty', please regenerate."
            )

        # build the sequence of AFTER-move boards and rewards
        boards_after: List[np.ndarray] = []
        rewards: List[float] = []
        for _, row in gdf.iterrows():
            boards_after.append(_extract_board_from_row(row))
            rewards.append(float(row["reward"]))

        prev_board = np.zeros_like(boards_after[0], dtype=np.int8)


        for idx, (board_after, ply, rew) in enumerate(zip(boards_after, plies, rewards)):
            diff = board_after - prev_board
            changed = np.argwhere(diff != 0)

            if changed.shape[0] != 1:
                raise ValueError(
                    f"Game {game_id}, ply {ply}: expected exactly 1 changed cell, "
                    f"got {changed.shape[0]}. Game appears dirty; please fix "
                    f"the seed generator or regenerate this dataset."
                )

            r, c = changed[0]
            delta = int(diff[r, c])
            if delta not in (-1, 1):
                raise ValueError(
                    f"Game {game_id}, ply {ply}: changed cell has delta={delta}, "
                    "expected ±1. Game appears dirty."
                )

            player = delta          # stone that was just placed
            action = int(c)         # column index 0..6

            yield {
                "game":         int(game_id),
                "ply":          int(ply),
                "player":       player,
                "action":       action,
                "board_before": prev_board.copy(),
                "board_after":  board_after.copy(),
                "reward":       float(rew),
            }

            prev_board = board_after


def _iter_all_transitions(dfs: Sequence[pd.DataFrame]) -> Iterable[Dict[str, Any]]:
    """Chain transitions from multiple DataFrames (each assumed 'clean')."""
    for idx, df in enumerate(dfs):
        yield from _iter_bc_transitions(df)


def _legal_mask_from_board(board: np.ndarray) -> np.ndarray:
    """
    Compute legal moves as a bool[7] mask from a (6,7) board.

    Dataset/env convention: row 0 is TOP, row 5 BOTTOM.
    A column is illegal if the TOP cell (row 0) is already occupied.
    """
    assert board.shape == (BOARD_ROWS, BOARD_COLS)
    return (board[0, :] == 0).astype(bool)


# =====================================================================
# Dataset builder with augmentation + reward weights
# =====================================================================

def _build_bc_dataset(
    data_frames: Sequence[pd.DataFrame],
    cfg: BCBootcampConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn DATA frames into arrays:

        states  : [N, 2, 6, 7] float32
        actions : [N] int64
        masks   : [N, 7] bool
        weights : [N] float32
    """
    states: List[np.ndarray] = []
    actions: List[int] = []
    masks: List[np.ndarray] = []
    weights: List[float] = []

    def reward_to_weight(r: float) -> float:
        if not cfg.use_reward_weights:
            return 1.0
        rc = max(-cfg.reward_clip, min(cfg.reward_clip, r))
        return 1.0 + cfg.reward_alpha * (rc / cfg.reward_clip)

    for trans in _iter_all_transitions(data_frames):
        board_before = trans["board_before"]
        player = int(trans["player"])
        action = int(trans["action"])
        reward = float(trans["reward"])

        w = reward_to_weight(reward)

        # original sample
        variants: List[Tuple[np.ndarray, int, int, float]] = [
            (board_before, player, action, w)
        ]

        # horizontal flip
        if cfg.augment_hflip:
            b_flip = np.flip(board_before, axis=1)  # flip columns
            a_flip = BOARD_COLS - 1 - action
            variants.append((b_flip, player, a_flip, w))

        # colour swap
        if cfg.augment_colorswap:
            b_cs = -board_before
            p_cs = -player
            variants.append((b_cs, p_cs, action, w))

            if cfg.augment_hflip:
                b_both = -np.flip(board_before, axis=1)
                a_both = BOARD_COLS - 1 - action
                p_both = -player
                variants.append((b_both, p_both, a_both, w))

        for b_var, p_var, a_var, w_var in variants:
            state_2ch = encode_two_channel_agent_centric(b_var, p_var).astype(
                np.float32
            )
            legal_mask = _legal_mask_from_board(b_var)

            states.append(state_2ch)
            actions.append(int(a_var))
            masks.append(legal_mask.astype(bool))
            weights.append(float(w_var))

    if not states:
        raise ValueError("BC boot camp: built empty dataset – no transitions found.")

    states_arr = np.stack(states, axis=0)              # [N, 2, 6, 7]
    actions_arr = np.asarray(actions, dtype=np.int64)  # [N]
    masks_arr = np.stack(masks, axis=0).astype(bool)   # [N, 7]
    weights_arr = np.asarray(weights, dtype=np.float32)# [N]

    return states_arr, actions_arr, masks_arr, weights_arr


# =====================================================================
# Training loop
# =====================================================================

def run_bc_bootcamp_bc(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    data_frames: Sequence[pd.DataFrame],
    cfg: BCBootcampConfig,
) -> Dict[str, List[float]]:
    """
    Behaviour-cloning boot camp.

    Parameters
    ----------
    policy     : ActorCritic to be updated in-place.
    optimizer  : torch optimizer for policy parameters.
    data_frames: list/tuple of DATA frames with complete games.
    cfg        : BCBootcampConfig.

    Returns
    -------
    metrics : dict with per-pass lists:
        'pass'       : [passes...]
        'loss_bc'    : [mean NLL per pass]
        'entropy'    : [mean entropy per pass]
        'num_samples': [N_used_per_pass]
    """
    if not data_frames:
        raise ValueError("run_bc_bootcamp_bc: data_frames list is empty.")

    policy.to(DEVICE)
    policy.train()

    # Build dataset once
    if cfg.verbose:
        print("Building BC dataset from offline games ...")
    states, actions, masks, weights = _build_bc_dataset(data_frames, cfg)
    N = states.shape[0]
    if cfg.verbose:
        print(f"BC dataset: {N} samples "
              f"(states {states.shape}, actions {actions.shape})")

    # Torch tensors (keep on CPU, move to GPU per batch)
    states_t = torch.from_numpy(states)      # [N,2,6,7], float32
    actions_t = torch.from_numpy(actions)    # [N], int64
    masks_t = torch.from_numpy(masks)        # [N,7], bool
    weights_t = torch.from_numpy(weights)    # [N], float32

    # Optionally scale LR during boot camp
    orig_lrs = [group["lr"] for group in optimizer.param_groups]
    if cfg.lr_scale != 1.0:
        for group in optimizer.param_groups:
            group["lr"] = group["lr"] * cfg.lr_scale
        if cfg.verbose:
            print(f"Scaled optimizer LR by factor {cfg.lr_scale:.3f} "
                  f"for BC boot camp.")

    metrics: Dict[str, List[float]] = {
        "pass": [],
        "loss_bc": [],
        "entropy": [],
        "num_samples": [],
    }

    try:
        pass_iter = range(1, cfg.passes + 1)
        if cfg.verbose:
            pass_iter = tqdm(pass_iter, desc="BC boot camp passes")

        for p in pass_iter:
            if cfg.shuffle_each_pass:
                perm = torch.randperm(N)
            else:
                perm = torch.arange(N)

            total_loss = 0.0
            total_entropy = 0.0
            total_count = 0

            for start in range(0, N, cfg.batch_size):
                end = min(start + cfg.batch_size, N)
                idx = perm[start:end]

                s_b = states_t[idx].to(DEVICE)    # [B,2,6,7]
                a_b = actions_t[idx].to(DEVICE)   # [B]
                m_b = masks_t[idx].to(DEVICE)     # [B,7]
                w_b = weights_t[idx].to(DEVICE)   # [B]

                optimizer.zero_grad(set_to_none=True)

                # evaluate_actions gives log pi(a|s), entropy, value
                logp, entropy, _value = policy.evaluate_actions(
                    s_b, a_b, legal_action_masks=m_b
                )
                # flatten to [B]
                logp = logp.view(-1)
                entropy = entropy.view(-1)

                # Behaviour-cloning loss: maximise logp -> minimise -logp
                loss_bc = -(w_b * logp).mean()

                loss_bc.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                optimizer.step()

                batch_size = a_b.shape[0]
                total_loss += float(loss_bc.item()) * batch_size
                total_entropy += float(entropy.mean().item()) * batch_size
                total_count += batch_size

            mean_loss = total_loss / total_count
            mean_entropy = total_entropy / total_count

            metrics["pass"].append(p)
            metrics["loss_bc"].append(mean_loss)
            metrics["entropy"].append(mean_entropy)
            metrics["num_samples"].append(float(total_count))

            if cfg.verbose:
                print(
                    f"[BC pass {p}/{cfg.passes}] "
                    f"N={total_count}, loss_bc={mean_loss:.4f}, "
                    f"entropy={mean_entropy:.4f}"
                )

    finally:
        # restore original LR
        for group, lr in zip(optimizer.param_groups, orig_lrs):
            group["lr"] = lr

    return metrics


# =====================================================================
# Simple per-pass summary plot
# =====================================================================

def plot_bc_bootcamp_bc(
    metrics: Dict[str, List[float]],
    title: str = "BC boot camp (behaviour cloning), per-pass summaries",
):
    """
    Quick diagnostic plot: loss_bc and entropy vs BC pass.
    """
    passes = np.asarray(metrics["pass"], dtype=float)
    loss_bc = np.asarray(metrics["loss_bc"], dtype=float)
    entropy = np.asarray(metrics["entropy"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    fig.suptitle(title)

    ax = axes[0]
    ax.plot(passes, loss_bc, marker="o")
    ax.set_xlabel("BC pass")
    ax.set_ylabel("loss_bc (mean NLL)")
    ax.set_title("Policy loss (per-pass mean)")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1]
    ax.plot(passes, entropy, marker="o")
    ax.set_xlabel("BC pass")
    ax.set_ylabel("entropy")
    ax.set_title("Policy entropy (per-pass mean)")
    ax.grid(True, linestyle="--", alpha=0.3)

    return fig
