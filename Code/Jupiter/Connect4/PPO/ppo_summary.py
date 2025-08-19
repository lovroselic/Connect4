# ppo_summary.py
# Summary logging utilities for PPO (dict-of-dicts, DQN-compatible)

from __future__ import annotations
import os, time
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd

# ----------------------------
# Small helpers
# ----------------------------

def _ma(x: List[float], k: int) -> Optional[float]:
    """Last value of a k-window moving average, or None if len(x) < k."""
    if k <= 1 or len(x) < k:
        return None
    w = np.ones(k, dtype=float) / k
    return np.convolve(np.asarray(x, dtype=float), w, mode="valid")[-1]

def init_summary_stats() -> Dict[int, Dict[str, Any]]:
    """Return dict keyed by episode (DQN-style)."""
    return {}

def _lk_to_int(lk: Optional[Union[int, str]]) -> int:
    """Map lookahead to numeric: None→0 (Random), 'self'→-1, int→itself."""
    if lk is None:
        return 0
    if isinstance(lk, str) and lk.lower() == "self":
        return -1
    return int(lk)

# ----------------------------
# Main logger
# ----------------------------

def log_summary_stats_ppo(
    summary_stats: Dict[int, Dict[str, Any]],
    *,
    episode: int,
    phase_name: str,
    lookahead_depth: Optional[Union[int, str]],
    reward_history: List[float],      # per-episode reward (your plotting choice)
    win_history: List[float],         # 1 for win, 0 for loss/draw (or 0.5 for draw if you prefer)
    win_count: int,
    loss_count: int,
    draw_count: int,
    ppo_metrics_history: Dict[str, List[float]],  # {"episodes","loss_pi","loss_v","entropy","approx_kl","clip_frac","explained_variance"}
    steps_per_update: int,
    lr: float,
    # DQN-table compatibility (optional, shown as columns in your Excel/plot):
    epsilon: Optional[float] = None,
    strategy_weights: Optional[List[float]] = None,
) -> None:
    """Append/replace the row for 'episode' in summary_stats (dict-of-dicts)."""

    # —— PPO update metrics (snapshot of the latest update) ——
    if ppo_metrics_history.get("episodes"):
        last = -1
        row_update = {
            "update_episode": int(ppo_metrics_history["episodes"][last]),
            "loss_pi": float(ppo_metrics_history["loss_pi"][last]),
            "loss_v": float(ppo_metrics_history["loss_v"][last]),
            "entropy": float(ppo_metrics_history["entropy"][last]),
            "approx_kl": float(ppo_metrics_history["approx_kl"][last]),
            "clip_frac": float(ppo_metrics_history["clip_frac"][last]),
            "explained_variance": float(ppo_metrics_history["explained_variance"][last]),
            "updates_total": int(len(ppo_metrics_history["episodes"])),
            "samples_seen": int(len(ppo_metrics_history["episodes"]) * steps_per_update),
        }
    else:
        row_update = {
            "update_episode": None,
            "loss_pi": None,
            "loss_v": None,
            "entropy": None,
            "approx_kl": None,
            "clip_frac": None,
            "explained_variance": None,
            "updates_total": 0,
            "samples_seen": 0,
        }

    # —— sliding windows expected by your plot_phase_summary ——
    # avg_reward_25
    if len(reward_history) > 0:
        k_r = min(25, len(reward_history))
        avg_reward_25 = float(np.mean(reward_history[-k_r:]))
    else:
        avg_reward_25 = float("nan")

    # win_rate_25
    if len(win_history) > 0:
        k_w = min(25, len(win_history))
        win_rate_25 = float(np.mean(win_history[-k_w:]))
    else:
        win_rate_25 = float("nan")

    # —— build row (NO 'win_rate_total' here; plot computes & joins it) ——
    summary_stats[int(episode)] = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episode": int(episode),
        "phase": str(phase_name),
        "lookahead": _lk_to_int(lookahead_depth),  # <-- supports None/"self"/int
        "epsilon": (None if epsilon is None else float(epsilon)),
        "strategy_weights": strategy_weights if strategy_weights is not None else None,
        "wins": int(win_count),
        "losses": int(loss_count),
        "draws": int(draw_count),

        "avg_reward_25": avg_reward_25,
        "win_rate_25": win_rate_25,

        # extras you may use elsewhere
        "reward_ma25": (_ma(reward_history, 25) or np.nan),
        "reward_ma100": (_ma(reward_history, 100) or np.nan),
        "reward_ma500": (_ma(reward_history, 500) or np.nan),
        "winrate_ma25": (_ma(win_history, 25) or np.nan),
        "winrate_ma250": (_ma(win_history, 250) or np.nan),

        "lr": float(lr),
        **row_update,
    }

# ----------------------------
# Conversions / Exports / Cleanup
# ----------------------------

def summary_stats_df(summary_stats: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """Convert dict-of-dicts to DataFrame sorted by episode (episodes remain as values; index is integer episode by your export)."""
    if not summary_stats:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(summary_stats, orient="index")
    # Keep index as episode for your Excel view, but also ensure a proper column exists:
    if "episode" not in df.columns:
        df["episode"] = df.index.astype(int)
    df = df.sort_values("episode")
    return df

def sanitize_for_plot(summary_stats: Dict[int, Dict[str, Any]]) -> None:
    """Remove fields that collide with plot_phase_summary joins (e.g., 'win_rate_total')."""
    for row in summary_stats.values():
        if isinstance(row, dict):
            row.pop("win_rate_total", None)

def save_summary_stats_excel(
    summary_stats: Dict[int, Dict[str, Any]],
    out_dir: str = "./logs",
    session_name: str = "PPO",
    filename: str | None = None,
) -> str:
    """Save to Excel with episodes as the index (DQN-style view)."""
    os.makedirs(out_dir, exist_ok=True)
    df = summary_stats_df(summary_stats)
    if filename is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{session_name}-training_summary_{stamp}.xlsx"
    path = os.path.join(out_dir, filename)
    # episodes as index in file
    df.to_excel(path, index=True)
    print(f"[summary] Excel saved → {path}")
    return path

def save_summary_stats_csv(
    summary_stats: Dict[int, Dict[str, Any]],
    out_dir: str = "./logs",
    session_name: str = "PPO",
    filename: str | None = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    df = summary_stats_df(summary_stats)
    if filename is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{session_name}-training_summary_{stamp}.csv"
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=True)
    print(f"[summary] CSV saved → {path}")
    return path
