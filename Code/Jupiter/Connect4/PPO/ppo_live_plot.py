# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 15:55:47 2025

@author: Lovro
"""

# ppo_live_plot.py
# Live plotting for PPO (episode-driven). Call this every N episodes.

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from DQN.dqn_utilities import _plot_bench_on_axis

def _moving_avg(x, k):
    if k <= 1 or len(x) < k: 
        return None
    w = np.ones(k, dtype=float) / k
    return np.convolve(np.asarray(x, dtype=float), w, mode='valid')

def plot_live_training_ppo(
    episode: int,
    reward_history,              # list[float]  per-episode total reward
    win_history,                 # list[int]    1 if win else 0 (draw/loss)
    phase_name: str,             # current phase name (for title)
    win_count: int,
    loss_count: int,
    draw_count: int,
    metrics_history: dict,       # dict of lists (see "How to use" below)
    benchmark_history: dict | None = None,
    title: str = "PPO Training",
    phases: dict | None = None,  # TRAINING_PHASES with 'length' filled (cumulative end ep)
    save: bool = False,
    save_path: str | None = None, # directory path ending with '/' or '\\'
    reward_ylim=(-200, 200),
):
    # --- Figure layout ---
    fig, ax = plt.subplots(5, 1, figsize=(10, 9), sharex=True)

    # 1) Rewards
    ax[0].plot(reward_history, label='Reward')
    for k, lw, ls in [(25, 2, '-'), (100, 1, '--'), (500, 1, '--')]:
        ma = _moving_avg(reward_history, k)
        if ma is not None:
            x0 = k - 1
            ax[0].plot(range(x0, x0 + len(ma)), ma, label=f'{k}-ep MA', linewidth=lw, linestyle=ls)
    ax[0].set_ylabel('Total Reward')
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_ylim(reward_ylim)

    # 2) Win rate (moving averages on 1/0 wins)
    if len(win_history) > 0:
        for k, color, ls in [(25, 'green', '-'), (250, '#999', '--')]:
            ma = _moving_avg(win_history, k)
            if ma is not None:
                x0 = k - 1
                ax[1].plot(range(x0, x0 + len(ma)), ma, label=f'Win Rate ({k})', color=color, linestyle=ls)
    ax[1].set_ylabel('Win Rate')
    ax[1].legend()
    ax[1].grid(True)

    # 3) PPO losses per update
    #    episodes_at_update aligns update metrics to when they were logged
    epi_u = metrics_history.get("episodes", [])
    if epi_u:
        ax[2].plot(epi_u, metrics_history.get("loss_pi", []), label='Loss π')
        ax[2].plot(epi_u, metrics_history.get("loss_v", []),  label='Loss V')
    ax[2].set_ylabel('Loss')
    ax[2].legend()
    ax[2].grid(True)

    # 4) Entropy / KL / ClipFrac per update
    if epi_u:
        ax[3].plot(epi_u, metrics_history.get("entropy", []),      label='Entropy')
        ax[3].plot(epi_u, metrics_history.get("approx_kl", []),    label='KL', linestyle='--')
        ax[3].plot(epi_u, metrics_history.get("clip_frac", []),    label='ClipFrac', linestyle=':')
    ax[3].set_xlabel('Episode')
    ax[3].set_ylabel('Policy Stats')
    ax[3].legend()
    ax[3].grid(True)
    
    # 5) Benchmark win rates vs fixed opponents (e.g., Random, L1)
    if benchmark_history and benchmark_history.get("episode"):
        _plot_bench_on_axis(
            ax[4],
            history=benchmark_history,
            smooth_k=3,
            training_phases=phases,
        )
    else:
        ax[4].set_visible(False)

    # Phase transition markers (vertical lines + labels)
    if phases:
        for name, meta in phases.items():
            L = meta.get("length", None)
            if L is None or np.isinf(L): 
                continue
            L = int(L)
            if L <= episode:
                for axis in ax:
                    axis.axvline(L, color='black', linestyle='dotted', linewidth=1)
                    # place label near top of each axis
                    y_top = axis.get_ylim()[1]
                    axis.text(L + 2, y_top * 0.95, name, rotation=90, va='top', ha='left', fontsize=8)

    # Title
    total = win_count + loss_count + draw_count
    wr = (win_count / total) if total > 0 else 0.0
    ev_list = metrics_history.get("explained_variance", [])
    ev_last = f"{ev_list[-1]:.3f}" if ev_list else "n/a"
    fig.suptitle(
        f"{title} - Episode {episode} — Phase: {phase_name} | "
        f"W/L/D: {win_count}/{loss_count}/{draw_count} | Win%={wr:.3f} | EV={ev_last}"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Live replace the previous figure in notebook
    clear_output(wait=True)
    display(fig)

    if save and save_path:
        fname = f"{save_path}{title}__complete_training_plot.png"
        fig.savefig(fname, dpi=120)
        print(f"[Saved] {fname}")

    plt.close(fig)
