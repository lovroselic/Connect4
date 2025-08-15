# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 20:18:59 2025

@author: Uporabnik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DQN.training_phases_config import TRAINING_PHASES


# plot_phase_summary.py

def plot_phase_summary(summary_stats):
    # Convert summary to DataFrame
    df = pd.DataFrame.from_dict(summary_stats, orient='index')

    # Keep last row per phase (for recent stats)
    recent_summary = df.groupby("phase").tail(1).set_index("phase")

    # Compute total win rate per phase
    total_stats = df.groupby("phase")[["wins", "losses", "draws"]].last()
    total_stats["total_games"] = total_stats[["wins", "losses", "draws"]].sum(axis=1)
    total_stats["win_rate_total"] = total_stats["wins"] / total_stats["total_games"]

    # Merge with recent summary
    merged = recent_summary.join(total_stats["win_rate_total"])

    # Sort phases in training order
    phase_order = [p for p in TRAINING_PHASES if p in merged.index]
    merged = merged.loc[phase_order]

    # --- Plot Win Rate: recent vs total ---
    x = np.arange(len(merged))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, merged["win_rate_25"], width, label="Last 25")
    plt.bar(x + width/2, merged["win_rate_total"], width, label="Total")
    plt.xticks(x, merged.index, rotation=90)
    plt.ylim(0, 1)
    plt.ylabel("Win Rate")
    plt.title("Win Rate per Phase (Recent vs Total)")
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # --- Plot Avg Reward (last 25) ---
    plt.figure(figsize=(10, 4))
    plt.bar(merged.index, merged["avg_reward_25"], color='steelblue')
    plt.xticks(x, merged.index, rotation=90)
    plt.ylabel("Avg Reward (Last 25)")
    plt.title("Avg Reward per Phase")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()