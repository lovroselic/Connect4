# ppo_live_plot.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import Counter
from DQN.dqn_utilities import _plot_bench_on_axis, _plot_h2h_axis,draw_phase_vlines,  _draw_openings_on_axes

def _plot_opponent_usage_axis(
    ax,
    opponent_timeline: list[str],
    up_to_episode: int | None = None,
    normalize: bool = True,
):
    """
    Bar chart of opponent usage over episodes.

    opponent_timeline: list of labels per episode, e.g. ["R","L1","POP","L2",...]
    up_to_episode:     if not None, only use prefix opponent_timeline[:up_to_episode]
    normalize:         if True, show fractions, else raw counts.
    """
    if not opponent_timeline:
        ax.set_visible(False)
        return

    # slice to requested prefix
    if up_to_episode is None:
        used = opponent_timeline
    else:
        used = opponent_timeline[: max(0, min(up_to_episode, len(opponent_timeline)))]

    if not used:
        ax.set_visible(False)
        return

    counts = Counter(used)
    total  = sum(counts.values())
    if total == 0:
        ax.set_visible(False)
        return

    # --- ordering: R, POP, L1..Ln, SP, then any other labels alphabetically ---
    def _opp_sort_key(lab: str):
        if lab == "R":      return (0, 0)
        if lab == "POP":    return (1, 0)
        # L<depth>
        if lab.startswith("L"):
            try:
                depth = int(lab[1:])
                return (2, depth)
            except ValueError:  
                return (3, lab)
        if lab == "SP":         
            return (4, 0)
        return (3, lab)  # other exotic labels in the middle group

    labels_sorted = sorted(counts.keys(), key=_opp_sort_key)

    # --- colors: fixed for R/POP/SP, auto palette for Lk / others ---
    base_colors = {
        "R":   "#1f77b4",   # Random
        "POP": "#000000",   # Hall-of-fame ensemble
        "SP":  "#9467bd",   # Self-play
    }

    # simple palette for L* and other custom labels
    palette = [
        "#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#e377c2",
        "#7f7f7f", "#bcbd22", "#17becf", "#1a9850", "#fee08b",
        "#e08214", "#543005", "#542788",
    ]

    color_map: dict[str, str] = dict(base_colors)
    palette_idx = 0

    for lab in labels_sorted:
        if lab in color_map:
            continue
        # assign from palette in order, wrap if needed
        color_map[lab] = palette[palette_idx % len(palette)]
        palette_idx += 1

    # --- build arrays for plotting ---
    labels = []
    vals   = []
    colors = []

    for lab in labels_sorted:
        c = counts.get(lab, 0)
        if c <= 0:
            continue
        v = c / total if normalize else float(c)
        labels.append(lab)
        vals.append(v)
        colors.append(color_map.get(lab, "#888888"))

    if not labels:
        ax.set_visible(False)
        return

    xs = np.arange(len(labels))
    ax.bar(xs, vals, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)

    if normalize:
        ax.set_ylabel("Fraction of episodes")
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylabel("Episodes")

    ax.set_title("Opponent usage (by episode)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for x, v in zip(xs, vals):
        if v > 0:
            ax.text(
                x, v,
                f"{v:.2f}" if normalize else f"{int(v)}",
                ha="center", va="bottom", fontsize=8,
            )


def _plot_ppo_action_mix_axis(ax, policy):
    """
    Bars for PPO action mix.
    Uses whatever keys policy.act_stats_summary(normalize=True) returns,
    keeping the classic order for known keys and appending any extras.
    """
    if policy is None or not hasattr(policy, "act_stats_summary"):
        ax.set_visible(False)
        return

    stats = policy.act_stats_summary(normalize=True)  # -> dict[str, float]
    if not stats:
        ax.set_visible(False)
        return

    # Keep old four in a nice order, then append any new modes
    canonical = ["win_now", "center", "guard", "policy"]
    known = [k for k in canonical if k in stats]
    extra = sorted(k for k in stats.keys() if k not in canonical)
    labels = known + extra

    vals = [float(stats.get(k, 0.0)) for k in labels]

    # --- use explicit x positions, then set ticks *and* labels ---
    xs = np.arange(len(labels))
    ax.bar(xs, vals)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("PPO action mix (heuristics vs policy)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    for x, v in zip(xs, vals):
        ax.text(x, min(0.98, v + 0.03), f"{100*v:.1f}%", ha="center", va="bottom", fontsize=8)


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
    metrics_history: dict,       # dict of lists: episodes, loss_pi, loss_v, entropy, approx_kl, clip_frac, explained_variance
    benchmark_history: dict | None = None,
    title: str = "PPO Training",
    phases: dict | None = None,  # TRAINING_PHASES with 'length' set (cumulative end ep)
    save: bool = False,
    save_path: str | None = None,  # path ending with '/' or '\\'
    reward_ylim=(-1000, 1000),
    opponent_timeline: list[str] | None = None,  # e.g. ["R","L1","SP",...]
    overlay_last: int = 100,                      # draw colored ticks for last N episodes
    h2h_history: dict | None = None,              # {"episode":[], "score":[], "lo":[], "hi":[], "n":[]}
    ensemble_h2h_history: dict | None = None,     # {"episode":[], "score":[], "lo":[], "hi":[], "n":[]}
    openings=None,
    openings_ylim: tuple[float, float] | None = (0.65, 1.05),  # limit
    policy=None,   # PPO policy, for action-mix stats
):
    """
    Draws a multi-panel live report:
      [ 0] Reward (w/ MAs) + opponent tick overlay
      [ 1] Win rate MAs
      [ 2] PPO losses (π, V)
      [ 3] Entropy / KL / ClipFrac
      [ 4] Benchmarks (RAW)
      [ 5] Benchmarks (MA 3)
      [ 6] Benchmarks (MA 7)
      [ 7] Benchmarks (MA 15)
      [ 8] Global benchmark score (depth-weighted)
      [ 9] H2H score ±95% CI
      [10] Ensemble H2H score ±95% CI
      [11] First-move histogram (agent vs opponent)      [if openings]
      [12] a0@center over time (with MA)                 [if openings]
      [13] PPO action mix (win_now / center / guard / policy)
      [14] Opponent usage (R, L1..L13, SP)
    """

    # 0..9 = core, 11 = openings hist, 12 = a0@center, 13 = action mix, 14 = opponent usage
    nrows = 15
    base_heights = [
        6.0,  # 0 reward
        2.8,  # 1 win rate
        5.0,  # 2 losses
        6.0,  # 3 entropy / KL / clipfrac
        6.0,  # 4 benchmarks raw
        2.3,  # 5 benchmarks MA3
        2.3,  # 6 benchmarks MA7
        2.3,  # 7 benchmarks MA15
        2.4,  # 8 global score
        2.6,  # 9 H2H
        2.6,  # 10 ensemble H2H
        2.1,  # 11 openings hist
        2.4,  # 12 a0@center
        2.3,  # 13 action mix
        2.3,  # 14 opponent usage
    ]

    heights = base_heights[:nrows]

    fig = plt.figure(figsize=(12.2, sum(heights) + 1.0))
    gs  = fig.add_gridspec(
        nrows, 1,
        height_ratios=heights,
        hspace=0.45,
    )
    gs = GridSpec(nrows, 1, height_ratios=heights, hspace=0.45)

    # --- time-series axes ---
    ax_reward = fig.add_subplot(gs[0, 0])
    ax_win    = fig.add_subplot(gs[1, 0], sharex=ax_reward)
    ax_loss   = fig.add_subplot(gs[2, 0], sharex=ax_reward)
    ax_stats  = fig.add_subplot(gs[3, 0], sharex=ax_reward)
    ax_bench  = fig.add_subplot(gs[4, 0], sharex=ax_reward)
    ax_b3     = fig.add_subplot(gs[5, 0], sharex=ax_reward)
    ax_b7     = fig.add_subplot(gs[6, 0], sharex=ax_reward)
    ax_b15    = fig.add_subplot(gs[7, 0], sharex=ax_reward)
    ax_global = fig.add_subplot(gs[8, 0], sharex=ax_reward)  # NEW
    ax_h2h    = fig.add_subplot(gs[9, 0], sharex=ax_reward)
    ax_eh2h   = fig.add_subplot(gs[10, 0], sharex=ax_reward)

    ax_hist = fig.add_subplot(gs[11, 0])               # x = column index
    ax_rate = fig.add_subplot(gs[12, 0], sharex=ax_reward)
    ax_mix  = fig.add_subplot(gs[13, 0])               # action mix
    ax_opp  = fig.add_subplot(gs[14, 0])               # opponent usage

    # 1) Rewards + moving averages
    ax_reward.plot(reward_history, label="Reward", alpha=0.55)
    for k, lw, ls in [(25, 2.0, "-"), (100, 1.5, "--"), (500, 1.5, "--")]:
        ma = _moving_avg(reward_history, k)
        if ma is not None:
            x0 = k - 1
            ax_reward.plot(range(x0, x0 + len(ma)), ma, label=f"{k}-ep MA", linewidth=lw, linestyle=ls)
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(True, alpha=0.35)
    ax_reward.set_ylim(reward_ylim)
    if ax_reward.lines: ax_reward.legend(loc="lower left", fontsize=8)

    # 2) Win rate MAs (on 0/1 series)
    if len(win_history) > 0:
        for k, color, ls in [(25, "green", "-"), (100, "#999", "--"), (250, "#444", "--")]:
            ma = _moving_avg(win_history, k)
            if ma is not None:
                x0 = k - 1
                ax_win.plot(range(x0, x0 + len(ma)), ma,
                            label=f"Win Rate ({k})", color=color, linestyle=ls)
    ax_win.set_ylabel("Win Rate")
    if ax_win.lines:
        ax_win.legend(loc="lower left", fontsize=8)
    ax_win.grid(True, alpha=0.35)

    # 3) PPO losses per update
    epi_u = metrics_history.get("episodes", [])
    if epi_u:
        ax_loss.plot(epi_u, metrics_history.get("loss_pi", []), label="Loss π")
        ax_loss.plot(epi_u, metrics_history.get("loss_v", []),  label="Loss V")
    ax_loss.set_ylabel("Loss")
    if ax_loss.lines:
        ax_loss.legend(loc="upper right", fontsize=8)
    ax_loss.grid(True, alpha=0.35)

    # 4) Entropy / KL / ClipFrac per update
    if epi_u:
        ax_stats.plot(epi_u, metrics_history.get("entropy", []),   label="Entropy")
        ax_stats.plot(epi_u, metrics_history.get("approx_kl", []), label="KL", linestyle="--")
        ax_stats.plot(epi_u, metrics_history.get("clip_frac", []), label="ClipFrac", linestyle=":")
    ax_stats.set_ylabel("Policy Stats")
    if ax_stats.lines:
        ax_stats.legend(loc="upper right", fontsize=8)
    ax_stats.grid(True, alpha=0.35)

    # 5–8) Benchmarks (raw + MA 3/7/15) – phase lines handled later
    if benchmark_history and benchmark_history.get("episode"):
        _plot_bench_on_axis(ax_bench, benchmark_history, smooth_k=0,  training_phases=None)
        ax_bench.set_title("Benchmarks (raw)", fontsize=10)

        _plot_bench_on_axis(ax_b3, benchmark_history, smooth_k=3,  training_phases=None)
        ax_b3.set_title("Benchmarks (MA 3)", fontsize=10)

        _plot_bench_on_axis(ax_b7, benchmark_history, smooth_k=7,  training_phases=None)
        ax_b7.set_title("Benchmarks (MA 7)", fontsize=10)

        _plot_bench_on_axis(ax_b15, benchmark_history, smooth_k=15, training_phases=None)
        ax_b15.set_title("Benchmarks (MA 15)", fontsize=10)
    else:
        for axis in (ax_bench, ax_b3, ax_b7, ax_b15):
            axis.set_visible(False)

    # 8) Global depth-weighted score panel
    if benchmark_history and benchmark_history.get("episode") and benchmark_history.get("global_score"):
        episodes_g = np.asarray(benchmark_history["episode"], dtype=float)
        g_scores   = np.asarray(benchmark_history["global_score"], dtype=float)

        ax_global.plot(episodes_g, g_scores, label="Global score", alpha=0.8)
        ma_g = _moving_avg(g_scores, 3)
        if ma_g is not None and len(episodes_g) >= 3:
            x0 = 3 - 1
            ax_global.plot(
                episodes_g[x0: x0 + len(ma_g)],
                ma_g,
                label="Global score (MA3)",
                linestyle="--",
                linewidth=1.5,
            )
        ax_global.set_ylabel("Global score")
        ax_global.set_ylim(0.0, 1.0)
        ax_global.grid(True, alpha=0.35)
        if ax_global.lines:
            ax_global.legend(loc="lower left", fontsize=8)
        ax_global.set_title("Depth-weighted global benchmark score", fontsize=10)
    else:
        ax_global.set_visible(False)

    # 9) H2H panel (score ±95% CI)
    _plot_h2h_axis(ax_h2h, h2h_history or {}, training_phases=None)

    # 10) Ensemble H2H panel (score ±95% CI)
    _plot_h2h_axis(ax_eh2h, ensemble_h2h_history or {}, training_phases=None, title="Ensemble H2H")

    # Openings (hist + a0@center over time)
    _draw_openings_on_axes(
        ax_hist,
        ax_rate,
        openings,
        training_phases=phases,
        rate_ylim=openings_ylim,
    )

    # PPO action mix (heuristics vs policy)
    _plot_ppo_action_mix_axis(ax_mix, policy)

    # Opponent usage bar chart
    _plot_opponent_usage_axis(
        ax_opp,
        opponent_timeline or [],
        up_to_episode=episode,
        normalize=True,
    )

    # Phase boundaries on episode-based axes
    if phases:
        axes_for_vlines = [
            ax_reward, ax_win, ax_loss, ax_stats,
            ax_bench, ax_b3, ax_b7, ax_b15,
            ax_global, ax_h2h, ax_eh2h,
        ]
        if ax_rate is not None:
            axes_for_vlines.append(ax_rate)

        for axis in axes_for_vlines:
            if axis.get_visible():
                draw_phase_vlines(axis, phases, up_to=episode, label=True)

    # Title
    total = win_count + loss_count + draw_count
    wr = (win_count / total) if total > 0 else 0.0
    ev_list = metrics_history.get("explained_variance", [])
    ev_last = f"{ev_list[-1]:.3f}" if ev_list else "n/a"
    fig.suptitle(
        f"{title} - Ep {episode} — Phase: {phase_name} | "
        f"W/L/D: {win_count}/{loss_count}/{draw_count} | Win%={wr:.3f} | EV={ev_last}",
        y=0.995,
    )

    #clear_output(wait=True)

    if save and save_path:
        fname = f"{save_path}{title}__complete_training_plot.png"
        fig.savefig(fname, dpi=120)
        print(f"[Saved] {fname}")

    return fig
