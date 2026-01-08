# ppo_live_plot.py

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
from collections import Counter
from DQN.dqn_utilities import _plot_bench_on_axis, _plot_h2h_axis, draw_phase_vlines, _draw_openings_on_axes


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
    total = sum(counts.values())
    if total == 0:
        ax.set_visible(False)
        return

    # --- ordering: R, POP, L1..Ln, SP, then any other labels alphabetically ---
    def _opp_sort_key(lab: str):
        if lab == "R":
            return (0, 0)
        if lab == "POP":
            return (1, 0)
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
        "R": "#1f77b4",   # Random
        "POP": "#000000",  # Hall-of-fame ensemble
        "SP": "#9467bd",   # Self-play
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
    vals = []
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
        ax.text(x, min(0.98, v + 0.03), f"{100 * v:.1f}%", ha="center", va="bottom", fontsize=8)


def _moving_avg(x, k):
    if k <= 1 or len(x) < k:
        return None
    w = np.ones(k, dtype=float) / k
    return np.convolve(np.asarray(x, dtype=float), w, mode="valid")


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
    reward_ylim=(-10000, 10000),
    opponent_timeline: list[str] | None = None,  # e.g. ["R","L1","SP",...]
    overlay_last: int = 100,                      # draw colored ticks for last N episodes
    h2h_history: dict | None = None,              # {"episode":[], "score":[], "lo":[], "hi":[], "n":[]}
    ensemble_h2h_history: dict | None = None,     # {"episode":[], "score":[], "lo":[], "hi":[], "n":[]}
    openings=None,
    openings_ylim: tuple[float, float] | None = (0.85, 1.05),
    policy=None,   # PPO policy, for action-mix stats
):
    """
    Multi-panel live report.
    Openings panels appear only if `openings` is provided.
    """

    use_openings = openings is not None

    # --- layout ---
    if use_openings:
        nrows = 16
        heights = [
            6.0,   # 0 reward
            10.0,  # 1 win rate
            5.0,   # 2 losses
            10.0,   # 3 entropy / KL / clipfrac (split y-axes)
            10.0,  # 4 benchmarks raw
            2.3,   # 5 benchmarks MA3
            2.3,   # 6 benchmarks MA7
            10.0,   # 7 benchmarks MA15
            2.4,   # 8 global score
            2.6,   # 9 H2H
            2.6,   # 10 ensemble H2H
            3.0,   # 11 checkpoint score
            2.1,   # 12 openings hist
            8.0,   # 13 a0@center
            2.3,   # 14 action mix
            2.3,   # 15 opponent usage
        ]
    else:
        nrows = 14
        heights = [
            6.0,   # 0 reward
            10.0,  # 1 win rate
            5.0,   # 2 losses
            10.0,   # 3 entropy / KL / clipfrac (split y-axes)
            10.0,  # 4 benchmarks raw
            2.3,   # 5 benchmarks MA3
            2.3,   # 6 benchmarks MA7
            10.0,   # 7 benchmarks MA15
            2.4,   # 8 global score
            2.6,   # 9 H2H
            2.6,   # 10 ensemble H2H
            3.0,   # 11 checkpoint score
            2.3,   # 12 action mix
            2.3,   # 13 opponent usage
        ]

    fig = plt.figure(figsize=(12.2, sum(heights) + 1.0))
    gs = fig.add_gridspec(nrows, 1, height_ratios=heights, hspace=0.45)

    # --- axes ---
    ax_reward = fig.add_subplot(gs[0, 0])
    ax_win = fig.add_subplot(gs[1, 0], sharex=ax_reward)
    ax_loss = fig.add_subplot(gs[2, 0], sharex=ax_reward)
    ax_stats = fig.add_subplot(gs[3, 0], sharex=ax_reward)
    ax_bench = fig.add_subplot(gs[4, 0], sharex=ax_reward)
    ax_b3 = fig.add_subplot(gs[5, 0], sharex=ax_reward)
    ax_b7 = fig.add_subplot(gs[6, 0], sharex=ax_reward)
    ax_b15 = fig.add_subplot(gs[7, 0], sharex=ax_reward)
    ax_global = fig.add_subplot(gs[8, 0], sharex=ax_reward)
    ax_h2h = fig.add_subplot(gs[9, 0], sharex=ax_reward)
    ax_eh2h = fig.add_subplot(gs[10, 0], sharex=ax_reward)
    ax_chkp = fig.add_subplot(gs[11, 0], sharex=ax_reward)

    if use_openings:
        ax_hist = fig.add_subplot(gs[12, 0])
        ax_rate = fig.add_subplot(gs[13, 0], sharex=ax_reward)
        ax_mix = fig.add_subplot(gs[14, 0])
        ax_opp = fig.add_subplot(gs[15, 0])
    else:
        ax_hist = None
        ax_rate = None
        ax_mix = fig.add_subplot(gs[12, 0])
        ax_opp = fig.add_subplot(gs[13, 0])

    # ---------------- helpers ----------------
    def _opp_color(lab: str) -> str:
        # keep consistent with usage bar chart intent
        if lab == "R":
            return "#1f77b4"
        if lab == "POP":
            return "#000000"
        if lab == "SP":
            return "#9467bd"
        if lab.startswith("L") and lab[1:].isdigit():
            # stable-ish palette by depth
            depth = int(lab[1:])
            palette = ["#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            return palette[depth % len(palette)]
        return "#888888"

    # ---------------- 1) Rewards ----------------
    ax_reward.plot(reward_history, label="Reward", alpha=0.55)

    for k, lw, ls in [(25, 2.0, "-"), (100, 1.5, "--"), (500, 1.5, "--")]:
        ma = _moving_avg(reward_history, k)
        if ma is not None:
            x0 = k - 1
            ax_reward.plot(range(x0, x0 + len(ma)), ma, label=f"{k}-ep MA", linewidth=lw, linestyle=ls)

    ax_reward.set_ylabel("Reward")
    ax_reward.grid(True, alpha=0.35)
    ax_reward.set_ylim(reward_ylim)
    ax_reward.legend(loc="lower left", fontsize=8)

    # opponent tick overlay (last N episodes)
    if opponent_timeline:
        n = len(opponent_timeline)
        a = max(0, n - int(overlay_last))
        xs = np.arange(a, n, dtype=int)
        if xs.size > 0:
            y0, y1 = ax_reward.get_ylim()
            y_tick = y0 + 0.03 * (y1 - y0)
            cols = [_opp_color(opponent_timeline[i]) for i in xs]
            ax_reward.scatter(
                xs,
                np.full(xs.shape, y_tick, dtype=float),
                marker="|",
                s=250,
                c=cols,
                alpha=0.9,
                zorder=4,
            )

    # ---------------- 2) Win rate ----------------
    if len(win_history) > 0:
        for k, color, ls in [(25, "green", "-"), (100, "#999", "--"), (250, "#444", "--")]:
            ma = _moving_avg(win_history, k)
            if ma is not None:
                x0 = k - 1
                ax_win.plot(range(x0, x0 + len(ma)), ma, label=f"Win Rate ({k})", color=color, linestyle=ls)

    ax_win.set_ylabel("Win Rate")
    if ax_win.lines:
        ax_win.legend(loc="lower left", fontsize=8)
    ax_win.grid(True, alpha=0.35)

    # ---------------- 3) PPO losses ----------------
    epi_u = metrics_history.get("episodes", [])
    if epi_u:
        ax_loss.plot(epi_u, metrics_history.get("loss_pi", []), label="Loss π")
        ax_loss.plot(epi_u, metrics_history.get("loss_v", []), label="Loss V")

    ax_loss.set_ylabel("Loss")
    if ax_loss.lines:
        ax_loss.legend(loc="upper right", fontsize=8)
    ax_loss.grid(True, alpha=0.35)

    # ---------------- 4) Entropy / KL / ClipFrac (split y-axes) ----------------
    if epi_u:
        kl_vals = metrics_history.get("approx_kl", [])
        cf_vals = metrics_history.get("clip_frac", [])
        ent_vals = metrics_history.get("entropy", [])

        # Primary axis: KL + ClipFrac
        line_kl = ax_stats.plot(epi_u, kl_vals, label="KL", linestyle="--", color="tab:orange")[0]
        line_cf = ax_stats.plot(epi_u, cf_vals, label="ClipFrac", linestyle=":", color="tab:green")[0]
        ax_stats.set_ylabel("KL / ClipFrac")
        ax_stats.grid(True, alpha=0.35)

        # Secondary axis: Entropy
        ax_ent = ax_stats.twinx()
        line_ent = ax_ent.plot(epi_u, ent_vals, label="Entropy", color="tab:blue", alpha=0.7)[0]
        ax_ent.set_ylabel("Entropy")

        # Combined legend (lines from both axes)
        lines = [line_kl, line_cf, line_ent]
        labels = [l.get_label() for l in lines]
        ax_stats.legend(lines, labels, loc="upper right", fontsize=8)
    else:
        ax_stats.grid(True, alpha=0.35)

    # ---------------- 5–8) Benchmarks ----------------
    if benchmark_history and benchmark_history.get("episode"):
        _plot_bench_on_axis(ax_bench, benchmark_history, smooth_k=0, training_phases=None)
        ax_bench.set_title("Benchmarks (raw)", fontsize=10)
        ax_bench.set_ylim(0.0, 1.02)

        _plot_bench_on_axis(ax_b3, benchmark_history, smooth_k=3, training_phases=None)
        ax_b3.set_title("Benchmarks (MA 3)", fontsize=10)

        _plot_bench_on_axis(ax_b7, benchmark_history, smooth_k=7, training_phases=None)
        ax_b7.set_title("Benchmarks (MA 7)", fontsize=10)

        _plot_bench_on_axis(ax_b15, benchmark_history, smooth_k=15, training_phases=None)
        ax_b15.set_title("Benchmarks (MA 15)", fontsize=10)
    else:
        for axis in (ax_bench, ax_b3, ax_b7, ax_b15):
            axis.set_visible(False)

    # ---------------- 8) Global depth-weighted score ----------------
    if benchmark_history and benchmark_history.get("episode") and benchmark_history.get("global_score"):
        xg = np.asarray(benchmark_history["episode"], dtype=float)
        yg = np.asarray(benchmark_history["global_score"], dtype=float)

        ax_global.plot(xg, yg, label="Global score", alpha=0.85)
        ma_g = _moving_avg(yg, 3)
        if ma_g is not None and xg.size >= 3:
            x0 = 2
            ax_global.plot(
                xg[x0:x0 + len(ma_g)],
                ma_g,
                label="Global score (MA3)",
                ls="--",
                lw=1.6,
            )

        ax_global.set_ylabel("Global score")
        ax_global.set_ylim(0.0, 1.0)
        ax_global.grid(True, alpha=0.35)
        ax_global.legend(loc="lower left", fontsize=8)
        ax_global.set_title("Depth-weighted global benchmark score", fontsize=10)
    else:
        ax_global.set_visible(False)

    # ---------------- 9–10) H2H panels ----------------
    _plot_h2h_axis(ax_h2h, h2h_history or {}, training_phases=None, title="H2H vs baseline (score ±95% CI)")
    _plot_h2h_axis(ax_eh2h, ensemble_h2h_history or {}, training_phases=None, title="Ensemble H2H (score ±95% CI)")

    # ---------------- 11) CheckPoint score ----------------
    if benchmark_history and benchmark_history.get("episode") and benchmark_history.get("check_score"):
        xc = np.asarray(benchmark_history["episode"], dtype=float)
        yc = np.asarray(benchmark_history["check_score"], dtype=float)

        ax_chkp.plot(xc, yc, label="CheckPoint score", alpha=0.85)
        ma_c = _moving_avg(yc, 3)
        if ma_c is not None and xc.size >= 3:
            x0 = 2
            ax_chkp.plot(
                xc[x0:x0 + len(ma_c)],
                ma_c,
                label="MA3",
                ls="--",
                lw=1.6,
            )

        ax_chkp.set_ylabel("CheckPoint")
        ax_chkp.set_ylim(0.0, 1.0)
        ax_chkp.grid(True, alpha=0.35)
        ax_chkp.legend(loc="lower left", fontsize=8)
    else:
        ax_chkp.set_visible(False)

    # ---------------- Openings panels ----------------
    if use_openings:
        # clamp openings_ylim to sane [0,1]
        oy = openings_ylim
        if oy is not None:
            oy = (0.85, 1.02)

        _draw_openings_on_axes(
            ax_hist,
            ax_rate,
            openings,
            training_phases=phases,
            rate_ylim=oy,
        )

    # ---------------- PPO action mix + opponent usage ----------------
    _plot_ppo_action_mix_axis(ax_mix, policy)
    _plot_opponent_usage_axis(ax_opp, opponent_timeline or [], up_to_episode=episode, normalize=True)

    # ---------------- phase vlines ----------------
    if phases:
        axes_for_vlines = [
            ax_reward, ax_win, ax_loss, ax_stats,
            ax_bench, ax_b3, ax_b7, ax_b15,
            ax_global, ax_h2h, ax_eh2h, ax_chkp,
        ]
        if use_openings and ax_rate is not None:
            axes_for_vlines.append(ax_rate)

        for ax in axes_for_vlines:
            if ax is not None and ax.get_visible():
                draw_phase_vlines(ax, phases, up_to=episode, label=True)

    # ---------------- title ----------------
    total = win_count + loss_count + draw_count
    wr = (win_count / total) if total > 0 else 0.0
    ev_list = metrics_history.get("explained_variance", [])
    ev_last = f"{ev_list[-1]:.3f}" if ev_list else "n/a"

    fig.suptitle(
        f"{title} - Ep {episode} — Phase: {phase_name} | "
        f"W/L/D: {win_count}/{loss_count}/{draw_count} | Win%={wr:.3f} | EV={ev_last}",
        y=0.995,
    )

    if save and save_path:
        fname = f"{save_path}{title}__complete_training_plot.png"
        fig.savefig(fname, dpi=120)
        print(f"[Saved] {fname}")

    return fig
