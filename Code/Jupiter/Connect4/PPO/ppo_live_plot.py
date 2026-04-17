# ppo_live_plot.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    metrics_history: dict,       # dict of lists: episodes, loss_pi, loss_v, entropy, approx_kl, approx_kl_ppo, clip_frac, explained_variance
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
    fig_width: float = 26.0,     # inches (18–26 on a big monitor)
    fig_dpi: int = 140,          # higher = sharper + heavier
    separate_eval_order=None,    # list[tuple[display_name, source_label]]
    separate_eval_cols: int = 4,
):
    """
    Multi-panel live report.
    Openings panels appear only if `openings` is provided.

    Separate per-opponent eval panels are read from:
        benchmark_history["by_opponent"][label]
    """

    use_openings = openings is not None

    if separate_eval_order is None:
        separate_eval_order = [
            ("Leftmost", "Leftmost"),
            ("Center", "Center"),
            ("Random", "Random"),
            ("L1", "Lookahead-1"),
            ("L2", "Lookahead-2"),
            ("L3", "Lookahead-3"),
            ("L4", "Lookahead-4"),
            ("L5", "Lookahead-5"),
            ("L6", "Lookahead-6"),
            ("L7", "Lookahead-7"),
            ("L9", "Lookahead-9"),
            ("L11", "Lookahead-11"),
            ("L13", "Lookahead-13"),
        ]

    # ---------------- helpers ----------------
    def _opp_color(lab: str) -> str:
        if lab == "R":
            return "#1f77b4"
        if lab == "POP":
            return "#000000"
        if lab == "SP":
            return "#9467bd"
        if lab.startswith("L") and lab[1:].isdigit():
            depth = int(lab[1:])
            palette = [
                "#ff7f0e", "#2ca02c", "#d62728", "#8c564b",
                "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            ]
            return palette[depth % len(palette)]
        return "#888888"

    def _get_eval_xy(hist, source_label):
        """
        Read from benchmark_history['by_opponent'][source_label].
        Fallback to top-level hist[source_label] only if needed.
        """
        if not hist or "episode" not in hist:
            return None, None

        ys = None
        by_opp = hist.get("by_opponent", None)

        if isinstance(by_opp, dict) and source_label in by_opp:
            ys = by_opp[source_label]
        elif source_label in hist:
            ys = hist[source_label]
        else:
            return None, None

        try:
            xs = np.asarray(hist["episode"], dtype=float)
            ys = np.asarray(ys, dtype=float)
        except Exception:
            return None, None

        n = min(len(xs), len(ys))
        if n <= 0:
            return None, None

        xs = xs[:n]
        ys = ys[:n]

        mask = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(mask):
            return None, None

        return xs[mask], ys[mask]

    def _plot_single_eval_axis(ax, hist, display_name, source_label):
        xs, ys = _get_eval_xy(hist, source_label)
        if xs is None or ys is None or len(xs) == 0:
            ax.set_visible(False)
            return

        ax.plot(xs, ys, alpha=0.45, lw=1.1)

        ma3 = _moving_avg(ys, 3)
        if ma3 is not None and len(xs) >= 3:
            ax.plot(xs[2:2 + len(ma3)], ma3, ls="--", lw=1.3)

        ma10 = _moving_avg(ys, 10)
        if ma10 is not None and len(xs) >= 10:
            ax.plot(xs[9:9 + len(ma10)], ma10, ls=":", lw=1.3)

        ax.set_title(display_name, fontsize=8, pad=2)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.28)

    eval_items = []
    if benchmark_history:
        for display_name, source_label in separate_eval_order:
            xs, ys = _get_eval_xy(benchmark_history, source_label)
            if xs is not None and ys is not None and len(xs) > 0:
                eval_items.append((display_name, source_label))

    eval_n = len(eval_items)
    eval_cols = max(1, int(separate_eval_cols))
    eval_rows = int(np.ceil(eval_n / eval_cols)) if eval_n > 0 else 1
    eval_block_height = max(5.0, 3.0 * eval_rows)

    # ---------------- layout ----------------
    if use_openings:
        nrows = 17
        heights = [
            6.0,   # 0 reward
            10.0,  # 1 win rate
            5.0,   # 2 losses
            10.0,  # 3 entropy / KL / clipfrac
            10.0,  # 4 benchmarks raw
            2.3,   # 5 benchmarks MA3
            2.3,   # 6 benchmarks MA7
            10.0,  # 7 benchmarks MA15
            16.0,  # 8 global score
            2.6,   # 9 H2H
            2.6,   # 10 ensemble H2H
            3.0,   # 11 checkpoint score
            eval_block_height,  # 12 separate evals
            2.1,   # 13 openings hist
            8.0,   # 14 a0@center
            2.3,   # 15 action mix
            2.3,   # 16 opponent usage
        ]
    else:
        nrows = 15
        heights = [
            6.0,   # 0 reward
            10.0,  # 1 win rate
            5.0,   # 2 losses
            10.0,  # 3 entropy / KL / clipfrac
            10.0,  # 4 benchmarks raw
            2.3,   # 5 benchmarks MA3
            2.3,   # 6 benchmarks MA7
            10.0,  # 7 benchmarks MA15
            16.0,  # 8 global score
            2.6,   # 9 H2H
            2.6,   # 10 ensemble H2H
            3.0,   # 11 checkpoint score
            eval_block_height,  # 12 separate evals
            2.3,   # 13 action mix
            2.3,   # 14 opponent usage
        ]

    fig = plt.figure(figsize=(fig_width, sum(heights) + 1.0), dpi=fig_dpi)
    fig.subplots_adjust(left=0.055, right=0.995)
    gs = fig.add_gridspec(nrows, 1, height_ratios=heights, hspace=0.45)

    # ---------------- axes ----------------
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

    eval_axes = []

    # ---------------- separate benchmark grid ----------------
    # FAIL-SAFE: if anything here breaks, the rest of the plot still renders
    try:
        if eval_n > 0:
            eval_gs = gs[12, 0].subgridspec(eval_rows, eval_cols, hspace=0.35, wspace=0.16)

            first_eval_ax = None
            for i, (display_name, source_label) in enumerate(eval_items):
                r = i // eval_cols
                c = i % eval_cols

                if first_eval_ax is None:
                    ax = fig.add_subplot(eval_gs[r, c], sharex=ax_reward)
                    first_eval_ax = ax
                else:
                    ax = fig.add_subplot(eval_gs[r, c], sharex=ax_reward, sharey=first_eval_ax)

                _plot_single_eval_axis(ax, benchmark_history, display_name, source_label)

                if r < eval_rows - 1:
                    ax.tick_params(labelbottom=False)

                if c == 0:
                    ax.set_ylabel("WR", fontsize=8)
                else:
                    ax.tick_params(labelleft=False)

                ax.tick_params(axis="both", labelsize=7)
                eval_axes.append(ax)

            # hide unused slots
            for j in range(eval_n, eval_rows * eval_cols):
                r = j // eval_cols
                c = j % eval_cols
                ax_unused = fig.add_subplot(eval_gs[r, c])
                ax_unused.set_visible(False)

            if eval_axes:
                eval_axes[0].text(
                    0.0,
                    1.18,
                    "Benchmarks by opponent",
                    transform=eval_axes[0].transAxes,
                    fontsize=10,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                )
        else:
            ax_eval_placeholder = fig.add_subplot(gs[12, 0], sharex=ax_reward)
            ax_eval_placeholder.set_visible(False)

    except Exception as e:
        print(f"[plot_live_training_ppo] separate benchmark panels disabled: {e}")
        try:
            ax_eval_placeholder = fig.add_subplot(gs[12, 0], sharex=ax_reward)
            ax_eval_placeholder.set_visible(False)
        except Exception:
            pass
        eval_axes = []

    # ---------------- lower axes ----------------
    if use_openings:
        ax_hist = fig.add_subplot(gs[13, 0])
        ax_rate = fig.add_subplot(gs[14, 0], sharex=ax_reward)
        ax_mix = fig.add_subplot(gs[15, 0])
        ax_opp = fig.add_subplot(gs[16, 0])
    else:
        ax_hist = None
        ax_rate = None
        ax_mix = fig.add_subplot(gs[13, 0])
        ax_opp = fig.add_subplot(gs[14, 0])

    # ---------------- 1) Rewards ----------------
    ax_reward.plot(reward_history, label="Reward", alpha=0.55)

    for k, lw, ls in [(25, 2.0, "-"), (100, 1.5, "--"), (500, 1.5, "--")]:
        ma = _moving_avg(reward_history, k)
        if ma is not None:
            x0 = k - 1
            ax_reward.plot(
                range(x0, x0 + len(ma)),
                ma,
                label=f"{k}-ep MA",
                linewidth=lw,
                linestyle=ls,
            )

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
                ax_win.plot(
                    range(x0, x0 + len(ma)),
                    ma,
                    label=f"Win Rate ({k})",
                    color=color,
                    linestyle=ls,
                )

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

    # ---------------- 4) Entropy / KL / ClipFrac ----------------
    if epi_u:
        kl_vals = metrics_history.get("approx_kl", [])
        kl_ppo_vals = metrics_history.get("approx_kl_ppo", [])
        cf_vals = metrics_history.get("clip_frac", [])
        ent_vals = metrics_history.get("entropy", [])

        line_kl = ax_stats.plot(
            epi_u, kl_vals, label="KL (old-new)", linestyle="--", color="tab:orange"
        )[0]

        line_klppo = None
        if kl_ppo_vals:
            line_klppo = ax_stats.plot(
                epi_u, kl_ppo_vals, label="KL (ppo)", linestyle="-.", color="tab:red", alpha=0.85
            )[0]

        line_cf = ax_stats.plot(
            epi_u, cf_vals, label="ClipFrac", linestyle=":", color="tab:green"
        )[0]

        ax_stats.set_ylabel("KL / ClipFrac")
        ax_stats.grid(True, alpha=0.35)

        ax_ent = ax_stats.twinx()
        line_ent = ax_ent.plot(
            epi_u, ent_vals, label="Entropy", color="tab:blue", alpha=0.7
        )[0]
        ax_ent.set_ylabel("Entropy")

        lines = [line_kl]
        if line_klppo is not None:
            lines.append(line_klppo)
        lines += [line_cf, line_ent]
        labels = [l.get_label() for l in lines]
        ax_stats.legend(lines, labels, loc="upper right", fontsize=8)
    else:
        ax_stats.grid(True, alpha=0.35)

    # ---------------- 5–8) Benchmarks aggregate ----------------
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

        n = min(xg.size, yg.size)
        xg = xg[:n]
        yg = yg[:n]

        # raw + smooth
        ax_global.plot(xg, yg, label="Global score", alpha=0.85)

        ma_g = _moving_avg(yg, 3)
        if ma_g is not None and xg.size >= 3:
            ax_global.plot(
                xg[2:2 + len(ma_g)],
                ma_g,
                label="Global score (MA3)",
                ls="--",
                lw=1.6,
            )

        ma10_g = _moving_avg(yg, 10)
        if ma10_g is not None and xg.size >= 10:
            ax_global.plot(
                xg[9:9 + len(ma10_g)],
                ma10_g,
                label="Global score (MA10)",
                ls=":",
                lw=1.6,
            )

        # baseline = first observed global score
        first_score = float(yg[0])
        ax_global.axhline(
            first_score,
            color="red",
            lw=1.4,
            ls="-",
            alpha=0.9,
            label=f"First eval = {first_score:.3f}",
            zorder=1,
        )

        # standard axis formatting
        ax_global.set_ylabel("Global score")
        ax_global.set_ylim(0.0, 1.0)

        # full left-axis labels every 0.1
        yticks_main = np.round(np.arange(0.0, 1.01, 0.1), 2)
        ax_global.set_yticks(yticks_main)
        ax_global.set_yticklabels([f"{v:.1f}" for v in yticks_main])

        # keep broader structure visible
        ax_global.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax_global.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
        ax_global.minorticks_on()

        ax_global.grid(True, which="major", alpha=0.45, lw=0.9)
        ax_global.grid(True, which="minor", alpha=0.18, lw=0.5)

        # emphasized upper-band guide lines: 0.90 .. 1.00
        upper_band_ticks = np.round(np.arange(0.90, 1.001, 0.02), 2)
        for yv in upper_band_ticks:
            ax_global.axhline(
                yv,
                color="crimson",
                lw=0.8 if yv < 1.0 else 1.0,
                ls=":" if yv < 1.0 else "--",
                alpha=0.28,
                zorder=0,
            )

        # right-side dense labels for the important region
        ax_global_hi = ax_global.twinx()
        ax_global_hi.set_ylim(ax_global.get_ylim())
        ax_global_hi.set_yticks(upper_band_ticks)
        ax_global_hi.set_yticklabels([f"{v:.2f}" for v in upper_band_ticks])
        ax_global_hi.tick_params(axis="y", labelsize=8, pad=2)
        ax_global_hi.set_ylabel("0.90–1.00 focus")

        ax_global.legend(loc="lower left", fontsize=8)
        ax_global.set_title("Depth-weighted global benchmark score", fontsize=10)
    else:
        ax_global.set_visible(False)

    # ---------------- 9–10) H2H panels ----------------
    _plot_h2h_axis(
        ax_h2h,
        h2h_history or {},
        training_phases=None,
        title="H2H vs baseline (score ±95% CI)",
    )
    _plot_h2h_axis(
        ax_eh2h,
        ensemble_h2h_history or {},
        training_phases=None,
        title="Ensemble H2H (score ±95% CI)",
    )

    # ---------------- 11) CheckPoint score ----------------
    if benchmark_history and benchmark_history.get("episode") and benchmark_history.get("check_score"):
        xc = np.asarray(benchmark_history["episode"], dtype=float)
        yc = np.asarray(benchmark_history["check_score"], dtype=float)

        n = min(xc.size, yc.size)
        xc = xc[:n]
        yc = yc[:n]

        ax_chkp.plot(xc, yc, label="CheckPoint score", alpha=0.85)
        ma_c = _moving_avg(yc, 3)
        if ma_c is not None and xc.size >= 3:
            ax_chkp.plot(
                xc[2:2 + len(ma_c)],
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

        axes_for_vlines.extend(eval_axes)

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

    fig.subplots_adjust(left=0.055, right=0.995)

    if save and save_path:
        fname = f"{save_path}{title}__complete_training_plot.png"
        fig.savefig(fname, dpi=120)
        print(f"[Saved] {fname}")

    return fig