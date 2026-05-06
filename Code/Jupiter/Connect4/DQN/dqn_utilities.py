# dqn_utilities.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from copy import deepcopy
from PPO.ppo_training_phases_config import TRAINING_PHASES
from C4.connect4_env import Connect4Env
from DQN.eval_utilities import evaluate_agent_model
from IPython.display import display, HTML
import torch
from C4.eval_oppo_dict import EVAL_CFG
from pathlib import Path
import shutil
import re
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from collections import Counter


TAU_DEFAULT = 0.006                         # phase controlled
GUARD_DEFAULT = 0.30                          # phase controlled
WIN_DEFAULT = 1.0                           # phase controlled
CENTER_START = 0.0                          # phase controlled
MIN_SEED_FRACTION = 0.0                           # phase controlled
MAX_SEED_FRACTION = 1.0                           # phase controlled
LR = 1e-4                          # phase controlled


# === QR-DQN training helpers  ===

def qr_training_updates(agent, env, batch_size: int):
    """
    Mirrors your 3-tier update schedule, but as a helper.
    Uses agent.per_mix_1step if not overridden.
    """
    mem_len = len(agent.memory)
    MIX = getattr(agent, "per_mix_1step", 0.70)

    WARMUP_HALF = max(batch_size * 2, 256)
    WARMUP_FULL = max(batch_size * 5, 1024)

    updates_done = 0

    if mem_len >= WARMUP_FULL:
        # Full batches
        # 4..16 updates per episode
        updates = 4 + min(12, env.ply // 2)
        for _ in range(updates):
            agent.replay(batch_size, mix_1step=MIX)
        updates_done = updates

    elif mem_len >= WARMUP_HALF:
        # Smaller batches
        # adaptive mini-batch
        b = max(32, min(batch_size, mem_len // 16))
        updates = 2 + min(6, env.ply // 3)                  # 2..8 updates
        for _ in range(updates):
            agent.replay(b, mix_1step=MIX)
        updates_done = updates

    else:
        if mem_len >= 64:
            for _ in range(2):
                agent.replay(32, mix_1step=MIX)
            updates_done = 2

    return updates_done


def maybe_phase_checkpoint(phase_changed, phase, agent, episode, model_dir, session_prefix):
    """
    Your exact checkpoint pattern, as a helper. No-op if phase didn't change.
    """
    import time
    import torch
    if phase_changed and phase is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = f"{model_dir}{session_prefix}_Temp_C4_{timestamp} eps-{episode+1} phase-{phase}.pt"
        default_model_path = f"phase-{phase} Interim C4.pt"
        torch.save(agent.model.state_dict(), model_path)
        torch.save(agent.model.state_dict(), default_model_path)
        print(f"Temp Model saved to {model_path}")


# ------------------------------- #

def reset_dir(path):
    p = Path(path).resolve()
    if str(p) in ("/", "C:\\", "C:/"):
        raise RuntimeError(f"Refusing to delete root path: {p}")
    shutil.rmtree(p, ignore_errors=True)   # delete dir and all contents
    p.mkdir(parents=True, exist_ok=True)   # recreate empty dir


@torch.no_grad()
def _flatten_params(module: torch.nn.Module) -> torch.Tensor:
    """Concatenate all trainable parameters into a single 1-D tensor (cpu, detached)."""
    if module is None:
        return None
    parts = []
    for p in module.parameters():
        if p.requires_grad:
            parts.append(p.detach().reshape(-1).cpu())
    return torch.cat(parts) if parts else torch.tensor([], dtype=torch.float32)


@torch.no_grad()
def compute_target_lag(agent, eps: float = 1e-12) -> float | None:
    """
    Return ||θ - θ̄|| / ||θ|| for the agent's online vs target networks.
    If unavailable, returns None.
    """
    if not hasattr(agent, "model") or not hasattr(agent, "target_model"):
        return None
    w = _flatten_params(agent.model)
    wt = _flatten_params(agent.target_model)
    if w.numel() == 0 or wt.numel() == 0 or w.numel() != wt.numel():
        return None
    num = torch.linalg.norm(w - wt, ord=2)
    den = torch.linalg.norm(w,     ord=2).clamp(min=eps)
    return float((num / den).item())


def maybe_log_target_lag(agent) -> float | None:
    """
    Compute lag and append to agent.target_lag_hist.
    """
    val = compute_target_lag(agent)
    if val is not None:
        if not hasattr(agent, "target_lag_hist"):
            agent.target_lag_hist = []
        agent.target_lag_hist.append(val)
    return val

# ---- Phase boundary helpers (episode-indexed) ----


def _phase_boundaries(training_phases: dict, up_to: int | None = None):
    """Return [(name, end_ep), ...] sorted by end_ep and clipped to up_to."""
    pairs = [
        (name, int(meta["length"]))
        for name, meta in training_phases.items()
        if meta.get("length") is not None
    ]
    pairs.sort(key=lambda p: p[1])
    if up_to is not None:
        pairs = [p for p in pairs if p[1] <= up_to]
    return pairs


def draw_phase_vlines(
    ax, training_phases: dict, up_to: int | None = None,
    label: bool = True, color: str = "k", ls: str = ":", lw: float = 1.0, alpha: float = 0.6,
    label_y: float | None = None, fontsize: int = 8, ha: str = "left", va: str = "top"
):
    """Draw vertical lines (and labels) at phase boundaries on an episode axis."""
    bounds = _phase_boundaries(training_phases, up_to)
    if not bounds:
        return

    ymin, ymax = ax.get_ylim()
    ytxt = (ymax * 0.95) if label_y is None else label_y
    for name, ep in bounds:
        ax.axvline(ep, color=color, linestyle=ls, linewidth=lw, alpha=alpha)
        if label:
            ax.text(ep, ytxt, name, rotation=90, ha=ha,
                    va=va, fontsize=fontsize, alpha=alpha)


class OpeningTracker:
    def __init__(self, cols=7, log_every=25, ma_window=25, create_figure: bool = False):
       self.cols = cols
       self.center = cols // 2
       self.log_every = int(log_every)
       self.ma_window = int(ma_window)
       self.kpis = init_opening_kpis(cols)
       self.episodes, self.a_center_rates, self.o_center_rates = [], [], []

       # Only create a separate figure if you explicitly want it
       self.fig = None
       if create_figure:
           self.fig, (self.ax_hist, self.ax_rate) = plt.subplots(
               2, 1, figsize=(7.5, 6.5), constrained_layout=True)
           self.fig.canvas.manager.set_window_title("Opening diagnostics")
           self._init_hist_axes()
           self._init_rate_axes()

       self.phase_marks = []

    # --- unchanged public API ---
    def get_last_logged_center_rate(self):
        if not self.a_center_rates: return 0.0
        return self.a_center_rates[-1]

    def on_first_move(self, col: int, is_agent: bool):
        record_first_move(self.kpis, col, "agent" if is_agent else "opp")

    def maybe_log(self, episode_idx: int):
        if episode_idx % self.log_every != 0:
            return
        a_tot = max(1, self.kpis["agent_first_total"])
        o_tot = max(1, self.kpis["opp_first_total"])
        a_rate = self.kpis["agent_first_counts"][self.center] / a_tot
        o_rate = self.kpis["opp_first_counts"][self.center] / o_tot
        self.episodes.append(episode_idx)
        self.a_center_rates.append(a_rate)
        self.o_center_rates.append(o_rate)

        # If a standalone fig exists, refresh it; otherwise embedding will draw
        if self.fig is not None:
            self.redraw()

    # ---------- plotting internals ----------
    def _init_hist_axes(self):
        self.ax_hist.set_title("First-move histogram (agent vs opponent)")
        self.ax_hist.set_xlabel("Column")
        self.ax_hist.set_ylabel("Rate")
        self.ax_hist.set_xticks(range(self.cols))
        # center highlight
        self.ax_hist.axvline(self.center, color="k",
                             linestyle="--", alpha=0.25)

    def _init_rate_axes(self):
        self.ax_rate.set_title("Agent center rate over time")
        self.ax_rate.set_xlabel("Episode")
        self.ax_rate.set_ylabel("Rate")
        self.ax_rate.grid(True, alpha=0.3)
        self.ax_rate.set_ylim(0.0, 1.0)
        self.a_line, = self.ax_rate.plot(
            [], [], label="agent a0@center", lw=1.5)
        self.o_line, = self.ax_rate.plot(
            [], [], label="opp a0@center", lw=1.0, alpha=0.6)
        self.a_ma_line, = self.ax_rate.plot(
            [], [], linestyle="--", lw=2.0, label=f"agent MA (w={self.ma_window})")
        self.ax_rate.legend(loc="lower right")

    def _moving_avg(self, y):
        w = self.ma_window
        if len(y) < max(2, w):
            return [], []
        # center the MA visually on the trailing point
        ma = np.convolve(y, np.ones(w)/w, mode="valid")
        x = self.episodes[len(self.episodes)-len(ma):]
        return x, ma

    def redraw(self):
        # --- histogram (rates) ---
        self.ax_hist.cla()
        self._init_hist_axes()
        a_tot = max(1, self.kpis["agent_first_total"])
        o_tot = max(1, self.kpis["opp_first_total"])
        a_rates = self.kpis["agent_first_counts"] / a_tot
        o_rates = self.kpis["opp_first_counts"] / o_tot
        xs = np.arange(self.cols)
        w = 0.38
        self.ax_hist.bar(xs - w/2, a_rates, width=w, label="Agent", alpha=0.9)
        self.ax_hist.bar(xs + w/2, o_rates, width=w,
                         label="Opponent", alpha=0.6)
        self.ax_hist.legend(loc="upper right")
        self.ax_hist.set_ylim(
            0, max(0.01, float(max(a_rates.max(), o_rates.max(), 0.3))))

        # --- time series ---
        self.a_line.set_data(self.episodes, self.a_center_rates)
        self.o_line.set_data(self.episodes, self.o_center_rates)
        x_ma, y_ma = self._moving_avg(self.a_center_rates)
        self.a_ma_line.set_data(x_ma, y_ma)

        # phase markers (optional)
        for ep, lab in self.phase_marks:
            self.ax_rate.axvline(ep, color="k", linestyle=":", alpha=0.2)
            self.ax_rate.text(ep, 0.02, lab, rotation=90,
                              va="bottom", ha="right", fontsize=8, alpha=0.6)

        # autoscale x only
        self.ax_rate.set_xlim(
            0 if not self.episodes else self.episodes[0], max(1, self.episodes[-1]))
        self.ax_rate.figure.canvas.draw_idle()
        self.ax_rate.figure.canvas.flush_events()

    # optional helper if you want vertical lines at phase changes
    def mark_phase(self, episode_idx: int, label: str):
        self.phase_marks.append((episode_idx, label))


def get_phase(episode):
    for phase_name, phase_data in TRAINING_PHASES.items():
        end_episode = phase_data["length"]
        if end_episode is None or episode < end_episode:
            return (
                phase_name,
                phase_data["weights"],
                phase_data["epsilon"],
                phase_data["memory_prune_low"],
                phase_data.get("epsilon_min", 0.0),
                phase_data.get("TU", 500),
                phase_data.get("TU_mode", "hard"),
                phase_data.get("tau", TAU_DEFAULT),
                phase_data.get("guard_prob", GUARD_DEFAULT),
                phase_data.get("center_start", CENTER_START),
                phase_data.get("win_now_prob", WIN_DEFAULT),
                phase_data.get("min_seed_fraction", MIN_SEED_FRACTION),
                phase_data.get("max_seed_fraction", MAX_SEED_FRACTION),
                phase_data.get("lr", LR),
            )


def handle_phase_change(
    agent,
    new_phase, current_phase,
    epsilon, memory_prune_low, epsilon_min,
    prev_frozen_opp,
    env, episode, device,
    Lookahead, training_session,
    guard_prob, center_start, win_now_prob,
    min_seed_fraction, max_seed_fraction,
    lr
):
    frozen_opp = prev_frozen_opp

    if new_phase.startswith("SelfPlay"):
        if frozen_opp is None:
            frozen_opp = deepcopy(agent)
            frozen_opp.model.eval()
            frozen_opp.target_model.eval()
            frozen_opp.epsilon = 0.0
            frozen_opp.epsilon_min = 0.0
            frozen_opp.guard_prob = 0.0
            frozen_opp.center_start = 0.0
    else:
        frozen_opp = None

    if new_phase != current_phase:
        # exploration resets
        agent.epsilon = epsilon
        agent.epsilon_min = epsilon_min

        # phase policy knobs
        agent.guard_prob = guard_prob
        agent.center_start = center_start
        agent.win_now_prob = win_now_prob

        # replay / sampling knobs
        agent.min_seed_frac = min_seed_fraction
        agent.max_seed_frac = max_seed_fraction
        agent.memory.prune(memory_prune_low, mode="low_priority")

        # optimizer LR
        for g in agent.optimizer.param_groups:
            g["lr"] = lr
        agent.lr = lr

        return new_phase, frozen_opp

    return current_phase, frozen_opp


def evaluate_final_result(env, agent_player):
    if env.winner == agent_player:
        return 1
    elif env.winner == -agent_player:
        return -1
    elif env.winner == 0:
        return 0.5
    else:
        return None


def map_final_result_to_reward(final_result):
    if final_result == 1:
        return Connect4Env.WIN_REWARD
    elif final_result == -1:
        return Connect4Env.LOSS_PENALTY
    elif final_result == 0.5:
        return Connect4Env.DRAW_REWARD
    else:
        return 0


def track_result(final_result, win_history):
    if final_result == 1:
        win_history.append(1)
        return 1, 0, 0
    elif final_result == -1:
        win_history.append(0)
        return 0, 1, 0
    elif final_result == 0.5:
        win_history.append(0)
        return 0, 0, 1
    else:
        print("DEBUG:", final_result, type(final_result))
        print(f"Final result before track_result: {final_result}")

        raise ValueError(
            "Invalid final_result — env.winner was not set correctly. Lovro, get a grip!")


def plot_moving_averages(ax, data, windows, colors=None, styles=None, labels=None):
    for i, win in enumerate(windows):
        if len(data) >= win:
            avg = np.convolve(data, np.ones(win) / win, mode='valid')
            offset = win - 1
            ax.plot(range(offset, len(data)), avg,
                    label=labels[i] if labels else f'{win}-ep Avg',
                    color=colors[i] if colors else None,
                    linestyle=styles[i] if styles else None)

# === FINAL SUMMARY PLOT ===


def save_final_winrate_plot(win_history, training_phases, save_path, session_name="default_session"):
    if len(win_history) < 25:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    win_avg = np.convolve(win_history, np.ones(25)/25, mode='valid')
    ax.plot(range(24, len(win_history)), win_avg,
            label='Win Rate (25 ep)', color='green')

    ax.set_title("Final Win Rate Over Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate")
    ax.grid(True)
    ax.legend()
    draw_phase_vlines(ax, training_phases, up_to=len(win_history), label=True)
    full_path = f"{save_path}DQN-{session_name}_final_winrate.png"
    fig.savefig(full_path)
    plt.close(fig)


# === PHASE-WISE SUMMARY LOGGING ===
def log_summary_stats(
    episode,
    reward_history,
    win_history,
    phase,
    strategy_weights,
    agent,
    win_count,
    loss_count,
    draw_count,
    summary_stats_dict
):
    recent_rewards = reward_history[-25:] if len(
        reward_history) >= 25 else reward_history
    recent_win_rate = np.mean(
        win_history[-25:]) if len(win_history) >= 25 else np.mean(win_history)

    summary_stats_dict[episode] = {
        "phase": phase,
        "strategy_weights": strategy_weights.copy() if strategy_weights else None,
        "epsilon": agent.epsilon,
        "wins": win_count,
        "losses": loss_count,
        "draws": draw_count,
        "avg_reward_25": round(np.mean(recent_rewards), 2),
        "win_rate_25": round(recent_win_rate, 2)
    }


# ----- benchmarking ----------------

def opponent_weight(label: str,
                    base: float = 1.4,
                    random_weight: float = 1.0,
                    default_weight: float = 1.0) -> float:
    """
    Exponential-ish weight by opponent depth.

    - 'Random' gets random_weight.
    - 'Lookahead-k' (or anything with a number) gets base**k.
    - Anything else falls back to default_weight.

    Examples with base=1.4:
        L1 -> 1.4
        L2 -> 1.4^2 ≈ 1.96
        L3 -> 1.4^3 ≈ 2.74
        L4 -> 1.4^4 ≈ 3.84
    """
    s = str(label)

    if "Random" in s:
        return float(random_weight)

    m = re.search(r"(\d+)", s)
    if m:
        depth = int(m.group(1))
        return float(base) ** depth

    return float(default_weight)


def global_score_from_results(results: dict,
                              base: float = 1.4) -> float:
    """
    Compute a single global score G in [0,1] from one benchmark snapshot:

        G = sum_i w_i * WR_i / sum_i w_i

    where WR_i is win_rate vs opponent i, w_i = opponent_weight(label_i).
    """
    num = 0.0
    den = 0.0
    for label, stats in results.items():
        wr = float(stats.get("win_rate", 0.0))
        w = opponent_weight(label, base=base)
        num += w * wr
        den += w
    return num / den if den > 1e-12 else float("nan")


# ---------- main function ----------

def update_benchmark_winrates(
    agent,
    env,
    device,
    Lookahead,
    episode: int,
    history: dict | None = None,
    save: str | None = None,
    opponents_cfg: dict = EVAL_CFG,
    score: float = 0.0,
    ensemble_score: float = 0.0,
    center_rate: float = 0.0,
    global_base: float = 1.4,      # <-- tweak exponential strength here
):
    # init history
    if history is None:
        history = {
            "episode":        [],
            "by_opponent":    {k: [] for k in opponents_cfg},
            "score":          [],
            "ensemble_score": [],
            "global_score":   [],
            "check":          [],
            "center_rate":    [],
            "check_score":    [],
        }

    is_dqn = hasattr(agent, "model") and hasattr(agent, "target_model")

    if is_dqn:
        # DQN path (unchanged)
        results = evaluate_agent_model(agent, env, opponents_cfg, device, Lookahead)

    else:
        # PPO / policy path (CNet192 1-ch POV):
        # - accept ActorCritic wrapper (agent.net) or raw nn.Module
        model = getattr(agent, "net", None)
        if model is None:
            model = agent  # already a torch.nn.Module

        # new eval uses bitboard utils; env not required
        from PPO.ppo_agent_eval import evaluate_suite

        la = None
        if Lookahead is not None:
            try:
                la = Lookahead()
            except Exception:
                la = None  # allow running without lookahead opponents

        suite_df = evaluate_suite(
            policy=model,
            opponents=opponents_cfg,          # dict[label -> n_games]
            device=device,
            lookahead=la,
            seed=int(episode) + 666,
            swap_sides=True,
            policy_deterministic=True,
            policy_temperature=1.0,
            progress=False,
        )

        # convert DF -> results dict compatible with global_score_from_results()
        results = {}
        for _, r in suite_df.iterrows():
            lab = str(r["opponent"])
            results[lab] = {
                "win_rate": float(r.get("win_rate", 0.0)),
                "loss_rate": float(r.get("loss_rate", 0.0)),
                "draw_rate": float(r.get("draw_rate", 0.0)),
                "wins": int(r.get("wins", 0)),
                "losses": int(r.get("losses", 0)),
                "draws": int(r.get("draws", 0)),
                "games": int(r.get("games", 0)),
                "score": float(r.get("score", 0.0)),
                "avg_plies": float(r.get("avg_plies", 0.0)),
            }

        # ensure all requested labels exist (robust logging)
        for label in opponents_cfg:
            results.setdefault(str(label), {"win_rate": 0.0})

    # per-benchmark global score (depth-weighted)
    gscore = global_score_from_results(results, base=global_base)

    # composite "check_score" (your heuristic)
    check_score = float(score) * float(score) * float(ensemble_score) * float(center_rate)

    # append to history (keep lengths aligned)
    history["episode"].append(int(episode))
    history["score"].append(float(score))
    history["ensemble_score"].append(float(ensemble_score))
    history["global_score"].append(float(gscore))
    history["center_rate"].append(float(center_rate))   
    history["check_score"].append(float(check_score))
    history["check"].append(None)

    for label in opponents_cfg:
        history["by_opponent"].setdefault(label, [])
        history["by_opponent"][label].append(float(results[str(label)]["win_rate"]))

    # persistent log (one row per benchmark)
    if save is not None:
        row = {
            "episode":        int(episode),
            "check_score":    float(check_score),
            "score":          float(score),
            "ensemble_score": float(ensemble_score),
            "global_score":   float(gscore),
            "check":          None,
            "center_rate":    float(center_rate),
        }
        for label in opponents_cfg:
            row[f"{label}_WR"] = float(results[str(label)]["win_rate"])

        df = pd.DataFrame([row])
        if os.path.exists(save):
            try:
                existing = pd.read_excel(save)
                out = pd.concat([existing, df], ignore_index=True)
            except Exception:
                out = df
            out.to_excel(save, index=False)
        else:
            df.to_excel(save, index=False)

    return history



def _plot_bench_on_axis(
    ax,
    history: dict,
    smooth_k: int | tuple | list = 0,
    training_phases: dict | None = None,
    epsilon_history=None,
    epsilon_min_history=None,
):
    """Draw benchmark win-rates.

    • smooth_k in {0,1} → RAW-ONLY: solid line + scatter + last-point label
    • smooth_k >= 2 or tuple/list → SMOOTHED-ONLY: solid smoothed line(s), plus last raw dot + label
    """
    if not history or not history.get("episode"):
        ax.set_visible(False)
        return

    # prepare smoothing spec
    if isinstance(smooth_k, (tuple, list)):
        ks = sorted({int(k) for k in smooth_k if int(k) >= 2})
    else:
        k = int(smooth_k)
        ks = [k] if k >= 2 else []

    x_all = np.asarray(history["episode"], dtype=int)

    for label, ys in history.get("by_opponent", {}).items():
        y = np.asarray(ys, dtype=float)
        if y.size == 0: continue
        x = x_all[-y.size:]

        if ks:
            # --- smoothed-only ---
            y_s = _smooth_same(y, ks[0])
            line, = ax.plot(x, y_s, linewidth=1.8, label=label,
                            solid_capstyle="round")
            col = line.get_color()
            for k in ks[1:]:
                ax.plot(x, _smooth_same(y, k), linewidth=2.2,
                        solid_capstyle="round", color=col)

            # last raw dot + label
            ax.scatter([x[-1]], [y[-1]], s=22, color=col, zorder=3)
            ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]),
                        textcoords="offset points", xytext=(6, -6),
                        ha="left", fontsize=8)
        else:
            # --- raw-only: solid line + scatter + last-value label ---
            line, = ax.plot(x, y, linewidth=1.6, label=label,
                            solid_capstyle="round")
            col = line.get_color()
            ax.scatter(x, y, s=14, alpha=0.95, color=col, label=None)
            ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]),
                        textcoords="offset points", xytext=(6, -6),
                        ha="left", fontsize=8)

    # epsilon traces
    if epsilon_history is not None and len(epsilon_history) > 0:
        x_eps = np.arange(len(epsilon_history), dtype=int)
        ax.plot(x_eps, epsilon_history, linewidth=1.2,
                color="#bbbbbb", label="ε")
    if epsilon_min_history is not None and len(epsilon_min_history) > 0:
        x_eps_min = np.arange(len(epsilon_min_history), dtype=int)
        ax.plot(x_eps_min, epsilon_min_history,
                linewidth=1.2, color="#bbbbbb", label="ε_min")

    if training_phases:
        draw_phase_vlines(ax, training_phases,
                          up_to=int(x_all.max()), label=True)

    ax.set_ylabel("Bench WR")
    ax.set_ylim(0.0, 1.0)

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, alpha=0.3, axis="y")

    ax.legend(loc="upper center", ncol=1, fontsize=8)


# first move tracking

def init_opening_kpis(cols=7):
    return {
        "agent_first_counts": np.zeros(cols, dtype=np.int64),
        "opp_first_counts":   np.zeros(cols, dtype=np.int64),
        "agent_first_total":  0,
        "opp_first_total":    0,
    }


def record_first_move(kpis, col, who):
    if col is None:
        return
    if who == "agent":
        kpis["agent_first_counts"][col] += 1
        kpis["agent_first_total"] += 1
    else:
        kpis["opp_first_counts"][col] += 1
        kpis["opp_first_total"] += 1


def summarize_opening_kpis(kpis):
    def _rate(cnts, total, col):
        return float(cnts[col] / total) if total > 0 else 0.0
    a_tot = kpis["agent_first_total"]
    o_tot = kpis["opp_first_total"]
    a_mode = int(np.argmax(kpis["agent_first_counts"])) if a_tot else None
    o_mode = int(np.argmax(kpis["opp_first_counts"])) if o_tot else None
    return {
        "a0_center_rate": _rate(kpis["agent_first_counts"], a_tot, 3),
        "o0_center_rate": _rate(kpis["opp_first_counts"],   o_tot, 3),
        "a0_mode": a_mode,
        "o0_mode": o_mode,
    }


def _plot_tu_axis(ax, tu_hist, tau_hist, mode_hist):
    """
    Plot target-update policy over episodes:
      • TU (hard) as a step line on left Y
      • tau (soft) on a twin right Y
      • shaded spans for soft vs hard phases
    """
    if not tu_hist or not tau_hist or not mode_hist:
        ax.set_visible(False)
        return

    import numpy as np
    x = np.arange(len(tu_hist))
    tu_vals = np.asarray(tu_hist, dtype=float)
    tau_vals = np.asarray(tau_hist, dtype=float)
    soft_mask = np.asarray(
        [m.lower() == "soft" for m in mode_hist], dtype=bool)

    # Show TU only when mode == hard
    tu_plot = tu_vals.copy()
    tu_plot[soft_mask] = np.nan
    ax.step(x, tu_plot, where="post", label="TU (hard, steps)", linewidth=1.5)
    ax.set_ylabel("TU (steps)")
    ax.grid(True, alpha=0.3)

    # Twin axis for tau, shown only when mode == soft
    ax2 = ax.twinx()
    tau_plot = tau_vals.copy()
    tau_plot[~soft_mask] = np.nan
    ax2.plot(x, tau_plot, linestyle="--", label="τ (soft)", alpha=0.9)
    ax2.set_ylabel("τ")

    # Shade spans for modes
    def _spans(mask: np.ndarray):
        spans, s = [], None
        for i, v in enumerate(mask):
            if v and s is None:
                s = i
            if (not v) and s is not None:
                spans.append((s, i))
                s = None
        if s is not None:
            spans.append((s, len(mask)))
        return spans

    for s, e in _spans(soft_mask):
        ax.axvspan(s, e, color="#7fd", alpha=0.10)   # soft
    for s, e in _spans(~soft_mask):
        ax.axvspan(s, e, color="#ddd", alpha=0.08)   # hard

    # Unified legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="upper right", fontsize=8)


def set_training_phases_length(DICT):
    length = 0
    for name, TF in DICT.items():

        length += int(TF["duration"])
        TF["length"] = length
        TF["sumWeights"] = float(np.sum(TF.get("weights", []))) if TF.get(
            "weights") is not None else 0.0
        opp = TF.get("opponent", [])
        TF["sumOppo"] = (float(np.sum(opp)) if isinstance(opp, (list, tuple, np.ndarray))
                         else (0.0 if opp is None else float(opp)))
    display_dict_as_tab(DICT)
    return length, name


def display_dict_as_tab(DICT):
    DF = pd.DataFrame.from_dict(DICT, orient="index")
    cols = [
        "length", "duration",
        "epsilon", "epsilon_min",
        "memory_prune_low",
        "sumWeights", "sumOppo",
        "TU_mode", "tau", "guard_prob", "center_start",
        "min_seed_fraction", "max_seed_fraction",
        "lr"
    ]

    DF = DF[cols]
    DF = DF.sort_values(by=['length'], ascending=True)
    display(HTML(DF.to_html()))


def _draw_openings_on_axes(ax_hist, ax_rate, openings, training_phases=None,
                           rate_ylim: tuple[float, float] | None = (0.50, 1.00)):
    if openings is None:
        ax_hist.set_visible(False)
        ax_rate.set_visible(False)
        return

    # --- Histogram ---
    ax_hist.cla()
    ax_hist.set_title("First-move histogram (agent vs opponent)")
    ax_hist.set_xlabel("Column")
    ax_hist.set_ylabel("Rate")
    ax_hist.set_xticks(range(openings.cols))
    ax_hist.set_xlim(-0.5, openings.cols - 0.5)
    ax_hist.axvline(openings.center, color="k", ls="--", alpha=0.25)

    a_tot = max(1, openings.kpis["agent_first_total"])
    o_tot = max(1, openings.kpis["opp_first_total"])
    a_rates = openings.kpis["agent_first_counts"] / a_tot
    o_rates = openings.kpis["opp_first_counts"] / o_tot
    xs = np.arange(openings.cols)
    w = 0.38
    ax_hist.bar(xs - w/2, a_rates, width=w, label="Agent",   alpha=0.9)
    ax_hist.bar(xs + w/2, o_rates, width=w, label="Opponent", alpha=0.6)
    ax_hist.legend(loc="upper left")
    ax_hist.set_ylim(0.0, max(0.01, float(
        max(a_rates.max(initial=0), o_rates.max(initial=0), 0.3))))

    # --- a0@center over time ---
    ax_rate.cla()
    ax_rate.set_title("a0@center over time")
    ax_rate.set_xlabel("Episode")
    ax_rate.set_ylabel("Rate")

    if rate_ylim is not None:
        ax_rate.set_ylim(*rate_ylim)       # <- clamp to [0.70, 1.00]
    else:
        ax_rate.set_ylim(0.0, 1.0)

    ax_rate.margins(x=0.02)
    ax_rate.grid(True, alpha=0.3)
    ax_rate.plot(openings.episodes, openings.a_center_rates,
                 lw=1.6, label="agent a0 center")
    ax_rate.plot(openings.episodes, openings.o_center_rates,
                 lw=1.0, alpha=0.8, label="opp a0 center")

    w = openings.ma_window
    if len(openings.a_center_rates) >= max(2, w):
        ma = np.convolve(openings.a_center_rates, np.ones(w)/w, mode="valid")
        ax_rate.plot(openings.episodes[-len(ma):], ma,
                     ls="--", lw=2.0, label=f"agent MA (w={w})")

    # Phase markers on the openings rate
    if training_phases:
        last_ep = openings.episodes[-1] if openings.episodes else None
        draw_phase_vlines(ax_rate, training_phases, up_to=last_ep, label=True)
    ax_rate.legend(loc="lower left", ncol=1, fontsize=8)


def _plot_action_mix_triplet(ax_explore, ax_guard, ax_epsdetail, agent, pad: float = 0.05):
    """
    Draw three bar charts from agent.act_stats:
      1) Explore vs Exploit
      2) Center / Guard / Win_now / None
      3) ε-branch detail: rand, L1..Lx (only keys present are plotted)
    """
    if agent is None or not hasattr(agent, "act_stats_summary"):
        for ax in (ax_explore, ax_guard, ax_epsdetail):
            ax.axis("off")
        return

    s = agent.act_stats_summary(normalize=True)  # fractions of total decisions

    # --- 1) Explore vs Exploit ---
    explore = float(s.get("epsilon", 0.0))
    exploit = float(s.get("exploit", 0.0))
    ax_explore.bar(["Exploit", "Explore (ε)"], [exploit, explore])
    ax_explore.set_ylim(0.0, 1.0)
    ax_explore.set_ylabel("Rate")
    ax_explore.set_title("Action Mix: Explore vs Exploit", fontsize=10)
    for i, v in enumerate([exploit, explore]):
        ax_explore.text(i, min(0.98, v + 0.02),
                        f"{100*v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax_explore.grid(True, axis="y", alpha=0.3)

    # --- 2) Center / Guard / Win_now / None ---
    center = float(s.get("center", 0.0))
    guard = float(s.get("guard",  0.0))
    win_now = float(s.get("win_now", 0.0))
    none = max(0.0, 1.0 - (center + guard + win_now))

    labels2 = ["Center", "Guard", "Win_now", "None"]
    vals2 = [center,   guard,   win_now,   none]
    ax_guard.bar(labels2, vals2)
    ax_guard.set_ylim(0.0, 1.0)
    ax_guard.set_ylabel("Rate")
    ax_guard.set_title(
        "Special Moves: Center / Guard / Win_now / None", fontsize=10)
    for i, v in enumerate(vals2):
        ax_guard.text(i, min(0.98, v + 0.02),
                      f"{100*v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax_guard.grid(True, axis="y", alpha=0.3)

    # --- 3) ε-branch detail: rand + Lk present in stats ---
    items = [("rand", float(s.get("rand", 0.0)))]
    items += [(k, float(v))
              for k, v in s.items() if k.startswith("L") and k[1:].isdigit()]
    if len(items) > 1:
        items = [items[0]] + sorted(items[1:], key=lambda kv: int(kv[0][1:]))

    labels3 = [k for k, _ in items]
    vals3 = [v for _, v in items]
    ax_epsdetail.bar(labels3, vals3)

    ymax = max(vals3) if vals3 else 0.0
    ax_epsdetail.set_ylim(0.0, max(0.05, min(1.0, ymax + pad)))
    ax_epsdetail.set_ylabel("Rate")
    ax_epsdetail.set_title("ε-Branch Breakdown", fontsize=10)
    for i, v in enumerate(vals3):
        ax_epsdetail.text(
            i, v + 0.02, f"{100*v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax_epsdetail.grid(True, axis="y", alpha=0.3)


def _plot_opp_epsdetail(ax, agent):
    if agent is None or not hasattr(agent, "opp_act_stats_summary"):
        ax.axis("off")
        return
    s = agent.opp_act_stats_summary(normalize=True)

    items = [("rand", s.get("rand", 0.0))] + sorted(
        [(k, v) for k, v in s.items() if k.startswith("L") and k[1:].isdigit()],
        key=lambda kv: int(kv[0][1:])
    )
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    ax.bar(labels, values)

    ymax = max(values) if values else 0.0
    ax.set_ylim(0, min(1.0, ymax + max(0.04, 0.10 * ymax)))
    ax.set_ylabel("Rate")
    ax.set_title("Opponent ε-Branch Breakdown", fontsize=10)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{100*v:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def _smooth_same(y: np.ndarray, k: int) -> np.ndarray:
    """
    Simple moving average with 'same' length output using edge-padding.
    k <= 1 → returns the original series.
    """
    k = int(k)
    if k <= 1 or y.size == 0:
        return y
    kernel = np.ones(k, dtype=float) / k
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    ypad = np.pad(y, (pad_left, pad_right), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


## ---- H2H --- ##

def h2h_append(hist: dict, episode: int, res: dict):
    lo, hi = res["A_score_CI95"]
    hist["episode"].append(int(episode))
    hist["score"].append(float(res["A_score_rate"]))
    hist["lo"].append(float(lo))
    hist["hi"].append(float(hi))
    hist["n"].append(int(res["games"]))


def _plot_h2h_axis(ax, history: dict, training_phases: dict | None = None, title: str = "H2H vs baseline (score ±95% CI)"):
    if not history or not history.get("episode"):
        ax.set_visible(False)
        return
    import numpy as np
    x = np.asarray(history["episode"], dtype=int)
    y = np.asarray(history["score"],  dtype=float)
    lo = np.asarray(history["lo"],     dtype=float)
    hi = np.asarray(history["hi"],     dtype=float)

    ax.plot(x, y, linewidth=1.8, label="A score vs baseline")
    ax.fill_between(x, lo, hi, alpha=0.20)
    ax.axhline(0.5, color="#888", ls="--", lw=1.2, label="even (0.5)")
    if y.size:
        ax.scatter([x[-1]], [y[-1]], s=22, zorder=3)
        ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), textcoords="offset points", xytext=(6, -6),
                    ha="left", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("H2H score")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    if training_phases:
        draw_phase_vlines(ax, training_phases, up_to=int(x.max()), label=True)


def _plot_h2h_dual_axis(
    ax,
    base_hist: dict | None,
    ens_hist: dict | None,
    training_phases: dict | None = None,
    title: str = "H2H (±95% CI), baseline + ensemble",
):
    import numpy as np

    base_ok = bool(base_hist and base_hist.get("episode"))
    ens_ok  = bool(ens_hist  and ens_hist.get("episode"))

    if not base_ok and not ens_ok:
        ax.set_visible(False)
        return

    ax.axhline(0.5, color="#888", ls="--", lw=1.2, label="even (0.5)")

    def _one(hist: dict, label: str):
        x  = np.asarray(hist["episode"], dtype=int)
        y  = np.asarray(hist["score"], dtype=float)
        lo = np.asarray(hist.get("lo", [np.nan] * len(x)), dtype=float)
        hi = np.asarray(hist.get("hi", [np.nan] * len(x)), dtype=float)

        line, = ax.plot(x, y, lw=1.8, label=label)
        col = line.get_color()

        # CI fill if present (and finite)
        if lo.size == y.size and hi.size == y.size and np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(x, lo, hi, alpha=0.15, color=col)

        if y.size:
            ax.scatter([x[-1]], [y[-1]], s=22, color=col, zorder=3)
            ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]),
                        textcoords="offset points", xytext=(6, -6),
                        ha="left", fontsize=8, color=col)

    if base_ok:
        _one(base_hist, "vs baseline")
    if ens_ok:
        _one(ens_hist, "vs ensemble")

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("H2H score")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    if training_phases:
        up_to = int(max(
            max(base_hist["episode"]) if base_ok else 0,
            max(ens_hist["episode"])  if ens_ok  else 0,
        ))
        draw_phase_vlines(ax, training_phases, up_to=up_to, label=True)
        
def _plot_global_score_axis(
    ax,
    bench_history: dict | None,
    training_phases: dict | None = None,
    title: str = "Depth-weighted global benchmark score",
):
    import numpy as np
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    if not bench_history or not bench_history.get("episode") or not bench_history.get("global_score"):
        ax.set_visible(False)
        return

    x = np.asarray(bench_history["episode"], dtype=float)
    y = np.asarray(bench_history["global_score"], dtype=float)

    ax.plot(x, y, label="Global score", alpha=0.85)

    def _ma(arr, k: int):
        k = int(k)
        if k <= 1 or arr.size < k:
            return None
        return np.convolve(arr, np.ones(k, dtype=float) / k, mode="valid")

    ma3 = _ma(y, 3)
    if ma3 is not None:
        ax.plot(x[2:], ma3, label="MA3", ls="--", lw=1.6)

    ma10 = _ma(y, 10)
    if ma10 is not None:
        ax.plot(x[9:], ma10, label="MA10", ls=":", lw=1.6)

    ax.set_ylabel("Global score")
    ax.set_ylim(0.0, 1.0)

    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.minorticks_on()

    ax.grid(True, which="major", alpha=0.45, lw=0.9)
    ax.grid(True, which="minor", alpha=0.25, lw=0.6)

    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", fontsize=8)

    if training_phases:
        draw_phase_vlines(ax, training_phases, up_to=int(x.max()), label=True)
        

def plot_is_weights(agent, bins: int = 60, last: int = 4096, return_figs: bool = True):
    """
    Histogram of importance-sampling weights.
    Uses agent.per_w_hist if available (preferred).
    Falls back to agent.last_replay_stats["is_w"] if present.
    Returns a matplotlib Figure or None.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    w = None

    # Preferred: a running history you maintain in the agent
    if getattr(agent, "per_w_hist", None):
        w = np.asarray(agent.per_w_hist, dtype=float)

    # Fallback: last replay batch weights (if you store them)
    elif getattr(agent, "last_replay_stats", None) is not None:
        st = agent.last_replay_stats
        if isinstance(st, dict) and "is_w" in st and st["is_w"] is not None:
            w = np.asarray(st["is_w"], dtype=float).reshape(-1)

    if w is None or w.size == 0:
        return None

    # keep it sane
    w = w[np.isfinite(w)]
    if w.size == 0:
        return None

    if last is not None and last > 0 and w.size > last:
        w = w[-last:]

    fig, ax = plt.subplots(figsize=(8.5, 3.5), dpi=120)
    ax.hist(w, bins=int(bins), alpha=0.85)
    ax.set_title("Importance-sampling weights (hist)")
    ax.set_xlabel("w")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)

    if return_figs:
        return fig
    return None



def _plot_opponent_usage_axis(
    ax,
    opponent_timeline: list[str],
    up_to_episode: int | None = None,
    normalize: bool = True,
):
    if not opponent_timeline:
        ax.set_visible(False)
        return

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

    def _opp_sort_key(lab: str):
        if lab == "R": return (0, 0)
        if lab == "POP": return (1, 0)
        if lab.startswith("L"):
            try:
                return (2, int(lab[1:]))
            except ValueError:
                return (3, lab)
        if lab == "SP": return (4, 0)
        return (3, lab)

    labels_sorted = sorted(counts.keys(), key=_opp_sort_key)

    base_colors = {
        "R": "#1f77b4",
        "POP": "#000000",
        "SP": "#9467bd",
    }

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
        color_map[lab] = palette[palette_idx % len(palette)]
        palette_idx += 1

    labels, vals, colors = [], [], []
    for lab in labels_sorted:
        c = counts.get(lab, 0)
        if c <= 0:
            continue
        v = (c / total) if normalize else float(c)
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
        ax.text(
            x, v,
            f"{v:.2f}" if normalize else f"{int(v)}",
            ha="center", va="bottom", fontsize=8,
        )

def _plot_a0_center_axis(
    ax,
    openings,
    bench_history: dict | None,
    training_phases: dict | None = None,
    rate_ylim: tuple[float, float] | None = None,
):
    """
    Robust a0@center panel:
      - Prefer OpeningTracker time series if it has points
      - Else fall back to bench_history['center_rate']
      - Never hides the axis; if no data, shows a placeholder
      - If rate_ylim clips all points, auto-expands
    """
    import numpy as np

    ax.cla()
    ax.set_title("a0@center over time", fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.grid(True, alpha=0.35)

    x = None
    y_agent = None
    y_opp = None
    source = None

    # 1) Prefer OpeningTracker if it has points
    if openings is not None:
        ep_list = getattr(openings, "episodes", None)
        ar_list = getattr(openings, "a_center_rates", None)
        if ep_list is not None and ar_list is not None and len(ep_list) > 0 and len(ar_list) > 0:
            x = np.asarray(ep_list, dtype=float)
            y_agent = np.asarray(ar_list, dtype=float)
            or_list = getattr(openings, "o_center_rates", None)
            if or_list is not None and len(or_list) == len(ar_list):
                y_opp = np.asarray(or_list, dtype=float)
            source = "openings"

    # 2) Fallback to benchmark history
    if source is None and bench_history is not None:
        if bench_history.get("episode") and bench_history.get("center_rate") is not None:
            if len(bench_history["episode"]) > 0 and len(bench_history["center_rate"]) > 0:
                x = np.asarray(bench_history["episode"], dtype=float)
                y_agent = np.asarray(bench_history["center_rate"], dtype=float)
                y_opp = None
                source = "bench"

    # 3) No data yet -> placeholder (do NOT hide axis)
    if source is None:
        ax.set_ylim(0.0, 1.0)
        ax.text(
            0.5, 0.5,
            "No a0@center data yet\nNeed openings.maybe_log(ep) or bench center_rate",
            transform=ax.transAxes,
            ha="center", va="center", fontsize=9, alpha=0.75
        )
        return

    # Plot main series
    label_main = "agent a0 center" if source == "openings" else "a0@center (bench)"
    ax.plot(x, y_agent, lw=1.6, label=label_main, alpha=0.90)

    if y_opp is not None and y_opp.size == y_agent.size:
        ax.plot(x, y_opp, lw=1.0, label="opp a0 center", alpha=0.70)

    # Moving averages for the agent line
    def _ma(arr, k: int):
        k = int(k)
        if k <= 1 or arr.size < k:
            return None
        return np.convolve(arr, np.ones(k, dtype=float) / k, mode="valid")

    ma3 = _ma(y_agent, 3)
    if ma3 is not None:
        ax.plot(x[2:], ma3, ls="--", lw=1.6, label="MA3")

    ma10 = _ma(y_agent, 10)
    if ma10 is not None:
        ax.plot(x[9:], ma10, ls=":", lw=1.6, label="MA10")

    # Y-lims: apply requested, but auto-expand if it would hide everything
    if rate_ylim is None:
        ax.set_ylim(0.0, 1.0)
    else:
        lo, hi = float(rate_ylim[0]), float(rate_ylim[1])

        yy = y_agent[np.isfinite(y_agent)]
        if yy.size > 0:
            ymin = float(yy.min())
            ymax = float(yy.max())
            if (ymax < lo) or (ymin > hi):
                pad = 0.05
                lo = min(lo, ymin - pad)
                hi = max(hi, ymax + pad)

        ax.set_ylim(lo, hi)

    if training_phases:
        try:
            draw_phase_vlines(ax, training_phases, up_to=int(np.nanmax(x)), label=True)
        except Exception:
            pass

    ax.legend(loc="lower left", fontsize=8)

    
### ----- final live plot ----- ###

def plot_live_training(
    episode, reward_history, win_history, epsilon_history, phase,
    win_count, loss_count, draw_count, title,
    epsilon_min_history, memory_prune_low_history,
    agent=None, save=False, path=None,
    bench_history=None, bench_smooth_k=3,
    tu_interval_history=None, tau_history=None, tu_mode_history=None,
    guard_prob_history=None, center_prob_history=None,
    openings=None,
    height_scale: float = 1.20,
    hspace: float = 0.48,
    dpi: int = 140,
    openings_ylim: tuple[float, float] | None = (-0.05, 1.05),
    h2h_history=None,
    pop_h2h_history=None,
    opponent_timeline=None,
    overlay_last: int = 100,
    return_figs=False,
    fig_width: float = 26,
    step_fig_width: float | None = None,
    full_width_css: bool = True,
):
    


    if step_fig_width is None:
        step_fig_width = fig_width

    # Optional notebook/JupyterLab display widening.
    # This does not change saved PNG size, only inline display behavior.
    if full_width_css:
        try:
            from IPython.display import display, HTML
            display(HTML("""
            <style>
                .jp-OutputArea-output img,
                .output_png img {
                    max-width: 100% !important;
                    height: auto !important;
                }
                .jp-OutputArea-output,
                .output_area {
                    width: 100% !important;
                }
            </style>
            """))
        except Exception:
            pass

    use_openings = openings is not None

    # --- layout ---
    if use_openings:
        nrows = 18
        base_heights = [
            3.8, 3.4, 2.6, 1.8, 2.6,
            10.0,
            3.2, 3.2,
            10.0,
            6.0,   # global
            6.0,   # h2h
            2.1, 2.1, 2.1, 2.1,  # action mix triplet + opp eps
            2.6,   # opponent usage
            2.6,   # openings hist
            6.0    # a0@center over time
        ]
    else:
        nrows = 16
        base_heights = [
            3.8, 3.4, 2.6, 1.8, 2.6,
            3.2, 3.2, 3.2,
            10.0,
            6.0,   # global
            2.6,   # h2h
            2.1, 2.1, 2.1, 2.1,  # action mix triplet + opp eps
            2.6    # opponent usage
        ]

    heights = [h * height_scale for h in base_heights]

    fig_ep = plt.figure(figsize=(fig_width, sum(heights) + 1.5), dpi=dpi)
    gs = GridSpec(nrows, 1, height_ratios=heights, hspace=hspace)

    # --- axes ---
    ax_reward = fig_ep.add_subplot(gs[0, 0])
    ax_win    = fig_ep.add_subplot(gs[1, 0], sharex=ax_reward)
    ax_eps    = fig_ep.add_subplot(gs[2, 0], sharex=ax_reward)
    ax_mem    = fig_ep.add_subplot(gs[3, 0], sharex=ax_reward)
    ax_tu     = fig_ep.add_subplot(gs[4, 0], sharex=ax_reward)

    ax_bench      = fig_ep.add_subplot(gs[5, 0], sharex=ax_reward)
    ax_bench_s35  = fig_ep.add_subplot(gs[6, 0], sharex=ax_reward)
    ax_bench_s57  = fig_ep.add_subplot(gs[7, 0], sharex=ax_reward)
    ax_bench_s15  = fig_ep.add_subplot(gs[8, 0], sharex=ax_reward)

    ax_global = fig_ep.add_subplot(gs[9, 0], sharex=ax_reward)
    ax_h2h    = fig_ep.add_subplot(gs[10, 0], sharex=ax_reward)

    # Action-mix triplet + opponent eps breakdown
    ax_mix_explore = fig_ep.add_subplot(gs[11, 0])
    ax_mix_guard   = fig_ep.add_subplot(gs[12, 0])
    ax_mix_eps     = fig_ep.add_subplot(gs[13, 0])
    ax_opp_eps     = fig_ep.add_subplot(gs[14, 0])

    # Opponent usage
    ax_opp_usage = fig_ep.add_subplot(gs[15, 0])

    # Openings
    ax_hist = None
    ax_rate = None
    if use_openings:
        ax_hist = fig_ep.add_subplot(gs[16, 0])
        ax_rate = fig_ep.add_subplot(gs[17, 0], sharex=ax_reward)

    # ---------------- helpers ----------------
    def _opp_color(lab: str) -> str:
        lab = str(lab).upper()

        if lab in ("R", "RANDOM"):
            return "#1f77b4"
        if lab == "POP":
            return "#000000"
        if lab == "SP":
            return "#9467bd"
        if lab in ("LEFT", "LEFTMOST"):
            return "#8c564b"
        if lab in ("C", "CENTER"):
            return "#2ca02c"

        if lab.startswith("LOOKAHEAD-"):
            lab = "L" + lab.split("-", 1)[1]
        elif lab.startswith("LA-"):
            lab = "L" + lab.split("-", 1)[1]
        elif lab.startswith("LA") and lab[2:].isdigit():
            lab = "L" + lab[2:]

        if lab.startswith("L") and lab[1:].isdigit():
            depth = int(lab[1:])
            palette = [
                "#ff7f0e", "#2ca02c", "#d62728", "#8c564b",
                "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
            ]
            return palette[depth % len(palette)]

        return "#888888"

    def _moving_avg(arr, k):
        if k <= 1 or arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        if arr.size < k:
            return None
        w = np.ones(k, dtype=float) / k
        return np.convolve(arr, w, mode="valid")

    def _plot_center_from_bench(ax, bench):
        if not bench or not bench.get("episode") or not bench.get("center_rate"):
            ax.set_visible(False)
            return

        x = np.asarray(bench["episode"], dtype=float)
        y = np.asarray(bench["center_rate"], dtype=float)

        ax.plot(x, y, label="a0@center (bench)", alpha=0.85)

        ma3 = _moving_avg(y, 3)
        if ma3 is not None and x.size >= 3:
            ax.plot(x[2:2 + len(ma3)], ma3, label="MA3", ls="--", lw=1.6)

        ma10 = _moving_avg(y, 10)
        if ma10 is not None and x.size >= 10:
            ax.plot(x[9:9 + len(ma10)], ma10, label="MA10", ls=":", lw=1.6)

        ax.set_title("a0@center over time", fontsize=10)
        ax.set_ylabel("Rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="lower left", fontsize=8)

    # ---------------- Reward ----------------
    plot_moving_averages(
        ax_reward, reward_history, [10, 25, 100, 500],
        colors=["red", "blue", "#666", "#000"],
        styles=["-", "-", "--", "--"],
        labels=["10-ep", "25-ep", "100-ep", "500-ep"]
    )
    ax_reward.plot(reward_history, label="Reward", alpha=0.45)
    ax_reward.set_ylabel("Reward")
    ax_reward.legend(loc="lower left", fontsize=8)
    ax_reward.grid(True, alpha=0.35)

    # opponent tick overlay on reward
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
                zorder=4
            )

    # ---------------- Win rate + epsilon overlay ----------------
    plot_moving_averages(
        ax_win, win_history, [25, 100, 250, 500],
        colors=["green", "red", "#11F", "#000"],
        styles=["-", "-", "--", "--"],
        labels=["25-ep", "100-ep", "250-ep", "500-ep"]
    )
    ax_win.plot(epsilon_history, label="Epsilon", color="orange", linestyle="--", alpha=0.8)
    ax_win.set_ylabel("Win Rate")
    ax_win.legend(loc="lower left", fontsize=8)
    ax_win.grid(True, alpha=0.35)

    # ---------------- Epsilon / guards ----------------
    ax_eps.plot(epsilon_history, label="Epsilon", color="orange")
    ax_eps.plot(epsilon_min_history, label="Epsilon_min", color="black", linestyle="--")

    if guard_prob_history:
        ax_eps.plot(guard_prob_history, label="Guard Prob", color="green", linestyle=":")

    if center_prob_history:
        ax_eps.plot(center_prob_history, label="Center Prob", color="red", linestyle=":")

    ax_eps.set_ylabel("Epsilon / Guard")
    ax_eps.legend(loc="lower left", fontsize=8)
    ax_eps.grid(True, alpha=0.35)

    # ---------------- Memory prune ----------------
    ax_mem.plot(memory_prune_low_history, label="Memory Prune", color="red")
    ax_mem.set_ylabel("Mem Prune")
    ax_mem.legend(loc="lower left", fontsize=8)
    ax_mem.grid(True, alpha=0.35)

    # ---------------- Target update ----------------
    _plot_tu_axis(ax_tu, tu_interval_history or [], tau_history or [], tu_mode_history or [])
    ax_tu.set_ylabel("Target Update")
    ax_tu.grid(True, alpha=0.35)

    # ---------------- Benchmarks ----------------
    _plot_bench_on_axis(
        ax_bench,
        bench_history,
        smooth_k=0,
        training_phases=None,
        epsilon_history=epsilon_history,
        epsilon_min_history=epsilon_min_history
    )
    ax_bench.set_xlabel("Episode")
    ax_bench.set_title("Benchmarks (raw)", fontsize=10)

    _plot_bench_on_axis(
        ax_bench_s35,
        bench_history,
        smooth_k=3,
        training_phases=None,
        epsilon_history=epsilon_history,
        epsilon_min_history=epsilon_min_history
    )
    ax_bench_s35.set_xlabel("Episode")
    ax_bench_s35.set_title("Benchmarks (MA 3)", fontsize=10)

    _plot_bench_on_axis(
        ax_bench_s57,
        bench_history,
        smooth_k=7,
        training_phases=None,
        epsilon_history=epsilon_history,
        epsilon_min_history=epsilon_min_history
    )
    ax_bench_s57.set_xlabel("Episode")
    ax_bench_s57.set_title("Benchmarks (MA 7)", fontsize=10)

    _plot_bench_on_axis(
        ax_bench_s15,
        bench_history,
        smooth_k=15,
        training_phases=None,
        epsilon_history=epsilon_history,
        epsilon_min_history=epsilon_min_history
    )
    ax_bench_s15.set_xlabel("Episode")
    ax_bench_s15.set_title("Benchmarks (MA 15)", fontsize=10)

    # ---------------- Global score ----------------
    _plot_global_score_axis(
        ax_global,
        bench_history,
        training_phases=TRAINING_PHASES
    )

    # ---------------- H2H ----------------
    _plot_h2h_dual_axis(
        ax_h2h,
        h2h_history or {},
        pop_h2h_history or {},
        training_phases=TRAINING_PHASES,
    )

    # ---------------- Action mix + opponent eps breakdown ----------------
    _plot_action_mix_triplet(ax_mix_explore, ax_mix_guard, ax_mix_eps, agent)
    _plot_opp_epsdetail(ax_opp_eps, agent)

    # ---------------- Opponent usage ----------------
    _plot_opponent_usage_axis(
        ax_opp_usage,
        opponent_timeline or [],
        up_to_episode=episode,
        normalize=True
    )

    # ---------------- Openings ----------------
    if use_openings and ax_hist is not None and ax_rate is not None:
        _draw_openings_on_axes(
            ax_hist,
            ax_rate,
            openings,
            training_phases=TRAINING_PHASES,
            rate_ylim=openings_ylim
        )

        has_any_data = False

        for ln in list(ax_rate.lines):
            if len(ln.get_xdata()) > 0 and len(ln.get_ydata()) > 0:
                has_any_data = True
                break

        if not has_any_data and getattr(ax_rate, "collections", None):
            for col in ax_rate.collections:
                try:
                    if len(col.get_offsets()) > 0:
                        has_any_data = True
                        break
                except Exception:
                    pass

        if not has_any_data:
            ax_rate.cla()
            _plot_center_from_bench(ax_rate, bench_history)

    # ---------------- Phase vlines ----------------
    axes = [
        ax_reward,
        ax_win,
        ax_eps,
        ax_mem,
        ax_tu,
        ax_bench,
        ax_bench_s35,
        ax_bench_s57,
        ax_bench_s15,
        ax_global,
        ax_h2h,
    ]

    if use_openings and ax_rate is not None and ax_rate.get_visible():
        axes.append(ax_rate)

    for ax in axes:
        draw_phase_vlines(ax, TRAINING_PHASES, up_to=episode, label=True)

    eps_last = float(epsilon_history[-1]) if epsilon_history else float("nan")

    fig_ep.suptitle(
        f"{title} - Ep {episode} | Phase: {phase} | Wins: {win_count}, "
        f"Losses: {loss_count}, Draws: {draw_count} | eps={eps_last:.3f}",
        y=0.995
    )

    fig_ep.subplots_adjust(
        left=0.045,
        right=0.985,
        top=0.985,
        bottom=0.015
    )

    # ---------- step-based metrics ----------
    fig_step = None

    if agent:
        plots = []

        if getattr(agent, "loss_hist", None):
            plots.append(("Loss",          agent.loss_hist,        "purple"))
        if getattr(agent, "td_hist", None):
            plots.append(("TD Error",      agent.td_hist,          "gray"))
        if getattr(agent, "per_w_hist", None):
            plots.append(("IS Weights",    agent.per_w_hist,       "brown"))
        if getattr(agent, "q_abs_hist", None):
            plots.append(("Q |mean|",      agent.q_abs_hist,       "teal"))
        if getattr(agent, "q_max_hist", None):
            plots.append(("Q |max|",       agent.q_max_hist,       "navy"))
        if getattr(agent, "target_abs_hist", None):
            plots.append(("|target| mean", agent.target_abs_hist,  "darkgreen"))
        if getattr(agent, "target_lag_hist", None):
            plots.append(("Target Lag",    agent.target_lag_hist,  "orange"))

        if plots:
            fig_step, ax_step = plt.subplots(
                len(plots),
                1,
                figsize=(step_fig_width, 3.0 * len(plots)),
                dpi=dpi
            )

            if len(plots) == 1:
                ax_step = [ax_step]

            for i, (label, data, color) in enumerate(plots):
                ax = ax_step[i]
                ax.plot(data, label=label, color=color, alpha=0.6)

                if label == "Loss" and len(data) >= 100:
                    ma = np.convolve(
                        np.asarray(data, dtype=float),
                        np.ones(100) / 100,
                        mode="valid"
                    )
                    ax.plot(
                        range(99, len(data)),
                        ma,
                        label="100-ep MA",
                        color="black",
                        linestyle="--"
                    )

                ax.set_ylabel(label)
                ax.legend(loc="lower left", fontsize=8)
                ax.grid(True, alpha=0.35)

            fig_step.suptitle("Step-based Metrics", y=0.98)
            fig_step.subplots_adjust(
                left=0.045,
                right=0.985,
                top=0.955,
                bottom=0.04,
                hspace=0.38
            )

    if save:
        out_png = os.path.join(path, f"{title}_training_report.png")
        fig_ep.savefig(out_png, dpi=dpi, bbox_inches="tight")

        if fig_step is not None:
            out_step_png = os.path.join(path, f"{title}_step_metrics.png")
            fig_step.savefig(out_step_png, dpi=dpi, bbox_inches="tight")

        if not return_figs and fig_step is not None:
            plt.close(fig_step)

    if return_figs:
        return fig_ep, fig_step