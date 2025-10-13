# dqn_utilities.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from copy import deepcopy
from DQN.training_phases_config import TRAINING_PHASES
from C4.connect4_env import Connect4Env
from DQN.eval_utilities import evaluate_agent_model
import os
from IPython.display import display, HTML
import torch


TAU_DEFAULT = 0.006                         # phase controlled
GUARD_DEFAULT = 0.30                        # phase controlled
CENTER_START = 0.30                         # phase controlled

# benchmarks
EVAL_CFG = {
    "Random": 101, 
    "Lookahead-1": 73,
    "Lookahead-2": 51,
    #"Lookahead-3": 13,
    #"Lookahead-4": 2,
    } 


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
    w  = _flatten_params(agent.model)
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
    # choose a y for labels after plots set their y-limits
    ymin, ymax = ax.get_ylim()
    ytxt = (ymax * 0.95) if label_y is None else label_y
    for name, ep in bounds:
        ax.axvline(ep, color=color, linestyle=ls, linewidth=lw, alpha=alpha)
        if label:
            ax.text(ep, ytxt, name, rotation=90, ha=ha, va=va, fontsize=fontsize, alpha=alpha)


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
               2, 1, figsize=(7.5, 6.5), constrained_layout=True
           )
           self.fig.canvas.manager.set_window_title("Opening diagnostics")
           self._init_hist_axes()
           self._init_rate_axes()

       self.phase_marks = []

    # --- unchanged public API ---
    def on_first_move(self, col:int, is_agent:bool):
        record_first_move(self.kpis, col, "agent" if is_agent else "opp")
    
    def maybe_log(self, episode_idx:int):
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
        self.ax_hist.axvline(self.center, color="k", linestyle="--", alpha=0.25)

    def _init_rate_axes(self):
        self.ax_rate.set_title("Agent center rate over time")
        self.ax_rate.set_xlabel("Episode")
        self.ax_rate.set_ylabel("Rate")
        self.ax_rate.grid(True, alpha=0.3)
        self.ax_rate.set_ylim(0.0, 1.0)
        self.a_line, = self.ax_rate.plot([], [], label="agent a0@center", lw=1.5)
        self.o_line, = self.ax_rate.plot([], [], label="opp a0@center", lw=1.0, alpha=0.6)
        self.a_ma_line, = self.ax_rate.plot([], [], linestyle="--", lw=2.0,
                                            label=f"agent MA (w={self.ma_window})")
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
        self.ax_hist.bar(xs + w/2, o_rates, width=w, label="Opponent", alpha=0.6)
        self.ax_hist.legend(loc="upper right")
        self.ax_hist.set_ylim(0, max(0.01, float(max(a_rates.max(), o_rates.max(), 0.3))))

        # --- time series ---
        self.a_line.set_data(self.episodes, self.a_center_rates)
        self.o_line.set_data(self.episodes, self.o_center_rates)
        x_ma, y_ma = self._moving_avg(self.a_center_rates)
        self.a_ma_line.set_data(x_ma, y_ma)

        # phase markers (optional)
        for ep, lab in self.phase_marks:
            self.ax_rate.axvline(ep, color="k", linestyle=":", alpha=0.2)
            self.ax_rate.text(ep, 0.02, lab, rotation=90, va="bottom", ha="right", fontsize=8, alpha=0.6)

        # autoscale x only
        self.ax_rate.set_xlim(0 if not self.episodes else self.episodes[0],
                              max(1, self.episodes[-1]))
        self.ax_rate.figure.canvas.draw_idle()
        self.ax_rate.figure.canvas.flush_events()

    # optional helper if you want vertical lines at phase changes
    def mark_phase(self, episode_idx:int, label:str):
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
            )


def handle_phase_change(agent, new_phase, current_phase, epsilon, memory_prune_low, epsilon_min, prev_frozen_opp, env, episode, device, Lookahead, training_session, guard_prob, center_start):
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
        agent.epsilon = epsilon
        agent.epsilon_min = epsilon_min
        agent.guard_prob = guard_prob
        agent.memory.prune(memory_prune_low, mode="low_priority")
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

        raise ValueError("Invalid final_result — env.winner was not set correctly. Lovro, get a grip!")


def plot_moving_averages(ax, data, windows, colors=None, styles=None, labels=None):
    for i, win in enumerate(windows):
        if len(data) >= win:
            avg = np.convolve(data, np.ones(win) / win, mode='valid')
            offset = win - 1
            ax.plot(range(offset, len(data)), avg,
                    label=labels[i] if labels else f'{win}-ep Avg',
                    color=colors[i] if colors else None,
                    linestyle=styles[i] if styles else None)



def plot_live_training(
    episode, reward_history, win_history, epsilon_history, phase,
    win_count, loss_count, draw_count, title,
    epsilon_min_history, memory_prune_low_history,
    agent=None, save=False, path=None,
    bench_history=None, bench_smooth_k=3,
    tu_interval_history=None, tau_history=None, tu_mode_history=None,
    guard_prob_history=None, center_prob_history=None,
    openings=None,
    height_scale: float = 1.20,             # scale everything taller
    hspace: float = 0.48,                   # more vertical room
    dpi: int = 120,
    openings_ylim: tuple[float, float] | None = (0.70, 1.00),  # clamp a0 center plot
    return_figs=False
):
    use_openings = openings is not None
    nrows = 8 if use_openings else 6

    base_heights = [3.8, 3.4, 2.6, 1.8, 2.6, 3.2] + ([2.6, 3.0] if use_openings else [])
    heights = [h * height_scale for h in base_heights]

    fig_ep = plt.figure(figsize=(12.5, sum(heights) + 1.5), dpi=dpi)
    gs = GridSpec(nrows, 1, height_ratios=heights, hspace=hspace)

    ax_reward = fig_ep.add_subplot(gs[0, 0])
    ax_win    = fig_ep.add_subplot(gs[1, 0], sharex=ax_reward)
    ax_eps    = fig_ep.add_subplot(gs[2, 0], sharex=ax_reward)
    ax_mem    = fig_ep.add_subplot(gs[3, 0], sharex=ax_reward)
    ax_tu     = fig_ep.add_subplot(gs[4, 0], sharex=ax_reward)
    ax_bench  = fig_ep.add_subplot(gs[5, 0], sharex=ax_reward)

    # Reward
    plot_moving_averages(ax_reward, reward_history, [10, 25, 100, 500],
                         colors=['red','blue', '#666', '#000'], styles=['-','-', '--', '--'],
                         labels=['10-ep','25-ep', '100-ep', '500-ep'])
    ax_reward.plot(reward_history, label='Reward', alpha=0.45)
    ax_reward.set_ylabel('Reward'); ax_reward.legend(loc='lower left', fontsize=8); ax_reward.grid(True, alpha=0.35)

    # Win rate + epsilon
    plot_moving_averages(ax_win, win_history, [25, 100, 250, 500],
                         colors=['green', 'red', '#11F', '#000'], styles=['-', '-', '--', '--'],
                         labels=['25-ep', '100-ep', '250-ep', '500-ep'])
    ax_win.plot(epsilon_history, label='Epsilon', color='orange', linestyle='--', alpha=0.8)
    ax_win.set_ylabel('Win Rate'); ax_win.legend(loc='lower left', fontsize=8); ax_win.grid(True, alpha=0.35)

    # Epsilon / guards
    ax_eps.plot(epsilon_history, label='Epsilon', color='orange')
    ax_eps.plot(epsilon_min_history, label='Epsilon_min', color='black', linestyle='--')
    if guard_prob_history:  ax_eps.plot(guard_prob_history,  label='Guard Prob',  color='green', linestyle=':')
    if center_prob_history: ax_eps.plot(center_prob_history, label='Center Prob', color='red',   linestyle=':')
    ax_eps.set_ylabel('Epsilon / Guard'); ax_eps.legend(loc='lower left', fontsize=8); ax_eps.grid(True, alpha=0.35)

    # Memory prune
    ax_mem.plot(memory_prune_low_history, label='Memory Prune', color='red')
    ax_mem.set_ylabel('Mem Prune'); ax_mem.legend(loc='lower left', fontsize=8); ax_mem.grid(True, alpha=0.35)

    # Target update
    _plot_tu_axis(ax_tu, tu_interval_history or [], tau_history or [], tu_mode_history or [])
    ax_tu.set_ylabel("Target Update"); ax_tu.grid(True, alpha=0.35)

    # Benchmarks
    _plot_bench_on_axis(ax_bench, bench_history, smooth_k=bench_smooth_k,
                        training_phases=None,
                        epsilon_history=epsilon_history,
                        epsilon_min_history=epsilon_min_history)
    ax_bench.set_xlabel("Episode")

    # Openings (hist + rate)
    if use_openings:
        ax_hist = fig_ep.add_subplot(gs[6, 0])                      # histogram has its own x
        ax_rate = fig_ep.add_subplot(gs[7, 0], sharex=ax_reward)    # time series shares episode axis
        
        _draw_openings_on_axes(ax_hist, ax_rate, openings,
                               training_phases=TRAINING_PHASES,
                               rate_ylim=openings_ylim)


    # Phase markers everywhere (bench + openings included)
    axes = [ax_reward, ax_win, ax_eps, ax_mem, ax_tu, ax_bench]
    if use_openings:
        axes += [ax_rate]  
    for ax in axes:
        draw_phase_vlines(ax, TRAINING_PHASES, up_to=episode, label=True)


    fig_ep.suptitle(
        f"{title} - Ep {episode} — Phase: {phase} | Wins: {win_count}, "
        f"Losses: {loss_count}, Draws: {draw_count} | ε={epsilon_history[-1]:.3f}",
        y=0.995
    )
    #plt.tight_layout(rect=[0, 0, 1, 0.965])


    # ---------- step-based metrics (unchanged) ----------
    if agent:
        plots = []
        if getattr(agent, "loss_hist", None):        plots.append(("Loss",          agent.loss_hist,        "purple"))
        if getattr(agent, "td_hist", None):          plots.append(("TD Error",      agent.td_hist,          "gray"))
        if getattr(agent, "per_w_hist", None):       plots.append(("IS Weights",    agent.per_w_hist,       "brown"))
        if getattr(agent, "q_abs_hist", None):       plots.append(("Q |mean|",      agent.q_abs_hist,       "teal"))
        if getattr(agent, "q_max_hist", None):       plots.append(("Q |max|",       agent.q_max_hist,       "navy"))
        if getattr(agent, "target_abs_hist", None):  plots.append(("|target| mean", agent.target_abs_hist,  "darkgreen"))
        if getattr(agent, "target_lag_hist", None):  plots.append(("Target Lag",    agent.target_lag_hist,  "orange"))

        if plots:
            fig_step, ax_step = plt.subplots(len(plots), 1, figsize=(12, 3 * len(plots)))
            if len(plots) == 1: ax_step = [ax_step]
            for i, (label, data, color) in enumerate(plots):
                ax = ax_step[i]; ax.plot(data, label=label, color=color, alpha=0.6)
                if label == "Loss" and len(data) >= 100:
                    ma = np.convolve(data, np.ones(100)/100, mode='valid')
                    ax.plot(range(99, len(data)), ma, label='100-ep MA', color='black', linestyle='--')
                ax.set_ylabel(label) 
                ax.legend() 
                ax.grid(True)
            fig_step.suptitle("Step-based Metrics", y=0.98) 
            #plt.tight_layout(rect=[0,0,1,0.95]) 


    if return_figs:
        return fig_ep, (fig_step if agent else None)

# === FINAL SUMMARY PLOT ===
def save_final_winrate_plot(win_history, training_phases, save_path, session_name="default_session"):
    if len(win_history) < 25:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    win_avg = np.convolve(win_history, np.ones(25)/25, mode='valid')
    ax.plot(range(24, len(win_history)), win_avg, label='Win Rate (25 ep)', color='green')

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
    recent_rewards = reward_history[-25:] if len(reward_history) >= 25 else reward_history
    recent_win_rate = np.mean(win_history[-25:]) if len(win_history) >= 25 else np.mean(win_history)

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
def update_benchmark_winrates(
    agent,
    env,
    device,
    Lookahead,
    episode: int,
    history: dict | None = None,
    save: str | None = None,
    opponents_cfg: dict = EVAL_CFG,
):

    if history is None:
        history = {"episode": [], "by_opponent": {k: [] for k in opponents_cfg}}

    is_dqn = hasattr(agent, "model") and hasattr(agent, "target_model")
    if is_dqn:
        results = evaluate_agent_model(agent, env, opponents_cfg, device, Lookahead)
    else:
        # Lazy import to avoid circular deps
        from PPO.ppo_eval import evaluate_actor_critic_model
        results = evaluate_actor_critic_model(
            agent=agent,
            env=env,
            evaluation_opponents=opponents_cfg,
            Lookahead=Lookahead,
            temperature=0.0,  # greedy eval
        )

    # Append
    history["episode"].append(episode)
    for label in opponents_cfg:
        history["by_opponent"].setdefault(label, [])
        history["by_opponent"][label].append(results[label]["win_rate"])

    # persistent log (one row per benchmark)
    if save is not None:
        row = {"episode": episode}
        for label in opponents_cfg:
            row[f"{label}_wr"] = results[label]["win_rate"]
        df = pd.DataFrame([row])
        if os.path.exists(save):
            try:
                existing = pd.read_excel(save)  # default sheet
                out = pd.concat([existing, df], ignore_index=True)
            except Exception:
                out = df
            out.to_excel(save, index=False)
        else:
            # Create new workbook
            df.to_excel(save, index=False)

    return history

def _plot_bench_on_axis(
    ax,
    history: dict,
    smooth_k: int = 0,
    training_phases: dict | None = None,
    epsilon_history=None,
    epsilon_min_history=None,
):
    """Draw benchmark win-rates as connected points and overlay epsilon traces."""
    if not history or not history.get("episode"):
        ax.set_visible(False)
        return

    x_all = np.asarray(history["episode"], dtype=int)

    # --- Bench points/lines ---
    for label, ys in history.get("by_opponent", {}).items():
        y = np.asarray(ys, dtype=float)
        if y.size == 0:
            continue

        # Align x to the last len(y) episodes in case lengths differ
        x = x_all[-y.size:]

        # Connect each point to the next, and show the points
        ax.plot(x, y, linewidth=1.2, label=label, solid_capstyle="round")
        ax.scatter(x, y, s=14, alpha=0.85)

        # Annotate last value
        ax.annotate(
            f"{y[-1]:.2f}",
            (x[-1], y[-1]),
            textcoords="offset points",
            xytext=(6, -6),
            ha="left",
            fontsize=8,
        )

    # --- Overlay epsilon on the same axis (scaled 0–1) ---
    # Use full episode index for epsilon curves
    if epsilon_history is not None and len(epsilon_history) > 0:
        x_eps = np.arange(len(epsilon_history), dtype=int)
        ax.plot(
            x_eps, epsilon_history,
            linewidth=1.2, color="#bbbbbb",
            label="ε"
        )

    if epsilon_min_history is not None and len(epsilon_min_history) > 0:
        x_eps_min = np.arange(len(epsilon_min_history), dtype=int)
        ax.plot(
            x_eps_min, epsilon_min_history,
            linewidth=1.2, color="#bbbbbb",
            linestyle=":",
            label="ε_min"
        )


    # Optional phase markers (use axis limits to place labels nicely)
    if training_phases: draw_phase_vlines(ax, training_phases, up_to=int(x_all.max()), label=True)

    ax.set_ylabel("Bench WR")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
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
    o_mode = int(np.argmax(kpis["opp_first_counts"]))   if o_tot else None
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
    soft_mask = np.asarray([m.lower() == "soft" for m in mode_hist], dtype=bool)

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
            if v and s is None: s = i
            if (not v) and s is not None:
                spans.append((s, i)); s = None
        if s is not None: spans.append((s, len(mask)))
        return spans

    for s, e in _spans(soft_mask):
        ax.axvspan(s, e, color="#7fd", alpha=0.10)   # soft
    for s, e in _spans(~soft_mask):
        ax.axvspan(s, e, color="#ddd", alpha=0.08)   # hard

    # Unified legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

def set_training_phases_length(DICT):
    length = 0
    for name, TF in DICT.items():

        length += int(TF["duration"])
        TF["length"] = length
        TF["sumWeights"] = float(np.sum(TF.get("weights", []))) if TF.get("weights") is not None else 0.0
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
        "TU", "TU_mode", "tau", "guard_prob", "center_start",
    ]
    DF = DF[cols]
    DF = DF.sort_values(by=['length'], ascending=True)
    display(HTML(DF.to_html()))
    
def _draw_openings_on_axes(ax_hist, ax_rate, openings, training_phases=None,
                           rate_ylim: tuple[float, float] | None = (0.50, 1.00)):
    if openings is None:
        ax_hist.set_visible(False); ax_rate.set_visible(False); return

    # --- Histogram ---
    ax_hist.cla()
    ax_hist.set_title("First-move histogram (agent vs opponent)")
    ax_hist.set_xlabel("Column"); ax_hist.set_ylabel("Rate")
    ax_hist.set_xticks(range(openings.cols)); ax_hist.set_xlim(-0.5, openings.cols - 0.5)
    ax_hist.axvline(openings.center, color="k", ls="--", alpha=0.25)

    a_tot = max(1, openings.kpis["agent_first_total"])
    o_tot = max(1, openings.kpis["opp_first_total"])
    a_rates = openings.kpis["agent_first_counts"] / a_tot
    o_rates = openings.kpis["opp_first_counts"]  / o_tot
    xs = np.arange(openings.cols); w = 0.38
    ax_hist.bar(xs - w/2, a_rates, width=w, label="Agent",   alpha=0.9)
    ax_hist.bar(xs + w/2, o_rates, width=w, label="Opponent", alpha=0.6)
    ax_hist.legend(loc="upper left")
    ax_hist.set_ylim(0.0, max(0.01, float(max(a_rates.max(initial=0), o_rates.max(initial=0), 0.3))))

    # --- a0@center over time ---
    ax_rate.cla()
    ax_rate.set_title("a0@center over time")
    ax_rate.set_xlabel("Episode"); ax_rate.set_ylabel("Rate")
    if rate_ylim is not None:
        ax_rate.set_ylim(*rate_ylim)       # <- clamp to [0.70, 1.00]
    else:
        ax_rate.set_ylim(0.0, 1.0)
    ax_rate.margins(x=0.02)
    
    ax_rate.grid(True, alpha=0.3)

    ax_rate.plot(openings.episodes, openings.a_center_rates, lw=1.6, label="agent a0 center")
    ax_rate.plot(openings.episodes, openings.o_center_rates, lw=1.0, alpha=0.8, label="opp a0 center")

    w = openings.ma_window
    if len(openings.a_center_rates) >= max(2, w):
        ma = np.convolve(openings.a_center_rates, np.ones(w)/w, mode="valid")
        ax_rate.plot(openings.episodes[-len(ma):], ma, ls="--", lw=2.0, label=f"agent MA (w={w})")

    # Phase markers on the openings rate
    if training_phases:
        last_ep = openings.episodes[-1] if openings.episodes else None
        draw_phase_vlines(ax_rate, training_phases, up_to=last_ep, label=True)
    ax_rate.legend(loc="lower left", ncol=1, fontsize=8)
