# A0/a0_plot.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

class LiveDiagnostics:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
        self._configure_axes()

        # cycle-level train history
        self.hist = {
            "cycle": [],
            "tr_loss": [], "tr_policy": [], "tr_value": [],
            "ent_model": [], "ent_target": [], "top1_match": [],
        }

        # eval history (own x-axis so points show only when they exist)
        self.eval_cycles: list[int] = []
        self.eval_wr_random: list[float] = []
        self.eval_wr_l1: list[float] = []

        # step-level (reset each cycle)
        self._step = {"loss": [], "policy": [], "value": [],
                      "ent_model": [], "ent_target": [], "top1": []}
        self._lines = {
            "step_loss": None, "step_policy": None, "step_value": None,
            "step_ent_m": None, "step_ent_t": None, "step_top1": None
        }

        # persistent display (no clear_output: tqdm stays)
        self._disp = display.display(self.fig, display_id=True)

    def _configure_axes(self):
        self.ax[0].set_title("Train Loss (live per-step)")
        self.ax[0].set_ylabel("loss")
        self.ax[1].set_title("Policy Diagnostics (live per-step)")
        self.ax[1].set_ylabel("entropy / acc")
        self.ax[2].set_title("Eval Win Rates (per-cycle)")
        self.ax[2].set_ylabel("win rate")
        self.ax[2].set_ylim(0, 1)
        self.ax[2].set_xlabel("cycle")

    # ---------- lifecycle ----------
    def start_cycle(self, cycle_idx: int):
        for k in self._step: self._step[k].clear()
        for k in self._lines: self._lines[k] = None
        self.fig.suptitle(f"Cycle {cycle_idx} • training starting...", fontsize=11)
        self.refresh()

    # ---------- self-play (title only) ----------
    def update_selfplay(self, cycle_idx: int, games_done: int, wins: int, losses: int, draws: int, avg_moves: float):
        self.fig.suptitle(
            f"Cycle {cycle_idx} • Self-play {games_done} | W {wins} / L {losses} / D {draws} • avg {avg_moves:.1f}",
            fontsize=11
        )

    # ---------- per-train-step ----------
    def update_step(self, cycle_idx: int, step_idx: int, stats: dict):
        self._step["loss"].append(stats["loss_total"])
        self._step["policy"].append(stats["loss_policy"])
        self._step["value"].append(stats["loss_value"])
        self._step["ent_model"].append(stats.get("ent_model", 0.0))
        self._step["ent_target"].append(stats.get("ent_target", 0.0))
        self._step["top1"].append(stats.get("top1_match", 0.0))
        x = np.arange(1, len(self._step["loss"]) + 1)

        # losses
        if self._lines["step_loss"] is None:
            (self._lines["step_loss"],)   = self.ax[0].plot(x, self._step["loss"],  label="loss")
            (self._lines["step_policy"],) = self.ax[0].plot(x, self._step["policy"],label="policy")
            (self._lines["step_value"],)  = self.ax[0].plot(x, self._step["value"], label="value")
            self.ax[0].legend(loc="upper right")
        else:
            self._lines["step_loss"].set_data(x, self._step["loss"])
            self._lines["step_policy"].set_data(x, self._step["policy"])
            self._lines["step_value"].set_data(x, self._step["value"])
        self.ax[0].relim(); self.ax[0].autoscale_view()

        # diagnostics
        if self._lines["step_ent_m"] is None:
            (self._lines["step_ent_m"],) = self.ax[1].plot(x, self._step["ent_model"], label="H(model)")
            (self._lines["step_ent_t"],) = self.ax[1].plot(x, self._step["ent_target"], label="H(target)")
            (self._lines["step_top1"],)  = self.ax[1].plot(x, self._step["top1"],      label="top1 acc")
            self.ax[1].legend(loc="upper right")
        else:
            self._lines["step_ent_m"].set_data(x, self._step["ent_model"])
            self._lines["step_ent_t"].set_data(x, self._step["ent_target"])
            self._lines["step_top1"].set_data(x, self._step["top1"])
        self.ax[1].relim(); self.ax[1].autoscale_view()

        self.fig.suptitle(
            f"Cycle {cycle_idx} • Step {step_idx} | "
            f"loss {stats['loss_total']:.3f} • pol {stats['loss_policy']:.3f} • "
            f"val {stats['loss_value']:.3f} • acc {stats.get('top1_match',0.):.2f} • "
            f"Hm {stats.get('ent_model',0.):.2f}",
            fontsize=11
        )

    # ---------- end-of-cycle train summary ----------
    def update_train(self, cycle_idx: int, tr: dict):
        if not tr or tr.get("tr_updates", 0) == 0: return
        self.hist["cycle"].append(cycle_idx)
        self.hist["tr_loss"].append(tr.get("tr_loss"))
        self.hist["tr_policy"].append(tr.get("tr_policy"))
        self.hist["tr_value"].append(tr.get("tr_value"))
        self.hist["ent_model"].append(tr.get("ent_model"))
        self.hist["ent_target"].append(tr.get("ent_target"))
        self.hist["top1_match"].append(tr.get("top1_match"))

    # ---------- per-cycle eval point (THIS FIXES THE EMPTY PLOT) ----------
    def update_eval_point(self, cycle_idx: int, wr_random: float | None, wr_l1: float | None):
        self.eval_cycles.append(cycle_idx)
        self.eval_wr_random.append(wr_random or 0.0)
        self.eval_wr_l1.append(wr_l1 or 0.0)
        # redraw bottom panel only
        self.ax[2].cla()
        self.ax[2].plot(self.eval_cycles, self.eval_wr_random, marker="o", label="vs Random")
        self.ax[2].plot(self.eval_cycles, self.eval_wr_l1,     marker="s", label="vs L1")
        self.ax[2].set_ylim(0, 1)
        self.ax[2].set_xlabel("cycle"); self.ax[2].set_ylabel("win rate")
        self.ax[2].legend(loc="upper left")
        self._configure_axes()

    # ---------- main-thread refresh ----------
    def refresh(self):
        self.fig.tight_layout()
        try:
            self._disp.update(self.fig)
        except Exception:
            self._disp = display.display(self.fig, display_id=True)

    # ---------- final bar chart ----------
    def show_final_eval_bars(self, results: list[dict]):
        if not results: return
        names = [r["opponent"] for r in results]
        wrs   = [r["win_rate"] for r in results]

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        bars = ax.bar(names, wrs)
        ax.set_ylim(0, 1)
        ax.set_ylabel("win rate")
        ax.set_title("Final Evaluation (win rate)")
        for b, wr in zip(bars, wrs):
            ax.text(b.get_x() + b.get_width()/2, wr + 0.02, f"{wr:.2f}", ha="center", va="bottom")
        display.display(fig)
