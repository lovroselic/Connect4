# C4/opening_tracker.py
from __future__ import annotations

import numpy as np


def init_opening_kpis(cols: int = 7):
    return {
        "agent_first_counts": np.zeros(cols, dtype=np.int64),
        "opp_first_counts":   np.zeros(cols, dtype=np.int64),
        "agent_first_total":  0,
        "opp_first_total":    0,
    }


def record_first_move(kpis, col: int | None, who: str) -> None:
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

    a_tot = int(kpis["agent_first_total"])
    o_tot = int(kpis["opp_first_total"])
    a_mode = int(np.argmax(kpis["agent_first_counts"])) if a_tot else None
    o_mode = int(np.argmax(kpis["opp_first_counts"])) if o_tot else None

    return {
        "a0_center_rate": _rate(kpis["agent_first_counts"], a_tot, 3),
        "o0_center_rate": _rate(kpis["opp_first_counts"],   o_tot, 3),
        "a0_mode": a_mode,
        "o0_mode": o_mode,
    }


class OpeningTracker:
    """
    Lightweight first-move diagnostics:
      - histogram counts for agent/opponent first move
      - periodic logging of center rate over episodes
    """

    def __init__(self, cols: int = 7, log_every: int = 25, ma_window: int = 25):
        self.cols = int(cols)
        self.center = self.cols // 2
        self.log_every = int(log_every)
        self.ma_window = int(ma_window)

        self.kpis = init_opening_kpis(self.cols)
        self.episodes: list[int] = []
        self.a_center_rates: list[float] = []
        self.o_center_rates: list[float] = []

        self.phase_marks: list[tuple[int, str]] = []

    def on_first_move(self, col: int, is_agent: bool) -> None:
        record_first_move(self.kpis, col, "agent" if is_agent else "opp")

    def get_last_logged_center_rate(self) -> float:
        return float(self.a_center_rates[-1]) if self.a_center_rates else 0.0

    def maybe_log(self, episode_idx: int) -> None:
        if episode_idx % self.log_every != 0:
            return

        a_tot = max(1, int(self.kpis["agent_first_total"]))
        o_tot = max(1, int(self.kpis["opp_first_total"]))

        a_rate = float(self.kpis["agent_first_counts"][self.center] / a_tot)
        o_rate = float(self.kpis["opp_first_counts"][self.center] / o_tot)

        self.episodes.append(int(episode_idx))
        self.a_center_rates.append(a_rate)
        self.o_center_rates.append(o_rate)

    def mark_phase(self, episode_idx: int, label: str) -> None:
        self.phase_marks.append((int(episode_idx), str(label)))
