# PPO/phase_tracker.py
# Phase boundaries + lookup helpers for PPO curriculum.
# Works with your newer schema:
#   TRAINING_PHASES = {
#       "PhaseName": {"duration": 100, "opponent_mix": {...}, "params": {...}},
#       ...
#   }
# It also tolerates legacy schemas that already have "length" cumulative ends.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


# -----------------------------
# Phase length normalization
# -----------------------------

def compute_phase_lengths(phases: Dict[str, Dict[str, Any]]) -> Tuple[Optional[int], str]:
    """
    Mutates `phases` in-place to ensure each phase has:
      - "duration": int | None
      - "start": int (episode start, inclusive)
      - "length": int | None (episode end, exclusive, cumulative)

    Supports:
      - New style: {"duration": N, ...}
      - Legacy style: {"length": cumulative_end, ...}

    Returns:
      (total_length, last_phase_name)
      total_length is None if an open-ended phase exists (duration/length is None).
    """
    end = 0
    last_name = None
    open_ended = False

    for name, meta in phases.items():
        last_name = name

        dur = meta.get("duration", None)
        cum_end = meta.get("length", None)

        # Determine duration / length
        if dur is None and cum_end is None:
            # Truly open-ended
            meta["start"] = int(end)
            meta["duration"] = None
            meta["length"] = None
            open_ended = True
            continue

        if dur is None and cum_end is not None:
            # Legacy: length is cumulative end
            cum_end = int(cum_end)
            dur = max(0, cum_end - end)

        if dur is not None:
            dur = int(dur)
            meta["duration"] = dur
            meta["start"] = int(end)
            end = end + dur
            meta["length"] = int(end)

    total = None if open_ended else int(end)
    return total, str(last_name)


def phase_boundaries(phases: Dict[str, Dict[str, Any]], up_to: Optional[int] = None) -> List[Tuple[str, int]]:
    """
    Return [(phase_name, end_episode), ...] sorted by end episode.
    Skips open-ended phases (length None).
    """
    items: List[Tuple[str, int]] = []
    for name, meta in phases.items():
        L = meta.get("length", None)
        if L is None:
            continue
        items.append((str(name), int(L)))

    items.sort(key=lambda p: p[1])
    if up_to is not None:
        items = [p for p in items if p[1] <= int(up_to)]
    return items


def draw_phase_vlines(
    ax,
    phases: Dict[str, Dict[str, Any]],
    up_to: Optional[int] = None,
    *,
    label: bool = True,
    color: str = "k",
    ls: str = ":",
    lw: float = 1.0,
    alpha: float = 0.6,
    label_y: Optional[float] = None,
    fontsize: int = 8,
    ha: str = "left",
    va: str = "top",
) -> None:
    """
    Draw vertical lines (and optional rotated labels) at phase boundaries.
    Requires matplotlib axis; intentionally kept import-free.
    """
    bounds = phase_boundaries(phases, up_to=up_to)
    if not bounds:
        return

    ymin, ymax = ax.get_ylim()
    ytxt = (ymax * 0.95) if label_y is None else float(label_y)

    for name, ep in bounds:
        ax.axvline(ep, color=color, linestyle=ls, linewidth=lw, alpha=alpha)
        if label:
            ax.text(ep, ytxt, name, rotation=90, ha=ha, va=va, fontsize=fontsize, alpha=alpha)


# -----------------------------
# Display helpers (notebook)
# -----------------------------

_DEFAULT_PARAM_COLS = [
    "lr", "clip", "entropy", "epochs", "batch_size", "steps_per_update",
    "vf_clip", "max_grad_norm", "target_kl", "temperature",
    "center_start", "guard_prob", "win_now_prob", "guard_ply_min", "guard_ply_max",
    "distill_coef", "mentor_depth", "mentor_prob", "mentor_coef",
]

def _mix_to_str(mix: Any) -> str:
    if not isinstance(mix, dict):
        return "" if mix is None else str(mix)
    # stable ordering: highest prob first, then key
    items = sorted(((k, float(v)) for k, v in mix.items()), key=lambda kv: (-kv[1], str(kv[0])))
    return ", ".join([f"{k}:{v:.2f}" for k, v in items])


def phases_to_dataframe(
    phases: Dict[str, Dict[str, Any]],
    *,
    param_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert phases dict -> dataframe with:
      start, duration, length, opponent_mix, plus selected params columns.
    """
    compute_phase_lengths(phases)

    cols = list(_DEFAULT_PARAM_COLS if param_cols is None else param_cols)
    rows: List[Dict[str, Any]] = []

    for name, meta in phases.items():
        params = dict(meta.get("params", {}) or {})
        row: Dict[str, Any] = {
            "phase": str(name),
            "start": meta.get("start", None),
            "duration": meta.get("duration", None),
            "length": meta.get("length", None),
            "opponent_mix": _mix_to_str(meta.get("opponent_mix", None)),
        }
        for c in cols:
            row[c] = params.get(c, None)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("phase")

    # Prefer sorting by start if available
    if "start" in df.columns and df["start"].notna().any():
        df = df.sort_values(by=["start"], ascending=True)

    return df


def display_phases_table(
    phases: Dict[str, Dict[str, Any]],
    *,
    param_cols: Optional[List[str]] = None,
    html: bool = True,
):
    """
    Notebook-friendly table display.
    Returns: (total_length, last_phase_name)

    Usage:
      PHASES = PhaseTracker(TRAINING_PHASES)
      L, last = display_phases_table(TRAINING_PHASES)
    """
    total, last = compute_phase_lengths(phases)
    df = phases_to_dataframe(phases, param_cols=param_cols)

    if html:
        try:
            from IPython.display import display, HTML
            display(HTML(df.to_html()))
        except Exception:
            print(df)
    else:
        print(df)

    return total, last


# -----------------------------
# PhaseTracker (fast lookup)
# -----------------------------

@dataclass(frozen=True)
class PhaseInfo:
    name: str
    start: int
    end: Optional[int]            # exclusive end, None if open-ended
    duration: Optional[int]
    opponent_mix: Dict[str, float]
    params: Dict[str, Any]

    @property
    def progress(self) -> Optional[float]:
        if self.end is None or self.duration in (None, 0):
            return None
        return 1.0


class PhaseTracker:
    """
    Episode-indexed phase helper.

    - Keeps a cached boundary list for fast lookup.
    - You can ask for current phase, detect changes, and fetch params.
    """

    def __init__(self, phases: Dict[str, Dict[str, Any]]):
        self.phases = phases
        self.total_length, self.last_phase = compute_phase_lengths(self.phases)
        self._cached = self._build_cache()
        self._current_name: Optional[str] = None

    def _build_cache(self) -> List[Tuple[str, int, Optional[int]]]:
        """
        Cache as list of (name, start, end).
        End can be None (open-ended).
        """
        cache: List[Tuple[str, int, Optional[int]]] = []
        for name, meta in self.phases.items():
            s = int(meta.get("start", 0))
            e = meta.get("length", None)
            e = None if e is None else int(e)
            cache.append((str(name), s, e))
        return cache

    def get_name(self, episode: int) -> str:
        ep = int(episode)
        for name, _s, e in self._cached:
            if e is None or ep < e:
                return name
        return str(self.last_phase)

    def get_info(self, episode: int) -> PhaseInfo:
        name = self.get_name(episode)
        meta = self.phases[name]
        params = dict(meta.get("params", {}) or {})
        mix_raw = meta.get("opponent_mix", {}) or {}
        mix = {str(k): float(v) for k, v in (mix_raw.items() if isinstance(mix_raw, dict) else [])}

        return PhaseInfo(
            name=name,
            start=int(meta.get("start", 0)),
            end=None if meta.get("length", None) is None else int(meta["length"]),
            duration=None if meta.get("duration", None) is None else int(meta["duration"]),
            opponent_mix=mix,
            params=params,
        )

    def maybe_advance(self, episode: int) -> Tuple[bool, PhaseInfo]:
        """
        Returns:
          (phase_changed, phase_info)
        """
        info = self.get_info(episode)
        changed = (info.name != self._current_name)
        self._current_name = info.name
        return changed, info

    # ✅ Notebook API compatibility
    def start_episode(self, episode: int) -> Tuple[PhaseInfo, bool]:
        """
        Notebook expects:
            phase_info, changed = PHASES.start_episode(ep)
        """
        changed, info = self.maybe_advance(episode)
        return info, changed

    def reset(self) -> None:
        self._current_name = None
