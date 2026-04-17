"""dqn_phase_manager.py

Tiny phase scheduler compatible with the PPO-style TRAINING_PHASES dict:

TRAINING_PHASES = {
  "Random": {
      "duration": 50,
      "opponent_mix": {"R": 1.0},
      "params": {"lr": 2e-4, "epsilon": 0.2, "epsilon_min": 0.05, ...}
  },
  ...
}

Use:
  PHASES = PhaseManager(TRAINING_PHASES)
  phase_info, changed = PHASES.start_episode(ep)

Episode numbers are 1-based.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class PhaseInfo:
    name: str
    start_ep: int
    end_ep: int


class PhaseManager:
    def __init__(self, phases: Dict[str, Dict[str, Any]]):
        if not phases:
            raise ValueError("TRAINING_PHASES is empty")

        self.phases = dict(phases)
        self._timeline = []  # list of (name, start, end)

        ep = 1
        for name, cfg in self.phases.items():
            dur = int(cfg.get("duration", 0))
            if dur <= 0:
                raise ValueError(f"Phase '{name}' must have duration > 0")
            start = ep
            end = ep + dur - 1
            self._timeline.append((name, start, end))
            ep = end + 1

        self._current_name: Optional[str] = None

    def start_episode(self, episode: int) -> Tuple[PhaseInfo, bool]:
        e = int(episode)
        if e <= 0:
            raise ValueError("episode must be 1-based and >= 1")

        for name, start, end in self._timeline:
            if start <= e <= end:
                changed = (name != self._current_name)
                self._current_name = name
                return PhaseInfo(name=name, start_ep=start, end_ep=end), changed

        # beyond configured phases: stick to last one
        name, start, end = self._timeline[-1]
        changed = (name != self._current_name)
        self._current_name = name
        return PhaseInfo(name=name, start_ep=start, end_ep=end), changed

    def current_name(self) -> Optional[str]:
        return self._current_name


__all__ = ["PhaseManager", "PhaseInfo"]
