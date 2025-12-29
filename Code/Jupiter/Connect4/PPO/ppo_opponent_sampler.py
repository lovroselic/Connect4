# PPO/ppo_opponent_sampler.py
from __future__ import annotations
import re
import numpy as np

class OpponentSampler:
    """
    Weighted opponent picker.

    Keys:
      "R"          -> Random
      "L1".."L13"  -> Lookahead depth
      "SP"         -> Self-play (same policy acting as -1)
      "POP"        -> Population ensemble (HOF), handled by caller
    """
    def __init__(self, weights: dict[str, float], seed: int | None = None, sort_keys: bool = False):
        keys = list(weights.keys())
        self.keys = sorted(keys) if sort_keys else keys

        w = np.array([float(weights[k]) for k in self.keys], dtype=np.float64)
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        s = float(w.sum())
        if s <= 0:
            raise ValueError("at least one positive weight required")

        self.p = w / s
        self.rng = np.random.default_rng(seed)

    def sample_key(self) -> str:
        i = int(self.rng.choice(len(self.keys), p=self.p))
        return self.keys[i]

    @staticmethod
    def key_to_mode(k: str):
        """
        Returns a mode suitable for select_opponent_action / helper logic:
          None      -> random
          "self"    -> policy acts as opponent
          int       -> lookahead depth

        NOTE: "POP" is NOT a mode; caller should treat it as a source switch.
        """
        if k == "R":
            return None
        if k == "SP":
            return "self"
        if k.startswith("L") and k[1:].isdigit():
            return int(k[1:])
        if k == "POP":
            return "self"   # keeps mode semantics consistent
        raise ValueError(f"Unknown opponent key: {k}")
