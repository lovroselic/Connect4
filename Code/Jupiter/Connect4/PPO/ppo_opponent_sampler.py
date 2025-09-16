# PPO/ppo_opponent_sampler.py
from __future__ import annotations
import numpy as np

class OpponentSampler:
    """
    Weighted opponent picker.

    Keys:
      "R"   -> Random
      "L1"  -> Lookahead-1
      "L2"  -> Lookahead-2
      "L3"  -> Lookahead-3
      "SP"  -> Self-play (same policy acting as -1)
    """
    def __init__(self, weights: dict[str, float], seed: int | None = None):
        self.keys = list(weights.keys())
        w = np.array([float(weights[k]) for k in self.keys], dtype=np.float64)
        assert np.all(w >= 0), "weights must be non-negative"
        assert w.sum() > 0, "at least one positive weight required"
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)

    def sample_key(self) -> str:
        i = int(self.rng.choice(len(self.keys), p=self.p))
        return self.keys[i]

    @staticmethod
    def key_to_mode(k: str):
        if k == "R":  return None       # Random
        if k == "SP": return "self"     # Self-play
        if k.startswith("L") and k[1:].isdigit():
            return int(k[1:])           # Lookahead depth
        raise ValueError(f"Unknown opponent key: {k}")
