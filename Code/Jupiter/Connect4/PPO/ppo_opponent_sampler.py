# PPO/ppo_opponent_sampler.py
from __future__ import annotations
import numpy as np

class OpponentSampler:
    """
    Weighted opponent picker.

    Keys:
      "R"         -> Random
      "L1"..."L13"-> Lookahead-1..13   (any 'L' + digits is treated as depth)
      "SP"        -> Self-play (same policy acting as -1)
      "POP"  -> Population ensemble (HOF)
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
        if k == "R":
            return None           # random opponent
        if k == "SP":
            return "self"         # self-play: use student policy
        if k == "POP":
            return "POP"          # NEW: hall-of-fame ensemble
        if k.startswith("L") and k[1:].isdigit():
            return int(k[1:])     # lookahead depth
        raise ValueError(f"Unknown opponent key: {k}")
