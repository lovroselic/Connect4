# A0/a0_buffer.py
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import torch
from A0.a0_selfplay import TrajectoryStep

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self._states: List[np.ndarray] = []
        self._pis: List[np.ndarray] = []
        self._zs: List[float] = []
        self._masks: List[np.ndarray] = []
        self._start = 0  # ring buffer logical start

    def __len__(self) -> int:
        return len(self._states)

    def _append(self, state: np.ndarray, pi: np.ndarray, z: float, mask: np.ndarray):
        if len(self) < self.capacity:
            self._states.append(state)
            self._pis.append(pi)
            self._zs.append(float(z))
            self._masks.append(mask)
        else:
            i = self._start
            self._states[i] = state
            self._pis[i] = pi
            self._zs[i] = float(z)
            self._masks[i] = mask
            self._start = (self._start + 1) % self.capacity

    def add_game(self, steps: List[TrajectoryStep], augment: bool = False):
        """
        Each TrajectoryStep must have fields:
          - state: [4,6,7] float32
          - pi:    [7]     float32
          - z:     scalar  float32
          - legal_mask: [7] bool
        """
        for st in steps:
            self._append(st.state, st.pi, st.z, st.legal_mask)
            if augment:
                # Horizontal mirror
                s_m = st.state[:, :, ::-1].copy()   # [4,6,7]
                pi_m = st.pi[::-1].copy()           # [7]
                mask_m = st.legal_mask[::-1].copy() # [7]
                self._append(s_m, pi_m, st.z, mask_m)

                # Color swap current/opponent planes (flip value label)
                s_c = st.state.copy()
                s_c[[0, 1]] = s_c[[1, 0]]
                self._append(s_c, st.pi.copy(), -st.z, st.legal_mask.copy())

    def sample_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        replace: bool = True,
        min_batch: int = 32,
    ) -> Dict[str, torch.Tensor]:
        """
        Draw a batch (with replacement by default).
        If buffer is small, still returns at least `min_batch` samples (by resampling).
        """
        n = len(self)
        assert n > 0, "ReplayBuffer is empty"

        # pick a practical batch size so training can start early
        bsz = min(batch_size, max(min_batch, n))
        idx = np.random.choice(n, size=bsz, replace=replace)

        states = np.stack([self._states[i] for i in idx], axis=0)  # [B,4,6,7]
        pis    = np.stack([self._pis[i]    for i in idx], axis=0)  # [B,7]
        zs     = np.array([self._zs[i]     for i in idx], dtype=np.float32)  # [B]
        masks  = np.stack([self._masks[i]  for i in idx], axis=0)  # [B,7]

        return {
            "states": torch.from_numpy(states).to(device=device, dtype=torch.float32),
            "target_pi": torch.from_numpy(pis).to(device=device, dtype=torch.float32),
            "target_z": torch.from_numpy(zs).to(device=device, dtype=torch.float32),
            "legal_mask": torch.from_numpy(masks).to(device=device, dtype=torch.bool),
        }

    # --------- (de)serialization for checkpointing ---------

    def state_dict(self) -> Dict[str, Any]:
        """Serialize the entire buffer to compact numpy arrays."""
        return {
            "capacity": int(self.capacity),
            "start": int(self._start),
            "states": np.asarray(self._states, dtype=np.float32),  # [N,4,6,7]
            "pis":    np.asarray(self._pis,    dtype=np.float32),  # [N,7]
            "zs":     np.asarray(self._zs,     dtype=np.float32),  # [N]
            "masks":  np.asarray(self._masks,  dtype=np.bool_),    # [N,7]
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        """Restore buffer contents from `state_dict`."""
        self.capacity = int(sd.get("capacity", self.capacity))
        self._start   = int(sd.get("start", 0))

        states = np.asarray(sd["states"], dtype=np.float32)
        pis    = np.asarray(sd["pis"],    dtype=np.float32)
        zs     = np.asarray(sd["zs"],     dtype=np.float32)
        masks  = np.asarray(sd["masks"],  dtype=np.bool_)

        n = int(states.shape[0])
        self._states = [states[i] for i in range(n)]
        self._pis    = [pis[i]    for i in range(n)]
        self._zs     = [float(zs[i]) for i in range(n)]
        self._masks  = [masks[i]  for i in range(n)]
