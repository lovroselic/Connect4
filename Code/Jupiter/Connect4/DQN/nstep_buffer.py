# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 16:49:06 2025

@author: Lovro

NStepBuffer keeps a rolling window of transitions and emits:
  • 1-step transitions
  • n-step transitions (with early-termination handling)

It will automatically use augmented pushers (push_1step_aug / push_nstep_aug)
if the provided replay memory implements them; otherwise it falls back to the
base pushers.
"""

from collections import deque


class NStepBuffer:
    """
    Usage:
        buf = NStepBuffer(n=3, gamma=0.97, memory=agent.memory)
        buf.reset()
        ...
        buf.append(s, a, r, s_next, done, player)
        ...
        buf.flush()
    """
    def __init__(self, n: int, gamma: float, memory):
        assert n >= 2, "n-step should be >= 2 to be useful"
        self.n = int(n)
        self.gamma = float(gamma)
        self.mem = memory
        self.roll = deque()

    def reset(self):
        self.roll.clear()

    def append(self, state, action, reward, next_state, done, player):
        """Append one environment step; may emit one or more items."""
        self.roll.append((state, action, reward, next_state, bool(done), int(player)))
        self._emit(force=False)

    def flush(self):
        """Flush remaining short tails at episode end."""
        self._emit(force=True)

    # ---- internal ----
    def _emit(self, force: bool):
        while self.roll and (force or len(self.roll) >= self.n):
            window = list(self.roll)[:self.n]

            # Compute R^(n) with early termination
            Rn = 0.0
            done_n = False
            next_state_n = None
            for i, (_, _, r_i, s_i1, d_i, _) in enumerate(window):
                Rn += (self.gamma ** i) * r_i
                if d_i:
                    done_n = True
                    next_state_n = s_i1
                    break
            if not done_n:
                next_state_n = window[-1][3]  # s_{t+n}

            # anchor (first step)
            s0, a0, r0, s1, d1, p0 = self.roll[0]
            n_used = min(self.n, len(window))

            # 1-step (with augmentation if available)
            if hasattr(self.mem, "push_1step_aug"):
                self.mem.push_1step_aug(s0, a0, r0, s1, d1, p0)
            else:
                self.mem.push_1step(s0, a0, r0, s1, d1, p0)

            # n-step (with augmentation if available)
            if hasattr(self.mem, "push_nstep_aug"):
                self.mem.push_nstep_aug(s0, a0, Rn, next_state_n, done_n, p0, n_used)
            else:
                self.mem.push_nstep(s0, a0, Rn, next_state_n, done_n, p0, n_used)

            # slide window
            self.roll.popleft()
