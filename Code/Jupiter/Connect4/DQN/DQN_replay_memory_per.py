# -*- coding: utf-8 -*-
import numpy as np
from collections import namedtuple

Transition      = namedtuple("Transition",      ["state", "action", "reward",   "next_state",   "done",   "player"])
NStepTransition = namedtuple("NStepTransition", ["state", "action", "reward_n", "next_state_n", "done_n", "player", "n_steps"])

class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay for DQN with 1-step and N-step banks.

    Policy: **priority-only**.
      - New samples are seeded at a chosen percentile of existing priorities
        (more stable than 'max'), then lightly boosted for terminals and
        for opponent-turn entries.
      - Pruning removes the **lowest-priority** fraction only. No age-based
        logic remains.

    Banks:
      bank_1 / prio_1 : 1-step transitions
      bank_n / prio_n : n-step transitions
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.7,          # PER exponent on priorities
        eps: float = 1e-3,           # small additive to avoid zero prob
        init_boost_terminal: float = 1.3,
        init_boost_oppmove: float   = 1.0,
        init_percentile: float = 98.0,  # seed new items at this percentile of existing prios
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.init_boost_terminal = float(init_boost_terminal)
        self.init_boost_oppmove  = float(init_boost_oppmove)
        self.init_percentile     = float(init_percentile)

        self.bank_1 = []
        self.prio_1 = np.zeros((self.capacity,), dtype=np.float32)
        self.pos_1  = 0

        self.bank_n = []
        self.prio_n = np.zeros((self.capacity,), dtype=np.float32)
        self.pos_n  = 0

    # ---------- helpers ----------
    @staticmethod
    def _finite_or(value, fallback):
        return value if np.isfinite(value) else fallback

    def _seed_priority(self, prio_vector, bank_len, done_flag, player_flag):
        """Seed initial priority using a percentile of existing, then boost."""
        if bank_len <= 0:
            base = 1.0
        else:
            pv = prio_vector[:bank_len]
            pv = pv[np.isfinite(pv) & (pv > 0)]
            base = np.percentile(pv, self.init_percentile) if pv.size else 1.0
        boost = 1.0
        if done_flag:
            boost *= self.init_boost_terminal
        if player_flag == -1:
            boost *= self.init_boost_oppmove
        init_p = max(self.eps, float(self._finite_or(base * boost, 1.0)))
        return init_p

    # ---------- pushes ----------
    def push_1step(self, s, a, r, ns, done, player):
        init_p = self._seed_priority(self.prio_1, len(self.bank_1), done, player)
        t = Transition(s, a, r, ns, bool(done), int(player))
        if len(self.bank_1) < self.capacity:
            self.bank_1.append(t)
        else:
            self.bank_1[self.pos_1] = t
        self.prio_1[self.pos_1] = init_p
        self.pos_1 = (self.pos_1 + 1) % self.capacity

    def push_nstep(self, s, a, rN, nsN, doneN, player, n_steps):
        init_p = self._seed_priority(self.prio_n, len(self.bank_n), doneN, player)
        t = NStepTransition(s, a, rN, nsN, bool(doneN), int(player), int(n_steps))
        if len(self.bank_n) < self.capacity:
            self.bank_n.append(t)
        else:
            self.bank_n[self.pos_n] = t
        self.prio_n[self.pos_n] = init_p
        self.pos_n = (self.pos_n + 1) % self.capacity


    # ---------- augmented pushers ----------
    def push_1step_aug(self, s, a, r, s2, done, player,
                       add_mirror=True, add_colorswap=True, add_mirror_colorswap=True):
        self.push_1step(s, a, r, s2, done, player)
        if add_mirror:
            fs, fs2, fa = np.flip(s, axis=-1).copy(), np.flip(s2, axis=-1).copy(), 6 - int(a)
            self.push_1step(fs, fa, r, fs2, done, player)
        if add_colorswap:
            cs, cs2 = -s, -s2
            self.push_1step(cs, a, -r, cs2, done, -int(player))
        if add_mirror_colorswap:
            mcs, mcs2, cfa = np.flip(-s, axis=-1).copy(), np.flip(-s2, axis=-1).copy(), 6 - int(a)
            self.push_1step(mcs, cfa, -r, mcs2, done, -int(player))

    def push_nstep_aug(self, s, a, rN, sN, doneN, player, n_steps,
                       add_mirror=True, add_colorswap=True, add_mirror_colorswap=True):
        self.push_nstep(s, a, rN, sN, doneN, player, n_steps)
        if add_mirror:
            fs, fsN, fa = np.flip(s, axis=-1).copy(), np.flip(sN, axis=-1).copy(), 6 - int(a)
            self.push_nstep(fs, fa, rN, fsN, doneN, player, n_steps)
        if add_colorswap:
            cs, csN = -s, -sN
            self.push_nstep(cs, a, -rN, csN, doneN, -int(player), n_steps)
        if add_mirror_colorswap:
            mcs, mcsN, cfa = np.flip(-s, axis=-1).copy(), np.flip(-sN, axis=-1).copy(), 6 - int(a)
            self.push_nstep(mcs, cfa, -rN, mcsN, doneN, -int(player), n_steps)

    # ---------- sampling ----------
    def _draw(self, bank, prio, count, beta):
        if count <= 0 or not bank:
            return [], np.empty((0,), np.int64), np.ones((0,), np.float32)

        size = len(bank)
        p = prio[:size].astype(np.float64, copy=True)

        # sanitize
        p[~np.isfinite(p)] = 0.0
        p[p < 0.0] = 0.0

        p = p ** self.alpha
        p[~np.isfinite(p)] = 0.0
        s = p.sum()
        if not np.isfinite(s) or s <= 0.0:
            p = np.ones(size, dtype=np.float64) / size
        else:
            p /= s

        idx = np.random.choice(size, count, p=p)
        w = ((size * p[idx]) ** (-beta)).astype(np.float32)
        w /= w.max() if w.size else 1.0
        return [bank[i] for i in idx], idx.astype(np.int64), w

    def sample_mixed(self, batch_size, mix=0.5, beta=0.4):
        b1 = int(batch_size * mix)
        bn = batch_size - b1
        s1, i1, w1 = self._draw(self.bank_1, self.prio_1, b1, beta)
        sn, in_, wn = self._draw(self.bank_n, self.prio_n, bn, beta)
        return (s1, sn), (i1, in_), (w1, wn)

    def sample(self, batch_size, beta=0.4):
        (s1, _), (i1, _), (w1, _) = self.sample_mixed(batch_size, mix=1.0, beta=beta)
        return s1, i1, w1

    def update_priorities(self, indices_1, td_errors_1, indices_n=None, td_errors_n=None):
        PRIO_CLIP = 3.5  # gentle; tune 3.0–10.0 if needed

        if indices_1 is not None and len(indices_1):
            for i, e in zip(indices_1, td_errors_1):
                e = float(e)
                if not np.isfinite(e):
                    e = 0.0
                ae = min(abs(e), PRIO_CLIP)
                self.prio_1[int(i)] = ae + self.eps

        if indices_n is not None and len(indices_n):
            for i, e in zip(indices_n, td_errors_n):
                e = float(e)
                if not np.isfinite(e):
                    e = 0.0
                ae = min(abs(e), PRIO_CLIP)
                self.prio_n[int(i)] = ae + self.eps


    # ---------- pruning (priority-only) ----------
    def _prune_low(self, bank, prio, fraction):
        """Drop the lowest-priority fraction."""
        n = len(bank)
        if n == 0:
            return bank, prio
        k = int(n * float(fraction))
        if k <= 0:
            return bank, prio
        idx_sorted = np.argsort(prio[:n])  # ascending
        drop = set(idx_sorted[:k].tolist())
        keep = [x for j, x in enumerate(bank) if j not in drop]
        new_prio = np.zeros_like(prio)
        new_prio[:len(keep)] = prio[[j for j in range(n) if j not in drop]]
        return keep, new_prio

    def prune(self, fraction, mode="low_priority"):
        """
        Prune a fraction of the buffer by **lowest priority** only.
        `mode` is kept for API compatibility; any value other than "low_priority"
        raises an error to make the policy explicit.
        """
        fraction = float(fraction)
        if fraction <= 0.0:
            return
        if mode != "low_priority":
            raise ValueError("Only 'low_priority' pruning is supported in priority-only mode.")

        self.bank_1, self.prio_1 = self._prune_low(self.bank_1, self.prio_1, fraction)
        self.bank_n, self.prio_n = self._prune_low(self.bank_n, self.prio_n, fraction)
        # write pointers remain valid modulo capacity (no circular age logic needed)
        self.pos_1 = min(self.pos_1, len(self.bank_1)) % max(1, self.capacity)
        self.pos_n = min(self.pos_n, len(self.bank_n)) % max(1, self.capacity)

    def __len__(self):
        return len(self.bank_1) + len(self.bank_n)
