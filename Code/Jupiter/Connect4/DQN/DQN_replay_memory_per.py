# -*- coding: utf-8 -*-
import numpy as np
from collections import namedtuple

Transition      = namedtuple("Transition",      ["state", "action", "reward",   "next_state",   "done",   "player"])
NStepTransition = namedtuple("NStepTransition", ["state", "action", "reward_n", "next_state_n", "done_n", "player", "n_steps"])

class PrioritizedReplayMemory:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        eps: float = 1e-3,
        init_boost_terminal: float = 1.2,
        init_boost_oppmove: float   = 1.0,
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.init_boost_terminal = float(init_boost_terminal)
        self.init_boost_oppmove  = float(init_boost_oppmove)

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

    # ---------- pushes ----------
    def push_1step(self, s, a, r, ns, done, player):
        maxp = self.prio_1.max() if len(self.bank_1) else 1.0
        boost = 1.0
        if done:         boost *= self.init_boost_terminal
        if player == -1: boost *= self.init_boost_oppmove
        init_p = maxp * boost
        # ensure finite, >= eps  (*** change ***)
        init_p = max(self.eps, float(self._finite_or(init_p, 1.0)))

        t = Transition(s, a, r, ns, bool(done), int(player))
        if len(self.bank_1) < self.capacity:
            self.bank_1.append(t)
        else:
            self.bank_1[self.pos_1] = t
        self.prio_1[self.pos_1] = init_p
        self.pos_1 = (self.pos_1 + 1) % self.capacity

    def push_nstep(self, s, a, rN, nsN, doneN, player, n_steps):
        maxp = self.prio_n.max() if len(self.bank_n) else 1.0
        boost = 1.0
        if doneN:        boost *= self.init_boost_terminal
        if player == -1: boost *= self.init_boost_oppmove
        init_p = maxp * boost
        # ensure finite, >= eps  (*** change ***)
        init_p = max(self.eps, float(self._finite_or(init_p, 1.0)))

        t = NStepTransition(s, a, rN, nsN, bool(doneN), int(player), int(n_steps))
        if len(self.bank_n) < self.capacity:
            self.bank_n.append(t)
        else:
            self.bank_n[self.pos_n] = t
        self.prio_n[self.pos_n] = init_p
        self.pos_n = (self.pos_n + 1) % self.capacity

    # Legacy compatibility
    def push(self, *args):
        if len(args) == 6:
            s, a, r, ns, done, p = args
            self.push_1step(s, a, r, ns, done, p)
        elif len(args) == 7:
            s, p, a, r, ns, _p2, done = args
            self.push_1step(s, a, r, ns, done, p)
        else:
            raise TypeError(f"push(...) unexpected signature with {len(args)} args.")

    # ---------- augmented pushers (unchanged) ----------
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
        # sanitize non-finite / negatives (*** change ***)
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
        PRIO_CLIP = 5.0  # gentle; try 3.0–10.0 if you like
    
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


    # ---------- pruning ----------
    def prune(self, fraction, mode="recent"):
        fraction = float(fraction)
        if fraction <= 0.0:
            return

        def _prune_recent(bank, prio):
            k = int(len(bank) * fraction)
            if k <= 0:
                return bank, prio
            keep = bank[k:]
            pr = np.zeros_like(prio)
            n_keep = len(keep)
            pr[:n_keep] = prio[k:k+n_keep]
            return keep, pr

        def _prune_low(bank, prio):
            n = len(bank)
            k = int(n * fraction)
            if k <= 0:
                return bank, prio
            idx_sorted = np.argsort(prio[:n])
            drop = set(idx_sorted[:k].tolist())
            keep = [x for j, x in enumerate(bank) if j not in drop]
            pr = np.zeros_like(prio)
            pr[:len(keep)] = prio[[j for j in range(n) if j not in drop]]
            return keep, pr

        if mode == "recent":
            self.bank_1, self.prio_1 = _prune_recent(self.bank_1, self.prio_1)
            self.bank_n, self.prio_n = _prune_recent(self.bank_n, self.prio_n)
        elif mode == "low_priority":
            self.bank_1, self.prio_1 = _prune_low(self.bank_1, self.prio_1)
            self.bank_n, self.prio_n = _prune_low(self.bank_n, self.prio_n)
        else:
            raise ValueError(f"Unknown prune mode: {mode}")

        self.pos_1 = min(self.pos_1, len(self.bank_1)) % max(1, self.capacity)
        self.pos_n = min(self.pos_n, len(self.bank_n)) % max(1, self.capacity)

    def __len__(self):
        return len(self.bank_1) + len(self.bank_n)
