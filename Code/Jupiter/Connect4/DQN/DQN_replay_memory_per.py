#DQN_preplay_memory_per.py

import numpy as np
from collections import namedtuple

Transition      = namedtuple("Transition",      ["state", "action", "reward",   "next_state",   "done",   "player"])
NStepTransition = namedtuple("NStepTransition", ["state", "action", "reward_n", "next_state_n", "done_n", "player", "n_steps"])

class PrioritizedReplayMemory:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.7,
        eps: float = 1e-3,
        init_boost_terminal: float = 1.3,
        init_boost_oppmove: float = 1.0,
        init_percentile: float = 98.0,
        init_boost_seed: float = 1.25,
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.init_boost_terminal = float(init_boost_terminal)
        self.init_boost_oppmove = float(init_boost_oppmove)
        self.init_percentile = float(init_percentile)
        self.init_boost_seed = float(init_boost_seed)

        self.bank_1, self.bank_n = [], []
        self.prio_1 = np.zeros((self.capacity,), dtype=np.float32)
        self.prio_n = np.zeros((self.capacity,), dtype=np.float32)
        self.pos_1 = self.pos_n = 0

        self.is_seed_1 = np.zeros((self.capacity,), dtype=bool)
        self.is_seed_n = np.zeros((self.capacity,), dtype=bool)

        self.seed_mode = False
        self.beta = 0.4

    # --- Helpers ---
    def _finite_or(self, value, fallback):
        return value if np.isfinite(value) else fallback

    def _compute_boost(self, done_flag, player_flag):
        boost = 1.0
        if done_flag:
            boost *= self.init_boost_terminal
        if player_flag == -1:
            boost *= self.init_boost_oppmove
        if self.seed_mode:
            boost *= self.init_boost_seed
        return boost

    def _seed_priority(self, prio_vector, bank_len, done_flag, player_flag):
        if bank_len <= 0:
            base = 1.0
        else:
            pv = prio_vector[:bank_len]
            pv = pv[np.isfinite(pv) & (pv > 0)]
            base = np.percentile(pv, self.init_percentile) if pv.size else 1.0
        boost = self._compute_boost(done_flag, player_flag)
        return max(self.eps, float(self._finite_or(base * boost, 1.0)))

    def _push_to_bank(self, bank, prio, pos_ptr, is_seed_arr, transition, init_p):
        if len(bank) < self.capacity:
            bank.append(transition)
            prio[len(bank) - 1] = init_p
            is_seed_arr[len(bank) - 1] = self.seed_mode
            return len(bank) % self.capacity
        else:
            pos = self._next_writable_pos(pos_ptr, is_seed_arr)
            bank[pos] = transition
            prio[pos] = init_p
            is_seed_arr[pos] = self.seed_mode
            return (pos + 1) % self.capacity

    def _flip_and_adjust(self, s, a, p, mirror=True, colorswap=True):
        sa_pairs = []
        if mirror:
            sa_pairs.append((np.flip(s, axis=-1).copy(), 6 - int(a), p))
        if colorswap:
            sa_pairs.append((-s, a, -p))
        if mirror and colorswap:
            sa_pairs.append((np.flip(-s, axis=-1).copy(), 6 - int(a), -p))
        return sa_pairs

    # --- Pushers ---
    def push_1step(self, s, a, r, ns, done, player):
        init_p = self._seed_priority(self.prio_1, len(self.bank_1), done, player)
        t = Transition(s, a, r, ns, bool(done), int(player))
        self.pos_1 = self._push_to_bank(self.bank_1, self.prio_1, self.pos_1, self.is_seed_1, t, init_p)

    def push_nstep(self, s, a, rN, nsN, doneN, player, n_steps):
        init_p = self._seed_priority(self.prio_n, len(self.bank_n), doneN, player)
        t = NStepTransition(s, a, rN, nsN, bool(doneN), int(player), int(n_steps))
        self.pos_n = self._push_to_bank(self.bank_n, self.prio_n, self.pos_n, self.is_seed_n, t, init_p)

    def push_1step_aug(self, s, a, r, s2, done, player, add_mirror=True, add_colorswap=True, add_mirror_colorswap=True):
        self.push_1step(s, a, r, s2, done, player)
        for ss, aa, rr in self._flip_and_adjust(s, a, r, mirror=add_mirror, colorswap=add_colorswap):
            self.push_1step(ss, aa, rr, self._flip_and_adjust(s2, a, r)[0][0], done, -int(player))

    def push_nstep_aug(self, s, a, rN, sN, doneN, player, n_steps, add_mirror=True, add_colorswap=True, add_mirror_colorswap=True):
        self.push_nstep(s, a, rN, sN, doneN, player, n_steps)
        for ss, aa, rr in self._flip_and_adjust(s, a, rN, mirror=add_mirror, colorswap=add_colorswap):
            self.push_nstep(ss, aa, rr, self._flip_and_adjust(sN, a, rN)[0][0], doneN, -int(player), n_steps)

    # --- Sampling ---
    def _draw(self, bank, prio, count, beta):
        if count <= 0 or not bank:
            return [], np.empty((0,), np.int64), np.ones((0,), np.float32)

        size = len(bank)
        p = prio[:size].astype(np.float64)
        p[~np.isfinite(p)] = 0.0
        p[p < 0.0] = 0.0
        p **= self.alpha
        s = p.sum()
        p = p / s if np.isfinite(s) and s > 0 else np.full(size, 1.0 / size)
        idx = np.random.choice(size, count, p=p)
        w = ((size * p[idx]) ** (-beta)).astype(np.float32)
        return [bank[i] for i in idx], idx, w / (w.max() if w.size else 1.0)

    def sample_mixed(self, batch_size, mix=0.5, beta=0.4):
        b1 = int(batch_size * mix)
        bn = batch_size - b1
        s1, i1, w1 = self._draw(self.bank_1, self.prio_1, b1, beta)
        sn, in_, wn = self._draw(self.bank_n, self.prio_n, bn, beta)
        return (s1, sn), (i1, in_), (w1, wn)

    def sample(self, batch_size, beta=0.4):
        return self.sample_mixed(batch_size, mix=1.0, beta=beta)[0]

    def update_priorities(self, indices_1, td_errors_1, indices_n=None, td_errors_n=None):
        PRIO_CLIP = 3.5
        for indices, errors, prios in [(indices_1, td_errors_1, self.prio_1), (indices_n, td_errors_n, self.prio_n)]:
            if indices is not None:
                for i, e in zip(indices, errors):
                    prios[int(i)] = min(abs(float(e)), PRIO_CLIP) + self.eps

    # --- Pruning ---
    def _prune_low(self, bank, prio, is_seed_flags, fraction):
        n = len(bank)
        if n == 0: return bank, prio, is_seed_flags
        cand_idx = np.arange(n)[~is_seed_flags[:n]]
        if cand_idx.size == 0: return bank, prio, is_seed_flags
        k = min(int(n * fraction), cand_idx.size)
        drop = set(cand_idx[np.argsort(prio[cand_idx])[:k]])
        keep_idx = [i for i in range(n) if i not in drop]
        new_bank = [bank[i] for i in keep_idx]
        new_prio = np.zeros_like(prio); new_seed = np.zeros_like(is_seed_flags)
        new_prio[:len(new_bank)] = prio[keep_idx]
        new_seed[:len(new_bank)] = is_seed_flags[keep_idx]
        return new_bank, new_prio, new_seed

    def prune(self, fraction, mode="low_priority"):
        if mode != "low_priority":
            raise ValueError("Only 'low_priority' pruning is supported.")
        self.bank_1, self.prio_1, self.is_seed_1 = self._prune_low(self.bank_1, self.prio_1, self.is_seed_1, fraction)
        self.bank_n, self.prio_n, self.is_seed_n = self._prune_low(self.bank_n, self.prio_n, self.is_seed_n, fraction)
        self.pos_1 = min(self.pos_1, len(self.bank_1)) % max(1, self.capacity)
        self.pos_n = min(self.pos_n, len(self.bank_n)) % max(1, self.capacity)

    def _next_writable_pos(self, pos, is_seed_flags):
        for _ in range(self.capacity):
            if not is_seed_flags[pos]: return pos
            pos = (pos + 1) % self.capacity
        return pos

    # --- Extras ---
    def __len__(self): return len(self.bank_1) + len(self.bank_n)
    def begin_seeding(self): self.seed_mode = True
    def end_seeding(self):   self.seed_mode = False
