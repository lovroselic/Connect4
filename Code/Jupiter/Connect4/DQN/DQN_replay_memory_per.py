# DQN_replay_memory_per.py
import numpy as np
from collections import namedtuple

Transition      = namedtuple("Transition",      ["state", "action", "reward",   "next_state",   "done",   "player"])
NStepTransition = namedtuple("NStepTransition", ["state", "action", "reward_n", "next_state_n", "done_n", "player", "n_steps"])

class PrioritizedReplayMemory:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.5, #0.6
        eps: float = 0.02 ,  #5e-3
        init_boost_terminal: float = 1.5,
        init_boost_oppmove: float = 1.05,
        init_percentile: float = 85.0,   # was 98.0 — softer so seeds don’t dominate #85
        init_boost_seed: float = 1.15,   # was 1.25 — softer initial seed priority #1.05
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
        self.beta = 0.55 

    # ----------------- helpers -----------------
    @staticmethod
    def _hflip_state(s):         # horizontal mirror (columns)
        return np.flip(s, axis=-1).copy()

    @staticmethod
    def _hflip_action(a):        # 7 columns → indices [0..6]
        return 6 - int(a)

    @staticmethod
    def _colorswap_state(s):     # agent ↔ opponent channel swap: here encoded as -state
        return -s

    def _finite_or(self, value, fallback):
        return value if np.isfinite(value) else fallback

    def _compute_boost(self, done_flag, player_flag):
        # optional initial priority boosts
        boost = 1.0
        if done_flag:
            boost *= self.init_boost_terminal
        if player_flag == -1:                # opponent moves can be slightly emphasized
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

    # ----------------- base pushers -----------------
    def push_1step(self, s, a, r, ns, done, player):
        init_p = self._seed_priority(self.prio_1, len(self.bank_1), done, player)
        t = Transition(s, a, r, ns, bool(done), int(player))
        self.pos_1 = self._push_to_bank(self.bank_1, self.prio_1, self.pos_1, self.is_seed_1, t, init_p)

    def push_nstep(self, s, a, rN, nsN, doneN, player, n_steps):
        init_p = self._seed_priority(self.prio_n, len(self.bank_n), doneN, player)
        t = NStepTransition(s, a, rN, nsN, bool(doneN), int(player), int(n_steps))
        self.pos_n = self._push_to_bank(self.bank_n, self.prio_n, self.pos_n, self.is_seed_n, t, init_p)


    def _apply_transform(self, state, action, player, mirror: bool, colorswap: bool):
        s = state
        a = int(action)
        p = int(player)
        if mirror:
            s = np.flip(s, axis=-1).copy()
            a = 6 - a
        if colorswap:
            s = -s
            p = -p
        return s, a, p
    
    def push_1step_aug(self, s, a, r, s2, done, player,
                       add_mirror=True, add_colorswap=True, add_mirror_colorswap=True):
        # base
        self.push_1step(s, a, r, s2, done, player)
    
        combos = []
        if add_mirror:             combos.append((True,  False))
        if add_colorswap:          combos.append((False, True))
        if add_mirror_colorswap:   combos.append((True,  True))
    
        for mir, col in combos:
            ss,  aa,  pp  = self._apply_transform(s,  a, player, mir, col)
            ss2, _,   _   = self._apply_transform(s2, a, player, mir, col)  # same transform
            rr = -r if col else r  # color swap flips reward sign
            self.push_1step(ss, aa, rr, ss2, done, pp)
    
    def push_nstep_aug(self, s, a, rN, sN, doneN, player, n_steps,
                       add_mirror=True, add_colorswap=True, add_mirror_colorswap=True):
        # base
        self.push_nstep(s, a, rN, sN, doneN, player, n_steps)
    
        combos = []
        if add_mirror:             combos.append((True,  False))
        if add_colorswap:          combos.append((False, True))
        if add_mirror_colorswap:   combos.append((True,  True))
    
        for mir, col in combos:
            ss,  aa,  pp  = self._apply_transform(s,  a, player, mir, col)
            ssN, _,   _   = self._apply_transform(sN, a, player, mir, col)
            rN_adj = -rN if col else rN
            self.push_nstep(ss, aa, rN_adj, ssN, doneN, pp, n_steps)



    # ----------------- sampling -----------------
    def _draw(self, bank, prio, count, beta, mask=None):
        if count <= 0 or not bank:
            return [], np.empty((0,), np.int64), np.ones((0,), np.float32)

        size = len(bank)
        p = prio[:size].astype(np.float64)
        p[~np.isfinite(p)] = 0.0
        p[p < 0.0] = 0.0
        p **= self.alpha
        s = p.sum()
        if not (np.isfinite(s) and s > 0):
            p = np.full(size, 1.0 / size)
        else:
            p = p / s

        # optional masked sampling (used by seed-aware sampler)
        if mask is not None:
            idx_all = np.where(mask[:size])[0]
            if idx_all.size == 0:
                return [], np.empty((0,), np.int64), np.ones((0,), np.float32)
            ps = p[idx_all]
            ps = ps / ps.sum()
            idx = np.random.choice(idx_all, count, p=ps)
        else:
            idx = np.random.choice(size, count, p=p)

        w = ((size * p[idx]) ** (-beta)).astype(np.float32) if idx.size else np.ones((0,), np.float32)
        return [bank[i] for i in idx], idx, w / (w.max() if w.size else 1.0)

    def sample_mixed(self, batch_size, mix=0.5, beta=0.4):
        # legacy sampler (no seed cap), kept for backward compatibility
        b1 = int(batch_size * mix)
        bn = batch_size - b1
        s1, i1, w1 = self._draw(self.bank_1, self.prio_1, b1, beta)
        sn, in_, wn = self._draw(self.bank_n, self.prio_n, bn, beta)
        return (s1, sn), (i1, in_), (w1, wn)

    #not used
    def sample_mixed_seedaware(self, batch_size, mix=0.5, beta=0.4, max_seed_frac=0.10):
        """Seed-aware sampler: caps seeds to at most max_seed_frac of each sub-batch."""
        b1 = int(batch_size * mix)
        bn = batch_size - b1

        def draw_seedaware(bank, prio, is_seed, count):
            if count <= 0 or not bank:
                return [], np.empty((0,), np.int64), np.ones((0,), np.float32)
            k_seed = min(int(round(count * max_seed_frac)), int(is_seed[:len(bank)].sum()))
            k_main = count - k_seed
            s_seed = self._draw(bank, prio, k_seed, beta, mask=is_seed)[0:3]
            s_main = self._draw(bank, prio, k_main, beta, mask=~is_seed)[0:3]
            # merge
            items  = (s_seed[0] + s_main[0])
            idx    = np.concatenate([s_seed[1], s_main[1]]) if s_seed[1].size or s_main[1].size else np.empty((0,), np.int64)
            weights= np.concatenate([s_seed[2], s_main[2]]) if s_seed[2].size or s_main[2].size else np.ones((0,), np.float32)
            return items, idx, weights

        s1, i1, w1 = draw_seedaware(self.bank_1, self.prio_1, self.is_seed_1, b1)
        sn, in_, wn = draw_seedaware(self.bank_n, self.prio_n, self.is_seed_n, bn)
        return (s1, sn), (i1, in_), (w1, wn)

    def sample(self, batch_size, beta=0.4):
        return self.sample_mixed(batch_size, mix=1.0, beta=beta)[0]

    def update_priorities(self, indices_1, td_errors_1, indices_n=None, td_errors_n=None):
        PRIO_CLIP = 3.5
        for indices, errors, prios in [(indices_1, td_errors_1, self.prio_1), (indices_n, td_errors_n, self.prio_n)]:
            if indices is not None:
                for i, e in zip(indices, errors):
                    prios[int(i)] = min(abs(float(e)), PRIO_CLIP) + self.eps

    # ----------------- pruning -----------------
    def _prune_low(self, bank, prio, is_seed_flags, fraction):
        n = len(bank)
        if n == 0:
            return bank, prio, is_seed_flags
        cand_idx = np.arange(n)[~is_seed_flags[:n]]  # do not prune seeds
        if cand_idx.size == 0:
            return bank, prio, is_seed_flags
        k = min(int(n * fraction), cand_idx.size)
        drop = set(cand_idx[np.argsort(prio[cand_idx])[:k]])
        keep_idx = [i for i in range(n) if i not in drop]
        new_bank = [bank[i] for i in keep_idx]
        new_prio = np.zeros_like(prio)
        new_seed = np.zeros_like(is_seed_flags)
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
            if not is_seed_flags[pos]:
                return pos
            pos = (pos + 1) % self.capacity
        return pos

    # ----------------- extras -----------------
    def __len__(self): return len(self.bank_1) + len(self.bank_n)
    def begin_seeding(self): self.seed_mode = True
    def end_seeding(self):   self.seed_mode = False
