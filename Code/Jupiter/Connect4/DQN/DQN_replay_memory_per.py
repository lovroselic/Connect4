"""DQN/dqn_replay_memory_per.py

Prioritized replay memory for Connect-4 DQN (single-channel POV states).

State convention (STRICT):
  - np.ndarray float32 with shape (1, 6, 7)
  - values in {-1, 0, +1} from the *player-to-move* perspective

Augmentations:
  - mirror: horizontal flip, action becomes 6-action
  - colorswap: state *= -1, next_state *= -1, reward *= -1
    (valid because POV alternates, colorswap corresponds to swapping who is to move)

N-step convention:
  - reward_n MUST already be computed from the POV of the player at the start state.
    If you build it from per-move env rewards (which are for the mover each ply),
    you must alternate signs: +r0 - r1 + r2 ...

This module is intentionally self-contained and does not depend on torch.
"""

from __future__ import annotations

from collections import namedtuple
import math
from typing import List, Tuple, Optional

import numpy as np


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
NStepTransition = namedtuple("NStepTransition", ["state", "action", "reward_n", "next_state_n", "done_n", "n_steps"])


def _ensure_state_1x6x7(x: np.ndarray) -> np.ndarray:
    """Return float32 state with shape (1,6,7)."""
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if x.ndim == 2 and x.shape == (6, 7):
        x = x[None, :, :]

    if x.ndim != 3 or x.shape != (1, 6, 7):
        raise ValueError(f"state must have shape (1,6,7) or (6,7), got {x.shape}")

    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)

    return x


class PrioritizedReplayMemory:
    """Dual-bank PER (1-step + N-step) with seed-aware sampling."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.60,
        eps: float = 1e-3,
        init_percentile: float = 85.0,
        init_prio_cap: float = 3.0,
        init_boost_seed: float = 1.05,
        # terminal boosts by sign
        init_boost_terminal_win: float = 1.40,
        init_boost_terminal_loss: float = 1.80,
        init_boost_terminal_draw: float = 0.95,
        deboost_small_reward: float = 0.80,
        small_reward_abs_threshold: float = 0.20,
        deboost_nonterminal_only: bool = True,
        nstep_closeness_k: float = 0.04,
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)

        self.init_percentile = float(init_percentile)
        self.init_prio_cap = float(init_prio_cap)
        self.init_boost_seed = float(init_boost_seed)

        self.init_boost_terminal_win = float(init_boost_terminal_win)
        self.init_boost_terminal_loss = float(init_boost_terminal_loss)
        self.init_boost_terminal_draw = float(init_boost_terminal_draw)

        self.deboost_small_reward = float(deboost_small_reward)
        self.small_reward_abs_threshold = float(small_reward_abs_threshold)
        self.deboost_nonterminal_only = bool(deboost_nonterminal_only)

        self.nstep_closeness_k = float(nstep_closeness_k)

        # banks
        self.bank_1: List[Transition] = []
        self.bank_n: List[NStepTransition] = []

        self.prio_1 = np.zeros((self.capacity,), dtype=np.float32)
        self.prio_n = np.zeros((self.capacity,), dtype=np.float32)

        self.is_seed_1 = np.zeros((self.capacity,), dtype=bool)
        self.is_seed_n = np.zeros((self.capacity,), dtype=bool)

        self.pos_1 = 0
        self.pos_n = 0

        self.seed_mode = False

    # ----------------- public controls -----------------

    def begin_seeding(self) -> None:
        self.seed_mode = True

    def end_seeding(self) -> None:
        self.seed_mode = False

    def __len__(self) -> int:
        return len(self.bank_1) + len(self.bank_n)

    # ----------------- augmentation helpers -----------------

    @staticmethod
    def _mirror_state(s: np.ndarray) -> np.ndarray:
        # (1,6,7) flip columns
        return np.flip(s, axis=-1).copy()

    @staticmethod
    def _mirror_action(a: int) -> int:
        return 6 - int(a)

    @staticmethod
    def _colorswap_state(s: np.ndarray) -> np.ndarray:
        # POV colorswap for single-channel signed board
        return (-s).copy()

    # ----------------- priority init -----------------

    def _compute_terminal_boost(self, reward: float) -> float:
        if reward > 0:
            return self.init_boost_terminal_win
        if reward < 0:
            return self.init_boost_terminal_loss
        return self.init_boost_terminal_draw

    def _compute_boost(
        self,
        done_flag: bool,
        reward: Optional[float] = None,
        n_steps: Optional[int] = None,
        is_nstep: bool = False,
    ) -> float:
        boost = 1.0

        if done_flag and (reward is not None) and np.isfinite(reward):
            boost *= self._compute_terminal_boost(float(reward))
        elif (not done_flag) and (reward is not None) and np.isfinite(reward):
            if abs(float(reward)) < self.small_reward_abs_threshold:
                if self.deboost_nonterminal_only:
                    boost *= self.deboost_small_reward

        if is_nstep and (self.nstep_closeness_k > 0) and (n_steps is not None) and (n_steps > 0):
            boost *= (1.0 + self.nstep_closeness_k / float(n_steps))

        if self.seed_mode:
            boost *= self.init_boost_seed

        return float(boost)

    def _seed_priority(
        self,
        prio_vec: np.ndarray,
        bank_len: int,
        done_flag: bool,
        reward_val: float,
        n_steps: Optional[int] = None,
        is_nstep: bool = False,
    ) -> float:
        base = 1.0
        if bank_len > 0:
            pv = prio_vec[:bank_len]
            pv = pv[np.isfinite(pv) & (pv > 0)]
            if pv.size:
                base = float(np.percentile(pv, self.init_percentile))

        boost = self._compute_boost(done_flag, reward=reward_val, n_steps=n_steps, is_nstep=is_nstep)
        init_val = min(self.init_prio_cap, base * boost)
        init_val = max(self.eps, float(init_val))
        if not np.isfinite(init_val):
            init_val = 1.0
        return float(init_val)

    def _next_writable_pos(self, pos: int, is_seed_flags: np.ndarray) -> int:
        # never overwrite seeds
        for _ in range(self.capacity):
            if not bool(is_seed_flags[pos]):
                return int(pos)
            pos = (pos + 1) % self.capacity
        return int(pos)

    def _push_to_bank(self, bank, prio, pos_ptr: int, is_seed_arr: np.ndarray, transition, init_p: float) -> int:
        if len(bank) < self.capacity:
            bank.append(transition)
            prio[len(bank) - 1] = float(init_p)
            is_seed_arr[len(bank) - 1] = bool(self.seed_mode)
            return len(bank) % self.capacity

        pos = self._next_writable_pos(int(pos_ptr), is_seed_arr)
        bank[pos] = transition
        prio[pos] = float(init_p)
        is_seed_arr[pos] = bool(self.seed_mode)
        return (pos + 1) % self.capacity

    # ----------------- push API -----------------

    def push_1step(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: bool) -> None:
        s = _ensure_state_1x6x7(s)
        ns = _ensure_state_1x6x7(ns)
        a = int(a)
        r = float(r)
        done = bool(done)

        init_p = self._seed_priority(self.prio_1, len(self.bank_1), done, r)
        t = Transition(s, a, r, ns, done)
        self.pos_1 = self._push_to_bank(self.bank_1, self.prio_1, self.pos_1, self.is_seed_1, t, init_p)

    def push_nstep(self, s: np.ndarray, a: int, rN: float, nsN: np.ndarray, doneN: bool, n_steps: int) -> None:
        s = _ensure_state_1x6x7(s)
        nsN = _ensure_state_1x6x7(nsN)
        a = int(a)
        rN = float(rN)
        doneN = bool(doneN)
        n_steps = int(n_steps)

        init_p = self._seed_priority(self.prio_n, len(self.bank_n), doneN, rN, n_steps=n_steps, is_nstep=True)
        t = NStepTransition(s, a, rN, nsN, doneN, n_steps)
        self.pos_n = self._push_to_bank(self.bank_n, self.prio_n, self.pos_n, self.is_seed_n, t, init_p)

    def push_1step_aug(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        ns: np.ndarray,
        done: bool,
        add_mirror: bool = True,
        add_colorswap: bool = True,
        add_mirror_colorswap: bool = True,
    ) -> None:
        """Push base transition plus optional augmentations."""
        self.push_1step(s, a, r, ns, done)

        s0 = _ensure_state_1x6x7(s)
        ns0 = _ensure_state_1x6x7(ns)
        a0 = int(a)
        r0 = float(r)
        done0 = bool(done)

        combos = []
        if add_mirror:
            combos.append((True, False))
        if add_colorswap:
            combos.append((False, True))
        if add_mirror_colorswap:
            combos.append((True, True))

        for mir, cs in combos:
            ss = s0
            nn = ns0
            aa = a0
            rr = r0

            if mir:
                ss = self._mirror_state(ss)
                nn = self._mirror_state(nn)
                aa = self._mirror_action(aa)

            if cs:
                ss = self._colorswap_state(ss)
                nn = self._colorswap_state(nn)
                rr = -rr

            self.push_1step(ss, aa, rr, nn, done0)

    def push_nstep_aug(
        self,
        s: np.ndarray,
        a: int,
        rN: float,
        nsN: np.ndarray,
        doneN: bool,
        n_steps: int,
        add_mirror: bool = True,
        add_colorswap: bool = True,
        add_mirror_colorswap: bool = True,
    ) -> None:
        self.push_nstep(s, a, rN, nsN, doneN, n_steps)

        s0 = _ensure_state_1x6x7(s)
        ns0 = _ensure_state_1x6x7(nsN)
        a0 = int(a)
        r0 = float(rN)
        done0 = bool(doneN)
        n0 = int(n_steps)

        combos = []
        if add_mirror:
            combos.append((True, False))
        if add_colorswap:
            combos.append((False, True))
        if add_mirror_colorswap:
            combos.append((True, True))

        for mir, cs in combos:
            ss = s0
            nn = ns0
            aa = a0
            rr = r0

            if mir:
                ss = self._mirror_state(ss)
                nn = self._mirror_state(nn)
                aa = self._mirror_action(aa)

            if cs:
                ss = self._colorswap_state(ss)
                nn = self._colorswap_state(nn)
                rr = -rr

            self.push_nstep(ss, aa, rr, nn, done0, n0)

    # ----------------- sampling -----------------

    def _draw(
        self,
        bank,
        prio: np.ndarray,
        count: int,
        beta: float,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        count = int(count)
        if count <= 0 or not bank:
            return [], np.empty((0,), np.int64), np.ones((0,), np.float32)

        size = len(bank)
        p = prio[:size].astype(np.float64, copy=True)
        p[~np.isfinite(p)] = 0.0
        p[p < 0.0] = 0.0
        p = np.power(p, self.alpha)

        s = float(p.sum())
        if not (np.isfinite(s) and s > 0.0):
            p = np.full(size, 1.0 / size, dtype=np.float64)
        else:
            p /= s

        if mask is not None:
            idx_all = np.where(mask[:size])[0]
            if idx_all.size == 0:
                return [], np.empty((0,), np.int64), np.ones((0,), np.float32)
            ps = p[idx_all]
            ps /= float(ps.sum())
            idx = rng.choice(idx_all, size=count, p=ps)
        else:
            idx = rng.choice(size, size=count, p=p)

        w = ((size * p[idx]) ** (-float(beta))).astype(np.float32)
        w /= (w.max() if w.size else 1.0)

        return [bank[int(i)] for i in idx], idx.astype(np.int64), w

    def sample_mixed(
        self,
        batch_size: int,
        mix_1step: float = 0.5,
        beta: float = 0.4,
        rng: Optional[np.random.Generator] = None,
    ):
        """Sample mixed batch without seed constraints."""
        if rng is None:
            rng = np.random.default_rng()

        batch_size = int(batch_size)
        b1 = int(round(batch_size * float(mix_1step)))
        bn = batch_size - b1

        s1, i1, w1 = self._draw(self.bank_1, self.prio_1, b1, beta, rng=rng)
        sn, in_, wn = self._draw(self.bank_n, self.prio_n, bn, beta, rng=rng)
        return (s1, sn), (i1, in_), (w1, wn)

    def sample_mixed_seedaware(
        self,
        batch_size: int,
        mix_1step: float = 0.5,
        beta: float = 0.4,
        max_seed_frac: float = 0.10,
        min_seed_frac: float = 0.03,
        rng: Optional[np.random.Generator] = None,
    ):
        """Seed-aware sampler, keeps seeds within [min_seed_frac, max_seed_frac] per sub-batch."""
        if rng is None:
            rng = np.random.default_rng()

        def draw_seedaware(bank, prio, is_seed, count: int):
            count = int(count)
            if count <= 0 or not bank:
                return [], np.empty((0,), np.int64), np.ones((0,), np.float32)

            n_items = len(bank)
            n_seed = int(is_seed[:n_items].sum())
            n_main = n_items - n_seed

            lo = int(math.ceil(count * float(min_seed_frac)))
            hi = int(math.floor(count * float(max_seed_frac)))
            hi = min(hi, count)
            if hi < lo:
                hi = lo

            k_seed = min(max(lo, 0), hi)
            k_seed = min(k_seed, n_seed)

            k_main = count - k_seed
            if k_main > n_main:
                deficit = k_main - n_main
                k_main = n_main
                k_seed = min(n_seed, k_seed + deficit)

            k_seed = min(max(k_seed, 0), count)
            k_main = count - k_seed

            seed_items, seed_idx, seed_w = self._draw(bank, prio, k_seed, beta, mask=is_seed, rng=rng)
            main_items, main_idx, main_w = self._draw(bank, prio, k_main, beta, mask=~is_seed, rng=rng)

            items = seed_items + main_items
            idx = np.concatenate([seed_idx, main_idx]) if (seed_idx.size or main_idx.size) else np.empty((0,), np.int64)
            w = np.concatenate([seed_w, main_w]) if (seed_w.size or main_w.size) else np.ones((0,), np.float32)
            return items, idx, w

        batch_size = int(batch_size)
        b1 = int(round(batch_size * float(mix_1step)))
        bn = batch_size - b1

        s1, i1, w1 = draw_seedaware(self.bank_1, self.prio_1, self.is_seed_1, b1)
        sn, in_, wn = draw_seedaware(self.bank_n, self.prio_n, self.is_seed_n, bn)
        return (s1, sn), (i1, in_), (w1, wn)

    # ----------------- priority updates -----------------

    def update_priorities(
        self,
        indices_1: np.ndarray,
        td_errors_1: np.ndarray,
        indices_n: Optional[np.ndarray] = None,
        td_errors_n: Optional[np.ndarray] = None,
        prio_clip: float = 5.0,
    ) -> None:
        prio_clip = float(prio_clip)

        def _upd(indices, errors, prio_vec):
            if indices is None or errors is None:
                return
            for i, e in zip(indices, errors):
                val = abs(float(e)) + self.eps
                if not np.isfinite(val):
                    val = 1.0
                prio_vec[int(i)] = min(val, prio_clip)

        _upd(indices_1, td_errors_1, self.prio_1)
        _upd(indices_n, td_errors_n, self.prio_n)

    # ----------------- pruning -----------------

    def prune_low_priority(self, fraction: float) -> None:
        """Drop the lowest priorities from both banks (never drops seeds due to overwrite rule only, so pruning can drop seeds)."""
        fraction = float(fraction)
        if fraction <= 0.0:
            return

        def _prune(bank, prio, is_seed):
            n = len(bank)
            if n == 0:
                return bank, prio, is_seed
            k = int(np.floor(n * fraction))
            if k <= 0:
                return bank, prio, is_seed
            if k >= n:
                return [], np.zeros_like(prio), np.zeros_like(is_seed)

            low_k = np.argpartition(prio[:n], k - 1)[:k]
            drop = set(low_k.tolist())
            keep_idx = [i for i in range(n) if i not in drop]

            new_bank = [bank[i] for i in keep_idx]
            new_len = len(new_bank)

            new_prio = np.zeros_like(prio)
            new_seed = np.zeros_like(is_seed)
            new_prio[:new_len] = prio[keep_idx]
            new_seed[:new_len] = is_seed[keep_idx]

            return new_bank, new_prio, new_seed

        self.bank_1, self.prio_1, self.is_seed_1 = _prune(self.bank_1, self.prio_1, self.is_seed_1)
        self.bank_n, self.prio_n, self.is_seed_n = _prune(self.bank_n, self.prio_n, self.is_seed_n)
        self.pos_1 = min(self.pos_1, len(self.bank_1)) % max(1, self.capacity)
        self.pos_n = min(self.pos_n, len(self.bank_n)) % max(1, self.capacity)


__all__ = [
    "Transition",
    "NStepTransition",
    "PrioritizedReplayMemory",
]
