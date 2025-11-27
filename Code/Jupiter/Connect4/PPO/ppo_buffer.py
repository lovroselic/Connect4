# -*- coding: utf-8 -*-
"""
ppo_buffer.py — PPO rollout buffer with reward scaling and data augmentation
(H-flip and color-swap) for Connect4 with fixed roles:
- Agent is always +1 (channel 0 in two-plane form)
- Opponent is always -1 (channel 1)

Color-swap is a ROLE SWAP:
- Channel 0 continues to mean "agent" but it becomes the original opponent plane.
- Reward is NEGATED for color-swapped variants.

add_augmented(...) returns [(idx, sign), ...] where sign=+1 for same-role
(original, hflip) and sign=-1 for role-swapped (colorswap, hflip+colorswap).
Use add_penalty_to_entries(...) to attribute terminal penalties consistently.
"""
from dataclasses import dataclass
from typing import Union, List, Dict, Any, Callable, Tuple
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class PPOHyperParams:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_adv: bool = True


class PPOBuffer:
    """
    On-policy rollout storage for PPO.
    Stores only AGENT turns (insert only when the agent acts).

     State format:
      - Scalar board (6,7) with values in {-1,0,+1}, or
      - Two planes (2,6,7) = [agent(+1), opp(-1)], or
      - Four planes (4,6,7) = [agent, opp, row, col].

    Reward scaling:
      - All rewards written through this class are multiplied by reward_scale.
    """

    def __init__(
        self,
        capacity: int,
        action_dim: int = 7,
        hparams: PPOHyperParams = PPOHyperParams(),
        reward_scale: float = 0.005,   # e.g., ±100 -> ±0.5
    ):
        self.capacity = capacity
        self.action_dim = action_dim
        self.h = hparams
        self.reward_scale = float(reward_scale)
        self.clear()

    # -----------------------
    # Lifecycle / settings
    # -----------------------

    def clear(self):
        self.states: List[np.ndarray] = []       # [T, 6,7] or [T, 2,6,7]
        self.actions: List[int] = []             # [T]
        self.logprobs: List[float] = []          # [T]
        self.values: List[float] = []            # [T]
        self.rewards: List[float] = []           # [T]  (scaled)
        self.dones: List[bool] = []              # [T]
        self.legal_masks: List[np.ndarray] = []  # [T, action_dim] bool
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None
        self.group_ids: list[int] = []      # one id per agent decision
        self.is_primary: list[bool] = []    # True for original (temporal link)
        self.signs: list[int] = []          # +1 for same-role, -1 for role-swapped
        self._next_group_id: int = 0        # increments per agent decision
        
    def _append_entry(self, s, a, lp, val, r, done, lm, *, gid: int, primary: bool, sign: int):
        self.states.append(s.astype(np.float32, copy=False))
        self.actions.append(int(a))
        self.logprobs.append(float(lp.item()))
        self.values.append(float(val.item()))
        self.rewards.append(float(r) * self.reward_scale)
        self.dones.append(bool(done))
        self.legal_masks.append(np.asarray(lm, dtype=bool))
        self.group_ids.append(int(gid))
        self.is_primary.append(bool(primary))
        self.signs.append(int(sign))
        return len(self.rewards) - 1


    def set_reward_scale(self, s: Union[int, float]):
        self.reward_scale = float(s)

    def __len__(self):
        return len(self.rewards)

    # -----------------------
    # Utilities: transforms
    # -----------------------

    def _hflip_state(self, s: np.ndarray) -> np.ndarray:
        """
        (6,7), (2,6,7) or (4,6,7) -> same shape, flipped horizontally.
        
        For 4-channel states [me, opp, row, col]:
          - reverse columns
          - negate the col-plane so coords follow the flip.
        """
        s = np.asarray(s)
        if s.ndim == 3 and s.shape[1:] == (6, 7) and s.shape[0] >= 4:
            sm = s[:, :, ::-1].copy()
            # channel 3 is col-plane
            sm[3] = -sm[3]
            return sm
        # scalar or 2-plane: just reverse columns
        return s[..., ::-1].copy()


    def _colorswap_state(self, s: np.ndarray) -> np.ndarray:
        """
        Swap agent/opponent channels or invert scalar board.
        For 4-channel states, assume [me, opp, row, col]:
          - swap channels 0 and 1
          - keep row/col as-is.
        """
        s = np.asarray(s)
        if s.ndim == 2 and s.shape == (6, 7):
            return (-s).copy()
        if s.ndim == 3:
            if s.shape[0] == 2:
                return s[[1, 0], :, :].copy()
            if s.shape[0] >= 4:
                out = s.copy()
                out[[0, 1]] = out[[1, 0]]
                return out
        raise ValueError(f"Unexpected state shape for colorswap: {s.shape}")


    def _hflip_action(self, a: int) -> int:
        return (self.action_dim - 1) - int(a)

    def _hflip_mask(self, lm: np.ndarray) -> np.ndarray:
        return lm[::-1].copy()

    # -----------------------
    # Inserts / mutations
    # -----------------------

    def add(
        self,
        state_np: Union[np.ndarray, torch.Tensor],
        action: int,
        logprob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
        legal_mask_bool: Union[np.ndarray, torch.Tensor],
    ):
        """
        Add a single (non-augmented) transition.
        Creates a new augmentation group with exactly one *primary* entry.
        """
        if len(self) >= self.capacity:
            raise RuntimeError("PPOBuffer capacity exceeded")
    
        # --- ensure augmentation bookkeeping exists (in case clear() wasn't patched yet) ---
        if not hasattr(self, "_next_group_id"):
            self.group_ids = []
            self.is_primary = []
            self.signs = []
            self._next_group_id = 0
    
        # --- normalize inputs ---
        if isinstance(state_np, torch.Tensor):
            s = state_np.detach().cpu().numpy()
        else:
            s = np.asarray(state_np)
        assert s.ndim in (2, 3), f"state must be (6,7) or (2,6,7); got {s.shape}"
        s = s.astype(np.float32, copy=False)
    
        if isinstance(legal_mask_bool, torch.Tensor):
            lm = legal_mask_bool.detach().cpu().numpy().astype(bool)
        else:
            lm = np.asarray(legal_mask_bool, dtype=bool)
        assert lm.shape == (self.action_dim,), f"legal_mask must be shape ({self.action_dim},)"
    
        # --- group bookkeeping ---
        gid = int(self._next_group_id)   # one real decision per group
        self._next_group_id += 1
    
        # --- append ---
        self.states.append(s)
        self.actions.append(int(action))
        self.logprobs.append(float(logprob.item()))
        self.values.append(float(value.item()))
        self.rewards.append(float(reward) * self.reward_scale)
        self.dones.append(bool(done))
        self.legal_masks.append(lm)
        self.group_ids.append(gid)
        self.is_primary.append(True)     # this is the temporal link
        self.signs.append(+1)            # same-role
    
        # (no return)


    def add_augmented(
        self,
        state_np: Union[np.ndarray, torch.Tensor],
        action: int,
        logprob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
        legal_mask_bool: Union[np.ndarray, torch.Tensor],
        *,
        hflip: bool = False,
        colorswap: bool = False,
        include_original: bool = True,
        recompute_fn: Callable[[np.ndarray, int, np.ndarray], Tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> List[Tuple[int, int]]:
        """
        Add original and/or augmented copies (H-flip, color-swap) **as one group**.
    
        Returns
        -------
        list[(idx, sign)]
            sign=+1 for same-role (original, hflip), sign=-1 for role-swapped
            (colorswap, hflip+colorswap). Use with add_penalty_to_entries(...).
    
        Note
        ----
        - We DO NOT call self.add() here, because that would create multiple groups.
        - If `recompute_fn` is provided, we use it to recompute (logprob, value)
          for each variant under the behavior policy; otherwise we reuse the given
          tensors (less correct for PPO, but safe fallback).
        """
        if len(self) >= self.capacity:
            raise RuntimeError("PPOBuffer capacity exceeded")
    
        # --- ensure augmentation bookkeeping exists (in case clear() wasn't patched yet) ---
        if not hasattr(self, "_next_group_id"):
            self.group_ids = []
            self.is_primary = []
            self.signs = []
            self._next_group_id = 0
    
        # --- normalize base state/mask ---
        if isinstance(state_np, torch.Tensor):
            base_state = state_np.detach().cpu().numpy()
        else:
            base_state = np.asarray(state_np)
        assert base_state.ndim in (2, 3), f"state must be (6,7) or (2,6,7); got {base_state.shape}"
        base_state = base_state.astype(np.float32, copy=False)
    
        if isinstance(legal_mask_bool, torch.Tensor):
            base_mask = legal_mask_bool.detach().cpu().numpy().astype(bool)
        else:
            base_mask = np.asarray(legal_mask_bool, dtype=bool)
        assert base_mask.shape == (self.action_dim,), f"legal_mask must be shape ({self.action_dim},)"
    
        # --- construct all variants for this ONE group ---
        gid = int(self._next_group_id)   # one real decision = one group
        self._next_group_id += 1
    
        variants: List[Tuple[np.ndarray, int, np.ndarray, int, bool]] = []

        if include_original:
            variants.append((base_state, int(action), base_mask, +1, True))  # primary
        
        if hflip:
            s = self._hflip_state(base_state)
            a = self._hflip_action(action)
            m = self._hflip_mask(base_mask)
            variants.append((s, a, m, +1, False))
        
        if colorswap:
            s = self._colorswap_state(base_state)
            a = int(action)        # same column in mirrored roles
            m = base_mask.copy()
            variants.append((s, a, m, -1, False))
        
        if hflip and colorswap:
            cs = self._colorswap_state(base_state)
            s = self._hflip_state(cs)
            a = self._hflip_action(action)
            m = self._hflip_mask(base_mask)
            variants.append((s, a, m, -1, False))
    
        if len(self) + len(variants) > self.capacity:
            raise RuntimeError("PPOBuffer capacity exceeded by augmented inserts")
    
        out: List[Tuple[int, int]] = []
        for s, a, m, sign, primary in variants:
            # recompute behavior logprob/value if provided
            if recompute_fn is not None:
                lp, val = recompute_fn(s, a, m)
            else:
                lp, val = logprob, value
    
            # append entry
            self.states.append(s.astype(np.float32, copy=False))
            self.actions.append(int(a))
            self.logprobs.append(float(lp.item()))
            self.values.append(float(val.item()))
            self.rewards.append(float(reward) * float(sign) * self.reward_scale)
            self.dones.append(bool(done))
            self.legal_masks.append(np.asarray(m, dtype=bool))
            self.group_ids.append(gid)
            self.is_primary.append(bool(primary))
            self.signs.append(int(sign))
    
            out.append((len(self.rewards) - 1, int(sign)))
    
        return out

    def add_to_reward(self, idx: int, delta_raw: float, *, scale_raw: bool = True):
        """Attribute extra reward/penalty to a single step."""
        if 0 <= idx < len(self.rewards):
            self.rewards[idx] += float(delta_raw) * (self.reward_scale if scale_raw else 1.0)

    def add_penalty_to_entries(self, entries: List[Tuple[int, int]], delta_raw: float, *, scale_raw: bool = True):
        """
        Attribute terminal penalty/bonus to multiple entries with signs produced by add_augmented().
        Each entry is (idx, sign); the delta is multiplied by sign.
        """
        for idx, sign in entries or []:
            if 0 <= idx < len(self.rewards):
                self.rewards[idx] += float(delta_raw) * sign * (self.reward_scale if scale_raw else 1.0)

    # -----------------------
    # GAE / tensors
    # -----------------------

    @torch.no_grad()
    def compute_gae(self, last_value: float = 0.0, last_done: bool = True):
        T = len(self.rewards)
        if T == 0:
            self.advantages = torch.empty(0, device=DEVICE)
            self.returns = torch.empty(0, device=DEVICE)
            return
    
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=DEVICE)
        values  = torch.tensor(self.values,  dtype=torch.float32, device=DEVICE)
        dones   = torch.tensor(self.dones,   dtype=torch.float32, device=DEVICE)
        signs   = torch.tensor(self.signs,   dtype=torch.float32, device=DEVICE)
    
        # indices of true temporal steps (original agent decisions)
        prim_idx = [i for i, p in enumerate(self.is_primary) if p]
        if not prim_idx:
            # degenerate: no primaries; fall back to per-step (r - v)
            adv = rewards - values
            ret = values + adv
            if self.h.normalize_adv and adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            self.advantages, self.returns = adv, ret
            return
    
        r_p = rewards[prim_idx]
        v_p = values[prim_idx]
        d_p = dones[prim_idx]
    
        # next values along primaries
        nv_p = torch.empty_like(v_p)
        nv_p[:-1] = v_p[1:]
        nv_p[-1]  = torch.tensor(float(last_value), device=DEVICE)
        if last_done:
            nv_p[-1] = 0.0
    
        not_done_p = 1.0 - d_p
        deltas_p = r_p + self.h.gamma * not_done_p * nv_p - v_p
    
        adv_p = torch.zeros_like(v_p)
        gae = 0.0
        for k in reversed(range(len(prim_idx))):
            gae = deltas_p[k] + self.h.gamma * self.h.gae_lambda * not_done_p[k] * gae
            adv_p[k] = gae
        ret_p = adv_p + v_p
    
        # scatter to full buffer
        adv = torch.zeros(T, dtype=torch.float32, device=DEVICE)
        ret = torch.zeros(T, dtype=torch.float32, device=DEVICE)
    
        # map group_id -> primary advantage
        # since primaries are one-per-group in insertion order, use their index
        gid_arr = np.asarray(self.group_ids, dtype=np.int64)
        prim_gid = gid_arr[prim_idx]
    
        # fill primaries
        for a_idx, g in zip(adv_p, prim_gid):
            adv[gid_arr == g] = a_idx  # will overwrite for augments next
    
        # apply signs for augments and compute returns per-entry
        adv = adv * signs
        ret = values + adv
    
        if self.h.normalize_adv and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    
        self.advantages = adv
        self.returns = ret


    def _stack_states(self) -> torch.Tensor:
        assert len(self.states) > 0, "No states to stack."
        first = self.states[0]
        if first.ndim == 2:   # [T,6,7]
            arr = np.stack(self.states, axis=0).astype(np.float32, copy=False)
        elif first.ndim == 3: # [T,2,6,7] (or [T,1,6,7])
            arr = np.stack(self.states, axis=0).astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unexpected state ndim {first.ndim}")
        return torch.tensor(arr, dtype=torch.float32, device=DEVICE)

    def get_tensors(self) -> Dict[str, Any]:
        assert self.advantages is not None and self.returns is not None, "Call compute_gae() first."
        states = self._stack_states()
        actions = torch.tensor(self.actions, dtype=torch.long, device=DEVICE)
        old_logprobs = torch.tensor(self.logprobs, dtype=torch.float32, device=DEVICE)
        values = torch.tensor(self.values, dtype=torch.float32, device=DEVICE)
        legal_masks = torch.tensor(np.stack(self.legal_masks, axis=0), dtype=torch.bool, device=DEVICE)
        return {
            "states": states,
            "actions": actions,
            "old_logprobs": old_logprobs,
            "values": values,
            "advantages": self.advantages,
            "returns": self.returns,
            "legal_masks": legal_masks,
        }

    def iter_minibatches(self, batch_size: int, shuffle: bool = True):
        T = len(self.rewards)
        idx = np.arange(T)
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, T, batch_size):
            yield idx[start:start + batch_size]
