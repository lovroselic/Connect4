# -*- coding: utf-8 -*-
"""
ppo_buffer.py — PPO rollout buffer with reward scaling and data augmentation
(H-flip and color-swap) for Connect4 with fixed roles:
- "Agent" is always +1, "Opponent" is always -1 in agent-centric encodings.

Color-swap is a ROLE SWAP augmentation:
- swap agent/opponent planes OR invert scalar board.
- reward is NEGATED for role-swapped variants.

Augmentation grouping:
- One real agent decision = one group_id.
- Exactly one primary entry per group (temporal link for GAE).
- Augmented entries share the primary's GAE advantage, multiplied by sign.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class PPOHyperParams:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_adv: bool = True


RecomputeFn = Callable[[np.ndarray, int, np.ndarray, int], Tuple[torch.Tensor, torch.Tensor]]
# returns (logprob_scalar_tensor, value_scalar_tensor)


class PPOBuffer:
    """
    On-policy rollout storage for PPO.

    Stores AGENT turns only (i.e., insert only when the training agent acts).

    State formats supported per entry:
      - Scalar POV board: (6,7) values in {-1,0,+1}
      - Planes: (2,6,7) = [agent, opp]
      - Planes: (4,6,7) = [agent, opp, row, col] (row/col assumed compatible with flip)
      - Also tolerates (1,6,7)

    Reward scaling:
      - stored_reward = raw_reward * reward_scale * sign
    """

    def __init__(
        self,
        capacity: int,
        action_dim: int = 7,
        hparams: PPOHyperParams = PPOHyperParams(),
        reward_scale: float = 0.005,
        device: torch.device | None = None,
    ):
        self.capacity = int(capacity)
        self.action_dim = int(action_dim)
        self.h = hparams
        self.reward_scale = float(reward_scale)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clear()

    # -----------------------
    # Lifecycle / settings
    # -----------------------

    def clear(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.logprobs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []   # scaled
        self.dones: List[bool] = []
        self.legal_masks: List[np.ndarray] = []  # (7,) bool

        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None

        # augmentation / grouping
        self.group_ids: List[int] = []
        self.is_primary: List[bool] = []
        self.signs: List[int] = []          # +1 same-role, -1 role-swapped
        self._next_group_id: int = 0

        # ply indices (per entry)
        self.ply_indices: List[int] = []

    def set_reward_scale(self, s: Union[int, float]) -> None:
        self.reward_scale = float(s)

    def __len__(self) -> int:
        return len(self.rewards)

    # -----------------------
    # Utilities: transforms
    # -----------------------

    def _hflip_state(self, s: np.ndarray) -> np.ndarray:
        """
        Flip horizontally on the column axis.

        Supports:
          - (6,7) scalar
          - (C,6,7) with C in {1,2,4,...}
        For 4-channel [me, opp, row, col]:
          - reverse columns
          - negate col-plane (assumes signed col encoding around center)
        """
        s = np.asarray(s)
        if s.ndim == 3 and s.shape[1:] == (6, 7) and s.shape[0] >= 4:
            sm = s[:, :, ::-1].copy()
            sm[3] = -sm[3]
            return sm
        return s[..., ::-1].copy()

    def _colorswap_state(self, s: np.ndarray) -> np.ndarray:
        """
        Role swap:
          - scalar: invert sign
          - (2,6,7): swap channels 0 and 1
          - (>=4,6,7): swap channels 0 and 1, keep others
        """
        s = np.asarray(s)
        if s.ndim == 2 and s.shape == (6, 7):
            return (-s).copy()
        if s.ndim == 3 and s.shape[1:] == (6, 7):
            C = s.shape[0]
            if C == 1:
                return (-s).copy()
            if C == 2:
                return s[[1, 0], :, :].copy()
            if C >= 4:
                out = s.copy()
                out[[0, 1]] = out[[1, 0]]
                return out
        raise ValueError(f"Unexpected state shape for colorswap: {s.shape}")

    def _hflip_action(self, a: int) -> int:
        return (self.action_dim - 1) - int(a)

    def _hflip_mask(self, lm: np.ndarray) -> np.ndarray:
        return np.asarray(lm, dtype=bool)[::-1].copy()

    # -----------------------
    # Inserts / mutations
    # -----------------------

    def add(
        self,
        state_np: Union[np.ndarray, torch.Tensor],
        action: int,
        logprob: Union[torch.Tensor, float],
        value: Union[torch.Tensor, float],
        reward: float,
        done: bool,
        legal_mask_bool: Union[np.ndarray, torch.Tensor],
        ply_idx: int | None = None,
    ) -> None:
        """Add a single (non-augmented) transition as its own group (primary entry)."""
        if len(self) >= self.capacity:
            raise RuntimeError("PPOBuffer capacity exceeded")

        s = state_np.detach().cpu().numpy() if isinstance(state_np, torch.Tensor) else np.asarray(state_np)
        s = s.astype(np.float32, copy=False)
        if s.ndim not in (2, 3):
            raise ValueError(f"state must be (6,7) or (C,6,7); got {s.shape}")

        lm = legal_mask_bool.detach().cpu().numpy().astype(bool) if isinstance(legal_mask_bool, torch.Tensor) else np.asarray(legal_mask_bool, dtype=bool)
        if lm.shape != (self.action_dim,):
            raise ValueError(f"legal_mask must be shape ({self.action_dim},), got {lm.shape}")

        lp = float(logprob.item()) if isinstance(logprob, torch.Tensor) else float(logprob)
        val = float(value.item()) if isinstance(value, torch.Tensor) else float(value)

        gid = int(self._next_group_id)
        self._next_group_id += 1

        self.states.append(s)
        self.actions.append(int(action))
        self.logprobs.append(lp)
        self.values.append(val)
        self.rewards.append(float(reward) * self.reward_scale)  # sign=+1
        self.dones.append(bool(done))
        self.legal_masks.append(lm)

        self.group_ids.append(gid)
        self.is_primary.append(True)
        self.signs.append(+1)
        self.ply_indices.append(int(ply_idx) if ply_idx is not None else -1)

    def add_augmented(
         self,
         state_np: Union[np.ndarray, torch.Tensor],
         action: int,
         logprob: Union[torch.Tensor, float],
         value: Union[torch.Tensor, float],
         reward: float,
         done: bool,
         legal_mask_bool: Union[np.ndarray, torch.Tensor],
         *,
         hflip: bool = False,
         colorswap: bool = False,
         include_original: bool = True,
         recompute_fn: RecomputeFn | None = None,
         ply_idx: int | None = None,
     ) -> List[Tuple[int, int]]:
         """
         Add original and/or augmented copies (H-flip, color-swap) as ONE group.
 
         Returns [(idx, sign), ...].  NOTE: for pure symmetry augmentation
         we keep sign = +1 for *all* variants so they share the same advantage.
         """
         # normalize base state/mask
         base_state = state_np.detach().cpu().numpy() if isinstance(state_np, torch.Tensor) else np.asarray(state_np)
         base_state = base_state.astype(np.float32, copy=False)
         if base_state.ndim not in (2, 3):
             raise ValueError(f"state must be (6,7) or (C,6,7); got {base_state.shape}")
 
         base_mask = legal_mask_bool.detach().cpu().numpy().astype(bool) if isinstance(legal_mask_bool, torch.Tensor) else np.asarray(legal_mask_bool, dtype=bool)
         if base_mask.shape != (self.action_dim,):
             raise ValueError(f"legal_mask must be shape ({self.action_dim},), got {base_mask.shape}")
 
         lp0 = torch.as_tensor(logprob, dtype=torch.float32, device=self.device) if not isinstance(logprob, torch.Tensor) else logprob.to(self.device)
         v0  = torch.as_tensor(value,   dtype=torch.float32, device=self.device) if not isinstance(value, torch.Tensor)   else value.to(self.device)
 
         gid = int(self._next_group_id)
         self._next_group_id += 1
 
         stored_ply = int(ply_idx) if ply_idx is not None else -1
 
         # (state, action, mask, sign, is_primary)
         variants: List[Tuple[np.ndarray, int, np.ndarray, int, bool]] = []
         if include_original:
             variants.append((base_state, int(action), base_mask, +1, True))
 
         if hflip:
             variants.append(
                 (
                     self._hflip_state(base_state),
                     self._hflip_action(action),
                     self._hflip_mask(base_mask),
                     +1,      # same-sign symmetry
                     False,
                 )
             )
 
         if colorswap:
             variants.append(
                 (
                     self._colorswap_state(base_state),
                     int(action),
                     base_mask.copy(),
                     +1,      # same-sign symmetry
                     False,
                 )
             )
 
         if hflip and colorswap:
             cs = self._colorswap_state(base_state)
             variants.append(
                 (
                     self._hflip_state(cs),
                     self._hflip_action(action),
                     self._hflip_mask(base_mask),
                     +1,      # same-sign symmetry
                     False,
                 )
             )
 
         if len(self) + len(variants) > self.capacity:
             raise RuntimeError("PPOBuffer capacity exceeded by augmented inserts")
 
         out: List[Tuple[int, int]] = []
 
         for s, a, m, sign, primary in variants:
             if recompute_fn is not None:
                 lp_t, v_t = recompute_fn(s, int(a), np.asarray(m, dtype=bool), stored_ply)
                 lp = float(lp_t.item())
                 val = float(v_t.item())
             else:
                 # fallback: reuse originals
                 lp = float(lp0.item())
                 val = float(v0.item())
 
             self.states.append(np.asarray(s, dtype=np.float32))
             self.actions.append(int(a))
             self.logprobs.append(lp)
             self.values.append(val)
             self.rewards.append(float(reward) * float(sign) * self.reward_scale)
             self.dones.append(bool(done))
             self.legal_masks.append(np.asarray(m, dtype=bool))
 
             self.group_ids.append(gid)
             self.is_primary.append(bool(primary))
             self.signs.append(int(sign))  # now always +1
             self.ply_indices.append(stored_ply)
 
             out.append((len(self.rewards) - 1, int(sign)))
 
         return out


    def add_to_reward(self, idx: int, delta_raw: float, *, scale_raw: bool = True) -> None:
        if 0 <= idx < len(self.rewards):
            self.rewards[idx] += float(delta_raw) * (self.reward_scale if scale_raw else 1.0)

    def add_penalty_to_entries(self, entries: List[Tuple[int, int]], delta_raw: float, *, scale_raw: bool = True) -> None:
        for idx, sign in entries or []:
            if 0 <= idx < len(self.rewards):
                self.rewards[idx] += float(delta_raw) * float(sign) * (self.reward_scale if scale_raw else 1.0)

    # -----------------------
    # GAE / tensors
    # -----------------------

    @torch.no_grad()
    def compute_gae(self, last_value: float = 0.0, last_done: bool = True) -> None:
        T = len(self.rewards)
        if T == 0:
            self.advantages = torch.empty(0, device=self.device)
            self.returns = torch.empty(0, device=self.device)
            return

        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        values  = torch.tensor(self.values,  dtype=torch.float32, device=self.device)
        dones   = torch.tensor(self.dones,   dtype=torch.float32, device=self.device)
        signs   = torch.tensor(self.signs,   dtype=torch.float32, device=self.device)

        prim_idx = [i for i, p in enumerate(self.is_primary) if p]
        if not prim_idx:
            adv = rewards - values
            ret = values + adv
            if self.h.normalize_adv and adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            self.advantages, self.returns = adv, ret
            return

        r_p = rewards[prim_idx]
        v_p = values[prim_idx]
        d_p = dones[prim_idx]

        nv_p = torch.empty_like(v_p)
        nv_p[:-1] = v_p[1:]
        nv_p[-1]  = torch.tensor(float(last_value), device=self.device)
        if last_done:
            nv_p[-1] = 0.0

        not_done_p = 1.0 - d_p
        deltas_p = r_p + self.h.gamma * not_done_p * nv_p - v_p

        adv_p = torch.zeros_like(v_p)
        gae = 0.0
        for k in reversed(range(len(prim_idx))):
            gae = deltas_p[k] + self.h.gamma * self.h.gae_lambda * not_done_p[k] * gae
            adv_p[k] = gae

        # scatter primary adv to all members of the group, then apply signs per entry
        gid_t = torch.tensor(self.group_ids, dtype=torch.long, device=self.device)
        prim_gid = gid_t[torch.tensor(prim_idx, dtype=torch.long, device=self.device)]

        adv = torch.zeros(T, dtype=torch.float32, device=self.device)
        for k in range(len(prim_gid)):
            g = prim_gid[k]
            adv[gid_t == g] = adv_p[k]

        adv = adv * signs
        ret = values + adv

        if self.h.normalize_adv and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        self.advantages = adv
        self.returns = ret

    def _stack_states(self) -> torch.Tensor:
        if not self.states:
            raise RuntimeError("No states to stack.")
        arr = np.stack(self.states, axis=0).astype(np.float32, copy=False)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def get_tensors(self) -> Dict[str, Any]:
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Call compute_gae() first.")

        states = self._stack_states()
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(self.logprobs, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.values, dtype=torch.float32, device=self.device)
        legal_masks = torch.tensor(np.stack(self.legal_masks, axis=0), dtype=torch.bool, device=self.device)
        ply_indices = torch.tensor(self.ply_indices, dtype=torch.long, device=self.device)

        return {
            "states": states,
            "actions": actions,
            "old_logprobs": old_logprobs,
            "values": values,
            "advantages": self.advantages,
            "returns": self.returns,
            "legal_masks": legal_masks,
            "ply_indices": ply_indices,
        }

    def iter_minibatches(self, batch_size: int, shuffle: bool = True):
        T = len(self.rewards)
        idx = np.arange(T)
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, T, batch_size):
            yield idx[start:start + batch_size]
