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
      - Two planes (2,6,7) = [agent(+1) plane, opp(-1) plane].

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

    def set_reward_scale(self, s: Union[int, float]):
        self.reward_scale = float(s)

    def __len__(self):
        return len(self.rewards)

    # -----------------------
    # Utilities: transforms
    # -----------------------

    def _hflip_state(self, s: np.ndarray) -> np.ndarray:
        # (6,7) or (2,6,7) -> same, flipped horizontally (last axis)
        return s[..., ::-1].copy()

    def _colorswap_state(self, s: np.ndarray) -> np.ndarray:
        # swap agent/opponent channels or invert scalar board
        if s.ndim == 3 and s.shape[0] == 2:
            return s[[1, 0], :, :].copy()
        return (-s).copy()  # scalar board

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
        Add a single transition (no augmentation).
        """
        if len(self) >= self.capacity:
            raise RuntimeError("PPOBuffer capacity exceeded")

        # store state as float32 np array, preserving rank
        if isinstance(state_np, torch.Tensor):
            s = state_np.detach().cpu().numpy()
        else:
            s = np.asarray(state_np)
        assert s.ndim in (2, 3), f"state must be (6,7) or (2,6,7); got {s.shape}"
        self.states.append(s.astype(np.float32, copy=False))

        self.actions.append(int(action))
        self.logprobs.append(float(logprob.item()))
        self.values.append(float(value.item()))

        # scale reward on write
        self.rewards.append(float(reward) * self.reward_scale)
        self.dones.append(bool(done))

        if isinstance(legal_mask_bool, torch.Tensor):
            lm = legal_mask_bool.detach().cpu().numpy().astype(bool)
        else:
            lm = np.asarray(legal_mask_bool, dtype=bool)
        assert lm.shape == (self.action_dim,), f"legal_mask must be shape ({self.action_dim},)"
        self.legal_masks.append(lm)

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
        Add original and/or augmented copies (H-flip, color-swap).

        RETURNS: list of (idx, sign)
          - sign = +1 for same-role variants (original, hflip)
          - sign = -1 for role-swapped variants (colorswap, hflip+colorswap)

        For PPO correctness, pass recompute_fn(state, action, legal_mask)
        that returns (logprob_tensor, value_tensor) under the behavior policy.
        """
        # normalize inputs to numpy
        if isinstance(state_np, torch.Tensor):
            base_state = state_np.detach().cpu().numpy()
        else:
            base_state = np.asarray(state_np)

        if isinstance(legal_mask_bool, torch.Tensor):
            base_mask = legal_mask_bool.detach().cpu().numpy().astype(bool)
        else:
            base_mask = np.asarray(legal_mask_bool, dtype=bool)

        # build variant specs: (state, action, mask, sign)
        variant_specs: List[Tuple[np.ndarray, int, np.ndarray, int]] = []

        if include_original:
            variant_specs.append((base_state, int(action), base_mask, +1))

        if hflip:
            s = self._hflip_state(base_state)
            a = self._hflip_action(action)
            m = self._hflip_mask(base_mask)
            variant_specs.append((s, a, m, +1))

        if colorswap:
            s = self._colorswap_state(base_state)
            a = int(action)          # same column index
            m = base_mask.copy()     # legality unchanged
            variant_specs.append((s, a, m, -1))  # role-swapped -> negate reward

        if hflip and colorswap:
            s = self._hflip_state(self._colorswap_state(base_state))
            a = self._hflip_action(action)
            m = self._hflip_mask(base_mask)
            variant_specs.append((s, a, m, -1))  # role-swapped

        if len(self) + len(variant_specs) > self.capacity:
            raise RuntimeError("PPOBuffer capacity exceeded by augmented inserts")

        out: List[Tuple[int, int]] = []
        for s, a, m, sign in variant_specs:
            if recompute_fn is not None:
                lp, val = recompute_fn(s, a, m)
            else:
                # Fallback (technically off-policy for PPO). Prefer passing recompute_fn.
                lp, val = logprob, value

            # NEGATE reward for role-swapped variants
            self.add(s, a, lp, val, reward * sign, done, m)
            out.append((len(self.rewards) - 1, sign))

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

        next_values = torch.empty_like(values)
        next_values[:-1] = values[1:]
        next_values[-1]  = torch.tensor(float(last_value), device=DEVICE)

        not_done = 1.0 - dones
        if last_done:
            next_values[-1] = 0.0

        deltas = rewards + self.h.gamma * not_done * next_values - values

        adv = torch.zeros(T, dtype=torch.float32, device=DEVICE)
        gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + self.h.gamma * self.h.gae_lambda * not_done[t] * gae
            adv[t] = gae

        ret = adv + values

        if self.h.normalize_adv and T > 1:
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
