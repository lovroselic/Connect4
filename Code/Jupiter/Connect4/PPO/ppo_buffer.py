# -*- coding: utf-8 -*-
"""
ppo_buffer.py — PPO rollout buffer with reward scaling and simple data augmentation
(H-flip and color-swap) suitable for Connect4.

Augmentation usage for PPO:
    buffer.add_augmented(
        state_np, action, logprob, value, reward, done, legal_mask,
        hflip=True, colorswap=True, include_original=True,
        recompute_fn=lambda s,a,lm: policy_logprob_value(s, a, lm)  # you supply this
    )
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

    Perspective & state format:
      - Agent is always +1, opponent -1 (agent-centric).
      - You may store states as scalar boards (6x7, values in {-1,0,+1})
        OR as two planes (2x6x7) = [agent(+1) plane, opp(-1) plane].
      - The buffer preserves what you give it and will stack accordingly
        in get_tensors() to [T,6,7] or [T,2,6,7].

    Reward scaling:
      - Pass reward_scale at construction (default 0.02 recommended).
      - All rewards/penalties written through this class are multiplied by reward_scale.
    """

    def __init__(
        self,
        capacity: int,
        action_dim: int = 7,
        hparams: PPOHyperParams = PPOHyperParams(),
        reward_scale: float = 0.02,   # global scale (±100 -> ±2)
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

        state_np: (6,7) OR (2,6,7)
        action:   int in [0, action_dim)
        logprob:  tensor scalar from behavior policy at this state/action
        value:    tensor scalar (network value head at this state)
        reward:   raw env reward (will be scaled)
        done:     episode ended after this agent move/opponent response
        legal_mask_bool: [action_dim] bool array/tensor (True for legal)
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
    ):
        """
        Add original and/or augmented copies (H-flip, color-swap).
        For PPO correctness, pass `recompute_fn` that returns (logprob, value)
        for a given (state, action, legal_mask) under the behavior policy.

        recompute_fn signature:
            (state_np, action_int, legal_mask_bool_np) -> (logprob_tensor, value_tensor)
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

        variants: List[Tuple[np.ndarray, int, np.ndarray]] = []

        if include_original:
            variants.append((base_state, int(action), base_mask))

        if hflip:
            s = self._hflip_state(base_state)
            a = self._hflip_action(action)
            m = self._hflip_mask(base_mask)
            variants.append((s, a, m))

        if colorswap:
            s = self._colorswap_state(base_state)
            a = int(action)          # same column index
            m = base_mask.copy()     # legality unchanged
            variants.append((s, a, m))

        if hflip and colorswap:
            s = self._hflip_state(self._colorswap_state(base_state))
            a = self._hflip_action(action)
            m = self._hflip_mask(base_mask)
            variants.append((s, a, m))

        if len(self) + len(variants) > self.capacity:
            raise RuntimeError("PPOBuffer capacity exceeded by augmented inserts")

        for s, a, m in variants:
            if recompute_fn is not None:
                lp, val = recompute_fn(s, a, m)
            else:
                # Fallback (technically off-policy for PPO). Prefer passing recompute_fn.
                lp, val = logprob, value
            self.add(s, a, lp, val, reward, done, m)

    def add_to_reward(self, idx: int, delta_raw: float, *, scale_raw: bool = True):
        """Attribute extra reward/penalty to an existing step (e.g., blame last agent move)."""
        if 0 <= idx < len(self.rewards):
            self.rewards[idx] += float(delta_raw) * (self.reward_scale if scale_raw else 1.0)

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
        if first.ndim == 2:   # [6,7]
            arr = np.stack(self.states, axis=0).astype(np.float32, copy=False)
        elif first.ndim == 3: # [C,6,7]
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
