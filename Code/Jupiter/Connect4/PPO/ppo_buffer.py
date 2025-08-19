# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 13:34:44 2025

@author: Lovro
"""

# ppo_buffer.py

from dataclasses import dataclass
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
    Stores only AGENT turns (your training loop should insert only when the agent acts).
    """
    def __init__(self, capacity: int, action_dim: int = 7, hparams: PPOHyperParams = PPOHyperParams()):
        self.capacity = capacity
        self.action_dim = action_dim
        self.h = hparams
        self.clear()

    def clear(self):
        self.states = []              # [T, 6, 7]  (np arrays or torch tensors)
        self.actions = []             # [T]
        self.logprobs = []            # [T]
        self.values = []              # [T]
        self.rewards = []             # [T]
        self.dones = []               # [T]  (True if episode ended immediately after this agent move)
        self.legal_masks = []         # [T, action_dim] bool
        self.advantages = None        # torch tensor [T]
        self.returns = None           # torch tensor [T]

    def __len__(self):
        return len(self.rewards)

    def add(self, state_np, action: int, logprob: torch.Tensor, value: torch.Tensor,
            reward: float, done: bool, legal_mask_bool: np.ndarray | torch.Tensor):
        """
        state_np: np.ndarray shape (6,7)
        action: int in [0, action_dim)
        logprob: tensor scalar (from policy.act with no_grad)
        value:   tensor scalar (from policy.act with no_grad)
        reward:  float
        done:    bool (True if env ended after opponent follows or we won immediately on our move)
        legal_mask_bool: [action_dim] bool array/tensor (True for legal)
        """
        if len(self) >= self.capacity:
            # On-policy buffer should not overflow; you can increase capacity or flush earlier.
            raise RuntimeError("PPOBuffer capacity exceeded")

        self.states.append(np.asarray(state_np, dtype=np.float32))
        self.actions.append(int(action))
        self.logprobs.append(float(logprob.item()))
        self.values.append(float(value.item()))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

        if isinstance(legal_mask_bool, torch.Tensor):
            legal_mask_bool = legal_mask_bool.detach().cpu().numpy().astype(bool)
        else:
            legal_mask_bool = np.asarray(legal_mask_bool, dtype=bool)
        assert legal_mask_bool.shape == (self.action_dim,)
        self.legal_masks.append(legal_mask_bool)

    @torch.no_grad()
    def compute_gae(self, last_value: float = 0.0, last_done: bool = True):
        """
        Compute GAE(λ) advantages and returns.
        last_value: V(s_{T}) for bootstrap if the trajectory segment ended by time truncation (not done).
        last_done:  whether the last transition ended an episode. If True, bootstrap is zeroed.
        """
        T = len(self.rewards)
        if T == 0:
            self.advantages = torch.empty(0, device=DEVICE)
            self.returns = torch.empty(0, device=DEVICE)
            return

        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=DEVICE)
        values  = torch.tensor(self.values,  dtype=torch.float32, device=DEVICE)
        dones   = torch.tensor(self.dones,   dtype=torch.float32, device=DEVICE)

        # Prepare v_{t+1} by shifting values and appending last_value
        next_values = torch.empty_like(values)
        next_values[:-1] = values[1:]
        next_values[-1]  = torch.tensor(float(last_value), device=DEVICE)

        # done masks (1 for done, 0 for not done) affect bootstrapping
        not_done = 1.0 - dones
        if last_done:
            # If the segment ended with an actual terminal, we should not bootstrap at T
            next_values[-1] = 0.0

        deltas = rewards + self.h.gamma * not_done * next_values - values

        adv = torch.zeros(T, dtype=torch.float32, device=DEVICE)
        gae = 0.0
        for t in reversed(range(T)):
            mask = not_done[t]
            gae = deltas[t] + self.h.gamma * self.h.gae_lambda * mask * gae
            adv[t] = gae

        ret = adv + values

        if self.h.normalize_adv and T > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        self.advantages = adv
        self.returns = ret

    def get_tensors(self):
        """
        Stack all fields into tensors on DEVICE.
        Call after compute_gae().
        """
        assert self.advantages is not None and self.returns is not None, "Call compute_gae() first."

        states = torch.tensor(np.stack(self.states, axis=0), dtype=torch.float32, device=DEVICE)  # [T,6,7]
        actions = torch.tensor(self.actions, dtype=torch.long, device=DEVICE)                      # [T]
        old_logprobs = torch.tensor(self.logprobs, dtype=torch.float32, device=DEVICE)            # [T]
        values = torch.tensor(self.values, dtype=torch.float32, device=DEVICE)                    # [T]
        legal_masks = torch.tensor(np.stack(self.legal_masks, axis=0), dtype=torch.bool, device=DEVICE)  # [T,A]
        adv = self.advantages                                                                       # [T]
        ret = self.returns                                                                          # [T]

        return {
            "states": states,
            "actions": actions,
            "old_logprobs": old_logprobs,
            "values": values,
            "advantages": adv,
            "returns": ret,
            "legal_masks": legal_masks,
        }

    def iter_minibatches(self, batch_size: int, shuffle: bool = True):
        """
        Yields index arrays for minibatching.
        """
        T = len(self.rewards)
        idx = np.arange(T)
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, T, batch_size):
            yield idx[start:start + batch_size]
