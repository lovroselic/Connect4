# actor_critic.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, action_dim: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # (6x7) -> (6x7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, padding=1)  # -> (5x6)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 6, 7)
            n_flat = self._forward_conv(dummy).view(1, -1).size(1)

        self.fc = nn.Linear(n_flat, 128)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head  = nn.Linear(128, 1)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:            # [6,7]
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:          # [B,6,7]
            x = x.unsqueeze(1)      # -> [B,1,6,7]
        x = x.to(dtype=torch.float32, device=DEVICE)

        x = self._forward_conv(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))

        logits = self.policy_head(x)              # [B, A]
        value  = self.value_head(x).squeeze(-1)   # [B]
        return logits, value

    @staticmethod
    def make_legal_action_mask(batch_legal_actions, action_dim: int = 7) -> torch.Tensor:
        B = len(batch_legal_actions)
        mask = torch.zeros(B, action_dim, dtype=torch.bool, device=DEVICE)
        for i, acts in enumerate(batch_legal_actions):
            if acts:  # guard against empty list
                idx = torch.as_tensor(list(acts), dtype=torch.long, device=DEVICE)
                mask[i, idx] = True
        return mask

    @torch.no_grad()
    def act(self, state_np: np.ndarray, legal_actions, temperature: float = 1.0):
        logits, value = self.forward(torch.from_numpy(state_np))

        # mask illegal actions
        mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
        if legal_actions:
            mask[0, torch.as_tensor(list(legal_actions), device=logits.device)] = True
        masked_logits = logits.masked_fill(~mask, -1e9)

        if temperature <= 0:
            probs = F.softmax(masked_logits, dim=-1)
            action = masked_logits.argmax(dim=-1)
            dist = Categorical(probs)
        else:
            probs = F.softmax(masked_logits / temperature, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()

        logprob = dist.log_prob(action)
        return (int(action.item()),
                logprob,           # already detached by no_grad
                value,             # [1]
                probs)

    def greedy_act(self, state_np: np.ndarray, legal_actions):
        return self.act(state_np, legal_actions, temperature=0.0)

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor,
                         legal_action_masks: torch.Tensor | None = None):
        logits, values = self.forward(states)
        if legal_action_masks is not None:
            logits = logits.masked_fill(~legal_action_masks, -1e9)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        logprobs = dist.log_prob(actions)
        entropy  = dist.entropy()
        return logprobs, entropy, values
