# actor_critic.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    """
    Two-channel Connect4 encoder (agent-centric):
      - channel 0: agent (+1) pieces
      - channel 1: opponent (-1) pieces

    Accepts single or batched inputs as ANY of:
      (6,7), (2,6,7), (1,6,7), (B,6,7), (B,1,6,7), (B,2,6,7)
    Values may be {-1,0,+1} (board) or {0,1} planes (already encoded).
    """
    def __init__(self, action_dim: int = 7):
        super().__init__()

        # trunk expects 2 input channels
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)   # -> (6x7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, padding=1)  # -> (5x6)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 2, 6, 7)
            n_flat = self._forward_conv(dummy).view(1, -1).size(1)

        self.fc = nn.Linear(n_flat, 128)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head  = nn.Linear(128, 1)

    # ---------- shape/encoding helpers ----------

    @staticmethod
    def _planes_from_scalar_board(s: torch.Tensor) -> torch.Tensor:
        """s: [..., 6, 7] with values in {-1,0,+1} -> returns two planes (agent, opp)."""
        agent = (s > 0.5).float()
        opp   = (s < -0.5).float()
        return agent, opp

    @staticmethod
    def _to_two_channel(x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input to [B, 2, 6, 7] on DEVICE.
        Handles:
          (6,7)               -> [1,2,6,7]
          (2,6,7)             -> [1,2,6,7]  (already planes)
          (1,6,7)             -> [1,2,6,7]  (scalar board)
          (B,6,7)             -> [B,2,6,7]
          (B,1,6,7)           -> [B,2,6,7]
          (B,2,6,7)           -> [B,2,6,7]
        """
        x = x.to(dtype=torch.float32, device=DEVICE)

        if x.dim() == 2:  # (6,7) scalar board
            a, o = ActorCritic._planes_from_scalar_board(x)
            return torch.stack([a, o], dim=0).unsqueeze(0)  # [1,2,6,7]

        if x.dim() == 3:
            # Ambiguous: could be (2,6,7) planes OR (B,6,7) scalar boards.
            if x.size(0) in (1, 2) and x.size(1) == 6 and x.size(2) == 7:
                # treat as channel-first single example: (C,6,7)
                C = x.size(0)
                if C == 2:                       # already two planes
                    return x.unsqueeze(0)        # [1,2,6,7]
                else:                            # C == 1 -> scalar board
                    s = x[0]
                    a, o = ActorCritic._planes_from_scalar_board(s)
                    return torch.stack([a, o], dim=0).unsqueeze(0)
            else:
                # treat as batch of scalar boards: (B,6,7)
                s = x
                a = (s > 0.5).float()
                o = (s < -0.5).float()
                return torch.stack([a, o], dim=1)  # [B,2,6,7]

        if x.dim() == 4:  # (B,C,6,7)
            C = x.size(1)
            if C == 2:                 # already planes
                return x
            if C == 1:                 # scalar board in channel dim
                s = x[:, 0, :, :]
                a, o = ActorCritic._planes_from_scalar_board(s)
                return torch.stack([a, o], dim=1)
            raise ValueError(f"Unexpected channel count {C}; expected 1 or 2.")

        raise ValueError(f"Unexpected input shape {tuple(x.shape)}")

    # ---------- core forward ----------

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

    def forward(self, x: torch.Tensor):
        x = self._to_two_channel(x)      # [B,2,6,7]
        x = self._forward_conv(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        logits = self.policy_head(x)              # [B, A]
        value  = self.value_head(x).squeeze(-1)   # [B]
        return logits, value

    # ---------- action helpers ----------

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
    def act(self, state_np: np.ndarray | torch.Tensor, legal_actions, temperature: float = 1.0):
        """
        Single-step action.
        - state_np can be any accepted shape (see class docstring).
        - legal_actions is an iterable of legal column indices.
        """
        state_t = torch.from_numpy(state_np) if isinstance(state_np, np.ndarray) else state_np
        logits, value = self.forward(state_t)     # shapes [1,A], [1]

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
                logprob,           # detached by no_grad
                value,             # [1]
                probs)

    def greedy_act(self, state_np: np.ndarray, legal_actions):
        return self.act(state_np, legal_actions, temperature=0.0)

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor,
                         legal_action_masks: torch.Tensor | None = None):
        """
        Batched evaluation for PPO loss.
        - states may be any accepted shape with a batch dimension.
        - legal_action_masks: [B, A] (True where legal)
        """
        logits, values = self.forward(states)
        if legal_action_masks is not None:
            logits = logits.masked_fill(~legal_action_masks, -1e9)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        logprobs = dist.log_prob(actions)
        entropy  = dist.entropy()
        return logprobs, entropy, values
