# PPO/actor_critic.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# Device is resolved once here; model can be moved later with .to(DEVICE)
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEG_INF = -1e9  # large negative for masked logits

class ActorCritic(nn.Module):
    """
    Two-channel Connect-4 encoder (agent-centric):
      - channel 0: agent (+1) pieces
      - channel 1: opponent (-1) pieces

    Accepts single or batched inputs as ANY of:
      (6,7), (2,6,7), (1,6,7), (B,6,7), (B,1,6,7), (B,2,6,7)
    Values may be {-1,0,+1} (board) or {0,1} planes (already encoded).

    Notes:
    - Keep boards agent-centric upstream (agent pieces == +1). If you only have
      an environment-centric board and the current player can be -1, flip with:
        s_agent = ActorCritic.ensure_agent_centric(s_env, current_player)
    """

    def __init__(self, action_dim: int = 7):
        super().__init__()

        # ---- Trunk (expects 2 channels) ----
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)   # -> (6x7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, padding=1)  # -> (5x6)
        self.flatten = nn.Flatten()

        # Infer flattened size robustly
        with torch.no_grad():
            dummy = torch.zeros(1, 2, 6, 7)
            n_flat = self._forward_conv(dummy).view(1, -1).size(1)

        self.fc = nn.Linear(n_flat, 128)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head  = nn.Linear(128, 1)

        self._init_weights()

    # ---------- shape/encoding helpers ----------

    @staticmethod
    def ensure_agent_centric(s: np.ndarray | torch.Tensor, current_player: int):
        """
        Returns a scalar-board (same type/shape) where agent is always +1.
        If current_player == -1, it flips signs.
        """
        if isinstance(s, np.ndarray):
            return s if current_player == 1 else -s
        t = s
        return t if current_player == 1 else -t

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
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
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
        x = self._to_two_channel(x)        # [B,2,6,7]
        x = self._forward_conv(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        logits = self.policy_head(x)                # [B, A]
        value  = self.value_head(x).squeeze(-1)     # [B]
        return logits, value

    # ---------- action helpers ----------

    @staticmethod
    def make_legal_action_mask(batch_legal_actions, action_dim: int = 7) -> torch.Tensor:
        """
        batch_legal_actions: iterable of iterables/sets of ints
        returns: [B, A] bool mask
        """
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
        Robust single-step action:
          - builds a legal-action mask and guarantees at least one legal action
          - avoids NaNs when all logits are masked or temperature==0
          - falls back to uniform over legal actions if probs get NaN
        Returns: (action:int, logprob:Tensor[1], value:Tensor[1], probs:Tensor[1,A])
        """
        logits, value = self.forward(state_np)

        # --- Build mask [1, A] and ensure at least one True ---
        B, A = logits.shape[0], logits.shape[1]
        mask = torch.zeros(B, A, dtype=torch.bool, device=logits.device)
        if legal_actions:
            idx = torch.as_tensor(list(legal_actions), dtype=torch.long, device=logits.device)
            mask[0, idx] = True
        if mask.sum(dim=-1).item() == 0:   # guard: allow all if empty
            mask[:] = True

        masked_logits = logits.masked_fill(~mask, NEG_INF)

        # --- Compute probs safely (avoid NaNs) ---
        if temperature <= 0:
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            probs = torch.exp(log_probs)
        else:
            t = max(float(temperature), 1e-6)
            log_probs = torch.log_softmax(masked_logits / t, dim=-1)
            probs = torch.exp(log_probs)

        if torch.isnan(probs).any():
            legal_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
            uniform = mask.float() / legal_counts
            probs = uniform
            log_probs = torch.log(probs.clamp_min(1e-12))

        # --- Sample / pick action ---
        if temperature <= 0:
            action = masked_logits.argmax(dim=-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()

        # Build dist after final probs to compute logprob consistently
        dist = Categorical(probs)
        logprob = dist.log_prob(action)

        return int(action.item()), logprob, value, probs

    def greedy_act(self, state_np: np.ndarray | torch.Tensor, legal_actions):
        return self.act(state_np, legal_actions, temperature=0.0)

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        legal_action_masks: torch.Tensor | None = None
    ):
        """
        states: [B,6,7], [B,1,6,7], or [B,2,6,7]; we encode internally.
        actions: [B] long
        legal_action_masks: [B, A] bool or None
        Returns: (logprobs[B], entropy[B], values[B])
        """
        logits, values = self.forward(states)

        mask = None
        if legal_action_masks is not None:
            mask = legal_action_masks.to(dtype=torch.bool, device=logits.device)
            # ---- Guard: ensure at least one legal action per row ----
            no_legal = (mask.sum(dim=-1) == 0)
            if no_legal.any():
                mask = mask.clone()
                mask[no_legal] = True  # fall back to all actions legal for those rows
            logits = logits.masked_fill(~mask, NEG_INF)

        probs = F.softmax(logits, dim=-1)

        # ---- Guard: replace any NaN rows with uniform over legal actions ----
        if torch.isnan(probs).any():
            if mask is None:
                probs = torch.full_like(probs, 1.0 / probs.size(-1))
            else:
                with torch.no_grad():
                    legal_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1)
                    uniform = mask.float() / legal_counts
                nan_rows = torch.isnan(probs).any(dim=-1, keepdim=True)
                probs = torch.where(nan_rows, uniform, probs)

        dist = Categorical(probs)
        actions = actions.long()
        logprobs = dist.log_prob(actions)
        entropy  = dist.entropy()
        return logprobs, entropy, values

    # ---------- init ----------

    def _init_weights(self):
        # Trunk: Kaiming for ReLU
        for layer in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc.bias)

        # Heads: Orthogonal for better PPO stability
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)  # small policy logits at start
        nn.init.zeros_(self.policy_head.bias)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)    # value ≈ N(0,1)
        nn.init.zeros_(self.value_head.bias)
