# PPO/ppo_hall_of_fame.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import os

import numpy as np
import torch
import torch.nn as nn

# Primary teacher model format (supervised)
from C4.CNet192 import load_cnet192

NEG_INF = -1e9
CENTER_COL = 3


def _argmax_legal_center_tiebreak(logits: Union[np.ndarray, torch.Tensor], legal_actions: Sequence[int]) -> int:
    """
    Greedy argmax over legal actions with deterministic center tie-break.
    If multiple actions share the same max logit, pick the one closest to CENTER_COL.
    """
    legal = [int(a) for a in legal_actions]
    if not legal:
        raise ValueError("No legal actions.")

    if isinstance(logits, torch.Tensor):
        vals_all = logits.detach().float().cpu().numpy()
    else:
        vals_all = np.asarray(logits, dtype=np.float32)

    m = float(max(vals_all[a] for a in legal))
    # tolerance for near-ties
    tie = [a for a in legal if abs(float(vals_all[a]) - m) <= 1e-8]

    if len(tie) == 1:
        return int(tie[0])

    tie.sort(key=lambda a: (abs(a - CENTER_COL), a))
    return int(tie[0])


def _state_to_1ch_tensor(state: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Normalize state to (B,1,6,7) float tensor.

    Accepts:
      - (6,7)                scalar board
      - (1,6,7)              already 1ch
      - (2,6,7)              [me, opp] -> (me - opp)
      - (4,6,7) or (6,6,7)   takes first two planes as [me, opp] -> (me - opp)
      - batched variants: (B,6,7), (B,1,6,7), (B,2,6,7), ...

    Assumes agent-centric encoding upstream when scalar (me=+1, opp=-1),
    and [me,opp] planes when multi-channel.
    """
    if not isinstance(state, torch.Tensor):
        x = torch.as_tensor(state, dtype=torch.float32, device=device)
    else:
        x = state.to(device=device, dtype=torch.float32)

    if x.dim() == 2:
        # (6,7) -> (1,1,6,7)
        if x.shape != (6, 7):
            raise ValueError(f"Expected (6,7), got {tuple(x.shape)}")
        return x.unsqueeze(0).unsqueeze(0)

    if x.dim() == 3:
        # (C,6,7) or (B,6,7)
        if x.shape[-2:] != (6, 7):
            raise ValueError(f"Expected last dims (6,7), got {tuple(x.shape)}")

        first = int(x.shape[0])

        # Channel-first single sample is the common case in your code: (1,6,7), (2,6,7), (4,6,7), (6,6,7)
        if first in (1, 2, 4, 6) and x.shape[1:] == (6, 7):
            if first == 1:
                return x.unsqueeze(0)  # (1,1,6,7)
            # use first two planes: me - opp
            scalar = x[0] - x[1]
            return scalar.unsqueeze(0).unsqueeze(0)

        # Otherwise interpret as batch: (B,6,7) -> (B,1,6,7)
        return x.unsqueeze(1)

    if x.dim() == 4:
        # (B,C,6,7)
        if x.shape[-2:] != (6, 7):
            raise ValueError(f"Expected last dims (6,7), got {tuple(x.shape)}")

        C = int(x.shape[1])
        if C == 1:
            return x
        if C >= 2:
            scalar = x[:, 0] - x[:, 1]
            return scalar.unsqueeze(1)

        raise ValueError(f"Unsupported channel count: C={C}")

    raise ValueError(f"Unsupported state shape: {tuple(x.shape)}")


class _FrozenPolicyWrapper(nn.Module):
    """
    Wrap a model into an ActorCritic-like opponent API:
      - forward(x) -> (logits[B,7], value[B])
      - act(state, legal_actions, temperature, ply_idx) -> (a, lp, v, info)

    Greedy by default (POP usage).
    """

    def __init__(self, model: nn.Module, device: torch.device, tag: str = "policy"):
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.tag = str(tag)

        self.to(self.device)
        self.model.eval()

        # Truly frozen
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Accepts x as (B,1,6,7). Returns:
          logits: (B,7)
          value : (B,)   (zeros if underlying model has no value head)
        """
        out = self.model(x)

        if isinstance(out, (tuple, list)):
            logits = out[0]
            value = out[1] if len(out) > 1 else None
        else:
            logits = out
            value = None

        if not isinstance(logits, torch.Tensor):
            raise TypeError("Model must return logits Tensor or (logits, value).")

        if logits.dim() != 2 or logits.size(-1) != 7:
            raise ValueError(f"Expected logits shape (B,7), got {tuple(logits.shape)}")

        if value is None:
            value = torch.zeros((logits.size(0),), dtype=logits.dtype, device=logits.device)
        else:
            if not isinstance(value, torch.Tensor):
                raise TypeError("Value must be a Tensor when provided.")
            # normalize to (B,)
            if value.dim() == 2 and value.size(-1) == 1:
                value = value.squeeze(-1)
            if value.dim() == 0:
                value = value.view(1).expand(logits.size(0))
            if value.dim() != 1 or value.size(0) != logits.size(0):
                raise ValueError(f"Expected value shape (B,), got {tuple(value.shape)}")

        return logits, value

    @torch.inference_mode()
    def act(
        self,
        state: np.ndarray,
        legal_actions,
        temperature: float = 1.0,
        ply_idx=None,
    ):
        legal = [int(a) for a in legal_actions]
        if not legal:
            raise ValueError("No legal actions for .act().")

        # single legal move shortcut
        if len(legal) == 1:
            a = int(legal[0])
            lp = torch.zeros((), dtype=torch.float32, device=self.device)
            v = torch.zeros((), dtype=torch.float32, device=self.device)
            return a, lp, v, {"mode": self.tag}

        x = _state_to_1ch_tensor(state, device=self.device)  # (1,1,6,7)
        logits, value = self.forward(x)                      # (1,7), (1,)

        # greedy with center tie-break (temperature ignored for POP)
        a = _argmax_legal_center_tiebreak(logits[0], legal)

        lp = torch.zeros((), dtype=torch.float32, device=self.device)  # opponent doesn't learn
        v = value[0].detach() if value.numel() else torch.zeros((), dtype=torch.float32, device=self.device)

        return int(a), lp, v, {"mode": self.tag}


@dataclass
class HOFMember:
    name: str
    ckpt_path: str
    metascore: Optional[float] = None
    policy: Optional[nn.Module] = None  # lazy-loaded (wrapped)


class PPOEnsemblePolicy(nn.Module):
    """
    Read-only weighted ensemble of frozen policies.

    Exposes:
      - forward(x) -> (logits,value) averaged across members
      - act(state, legal_actions, temperature, ply_idx) -> greedy action
    """

    def __init__(
        self,
        members: Sequence[nn.Module],
        weights: Optional[Sequence[float]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        if not members:
            raise ValueError("PPOEnsemblePolicy needs at least one member.")

        self.members = nn.ModuleList(list(members))
        self.device = torch.device(device)

        if weights is None:
            w = np.ones(len(self.members), dtype=np.float32)
        else:
            w = np.asarray(list(weights), dtype=np.float32)
            if w.shape[0] != len(self.members):
                raise ValueError("weights length must match number of members")

        w = np.maximum(w, 1e-8)
        w = w / w.sum()

        self.register_buffer("weights", torch.from_numpy(w).float())

        self.to(self.device)
        for m in self.members:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,1,6,7)
        logits_list: List[torch.Tensor] = []
        values_list: List[torch.Tensor] = []

        for m in self.members:
            out = m(x)
            if isinstance(out, (tuple, list)):
                l = out[0]
                v = out[1] if len(out) > 1 else None
            else:
                l = out
                v = None

            if v is None:
                v = torch.zeros((l.size(0),), dtype=l.dtype, device=l.device)
            else:
                if v.dim() == 2 and v.size(-1) == 1:
                    v = v.squeeze(-1)

            logits_list.append(l)  # (B,7)
            values_list.append(v)  # (B,)

        logits_stacked = torch.stack(logits_list, dim=0)  # (E,B,7)
        values_stacked = torch.stack(values_list, dim=0)  # (E,B)

        w = self.weights.view(-1, 1, 1)                   # (E,1,1)
        logits = (w * logits_stacked).sum(dim=0)          # (B,7)

        wv = self.weights.view(-1, 1)                     # (E,1)
        values = (wv * values_stacked).sum(dim=0)         # (B,)

        return logits, values

    @torch.inference_mode()
    def act(self, state: np.ndarray, legal_actions, temperature: float = 1.0, ply_idx=None):
        legal = [int(a) for a in legal_actions]
        if not legal:
            raise ValueError("No legal actions for ensemble .act().")

        if len(legal) == 1:
            a = int(legal[0])
            lp = torch.zeros((), dtype=torch.float32, device=self.device)
            v = torch.zeros((), dtype=torch.float32, device=self.device)
            return a, lp, v, {"mode": "ensemble"}

        x = _state_to_1ch_tensor(state, device=self.device)  # (1,1,6,7)
        logits, value = self.forward(x)                      # (1,7), (1,)
        a = _argmax_legal_center_tiebreak(logits[0], legal)

        lp = torch.zeros((), dtype=torch.float32, device=self.device)
        v = value[0].detach() if value.numel() else torch.zeros((), dtype=torch.float32, device=self.device)

        return int(a), lp, v, {"mode": "ensemble"}


class PPOHallOfFame:
    """
    Hall-of-Fame registry for frozen policies.

    Supports:
      - Supervised CNet192 checkpoints (preferred)
      - PPO ActorCritic-style checkpoints (fallback, for later)
    """

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self.members: Dict[str, HOFMember] = {}

    # ---------- Member management ----------

    def add_member(
        self,
        name: str,
        ckpt_path: str,
        metascore: Optional[float] = None,
        policy: Optional[nn.Module] = None,
    ) -> None:
        self.members[name] = HOFMember(
            name=str(name),
            ckpt_path=str(ckpt_path),
            metascore=metascore,
            policy=policy,
        )

    def remove_member(self, name: str) -> None:
        self.members.pop(name, None)

    def list_members(self) -> List[str]:
        return list(self.members.keys())

    def get_member(self, name: str) -> HOFMember:
        if name not in self.members:
            raise KeyError(f"HOF member '{name}' not found.")
        return self.members[name]

    def get_policy(self, name: str) -> nn.Module:
        """Convenience: ensure_loaded + return the wrapped policy."""
        return self.ensure_loaded(name)

    def ensure_all_loaded(self) -> None:
        for n in list(self.members.keys()):
            self.ensure_loaded(n)

    # ---------- Loading ----------

    def _load_supervised_cnet192(self, ckpt_path: str) -> nn.Module:
        model, _ = load_cnet192(path=ckpt_path, device=self.device, strict=True)
        model.eval()
        return _FrozenPolicyWrapper(model, device=self.device, tag="HOF_CNet192")

    def _load_ppo_actor_critic_fallback(self, ckpt_path: str) -> nn.Module:
        """
        Fallback loader for older PPO checkpoints if/when you add them to HOF.
        Tries common keys: policy_state_dict/state_dict/model_state_dict.
        """
        from PPO.actor_critic import ActorCritic  # local import to avoid circulars

        ckpt = torch.load(ckpt_path, map_location=self.device)

        # Try to match architecture if cfg exists
        use_mid = True
        if isinstance(ckpt, dict):
            cfg = ckpt.get("cfg", {}) or {}
            if isinstance(cfg, dict) and "use_mid_3x3" in cfg:
                use_mid = bool(cfg["use_mid_3x3"])

        policy = ActorCritic(use_mid_3x3=use_mid).to(self.device)

        if not isinstance(ckpt, dict):
            raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

        if "policy_state_dict" in ckpt:
            state_dict = ckpt["policy_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            # If it looks like a raw state_dict
            if all(isinstance(k, str) for k in ckpt.keys()) and any("." in k for k in ckpt.keys()):
                state_dict = ckpt
            else:
                raise ValueError(f"Unknown PPO checkpoint format: keys={list(ckpt.keys())}")

        policy.load_state_dict(state_dict, strict=True)
        policy.eval()

        # Wrap to enforce consistent 1ch input handling + greedy POP behavior
        return _FrozenPolicyWrapper(policy, device=self.device, tag="HOF_PPO")

    def _load_policy_from_ckpt(self, ckpt_path: str) -> nn.Module:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # 1) Prefer supervised CNet192 format
        try:
            return self._load_supervised_cnet192(ckpt_path)
        except FileNotFoundError:
            raise
        except (RuntimeError, ValueError, KeyError, TypeError):
            pass

        # 2) Fallback to PPO ActorCritic-style checkpoint
        return self._load_ppo_actor_critic_fallback(ckpt_path)

    def ensure_loaded(self, name: str) -> nn.Module:
        mem = self.get_member(name)
        if mem.policy is None:
            mem.policy = self._load_policy_from_ckpt(mem.ckpt_path)
        return mem.policy

    # ---------- Ensemble ----------

    def build_ensemble(self, names: Sequence[str], use_metascore_weights: bool = True) -> PPOEnsemblePolicy:
        if not names:
            raise ValueError("Cannot build ensemble from empty list.")

        policies: List[nn.Module] = []
        weights: List[float] = []

        for n in names:
            mem = self.get_member(n)
            pol = self.ensure_loaded(n)
            policies.append(pol)

            if use_metascore_weights:
                w = 1.0 if mem.metascore is None else float(mem.metascore)
                weights.append(w)
            else:
                weights.append(1.0)

        ens = PPOEnsemblePolicy(
            members=policies,
            weights=weights if use_metascore_weights else None,
            device=self.device,
        )
        ens.eval()
        return ens


__all__ = [
    "HOFMember",
    "PPOHallOfFame",
    "PPOEnsemblePolicy",
]
