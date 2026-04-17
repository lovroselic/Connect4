"""dqn_hall_of_fame.py

Hall-of-Fame (HOF) + metascore-weighted ensemble for Connect-4 opponents.

Designed for DQN training, but intentionally model-agnostic:
  - accepts supervised CNet192 checkpoints via `load_cnet192()`
  - accepts DQN checkpoints saved by `DQNAgent.save()` (keys: q_state, ...)

All members are frozen and used in a greedy (argmax) manner with a
center-column tie-break.

Input convention:
  - state is a numpy array from *player-to-move* POV
    shape (6,7) or (1,6,7) with values in {-1,0,+1}.

Output convention:
  - models return either:
      logits: (B,7)
    or:
      (logits, value)
    We treat `logits` as Q-values / action scores.

Lovro-friendly philosophy: strict about shapes, flexible about checkpoint formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import os

import numpy as np
import torch
import torch.nn as nn

from C4.CNet192 import load_cnet192


COLS = 7
CENTER_COL = 3
NEG_INF = -1e9


def _state_to_1ch_tensor(state: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Normalize state to (B,1,6,7) float tensor."""
    if not isinstance(state, torch.Tensor):
        x = torch.as_tensor(state, dtype=torch.float32, device=device)
    else:
        x = state.to(device=device, dtype=torch.float32)

    if x.dim() == 2:
        if x.shape != (6, 7):
            raise ValueError(f"Expected (6,7), got {tuple(x.shape)}")
        return x.unsqueeze(0).unsqueeze(0)

    if x.dim() == 3:
        # (1,6,7) OR (B,6,7)
        if x.shape[-2:] != (6, 7):
            raise ValueError(f"Expected last dims (6,7), got {tuple(x.shape)}")

        # common single-sample channel-first (1,6,7)
        if x.shape[0] == 1 and x.shape[1:] == (6, 7):
            return x.unsqueeze(0)

        # batch (B,6,7)
        return x.unsqueeze(1)

    if x.dim() == 4:
        # (B,1,6,7)
        if x.shape[1:] != (1, 6, 7) and x.shape[-2:] == (6, 7):
            # tolerate (B,C,6,7) by taking channel 0
            if x.shape[1] >= 1:
                x = x[:, :1]
        if x.shape[1:] != (1, 6, 7):
            raise ValueError(f"Expected (B,1,6,7), got {tuple(x.shape)}")
        return x

    raise ValueError(f"Unsupported state shape: {tuple(x.shape)}")


def _argmax_legal_center_tiebreak(scores: Union[np.ndarray, torch.Tensor], legal_actions: Sequence[int]) -> int:
    legal = [int(a) for a in legal_actions]
    if not legal:
        raise ValueError("No legal actions")

    if isinstance(scores, torch.Tensor):
        vals = scores.detach().float().cpu().numpy()
    else:
        vals = np.asarray(scores, dtype=np.float32)

    m = float(max(vals[a] for a in legal))
    tied = [a for a in legal if abs(float(vals[a]) - m) <= 1e-8]
    if len(tied) == 1:
        return int(tied[0])

    tied.sort(key=lambda a: (abs(a - CENTER_COL), a))
    return int(tied[0])


def _legal_actions_from_state_1ch(state: np.ndarray) -> List[int]:
    """Legal if top cell is empty."""
    s = np.asarray(state)
    if s.ndim == 3:
        s = s[0]
    if s.shape != (6, 7):
        raise ValueError(f"Expected (6,7) or (1,6,7), got {tuple(s.shape)}")
    return [c for c in range(7) if s[0, c] == 0]


class FrozenQWrapper(nn.Module):
    """Wrap a model so it exposes:

      - forward(x) -> scores (B,7)
      - act(state, legal_actions) -> int

    Underlying model may return (scores, value) or just scores.
    """

    def __init__(self, model: nn.Module, device: torch.device, tag: str = "HOF"):
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.tag = str(tag)

        self.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            scores = out[0]
        else:
            scores = out

        if not isinstance(scores, torch.Tensor) or scores.dim() != 2 or scores.size(-1) != 7:
            raise ValueError(f"Model must return (B,7) scores, got {type(scores)} {getattr(scores,'shape',None)}")

        return scores

    @torch.inference_mode()
    def act(self, state: np.ndarray, legal_actions: Sequence[int]) -> int:
        if not legal_actions:
            raise ValueError("No legal actions")
        if len(legal_actions) == 1:
            return int(legal_actions[0])

        x = _state_to_1ch_tensor(state, device=self.device)  # (1,1,6,7)
        scores = self.forward(x)[0]                          # (7,)
        return _argmax_legal_center_tiebreak(scores, legal_actions)


@dataclass
class HOFMember:
    name: str
    ckpt_path: str
    metascore: Optional[float] = None
    policy: Optional[nn.Module] = None


class DQNEnsemblePolicy(nn.Module):
    """Read-only weighted ensemble of FrozenQWrapper members."""

    def __init__(
        self,
        members: Sequence[FrozenQWrapper],
        weights: Optional[Sequence[float]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        if not members:
            raise ValueError("DQNEnsemblePolicy needs at least one member")

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
                p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores_list: List[torch.Tensor] = []
        for m in self.members:
            scores_list.append(m(x))  # (B,7)

        stacked = torch.stack(scores_list, dim=0)  # (E,B,7)
        w = self.weights.view(-1, 1, 1)
        return (w * stacked).sum(dim=0)            # (B,7)

    @torch.inference_mode()
    def act(self, state: np.ndarray, legal_actions: Sequence[int]) -> int:
        if not legal_actions:
            raise ValueError("No legal actions")
        if len(legal_actions) == 1:
            return int(legal_actions[0])

        x = _state_to_1ch_tensor(state, device=self.device)
        scores = self.forward(x)[0]
        return _argmax_legal_center_tiebreak(scores, legal_actions)


class DQNHallOfFame:
    """Registry + lazy-loading for frozen opponents."""

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self.members: Dict[str, HOFMember] = {}

    def add_member(self, name: str, ckpt_path: str, metascore: Optional[float] = None) -> None:
        self.members[str(name)] = HOFMember(name=str(name), ckpt_path=str(ckpt_path), metascore=metascore)

    def remove_member(self, name: str) -> None:
        self.members.pop(name, None)

    def list_members(self) -> List[str]:
        return list(self.members.keys())

    def get_member(self, name: str) -> HOFMember:
        if name not in self.members:
            raise KeyError(f"HOF member '{name}' not found")
        return self.members[name]

    # ---------- loading ----------

    def _load_supervised_cnet192(self, ckpt_path: str) -> FrozenQWrapper:
        model, _ = load_cnet192(path=ckpt_path, device=self.device, strict=True)
        model.eval()
        return FrozenQWrapper(model, device=self.device, tag="HOF_CNet192")

    def _load_dqn_checkpoint(self, ckpt_path: str) -> FrozenQWrapper:
        """Load DQNAgent.save() format: expects 'q_state' in dict."""
        try:
            from DQN.dqn_agent import CNet192_Q  # type: ignore
        except Exception:
            from dqn_agent import CNet192_Q  # local import fallback

        ckpt = torch.load(ckpt_path, map_location=self.device)
        if not isinstance(ckpt, dict) or "q_state" not in ckpt:
            raise ValueError("Not a DQN checkpoint (missing q_state)")

        sd = ckpt["q_state"]
        # infer whether conv_mid exists
        use_mid = any(k.startswith("conv_mid") for k in sd.keys())

        q = CNet192_Q(in_channels=1, use_mid_3x3=use_mid).to(self.device)
        q.load_state_dict(sd, strict=True)
        q.eval()
        return FrozenQWrapper(q, device=self.device, tag="HOF_DQN")

    def _load_policy_from_ckpt(self, ckpt_path: str) -> FrozenQWrapper:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # 1) supervised CNet192
        try:
            return self._load_supervised_cnet192(ckpt_path)
        except FileNotFoundError:
            raise
        except Exception:
            pass

        # 2) DQN checkpoint
        return self._load_dqn_checkpoint(ckpt_path)

    def ensure_loaded(self, name: str) -> FrozenQWrapper:
        mem = self.get_member(name)
        if mem.policy is None:
            mem.policy = self._load_policy_from_ckpt(mem.ckpt_path)
        return mem.policy  # type: ignore[return-value]

    # ---------- ensemble ----------

    def build_ensemble(self, names: Sequence[str], use_metascore_weights: bool = True) -> DQNEnsemblePolicy:
        if not names:
            raise ValueError("Cannot build ensemble from empty names list")

        policies: List[FrozenQWrapper] = []
        weights: List[float] = []

        for n in names:
            mem = self.get_member(n)
            pol = self.ensure_loaded(n)
            policies.append(pol)

            if use_metascore_weights:
                weights.append(float(mem.metascore) if mem.metascore is not None else 1.0)
            else:
                weights.append(1.0)

        ens = DQNEnsemblePolicy(policies, weights=weights if use_metascore_weights else None, device=self.device)
        ens.eval()
        return ens


__all__ = [
    "HOFMember",
    "DQNHallOfFame",
    "DQNEnsemblePolicy",
    "FrozenQWrapper",
    "_legal_actions_from_state_1ch",
]
