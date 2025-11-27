# PPO/ppo_hall_of_fame.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import numpy as np

from PPO.actor_critic import ActorCritic
from PPO.ppo_agent_eval import _argmax_legal_center_tiebreak


@dataclass
class HOFMember:
    name: str
    ckpt_path: str
    metascore: Optional[float] = None
    policy: Optional[ActorCritic] = None  # lazy-loaded


class PPOEnsemblePolicy(nn.Module):
    """
    Read-only ensemble of multiple PPO ActorCritic policies.

    Exposes .act(state, legal_actions, temperature, ply_idx)
    so it can be used anywhere your normal ActorCritic is used as an opponent.
    """

    def __init__(self, members, weights=None, device="cpu"):
        super().__init__()

        if not members:
            raise ValueError("PPOEnsemblePolicy needs at least one member.")

        self.members = nn.ModuleList(members)
        self.device = torch.device(device)

        # Normalize weights or use uniform
        if weights is None:
            w = np.ones(len(self.members), dtype=np.float32)
        else:
            w = np.array(weights, dtype=np.float32)
        w = np.maximum(w, 1e-6)
        w = w / w.sum()

        # store as buffer so it moves with .to(...)
        self.register_buffer("weights", torch.from_numpy(w).float())

        # move whole ensemble to device & eval
        self.to(self.device)
        for m in self.members:
            m.eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor):
        """
        Weighted-average logits & values.
        x: [1, C, 6, 7]
        Returns (logits[1,7], value[1,1])
        """
        logits_list = []
        values_list = []

        for m in self.members:
            l, v = m(x)
            logits_list.append(l)
            values_list.append(v)

        logits_stacked = torch.stack(logits_list, dim=0)   # [E,1,7]
        values_stacked = torch.stack(values_list, dim=0)   # [E,1,1]

        w = self.weights.view(-1, 1, 1)                    # [E,1,1]
        logits = (w * logits_stacked).sum(dim=0)           # [1,7]
        value  = (w * values_stacked).sum(dim=0)           # [1,1]

        return logits, value

    @torch.inference_mode()
    def act(self, state: np.ndarray, legal_actions, temperature: float = 1.0, ply_idx=None):
        """
        Minimal ActorCritic-like interface used by ppo_opponent_step / ppo_maybe_opponent_opening.

        - state: np.ndarray, same format you pass to ActorCritic.act during training
                 (your encode_two_channel_agent_centric output)
        - legal_actions: iterable of ints (columns)
        - temperature: ignored (we use pure greedy)
        - ply_idx: ignored here (we do not re-implement heuristics)

        Returns: (action:int, log_prob:Tensor, value:Tensor, info:dict)
        """
        legal_actions = list(legal_actions)
        if not legal_actions: raise ValueError("No legal actions for ensemble .act()")

        # trivial case: only one legal move
        if len(legal_actions) == 1:
            a = int(legal_actions[0])
            log_prob = torch.zeros((), dtype=torch.float32, device=self.device)
            value = torch.zeros((), dtype=torch.float32, device=self.device)
            return a, log_prob, value, {"mode": "ensemble"}

        # encode to tensor
        x_np = np.ascontiguousarray(state)
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(self.device)   # [1,C,6,7]

        # weighted ensemble forward
        logits, value = self.forward(x)                                   # [1,7], [1,1]
        logits_np = logits.squeeze(0).detach().cpu().numpy()              # [7]
        a = _argmax_legal_center_tiebreak(logits_np, legal_actions)

        # opponent doesn't use log_prob for learning => dummy 0
        log_prob = torch.zeros((), dtype=torch.float32, device=self.device)
        value_out = value.squeeze(0).detach()

        return int(a), log_prob, value_out, {"mode": "ensemble"}

class PPOHallOfFame:
    """
    Keeps a small population of frozen PPO policies.
    Can build an ensemble over any subset of them.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.members: Dict[str, HOFMember] = {}

    # ---------- Member management ----------

    def add_member(
        self,
        name: str,
        ckpt_path: str,
        metascore: Optional[float] = None,
        policy: Optional[ActorCritic] = None,
    ):
        """
        Register a Hall-of-Fame entry.

        - ckpt_path: path to checkpoint (relative or absolute)
        - metascore: optional scalar performance signal (0..1); used as weight.
        - policy: optional already-loaded ActorCritic.
        """
        self.members[name] = HOFMember(
            name=name,
            ckpt_path=ckpt_path,
            metascore=metascore,
            policy=policy,
        )

    def remove_member(self, name: str):
        self.members.pop(name, None)

    def list_members(self) -> List[str]:
        return list(self.members.keys())

    def get_member(self, name: str) -> HOFMember:
        if name not in self.members:
            raise KeyError(f"HOF member '{name}' not found.")
        return self.members[name]

    # ---------- Loading ----------

    def _load_policy_from_ckpt(self, ckpt_path: str) -> ActorCritic:
        ckpt = torch.load(ckpt_path, map_location=self.device)

        policy = ActorCritic().to(self.device)

        # Try to guess the right field in the checkpoint
        if isinstance(ckpt, dict):
            if "policy_state_dict" in ckpt:
                state_dict = ckpt["policy_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                # Heuristic: if keys look like module params, assume this is the state_dict itself
                if all(isinstance(k, str) for k in ckpt.keys()) and any("." in k for k in ckpt.keys()):
                    state_dict = ckpt
                else:
                    raise ValueError(
                        f"Unknown checkpoint format in '{ckpt_path}'. "
                        f"Available keys: {list(ckpt.keys())}"
                    )
        else:
            raise TypeError(f"Unexpected checkpoint type for '{ckpt_path}': {type(ckpt)}")

        policy.load_state_dict(state_dict)
        policy.eval()
        return policy

    def ensure_loaded(self, name: str) -> ActorCritic:
        """
        Make sure the member's policy is loaded and moved to device.
        """
        member = self.get_member(name)
        if member.policy is None: member.policy = self._load_policy_from_ckpt(member.ckpt_path)
        return member.policy

    # ---------- Ensemble building ----------

    def build_ensemble(
        self,
        names: Sequence[str],
        use_metascore_weights: bool = True,
    ) -> PPOEnsemblePolicy:
        """
        Build an ensemble policy over the given member names.

        If use_metascore_weights=True, weight members by their metascore.
        If a member has metascore=None, its weight defaults to 1.0.
        """
        if len(names) == 0:                raise ValueError("Cannot build ensemble from empty name list.")

        policies: List[ActorCritic] = []
        weight_list: List[float] = []

        for name in names:
            member = self.get_member(name)
            policy = self.ensure_loaded(name)   # ActorCritic
            policies.append(policy)

            if use_metascore_weights:
                w = 1.0 if member.metascore is None else float(member.metascore)
                weight_list.append(w)
            else:
                weight_list.append(1.0)

        ensemble = PPOEnsemblePolicy(
            members=policies,
            weights=weight_list if use_metascore_weights else None,
            device=self.device,
        )
        ensemble.eval()
        return ensemble
    
