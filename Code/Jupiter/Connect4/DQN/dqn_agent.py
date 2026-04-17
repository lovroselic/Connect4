"""
DQN/dqn_agent.py

DQN agent for Connect-4 using CNet192 as Q-network (single-channel POV input).

Conventions:
- State is float32 shape (1,6,7) from player-to-move POV.
- Rewards are for the mover at that state.
- POV flips each ply, so bootstrap gets parity sign:
    target = r_n + (1-done_n) * (gamma**n) * ((-1)**n) * V(next)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from C4.CNet192 import CNet192, load_cnet192
from DQN.dqn_replay_memory_per import PrioritizedReplayMemory, NStepTransition

COLS = 7
NEG_INF = -1e9



# ---------------- checkpoint helpers ----------------



def _infer_cnet192_cfg_from_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[int, bool]:
    """
    Infer (in_channels, use_mid_3x3) from a CNet192/CNet192_Q state_dict.
    Best-effort: if uncertain, defaults to (1, True).
    """
    in_ch = 1
    use_mid = True

    # conv1 weight shape: (out_ch, in_ch, kh, kw)
    w1 = sd.get("conv1.weight", None)
    if isinstance(w1, torch.Tensor) and w1.ndim == 4:
        in_ch = int(w1.shape[1])

    # presence of conv_mid params indicates mid block
    # (your CNet192 uses conv_mid if use_mid_3x3=True)
    has_mid = any(k.startswith("conv_mid.") for k in sd.keys())
    use_mid = bool(has_mid)

    return in_ch, use_mid


def save_dqn_checkpoint(agent, path: str, **meta):
    # infer cfg from model
    in_ch = int(getattr(agent.q_net.conv1, "in_channels", 1)) if hasattr(agent.q_net, "conv1") else 1
    use_mid = (getattr(agent.q_net, "conv_mid", None) is not None)
    freeze_conv = False
    try:
        freeze_conv = (not next(agent.q_net.conv1.parameters()).requires_grad)
    except Exception:
        pass

    meta = dict(meta)
    meta.setdefault("in_channels", in_ch)
    meta.setdefault("use_mid_3x3", bool(use_mid))
    meta.setdefault("freeze_conv", bool(freeze_conv))

    payload = {
        "q_net": agent.q_net.state_dict(),
        "meta": meta,
    }

    tgt = getattr(agent, "target_net", None)
    if tgt is not None:
        payload["target_net"] = tgt.state_dict()

    opt = getattr(agent, "optimizer", None)
    if opt is not None:
        payload["optimizer"] = opt.state_dict()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


class CNet192_Q(CNet192):
    """DQN view of CNet192: treat policy logits as Q-values."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, _v = super().forward(x)
        return q  # (B,7)


def freeze_cnet192_conv_block(m: CNet192, freeze: bool = True) -> None:
    """Freeze/unfreeze conv1/conv2/(conv_mid) parameters."""
    convs = [m.conv1, m.conv2]
    if getattr(m, "conv_mid", None) is not None:
        convs.append(m.conv_mid)
    for layer in convs:
        for p in layer.parameters():
            p.requires_grad = not bool(freeze)


def _ensure_b1hw(state: np.ndarray) -> np.ndarray:
    """Ensure (1,6,7) float32."""
    if not isinstance(state, np.ndarray):
        state = np.asarray(state)

    if state.ndim == 2 and state.shape == (6, 7):
        state = state[None, :, :]

    if state.ndim != 3 or state.shape != (1, 6, 7):
        raise ValueError(f"state must be (1,6,7) or (6,7), got {state.shape}")

    return state.astype(np.float32, copy=False)


def _pack_states(states: List[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.asarray([_ensure_b1hw(s) for s in states], dtype=np.float32)  # (B,1,6,7)
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def _legal_mask_from_state(x: torch.Tensor) -> torch.Tensor:
    """Return legal mask (B,7) from (B,1,6,7). Column legal if top cell empty."""
    top = x[:, 0, 0, :]  # row 0 is top row in your env
    return (top == 0)


@dataclass
class DQNHyperParams:
    lr: float = 2e-4
    gamma: float = 0.99

    epsilon: float = 0.10
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.999

    batch_size: int = 256
    reward_scale: float = 1.0

    # PER
    per_alpha: float = 0.60
    per_eps: float = 1e-3
    per_beta_start: float = 0.40
    per_beta_end: float = 0.90
    per_beta_steps: int = 250_000
    per_mix_1step: float = 0.70

    # target updates
    target_update_mode: str = "soft"      # "soft" or "hard"
    target_update_interval: int = 1000    # hard-update period in replay steps
    tau: float = 0.005                    # soft update coefficient

    # misc
    grad_clip: float = 5.0


class DQNAgent:
    def __init__(
        self,
        q_net: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        memory_capacity: int = 500_000,
        hparams: Optional[DQNHyperParams] = None,
        seed: int = 0,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hp = hparams or DQNHyperParams()
        self.rng = np.random.default_rng(int(seed))

        # online net
        self.q_net: nn.Module = (q_net or CNet192_Q(in_channels=1, use_mid_3x3=True)).to(self.device)

        # target net with matching arch
        if isinstance(self.q_net, (CNet192, CNet192_Q)):
            in_ch = int(getattr(self.q_net.conv1, "in_channels", 1))
            use_mid = getattr(self.q_net, "conv_mid", None) is not None
            self.target_net = CNet192_Q(in_channels=in_ch, use_mid_3x3=use_mid).to(self.device)
        else:
            import copy
            self.target_net = copy.deepcopy(self.q_net).to(self.device)

        # compatibility aliases (older utilities)
        self.model = self.q_net
        self.target_model = self.target_net

        # replay memory
        self.memory = PrioritizedReplayMemory(
            capacity=int(memory_capacity),
            alpha=float(self.hp.per_alpha),
            eps=float(self.hp.per_eps),
        )

        # PER beta schedule
        self.per_beta = float(self.hp.per_beta_start)
        steps = max(1, int(self.hp.per_beta_steps))
        self.per_beta_step = (float(self.hp.per_beta_end) - self.per_beta) / steps

        # counters
        self.global_step = 0
        self._last_hard_update = 0

        # plotting/debug hook
        self.last_replay_stats: Optional[Dict[str, Any]] = None

        # optimizer (trainable params only)
        self._rebuild_optimizer()

        self.sync_target()
        self.q_net.train(True)
        self.target_net.eval()
        
        self.per_w_hist = deque(maxlen=200_000)
        self.td_hist    = deque(maxlen=200_000)

    # ---------- construction ----------
    @classmethod
    def from_dqn_checkpoint(
        cls,
        ckpt_path: str,
        device: Optional[torch.device] = None,
        strict: bool = True,
        freeze_conv: Optional[bool] = None,
        memory_capacity: int = 500_000,
        hparams: Optional["DQNHyperParams"] = None,
        seed: int = 0,
        load_optimizer: bool = False,
        load_target: bool = True,
    ) -> Tuple["DQNAgent", Dict[str, Any]]:
        """
        Robust loader for multiple checkpoint formats.
    
        Supports:
        A) save_dqn_checkpoint(): {"q_net":..., "target_net":..., "optimizer":..., "meta":...}
        B) DQNAgent.save():       {"q_state":..., "t_state":..., "opt":..., "hparams":..., ...}
        C) raw state_dict:        {"conv1.weight":..., ...}
        """
        def _strip_prefix_if_all(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
            ks = list(sd.keys())
            if ks and all(k.startswith(prefix) for k in ks):
                return {k[len(prefix):]: v for k, v in sd.items()}
            return sd
    
        def _normalize_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            if not isinstance(sd, dict):
                raise ValueError("state_dict is not a dict")
    
            # common wrappers
            sd = _strip_prefix_if_all(sd, "module.")
            sd = _strip_prefix_if_all(sd, "q_net.")
            sd = _strip_prefix_if_all(sd, "model.")
            return sd
    
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        raw = torch.load(ckpt_path, map_location=device)
    
        # ---------------- detect format ----------------
        meta: Dict[str, Any] = {}
        q_sd: Optional[Dict[str, torch.Tensor]] = None
        t_sd: Optional[Dict[str, torch.Tensor]] = None
        opt_sd: Optional[Dict[str, Any]] = None
    
        if isinstance(raw, dict):
            # Format A: save_dqn_checkpoint
            if "q_net" in raw and isinstance(raw["q_net"], dict):
                q_sd = raw["q_net"]
                t_sd = raw.get("target_net", None) if isinstance(raw.get("target_net", None), dict) else None
                opt_sd = raw.get("optimizer", None) if isinstance(raw.get("optimizer", None), dict) else None
                meta = raw.get("meta", {}) if isinstance(raw.get("meta", {}), dict) else {}
    
            # Format B: DQNAgent.save
            elif "q_state" in raw and isinstance(raw["q_state"], dict):
                q_sd = raw["q_state"]
                t_sd = raw.get("t_state", None) if isinstance(raw.get("t_state", None), dict) else None
                opt_sd = raw.get("opt", None) if isinstance(raw.get("opt", None), dict) else None
    
                # optional extra/meta pattern
                extra = raw.get("extra", {}) if isinstance(raw.get("extra", {}), dict) else {}
                meta = extra.get("meta", {}) if isinstance(extra.get("meta", {}), dict) else raw.get("meta", {})
    
            # Format C: common "state_dict" wrapper
            elif "state_dict" in raw and isinstance(raw["state_dict"], dict):
                q_sd = raw["state_dict"]
                meta = raw.get("meta", {}) if isinstance(raw.get("meta", {}), dict) else {}
    
            # Format D: raw looks like a state_dict already (tensor values)
            else:
                # heuristic: if most values are tensors, assume raw is a state_dict
                tensor_vals = sum(1 for v in raw.values() if isinstance(v, torch.Tensor))
                if tensor_vals >= max(1, int(0.8 * len(raw))):
                    q_sd = raw
                else:
                    raise ValueError(
                        f"Unrecognized checkpoint format: {ckpt_path}. "
                        f"Top-level keys: {list(raw.keys())[:50]}"
                    )
        else:
            raise ValueError(f"Checkpoint is not a dict: {ckpt_path} (type={type(raw)})")
    
        q_sd = _normalize_state_dict(q_sd)
    
        # ---------------- decide arch ----------------
        in_ch = None
        use_mid = None
    
        if isinstance(meta, dict):
            if "in_channels" in meta:
                try:
                    in_ch = int(meta["in_channels"])
                except Exception:
                    pass
            if "use_mid_3x3" in meta:
                use_mid = bool(meta["use_mid_3x3"])
    
        if in_ch is None or use_mid is None:
            inf_in_ch, inf_use_mid = _infer_cnet192_cfg_from_state_dict(q_sd)
            if in_ch is None:
                in_ch = inf_in_ch
            if use_mid is None:
                use_mid = inf_use_mid
    
        in_ch = 1 if in_ch is None else int(in_ch)
        use_mid = True if use_mid is None else bool(use_mid)
    
        # ---------------- build q_net + agent ----------------
        q_net = CNet192_Q(in_channels=in_ch, use_mid_3x3=use_mid).to(device)
        q_net.load_state_dict(q_sd, strict=bool(strict))
    
        if freeze_conv is None:
            freeze_conv = bool(meta.get("freeze_conv", False)) if isinstance(meta, dict) and "freeze_conv" in meta else False
        if freeze_conv:
            freeze_cnet192_conv_block(q_net, freeze=True)
    
        agent = cls(q_net=q_net, device=device, memory_capacity=memory_capacity, hparams=hparams, seed=seed)
    
        # ---------------- optional target ----------------
        if load_target and isinstance(t_sd, dict):
            try:
                t_sd = _normalize_state_dict(t_sd)
                agent.target_net.load_state_dict(t_sd, strict=bool(strict))
                agent.target_net.eval()
            except Exception:
                agent.sync_target()
        else:
            agent.sync_target()
    
        # ---------------- optional optimizer ----------------
        if load_optimizer and isinstance(opt_sd, dict):
            try:
                agent.optimizer.load_state_dict(opt_sd)
            except Exception:
                pass
    
        return agent, raw
    
    
    @classmethod
    def from_cnet192_checkpoint(
        cls,
        model_path: str,
        device: Optional[torch.device] = None,
        strict: bool = True,
        freeze_conv: bool = True,
        memory_capacity: int = 500_000,
        hparams: Optional[DQNHyperParams] = None,
        seed: int = 0,
    ) -> Tuple["DQNAgent", Dict[str, Any]]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnet, ckpt = load_cnet192(model_path, device=device, strict=bool(strict))

        cfg = (ckpt.get("cfg", {}) or {}) if isinstance(ckpt, dict) else {}
        in_ch = int(cfg.get("in_channels", cfg.get("input_channels", 1)))
        use_mid = bool(cfg.get("use_mid_3x3", True))

        q_net = CNet192_Q(in_channels=in_ch, use_mid_3x3=use_mid).to(device)
        q_net.load_state_dict(cnet.state_dict(), strict=True)

        if freeze_conv:
            freeze_cnet192_conv_block(q_net, freeze=True)

        agent = cls(q_net=q_net, device=device, memory_capacity=memory_capacity, hparams=hparams, seed=seed)
        agent.sync_target()
        return agent, ckpt

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
    def board_to_state(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Convert env board -> (1,6,7) float32 POV state (+1 = mover pieces, -1 = opponent).
        Supports boards encoded as:
          - signed: {-1,0,+1}
          - id: {0,1,2}
        Supports player as:
          - signed: +1 / -1
          - id: 1 / 2
        """
        b = np.asarray(board)
    
        # already state-shaped
        if b.shape == (1, 6, 7):
            return b.astype(np.float32, copy=False)
    
        if b.shape != (6, 7):
            raise ValueError(f"board must be (6,7) or (1,6,7), got {b.shape}")
    
        # Determine encoding style
        bmin = int(b.min())
        bmax = int(b.max())
    
        # --- Case A: player-id board (0,1,2) ---
        if bmin >= 0 and bmax <= 2:
            # normalize player id
            if player in (1, 2):
                pid = int(player)
            elif player in (+1, -1):
                # conventional mapping: +1 -> player 1, -1 -> player 2
                pid = 1 if int(player) == +1 else 2
            else:
                raise ValueError(f"player must be 1/2 or +/-1, got {player}")
    
            s = np.zeros((6, 7), dtype=np.float32)
            s[b == pid] = 1.0
            s[(b != 0) & (b != pid)] = -1.0
            return s[None, :, :]
    
        # --- Case B: signed board (-1,0,+1) ---
        if bmin >= -1 and bmax <= 1:
            if player in (+1, -1):
                ps = float(player)
            elif player in (1, 2):
                ps = 1.0 if int(player) == 1 else -1.0
            else:
                raise ValueError(f"player must be 1/2 or +/-1, got {player}")
    
            return (b.astype(np.float32) * ps)[None, :, :]
    
        raise ValueError(f"Unsupported board encoding range: min={bmin}, max={bmax}")

    # ---------- optimizer / freezing ----------

    def _rebuild_optimizer(self) -> None:
        params = [p for p in self.q_net.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=float(self.hp.lr), weight_decay=1e-5)

    def freeze_conv_block(self, freeze: bool = True, rebuild_optimizer: bool = True) -> None:
        if isinstance(self.q_net, (CNet192, CNet192_Q)):
            freeze_cnet192_conv_block(self.q_net, freeze=bool(freeze))
        if isinstance(self.target_net, (CNet192, CNet192_Q)):
            freeze_cnet192_conv_block(self.target_net, freeze=bool(freeze))
        if rebuild_optimizer:
            self._rebuild_optimizer()

    # ---------- setters (loop-friendly) ----------

    def set_lr(self, lr: float) -> None:
        lr = float(lr)
        self.hp.lr = lr
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def set_epsilon(self, eps: float) -> None:
        self.hp.epsilon = float(eps)

    def set_epsilon_min(self, eps_min: float) -> None:
        self.hp.epsilon_min = float(eps_min)

    def set_epsilon_decay(self, decay: float) -> None:
        self.hp.epsilon_decay = float(decay)

    def set_per_mix_1step(self, mix: float) -> None:
        self.hp.per_mix_1step = float(mix)

    def set_target_update_interval(self, interval: int) -> None:
        self.hp.target_update_interval = int(interval)

    def reset_target_update_timer(self, step: int = 0) -> None:
        self._last_hard_update = int(step)

    def update_target_model(self) -> None:
        self.sync_target()

    # ---------- epsilon ----------

    @property
    def epsilon(self) -> float:
        return float(self.hp.epsilon)

    def decay_epsilon(self) -> None:
        if self.hp.epsilon > self.hp.epsilon_min:
            self.hp.epsilon = max(self.hp.epsilon_min, self.hp.epsilon * self.hp.epsilon_decay)

    # ---------- acting ----------

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        s = _ensure_b1hw(state)
        x = torch.from_numpy(s).unsqueeze(0).to(self.device, dtype=torch.float32)  # (1,1,6,7)
        q = self.q_net(x).squeeze(0).detach().cpu().numpy()
        if q.shape != (COLS,):
            raise ValueError(f"Expected q shape (7,), got {q.shape}")
        return q.astype(np.float32, copy=False)

    def _argmax_legal_center_tiebreak(self, q: np.ndarray, legal_actions: List[int]) -> int:
        la = np.asarray(list(legal_actions), dtype=np.int64)
        if la.size == 0:
            raise ValueError("No legal actions")
        masked = np.full((COLS,), -np.inf, dtype=np.float64)
        masked[la] = np.asarray(q, dtype=np.float64)[la]
        qmax = float(masked[la].max())
        tied = la[masked[la] >= (qmax - 1e-12)]
        if tied.size == 1:
            return int(tied[0])
        if 3 in tied:
            return 3
        return int(self.rng.choice(tied))

    def act(self, state: np.ndarray, legal_actions: List[int], epsilon_override: Optional[float] = None) -> int:
        eps = float(self.hp.epsilon if epsilon_override is None else epsilon_override)
        if (eps > 0.0) and (self.rng.random() < eps):
            return int(self.rng.choice(legal_actions))
        q = self.q_values(state)
        return self._argmax_legal_center_tiebreak(q, legal_actions)

    # ---------- memory ----------

    def remember_1step(self, s, a, r, s2, done, add_mirror=True, add_colorswap=True, add_mirror_colorswap=True) -> None:
        self.memory.push_1step_aug(
            _ensure_b1hw(s), int(a), float(r), _ensure_b1hw(s2), bool(done),
            add_mirror=bool(add_mirror),
            add_colorswap=bool(add_colorswap),
            add_mirror_colorswap=bool(add_mirror_colorswap),
        )

    def remember_nstep(self, s, a, rN, sN, doneN, n_steps, add_mirror=True, add_colorswap=True, add_mirror_colorswap=True) -> None:
        self.memory.push_nstep_aug(
            _ensure_b1hw(s), int(a), float(rN), _ensure_b1hw(sN), bool(doneN), int(n_steps),
            add_mirror=bool(add_mirror),
            add_colorswap=bool(add_colorswap),
            add_mirror_colorswap=bool(add_mirror_colorswap),
        )

    # ---------- training ----------

    def _parity_sign(self, n_steps: torch.Tensor) -> torch.Tensor:
        even = (n_steps % 2 == 0)
        return torch.where(
            even,
            torch.ones_like(n_steps, dtype=torch.float32),
            -torch.ones_like(n_steps, dtype=torch.float32),
        )

    def replay(self, batch_size: Optional[int] = None) -> Optional[float]:
        if batch_size is not None:
            self.hp.batch_size = int(batch_size)

        bs = int(self.hp.batch_size)
        if len(self.memory) < bs:
            return None

        (b1, bn), (i1, in_), (w1, wn) = self.memory.sample_mixed_seedaware(
            batch_size=bs,
            mix_1step=float(self.hp.per_mix_1step),
            beta=float(self.per_beta),
            max_seed_frac=0.90,
            min_seed_frac=0.10,
            rng=self.rng,
        )

        batch_all = list(b1) + list(bn)
        n1 = len(b1)
        if not batch_all:
            return None

        states, next_states, actions, rewards, dones, n_steps = [], [], [], [], [], []
        for t in batch_all:
            if isinstance(t, NStepTransition) or hasattr(t, "reward_n"):
                states.append(t.state)
                next_states.append(t.next_state_n)
                actions.append(int(t.action))
                rewards.append(float(t.reward_n) * float(self.hp.reward_scale))
                dones.append(bool(t.done_n))
                n_steps.append(int(t.n_steps))
            else:
                states.append(t.state)
                next_states.append(t.next_state)
                actions.append(int(t.action))
                rewards.append(float(t.reward) * float(self.hp.reward_scale))
                dones.append(bool(t.done))
                n_steps.append(1)

        X  = _pack_states(states, self.device)
        Xn = _pack_states(next_states, self.device)

        a_t = torch.tensor(actions, device=self.device, dtype=torch.long).view(-1, 1)
        r_t = torch.tensor(rewards, device=self.device, dtype=torch.float32).view(-1, 1)
        d_t = torch.tensor(dones, device=self.device, dtype=torch.bool).view(-1, 1)
        n_t = torch.tensor(n_steps, device=self.device, dtype=torch.long).view(-1, 1)

        q_all = self.q_net(X)
        q_sa = q_all.gather(1, a_t)

        with torch.no_grad():
            legal = _legal_mask_from_state(Xn)

            q_next_online = self.q_net(Xn).masked_fill(~legal, NEG_INF)
            a_next = q_next_online.argmax(dim=1, keepdim=True)

            q_next_tgt = self.target_net(Xn).masked_fill(~legal, NEG_INF)
            v_next = q_next_tgt.gather(1, a_next)

            gamma_pow = (float(self.hp.gamma) ** n_t.to(torch.float32))
            sign_boot = self._parity_sign(n_t).to(self.device).view(-1, 1)

            target = r_t + (~d_t).to(torch.float32) * gamma_pow * sign_boot * v_next

        td = (target - q_sa)
        per_sample = F.smooth_l1_loss(q_sa, target, reduction="none").squeeze(1)

        is_w = np.concatenate([w1, wn]) if (w1.size or wn.size) else np.ones((len(batch_all),), dtype=np.float32)
        is_w_t = torch.as_tensor(is_w, device=self.device, dtype=per_sample.dtype)
        is_w_t = is_w_t / (is_w_t.mean() + 1e-8)
        is_w_t = torch.clamp(is_w_t, 0.5, 3.0)
        
        self.per_w_hist.extend(is_w.astype(np.float32).tolist())
        self.td_hist.extend(td.detach().squeeze(1).cpu().numpy().astype(np.float32).tolist())

        loss = (is_w_t * per_sample).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), float(self.hp.grad_clip))
        self.optimizer.step()

        prios = td.detach().abs().squeeze(1).cpu().numpy()
        prios = np.nan_to_num(prios, nan=0.0, posinf=1e6, neginf=1e6)
        self.memory.update_priorities(i1, prios[:n1], indices_n=in_, td_errors_n=prios[n1:])

        if self.per_beta < float(self.hp.per_beta_end):
            self.per_beta = min(float(self.hp.per_beta_end), self.per_beta + float(self.per_beta_step))

        self.global_step += 1
        self.maybe_update_target(self.global_step)

        loss_val = float(loss.detach().cpu().item())
        with torch.no_grad():
            td_abs = td.detach().abs()
            self.last_replay_stats = {
                "loss": loss_val,
                "td_abs_mean": float(td_abs.mean().cpu().item()),
                "td_abs_max": float(td_abs.max().cpu().item()),
                "q_mean": float(q_sa.detach().mean().cpu().item()),
                "q_abs_mean": float(q_sa.detach().abs().mean().cpu().item()),
                "w_mean": float(is_w_t.detach().mean().cpu().item()),
                "w_max": float(is_w_t.detach().max().cpu().item()),
                "per_beta": float(self.per_beta),
                "global_step": int(self.global_step),
            }

        return loss_val

    # ---------- target updates ----------

    def configure_target_update(
        self,
        mode: str = "hard",
        tau: Optional[float] = None,
        interval: Optional[int] = None,
        reset_timer: bool = True,
    ) -> None:
        mode = str(mode).lower().strip()
        if mode == "polyak":
            mode = "soft"
        if mode not in ("hard", "soft"):
            raise ValueError(f"Unknown target update mode: {mode}")

        self.hp.target_update_mode = mode
        if tau is not None:
            self.hp.tau = float(tau)
        if interval is not None:
            self.hp.target_update_interval = int(interval)

        if reset_timer:
            self._last_hard_update = int(self.global_step)

    def soft_update_target(self, tau: Optional[float] = None) -> None:
        tau = float(self.hp.tau if tau is None else tau)
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"tau must be in (0,1], got {tau}")
        with torch.no_grad():
            for pt, ps in zip(self.target_net.parameters(), self.q_net.parameters()):
                pt.data.lerp_(ps.data, tau)

    def maybe_update_target(self, step: int, force_hard: bool = False) -> None:
        mode = str(self.hp.target_update_mode).lower().strip()
        if mode == "polyak":
            mode = "soft"

        if force_hard:
            self.sync_target()
            self._last_hard_update = int(step)
            return

        if mode == "soft":
            self.soft_update_target(float(self.hp.tau))
            return

        if (int(step) - int(self._last_hard_update)) >= int(self.hp.target_update_interval):
            self.sync_target()
            self._last_hard_update = int(step)

    # ---------- save/load ----------

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        ckpt = {
            "q_state": self.q_net.state_dict(),
            "t_state": self.target_net.state_dict(),
            "opt": self.optimizer.state_dict(),
            "hparams": self.hp.__dict__.copy(),
            "per_beta": float(self.per_beta),
            "global_step": int(self.global_step),
        }
        if extra:
            ckpt["extra"] = dict(extra)
        torch.save(ckpt, path)

    def load(self, path: str, strict: bool = True) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_state"], strict=bool(strict))
        self.target_net.load_state_dict(ckpt["t_state"], strict=bool(strict))
        self.optimizer.load_state_dict(ckpt["opt"])

        hp = ckpt.get("hparams", None)
        if isinstance(hp, dict):
            for k, v in hp.items():
                if hasattr(self.hp, k):
                    setattr(self.hp, k, v)

        self.per_beta = float(ckpt.get("per_beta", self.per_beta))
        self.global_step = int(ckpt.get("global_step", self.global_step))
        return ckpt


__all__ = ["DQNHyperParams", "DQNAgent", "CNet192_Q", "freeze_cnet192_conv_block"]
