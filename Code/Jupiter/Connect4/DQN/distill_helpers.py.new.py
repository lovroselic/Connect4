# DQN.distill_helpers.py

from __future__ import annotations
import random
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch import amp
import torch.nn.functional as F
import math

from DQN.dqn_agent import DQNAgent
from DQN.dqn_model import DQN
from C4.connect4_env import Connect4Env
from C4.fast_connect4_lookahead import Connect4Lookahead
from DQN.agent_eval import load_agent_from_ckpt 
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import os, json, hashlib
from pathlib import Path

_ORACLE_LA = None
_ORACLE_CACHE = OrderedDict()
_ORACLE_CACHE_MAX = 1_000_000  # tune to RAM

_ORACLE_DEPTH = 4
_ORACLE_TAU = 0.55
_ORACLE_FRACTION = 0.30
_CONFIDENCE_TRESHOLD= 1.35

ema_period           = 8
use_amp              = True
num_workers          = 4

def _fingerprint_payload(payload: dict) -> str:
    """Stable short fingerprint for caching."""
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

@torch.no_grad()
def collect_states_cached(
    total: int,
    split: Dict[str, float],
    *,
    cache_path: str | os.PathLike,
    overwrite: bool = False,
    device: Optional[torch.device] = None,
    shuffle: bool = True,
    max_plies_per_game: Optional[int] = None,
    enable_cache: bool = True,
    phase_split=(0.35, 0.45, 0.20),
    cache_filename: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Wrapper around collect_states that caches to disk.
    Returns: (states_cpu, mask_cpu, meta)
    """
    cache_dir = Path(cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build a fingerprint so different settings don't collide
    meta = {
        "version": "c4-distill-cache-v1",
        "total": int(total),
        "split": split,
        "shuffle": bool(shuffle),
        "max_plies_per_game": None if max_plies_per_game is None else int(max_plies_per_game),
        "enable_cache": bool(enable_cache),
        "phase_split": tuple(float(x) for x in phase_split),
    }
    fp = _fingerprint_payload(meta)
    fname = cache_filename or f"states_{total}_{fp}.pt"
    fpath = cache_dir / fname

    if fpath.exists() and not overwrite:
        blob = torch.load(fpath, map_location="cpu")
        states_cpu: torch.Tensor = blob["states"]  # (N,4,6,7) float32 on CPU
        mask_cpu: torch.Tensor   = blob["mask"]    # (N,7) bool on CPU
        cached_meta: dict        = blob.get("meta", {})
        tqdm.write(f"[cache] loaded {states_cpu.shape[0]:,} states from {str(fpath)}")
        return states_cpu, mask_cpu, cached_meta

    # Generate fresh
    states = collect_states(
        total, split,
        device=None,                 # keep on CPU to avoid big device copies
        shuffle=shuffle,
        max_plies_per_game=max_plies_per_game,
        enable_cache=enable_cache,
        phase_split=phase_split
    )  # (N,4,6,7) float32 CPU
    mask = legal_mask_from_states(states)  # (N,7) bool

    blob = {"states": states.cpu(), "mask": mask.cpu(), "meta": meta}
    torch.save(blob, fpath)
    tqdm.write(f"[cache] saved {states.shape[0]:,} states to {str(fpath)}")
    return blob["states"], blob["mask"], meta


def _oracle_cache_get(board_bytes, player, depth):
    key = (board_bytes, int(player), int(depth))
    val = _ORACLE_CACHE.get(key)
    if val is not None:
        _ORACLE_CACHE.move_to_end(key)
    return val

def _oracle_cache_put(board_bytes, player, depth, action):
    key = (board_bytes, int(player), int(depth))
    _ORACLE_CACHE[key] = int(action)
    if len(_ORACLE_CACHE) > _ORACLE_CACHE_MAX:
        _ORACLE_CACHE.popitem(last=False)  # oldest


@torch.no_grad()
def _infer_player_from_state_tensor(x4: torch.Tensor) -> int:
    c0 = int(x4[0].sum().item())
    c1 = int(x4[1].sum().item())
    return +1 if c0 == c1 else -1


def _get_oracle():
    global _ORACLE_LA
    if _ORACLE_LA is None:
        _ORACLE_LA = Connect4Lookahead()
    return _ORACLE_LA

@torch.no_grad()
def _oracle_actions_subset(X, mask, idxs, depth):
    if depth is None or depth <= 0 or len(idxs) == 0:
        return torch.full((X.shape[0],), -1, dtype=torch.long, device=X.device)

    LA = _get_oracle()
    out = torch.full((X.shape[0],), -1, dtype=torch.long, device=X.device)

    for i in idxs:
        xi = X[i]
        valid = mask[i].nonzero(as_tuple=False).squeeze(1).tolist()
        if not valid: 
            continue
        board = (xi[0] - xi[1]).detach().cpu().numpy().astype(np.int8)
        player = _infer_player_from_state_tensor(xi)
        bbytes = board.tobytes()

        a = _oracle_cache_get(bbytes, player, depth)
        if a is None:
            a = LA.n_step_lookahead(board, player=player, depth=depth)
            _oracle_cache_put(bbytes, player, depth, a)

        out[i] = a if a in valid else random.choice(valid)
    return out


@torch.no_grad()
def _margin_confidence(q: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    q_masked = q.masked_fill(~mask, -1e9)
    top2 = torch.topk(q_masked, k=2, dim=1).values  # (B,2)
    margin = (top2[:, 0] - top2[:, 1])
    # rows with <2 legal actions → treat as very confident (large margin)
    legal_counts = mask.sum(dim=1)
    margin = torch.where(legal_counts >= 2, margin, torch.as_tensor(5.0, device=q.device))
    margin = margin.clamp_min(0.0)
    mean_m = margin.mean().clamp_min(eps)
    return (margin / mean_m).unsqueeze(1)


@torch.no_grad()
def _sigmoid_mix(w: torch.Tensor, steer_to_A: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Nudges w toward A (1.0) or toward B (0.0) on rows where exactly one matches oracle.
    steer_to_A: (B,1) in {0, 1} but only meaningful where XOR(matchA, matchB)=True.
    tau in [0,1]: how much to trust the oracle on those rows.
    """
    if tau <= 0:
        return w
    return (1.0 - tau) * w + tau * steer_to_A

@torch.no_grad()
def _masked_argmax(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Argmax over legal actions only. q:(B,7), mask:(B,7)-> (B,) long"""
    qm = q.clone()
    qm[~mask] = -1e9
    return torch.argmax(qm, dim=1)

@torch.no_grad()
def sanity_eval(student: torch.nn.Module,
                teacherA: Optional[DQNAgent],
                teacherB: Optional[DQNAgent],
                states: torch.Tensor,
                *,
                alpha: float = 0.5,
                z_norm: bool = True,
                samples: int = 10000,
                batch_size: int = 512,
                device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Fast agreement & loss checks vs blended teachers on a subset of `states`.
    Iterations are tqdm-wrapped; summary via tqdm.write.
    Returns a dict of metrics.
    """

    if device is None:
        device = next(student.parameters()).device
    student.eval()
    mA = teacherA.model if teacherA is not None else None
    mB = teacherB.model if teacherB is not None else None
    if mA is not None: mA.eval()
    if mB is not None: mB.eval()

    N = min(samples, states.shape[0])
    # sample without replacement on CPU to avoid cuda rng sync
    idx = torch.randperm(states.shape[0])[:N]
    Xsub = states[idx]

    # accumulators
    mse_sum = 0.0
    huber_sum = 0.0
    n_masked = 0
    agree = 0
    illegal_pref = 0
    seen = 0

    # iterate in batches
    num_batches = (N + batch_size - 1) // batch_size
    pbar = tqdm(range(num_batches), desc="sanity batches", leave=True)
    for b in pbar:
        s = b * batch_size
        e = min(N, s + batch_size)
        X = Xsub[s:e].to(device, non_blocking=True)              # (B,4,6,7)
        Bsz = X.shape[0]
        
        mask = legal_mask_from_states(X)                         # (B,7)

        # targets
        if (mA is not None) and (mB is not None):
            targ = teacher_targets(mA, mB, X, alpha=alpha, z_norm=z_norm)
        elif mA is not None:
            targ = mA(X);  targ = z_norm_per_state(targ) if z_norm else targ
        elif mB is not None:
            targ = mB(X);  targ = z_norm_per_state(targ) if z_norm else targ
        else:
            raise ValueError("At least one teacher must be provided.")
        pred = student(X)

        # masked losses
        diff = pred - targ
        m = mask
        mse_sum += (diff[m]**2).sum().item()
        # SmoothL1 per-element (same delta as F.smooth_l1_loss defaults)
        huber_sum += F.smooth_l1_loss(pred[m], targ[m], reduction="sum").item()
        n_masked += int(m.sum().item())

        # top-1 agreement among legal actions
        a_pred = _masked_argmax(pred, mask)
        a_targ = _masked_argmax(targ, mask)
        agree += int((a_pred == a_targ).sum().item())

        # how often the raw argmax (without masking) is illegal (should be ~0)
        raw_arg = torch.argmax(pred, dim=1)                      # (B,)
        illegal_pref += int((~mask.gather(1, raw_arg.view(-1,1)).squeeze(1)).sum().item())

        seen += Bsz
        if (b+1) % 10 == 0 or b == 0:
            pbar.set_postfix(seen=seen, agree=f"{agree/seen:.3f}")
    

    metrics = {
        "masked_mse": mse_sum / max(1, n_masked),
        "masked_huber": huber_sum / max(1, n_masked),
        "top1_agreement": agree / max(1, seen),
        "illegal_pref_rate": illegal_pref / max(1, seen),
        "samples": float(seen),
    }
    tqdm.write(
        "[sanity] mse={masked_mse:.6f} huber={masked_huber:.6f} "
        "agree={top1_agreement:.3f} illegal_pref={illegal_pref_rate:.4f} (N={samples:.0f})"
        .format(**metrics)
    )
    
    return metrics


def save_student(student: torch.nn.Module, path: str) -> None:
    # write a *pure* state_dict (compatible with DQNAgent)
    torch.save(student.state_dict(), path)
    tqdm.write(f"[save] student weights -> {path}")

class _StateDatasetIdx(Dataset):
    """Returns (row_index, state). Lets us slice precomputed Qa/Qb quickly."""
    def __init__(self, states: torch.Tensor):
        self.states = states
    def __len__(self): return self.states.shape[0]
    def __getitem__(self, i): return i, self.states[i]

@torch.no_grad()
def _precompute_teacher_Q(states: torch.Tensor,
                          teacherA: Optional[torch.nn.Module],
                          teacherB: Optional[torch.nn.Module],
                          *,
                          z_norm: bool,
                          device: torch.device,
                          batch_size: int,
                          dtype: torch.dtype,
                          progress: bool = True):
    """
    Forward all states through teacher nets once. Returns (Qa_all, Qb_all) on CPU in `dtype`,
    or None if that teacher is not provided.
    """
    N = states.shape[0]
    Qa_all = None
    Qb_all = None

    # set up loaders to avoid copying everything to GPU at once
    ds = _StateDatasetIdx(states)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=0, pin_memory=(device.type == "cuda"),
                    drop_last=False)

    if teacherA is not None:
        Qa_all = torch.empty((N, 7), dtype=dtype, device="cpu")
    if teacherB is not None:
        Qb_all = torch.empty((N, 7), dtype=dtype, device="cpu")

    rng = tqdm(dl, desc="precompute teachers", leave=True) if progress else dl
    for idxs, X in rng:
        Xd = X.to(device, non_blocking=True)
        if teacherA is not None:
            Qa = teacherA(Xd)                           # (B,7) on device
            if z_norm: Qa = z_norm_per_state(Qa)
            Qa_all[idxs] = Qa.detach().to("cpu", dtype=dtype)
        if teacherB is not None:
            Qb = teacherB(Xd)
            if z_norm: Qb = z_norm_per_state(Qb)
            Qb_all[idxs] = Qb.detach().to("cpu", dtype=dtype)

    return Qa_all, Qb_all

@torch.no_grad()
def _row_hardness(Qa: Optional[torch.Tensor],
                  Qb: Optional[torch.Tensor],
                  mask: torch.Tensor,
                  eps: float = 1e-6) -> torch.Tensor:
    """
    Cheap per-row weight. If both teachers given, inverse of combined margin.
    If only one given, inverse of that teacher's margin.
    Returns (B,) normalized around 1.0 and clipped to [0.5, 2.0].
    """
    if Qa is not None and Qb is not None:
        ca = _margin_confidence(Qa, mask).squeeze(1)  # big = easy
        cb = _margin_confidence(Qb, mask).squeeze(1)
        w = 2.0 / (ca + cb + eps)                      # big = hard
    else:
        Q  = Qa if Qa is not None else Qb
        qM = Q.masked_fill(~mask, -1e9)
        top2 = torch.topk(qM, k=2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
        w = 1.0 / (margin + eps)
    w = w / (w.mean() + eps)
    return w.clamp(0.5, 2.0)

def masked_huber_weighted(pred: torch.Tensor,
                          target: torch.Tensor,
                          mask: torch.Tensor,
                          row_w: torch.Tensor) -> torch.Tensor:
    """
    Huber over legal actions, scaled per-row by row_w (B,).
    """
    per_elem = F.smooth_l1_loss(pred, target, reduction="none")
    w = row_w.view(-1, 1).expand_as(per_elem)
    return (per_elem * w)[mask].mean()

def kl_policy_loss(pred: torch.Tensor,
                   target: torch.Tensor,
                   mask: torch.Tensor,
                   T: float = 1.5,
                   eps: float = 1e-8) -> torch.Tensor:
    """
    KL( teacher || student ) over legal actions (masked + renormalized).
    """
    p = F.softmax(pred / T, dim=1)
    q = F.softmax(target / T, dim=1)
    p = (p * mask.float()); p = p / (p.sum(dim=1, keepdim=True) + eps)
    q = (q * mask.float()); q = q / (q.sum(dim=1, keepdim=True) + eps)
    kl = (q * (q.add(eps).log() - p.add(eps).log())).sum(dim=1)
    return kl.mean()

class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, opt, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-5, last_epoch: int = -1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps  = max(self.warmup_steps + 1, int(total_steps))
        self.min_lr = float(min_lr)
        super().__init__(opt, last_epoch)
    def get_lr(self):
        s = self.last_epoch + 1
        out = []
        for base_lr in self.base_lrs:
            if s <= self.warmup_steps:
                lr = base_lr * s / self.warmup_steps
            else:
                t = (s - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))
            out.append(lr)
        return out

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        model.load_state_dict(self.shadow, strict=False)
# -----------------------------------------------------------------------------


@torch.no_grad()
def _blend_targets(
    Qa: torch.Tensor, Qb: torch.Tensor, mask: torch.Tensor, X: torch.Tensor,
    *, dynamic_alpha: bool = True, alpha: float = 0.5,
    oracle_depth: int = _ORACLE_DEPTH,
    oracle_tau: float = _ORACLE_TAU,
    oracle_fraction: float = _ORACLE_FRACTION,
    conf_thresh: float = _CONFIDENCE_TRESHOLD
) -> torch.Tensor:
    """
    Blend precomputed teacher Q-tables (already z-normed if desired).
    Applies oracle steering on rows that are (disagree OR low-confidence).
    Qa/Qb: (B,7), mask: (B,7) bool, X: (B,4,6,7)
    """
    if Qa is None and Qb is None:
        raise ValueError("Need at least one teacher.")
    if Qa is None: return Qb
    if Qb is None: return Qa

    # base weights
    if dynamic_alpha:
        cA = _margin_confidence(Qa, mask)                 # (B,1)
        cB = _margin_confidence(Qb, mask)                 # (B,1)
        w  = (cA / (cA + cB + 1e-8)).clamp(0.05, 0.95)
    else:
        w  = torch.full((Qa.shape[0], 1), float(alpha), device=Qa.device)

    # oracle steering
    if oracle_depth > 0 and oracle_fraction > 0 and oracle_tau > 0:
        aA = _masked_argmax(Qa, mask)
        aB = _masked_argmax(Qb, mask)
        cA = _margin_confidence(Qa, mask)
        cB = _margin_confidence(Qb, mask)

        disagree = (aA != aB)

        # treat rows with <2 legal actions as high-confidence (skip oracle)
        legal_counts = mask.sum(dim=1)
        min_conf = torch.minimum(cA, cB).squeeze(1)
        amb = (min_conf < float(conf_thresh)) & (legal_counts >= 2)

        cand = (disagree | amb).nonzero(as_tuple=False).squeeze(1)
        if cand.numel() > 0:
            k = max(1, int(torch.ceil(torch.tensor(len(cand) * float(oracle_fraction))).item()))
            idxs = cand[torch.randperm(len(cand))[:k]].tolist()

            ao = _oracle_actions_subset(X=X, mask=mask, idxs=idxs, depth=int(oracle_depth))  # (B,)
            has_o = (ao >= 0)

            matchA = torch.zeros_like(aA, dtype=torch.bool)
            matchB = torch.zeros_like(aB, dtype=torch.bool)
            matchA[idxs] = (aA[idxs] == ao[idxs]) & has_o[idxs]
            matchB[idxs] = (aB[idxs] == ao[idxs]) & has_o[idxs]
            xor = matchA ^ matchB
            if xor.any():
                minm  = torch.minimum(cA.squeeze(1), cB.squeeze(1)).clamp_min(1e-6)
                boost = (1.6 / (minm + 0.6)).clamp(0.0, 1.0) * float(oracle_tau)
                steer = torch.zeros_like(w)
                steer[xor] = matchA[xor].float().unsqueeze(1)
                w[xor] = (1.0 - boost[xor].unsqueeze(1)) * w[xor] + boost[xor].unsqueeze(1) * steer[xor]

    return w * Qa + (1.0 - w) * Qb




def distill_train(student: torch.nn.Module,
                  teacherA: Optional[DQNAgent],
                  teacherB: Optional[DQNAgent],
                  states: torch.Tensor,
                  *,
                  # core
                  z_norm: bool = True,
                  alpha: float = 0.5,
                  batch_size: int = 2048,
                  epochs: int = 3,
                  lr: float = 1e-3,
                  weight_decay: float = 0.0,
                  grad_clip: Optional[float] = 1.0,
                  num_workers: int = 4,
                  device: Optional[torch.device] = None,
                  # speed/quality knobs
                  use_amp: bool = True,
                  ema_decay: float = 0.999,
                  ema_period: int = 8,
                  mirror_prob: float = 0.50,
                  # regularizers
                  kl_weight: float = 0.05,
                  T_kl: float = 1.5,
                  illegal_weight: float = 0.05,
                  illegal_margin: float = 0.0,
                  # scheduler
                  warmup_steps: int = 200,
                  total_steps: Optional[int] = None,
                  min_lr: float = 1e-5,
                  # NEW: built-in precompute
                  precompute: str = "auto",        # "auto" | True | False
                  precompute_dtype: str = "fp32",  # "fp16" | "fp32"
                  max_precompute_bytes: int = 1_200_000_000  # ~1.2GB guard
                  ) -> None:
    """
    Distill `student` on `states` with optional teacher precompute.

    precompute:
      - "auto": estimate memory; precompute if it fits
      - True: always precompute
      - False: never precompute

    precompute_dtype:
      - "fp16" (default) halves memory; good for targets
      - "fp32" for maximum fidelity (slower/heavier)
    """
    if device is None:
        device = next(student.parameters()).device
    student.train()

    mA = teacherA.model.eval() if teacherA is not None else None
    mB = teacherB.model.eval() if teacherB is not None else None

    # --- precompute legal mask once (CPU) ---
    mask_all = legal_mask_from_states(states).cpu()  # (N,7) bool

    # --- decide precompute plan ---
    N = states.shape[0]
    n_teachers = (1 if mA is not None else 0) + (1 if mB is not None else 0)
    bytes_per = 2 if precompute_dtype == "fp16" else 4
    need_bytes = N * 7 * bytes_per * n_teachers

    do_precomp = (precompute is True) or (precompute == "auto" and need_bytes <= max_precompute_bytes)
    dtype_cpu = torch.float16 if precompute_dtype == "fp16" else torch.float32

    # --- optional precompute ---
    Qa_all = Qb_all = None
    if do_precomp and n_teachers > 0:
        Qa_all, Qb_all = _precompute_teacher_Q(
            states=states,
            teacherA=mA, teacherB=mB,
            z_norm=z_norm,
            device=device,
            batch_size=max(1024, batch_size),  # push large for speed
            dtype=dtype_cpu,
            progress=True
        )

    ds = _StateDatasetIdx(states)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers,
                    pin_memory=(device.type == "cuda"),
                    drop_last=False)

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(ds) / batch_size)
    if total_steps is None:
        total_steps = steps_per_epoch * epochs
    sch = CosineWithWarmup(opt, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr)

    scaler = amp.GradScaler(device_type="cuda")
    ema = EMA(student, decay=ema_decay)
    step_idx = 0

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", leave=True):
        running, seen = 0.0, 0
        with tqdm(total=steps_per_epoch, desc=f"epoch {epoch}/{epochs}", leave=True) as pbar:
            for idxs, X in dl:
                # pull precomputed mask (CPU) and move as needed
                mask = mask_all[idxs]  # CPU bool
                X = X.to(device, non_blocking=True)

                # optional mirror augmentation: flip state, mask, and precomputed teacher Q
                do_mirror = (mirror_prob > 0 and torch.rand(1).item() < mirror_prob)
                if do_mirror:
                    X = X.flip(dims=(3,))
                    mask = mask.flip(dims=(1,))

                with torch.no_grad():
                    if Qa_all is not None or Qb_all is not None:
                        # slice precomputed and ship to device
                        Qa = Qa_all[idxs].to(device, dtype=torch.float32) if Qa_all is not None else None
                        Qb = Qb_all[idxs].to(device, dtype=torch.float32) if Qb_all is not None else None
                        if do_mirror:
                            if Qa is not None: Qa = Qa.flip(dims=(1,))
                            if Qb is not None: Qb = Qb.flip(dims=(1,))
                    else:
                        # on-the-fly teacher forward
                        Qa = mA(X) if mA is not None else None
                        Qb = mB(X) if mB is not None else None
                        if z_norm and Qa is not None: Qa = z_norm_per_state(Qa)
                        if z_norm and Qb is not None: Qb = z_norm_per_state(Qb)

                    # NOTE: mask is CPU—keep as such for cheaper boolean ops; move later if needed
                    mask_d = mask.to(device, non_blocking=True)

                    target = _blend_targets(
                        Qa, Qb, mask_d, X,
                        dynamic_alpha=True,
                        alpha=alpha,
                        oracle_depth=_ORACLE_DEPTH,
                        oracle_tau=_ORACLE_TAU,
                        oracle_fraction=_ORACLE_FRACTION,
                        conf_thresh=_CONFIDENCE_TRESHOLD
                    )

                    # row hardness weights from teacher Q
                    row_w = _row_hardness(Qa, Qb, mask_d)    # (B,)

                # --- student update ---
                opt.zero_grad(set_to_none=True)
                
                with amp.autocast(
                    device_type="cuda",
                    enabled=(use_amp and device.type == "cuda"),
                    dtype=torch.float16,
                ):
                    pred = student(X)
                    loss = masked_huber_weighted(pred, target, mask_d, row_w)
                
                    if kl_weight > 0:
                        loss = loss + kl_weight * kl_policy_loss(pred, target, mask_d, T=T_kl)
                
                    if illegal_weight > 0 and (~mask_d).any():
                        min_legal = pred.masked_fill(~mask_d, float("inf")).min(dim=1).values.unsqueeze(1)
                        over = F.relu(pred - (min_legal - illegal_margin))
                        illegal_pen = (over[~mask_d] ** 2).mean()
                        loss = loss + illegal_weight * illegal_pen
                
                scaler.scale(loss).backward()
                
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                
                # Track whether optimizer actually stepped
                prev_steps = getattr(opt, "_step_count", 0)
                scaler.step(opt)
                scaler.update()
                did_step = getattr(opt, "_step_count", 0) > prev_steps
                
                # Only advance the scheduler if we truly stepped
                if did_step:
                    sch.step()
                
                # EMA after the update
                if (step_idx % max(1, int(ema_period))) == 0:
                    ema.update(student)
                step_idx += 1

                running += float(loss.item())
                seen += 1
                if (seen % 10) == 0 or seen == 1:
                    pbar.set_postfix(loss=f"{running/seen:.4f}")
                pbar.update(1)

        tqdm.write(f"[distill] epoch {epoch}: avg_loss={running/max(1,seen):.5f}, steps={seen}")

    # keep online weights; if you prefer EMA weights for export:
    # ema.copy_to(student)
    student.eval()
    tqdm.write("[distill] training complete.")

# -------------------- Teacher loading (wrap existing loader) --------------------
def load_teacher(path: str,
                 device: Optional[torch.device] = None,
                 epsilon: float = 0.0,
                 guard_prob: float = 0.0) -> DQNAgent:
    """
    Thin wrapper around agent_eval.load_agent_from_ckpt for symmetry.
    Returns the loaded agent in eval mode.
    """
    agent = load_agent_from_ckpt(path, device=device, epsilon=epsilon, guard_prob=guard_prob)
    # Ensure pure eval behavior
    agent.model.eval(); agent.target_model.eval()
    agent.epsilon = agent.epsilon_min = float(epsilon)
    agent.guard_prob = float(guard_prob)
    return agent


# -------------------- Masks & normalization ------------------------------------
@torch.no_grad()
def legal_mask_from_states(X: torch.Tensor) -> torch.Tensor:
    """
    X: (N, 4, 6, 7)  →  bool mask (N, 7) where top cell empty => legal.
    Assumes channel 0 is +1 plane, channel 1 is -1 plane.
    """
    return (X[:, 0, 0, :] + X[:, 1, 0, :]) == 0


def z_norm_per_state(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-row (per-state) z-normalization: (q - mean)/std.
    q: (N, 7) or (B, 7)
    """
    m = q.mean(dim=1, keepdim=True)
    s = q.std(dim=1, keepdim=True).clamp_min(eps)
    return (q - m) / s


# -------------------- Teacher target blending ----------------------------------
@torch.no_grad()
def teacher_targets(modelA: torch.nn.Module, modelB: torch.nn.Module,
                    X: torch.Tensor, alpha: float = 0.5, z_norm: bool = True) -> torch.Tensor:
    """
    Compute blended teacher targets: Q* = α QA + (1-α) QB
    Optionally z-normalize QA, QB per-state to reduce scale mismatches.
    """
    Qa = modelA(X)
    Qb = modelB(X)
    if z_norm:
        Qa = z_norm_per_state(Qa)
        Qb = z_norm_per_state(Qb)
    return alpha * Qa + (1.0 - alpha) * Qb


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    SmoothL1 loss over legal columns only.
    pred/target: (B, 7)
    mask:        (B, 7) bool
    """
    pred_m = pred[mask]
    targ_m = target[mask]
    if pred_m.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.smooth_l1_loss(pred_m, targ_m, reduction="mean")


# -------------------- State collection via scripted opponents ------------------
def _scripted_move(label: str, board: np.ndarray, player: int,
                   valid_actions: Iterable[int], LA: Connect4Lookahead) -> int:
    """
    Select an action for the scripted opponent bucket (Random | Lookahead-k).
    `board` is 2D int array (6x7) with {+1,-1,0}, `player` is current player sign (+1/-1).
    """
    valid_actions = list(valid_actions)
    if label == "Random":
        return random.choice(valid_actions)
    # label like "Lookahead-1", "Lookahead-2", "Lookahead-3"
    depth = int(label.split("-")[1])
    a = LA.n_step_lookahead(board, player=player, depth=depth)
    return a if a in valid_actions else random.choice(valid_actions)


def _log_clean(msg: str):
    # prints atomically with tqdm, never mangles bars
    tqdm.write(msg)



@torch.no_grad()
def collect_states(total: int, split: Dict[str, float],
                   device: Optional[torch.device] = None,
                   shuffle: bool = True,
                   max_plies_per_game: Optional[int] = None,
                   enable_cache: bool = True,
                   phase_split=(0.35, 0.45, 0.20)) -> torch.Tensor:
    """
    Collect `total` states across scripted buckets in `split`, with late-game upweighting.
    split example: {"Random":0.40, "Lookahead-1":0.30, "Lookahead-2":0.20, "Lookahead-3":0.10}
    phase_split: probability of keeping a ply from (early, mid, late).
    """
    env = Connect4Env()
    LA  = Connect4Lookahead()

    target_counts = {k: int(round(total * v)) for k, v in split.items()}
    # fix rounding drift
    diff = total - sum(target_counts.values())
    if diff != 0:
        kmax = max(target_counts, key=target_counts.get)
        target_counts[kmax] += diff

    def _cached_LA(board2d: np.ndarray, player: int, depth: int, valids):
        if not enable_cache:
            a = LA.n_step_lookahead(board2d, player=player, depth=depth)
            return a if a in valids else random.choice(valids)
        key = (board2d.tobytes(), int(player), int(depth))
        a = _ORACLE_CACHE.get(key)
        if a is None:
            a = LA.n_step_lookahead(board2d, player=player, depth=depth)
            _ORACLE_CACHE[key] = int(a)
            if len(_ORACLE_CACHE) > _ORACLE_CACHE_MAX:
                _ORACLE_CACHE.popitem(last=False)
        return a if a in valids else random.choice(valids)

    buffers = {k: [] for k in split.keys()}
    labels = list(target_counts.keys())

    def _phase_idx(ply:int) -> int:
        # rough phases; adjust thresholds if you want
        if ply < 8: return 0      # early
        if ply < 20: return 1     # mid
        return 2                   # late

    with tqdm(total=len(labels), desc="Buckets", position=0, leave=True) as outer:
        for label in labels:
            need = target_counts[label]
            with tqdm(total=need, desc=f"{label} states", position=1, leave=True) as pbar:
                while len(buffers[label]) < need:
                    s = env.reset()
                    done, ply = False, 0
                    while not done:
                        # keep this state with phase-dependent probability
                        if len(buffers[label]) < need:
                            if random.random() < phase_split[_phase_idx(ply)]:
                                buffers[label].append(s.copy()); pbar.update(1)
                                if len(buffers[label]) >= need:
                                    break
                        if (max_plies_per_game is not None) and (ply >= max_plies_per_game):
                            break

                        valids = env.available_actions()
                        board  = (s[0] - s[1]).astype(np.int8)
                        if label == "Random":
                            a = random.choice(valids)
                        else:
                            depth = int(label.split("-")[1])
                            a = _cached_LA(board, env.current_player, depth, valids)

                        s, _, done = env.step(a)
                        ply += 1
            outer.update(1)

    all_states = np.concatenate([np.stack(v, axis=0) for v in buffers.values()], axis=0)
    if shuffle:
        idx = np.arange(all_states.shape[0]); np.random.shuffle(idx)
        all_states = all_states[idx]

    out = torch.tensor(all_states, dtype=torch.float32, device=device) if device is not None \
          else torch.tensor(all_states, dtype=torch.float32)
    tqdm.write(f"[collect] target_counts={target_counts}  total={out.shape[0]:,}")
    return out


# -------------------- Student creation ----------------------------------------
def build_student(init_from: Optional[str] = None,
                  teacherA: Optional[DQNAgent] = None,
                  teacherB: Optional[DQNAgent] = None,
                  device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Create a fresh DQN student, optionally initialized from a teacher's weights.
      init_from: None | "A" | "B"
    """
    student = DQN().to(device if device is not None else torch.device("cpu"))
    if init_from == "A" and teacherA is not None:
        student.load_state_dict(teacherA.model.state_dict(), strict=False)
    elif init_from == "B" and teacherB is not None:
        student.load_state_dict(teacherB.model.state_dict(), strict=False)
    return student


# -------------------- Merit-based alpha (optional helper) ----------------------
def alpha_from_merit(weights: Dict[str, float],
                     metrics_A: Dict[str, float],
                     metrics_B: Dict[str, float]) -> float:
    """
    Compute α = J_A / (J_A + J_B) from per-opponent win-rates, given weights.
    Example:
      weights = {"R":0.15, "L1":0.25, "L2":0.45, "L3":0.15}
      metrics_A = {"R":0.773, "L1":0.0, "L2":0.302, "L3":0.182}
      metrics_B = {"R":0.768, "L1":0.079, "L2":0.0, "L3":0.0}
    """
    JA = sum(weights[k] * metrics_A.get(k, 0.0) for k in weights)
    JB = sum(weights[k] * metrics_B.get(k, 0.0) for k in weights)
    if JA + JB <= 0:
        return 0.5
    return float(JA / (JA + JB))


