# DQN.distill_helpers.py

from __future__ import annotations
import random
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

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
_ORACLE_CACHE_MAX = 200_000  # tune to RAM

_ORACLE_DEPTH = 4                   #2->4
_ORACLE_TAU = 0.35                  #0.33 -->0.35
_ORACLE_FRACTION = 0.25             #0.1 --> 0.2
_CONFIDENCE_TRESHOLD= 1.3           #1.0 --> 1.2

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
def _oracle_actions_batch(X, mask, depth=3, fraction=0.25):
    if depth is None or depth <= 0 or fraction <= 0:
        return torch.full((X.shape[0],), -1, dtype=torch.long, device=X.device)
    LA = _get_oracle()
    B = X.shape[0]
    use_idx = (torch.rand(B) < float(fraction))   # CPU sampling to avoid CUDA sync
    idxs = use_idx.nonzero(as_tuple=False).squeeze(1).tolist()
    out = torch.full((B,), -1, dtype=torch.long, device=X.device)
    for i in idxs:
        xi = X[i]
        valid = mask[i].nonzero(as_tuple=False).squeeze(1).tolist()
        if not valid: continue
        board = (xi[0] - xi[1]).detach().cpu().numpy().astype(np.int8)
        player = _infer_player_from_state_tensor(xi)
        a = LA.n_step_lookahead(board, player=player, depth=depth)
        out[i] = a if a in valid else random.choice(valid)
    return out

@torch.no_grad()
def _margin_confidence(q: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    q:(B,7), mask:(B,7)-> per-row confidence = (top1 - top2) among LEGAL actions, normalized.
    Returns c in (B,1).
    """
    q_masked = q.masked_fill(~mask, -1e9)
    top2 = torch.topk(q_masked, k=2, dim=1).values  # (B,2)
    margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)  # (B,)
    # Normalize by batch mean margin for stability
    mean_m = margin.mean().clamp_min(eps)
    c = (margin / mean_m).unsqueeze(1)  # (B,1)
    return c

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

class _StateDataset(Dataset):
    def __init__(self, states: torch.Tensor):
        # states: (N, 4, 6, 7), on CPU or device; we’ll move per-batch.
        self.states = states
    def __len__(self): return self.states.shape[0]
    def __getitem__(self, i): return self.states[i]

def _as_teacher_tuple(teacherA: Optional[DQNAgent], teacherB: Optional[DQNAgent]):
    mA = teacherA.model if teacherA is not None else None
    mB = teacherB.model if teacherB is not None else None
    if mA is not None: mA.eval()
    if mB is not None: mB.eval()
    return mA, mB

def _blend_targets(mA, mB, X, alpha, z_norm, *, mask=None,
                   dynamic_alpha=True,
                   oracle_depth=None, oracle_tau=0.25, oracle_fraction=0.25,
                   conf_thresh=0.8):
    Qa = mA(X) if mA is not None else None
    Qb = mB(X) if mB is not None else None
    if Qa is None and Qb is None: raise ValueError("Need a teacher.")
    if z_norm and Qa is not None: Qa = z_norm_per_state(Qa)
    if z_norm and Qb is not None: Qb = z_norm_per_state(Qb)
    if Qa is not None and Qb is None: return Qa
    if Qb is not None and Qa is None: return Qb

    if mask is None:
        mask = legal_mask_from_states(X)

    # dynamic alpha (per-state)
    if dynamic_alpha:
        cA = _margin_confidence(Qa, mask)          # (B,1)
        cB = _margin_confidence(Qb, mask)          # (B,1)
        w  = (cA / (cA + cB + 1e-8)).clamp(0.05, 0.95)
    else:
        w  = torch.full((X.shape[0], 1), float(alpha), device=X.device)

    # === Oracle gating to save time ===
    if oracle_depth and oracle_fraction > 0 and oracle_tau > 0:
        aA = _masked_argmax(Qa, mask)              # (B,)
        aB = _masked_argmax(Qb, mask)              # (B,)
        disagree = (aA != aB)

        # "ambiguous" if the weaker of the two margins is small
        amb = torch.minimum(cA, cB).squeeze(1) < float(conf_thresh)

        # only consider rows that are disagree OR ambiguous
        cand = (disagree | amb).nonzero(as_tuple=False).squeeze(1)
        if cand.numel() > 0:
            # sample a fraction for cost control
            k = max(1, int(torch.ceil(torch.tensor(len(cand) * float(oracle_fraction))).item()))
            # CPU sampling to avoid sync
            perm = torch.randperm(len(cand)).cpu().numpy().tolist()
            idxs = cand[perm[:k]].tolist()

            ao = _oracle_actions_subset(X, mask, idxs, depth=int(oracle_depth))  # (B,)
            has_o = (ao >= 0)

            matchA = (aA == ao) & has_o
            matchB = (aB == ao) & has_o
            xor    = matchA ^ matchB

            if xor.any():
                steer_to_A = torch.zeros_like(w)
                steer_to_A[xor] = matchA[xor].float().unsqueeze(1)
                w = (1.0 - oracle_tau) * w + oracle_tau * steer_to_A

    return w * Qa + (1.0 - w) * Qb




def distill_train(student: torch.nn.Module,
                  teacherA: Optional[DQNAgent],
                  teacherB: Optional[DQNAgent],
                  states: torch.Tensor,
                  *,
                  alpha: float = 0.5,
                  z_norm: bool = True,
                  batch_size: int = 2048,
                  epochs: int = 3,
                  lr: float = 1e-3,
                  weight_decay: float = 0.0,
                  grad_clip: Optional[float] = 1.0,
                  num_workers: int = 0,
                  device: Optional[torch.device] = None) -> None:
    """
    Distill student on a static corpus of states.
    All iterations wrapped in tqdm; logs use tqdm.write to avoid mangling.
    """
    if device is None:
        device = next(student.parameters()).device
    student.train()

    mA, mB = _as_teacher_tuple(teacherA, teacherB)

    ds = _StateDataset(states)  # states can live on CPU; we transfer per batch
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True if device.type == "cuda" else False,
                    drop_last=False)

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    
    # --- illegal-margin penalty (keeps illegal actions below the worst legal one) ---
    illegal_weight = 0.05   # start small (0.02–0.10 is typical)
    illegal_margin = 0.0    # or 0.1 if not z-norming


    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", position=0, leave=True):
        running_loss, steps = 0.0, 0

        with tqdm(total=len(dl), desc=f"epoch {epoch} / {epochs}", position=1, leave=True) as pbar:
            for X in dl:
                X = X.to(device, non_blocking=True)                # (B,4,6,7)
                mask = legal_mask_from_states(X)                   # (B,7) bool

                with torch.no_grad():
                    target = _blend_targets(
                        mA, mB, X, alpha, z_norm,
                        mask=mask,                   # <— pass it in
                        dynamic_alpha=True,
                        oracle_depth=_ORACLE_DEPTH,
                        oracle_tau=_ORACLE_TAU,
                        oracle_fraction=_ORACLE_FRACTION,
                        conf_thresh=_CONFIDENCE_TRESHOLD     # oracle only when margins are small or teachers disagree
                    )


                pred = student(X)                                   # (B,7)
                loss = masked_huber(pred, target, mask)

                if illegal_weight > 0:
                    # min legal Q per row
                    min_legal = pred.masked_fill(~mask, float('inf')).min(dim=1).values.unsqueeze(1)  # (B,1)
                    # how much illegal Q exceeds (min legal - margin)
                    over = F.relu(pred - (min_legal - illegal_margin))                                # (B,7)
                    # penalize only illegal positions
                    if (~mask).any():
                        illegal_penalty = (over[~mask] ** 2).mean()
                        loss = loss + illegal_weight * illegal_penalty


                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                opt.step()

                steps += 1
                running_loss += float(loss.item())
                if steps % 10 == 0 or steps == 1:
                    pbar.set_postfix(loss=f"{running_loss/steps:.4f}")
                pbar.update(1)

        avg = running_loss / max(1, steps)
        tqdm.write(f"[distill] epoch {epoch}: avg_loss={avg:.5f}, batches={steps}")

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
                   enable_cache: bool = True) -> torch.Tensor:
    env = Connect4Env()
    LA  = Connect4Lookahead()

    target_counts = {k: int(round(total * v)) for k, v in split.items()}
    diff = total - sum(target_counts.values())
    if diff != 0:
        kmax = max(target_counts, key=target_counts.get)
        target_counts[kmax] += diff

    _log_clean(f"[collect] target_counts={target_counts} (sum={sum(target_counts.values())})")

    buffers = {k: [] for k in split.keys()}
    cache = {} if enable_cache else None

    def _cached_lookahead(board2d: np.ndarray, player: int, depth: int, valid_actions):
        if cache is None:
            a = LA.n_step_lookahead(board2d, player=player, depth=depth)
            return a if a in valid_actions else random.choice(valid_actions)
        key = (board2d.tobytes(), int(player), int(depth))
        a = cache.get(key)
        if a is None:
            a = LA.n_step_lookahead(board2d, player=player, depth=depth)
            cache[key] = a
        return a if a in valid_actions else random.choice(valid_actions)

    labels = list(target_counts.keys())
    # leave=False ensures bars are cleared before the next print happens
    with tqdm(total=len(labels), desc="Buckets", position=0, leave=True) as outer:
        for label in labels:
            need = target_counts[label]
            with tqdm(total=need, desc=f"{label} states", position=1, leave=True) as pbar:
                while len(buffers[label]) < need:
                    s = env.reset()
                    done = False
                    ply = 0
                    while not done:
                        if len(buffers[label]) < need:
                            buffers[label].append(s.copy())
                            pbar.update(1)
                            if len(buffers[label]) >= need:
                                break
                        ply += 1
                        if (max_plies_per_game is not None) and (ply >= max_plies_per_game):
                            break
                        valid = env.available_actions()
                        board = (s[0] - s[1]).astype(np.int8)
                        if label == "Random":
                            a = random.choice(valid)
                        else:
                            depth = int(label.split("-")[1])
                            a = _cached_lookahead(board, env.current_player, depth, valid)
                        s, _, done = env.step(a)
            outer.update(1)

    all_states = np.concatenate([np.stack(v, axis=0) for v in buffers.values()], axis=0)
    if shuffle:
        idx = np.arange(all_states.shape[0]); np.random.shuffle(idx)
        all_states = all_states[idx]

    out = torch.tensor(all_states, dtype=torch.float32, device=device) if device is not None \
          else torch.tensor(all_states, dtype=torch.float32)

    _log_clean(f"[collect] Collected {out.shape[0]:,} states."
               + (f"  [cache={len(cache):,}]" if enable_cache else ""))

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

