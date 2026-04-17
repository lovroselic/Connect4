# DQN.replay_inspect.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def _to_float(x) -> float:
    # robust float conversion (supports torch, numpy scalars)
    try:
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
    except Exception: pass
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x).reshape(()).item())

def _np(a):
    return np.asarray(a, dtype=float)

def _slice_flags(flags, n):
    return _np(flags[:n]).astype(bool) if flags is not None and n > 0 else np.zeros((0,), dtype=bool)

def _stats_block(name: str, arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"set": name, "n": 0, "mean": np.nan, "std": np.nan, "min": np.nan,
                "q05": np.nan, "q25": np.nan, "median": np.nan, "q75": np.nan, "q95": np.nan, "max": np.nan}
    q = np.quantile(arr, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "set": name,
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(q[0]), "q05": float(q[1]), "q25": float(q[2]),
        "median": float(q[3]), "q75": float(q[4]), "q95": float(q[5]), "max": float(q[6]),
    }

def _auto_bins_float(x: np.ndarray, nbins: int | None):
    if x.size == 0: return "auto"
    if isinstance(nbins, int) and nbins > 0:
        return nbins
    return "auto"

def snapshot_reward_dists(
    memory,
    *,
    show_seed_splits: bool = True,
    bins_1step: int | None = None,
    bins_nstep:  int | None = None,
    title_prefix: str = "Replay buffer",
    preview_k: int = 8,
    return_stats_df: bool = True,
):
    """
    Inspect replay rewards (already *scaled* as stored in memory).

    Parameters
    ----------
    memory : PrioritizedReplayMemory
        Your PER instance (must expose bank_1/bank_n lists and is_seed_1/is_seed_n flags).
    show_seed_splits : bool
        If True, draw extra histograms for seeds only when present.
    bins_1step, bins_nstep : Optional[int]
        Number of bins for histograms (None → 'auto').
    title_prefix : str
        Figure title prefix.
    preview_k : int
        Rows to preview from each bank (head).
    return_stats_df : bool
        If True, returns (stats_df, preview_df); else returns None.

    Returns
    -------
    (stats_df, preview_df) or None
    """
    # --- collect rewards (robust to tensors) ---
    r1 = _np([_to_float(t.reward)   for t in getattr(memory, "bank_1", [])])
    rn = _np([_to_float(t.reward_n) for t in getattr(memory, "bank_n", [])])

    n1, nn = r1.size, rn.size
    s1 = _slice_flags(getattr(memory, "is_seed_1", None), n1)
    sn = _slice_flags(getattr(memory, "is_seed_n", None), nn)

    # --- quick console snapshot ---
    print("=== Memory snapshot ===")
    print(f"bank_1 (1-step): n={n1}  seeds={int(s1.sum()) if s1.size else 0}")
    print(f"bank_n (n-step): n={nn}  seeds={int(sn.sum()) if sn.size else 0}")
    for fn in ["alpha", "eps", "beta", "init_percentile", "init_prio_cap"]:
        if hasattr(memory, fn): print(f"{fn}={getattr(memory, fn)}", end="  ")
    print("\n")

    # --- stats table ---
    rows = [
        _stats_block("r1_all", r1),
        _stats_block("rn_all", rn),
    ]
    if show_seed_splits and s1.size and s1.any():
        rows.append(_stats_block("r1_seed", r1[s1]))
        rows.append(_stats_block("r1_nonseed", r1[~s1]))
    if show_seed_splits and sn.size and sn.any():
        rows.append(_stats_block("rn_seed", rn[sn]))
        rows.append(_stats_block("rn_nonseed", rn[~sn]))

    stats_df = pd.DataFrame(rows).set_index("set")
    print(stats_df.to_string(float_format=lambda v: f"{v:.4f}"))

    # --- plots: 1-step ---
    if n1:
        plt.figure(figsize=(8, 4.5))
        plt.hist(r1, bins=_auto_bins_float(r1, bins_1step), edgecolor="black")
        plt.axvline(r1.mean(),   linestyle="--", linewidth=2, label=f"mean={r1.mean():.4f}")
        plt.axvline(np.median(r1), linestyle=":",  linewidth=2, label=f"median={np.median(r1):.4f}")
        plt.title(f"{title_prefix}: 1-step rewards (scaled floats)")
        plt.xlabel("reward"); plt.ylabel("count")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    if n1 and show_seed_splits and s1.size and s1.any():
        plt.figure(figsize=(8, 4.5))
        plt.hist(r1[s1], bins=_auto_bins_float(r1[s1], bins_1step), edgecolor="black")
        mu, md = r1[s1].mean(), np.median(r1[s1])
        plt.axvline(mu, linestyle="--", linewidth=2, label=f"mean={mu:.4f}")
        plt.axvline(md, linestyle=":",  linewidth=2, label=f"median={md:.4f}")
        plt.title(f"{title_prefix}: 1-step rewards (SEEDS only)")
        plt.xlabel("reward"); plt.ylabel("count")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    # --- plots: N-step ---
    if nn:
        plt.figure(figsize=(8, 4.5))
        plt.hist(rn, bins=_auto_bins_float(rn, bins_nstep), edgecolor="black")
        plt.axvline(rn.mean(),   linestyle="--", linewidth=2, label=f"mean={rn.mean():.4f}")
        plt.axvline(np.median(rn), linestyle=":",  linewidth=2, label=f"median={np.median(rn):.4f}")
        plt.title(f"{title_prefix}: N-step returns (scaled, discounted)")
        plt.xlabel("n-step return"); plt.ylabel("count")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    # --- small preview table (head) ---
    preview_df = pd.DataFrame({
        "r1": pd.Series(r1[:preview_k]),
        "r1_seed": pd.Series(s1[:preview_k]) if s1.size else pd.Series(dtype=bool),
        "rn": pd.Series(rn[:preview_k]),
        "rn_seed": pd.Series(sn[:preview_k]) if sn.size else pd.Series(dtype=bool),
    })

    return (stats_df, preview_df) if return_stats_df else None
