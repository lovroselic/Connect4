#DQN.TD_error.py

import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import Counter

def plot_td_error_hist(agent, bins=80):
    td = np.array(agent.td_hist, dtype=np.float32)
    if td.size == 0:
        print("No TD data yet."); return
    plt.figure(figsize=(8,4))
    plt.hist(td, bins=bins, density=True, alpha=0.8)
    plt.title(f"TD-error distribution (mean={td.mean():.4f}, |td|={np.mean(np.abs(td)):.4f})")
    plt.xlabel("TD error"); plt.ylabel("Density"); plt.grid(True, ls='--', alpha=0.4)
    plt.show()

def plot_td_running(agent, window=500):
    td = np.array(agent.td_hist, dtype=np.float32)
    if td.size < window:
        print("Not enough TD samples for running plot."); return
    abs_td = np.abs(td)
    k = window
    run = np.convolve(abs_td, np.ones(k)/k, mode='valid')
    plt.figure(figsize=(10,3.5))
    plt.plot(run)
    plt.title(f"|TD| running mean (window={window}) ↓ should trend down / stabilize")
    plt.xlabel("sample idx"); plt.ylabel("running |td|"); plt.grid(True, ls='--', alpha=0.4)
    plt.show()

def plot_is_weights(agent, bins=60, return_figs=False):
    w = np.array(agent.per_w_hist, dtype=np.float32)
    if w.size == 0:
        print("No IS weights yet."); return
    fig_histogram = plt.figure(figsize=(8,4))
    plt.hist(w, bins=bins, density=True, alpha=0.8)
    plt.title(f"Importance weights (mean={w.mean():.3f}, max={w.max():.3f})")
    plt.xlabel("w") 
    plt.ylabel("Density") 
    plt.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    
    if return_figs:
        return fig_histogram
    else: 
        plt.show()

def check_per_priority_correlation(agent, sample=2048):
    # draw a mixed batch and compute |td| vs. updated priority to see correlation
    if (len(agent.memory.bank_1)+len(agent.memory.bank_n)) < sample: print("Not enough memory to probe."); return
    (b1, bn), (i1, in_), (w1, wn) = agent.memory.sample_mixed(sample, mix=0.5, beta=agent.per_beta)
    batch = list(b1)+list(bn)
    # Quick forward pass to compute provisional td
    states, next_states, s_players, ns_players, actions, rewards, dones, gpow = [],[],[],[],[],[],[],[]
    for t in batch:
        if hasattr(t, "reward_n"):
            s,a,rN,nsN,dN,p,n_used=t
            p_next = p if (n_used % 2 == 0) else -p
            states.append(s); next_states.append(nsN)
            s_players.append(p); ns_players.append(p_next)
            actions.append(a); rewards.append(rN*agent.reward_scale)
            dones.append(bool(dN)); gpow.append(agent.gamma**n_used)
        else:
            s,a,r,ns,d,p=t
            p_next = -p
            states.append(s); next_states.append(ns)
            s_players.append(p); ns_players.append(p_next)
            actions.append(a); rewards.append(r*agent.reward_scale)
            dones.append(bool(d)); gpow.append(agent.gamma)
    def pack(boards, players):
        planes = [agent._agent_planes(b, p) for b,p in zip(boards,players)]
        return torch.tensor(np.asarray(planes), dtype=torch.float32, device=agent.device)
    X = pack(states, s_players); Xn = pack(next_states, ns_players)
    a_t = torch.tensor(actions, dtype=torch.long, device=agent.device).view(-1,1)
    r_t = torch.tensor(rewards, dtype=torch.float32, device=agent.device)
    d_t = torch.tensor(dones, dtype=torch.bool, device=agent.device)
    g_t = torch.tensor(gpow, dtype=torch.float32, device=agent.device)

    with torch.no_grad():
        q_all = agent.model(X); q_sa = q_all.gather(1, a_t).squeeze(1)
        next_a = agent.model(Xn).argmax(dim=1, keepdim=True)
        next_q = agent.target_model(Xn).gather(1, next_a).squeeze(1)
        target = r_t + (~d_t).float() * g_t * next_q
        td = (q_sa - target).detach().cpu().numpy()
    # Correlation proxy: higher |td| should map to higher priority after update
    print(f"Probe |td|: mean={np.mean(np.abs(td)):.4f}, std={np.std(td):.4f}")

def show_sample_mix(agent, last_k=5000):
    if len(agent.sample_mix_hist)==0:
        print("No sample mix yet."); return
    arr = list(agent.sample_mix_hist)[-last_k:]
    is_n, p = zip(*arr)
    cnt_n = Counter(is_n); cnt_p = Counter(p)
    total = len(arr)
    print(f"Sample mix (last {total}): n-step={cnt_n.get(1,0)/total:.2%}, 1-step={cnt_n.get(0,0)/total:.2%};  p=+1={cnt_p.get(1,0)/total:.2%}, p=-1={cnt_p.get(-1,0)/total:.2%}")
    

def plot_td_running_by_phase(agent, phases_dict, window=500):
    """
    Plot phase-segmented running |TD| using TRAINING_PHASES-style dict.

    phases_dict: ordered dict like your TRAINING_PHASES where each value has "duration" in EPISODES.
    window: moving average window over |TD| samples.
    """
    # --- TD data ---
    td = np.asarray(getattr(agent, "td_hist", []), dtype=np.float32).ravel()
    if td.size == 0:
        print("[plot_td_running_by_phase] No TD samples in agent.td_hist."); return
    k = max(1, int(window))
    if td.size < k:
        print(f"[plot_td_running_by_phase] Not enough TD samples ({td.size}) for window={k}."); return
    run = np.convolve(np.abs(td), np.ones(k, dtype=np.float32)/k, mode="valid")  # length = N-k+1
    if run.size == 0:
        print("[plot_td_running_by_phase] Running series empty; check window/data."); return

    # --- read phases (dict is ordered in Py>=3.7) ---
    names = list(phases_dict.keys())
    durs_ep = [int(phases_dict[n].get("duration", 0)) for n in names]
    if not durs_ep or sum(durs_ep) <= 0:
        print("[plot_td_running_by_phase] No positive phase durations found."); return
    cum_ep = np.cumsum([0] + durs_ep)  # episode marks at phase starts; len = len(names)+1
    total_ep = cum_ep[-1]

    # --- episode->TD mapping (best available) ---
    def ep_to_td(ep):
        # exact lookup array?
        if hasattr(agent, "td_idx_at_ep"):
            arr = np.asarray(getattr(agent, "td_idx_at_ep"))
            if arr.size > ep:
                try:
                    return int(arr[ep])
                except Exception:
                    pass
        # callable mapper?
        if hasattr(agent, "ep_to_td") and callable(getattr(agent, "ep_to_td")):
            try:
                return int(agent.ep_to_td(int(ep)))
            except Exception:
                pass
        # fallback: proportional (approx)
        return int(round((ep / max(1, total_ep)) * (td.size - 1)))

    # convert episode marks to TD-sample marks, then to running-index space
    td_marks = [ep_to_td(ep) for ep in cum_ep]
    def to_run_idx(sample_idx):
        return max(0, min(run.size - 1, int(sample_idx) - (k - 1)))
    run_marks = [to_run_idx(i) for i in td_marks]

    # dedupe/monotone boundaries
    boundaries = [0]
    for i in run_marks:
        if i != boundaries[-1]:
            boundaries.append(i)
    if boundaries[-1] != run.size - 1:
        boundaries.append(run.size - 1)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(run, lw=1.6)
    ax.set_title(f"|TD| running mean by phase (window={k})")
    ax.set_xlabel("TD sample (running index)")
    ax.set_ylabel("running |td|")
    ax.grid(True, ls="--", alpha=0.35)

    # shade segments + per-segment mean
    for seg_idx, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if a >= b: 
            continue
        seg = run[a:b]
        ax.axvspan(a, b, color=("0.95" if (seg_idx % 2 == 0) else "0.90"), alpha=0.9, lw=0)
        ax.hlines(seg.mean(), a, b, ls="--", lw=1.0, alpha=0.9)

    # verticals & labels at phase starts
    y_top = ax.get_ylim()[1]
    for x, lbl in zip(run_marks[:-1], names):  # exclude final end mark
        ax.axvline(x, color="k", lw=1, ls=":", alpha=0.7)
        ax.text(x + 3, y_top * 0.98, lbl, rotation=90, va="top", ha="left", fontsize=8, alpha=0.75)

    plt.tight_layout()
    plt.show()



