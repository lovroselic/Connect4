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

def plot_is_weights(agent, bins=60):
    w = np.array(agent.per_w_hist, dtype=np.float32)
    if w.size == 0:
        print("No IS weights yet."); return
    plt.figure(figsize=(8,4))
    plt.hist(w, bins=bins, density=True, alpha=0.8)
    plt.title(f"Importance weights (mean={w.mean():.3f}, max={w.max():.3f})")
    plt.xlabel("w"); plt.ylabel("Density"); plt.grid(True, ls='--', alpha=0.4)
    plt.show()

def check_per_priority_correlation(agent, sample=2048):
    # draw a mixed batch and compute |td| vs. updated priority to see correlation
    if (len(agent.memory.bank_1)+len(agent.memory.bank_n)) < sample:
        print("Not enough memory to probe."); return
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

