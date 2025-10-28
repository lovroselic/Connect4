# agent_eval.py
from tqdm.auto import tqdm
import os, random, numpy as np, torch
from DQN.dqn_agent import DQNAgent
from C4.connect4_env import Connect4Env
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="You are using `torch.load`.*weights_only=False")


# ----------------------------- Eval helpers -----------------------------

def set_eval_mode(agent):
    agent.model.eval(); 
    agent.target_model.eval()
    agent.epsilon = 0.0; 
    agent.epsilon_min = 0.0
    agent.guard_prob = 0.0
    agent.use_symmetry_policy = True
    agent.prefer_center = False
    agent.tie_tol = 0.0
    agent.center_start = 0.0                    # disable a0=center forcing
    agent.center_forced_used = True            

def _extract_state_dict(sd):
    if isinstance(sd, dict):
        for k in ("model", "state_dict", "net", "params"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    if not isinstance(sd, dict): raise TypeError("Checkpoint does not contain a state_dict dict.")

    clean = {}
    for k, v in sd.items():
        new_k = k[7:] if k.startswith("module.") else k
        clean[new_k] = v
    return clean


def load_agent_from_ckpt(path: str, device=None, epsilon=0.0, guard_prob=0.0):
    agent = DQNAgent(device=device)
    try: sd = torch.load(path, map_location=agent.device, weights_only=True)
    except TypeError: sd = torch.load(path, map_location=agent.device)
    state_dict = _extract_state_dict(sd)
    missing, unexpected = agent.model.load_state_dict(state_dict, strict=False)
    agent.update_target_model()
    set_eval_mode(agent)
    return agent


def mean_ci(scores, z=1.96):
    """CI for mean of scores in {0, 0.5, 1}. Normal approximation."""
    x = np.asarray(scores, float)
    n = len(x)
    m = float(x.mean()) if n else 0.0
    s = float(x.std(ddof=1)) if n > 1 else 0.0
    half = z * s / (n ** 0.5) if n > 1 else 0.0
    return (max(0.0, m - half), min(1.0, m + half))


# ----------------------------- Pure-Q action -----------------------------

def q_act_pure(agent, env):
    """
    Pure Q action from the current mover's POV.
    Uses symmetry-averaged Q if agent.use_symmetry_policy=True.
    Legal-aware tie-break matches training.
    """
    s = env.get_state(perspective=env.current_player)            # (4,6,7), mover first
    va = env.available_actions()
    with torch.no_grad():
        q = agent._symmetry_avg_q(s) if agent.use_symmetry_policy else agent._q_values_np(s)
    return agent._argmax_legal_with_tiebreak(q, va)

# ----------------------------- Game loop -----------------------------

def agent_for_side(side: int, agentA, agentB, A_color: int):
    """
    side: env.current_player (+1 or -1).
    Returns (agent_for_that_side, side) so the model is oriented to the true mover.
    """
    return (agentA if side == A_color else agentB), side


def play_game(agentA, agentB, A_color:int=+1, opening_noise_k:int=0, seed:int=0):
    env = Connect4Env()
    env.reset(); 

    while not env.done:
        side = env.current_player                                       # +1 or -1
        ag = (agentA if side == A_color else agentB)
        a = q_act_pure(ag, env)                                         # <-- uses mover POV internally
        env.step(a)

    return 0.5 if env.winner == 0 else (1.0 if env.winner == A_color else 0.0)



# ----------------------------- Head-to-head -----------------------------

def head_to_head(ckptA:str, ckptB:str, n_games:int=200, device=None,
                 epsilon=0.0, guard_prob=0.0, opening_noise_k:int=0, seed:int=666,
                 progress: bool = True, center_start: float = 0.0,
                 use_symmetry_policy: bool = True,
                 prefer_center: bool = False, tie_tol: float = 0.0):
    
    A = load_agent_from_ckpt(ckptA, device=device, epsilon=epsilon, guard_prob=guard_prob)
    B = load_agent_from_ckpt(ckptB, device=device, epsilon=epsilon, guard_prob=guard_prob)
    
    set_eval_mode(A)
    set_eval_mode(B)

    rng = random.Random(seed)
    colors = ([+1, -1] * ((n_games + 1)//2))[:n_games]
    iterable = tqdm(colors, total=n_games, desc=f"{os.path.basename(ckptA)} vs {os.path.basename(ckptB)}") if progress else colors

    timeline = []; wins = draws = losses = 0
    for A_color in iterable:

        res = play_game(A, B, A_color=A_color, opening_noise_k=opening_noise_k, seed=rng.randint(0,10**9))
        timeline.append(res)
        if res == 1.0: wins += 1
        elif res == 0.5: draws += 1
        else: losses += 1

        if progress:
            n = len(timeline); score = (wins + 0.5*draws) / n
            iterable.set_postfix(W=wins, D=draws, L=losses, score=f"{score:.3f}")

    n = len(timeline)
    score = (wins + 0.5*draws) / n
    ci = mean_ci(timeline, z=1.96)
    return {"A_path": ckptA, "B_path": ckptB, "games": n,
            "A_wins": wins, "draws": draws, "A_losses": losses,
            "A_score_rate": score, "A_score_CI95": ci}


# ----------------------------- Plotting -----------------------------

def plot_winrate_bar(res, title=None, show_score_bars=True):
    """
    Draws a stacked bar of A's Win/Draw/Loss rates vs B (and optionally a 2-bar A vs B score).
    Expects the dict returned by head_to_head(...).
    """
    n   = int(res["games"])
    Aw  = int(res["A_wins"])
    Ad  = int(res["draws"])
    Al  = int(res["A_losses"])
    A_score = float(res["A_score_rate"])
    B_score = 1.0 - A_score

    pw, pd, pl = (Aw/n if n else 0.0), (Ad/n if n else 0.0), (Al/n if n else 0.0)
    a_name = os.path.basename(res["A_path"])
    b_name = os.path.basename(res["B_path"])
    title = title or f"{a_name} vs {b_name}"

    # --- Stacked Win/Draw/Loss bar (A perspective) ---
    fig, ax = plt.subplots(figsize=(6.2, 1.6))
    left = 0.0
    ax.barh([0], [pw], left=left, color="#4CAF50", label="Win")
    left += pw
    ax.barh([0], [pd], left=left, color="#9E9E9E", label="Draw")
    left += pd
    ax.barh([0], [pl], left=left, color="#F44336", label="Loss")

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Proportion")
    ax.set_title(f"{title} — A Win/Draw/Loss")

    # annotate counts
    if n > 0:
        ax.text(pw/2 if pw>0 else 0.01, 0, f"W {Aw}", ha="center", va="center", color="white", fontsize=10)
        ax.text(pw + pd/2 if pd>0 else pw+0.01, 0, f"D {Ad}", ha="center", va="center", color="white", fontsize=10)
        ax.text(pw + pd + pl/2 if pl>0 else pw+pd+0.01, 0, f"L {Al}", ha="center", va="center", color="white", fontsize=10)

    ax.legend(ncols=3, loc="lower right", frameon=False)
    plt.tight_layout()
    plt.show()

    if not show_score_bars: return

    # --- A vs B overall score bar (A gets 1, draw=0.5) ---
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.bar([a_name, b_name], [A_score, B_score])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score rate")
    ax.set_title(f"{title} — overall score")
    for i, v in enumerate([A_score, B_score]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.grid(axis="y", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
