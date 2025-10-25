# agent_eval.py
# DQN.agent_eval (corrected for player perspective + mean CI)
from tqdm.auto import tqdm
import os, random, numpy as np, torch
from DQN.dqn_agent import DQNAgent
from C4.connect4_env import Connect4Env
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="You are using `torch.load`.*weights_only=False")


# ----------------------------- Eval helpers -----------------------------

def set_eval_mode(agent, epsilon=0.0, guard_prob=0.0):
    agent.model.eval() 
    agent.target_model.eval()
    agent.epsilon = float(epsilon)
    agent.epsilon_min = float(epsilon)
    agent.guard_prob = float(guard_prob)


def _extract_state_dict(sd):
    """
    Accepts:
      - raw state_dict (param_name -> tensor)
      - checkpoint dict with keys like: 'model', 'state_dict', 'net', 'params'
    Strips a leading 'module.' if present (DataParallel).
    """
    if isinstance(sd, dict):
        for k in ("model", "state_dict", "net", "params"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    if not isinstance(sd, dict):
        raise TypeError("Checkpoint does not contain a state_dict dict.")

    clean = {}
    for k, v in sd.items():
        new_k = k[7:] if k.startswith("module.") else k
        clean[new_k] = v
    return clean


def load_agent_from_ckpt(path: str, device=None, epsilon=0.0, guard_prob=0.0):
    agent = DQNAgent(device=device)
    try:
        sd = torch.load(path, map_location=agent.device, weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location=agent.device)

    state_dict = _extract_state_dict(sd)
    missing, unexpected = agent.model.load_state_dict(state_dict, strict=False)
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    matched = sum(1 for n,p in agent.model.named_parameters()
                  if n in state_dict and state_dict[n].shape == p.shape)
    total = sum(1 for _ in agent.model.parameters())
    print(f"[load] matched params: {matched}/{total}")

    agent.update_target_model()
    set_eval_mode(agent, epsilon=epsilon, guard_prob=guard_prob)
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

def q_act_pure(agent, state_abs, valid_actions, player):
    """
    Pure Q evaluation with symmetry averaging + legal-aware tie-break,
    oriented to the true mover 'player' (+1/-1).
    """
    assert state_abs.shape == (4, 6, 7), f"state shape {state_abs.shape} unexpected"
    assert len(valid_actions) > 0, "no valid actions"

    # Re-orient absolute planes so channel 0 is the *current mover*
    oriented = agent._agent_planes(state_abs, player)   # (4,6,7)

    # Symmetry-averaged Q to match training exploitation
    with torch.no_grad():
        q = agent._symmetry_avg_q(oriented)  # returns np.ndarray shape (7,)

    # Same legal & tie behavior as in training
    return agent._argmax_legal_with_tiebreak(q, valid_actions)



# ----------------------------- Game loop -----------------------------

def agent_for_side(side: int, agentA, agentB, A_color: int):
    """
    side: env.current_player (+1 or -1).
    Returns (agent_for_that_side, side) so the model is oriented to the true mover.
    """
    return (agentA if side == A_color else agentB), side


def play_game(agentA, agentB, A_color:int=+1, opening_noise_k:int=0, seed:int=0):
    random.seed(seed); np.random.seed(seed)
    env = Connect4Env()
    s = env.reset(); done = False

    # Optional opening noise (consider k=0 for fair eval)
    for _ in range(opening_noise_k):
        if done: break
        a = random.choice(env.available_actions())
        s, r, done = env.step(a)

    while not done:
        side = env.current_player                  # +1 or -1
        ag, pflag = agent_for_side(side, agentA, agentB, A_color)  # pflag == side
        a = q_act_pure(ag, s, env.available_actions(), player=pflag)
        s, r, done = env.step(a)

    winner = env.winner
    if winner == 0: return 0.5
    return 1.0 if winner == A_color else 0.0


# ----------------------------- Head-to-head -----------------------------

def head_to_head(ckptA:str, ckptB:str, n_games:int=200, device=None,
                 epsilon=0.0, guard_prob=0.0, opening_noise_k:int=0, seed:int=123,
                 progress: bool = True, center_start: float = 0.0,
                 disable_init_guard: bool = True):
    """
    Evaluates A vs B. We alternate A's color (+1 then -1) to balance first-move advantage.
    By default we do not inject random opening noise (opening_noise_k=0).
    center_start is forced to 0.0 by default (no forced a0=center).
    If disable_init_guard=True, the early-plies guard is disabled as well.
    """
    A = load_agent_from_ckpt(ckptA, device=device, epsilon=epsilon, guard_prob=guard_prob)
    B = load_agent_from_ckpt(ckptB, device=device, epsilon=epsilon, guard_prob=guard_prob)

    # --- ensure eval is purely policy-driven ---
    for ag in (A, B):
        ag.epsilon = 0.0
        ag.epsilon_min = 0.0
        ag.guard_prob = float(guard_prob)
        ag.center_start = float(center_start)   # <<< disable forced center
        if disable_init_guard:
            ag.init_guard = 0.0               # <<< otherwise it stays 0.5 for first 4 plies
            ag.init_guard_ply = 0

    rng = random.Random(seed)
    colors = ([+1, -1] * ((n_games + 1)//2))[:n_games]
    iterable = tqdm(colors, total=n_games, desc=f"{os.path.basename(ckptA)} vs {os.path.basename(ckptB)}") if progress else colors

    timeline = []
    wins = draws = losses = 0

    for A_color in iterable:
        # reset per-game flag to be extra safe
        A.center_forced_used = False
        B.center_forced_used = False

        res = play_game(A, B, A_color=A_color, opening_noise_k=opening_noise_k, seed=rng.randint(0,10**9))
        timeline.append(res)
        if res == 1.0: wins += 1
        elif res == 0.5: draws += 1
        else: losses += 1

        if progress:
            games_done = len(timeline)
            score = (wins + 0.5*draws) / games_done
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

    if not show_score_bars:
        return

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
