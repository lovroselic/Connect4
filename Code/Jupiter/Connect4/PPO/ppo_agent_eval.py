# PPO/ppo_agent_eval.py
# PPO model-vs-model evaluation (alternating colors, CI, simple plots)

import os, random, warnings
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from C4.connect4_env import Connect4Env
from PPO.actor_critic import ActorCritic
from PPO.checkpoint import load_checkpoint

warnings.filterwarnings("ignore", message="You are using `torch.load`.*weights_only=False")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- Utilities -----------------------------

def _extract_state_dict_ppo(obj: dict) -> dict:
    """
    Accepts:
      - checkpoint dict with 'model_state' (our PPO saver)
      - generic dict with 'state_dict'/'model'/'net'/...
      - raw state_dict
    Strips 'module.' if present (DataParallel).
    """
    sd = None
    if isinstance(obj, dict):
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            sd = obj["model_state"]
        else:
            for k in ("state_dict", "model", "net", "params"):
                if k in obj and isinstance(obj[k], dict):
                    sd = obj[k]
                    break
    if sd is None:
        if isinstance(obj, dict):
            sd = obj  # assume raw state_dict
        else:
            raise TypeError("Checkpoint does not contain a usable state_dict.")

    clean = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        clean[nk] = v
    return clean


def load_policy_from_ckpt(path: str, device=DEVICE, strict: bool = False):
    model, _, _ = load_checkpoint(path, device=device, strict=strict)
    model.eval()
    return model


def mean_ci(scores, z=1.96):
    """95% CI for mean of scores in {0, 0.5, 1} via normal approximation."""
    x = np.asarray(scores, dtype=float)
    n = len(x)
    m = float(x.mean()) if n else 0.0
    s = float(x.std(ddof=1)) if n > 1 else 0.0
    half = z * s / (n ** 0.5) if n > 1 else 0.0
    return max(0.0, m - half), min(1.0, m + half)


# ----------------------------- Encoding & action -----------------------------

def _encode_two_planes(board_2d: np.ndarray, perspective: int) -> np.ndarray:
    """
    board_2d: (6,7) in {-1,0,+1}
    perspective: +1 or -1 (the true mover)
    returns (2,6,7) float32: [me, opp]
    """
    b = (board_2d.astype(np.int8) * int(perspective))
    me  = (b == +1).astype(np.float32)
    opp = (b == -1).astype(np.float32)
    return np.stack([me, opp], axis=0)


@torch.inference_mode()
def ppo_act_greedy(policy: ActorCritic, board_2d: np.ndarray, legal_actions, mover: int) -> int:
    """
    Deterministic action: argmax over masked logits.
    - policy: ActorCritic
    - board_2d: current env.board (6,7) in {-1,0,+1}
    - legal_actions: iterable of ints
    - mover: +1/-1 (env.current_player)
    """
    assert len(legal_actions) > 0, "No legal actions."
    planes = _encode_two_planes(board_2d, perspective=mover)  # (2,6,7)

    x = torch.from_numpy(planes).float().unsqueeze(0).to(policy.device if hasattr(policy, "device") else next(policy.parameters()).device)
    logits, _ = policy.forward(x)                            # (1,7)
    logits = logits.squeeze(0)                               # (7,)

    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[torch.as_tensor(list(legal_actions), dtype=torch.long, device=logits.device)] = True
    masked = logits.masked_fill(~mask, float("-inf"))
    return int(torch.argmax(masked).item())


# ----------------------------- Game loop -----------------------------

def _agent_for_side(side: int, A: ActorCritic, B: ActorCritic, A_color: int) -> ActorCritic:
    """Return the model that plays the given side (+1/-1)."""
    return A if side == A_color else B


def play_game_ppo(A: ActorCritic, B: ActorCritic, A_color: int = +1,
                  opening_noise_k: int = 0, seed: int = 0) -> float:
    """
    Plays one game. A plays as +1 for half the games (alternated by the caller).
    Returns score from A's perspective (1=win, 0.5=draw, 0=loss).
    """
    random.seed(seed); np.random.seed(seed)
    env = Connect4Env()
    s = env.reset(); done = False

    for _ in range(opening_noise_k):
        if done: break
        a = random.choice(env.available_actions())
        s, r, done = env.step(a)

    while not done:
        mover = env.current_player                       # +1 or -1
        model = _agent_for_side(mover, A, B, A_color)
        action = ppo_act_greedy(model, env.board, env.available_actions(), mover)
        s, r, done = env.step(action)

    if env.winner == 0: return 0.5
    return 1.0 if env.winner == A_color else 0.0


# ----------------------------- Head-to-head -----------------------------

def head_to_head(ckptA: str, ckptB: str, n_games: int = 200, device=DEVICE,
                 opening_noise_k: int = 0, seed: int = 123, progress: bool = True):
    """
    Evaluate PPO model A vs PPO model B.
    Alternate A's color each game to cancel first-move advantage.
    """
    A = load_policy_from_ckpt(ckptA, device=device)
    B = load_policy_from_ckpt(ckptB, device=device)

    rng = random.Random(seed)
    colors = ([+1, -1] * ((n_games + 1) // 2))[:n_games]
    iterable = tqdm(colors, total=n_games,
                    desc=f"{os.path.basename(ckptA)} vs {os.path.basename(ckptB)}") if progress else colors

    timeline = []
    wins = draws = losses = 0

    for A_color in iterable:
        res = play_game_ppo(A, B, A_color=A_color, opening_noise_k=opening_noise_k, seed=rng.randint(0, 10**9))
        timeline.append(res)
        if res == 1.0: wins += 1
        elif res == 0.5: draws += 1
        else: losses += 1

        if progress:
            g = len(timeline)
            score = (wins + 0.5 * draws) / g
            iterable.set_postfix(W=wins, D=draws, L=losses, score=f"{score:.3f}")

    n = len(timeline)
    score = (wins + 0.5 * draws) / n if n else 0.0
    ci95 = mean_ci(timeline, z=1.96)

    return {
        "A_path": ckptA, "B_path": ckptB, "games": n,
        "A_wins": wins, "draws": draws, "A_losses": losses,
        "A_score_rate": score, "A_score_CI95": ci95,
        # "timeline": timeline,  # keep if you want raw results
    }


# ----------------------------- Plotting -----------------------------

def plot_winrate_bar(res: dict, title: str | None = None, show_score_bars: bool = True):
    """
    Stacked W/D/L bar for A vs B (A perspective), plus A vs B score bars.
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

    fig, ax = plt.subplots(figsize=(6.2, 1.6))
    left = 0.0
    ax.barh([0], [pw], left=left, color="#4CAF50", label="Win"); left += pw
    ax.barh([0], [pd], left=left, color="#9E9E9E", label="Draw"); left += pd
    ax.barh([0], [pl], left=left, color="#F44336", label="Loss")

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Proportion")
    ax.set_title(f"{title} — A Win/Draw/Loss")

    if n > 0:
        ax.text(pw/2 if pw>0 else 0.01, 0, f"W {Aw}", ha="center", va="center", color="white", fontsize=10)
        ax.text(pw + (pd/2 if pd>0 else 0.01), 0, f"D {Ad}", ha="center", va="center", color="white", fontsize=10)
        ax.text(pw + pd + (pl/2 if pl>0 else 0.01), 0, f"L {Al}", ha="center", va="center", color="white", fontsize=10)

    ax.legend(ncols=3, loc="lower right", frameon=False)
    plt.tight_layout(); plt.show()

    if not show_score_bars:
        return

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.bar([a_name, b_name], [A_score, B_score])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score rate")
    ax.set_title(f"{title} — overall score")
    for i, v in enumerate([A_score, B_score]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.grid(axis="y", ls="--", alpha=0.3)
    plt.tight_layout(); plt.show()
