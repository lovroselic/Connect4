# PPO/ppo_agent_eval.py
# Single-source PPO eval helpers, with a simple DQN-style loader.

import os, re, random, warnings
from typing import Iterable, Dict, Tuple, Any, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from C4.connect4_env import Connect4Env
from C4.fast_connect4_lookahead import Connect4Lookahead
from PPO.actor_critic import ActorCritic
from PPO.checkpoint import load_checkpoint  # uses your legacy save/load format


warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", message="You are using `torch.load`.*weights_only=False")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEG_INF = -1e9


# =============================
# Loading (DQN-style, via PPO.checkpoint)
# =============================

def load_policy_simple(
    name_or_path: str,
    device=DEVICE,
    default_suffix: str = " PPO model.pt",
):
    """
    Minimal loader using PPO.checkpoint.load_checkpoint:
      - If you pass a filename with .pt/.pth/.ckpt -> load it.
      - Else append default_suffix (e.g., 'RND_3' -> 'RND_3 PPO model.pt').
    Returns (policy.eval(), resolved_path, meta_dict).
    """
    path = name_or_path
    if not any(path.endswith(ext) for ext in (".pt", ".pth", ".ckpt")):
        path = f"{name_or_path}{default_suffix}"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model, optim_state, meta = load_checkpoint(path, device=device, strict=True)
    return model.eval(), path, meta or {}


# =============================
# Encoding & action helpers
# =============================

def _mirror_state_4ch(s: np.ndarray) -> np.ndarray:
    """Mirror horizontally; negate col-plane; ensure positive strides."""
    assert s.shape[0] >= 4 and s.shape[1:] == (6, 7), f"Unexpected state shape {s.shape}"
    sm = s[:, :, ::-1].copy()
    sm[3] = -sm[3]
    return sm[:4]


@torch.inference_mode()
def _logits_np(policy: ActorCritic, s_4: np.ndarray) -> np.ndarray:
    x_np = np.ascontiguousarray(s_4)
    x = torch.from_numpy(x_np).float().unsqueeze(0).to(next(policy.parameters()).device)
    logits, _ = policy.forward(x)
    return logits.squeeze(0).detach().cpu().numpy()


def _argmax_legal_center_tiebreak(
    logits: np.ndarray,
    legal: Iterable[int],
    center: int = 3,
    tol: float = 1e-12,
) -> int:
    legal = list(legal)
    if not legal:
        raise ValueError("No legal actions passed to _argmax_legal_center_tiebreak")

    logits = np.asarray(logits, dtype=float)
    vals = np.full(7, -np.inf, dtype=float)

    for c in legal:
        if 0 <= c < logits.shape[-1]:
            vals[c] = float(logits[c])

    m = np.max(vals)
    diff = np.abs(vals - m)
    tied = np.where(diff <= tol)[0]

    if len(tied) == 0:
        return int(sorted(legal, key=lambda c: (abs(c - center), c))[0])
    if len(tied) == 1:
        return int(tied[0])

    # Multiple argmax actions, break ties by closeness to center, then index
    return int(sorted(tied, key=lambda c: (abs(c - center), c))[0])


@torch.inference_mode()
def ppo_act_greedy(policy: ActorCritic, env: Connect4Env) -> int:
    s = env.get_state(perspective=env.current_player)
    logits = _logits_np(policy, s)
    return _argmax_legal_center_tiebreak(logits, env.available_actions())


# =============================
# Symmetry & sanity
# =============================

def symmetry_check_once(
    policy: ActorCritic,
    env: Connect4Env,
    verbose: bool = True,
) -> Dict[str, Any]:
    s = env.get_state(env.current_player)
    q = _logits_np(policy, s)
    qm = _logits_np(policy, _mirror_state_4ch(s))[::-1]
    diff = float(np.max(np.abs(q - qm)))
    if verbose:
        print(f"[SYM] max|q - mirror(q)| = {diff:.6g}")
    return {"q": q, "qm_unflip": qm, "max_abs_diff": diff}


def random_states_symmetry_probe(
    policy: ActorCritic,
    n_random: int = 32,
    rng_seed: int = 0,
    atol_warn: float = 1e-5,
):
    rng = np.random.default_rng(rng_seed)
    diffs = []
    for _ in range(n_random):
        env = Connect4Env()
        env.reset()
        for _ in range(int(rng.integers(0, 8))):
            if env.done:
                break
            legal = env.available_actions()
            if not legal:
                break
            env.step(int(rng.choice(legal)))
        s = env.get_state(env.current_player)
        q = _logits_np(policy, s)
        qm = _logits_np(policy, _mirror_state_4ch(s))[::-1]
        diffs.append(float(np.max(np.abs(q - qm))))
    mx, mean = float(np.max(diffs)), float(np.mean(diffs))
    print(f"[SYM] random-states: max={mx:.6g}, mean={mean:.6g}, >{atol_warn}: {sum(d > atol_warn for d in diffs)}/{n_random}")
    return {"max": mx, "mean": mean, "diffs": diffs}


def self_test(
    policy: ActorCritic,
    atol: float = 1e-2,
    n_random: int = 32,
    rng_seed: int = 0,
):
    env = Connect4Env()
    env.reset()
    res_raw = symmetry_check_once(policy, env, verbose=True)
    diff = res_raw["max_abs_diff"]
    if not diff < atol:
        print(f"RAW symmetry broken on empty board (diff={diff:.6g} > atol={atol}).")

    # report center pick (no assert, ZERO can be slightly off)
    s0 = env.get_state(env.current_player)
    a0 = _argmax_legal_center_tiebreak(_logits_np(policy, s0), env.available_actions())
    print(f"[CENTER] empty-board argmax = {a0}")

    random_states_symmetry_probe(policy, n_random=n_random, rng_seed=rng_seed)
    print("✅ PPO self-test passed.")


# =============================
# Opponents & evaluation
# =============================

def _parse_depth(label: str, default=2) -> int:
    m = re.search(r"(\d+)", label)
    return int(m.group(1)) if (label.startswith("Lookahead") and m) else default


def opponent_action(env: Connect4Env, label: str, la: Connect4Lookahead, rng: random.Random) -> int:
    legal = env.available_actions()
    if not legal:
        raise ValueError("No legal actions.")
    if label == "Random":
        return rng.choice(list(legal))
    d = _parse_depth(label, default=2)
    a = la.n_step_lookahead(env.board, player=-1, depth=d)
    return a if a in legal else rng.choice(list(legal))


def play_single_game_ppo(policy: ActorCritic, opponent_label: str, seed: int = 0) -> Tuple[float, np.ndarray]:
    rng = random.Random(seed)
    env = Connect4Env()
    env.reset()
    la = Connect4Lookahead()
    AGENT, OPP = +1, -1
    done = False

    # strict alternation: half the games opponent opens
    if (seed % 2) != 0:
        env.current_player = OPP
        a = opponent_action(env, opponent_label, la, rng)
        _, _, done = env.step(a)

    while not done:
        env.current_player = AGENT
        a = ppo_act_greedy(policy, env)
        _, _, done = env.step(a)
        if done:
            break

        env.current_player = OPP
        a = opponent_action(env, opponent_label, la, rng)
        _, _, done = env.step(a)

    if env.winner == AGENT:
        outcome = 1.0
    elif env.winner == OPP:
        outcome = -1.0
    else:
        outcome = 0.5

    return outcome, env.board.copy()


def evaluate_actor_critic_model(
    policy: ActorCritic,
    evaluation_opponents: Dict[str, int],
    seed: int = 666,
) -> Dict[str, Dict[str, float]]:
    results = {}
    for label, n_games in evaluation_opponents.items():
        wins = losses = draws = 0
        with tqdm(total=n_games, desc=f"Opponent: {label}", leave=False, position=1) as pbar:
            for g in range(n_games):
                outcome, _ = play_single_game_ppo(policy, label, seed=seed + g)
                if outcome == 1.0:
                    wins += 1
                elif outcome == -1.0:
                    losses += 1
                else:
                    draws += 1
                pbar.update(1)
        results[label] = dict(
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=round(wins / n_games, 3),
            loss_rate=round(losses / n_games, 3),
            draw_rate=round(draws / n_games, 3),
            games=n_games,
        )
    return results


# =============================
# Extra: exploitation histogram & reporting
# =============================

@torch.inference_mode()
def exploitation_histogram(policy: ActorCritic, n_states: int = 256, rng_seed: int = 0) -> np.ndarray:
    counts = np.zeros(7, dtype=int)
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_states):
        env = Connect4Env()
        env.reset()
        for _ in range(int(rng.integers(0, 12))):
            if env.done:
                break
            legal = env.available_actions()
            if not legal:
                break
            env.step(int(rng.choice(legal)))
        a = ppo_act_greedy(policy, env)
        counts[a] += 1
    return counts


def results_to_row(run_tag: str, results: dict, elapsed_h: float, episodes=None) -> pd.DataFrame:
    row = {"TRAINING_SESSION": run_tag, "TIME [h]": round(float(elapsed_h), 6), "EPISODES": episodes}
    for label, m in results.items():
        row[label] = m["win_rate"]
    return pd.DataFrame([row]).set_index("TRAINING_SESSION")


def append_eval_row_to_excel(df_row: pd.DataFrame, excel_path: str) -> None:
    if os.path.exists(excel_path):
        old = pd.read_excel(excel_path)
        if not old.empty and not old.isnull().all().all():
            new = pd.concat([old, df_row.reset_index()], ignore_index=True)
        else:
            new = df_row.reset_index()
    else:
        new = df_row.reset_index()
    new.to_excel(excel_path, index=False)


# =============================
# H2H utilities (shared openings + paired games)
# =============================

def mean_ci(scores, z=1.96):
    x = np.asarray(scores, dtype=float)
    n = len(x)
    m = float(x.mean()) if n else 0.0
    s = float(x.std(ddof=1)) if n > 1 else 0.0
    half = z * s / (n ** 0.5) if n > 1 else 0.0
    return max(0.0, m - half), min(1.0, m + half)


@torch.inference_mode()
def ppo_act_greedy_for_h2h(policy: ActorCritic, env: Connect4Env) -> int:
    return ppo_act_greedy(policy, env)


def _agent_for_side(side: int, A: ActorCritic, B: ActorCritic, A_color: int) -> ActorCritic:
    return A if side == A_color else B


def _resolve_opening_k(rng: random.Random, opening_noise_k: Union[int, Dict[int, float]]) -> int:
    """
    opening_noise_k can be:
      - int: fixed k
      - dict[int->float]: categorical distribution over k, e.g. {0:0.70, 1:0.20, 2:0.10}
    """
    if isinstance(opening_noise_k, dict):
        items = [(int(k), float(p)) for k, p in opening_noise_k.items() if float(p) > 0.0]
        if not items:
            return 0
        ks, ps = zip(*items)
        s = float(sum(ps))
        psn = [p / s for p in ps]
        r = rng.random()
        c = 0.0
        for k, p in zip(ks, psn):
            c += p
            if r <= c:
                return int(max(0, k))
        return int(max(0, ks[-1]))
    return int(max(0, int(opening_noise_k)))


def _sample_opening_action(
    rng: random.Random,
    legal: List[int],
    bias: str = "uniform",
    center: int = 3,
) -> int:
    """
    bias:
      - "uniform": uniform random over legal
      - "center": center-biased random over legal (still random, but likes the middle)
    """
    if not legal:
        raise ValueError("No legal actions for opening move.")

    if bias == "uniform":
        return rng.choice(legal)

    if bias == "center":
        # distance-to-center weights (tweakable)
        # dist 0:1.00, dist 1:0.55, dist 2:0.25, dist 3:0.10
        wmap = {0: 1.00, 1: 0.55, 2: 0.25, 3: 0.10}
        weights = [wmap.get(abs(c - center), 0.05) for c in legal]
        s = sum(weights)
        if s <= 0:
            return rng.choice(legal)
        weights = [w / s for w in weights]
        return rng.choices(legal, weights=weights, k=1)[0]

    raise ValueError(f"Unknown opening bias: {bias!r}")


def _generate_opening_moves(
    opening_noise_k: Union[int, Dict[int, float]],
    seed: int,
    opening_bias: str = "uniform",
) -> List[int]:
    """
    Generate a *legal* opening sequence from the empty board (independent of A/B).
    """
    rng = random.Random(seed)
    env = Connect4Env()
    env.reset()
    done = False

    k = _resolve_opening_k(rng, opening_noise_k)
    moves: List[int] = []

    for _ in range(k):
        if done:
            break
        legal = list(env.available_actions())
        if not legal:
            break
        a = _sample_opening_action(rng, legal, bias=opening_bias)
        moves.append(int(a))
        _, _, done = env.step(int(a))

    return moves


def play_game_ppo(
    A: ActorCritic,
    B: ActorCritic,
    A_color: int = +1,
    opening_noise_k: Union[int, Dict[int, float]] = 0,
    seed: int = 0,
    opening_moves: Optional[List[int]] = None,
    opening_bias: str = "uniform",
) -> float:
    """
    One greedy H2H game.

    - If opening_moves is provided, those moves are applied first (shared openings).
    - Else, opening_noise_k is used to sample a legal opening from the empty board.

    Returns:
      1.0 if A wins, 0.5 draw, 0.0 if A loses
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    env = Connect4Env()
    env.reset()
    done = False

    # Apply opening (shared or sampled)
    if opening_moves is None:
        k = _resolve_opening_k(rng, opening_noise_k)
        for _ in range(k):
            if done:
                break
            legal = list(env.available_actions())
            if not legal:
                break
            a = _sample_opening_action(rng, legal, bias=opening_bias)
            _, _, done = env.step(int(a))
    else:
        for a in opening_moves:
            if done:
                break
            legal = list(env.available_actions())
            if not legal:
                break
            aa = int(a)
            if aa not in legal:
                # Safety: if somehow invalid, resample a legal opening move
                aa = _sample_opening_action(rng, legal, bias=opening_bias)
            _, _, done = env.step(int(aa))

    # Main deterministic greedy play
    while not done:
        legal = env.available_actions()
        if not legal:
            return 0.5

        mover = env.current_player
        model = _agent_for_side(mover, A, B, A_color)
        a = ppo_act_greedy_for_h2h(model, env)
        _, _, done = env.step(int(a))

    if env.winner == 0:
        return 0.5
    return 1.0 if env.winner == A_color else 0.0


def head_to_head(
    ckptA: str,
    ckptB: str,
    n_games: int = 200,
    device=DEVICE,
    opening_noise_k: Union[int, Dict[int, float]] = 0,
    seed: int = 666,
    progress: bool = True,
    paired_openings: bool = True,
    opening_bias: str = "uniform",
):
    """
    Head-to-head match between two checkpoints.

    Update: supports shared openings + paired games (variance reduction).
      - paired_openings=True:
          each sampled opening is played twice (A as +1 and A as -1),
          using the exact same opening move sequence in both games.

    opening_noise_k:
      - int: fixed number of opening moves
      - dict: distribution over {k}, e.g. {0:0.70, 1:0.20, 2:0.10}

    opening_bias:
      - "uniform" or "center"
    """
    A, A_path, _ = load_policy_simple(ckptA, device=device)
    B, B_path, _ = load_policy_simple(ckptB, device=device)

    rng = random.Random(seed)
    timeline: List[float] = []
    wins = draws = losses = 0

    total = int(n_games)

    if not paired_openings:
        colors = ([+1, -1] * ((total + 1) // 2))[:total]
        iterable = tqdm(colors, total=total, desc=f"{os.path.basename(ckptA)} vs {os.path.basename(ckptB)}") if progress else colors

        for A_color in iterable:
            res = play_game_ppo(
                A,
                B,
                A_color=A_color,
                opening_noise_k=opening_noise_k,
                seed=rng.randint(0, 10**9),
                opening_moves=None,
                opening_bias=opening_bias,
            )
            timeline.append(res)
            if res == 1.0:
                wins += 1
            elif res == 0.5:
                draws += 1
            else:
                losses += 1

            if progress:
                g = len(timeline)
                score = (wins + 0.5 * draws) / g
                iterable.set_postfix(W=wins, D=draws, L=losses, score=f"{score:.3f}")

    else:
        n_pairs = total // 2
        extra = total % 2

        pbar = tqdm(total=total, desc=f"{os.path.basename(ckptA)} vs {os.path.basename(ckptB)}") if progress else None

        def _update_postfix():
            if pbar is None:
                return
            g = len(timeline)
            score = (wins + 0.5 * draws) / g if g else 0.0
            pbar.set_postfix(W=wins, D=draws, L=losses, score=f"{score:.3f}")

        for _ in range(n_pairs):
            opening_seed = rng.randint(0, 10**9)
            opening_moves = _generate_opening_moves(opening_noise_k, seed=opening_seed, opening_bias=opening_bias)

            for A_color in (+1, -1):
                res = play_game_ppo(
                    A,
                    B,
                    A_color=A_color,
                    opening_noise_k=0,              # important: opening is controlled by opening_moves
                    seed=rng.randint(0, 10**9),
                    opening_moves=opening_moves,    # shared
                    opening_bias=opening_bias,
                )
                timeline.append(res)
                if res == 1.0:
                    wins += 1
                elif res == 0.5:
                    draws += 1
                else:
                    losses += 1

                if pbar is not None:
                    pbar.update(1)
                    _update_postfix()

        if extra:
            opening_seed = rng.randint(0, 10**9)
            opening_moves = _generate_opening_moves(opening_noise_k, seed=opening_seed, opening_bias=opening_bias)
            res = play_game_ppo(
                A,
                B,
                A_color=+1,
                opening_noise_k=0,
                seed=rng.randint(0, 10**9),
                opening_moves=opening_moves,
                opening_bias=opening_bias,
            )
            timeline.append(res)
            if res == 1.0:
                wins += 1
            elif res == 0.5:
                draws += 1
            else:
                losses += 1

            if pbar is not None:
                pbar.update(1)
                _update_postfix()

        if pbar is not None:
            pbar.close()

    n = len(timeline)
    score = (wins + 0.5 * draws) / n if n else 0.0
    ci95 = mean_ci(timeline, z=1.96)

    return {
        "A_path": ckptA,
        "B_path": ckptB,
        "games": n,
        "A_wins": wins,
        "draws": draws,
        "A_losses": losses,
        "A_score_rate": score,
        "A_score_CI95": ci95,
        "paired_openings": bool(paired_openings),
        "opening_bias": opening_bias,
        "opening_noise_k": opening_noise_k,
    }


def plot_winrate_bar(res: dict, title: str | None = None, show_score_bars: bool = True):
    n = int(res["games"])
    Aw = int(res["A_wins"])
    Ad = int(res["draws"])
    Al = int(res["A_losses"])
    A_score = float(res["A_score_rate"])
    B_score = 1.0 - A_score
    pw, pd, pl = (Aw / n if n else 0.0), (Ad / n if n else 0.0), (Al / n if n else 0.0)
    a_name = os.path.basename(res["A_path"])
    b_name = os.path.basename(res["B_path"])
    title = title or f"{a_name} vs {b_name}"
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
    if n > 0:
        ax.text(pw / 2 if pw > 0 else 0.01, 0, f"W {Aw}", ha="center", va="center", color="white", fontsize=10)
        ax.text(pw + (pd / 2 if pd > 0 else 0.01), 0, f"D {Ad}", ha="center", va="center", color="white", fontsize=10)
        ax.text(pw + pd + (pl / 2 if pl > 0 else 0.01), 0, f"L {Al}", ha="center", va="center", color="white", fontsize=10)
    ax.legend(ncols=3, loc="lower right", frameon=False)
    plt.tight_layout()
    plt.show()
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
    plt.tight_layout()
    plt.show()


def plot_exploitation_histogram(counts: np.ndarray, title: str = "PPO greedy exploit histogram"):
    counts = np.asarray(counts, dtype=int)
    cols = np.arange(7)

    # Frequencies
    freqs = counts / max(1, counts.sum())
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(cols, freqs)
    ax.set_xticks(cols)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Column")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{title} (normalized)")
    for i, v in enumerate(freqs):
        if v > 0:
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    # plt.show()
    return fig


def head_to_head_models(
    A,
    B,
    n_games: int = 200,
    A_label: str = "student",
    B_label: str = "POP_ensemble",
    opening_noise_k: Union[int, Dict[int, float]] = 0,
    seed: int = 123,
    progress: bool = True,
    paired_openings: bool = True,
    opening_bias: str = "uniform",
):
    """
    Head-to-head match between two *models* (ActorCritic or PPOEnsemblePolicy),
    using the same logic as head_to_head(ckptA, ckptB) but without loading.

    Update: supports shared openings + paired games (variance reduction).

    Returns a dict compatible with plot_winrate_bar().
    """
    rng = random.Random(seed)

    timeline: List[float] = []
    wins = draws = losses = 0
    total = int(n_games)

    if not paired_openings:
        colors = ([+1, -1] * ((total + 1) // 2))[:total]
        iterable = tqdm(colors, total=total, desc=f"{A_label} vs {B_label}") if progress else colors

        for A_color in iterable:
            res = play_game_ppo(
                A,
                B,
                A_color=A_color,
                opening_noise_k=opening_noise_k,
                seed=rng.randint(0, 10**9),
                opening_moves=None,
                opening_bias=opening_bias,
            )
            timeline.append(res)
            if res == 1.0:
                wins += 1
            elif res == 0.5:
                draws += 1
            else:
                losses += 1

            if progress:
                g = len(timeline)
                score = (wins + 0.5 * draws) / g
                iterable.set_postfix(W=wins, D=draws, L=losses, score=f"{score:.3f}")

    else:
        n_pairs = total // 2
        extra = total % 2

        pbar = tqdm(total=total, desc=f"{A_label} vs {B_label}") if progress else None

        def _update_postfix():
            if pbar is None:
                return
            g = len(timeline)
            score = (wins + 0.5 * draws) / g if g else 0.0
            pbar.set_postfix(W=wins, D=draws, L=losses, score=f"{score:.3f}")

        for _ in range(n_pairs):
            opening_seed = rng.randint(0, 10**9)
            opening_moves = _generate_opening_moves(opening_noise_k, seed=opening_seed, opening_bias=opening_bias)

            for A_color in (+1, -1):
                res = play_game_ppo(
                    A,
                    B,
                    A_color=A_color,
                    opening_noise_k=0,
                    seed=rng.randint(0, 10**9),
                    opening_moves=opening_moves,
                    opening_bias=opening_bias,
                )
                timeline.append(res)
                if res == 1.0:
                    wins += 1
                elif res == 0.5:
                    draws += 1
                else:
                    losses += 1

                if pbar is not None:
                    pbar.update(1)
                    _update_postfix()

        if extra:
            opening_seed = rng.randint(0, 10**9)
            opening_moves = _generate_opening_moves(opening_noise_k, seed=opening_seed, opening_bias=opening_bias)
            res = play_game_ppo(
                A,
                B,
                A_color=+1,
                opening_noise_k=0,
                seed=rng.randint(0, 10**9),
                opening_moves=opening_moves,
                opening_bias=opening_bias,
            )
            timeline.append(res)
            if res == 1.0:
                wins += 1
            elif res == 0.5:
                draws += 1
            else:
                losses += 1

            if pbar is not None:
                pbar.update(1)
                _update_postfix()

        if pbar is not None:
            pbar.close()

    n = len(timeline)
    score = (wins + 0.5 * draws) / n if n else 0.0
    ci95 = mean_ci(timeline, z=1.96)

    return {
        "A_path": A_label,
        "B_path": B_label,
        "games": n,
        "A_wins": wins,
        "draws": draws,
        "A_losses": losses,
        "A_score_rate": score,
        "A_score_CI95": ci95,
        "paired_openings": bool(paired_openings),
        "opening_bias": opening_bias,
        "opening_noise_k": opening_noise_k,
    }


# =============================
# Policy visualization helpers
# =============================

@torch.inference_mode()
def state_action_distribution(
    policy: ActorCritic,
    state_4: np.ndarray,
    title: str = "policy — action distribution",
    annotate: bool = True,
    ylim_pad: float = 0.15,
):
    """
    Given a single mover-centric state (4,6,7), compute logits, turn them
    into a softmax probability distribution over columns, and plot it.

    Returns fig.
    """
    logits = _logits_np(policy, state_4)  # shape (7,)

    # numerically-stable softmax on numpy side
    probs = np.exp(logits - logits.max())
    s = probs.sum()
    if s > 0.0:
        probs /= s

    argmax = int(np.argmax(probs))

    cols = np.arange(len(probs))
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.bar(cols, probs)
    ax.set_xticks(cols)
    ax.set_ylim(0, max(0.01, probs.max() * (1.0 + ylim_pad)))
    ax.set_xlabel("Column")
    ax.set_ylabel("Probability")
    ax.set_title(f"{title}; argmax: {argmax}")
    ax.grid(axis="y", ls="--", alpha=0.3)

    if annotate:
        for i, p in enumerate(probs):
            if p > 0:
                ax.text(i, p, f"{p:.5f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def plot_empty_board_action_distribution(
    policy: ActorCritic,
    tag: str = "policy",
):
    """
    Convenience wrapper: reset env, get empty-board state and call
    state_action_distribution().

    Example:
        plot_empty_board_action_distribution(policy, tag="ZERO")

    Returns fig.
    """
    env = Connect4Env()
    env.reset()
    s0 = env.get_state(env.current_player)  # (4,6,7), mover-centric
    title = f"{tag} — empty board action distribution"
    return state_action_distribution(policy, s0, title=title)
