# PPO/ppo_eval_utilities.py
# PPO evaluation helpers mirroring  DQN flow (env is mover-centric).

from __future__ import annotations
import os, re, glob, random
from typing import Dict, List, Tuple, Iterable, Any, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from C4.connect4_env import Connect4Env
from C4.fast_connect4_lookahead import Connect4Lookahead
from PPO.actor_critic import ActorCritic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEG_INF = -1e9


# -----------------------------
# Checkpoint loading
# -----------------------------

def _extract_state_dict_ppo(obj: dict) -> dict:
    sd = None
    if isinstance(obj, dict):
        for k in ("model_state","state_dict","model","net","params"):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                break
    if sd is None:
        sd = obj if isinstance(obj, dict) else None
    if sd is None:
        raise TypeError("Checkpoint does not contain a usable state_dict.")
    return { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }

def find_checkpoint(tag_or_path: str,
                    search_dirs: Optional[List[str]] = None,
                    exts: Tuple[str,...] = (".pt",".pth",".ckpt")) -> str:
    """
    If tag_or_path is a file that exists, return it.
    Otherwise search typical dirs for tag.*ext (case-insensitive).
    """
    if os.path.isfile(tag_or_path):
        return tag_or_path

    tag = tag_or_path
    search_dirs = search_dirs or ["./", "checkpoints", "PPO", "PPO/checkpoints"]
    patterns = [os.path.join(d, f"{tag}{e}") for d in search_dirs for e in exts]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))
        # case-insensitive pass
        d = os.path.dirname(pat)
        b, e = os.path.splitext(os.path.basename(pat))
        if os.path.isdir(d):
            for f in os.listdir(d):
                if os.path.splitext(f)[1].lower() in exts and os.path.splitext(f)[0].lower() == b.lower():
                    matches.append(os.path.join(d, f))
    if not matches:
        # fallback: substring search
        for d in search_dirs:
            for e in exts:
                matches.extend(glob.glob(os.path.join(d, f"*{tag}*{e}"), recursive=True))

    if not matches:
        raise FileNotFoundError(f"Checkpoint not found for tag '{tag_or_path}' in {search_dirs}")

    # pick shortest path (most specific)
    matches = sorted(set(matches), key=lambda p: (len(p), p))
    return matches[0]

def load_policy(tag_or_path: str, device=DEVICE) -> ActorCritic:
    """
    Flexible loader: accepts a direct filename or a tag to be resolved via find_checkpoint().
    Loads state_dict into a fresh ActorCritic and returns .eval().
    """
    path = find_checkpoint(tag_or_path)
    try:
        raw = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        raw = torch.load(path, map_location=device)
    sd  = _extract_state_dict_ppo(raw)
    m = ActorCritic().to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


# -----------------------------
# Encoding, actions, mirror
# -----------------------------

def _mirror_state_4ch(s: np.ndarray) -> np.ndarray:
    """
    s: (4,6,7)[me,opp,row,col] mover-centric → mirrored horizontally (contiguous).
    """
    assert s.shape[0] >= 4 and s.shape[1:] == (6,7), f"Unexpected state shape {s.shape}"
    sm = s[:, :, ::-1].copy()  # ensure positive strides
    sm[3] = -sm[3]             # flip sign of col-plane
    return sm[:4]

@torch.inference_mode()
def _logits_np(policy: ActorCritic, s4: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(s4)).float().unsqueeze(0).to(next(policy.parameters()).device)
    logits, _ = policy.forward(x)
    return logits.squeeze(0).detach().cpu().numpy()

def _argmax_legal_center_tiebreak(logits: np.ndarray, legal: Iterable[int], center: int = 3, tol: float = 1e-12) -> int:
    legal = list(legal)
    if not legal: raise ValueError("No legal actions.")
    vals = np.full(7, -np.inf, dtype=float)
    vals[legal] = logits[legal]
    m = np.max(vals)
    tied = np.where(np.abs(vals - m) <= tol)[0]
    if len(tied) == 1:
        return int(tied[0])
    return int(sorted(tied, key=lambda c: (abs(c-center), c))[0])

@torch.inference_mode()
def ppo_greedy_action(policy: ActorCritic, env: Connect4Env) -> int:
    s4 = env.get_state(env.current_player)
    logits = _logits_np(policy, s4)
    return _argmax_legal_center_tiebreak(logits, env.available_actions())


# -----------------------------
# Symmetry & sanity
# -----------------------------

def symmetry_check_once(policy: ActorCritic, env: Connect4Env, verbose: bool = True) -> Dict[str, Any]:
    s = env.get_state(env.current_player)
    q  = _logits_np(policy, s)
    qm = _logits_np(policy, _mirror_state_4ch(s))[::-1]  # unflip
    diff = float(np.max(np.abs(q - qm)))
    if verbose:
        print(f"[SYM] max|q - mirror(q)| = {diff:.6g}")
    return {"q": q, "qm_unflip": qm, "max_abs_diff": diff}

def self_test(policy: ActorCritic, atol: float = 1e-2, n_random: int = 32, rng_seed: int = 0):
    env = Connect4Env(); env.reset()
    res_raw = symmetry_check_once(policy, env, verbose=True)
    diff = res_raw["max_abs_diff"]
    assert diff < atol, f"RAW symmetry broken on empty board (diff={diff:.6g} > atol={atol})."

    # report center pick on empty board (no assert, to allow ZERO)
    s0 = env.get_state(env.current_player)
    a0 = _argmax_legal_center_tiebreak(_logits_np(policy, s0), env.available_actions())
    print(f"[CENTER] empty-board argmax = {a0}")

    # random states probe
    rng = np.random.default_rng(rng_seed)
    diffs = []
    for _ in range(n_random):
        env = Connect4Env(); env.reset()
        for _ in range(int(rng.integers(0, 8))):
            if env.done: break
            legal = env.available_actions()
            if not legal: break
            env.step(int(rng.choice(legal)))
        s  = env.get_state(env.current_player)
        q  = _logits_np(policy, s)
        qm = _logits_np(policy, _mirror_state_4ch(s))[::-1]
        diffs.append(float(np.max(np.abs(q - qm))))
    mx, mean = float(np.max(diffs)), float(np.mean(diffs))
    print(f"[SYM] random-states: max={mx:.6g}, mean={mean:.6g}, >{atol}: {sum(d>atol for d in diffs)}/{n_random}")
    print("✅ PPO self-test passed.")


# -----------------------------
# Opponent adapter (Random / Lookahead-k)
# -----------------------------

def _parse_depth(label: str, default=2) -> int:
    m = re.search(r"(\d+)", label)
    return int(m.group(1)) if (label.startswith("Lookahead") and m) else default

def opponent_action(env: Connect4Env, label: str, la: Connect4Lookahead, rng: random.Random) -> int:
    legal = env.available_actions()
    if not legal: raise ValueError("No legal actions.")
    if label == "Random":
        return rng.choice(legal)
    d = _parse_depth(label, default=2)
    a = la.n_step_lookahead(env.board, player=-1, depth=d)
    return a if a in legal else rng.choice(legal)


# -----------------------------
# One game, batch evaluate, exploitation histogram
# -----------------------------

def play_single_game_ppo(policy: ActorCritic, opponent_label: str, seed: int = 0) -> Tuple[float, np.ndarray]:
    rng = random.Random(seed)
    env = Connect4Env(); env.reset()
    la  = Connect4Lookahead()
    AGENT, OPP = +1, -1
    done = False

    # strict alternation: half the games opponent opens
    if (seed % 2) != 0:
        env.current_player = OPP
        a = opponent_action(env, opponent_label, la, rng)
        _, _, done = env.step(a)

    while not done:
        env.current_player = AGENT
        a = ppo_greedy_action(policy, env)
        _, _, done = env.step(a)
        if done: break

        env.current_player = OPP
        a = opponent_action(env, opponent_label, la, rng)
        _, _, done = env.step(a)

    if env.winner == AGENT:   outcome = 1.0
    elif env.winner == OPP:   outcome = -1.0
    else:                     outcome = 0.5

    return outcome, env.board.copy()

def evaluate_actor_critic_model(policy: ActorCritic,
                                evaluation_opponents: Dict[str,int],
                                seed: int = 666) -> Dict[str, Dict[str, float]]:
    results = {}
    for label, n_games in evaluation_opponents.items():
        wins = losses = draws = 0
        with tqdm(total=n_games, desc=f"Opponent: {label}", leave=False, position=1) as pbar:
            for g in range(n_games):
                outcome, _ = play_single_game_ppo(policy, label, seed=seed+g)
                if outcome == 1.0: wins += 1
                elif outcome == -1.0: losses += 1
                else: draws += 1
                pbar.update(1)
        results[label] = dict(
            wins=wins, losses=losses, draws=draws,
            win_rate=round(wins / n_games, 3),
            loss_rate=round(losses / n_games, 3),
            draw_rate=round(draws / n_games, 3),
            games=n_games,
        )
    return results

def exploitation_histogram(policy: ActorCritic, n_states: int = 256, rng_seed: int = 0) -> np.ndarray:
    """
    Greedy argmax decisions on random legal states. Returns counts[7].
    """
    rng = np.random.default_rng(rng_seed)
    counts = np.zeros(7, dtype=int)
    for i in range(n_states):
        env = Connect4Env(); env.reset()
        for _ in range(int(rng.integers(0, 12))):
            if env.done: break
            legal = env.available_actions()
            if not legal: break
            env.step(int(rng.choice(legal)))
        a = ppo_greedy_action(policy, env)
        counts[a] += 1
    return counts


# -----------------------------
# Formatting to DataFrame / Excel
# -----------------------------

def results_to_row(run_tag: str, results: Dict[str, Dict[str, float]], elapsed_h: float,
                   extra: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Builds a one-row dataframe like your DQN eval sheet:
    [TRAINING_SESSION, TIME [h], <opponent-keys...>]
    """
    row = {
        "TRAINING_SESSION": run_tag,
        "TIME [h]": round(float(elapsed_h), 6),
        "EPISODES": extra.get("EPISODES") if extra else None,
    }
    for label, m in results.items():
        row[label] = m["win_rate"]
    df = pd.DataFrame([row]).set_index("TRAINING_SESSION")
    return df

def append_eval_row_to_excel(df_row: pd.DataFrame, excel_path: str) -> None:
    """
    Appends/creates a flat sheet identical to your DQN layout.
    """
    if os.path.exists(excel_path):
        old = pd.read_excel(excel_path)
        new = pd.concat([old, df_row.reset_index()], ignore_index=True)
    else:
        new = df_row.reset_index()
    new.to_excel(excel_path, index=False)
