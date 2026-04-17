# c4_move_data_generator.py
"""
Connect-4 move-data generator (Excel-first).

Produces per-move rows (one row per action taken) that can be used for:
- seeding replay buffers
- supervised / distillation pretraining (store state BEFORE the move + action)
- analytics (reward distributions, ply distributions, etc.)

Key design choices:
- Fast legality checks use bitboard mask & TOP_MASK[c]
- Optional "perfect" vs "noisy" move selection
- Optional "teacher policy probs" stored as 7 columns (uniform among best-scored legal moves)

Expected external API (from your project):
- C4.connect4_env.Connect4Env with:
    - reset() -> obs
    - step(action) -> (obs, reward, done) or (obs, reward, done, info)
    - board: numpy array shape (6,7), values in {0,1,2} or {0,1,-1}, top row index 0
    - current_player: int in {1,2} (or possibly {1,-1}; we normalize in lookahead)
- C4.numba_connect4_lookahead.Connect4Lookahead (your Numba bitboard lookahead)
"""

from __future__ import annotations

from typing import Dict, List, Iterable, Any, Optional, Tuple
from pathlib import Path
import random

import numpy as np
import pandas as pd


from C4.connect4_env import Connect4Env
from C4.fast_connect4_lookahead import Connect4Lookahead
import torch
from PPO.actor_critic import ActorCritic


MIN_LA_DEPTH_STORED = 7

# --------------------------- Row / DF helpers ---------------------------

def _is_ppo_path(x: Any) -> bool:
    return isinstance(x, str) and x.strip().lower().endswith(".pt") and (not _is_random_spec(x))


# --- PPO support (lazy import so pure-LA notebooks still work) ---
_PPO_CACHE: Dict[str, Any] = {}

def _env_board_to_pov_robust(board: np.ndarray, player_to_move: int) -> np.ndarray:
    """
    Robust env board -> 1ch POV board in {-1,0,+1}.
    Supports env boards in {0,1,2} or {-1,0,+1}.
    """
    b = np.asarray(board, dtype=np.int8)

    # Already looks like {-1,0,+1}
    if b.min() >= -1 and b.max() <= 1 and np.any(b < 0):
        # player_to_move: 1 => keep, 2 => flip
        sign = 1 if int(player_to_move) == 1 else -1
        return (b * sign).astype(np.int8)

    # Assume {0,1,2}
    p = int(player_to_move)
    me, opp = (1, 2) if p == 1 else (2, 1)

    pov = np.zeros_like(b, dtype=np.int8)
    pov[b == me] = 1
    pov[b == opp] = -1
    return pov


def load_ppo_actor_critic(path: str, device=None):
    """
    Loads ActorCritic from a .pt checkpoint.
    Supports:
      1) CNet192 checkpoint with {'model_state_dict','cfg'} via ActorCritic.from_cnet192_checkpoint
      2) raw state_dict / dict{'state_dict': ...}
    Cached by path.
    """
    path = str(path)
    if path in _PPO_CACHE:
        return _PPO_CACHE[path]

    

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    ckpt = torch.load(path, map_location=dev, weights_only=True)

    # Format 1: CNet192 checkpoint
    if isinstance(ckpt, dict) and ("model_state_dict" in ckpt) and ("cfg" in ckpt):
        ac = ActorCritic.from_cnet192_checkpoint(path=path, device=dev)
        ac.eval()
        _PPO_CACHE[path] = ac
        return ac

    # Format 2: state_dict
    ac = ActorCritic().to(dev)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    ac.load_state_dict(state, strict=False)
    ac.eval()
    _PPO_CACHE[path] = ac
    return ac


def ppo_probs_and_action(
    ac,
    env_board: np.ndarray,
    player_to_move: int,
    legal_actions: List[int],
    *,
    temperature: float = 1.0,
    action_mode: str = "sample",
) -> Tuple[int, np.ndarray]:
    """
    Return (action, probs[7]) for PPO policy, masked to legal actions.
    """


    NEG_INF = -1e9

    pov = _env_board_to_pov_robust(env_board, player_to_move)

    dev = next(ac.parameters()).device
    x = torch.as_tensor(pov, dtype=torch.float32, device=dev)

    logits, _ = ac.forward(x)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, legal_actions] = True

    T = max(float(temperature), 1e-6)
    masked_logits = logits.masked_fill(~mask, NEG_INF) / T
    probs_t = torch.softmax(masked_logits, dim=-1)[0]
    probs = probs_t.detach().cpu().numpy().astype(np.float64, copy=False)

    if action_mode.lower() == "argmax":
        a = int(np.argmax(probs))
    else:
        # sample
        r = random.random()
        c = 0.0
        a = int(np.argmax(probs))
        for i, p in enumerate(probs):
            c += float(p)
            if r <= c:
                a = int(i)
                break

    return a, probs


def _is_random_spec(x: Any) -> bool:
    if isinstance(x, str):
        s = x.strip().upper()
        return s == "R" or s == "RANDOM" or s.startswith("R_")
    return False

def _is_center_spec(x: Any) -> bool:
    if isinstance(x, str):
        s = x.strip().upper()
        return s == "C" or s == "CENTER" or s.startswith("C_")
    return False

def _is_left_spec(x: Any) -> bool:
    if isinstance(x, str):
        s = x.strip().upper()
        return s == "L" or s == "LEFT" or s.startswith("L_")
    return False


def _legal_cols_from_board_toprow(board: np.ndarray) -> np.ndarray:
    """
    Legal columns = columns where top cell is empty (0).
    Works with boards in {0,1,2} or {0,1,-1} as long as empty is 0.
    """
    top = board[0]
    return np.flatnonzero(top == 0)


def _uniform_legal_probs(legal_cols: np.ndarray, n_actions: int = 7) -> np.ndarray:
    probs = np.zeros(n_actions, dtype=np.float64)
    if legal_cols is None or len(legal_cols) == 0:
        return probs
    probs[legal_cols] = 1.0 / float(len(legal_cols))
    return probs

def _uniform_probs_over(cols: Iterable[int], n_actions: int = 7) -> np.ndarray:
    """
    probs[7] uniform over given columns.
    """
    probs = np.zeros(n_actions, dtype=np.float64)
    cols = list(cols)
    if not cols:
        return probs
    w = 1.0 / float(len(cols))
    for c in cols:
        probs[int(c)] = w
    return probs



def choose_move_mixed(
    env: Connect4Env,
    la: Connect4Lookahead,
    spec: int | str,
    noise_p: float,
    force_loss_p: float,
    rng: random.Random,
    teacher_mode: str = "scores",
    return_policy_probs: bool = False,
    *,
    ppo_model=None,
    ppo_temperature: float = 1.0,
    ppo_action: str = "sample",
) -> Tuple[int, Optional[np.ndarray]]:
    """
    spec:
      - int depth => lookahead teacher (optionally noisy)
      - "R"/"Random" => uniform random legal move
      - "....pt" => PPO model checkpoint (ActorCritic)

    Returns:
      action, teacher_probs (optional)
    """
    # --- Random agent ---
    if _is_random_spec(spec):
        legal_cols = _legal_cols_from_board_toprow(env.board)
        if len(legal_cols) == 0:
            a = 3
            probs = None
        else:
            a = int(rng.choice(legal_cols))
            probs = _uniform_legal_probs(legal_cols) if return_policy_probs else None
        return a, probs

    # --- PPO agent ---
    if _is_ppo_path(spec):
        ac = ppo_model or load_ppo_actor_critic(str(spec))
        legal_cols = _legal_cols_from_board_toprow(env.board)
        legal = [int(c) for c in legal_cols.tolist()]
        if len(legal) == 0:
            return 3, (np.zeros(7, dtype=np.float64) if return_policy_probs else None)

        a, probs = ppo_probs_and_action(
            ac,
            env.board,
            int(env.current_player),
            legal,
            temperature=float(ppo_temperature),
            action_mode=str(ppo_action),
        )
        return int(a), (probs if return_policy_probs else None)
    
    # --- Center agent ---
    if _is_center_spec(spec):
        legal_cols = _legal_cols_from_board_toprow(env.board)
        if len(legal_cols) == 0:
            return 3, (np.zeros(7, dtype=np.float64) if return_policy_probs else None)
    
        # pick closest-to-center (3), tie-broken by your usual center-first order
        CENTER_ORDER = (3, 4, 2, 5, 1, 6, 0)
        legal_set = set(int(c) for c in legal_cols.tolist())
        a = next((c for c in CENTER_ORDER if c in legal_set), int(legal_cols[0]))
    
        # policy probs: uniform among all legal cols at minimal distance to center
        dmin = min(abs(int(c) - 3) for c in legal_cols.tolist())
        best = [int(c) for c in legal_cols.tolist() if abs(int(c) - 3) == dmin]
        probs = _uniform_probs_over(best) if return_policy_probs else None
        return int(a), probs
    
    # --- Left agent ---
    if _is_left_spec(spec):
        legal_cols = _legal_cols_from_board_toprow(env.board)
        if len(legal_cols) == 0:
            return 3, (np.zeros(7, dtype=np.float64) if return_policy_probs else None)
    
        # always leftmost legal column
        a = int(legal_cols.min())
    
        # policy probs: one-hot (uniform over the singleton "best" set)
        probs = _uniform_probs_over([a]) if return_policy_probs else None
        return int(a), probs

    # --- Lookahead agent ---
    depth = int(spec)
    a, probs = choose_move_noisy(
        env, la, depth=depth,
        noise_p=noise_p, force_loss_p=force_loss_p,
        rng=rng, teacher_mode=teacher_mode,
    )
    if not return_policy_probs:
        probs = None
    return int(a), probs



def _board_to_cells(board: np.ndarray) -> Dict[str, int]:
    """
    Flatten board into dict columns named "r-c" for r in [0..5], c in [0..6],
    where r=0 is the TOP row (same as board[0, :]).
    """
    rows, cols = board.shape
    out: Dict[str, int] = {}
    for r in range(rows):
        for c in range(cols):
            out[f"{r}-{c}"] = int(board[r, c])
    return out


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of row dicts into a stable, column-ordered DataFrame.

    We preserve your original column order:
      label, reward, game, ply, <optional extras>, 0-0..5-6
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)

    # Board columns in your legacy order
    board_cols = [f"{r}-{c}" for r in range(6) for c in range(7)]

    head_cols = [c for c in ["label", "reward", "game", "ply"] if c in df.columns]
    # keep any extra (action/player/done/probs/scores/etc) but not board, in a stable order
    extra_cols = [c for c in df.columns if c not in set(head_cols + board_cols)]
    df = df[head_cols + sorted(extra_cols) + [c for c in board_cols if c in df.columns]]

    return df


def upsert_excel(df_new: pd.DataFrame, xlsx_path: str | Path, sheet_name: str = "Sheet1") -> pd.DataFrame:
    """
    Append df_new into xlsx_path/sheet_name, drop exact duplicates, and rewrite the sheet.

    This is intentionally simple:
    - duplicates are removed by full-row equality
    - the sheet is rewritten (replace), so it stays clean
    """
    xlsx_path = Path(xlsx_path)
    if df_new is None or len(df_new) == 0:
        return df_new

    if xlsx_path.exists():
        try:
            df_old = pd.read_excel(xlsx_path, sheet_name=sheet_name)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_all = df_new.copy()
        df_all = df_all.drop_duplicates().reset_index(drop=True)

        with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            df_all.to_excel(w, sheet_name=sheet_name, index=False)
        return df_all
    else:
        df_new = df_new.drop_duplicates().reset_index(drop=True)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="w") as w:
            df_new.to_excel(w, sheet_name=sheet_name, index=False)
        return df_new


# --------------------------- Teacher policy helpers ---------------------------

def _uniform_among_best(scores: np.ndarray) -> np.ndarray:
    """
    Given scores[7] where illegal columns are -inf,
    return probs[7] uniform over the best-scored legal actions.
    """
    probs = np.zeros_like(scores, dtype=np.float64)
    best = np.nanmax(scores)
    if not np.isfinite(best):
        return probs
    idx = np.flatnonzero(scores == best)
    if idx.size == 0:
        return probs
    probs[idx] = 1.0 / float(idx.size)
    return probs


def _sample_from_probs(probs: np.ndarray, rng: random.Random) -> int:
    """
    Sample action from probs[7] using Python RNG (deterministic if seeded).
    probs need not sum to 1 exactly.
    """
    p = probs.astype(np.float64, copy=False)
    s = float(p.sum())
    if s <= 0.0:
        return 0
    r = rng.random() * s
    acc = 0.0
    for a in range(p.shape[0]):
        acc += float(p[a])
        if r <= acc:
            return int(a)
    return int(p.shape[0] - 1)


# --------------------------- Noise / safety helpers ---------------------------

def _safe_unsafe_alternatives(
    la: Connect4Lookahead,
    board: np.ndarray,
    player: int,
    exclude: Optional[Iterable[int]] = None,
) -> Tuple[List[int], List[int]]:
    """
    Split legal actions into:
      - safe: does NOT give opponent an immediate win next ply
      - unsafe: does give opponent an immediate win next ply

    Uses the Numba core bitboard routines (fast).
    """
    exclude_set = set(exclude) if exclude is not None else set()

    p1, p2, mask = la._parse_board_bitboards(board)
    me_mark = la._p(player)
    pos = p1 if me_mark == 1 else p2

    N = la._N
    safe: List[int] = []
    unsafe: List[int] = []

    for c in la._CENTER_ORDER:
        if c in exclude_set:
            continue
        if (mask & la.TOP_MASK[c]) != 0:
            continue

        mv = int(la._play_bit_py(int(mask), int(c)))
        nm = int(mask | mv)
        opp_pos = nm ^ (pos | mv)  # position for the next player under Pons convention

        # If opponent has ANY immediate winning reply, this is unsafe
        opp_wins = int(N["count_immediate_wins_bits"](
            np.uint64(opp_pos), np.uint64(nm),
            N["CENTER_ORDER"], N["TOP_MASK"], N["BOTTOM_MASK"], N["COL_MASK"],
            np.int32(la.STRIDE)
        ))
        if opp_wins > 0:
            unsafe.append(int(c))
        else:
            safe.append(int(c))

    return safe, unsafe


def choose_move_noisy(
    env: Connect4Env,
    la: Connect4Lookahead,
    depth: int,
    noise_p: float,
    force_loss_p: float,
    rng: random.Random,
    teacher_mode: str = "scores",
) -> Tuple[int, Optional[np.ndarray]]:
    """
    Noisy teacher move:
      - compute teacher policy (uniform among best) and sample from it
      - with probability noise_p, sample from "safe alternatives" excluding all best moves
      - with probability force_loss_p (evaluated only if we are in noise), sample from unsafe moves
        (useful to seed 'punishment' experiences)

    Returns:
      action, teacher_probs (or None if teacher_mode != "scores")
    """
    teacher_probs = None

    if teacher_mode == "scores":
        scores = la.n_step_action_scores(env.board, env.current_player, depth=depth)
        teacher_probs = _uniform_among_best(scores)
        best_actions = set(np.flatnonzero(teacher_probs > 0.0).tolist())
        best_action = _sample_from_probs(teacher_probs, rng)
    else:
        # fast, deterministic best (center-first ordering)
        best_action = int(la.n_step_lookahead(env.board, env.current_player, depth=depth, tie_break="random"))
        best_actions = {best_action}

    # inject noise?
    if noise_p > 0.0 and rng.random() < float(noise_p):
        safe, unsafe = _safe_unsafe_alternatives(la, env.board, env.current_player, exclude=best_actions)

        if force_loss_p > 0.0 and rng.random() < float(force_loss_p) and len(unsafe) > 0:
            return int(rng.choice(unsafe)), teacher_probs

        if len(safe) > 0:
            return int(rng.choice(safe)), teacher_probs
        # if no safe alternatives exist, fall back to best
        return int(best_action), teacher_probs

    return int(best_action), teacher_probs


# --------------------------- Main generation ---------------------------

def _play_one_game_rows(
    lookA: int | str,
    lookB: int | str,
    label: str,
    game_index: int,
    seed: int = 666,
    CFG: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Play ONE game and return a list of ROW dicts, one per MOVE *we choose to record*.

    IMPORTANT (supervised safety):
      - If the current mover is Random ("R"/"Random"/"R_*"), we still PLAY the move
        to diversify positions, but we DO NOT EMIT a row for that move.
        (Random is poison for supervised labels.)

    lookA/lookB can be:
      - int depth (lookahead)
      - "R" / "Random" (uniform random legal move)

    CFG supports:
      - noiseA, noiseB: float, probability of deviating from teacher best on each move (lookahead only)
      - forceLoss: float, probability to pick an unsafe move when deviating (lookahead only)
      - teacher_mode: "scores" (uniform among best) or "fast" (deterministic)
      - state_mode: "after" (default) or "before"
      - store_action: bool (default True)
      - store_player: bool (default True)
      - store_done: bool (default False)
      - store_policy_probs: bool (default False)
          If True (and move is recorded):
            - lookahead + teacher_mode="scores": p0..p6 = uniform among best
            - (Random mover rows are never stored, so probs never stored for Random)
    """
    CFG = CFG or {}
    rng = random.Random(seed)
    
    

    noiseA = float(CFG.get("noiseA", 0.0))
    noiseB = float(CFG.get("noiseB", 0.0))
    forceLoss = float(CFG.get("forceLoss", 0.0))

    teacher_mode = str(CFG.get("teacher_mode", "scores")).lower()
    state_mode = str(CFG.get("state_mode", "before")).lower()

    store_action = bool(CFG.get("store_action", True))
    store_player = bool(CFG.get("store_player", True))
    store_done = bool(CFG.get("store_done", False))
    store_probs = bool(CFG.get("store_policy_probs", False))
    
    min_supervised_depth = CFG.get("supervised_min_lookahead_depth", MIN_LA_DEPTH_STORED)
    min_supervised_depth = None if min_supervised_depth is None else int(min_supervised_depth)

    #exclude_ppo_supervised = bool(CFG.get("exclude_ppo_supervised", True))  # optional

    env = Connect4Env()
    la = Connect4Lookahead()
    
    acA = load_ppo_actor_critic(lookA) if _is_ppo_path(lookA) else None
    acB = load_ppo_actor_critic(lookB) if _is_ppo_path(lookB) else None




    _ = env.reset()
    
    if acA is not None:
        acA.begin_episode()
    if acB is not None:
        acB.begin_episode()


    rows: List[Dict[str, Any]] = []
    ply = 0

    while True:
        ply += 1
        player = int(env.current_player)

        spec = (lookA if player == 1 else lookB)
        noise_p = noiseA if player == 1 else noiseB
        
        is_weak_lookahead = (
            (min_supervised_depth is not None)
            and isinstance(spec, (int, np.integer))
            and int(spec) < min_supervised_depth
        )

        is_excluded_turn = (
            _is_random_spec(spec)
            or _is_center_spec(spec)
            or _is_left_spec(spec)
            or is_weak_lookahead
            or _is_ppo_path(spec)   # NEW: don't emit rows for PPO moves
        )

        # Only copy "before" state if we're going to record and we need it
        board_before = None
        if (not is_excluded_turn) and (state_mode == "before"):
            board_before = np.array(env.board, copy=True)


        # Choose action. If Random turn, we do not request policy probs at all.
        ppo_T = float(CFG.get("ppo_temperature", 1.0))
        ppo_mode = str(CFG.get("ppo_action", "sample")).lower()


        action, teacher_probs = choose_move_mixed(
            env,
            la,
            spec=spec,
            noise_p=noise_p,
            force_loss_p=forceLoss,
            rng=rng,
            teacher_mode=teacher_mode,
            return_policy_probs=(store_probs and (not is_excluded_turn)),
            ppo_model=(acA if player == 1 else acB),
            ppo_temperature=ppo_T,
            ppo_action=ppo_mode,
        )


        step_out = env.step(int(action))
        if isinstance(step_out, (tuple, list)):
            if len(step_out) >= 3:
                reward, done = step_out[1], step_out[2]
            else:
                reward, done = 0.0, False
        else:
            reward, done = 0.0, False

        # Supervised safety: NEVER emit a row when Random was the mover.
        if not is_excluded_turn:
            if state_mode == "after":
                board_store = np.array(env.board, copy=True)
            else:
                board_store = board_before  # already copied

            row: Dict[str, Any] = {
                "label": str(label),
                "reward": float(reward),
                "game": int(game_index),
                "ply": int(ply),
                **_board_to_cells(board_store),
            }

            if store_player:
                row["player"] = int(player)
            if store_action:
                row["action"] = int(action)
            if store_done:
                row["done"] = bool(done)

            if store_probs and teacher_probs is not None:
                for a in range(7):
                    row[f"p{a}"] = float(teacher_probs[a])

            rows.append(row)

        if bool(done):
            break

    return rows



def generate_dataset(
    PLAYS: Dict[str, Dict[str, Any]],
    seed: int = 666,
    CFG: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate a dataset from a PLAYS dict like:
      PLAYS = {
        "L5L7":     {"A": 5,   "B": 7,   "games": 10},
        "L7_vs_R":  {"A": 7,   "B": "R", "games": 50},
        "R_vs_L7":  {"A": "R", "B": 7,   "games": 50},
      }

    Returns:
      list of row dicts (moves across all games/labels)
    """
    CFG = CFG or {}

    all_rows: List[Dict[str, Any]] = []
    base_seed = int(seed)

    for label, spec in PLAYS.items():
        lookA = spec.get("A", 1)
        lookB = spec.get("B", 1)
        games = int(spec.get("games", 1))

        # normalize A/B: keep "R" as-is, cast others to int
        if (not _is_random_spec(lookA)) and (not _is_center_spec(lookA)) and (not _is_left_spec(lookA)) and (not _is_ppo_path(lookA)):
            lookA = int(lookA)
        if (not _is_random_spec(lookB)) and (not _is_center_spec(lookB)) and (not _is_left_spec(lookB)) and (not _is_ppo_path(lookB)):
            lookB = int(lookB)


        for g in range(games):
            s = base_seed + (hash(label) & 0xFFFF) + g * 9973
            rows = _play_one_game_rows(
                lookA=lookA,
                lookB=lookB,
                label=str(label),
                game_index=g,
                seed=s,
                CFG=CFG,
            )
            all_rows.extend(rows)

    return all_rows
