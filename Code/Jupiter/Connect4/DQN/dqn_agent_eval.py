# DQN/dqn_agent_eval.py
"""
Unified evaluation helpers for DQN agents (Q-net) in the "new line" Connect-4 codebase.

Assumptions:
- board: np.ndarray (6,7), dtype int8 preferred, values typically in {0, +1, -1}
- DQN agent:
    agent.board_to_state(board, player) -> (C,6,7) or (1,C,6,7)
    agent.q_net(state_tensor) -> (B,7) Q-values

Opponents:
- Random / Leftmost / Center
- Lookahead-k via Connect4Lookahead (if provided)

Excel row includes:
- per-opponent win_rate columns
- GLOBAL_SCORE (depth-weighted)
- check_score / ensemble_score / center_rate (optional extras you asked for)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from C4.connect4_env import (
    ROWS, COLS, STRIDE,
    TOP_MASK, FULL_MASK,
    _bb_has_won,
    UINT,
)

NEG_INF = -1e9

OpponentFn = Callable[[np.ndarray, int, np.uint64], int]

def _resolve_model(agent: Any, model: Optional[torch.nn.Module]) -> Optional[torch.nn.Module]:
    if model is not None:
        return model
    if agent is None:
        return None

    # common DQNAgent field
    qn = getattr(agent, "q_net", None)
    if isinstance(qn, torch.nn.Module):
        return qn

    # sometimes agent itself is the model
    if isinstance(agent, torch.nn.Module):
        return agent

    # wrappers
    for attr in ("model", "net", "policy", "student"):
        m = getattr(agent, attr, None)
        if isinstance(m, torch.nn.Module):
            return m

    return None

def _resolve_board_to_state_fn(agent: Any, board_to_state_fn: Optional[Callable]) -> Optional[Callable]:
    if callable(board_to_state_fn):
        return board_to_state_fn
    if agent is None:
        return None

    # direct names (your project has historically used board_to_state)
    candidates = (
        "board_to_state",
        "board_to_state_4ch",
        "board_to_state4",
        "encode_board_to_state",
        "board_to_state_fn",
    )
    for name in candidates:
        fn = getattr(agent, name, None)
        if callable(fn):
            return fn

    # nested wrappers, just in case
    for wrap in ("agent", "student", "dqn", "inner", "wrapped"):
        sub = getattr(agent, wrap, None)
        if sub is None:
            continue
        for name in candidates:
            fn = getattr(sub, name, None)
            if callable(fn):
                return fn

    return None


def _hint_state_fns(obj: Any) -> str:
    if obj is None:
        return ""
    names = [n for n in dir(obj) if ("state" in n.lower()) or ("encode" in n.lower())]
    names = sorted(names)[:40]
    if not names:
        return ""
    return "Found these related attrs: " + ", ".join(names)


# ----------------------------- scoring helpers -----------------------------

def opponent_weight(
    label: str,
    base: float = 1.4,
    random_weight: float = 1.0,
    default_weight: float = 1.0,
) -> float:
    s = str(label)
    if "random" in s.lower() or s.strip().upper() in ("R", "RND"):
        return float(random_weight)

    m = re.search(r"(\d+)", s)
    if m:
        depth = int(m.group(1))
        return float(base) ** depth

    return float(default_weight)


def global_score_from_suite_df(
    suite_df: pd.DataFrame,
    base: float = 1.4,
) -> float:
    if suite_df is None or suite_df.empty:
        return float("nan")

    num = 0.0
    den = 0.0
    for _, r in suite_df.iterrows():
        label = str(r.get("opponent", ""))
        wr = float(r.get("win_rate", 0.0))
        w = opponent_weight(label, base=float(base))
        num += w * wr
        den += w
    return float(num / den) if den > 1e-12 else float("nan")


# ----------------------------- internal helpers -----------------------------

def _as_uint64(x: np.uint64 | int) -> np.uint64:
    return UINT(int(x))


def _require_board(board: np.ndarray) -> None:
    if not isinstance(board, np.ndarray):
        raise TypeError(f"board must be np.ndarray, got {type(board)}")
    if board.shape != (ROWS, COLS):
        raise ValueError(f"board must have shape ({ROWS},{COLS}), got {board.shape}")


def _legal_mask_to_cols(mask: np.uint64) -> List[int]:
    out: List[int] = []
    for c in range(COLS):
        if (mask & TOP_MASK[c]) == 0:
            out.append(c)
    return out


def _apply_action_inplace(
    board: np.ndarray,
    heights: np.ndarray,
    pos1: np.uint64,
    pos2: np.uint64,
    mask: np.uint64,
    col: int,
    player: int,
) -> Tuple[np.uint64, np.uint64, np.uint64]:
    _require_board(board)

    c = int(col)
    if c < 0 or c >= COLS:
        raise ValueError(f"Illegal move: col={c}")

    h = int(heights[c])
    if h >= ROWS:
        raise ValueError(f"Illegal move: column {c} is full")

    bit = _as_uint64(1) << _as_uint64(c * STRIDE + h)
    mask = mask | bit
    if player == 1:
        pos1 = pos1 | bit
    elif player == -1:
        pos2 = pos2 | bit
    else:
        raise ValueError(f"player must be +1 or -1, got {player}")

    heights[c] = h + 1
    board[ROWS - 1 - h, c] = np.int8(player)
    return pos1, pos2, mask


def _argmax_legal_center_tiebreak(
    qvals: np.ndarray,
    legal: List[int],
    center: int = 3,
    tol: float = 1e-12,
) -> int:
    if not legal:
        return 0
    best = np.max([qvals[c] for c in legal])
    tied = [c for c in legal if abs(float(qvals[c]) - float(best)) <= tol]
    if len(tied) == 1:
        return int(tied[0])
    return int(sorted(tied, key=lambda c: (abs(c - center), c))[0])


def _to_state_tensor(state: Any, device: torch.device) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        x = state
    else:
        x = torch.from_numpy(np.asarray(state))
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"state must be (C,6,7) or (1,C,6,7), got {tuple(x.shape)}")
    return x.to(device=device, dtype=torch.float32)


def _dqn_greedy_action(
    model: torch.nn.Module,
    board: np.ndarray,
    player: int,
    device: torch.device,
    board_to_state_fn: Callable[[np.ndarray, int], Any],
    mask: Optional[np.uint64] = None,
) -> int:
    _require_board(board)

    if mask is None:
        legal = [c for c in range(COLS) if board[0, c] == 0]
    else:
        legal = _legal_mask_to_cols(_as_uint64(mask))
    if not legal:
        return 0

    st = board_to_state_fn(board, int(player))
    x = _to_state_tensor(st, device=device)

    was_training = bool(getattr(model, "training", False))
    model.eval()
    with torch.no_grad():
        out = model(x)
    if was_training:
        model.train(True)

    if isinstance(out, (tuple, list)):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise TypeError("Q-model output must be a Tensor (or tuple/list with first Tensor first).")

    q = out.detach().cpu().numpy().astype(np.float64, copy=False)
    if q.ndim == 2:
        q = q[0]
    if q.shape[0] != COLS:
        raise ValueError(f"Q-values must have length {COLS}, got {q.shape}")

    q_masked = q.copy()
    illegal = set(range(COLS)) - set(legal)
    for c in illegal:
        q_masked[c] = -1e18

    return _argmax_legal_center_tiebreak(q_masked, legal)


# ----------------------------- opponent factory -----------------------------

def _parse_lookahead_depth(label: str) -> Optional[int]:
    m = re.search(r"(\d+)", str(label).strip())
    return int(m.group(1)) if m else None


def make_opponent(
    label: str,
    lookahead: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
) -> OpponentFn:
    if rng is None:
        rng = np.random.default_rng()

    s0 = (label or "").strip()
    s = s0.lower()

    has_la_baselines = (lookahead is not None) and hasattr(lookahead, "baseline_action")

    if s in ("random", "rnd", "r"):
        if has_la_baselines:
            def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
                return int(lookahead.baseline_action(board, kind="random", rng=rng))
            return _op
        def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
            legal = _legal_mask_to_cols(_as_uint64(mask))
            return int(rng.choice(legal)) if legal else 0
        return _op

    if s in ("leftmost", "left", "lm"):
        if has_la_baselines:
            def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
                return int(lookahead.baseline_action(board, kind="leftmost", rng=rng))
            return _op
        def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
            m = _as_uint64(mask)
            for c in range(COLS):
                if (m & TOP_MASK[c]) == 0:
                    return int(c)
            return 0
        return _op

    if s in ("center", "centre", "c"):
        if has_la_baselines:
            def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
                return int(lookahead.baseline_action(board, kind="center", rng=rng))
            return _op
        order = (3, 4, 2, 5, 1, 6, 0)
        def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
            m = _as_uint64(mask)
            for c in order:
                if (m & TOP_MASK[c]) == 0:
                    return int(c)
            return 0
        return _op

    if s.startswith("lookahead") or (len(s0) >= 2 and s0[0].upper() == "L" and s0[1:].isdigit()):
        if lookahead is None:
            raise ValueError("Lookahead requested but lookahead=None. Pass your Connect4Lookahead instance.")
        depth = _parse_lookahead_depth(s0)
        depth = int(depth) if depth is not None else 3
        def _op(board: np.ndarray, player: int, mask: np.uint64) -> int:
            return int(lookahead.n_step_lookahead(board, player, depth=depth))
        return _op

    raise ValueError(f"Unknown opponent label: {label!r}")


# ----------------------------- core eval loop -----------------------------

@dataclass
class GameResult:
    winner: int
    plies: int


def play_one_game(
    model: torch.nn.Module,
    board_to_state_fn: Callable[[np.ndarray, int], Any],
    opponent: OpponentFn,
    device: torch.device,
    seed: int = 666,
    policy_player: int = 1,
    max_plies: int = 42,
) -> GameResult:
    if policy_player not in (+1, -1):
        raise ValueError(f"policy_player must be +1/-1, got {policy_player}")

    rng = np.random.default_rng(int(seed))

    board = np.zeros((ROWS, COLS), dtype=np.int8)
    heights = np.zeros(COLS, dtype=np.int8)
    pos1 = UINT(0)
    pos2 = UINT(0)
    mask = UINT(0)

    player = 1
    plies = 0

    for _ in range(int(max_plies)):
        if player == policy_player:
            a = _dqn_greedy_action(
                model=model,
                board=board,
                player=player,
                device=device,
                board_to_state_fn=board_to_state_fn,
                mask=mask,
            )
        else:
            a = int(opponent(board, player, mask))

        if (a < 0) or (a >= COLS) or ((mask & TOP_MASK[int(a)]) != 0):
            return GameResult(winner=-player, plies=plies)

        pos1, pos2, mask = _apply_action_inplace(board, heights, pos1, pos2, mask, int(a), player)
        plies += 1

        me_bb = pos1 if player == 1 else pos2
        if _bb_has_won(me_bb, STRIDE):
            return GameResult(winner=player, plies=plies)

        if mask == FULL_MASK:
            return GameResult(winner=0, plies=plies)

        player = -player

    return GameResult(winner=0, plies=plies)


def evaluate_vs_opponent(
    model: torch.nn.Module,
    board_to_state_fn: Callable[[np.ndarray, int], Any],
    opponent_label: str,
    device: torch.device,
    lookahead: Optional[Any] = None,
    n_games: int = 200,
    seed: int = 666,
    swap_sides: bool = True,
    progress: bool = True,
) -> Dict[str, Any]:
    n_games = int(n_games)
    if n_games <= 0:
        raise ValueError("n_games must be > 0")
    if swap_sides and (n_games % 2 == 1):
        n_games += 1

    rng = np.random.default_rng(int(seed))
    opp = make_opponent(opponent_label, lookahead=lookahead, rng=rng)

    wins = losses = draws = 0
    total_plies = 0

    it = range(n_games)
    if progress:
        it = tqdm(it, total=n_games, desc=f"Eval vs {opponent_label}", leave=True)

    for i in it:
        policy_player = 1 if (not swap_sides or (i % 2 == 0)) else -1
        gr = play_one_game(
            model=model,
            board_to_state_fn=board_to_state_fn,
            opponent=opp,
            device=device,
            seed=int(rng.integers(0, 2**31 - 1)),
            policy_player=policy_player,
        )

        total_plies += int(gr.plies)

        if gr.winner == 0:
            draws += 1
        elif gr.winner == policy_player:
            wins += 1
        else:
            losses += 1

    score = (wins + 0.5 * draws) / float(n_games)

    return {
        "opponent": str(opponent_label),
        "games": int(n_games),
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "win_rate": float(wins / n_games),
        "loss_rate": float(losses / n_games),
        "draw_rate": float(draws / n_games),
        "score": float(score),
        "avg_plies": float(total_plies / n_games),
    }


def evaluate_suite(
    model: torch.nn.Module,
    board_to_state_fn: Callable[[np.ndarray, int], Any],
    opponents: Union[Dict[str, int], Iterable[str]],
    device: torch.device,
    lookahead: Optional[Any] = None,
    n_games_default: int = 200,
    seed: int = 666,
    swap_sides: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    if isinstance(opponents, dict):
        items = [(str(k), int(v)) for k, v in opponents.items()]
    else:
        items = [(str(o), int(n_games_default)) for o in opponents]

    rows: List[Dict[str, Any]] = []
    for label, ng in items:
        rows.append(
            evaluate_vs_opponent(
                model=model,
                board_to_state_fn=board_to_state_fn,
                opponent_label=label,
                device=device,
                lookahead=lookahead,
                n_games=int(ng),
                seed=int(seed),
                swap_sides=bool(swap_sides),
                progress=bool(progress),
            )
        )
    return pd.DataFrame(rows)


# ----------------------------- Excel logging -----------------------------

def suite_df_to_row(
    run_tag: str,
    suite_df: pd.DataFrame,
    elapsed_h: float,
    episodes: Optional[int] = None,
    global_base: float = 1.4,
    check_score: Optional[float] = None,
    ensemble_score: Optional[float] = None,
    center_rate: Optional[float] = None,
) -> pd.DataFrame:
    row: Dict[str, Any] = {
        "TRAINING_SESSION": str(run_tag),
        "TIME [h]": round(float(elapsed_h), 6),
        "EPISODES": episodes,
    }

    # per-opponent win rates
    for _, r in suite_df.iterrows():
        row[str(r.get("opponent", ""))] = float(r.get("win_rate", 0.0))

    # depth-weighted global score (same definition as your PPO helper)
    row["GLOBAL_SCORE"] = float(global_score_from_suite_df(suite_df, base=float(global_base)))

    # extras you requested
    if check_score is not None:
        row["check_score"] = float(check_score)
    if ensemble_score is not None:
        row["ensemble_score"] = float(ensemble_score)
    if center_rate is not None:
        row["center_rate"] = float(center_rate)

    return pd.DataFrame([row]).set_index("TRAINING_SESSION")


def append_eval_row_to_excel(df_row: pd.DataFrame, excel_path: str) -> None:
    excel_path = str(excel_path)

    if os.path.exists(excel_path):
        old = pd.read_excel(excel_path)
        if not old.empty and not old.isnull().all().all():
            new = pd.concat([old, df_row.reset_index()], ignore_index=True)
        else:
            new = df_row.reset_index()
    else:
        new = df_row.reset_index()

    os.makedirs(os.path.dirname(excel_path) or ".", exist_ok=True)
    new.to_excel(excel_path, index=False)



def evaluate_and_log_to_excel(
    *,
    agent: Any = None,
    model: Optional[torch.nn.Module] = None,
    board_to_state_fn: Optional[Callable] = None,
    opponents_cfg: Union[Dict[str, int], Iterable[str], None] = None,
    excel_path: str = "EVAL_DQN_results.xlsx",
    run_tag: str = "DQN",
    device: Optional[torch.device] = None,
    lookahead: Optional[Any] = None,
    seed: int = 666,
    episodes: Optional[int] = None,
    progress: bool = True,
    swap_sides: bool = True,
    global_base: float = 1.4,
    check_score: Optional[float] = None,
    ensemble_score: Optional[float] = None,
    center_rate: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    DQN analogue of PPO.ppo_agent_eval.evaluate_and_log_to_excel().

    Accepts either:
      - agent=... (expects agent.q_net and agent.board_to_state-like fn), or
      - model=... and board_to_state_fn=... explicitly.

    Also logs GLOBAL_SCORE + check_score/ensemble_score/center_rate into the Excel row.
    """
    if opponents_cfg is None:
        raise ValueError("evaluate_and_log_to_excel needs opponents_cfg.")

    model = _resolve_model(agent, model)
    if model is None:
        raise ValueError("evaluate_and_log_to_excel needs model (or agent with .q_net).")

    board_to_state_fn = _resolve_board_to_state_fn(agent, board_to_state_fn)
    if board_to_state_fn is None:
        msg = (
            "evaluate_and_log_to_excel needs board_to_state_fn "
            "(or agent exposing board_to_state-like method).\n"
            + _hint_state_fns(agent)
        )
        raise ValueError(msg)

    t0 = time.time()

    suite_df = evaluate_suite(
        model=model,
        board_to_state_fn=board_to_state_fn,
        opponents=opponents_cfg,
        device=device,
        lookahead=lookahead,
        seed=int(seed),
        swap_sides=bool(swap_sides),
        progress=bool(progress),
    )

    elapsed_h = (time.time() - t0) / 3600.0

    row_df = suite_df_to_row(
        run_tag=run_tag,
        suite_df=suite_df,
        elapsed_h=elapsed_h,
        episodes=episodes,
        global_base=float(global_base),
        check_score=check_score,
        ensemble_score=ensemble_score,
        center_rate=center_rate,
    )

    append_eval_row_to_excel(row_df, excel_path)

    return suite_df, row_df


# ----------------------------- plotting -----------------------------

def plot_eval_bar_summary_from_suite_df(
    suite_df: pd.DataFrame,
    model_name: str = "DQN",
    save_path: Optional[str] = None,
    rotation: int = 15,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 150,
):
    import matplotlib.pyplot as plt

    labels = [str(x) for x in suite_df["opponent"].tolist()]
    win_rates = (suite_df["win_rate"].astype(float) * 100.0).tolist()
    loss_rates = (suite_df["loss_rate"].astype(float) * 100.0).tolist()
    draw_rates = (suite_df["draw_rate"].astype(float) * 100.0).tolist()

    x = np.arange(len(labels), dtype=np.float64)
    bar_w = 0.25

    fig = plt.figure(figsize=figsize, dpi=int(dpi))
    ax = fig.add_subplot(111)

    ax.bar(x, win_rates, width=bar_w, label="Win %")
    ax.bar(x + bar_w, loss_rates, width=bar_w, label="Loss %")
    ax.bar(x + 2 * bar_w, draw_rates, width=bar_w, label="Draw %")

    ax.set_xlabel("Opponent")
    ax.set_ylabel("Percentage")
    ax.set_title(f"{model_name} vs opponents")
    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(labels, rotation=rotation, ha="right")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=int(dpi))
        print(f"[Saved] {save_path}")

    return fig


__all__ = [
    "opponent_weight",
    "global_score_from_suite_df",
    "make_opponent",
    "evaluate_vs_opponent",
    "evaluate_suite",
    "evaluate_and_log_to_excel",
    "plot_eval_bar_summary_from_suite_df",
]