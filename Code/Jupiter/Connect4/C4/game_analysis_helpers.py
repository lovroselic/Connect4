#C4.game_analysis_helpers.py

import numpy as np
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from C4.fast_connect4_lookahead import Connect4Lookahead  

ROWS, COLS = 6,7
la = Connect4Lookahead()

def board_list_to_np(board_list, rows=6, cols=7) -> np.ndarray:
    """Kaggle board list -> (rows, cols) with row 0 = top."""
    arr = np.array(board_list, dtype=np.int8)
    return arr.reshape((rows, cols))

def find_board_in_pair(step_pair):
    """Return (board_np, step_no) if a board exists in either agent observation, else (None, None)."""
    for agent_entry in step_pair:
        obs = agent_entry.get("observation", {})
        if "board" in obs and obs["board"] is not None:
            b = board_list_to_np(obs["board"], ROWS, COLS)
            return b, int(obs.get("step", -1))
    return None, None

def get_overage_in_pair(step_pair):
    """Return [overage_agent0, overage_agent1] where values may be None if missing."""
    out = [None, None]
    for i, agent_entry in enumerate(step_pair):
        obs = agent_entry.get("observation", {})
        if "remainingOverageTime" in obs:
            out[i] = float(obs["remainingOverageTime"])
    return out

def diff_one_move(prev: np.ndarray, cur: np.ndarray):
    """
    Find the single placed disc from prev->cur.
    Returns (mark, (r,c)). mark in {1,2}. r is row index (top-based).
    """
    d = (cur != prev)
    idx = np.argwhere(d)
    if idx.shape[0] != 1:
        return None
    r, c = int(idx[0, 0]), int(idx[0, 1])
    mark = int(cur[r, c])
    return mark, (r, c)

def apply_move(board: np.ndarray, col: int, mark: int) -> np.ndarray:
    """Return new board after dropping mark (1 or 2) into col."""
    b = board.copy()
    if b[0, col] != 0:
        raise ValueError(f"Illegal move: column {col} is full")
    for r in range(ROWS - 1, -1, -1):
        if b[r, col] == 0:
            b[r, col] = mark
            return b
    raise ValueError("Unreachable: column said not full but no empty cell found")

def legal_cols(board: np.ndarray):
    return [c for c in range(COLS) if board[0, c] == 0]



def draw_board(ax, board: np.ndarray, title: str = ""):
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_title(title)

    for rr in range(ROWS):
        for cc in range(COLS):
            v = int(board[rr, cc])
            if v == 1:
                face = "red"
                mark = "X"
            elif v == 2:
                face = "gold"
                mark = "O"
            else:
                face = "white"
                mark = ""

            ax.add_patch(Circle((cc, rr), 0.42, facecolor=face, edgecolor="black", linewidth=1))
            if mark:
                ax.text(cc, rr, mark, ha="center", va="center", fontsize=18, fontweight="bold", color="black")

    ax.grid(True, linestyle="--", alpha=0.25)

def show_two_boards(boardA, titleA, boardB, titleB, suptitle=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    draw_board(axes[0], boardA, titleA)
    draw_board(axes[1], boardB, titleB)
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

def opp_mark(mark: int) -> int:
    return 2 if mark == 1 else 1

def win_cols(board: np.ndarray, mark: int):
    # LA helper returns immediate winning columns for that mark
    return list(la.count_immediate_wins(board, mark))

def safe_moves(board: np.ndarray, mark: int):
    """Moves that do NOT give opponent an immediate win next."""
    opp = opp_mark(mark)
    safe = []
    for c in legal_cols(board):
        after = apply_move(board, c, mark)
        if len(win_cols(after, opp)) == 0:
            safe.append(c)
    return safe

def pure_features(board: np.ndarray, mark: int):
    """A minimal feature set that maps to your heuristic knobs."""
    opp = opp_mark(mark)
    return {
        "me_pure2": int(la.count_pure(board, mark, 2)),
        "me_pure3": int(la.count_pure(board, mark, 3)),
        "opp_pure2": int(la.count_pure(board, opp, 2)),
        "opp_pure3": int(la.count_pure(board, opp, 3)),
        "me_imw": len(win_cols(board, mark)),
        "opp_imw": len(win_cols(board, opp)),
        "center_occupied": int(board[ROWS-1, 3] != 0),  # bottom center
    }

def feature_delta(f_after: dict, f_before: dict):
    return {k: (f_after[k] - f_before[k]) for k in f_before.keys()}

def toList(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    try:
        return list(x)
    except Exception:
        return [x]

def la_move_after(board0: np.ndarray, mark: int, col: int) -> np.ndarray:
    if col not in legal_cols(board0):
        return None
    return apply_move(board0, col, mark)

def move_eval_pack(board0: np.ndarray, mark: int, col: int) -> dict:
    b1 = la_move_after(board0, mark, col)
    if b1 is None:
        return None

    opp = opp_mark(mark)

    my_imw  = len(toList(la.count_immediate_wins(b1, mark)))
    opp_imw = len(toList(la.count_immediate_wins(b1, opp)))

    my_p3   = int(la.count_pure(b1, mark, 3))
    opp_p3  = int(la.count_pure(b1, opp, 3))

    fork = la.compute_fork_signals(board0, b1, mover=mark)
    my_fork_after  = int(fork.get("my_after", 0))
    opp_fork_after = int(fork.get("opp_after", 0))

    return {
        "col": int(col),
        "my_imw": my_imw,
        "opp_imw": opp_imw,
        "my_p3": my_p3,
        "opp_p3": opp_p3,
        "my_fork_after": my_fork_after,
        "opp_fork_after": opp_fork_after,
    }

def argmin_lex(packs, keys):
    best = None
    best_key = None
    for p in packs:
        if p is None:
            continue
        kk = tuple(p[k] for k in keys)
        if best is None or kk < best_key:
            best, best_key = p, kk
    return best

def argmax_lex(packs, keys):
    best = None
    best_key = None
    for p in packs:
        if p is None:
            continue
        kk = tuple(p[k] for k in keys)
        if best is None or kk > best_key:
            best, best_key = p, kk
    return best


def get_la_knobs(la):
    knobs = {
        "DEFENSIVE": float(getattr(la, "DEFENSIVE", np.nan)),
        "FLOATING_NEAR": float(getattr(la, "FLOATING_NEAR", np.nan)),
        "FLOATING_FAR": float(getattr(la, "FLOATING_FAR", np.nan)),
        "CENTER_BONUS": float(getattr(la, "CENTER_BONUS", np.nan)),
        "PARITY_BONUS": float(getattr(la, "PARITY_BONUS", np.nan)),
        "VERT_MUL": float(getattr(la, "VERT_MUL", np.nan)),
        "VERT_3_READY_BONUS": float(getattr(la, "VERT_3_READY_BONUS", np.nan)),
        "TEMPO_W": float(getattr(la, "TEMPO_W", np.nan)),
        "THREATSPACE_W": float(getattr(la, "THREATSPACE_W", np.nan)),
        "PARITY_MOVE_W": float(getattr(la, "PARITY_MOVE_W", np.nan)),
        "PARITY_UNLOCK_W": float(getattr(la, "PARITY_UNLOCK_W", np.nan)),
        "immediate_w": float(getattr(la, "immediate_w", np.nan)),
        "fork_w": float(getattr(la, "fork_w", np.nan)),
    }
    w = getattr(la, "weights", {})
    knobs["W2"] = float(w.get(2, np.nan))
    knobs["W3"] = float(w.get(3, np.nan))
    knobs["W4"] = float(w.get(4, np.nan))
    return knobs

def clamp(x, lo=None, hi=None):
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x

def compute_coward_rate_lovro(df, LOVRO):
    """
    Lovro-only.
    'Coward' = LA picks a SAFE move (opp_imw==0) but ignores a *safe* alternative that gives
    strictly better attack signals (my_imw/my_p3/my_fork_after).
    """
    if df is None or len(df) == 0:
        return 0.0
    if "board_before" not in df.columns:
        return 0.0

    # Filter to Lovro (robust even if df already filtered)
    df_local = df.copy()
    if "mark" in df_local.columns:
        df_local = df_local[df_local["mark"] == LOVRO]
    df_local = df_local.reset_index(drop=True)

    coward = 0
    denom = 0

    for i in range(len(df_local)):
        b0 = df_local.iloc[i]["board_before"]
        mark = int(df_local.iloc[i]["mark"])   # == LOVRO
        la_col = int(df_local.iloc[i]["la_col"])

        legal0 = legal_cols(b0)
        if not legal0:
            continue

        packs = [move_eval_pack(b0, mark, c) for c in legal0]
        pack_la = move_eval_pack(b0, mark, la_col)
        if pack_la is None:
            continue

        safe = [p for p in packs if (p is not None and p["opp_imw"] == 0)]
        if len(safe) < 2:
            continue

        pack_safe_att = argmax_lex(safe, keys=("my_imw", "my_p3", "my_fork_after"))
        if pack_safe_att is None:
            continue

        if pack_la["opp_imw"] != 0:
            continue

        denom += 1

        worse_p3 = (pack_safe_att["my_p3"] - pack_la["my_p3"]) >= 1
        worse_fork = (pack_safe_att["my_fork_after"] - pack_la["my_fork_after"]) >= 1
        worse_imw = (pack_safe_att["my_imw"] - pack_la["my_imw"]) >= 1

        if (worse_imw or worse_p3 or worse_fork) and (pack_safe_att["col"] != pack_la["col"]):
            coward += 1

    return 100.0 * (coward / denom) if denom > 0 else 0.0

def patch_to_rows(patch: dict, label: str, kn):
    rows = []
    for k, newv in patch.items():
        cur = kn.get(k, np.nan)
        rows.append({
            "patch": label,
            "knob": k,
            "current": cur,
            "proposed": float(newv),
            "delta": float(newv) - float(cur) if (cur == cur) else np.nan,
        })
    return rows

def _build_WARR_from_la(la):
    WARR = np.zeros(la.K + 1, dtype=np.float64)
    for k in (2, 3, 4):
        WARR[k] = float(la.weights.get(k, 0.0))
    return WARR

def root_bias_components(la, board, player, col):
    """
    Replicate root_select_fixed's bias terms for one move:
      - parity move preference
      - parity unlock penalty
      - threat-space (immediate-win count after move)
    """
    p1, p2, mask = la._parse_board_bitboards(board)
    me_mark = la._p(player)
    me = p1 if me_mark == 1 else p2

    role_first_mark, parity_enabled = la._role_first_from_center(mask, p1, p2)
    root_pos_is_first = bool(parity_enabled and (me_mark == role_first_mark))

    # column heights at root (landing row r)
    H = [(mask & la.COL_MASK[c]).bit_count() for c in range(la.COLS)]
    r = H[col]

    bias_parity_move = 0.0
    bias_parity_unlock = 0.0

    if parity_enabled:
        root_is_first = root_pos_is_first
        pref_parity_root = 0 if root_is_first else 1
        pref_parity_opp  = 1 if root_is_first else 0

        if (r & 1) == pref_parity_root:
            bias_parity_move += float(la.PARITY_MOVE_W)
        else:
            bias_parity_move -= float(la.PARITY_MOVE_W)

        r2 = r + 1
        if r2 < la.ROWS and ((r2 & 1) == pref_parity_opp):
            bias_parity_unlock -= float(la.PARITY_UNLOCK_W)

    # play move in bitboard form
    mv = la._play_bit_py(mask, col)
    nm = mask | mv
    my_after = me | mv
    opp_after = nm ^ my_after

    # threat-space counts (uses the same numba helper as root_select_fixed)
    my_threats = int(la._N["count_immediate_wins_bits"](
        np.uint64(my_after),
        np.uint64(nm),
        la._N["CENTER_ORDER"],
        la._N["TOP_MASK"],
        la._N["BOTTOM_MASK"],
        la._N["COL_MASK"],
        np.int32(la.STRIDE),
    ))
    opp_threats = int(la._N["count_immediate_wins_bits"](
        np.uint64(opp_after),
        np.uint64(nm),
        la._N["CENTER_ORDER"],
        la._N["TOP_MASK"],
        la._N["BOTTOM_MASK"],
        la._N["COL_MASK"],
        np.int32(la.STRIDE),
    ))

    bias_threatspace = 0.0
    if float(la.THREATSPACE_W) != 0.0:
        bias_threatspace += float(la.THREATSPACE_W) * float(my_threats)
        bias_threatspace -= float(la.DEFENSIVE) * float(la.THREATSPACE_W) * 0.25 * float(opp_threats)

    bias_total = float(bias_parity_move + bias_parity_unlock + bias_threatspace)

    return {
        "parity_enabled": bool(parity_enabled),
        "root_pos_is_first": bool(root_pos_is_first),
        "landing_row_r": int(r),
        "bias_parity_move": float(bias_parity_move),
        "bias_parity_unlock": float(bias_parity_unlock),
        "bias_threatspace": float(bias_threatspace),
        "bias_total": bias_total,
        "my_threats": int(my_threats),
        "opp_threats": int(opp_threats),
    }

def child_search_value(la, board, player, depth, col):
    """
    Compute the 'search part' used at root for a move:
      val_search = -negamax(next_pos, nm, depth-1, ...)    (no root bias)
    This matches root_select_fixed's 'val = -v + bias'.
    """
    p1, p2, mask = la._parse_board_bitboards(board)
    me_mark = la._p(player)
    me = p1 if me_mark == 1 else p2

    role_first_mark, parity_enabled = la._role_first_from_center(mask, p1, p2)
    root_pos_is_first = np.int8(1 if (parity_enabled and me_mark == role_first_mark) else 0)
    parity_enabled_i8 = np.int8(1 if parity_enabled else 0)

    mv = la._play_bit_py(mask, col)
    nm = mask | mv
    next_pos = nm ^ (me | mv)  # opponent to move in negamax convention

    WARR = _build_WARR_from_la(la)

    killers = np.full((64, 2), -1, dtype=np.int8)
    history = np.zeros(la.COLS, dtype=np.int32)

    TT_SIZE = 1 << 16
    TT_pos = np.zeros(TT_SIZE, dtype=np.uint64)
    TT_mask = np.zeros(TT_SIZE, dtype=np.uint64)
    TT_depth = np.full(TT_SIZE, -1, dtype=np.int16)
    TT_flag = np.zeros(TT_SIZE, dtype=np.int8)
    TT_val = np.zeros(TT_SIZE, dtype=np.float64)
    TT_move = np.full(TT_SIZE, -1, dtype=np.int8)

    node_counter = np.zeros(1, dtype=np.int64)
    max_nodes = np.int64(9_000_000_000_000)

    v, _ = la._N["negamax"](
        np.uint64(next_pos),
        np.uint64(nm),
        np.int16(depth - 1),
        -float(la.MATE_SCORE),
        float(la.MATE_SCORE),
        np.int16(1),  # root_select_fixed starts child at ply=1
        root_pos_is_first,
        parity_enabled_i8,
        node_counter,
        max_nodes,
        float(la.MATE_SCORE),
        la._N["COL_MASK"],
        la._N["WIN_MASKS"],
        la._N["WIN_KIND"],
        la._N["WIN_B"],
        la._N["WIN_C"],
        la._N["WIN_R"],
        WARR,
        float(la.DEFENSIVE),
        float(la.FLOATING_NEAR),
        float(la.FLOATING_FAR),
        np.uint64(la._N["CENTER_MASK"]),
        np.uint64(la._N["ODD_MASK"]),
        np.uint64(la._N["EVEN_MASK"]),
        float(la.PARITY_BONUS),
        float(la.immediate_w),
        float(la.fork_w),
        float(la.CENTER_BONUS),
        float(la.VERT_MUL),
        float(la.VERT_3_READY_BONUS),
        float(la.TEMPO_W),
        la._N["CENTER_ORDER"],
        la._N["TOP_MASK"],
        la._N["BOTTOM_MASK"],
        np.int32(la.STRIDE),
        np.int32(la.ROWS),
        np.int32(la.COLS),
        killers,
        history,
        np.uint64(la._N["FULL_MASK"]),
        TT_pos,
        TT_mask,
        TT_depth,
        TT_flag,
        TT_val,
        TT_move,
    )
    return -float(v)

# -------------------- Nudge dominance warning --------------------

def top2(df, col, descending=True):
    d = df.sort_values(col, ascending=not descending).reset_index(drop=True)
    if len(d) == 0:
        return None, None
    if len(d) == 1:
        return d.iloc[0], None
    return d.iloc[0], d.iloc[1]