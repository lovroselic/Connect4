# make_c4_la_depth_notebook.py
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(r"""
# C4 Lookahead Depth Comparator (Fast LA vs Kaggle bbLA)

This notebook compares, from a **fixed board position**:
- **Fast Numba Lookahead** (`Connect4Lookahead`) for depths `d_min..d_max`
- **Kaggle bbLA** (`N_step_lookahead_bitboard`) action

It reports:
- chosen **action** (column)
- **outcome/reward** under a few continuation policies
- optional **win rate vs random** (Monte Carlo)

Notes:
- Board is expected as a `6x7` array (top row first) with values in `{0,1,2}` (or `{0,1,-1}` which will be canonicalized to `{0,1,2}`).
- `to_move` is `1` or `2`.
"""))

cells.append(nbf.v4.new_code_cell(r"""
import os, sys, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Optional widgets
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    HAVE_WIDGETS = True
except Exception:
    HAVE_WIDGETS = False

# Make project imports easier
PROJECT_ROOT = os.path.abspath(".")
CANDIDATE_PATHS = [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "C4"),
    os.path.join(PROJECT_ROOT, "Kaggle"),
    "/mnt/data",  # you uploaded helpers here
]
for p in CANDIDATE_PATHS:
    if p and os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("Added paths:", [p for p in CANDIDATE_PATHS if p and os.path.exists(p)])
"""))

cells.append(nbf.v4.new_code_cell(r"""
# --- Try to import your helper utilities (optional, but nice) ---
try:
    import game_analysis_helpers as gah
    print("Loaded game_analysis_helpers from:", gah.__file__)
except Exception as e:
    gah = None
    print("game_analysis_helpers not available:", repr(e))
"""))

cells.append(nbf.v4.new_code_cell(r"""
# --- Import Fast LA and Kaggle bbLA ---
def import_fast_la():
    # Common layouts
    try:
        from C4.fast_connect4_lookahead import Connect4Lookahead
        return Connect4Lookahead
    except Exception:
        from fast_connect4_lookahead import Connect4Lookahead
        return Connect4Lookahead

def import_kaggle_bbla():
    # Common layouts
    try:
        from N_step_lookahead_bitboard import N_step_lookahead_bitboard
        return N_step_lookahead_bitboard
    except Exception:
        try:
            from Kaggle.N_step_lookahead_bitboard import N_step_lookahead_bitboard
            return N_step_lookahead_bitboard
        except Exception as e:
            raise ImportError(
                "Could not import N_step_lookahead_bitboard. "
                "Expected either N_step_lookahead_bitboard.py in sys.path, "
                "or Kaggle/N_stepS module layout."
            ) from e

Connect4Lookahead = import_fast_la()
N_step_lookahead_bitboard = import_kaggle_bbla()

print("Connect4Lookahead:", Connect4Lookahead)
print("N_step_lookahead_bitboard:", N_step_lookahead_bitboard)
"""))

cells.append(nbf.v4.new_code_cell(r"""
# =========================
# Board utilities
# =========================

ROWS, COLS = 6, 7

def canonicalize_board(board):
    b = np.asarray(board, dtype=np.int8).copy()
    assert b.shape == (ROWS, COLS), f"Expected (6,7), got {b.shape}"
    # Accept -1 as player2
    b[b == -1] = 2
    return b

def legal_moves(board):
    b = canonicalize_board(board)
    return [c for c in range(COLS) if b[0, c] == 0]

def apply_move(board, col, mark):
    b = canonicalize_board(board)
    if b[0, col] != 0:
        raise ValueError(f"Column {col} is full")
    for r in range(ROWS - 1, -1, -1):
        if b[r, col] == 0:
            b[r, col] = int(mark)
            return b
    raise RuntimeError("apply_move failed unexpectedly")

def check_winner(board):
    b = canonicalize_board(board)

    def four_in_a_row(mark):
        # horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if (b[r, c] == mark and b[r, c+1] == mark and b[r, c+2] == mark and b[r, c+3] == mark):
                    return True
        # vertical
        for c in range(COLS):
            for r in range(ROWS - 3):
                if (b[r, c] == mark and b[r+1, c] == mark and b[r+2, c] == mark and b[r+3, c] == mark):
                    return True
        # diag down-right
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if (b[r, c] == mark and b[r+1, c+1] == mark and b[r+2, c+2] == mark and b[r+3, c+3] == mark):
                    return True
        # diag down-left
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                if (b[r, c] == mark and b[r+1, c-1] == mark and b[r+2, c-2] == mark and b[r+3, c-3] == mark):
                    return True
        return False

    if four_in_a_row(1):
        return 1
    if four_in_a_row(2):
        return 2
    if np.all(b != 0):
        return 3  # draw
    return 0  # ongoing

def reward_from_winner(winner, perspective_mark):
    if winner == 3:
        return 0
    if winner == 0:
        return 0
    return 1 if winner == perspective_mark else -1

def board_to_kaggle_flat(board):
    # Kaggle ConnectX expects top row first, row-major list length 42
    b = canonicalize_board(board)
    return b.reshape(-1).tolist()

def kaggle_action(board, mark, seed=0):
    # Kaggle agent has some random choices in opening book, seed it for repeatability
    import random
    random.seed(int(seed))

    obs = {"board": board_to_kaggle_flat(board), "mark": int(mark)}
    config = {"rows": 6, "columns": 7, "inarow": 4}
    return int(N_step_lookahead_bitboard(obs, config))

def plot_board_matplotlib(board, title=None):
    b = canonicalize_board(board)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(title or "")
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.5, ROWS - 0.5)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.invert_yaxis()
    ax.grid(True)

    # Draw discs as markers, default matplotlib colors are fine
    for r in range(ROWS):
        for c in range(COLS):
            v = b[r, c]
            if v == 0:
                continue
            ax.plot(c, r, "o", markersize=22)

            # label the disc with 1/2 so color is not the only signal
            ax.text(c, r, str(int(v)), ha="center", va="center", fontsize=10)

    plt.show()

def show_board(board, title=None):
    # Use your helper if it exists, else fallback
    if gah is not None:
        for fn_name in ["show_board", "plot_board", "draw_board"]:
            if hasattr(gah, fn_name):
                try:
                    getattr(gah, fn_name)(board, title=title)
                    return
                except TypeError:
                    getattr(gah, fn_name)(board)
                    return
                except Exception:
                    pass
    plot_board_matplotlib(board, title=title)
"""))

cells.append(nbf.v4.new_code_cell(r"""
# =========================
# Fast LA knobs (edit these)
# =========================

# Base eval weights (w2)
LA_WEIGHTS = {2: 1.0, 3: 1000.0, 4: 100000.0}

# Heuristic knobs, mirror your fast_connect4_lookahead defaults
KNOBS = dict(
    immediate_w=100000.0,
    fork_w=100000.0,

    DEFENSIVE=1.5,
    FLOATING_NEAR=0.25,
    FLOATING_FAR=0.075,
    CENTER_BONUS=7.0,
    PARITY_BONUS=1.0,

    VERT_MUL=0.666,
    VERT_3_READY_BONUS=0.0,
    TEMPO_W=2.0,

    PARITY_MOVE_W=0.3,
    PARITY_UNLOCK_W=0.2,
    THREATSPACE_W=1.0,

    # opening book behavior inside class (if your implementation uses this)
    OPENING_RANDOM=False,
)

def make_fast_la(weights=None, knobs=None):
    w = LA_WEIGHTS if weights is None else dict(weights)
    k = KNOBS if knobs is None else dict(knobs)

    la = Connect4Lookahead(weights=w)

    # Apply knobs (fields exist on the class)
    for key, val in k.items():
        if hasattr(la, key):
            setattr(la, key, val)
        elif hasattr(Connect4Lookahead, key):
            setattr(Connect4Lookahead, key, val)
        else:
            # Keep it loud so you notice typos
            print("Warning, knob not found:", key)

    return la

FAST_LA = make_fast_la()
print("FAST_LA ready. Weights:", FAST_LA.weights)
for k in sorted(KNOBS.keys()):
    if hasattr(FAST_LA, k):
        print(f"{k} = {getattr(FAST_LA, k)}")
"""))

cells.append(nbf.v4.new_code_cell(r"""
# =========================
# Paste / set the board here
# =========================

# Replace this with your imported board_before from the analyzer notebook.
# Must be 6x7, top row first.
board_before = np.zeros((6, 7), dtype=np.int8)

# Player to move: 1 or 2
to_move = 1

show_board(board_before, title=f"board_before, to_move={to_move}")
print("Legal:", legal_moves(board_before))
"""))

cells.append(nbf.v4.new_code_cell(r"""
# =========================
# Policies + game rollout
# =========================

def policy_fast_la(depth):
    depth = int(depth)
    def _pol(board, mark):
        return int(FAST_LA.n_step_lookahead(canonicalize_board(board), int(mark), depth=depth))
    return _pol

def policy_random(seed=0):
    rng = np.random.default_rng(int(seed))
    def _pol(board, mark):
        lm = legal_moves(board)
        return int(rng.choice(lm)) if lm else 0
    return _pol

def play_game_from(board, to_move, p1_policy, p2_policy, max_plies=128):
    b = canonicalize_board(board)
    mark = int(to_move)
    plies = 0
    first_move = None

    while True:
        w = check_winner(b)
        if w != 0:
            return dict(
                winner=w,
                reward_p1=reward_from_winner(w, 1),
                reward_p2=reward_from_winner(w, 2),
                plies=plies,
                first_move=first_move,
                final_board=b,
            )

        if plies >= max_plies:
            # Should not happen in normal connect4, but just in case
            return dict(
                winner=3,
                reward_p1=0,
                reward_p2=0,
                plies=plies,
                first_move=first_move,
                final_board=b,
            )

        pol = p1_policy if mark == 1 else p2_policy
        col = int(pol(b, mark))

        lm = legal_moves(b)
        if col not in lm:
            # fail-safe, keep going
            col = int(lm[0]) if lm else 0

        if plies == 0:
            first_move = col

        b = apply_move(b, col, mark)
        mark = 2 if mark == 1 else 1
        plies += 1
"""))

cells.append(nbf.v4.new_code_cell(r"""
# =========================
# Depth sweep (Fast LA) + Kaggle action
# =========================

d_min, d_max = 1, 9
DEPTHS = list(range(d_min, d_max + 1))

# Continuation comparisons:
# 1) depth d vs depth d (self-play)
# 2) depth d vs reference depth (default: 9)
REF_DEPTH = 9

# Optional: Monte Carlo vs random opponent
DO_MC_VS_RANDOM = True
MC_GAMES = 200
MC_SEED0 = 123

results = []

for d in DEPTHS:
    # self-play
    g_self = play_game_from(board_before, to_move, policy_fast_la(d), policy_fast_la(d))
    a_self = g_self["first_move"]
    rew_self = g_self["reward_p1"] if to_move == 1 else g_self["reward_p2"]

    # vs reference depth
    if to_move == 1:
        g_ref = play_game_from(board_before, to_move, policy_fast_la(d), policy_fast_la(REF_DEPTH))
        rew_ref = g_ref["reward_p1"]
    else:
        g_ref = play_game_from(board_before, to_move, policy_fast_la(REF_DEPTH), policy_fast_la(d))
        rew_ref = g_ref["reward_p2"]
    a_ref = g_ref["first_move"]

    # optional MC vs random (randomness only on the random side)
    mc_win = mc_loss = mc_draw = None
    if DO_MC_VS_RANDOM:
        win = loss = draw = 0
        for i in tqdm(range(MC_GAMES), desc=f"MC vs random (d={d})", leave=False):
            seed = MC_SEED0 + i
            rnd = policy_random(seed=seed)

            if to_move == 1:
                g = play_game_from(board_before, to_move, policy_fast_la(d), rnd)
                r = g["reward_p1"]
            else:
                g = play_game_from(board_before, to_move, rnd, policy_fast_la(d))
                r = g["reward_p2"]

            if r > 0: win += 1
            elif r < 0: loss += 1
            else: draw += 1

        mc_win = win / MC_GAMES
        mc_loss = loss / MC_GAMES
        mc_draw = draw / MC_GAMES

    results.append(dict(
        depth=d,
        action_self=a_self,
        reward_self=rew_self,
        action_vs_ref=a_ref,
        reward_vs_ref=rew_ref,
        mc_win=mc_win,
        mc_loss=mc_loss,
        mc_draw=mc_draw,
    ))

df = pd.DataFrame(results)

# Kaggle action on the same board
k_action = kaggle_action(board_before, to_move, seed=0)

df, k_action
"""))

cells.append(nbf.v4.new_code_cell(r"""
# =========================
# Display, plots, and board previews
# =========================

display(df)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(df["depth"], df["reward_self"], marker="o", label="reward: self-play (d vs d)")
ax.plot(df["depth"], df["reward_vs_ref"], marker="o", label=f"reward: vs ref (ref={REF_DEPTH})")
ax.set_xlabel("depth")
ax.set_ylabel("reward from to_move POV")
ax.set_xticks(df["depth"].tolist())
ax.legend()
plt.show()

if DO_MC_VS_RANDOM and df["mc_win"].notna().any():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["depth"], df["mc_win"], marker="o", label="win rate vs random")
    ax.plot(df["depth"], df["mc_loss"], marker="o", label="loss rate vs random")
    ax.plot(df["depth"], df["mc_draw"], marker="o", label="draw rate vs random")
    ax.set_xlabel("depth")
    ax.set_ylabel("rate")
    ax.set_xticks(df["depth"].tolist())
    ax.legend()
    plt.show()

best_d = int(df.sort_values(["reward_vs_ref", "reward_self", "depth"], ascending=[False, False, True]).iloc[0]["depth"])
best_a = int(df[df["depth"] == best_d]["action_vs_ref"].iloc[0])

print("Kaggle action:", k_action)
print("Best depth by reward_vs_ref then reward_self:", best_d, "action:", best_a)

# Visualize the move consequences (one-step)
show_board(board_before, title=f"Before, to_move={to_move}")
show_board(apply_move(board_before, best_a, to_move), title=f"After FastLA depth={best_d} move={best_a}")
show_board(apply_move(board_before, k_action, to_move), title=f"After Kaggle move={k_action}")
"""))

cells.append(nbf.v4.new_code_cell(r"""
# =========================
# Per-column action scores (Fast LA)
# =========================

depth_for_scores = 7  # change quickly
scores = FAST_LA.n_step_action_scores(board_before, to_move, depth=depth_for_scores)

lm = legal_moves(board_before)
print("Legal moves:", lm)

# Replace -inf with nan for plotting readability
sc = scores.copy()
sc[np.isneginf(sc)] = np.nan

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(np.arange(COLS), sc)
ax.set_xticks(range(COLS))
ax.set_xlabel("column")
ax.set_ylabel(f"score (depth={depth_for_scores})")
plt.show()

best = int(np.nanargmax(sc)) if np.any(np.isfinite(sc)) else (lm[0] if lm else 0)
print("Depth", depth_for_scores, "argmax action:", best, "score:", scores[best])
"""))

cells.append(nbf.v4.new_code_cell(r"""
# =========================
# (Optional) Small interactive panel
# =========================

if not HAVE_WIDGETS:
    print("ipywidgets not available.")
else:
    d_slider = widgets.IntSlider(value=7, min=1, max=12, step=1, description="depth")
    seed_box = widgets.IntText(value=0, description="k_seed")
    out = widgets.Output()

    def _run(depth, k_seed):
        with out:
            clear_output(wait=True)
            depth = int(depth)
            k_seed = int(k_seed)

            a_fast = int(FAST_LA.n_step_lookahead(board_before, to_move, depth=depth))
            a_k = kaggle_action(board_before, to_move, seed=k_seed)

            show_board(board_before, title=f"Before, to_move={to_move}")
            show_board(apply_move(board_before, a_fast, to_move), title=f"FastLA depth={depth}, move={a_fast}")
            show_board(apply_move(board_before, a_k, to_move), title=f"Kaggle (seed={k_seed}), move={a_k}")

            # quick continuation vs ref
            if to_move == 1:
                g_ref = play_game_from(board_before, to_move, policy_fast_la(depth), policy_fast_la(REF_DEPTH))
                rew = g_ref["reward_p1"]
            else:
                g_ref = play_game_from(board_before, to_move, policy_fast_la(REF_DEPTH), policy_fast_la(depth))
                rew = g_ref["reward_p2"]

            print("FastLA action:", a_fast)
            print("Kaggle action:", a_k)
            print(f"Reward vs ref={REF_DEPTH} (FastLA depth={depth}):", rew)

            sc = FAST_LA.n_step_action_scores(board_before, to_move, depth=depth)
            sc2 = sc.copy()
            sc2[np.isneginf(sc2)] = np.nan

            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(np.arange(COLS), sc2)
            ax.set_xticks(range(COLS))
            ax.set_xlabel("column")
            ax.set_ylabel("score")
            plt.show()

    ui = widgets.HBox([d_slider, seed_box])
    display(ui, out)

    def _on_change(change):
        _run(d_slider.value, seed_box.value)

    d_slider.observe(_on_change, names="value")
    seed_box.observe(_on_change, names="value")

    _run(d_slider.value, seed_box.value)
"""))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
}

out_path = "C4_LA_Depth_Comparator.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Wrote:", out_path)
