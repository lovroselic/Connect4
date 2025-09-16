# A0/a0_config.py
from __future__ import annotations

# --- Self-play / MCTS ---
MCTS_SIMS        = 256     # sims per move during self-play
CPUCT            = 2.0
TAU_EARLY        = 1.0
TAU_LATE         = 0.0
TAU_SWITCH_PLY   = 10      # sharpen targets a bit earlier
AUGMENT          = True    # mirror + color-swap

# --- Replay & Optimization ---
REPLAY_CAPACITY  = 200_000
BATCH_SIZE       = 256
LR               = 2e-4
WEIGHT_DECAY     = 1e-4
GRAD_CLIP        = 1.0
USE_AMP          = None     # let trainer choose (uses 'cuda' amp if available)

# --- Loop sizing ---
GAMES_PER_CYCLE  = 32
UPDATES_PER_CYCLE= 128

# --- Evaluations ---
EVAL_GAMES_INTERIM = 10     # short, plotted during training
EVAL_GAMES_FINAL   = 20    # longer final benchmark (bar chart)
EVAL_MCTS_SIMS     = 256    # stronger planning at eval

# --- UI ---
USE_TQDM = True
