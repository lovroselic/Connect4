# -*- coding: utf-8 -*-
"""
PPO utilities — cleaned for the new on-policy setup
(Agent always uses its policy; 'teacher' appears only as opponent.)
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import pandas as pd
from IPython.display import HTML, display
import torch
import numpy as np

from C4.fast_connect4_lookahead import Connect4Lookahead
from C4.connect4_env import Connect4Env
from PPO.ppo_training_phases_config import TRAINING_PHASES
from PPO.ppo_update import PPOUpdateCfg

# Singletons (stateless enough for reuse)
Lookahead = Connect4Lookahead()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Connect4Env()

# Precompute coord planes in [-1, 1], same as ActorCritic.coord_row / coord_col
_ROW_PLANE = np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(1, 6, 1)
_ROW_PLANE = np.repeat(_ROW_PLANE, 7, axis=2)  # (1, 6, 7)

_COL_PLANE = np.linspace(-1.0, 1.0, 7, dtype=np.float32).reshape(1, 1, 7)
_COL_PLANE = np.repeat(_COL_PLANE, 6, axis=1)  # (1, 6, 7)

# ------------------------------
# Encoding helpers
# ------------------------------

import numpy as np

def encode_two_channel_agent_centric(state_np: np.ndarray, perspective: int) -> np.ndarray:
    """
    Agent-centric 2-plane encoding.

    Accepts any of:
      - (6,7) scalar board in {-1, 0, +1}
      - (2,6,7) planes [me, opp] (0/1)
      - (4,6,7) planes [me, opp, row, col] (row/col ignored)

    Returns
    -------
    (2,6,7) float32:
        [0] = agent (perspective) pieces == +1
        [1] = opponent pieces        == -1

    `perspective` is +1 or -1 in the scalar-board convention.
    """
    x = np.asarray(state_np)

    # --- reconstruct scalar board in {-1,0,+1} ---
    if x.shape == (6, 7):
        # scalar board already
        board = x.astype(np.int8)

    elif x.shape == (2, 6, 7):
        # planes [me, opp]
        me, opp = x[0] > 0.5, x[1] > 0.5
        board = me.astype(np.int8) - opp.astype(np.int8)

    elif x.shape == (4, 6, 7):
        # planes [me, opp, row, col] – ignore coord planes
        me, opp = x[0] > 0.5, x[1] > 0.5
        board = me.astype(np.int8) - opp.astype(np.int8)

    else:
        raise ValueError(
            f"Unsupported state shape {x.shape}; "
            f"expected (6,7), (2,6,7) or (4,6,7)."
        )

    # --- agent-centric relabelling ---
    # perspective == +1: "+1" on board is agent
    # perspective == -1: "-1" on board is agent (flip signs)
    p = int(np.sign(perspective) or 1)   # just in case someone passes 0
    board = (board * p).astype(np.int8)

    me_plane  = (board == +1).astype(np.float32)
    opp_plane = (board == -1).astype(np.float32)

    out = np.stack([me_plane, opp_plane], axis=0)  # (2,6,7)
    return np.ascontiguousarray(out)


def encode_four_channel_agent_centric(state_np: np.ndarray, perspective: int) -> np.ndarray:
    """
    Agent-centric 4-plane encoding for PPO:

      (4,6,7) = [agent_plane, opp_plane, row, col]

    Accepts the same inputs as encode_two_channel_agent_centric:
      - (6,7) scalar board in {-1,0,+1}
      - (2,6,7) [me, opp]
      - (4,6,7) [me, opp, row, col]  (row/col of input are ignored here)

    Coord planes row/col are always generated fresh in [-1,1], matching
    ActorCritic.coord_row / coord_col.
    """
    two = encode_two_channel_agent_centric(state_np, perspective)  # (2,6,7)
    row = _ROW_PLANE.astype(np.float32, copy=False)
    col = _COL_PLANE.astype(np.float32, copy=False)
    out = np.concatenate([two, row, col], axis=0)  # (4,6,7)
    return np.ascontiguousarray(out)

# ------------------------------
# Phase params / display
# ------------------------------

def _cfg_fallback(cfg: PPOUpdateCfg, names: List[str], default):
    """Return the first present attribute from 'names' on cfg, else default."""
    for n in names:
        if hasattr(cfg, n):
            return getattr(cfg, n)
    return default

def params_for_phase(phase_name: str, cfg) -> dict:
    phase = TRAINING_PHASES[phase_name]
    p_raw = phase.get("params", {})

    p = {
        "lr":                   p_raw.get("lr",        3e-4),
        "clip":                 p_raw.get("clip",      0.20),
        "entropy":              p_raw.get("entropy",   0.0),
        "epochs":               p_raw.get("epochs",    4),
        "batch_size":           p_raw.get("batch_size",      256),
        "steps_per_update":     p_raw.get("steps_per_update", 512),
        "vf_clip":              p_raw.get("vf_clip",   0.2),
        "max_grad_norm":        p_raw.get("max_grad_norm", 0.5),
        "target_kl":            p_raw.get("target_kl", 0.5),
        "temperature":          p_raw.get("temperature", 1.0),
        "vf_coef":              p_raw.get("vf_coef",   0.01),
        "distill_coef":         p_raw.get("distill_coef",   0.00),
        "mentor_depth":         p_raw.get("mentor_depth",   1),     
        "mentor_prob":          p_raw.get("mentor_prob",   0.1),   
        "mentor_coef":          p_raw.get("mentor_coef",   0.1),   
    }

    # --- pass through heuristic knobs untouched ---
    for k in ("center_start", "guard_prob", "win_now_prob",
              "guard_ply_min", "guard_ply_max"):
        if k in p_raw:
            p[k] = p_raw[k]

    return p



@dataclass(frozen=True)
class PhaseInfo:
    name: str
    start_ep: int           # inclusive
    end_ep: Optional[int]   # inclusive; None => infinity


class PhaseTracker:
    """
    Episode-based curriculum tracker using 'duration' per phase.
    Builds a timeline of [start_ep, end_ep] for each phase. Final can be open-ended.
    Also annotates the source dict with 'start_ep', 'end_ep', and cumulative 'length' (=end_ep).
    """
    def __init__(self, phases_cfg: Dict[str, Dict]):
        self.phases_cfg = phases_cfg
        self.timeline: List[PhaseInfo] = self._build_timeline(phases_cfg)
        self.current_name: Optional[str] = None

    @staticmethod
    def _build_timeline(phases_cfg: Dict[str, Dict]) -> List[PhaseInfo]:
        timeline: List[PhaseInfo] = []
        cursor = 1
        cumulative = 0
        for name, cfg in phases_cfg.items():
            dur = cfg.get("duration", None)
            if dur is None:
                # open-ended final phase
                cfg["start_ep"] = cursor
                cfg["end_ep"] = None
                cfg["length"] = float("inf")
                timeline.append(PhaseInfo(name, start_ep=cursor, end_ep=None))
                break
            dur = int(dur)
            start_ep = cursor
            end_ep = cursor + dur - 1
            cursor = end_ep + 1
            cumulative += dur
            cfg["start_ep"] = start_ep
            cfg["end_ep"] = end_ep
            cfg["length"] = cumulative
            timeline.append(PhaseInfo(name, start_ep=start_ep, end_ep=end_ep))
        return timeline

    def get_phase(self, episode: int) -> PhaseInfo:
        for p in self.timeline:
            if p.end_ep is None:
                if episode >= p.start_ep:
                    return p
            elif p.start_ep <= episode <= p.end_ep:
                return p
        return self.timeline[-1]

    def start_episode(self, episode: int) -> Tuple[PhaseInfo, bool]:
        p = self.get_phase(episode)
        changed = (p.name != self.current_name)
        self.current_name = p.name
        return p, changed


def display_phases_table(phases: dict) -> Tuple[int, str]:
    """
    Build timeline in original config order.
    Returns (final_end_ep, last_phase_name).
    """
    rows = []
    cursor = 1  # episode counter
    last_phase = None

    for name, TF in phases.items():   # preserve insertion order
        dur = TF.get("duration")
        if dur is None:
            dur = TF.get("length")
        dur = int(dur)

        start_ep = cursor
        end_ep   = cursor + dur - 1
        cursor   = end_ep + 1
        last_phase = name

        p = TF.get("params", {}) or {}
        rows.append({
            "phase":       name,
            "start_ep":    start_ep,
            "end_ep":      end_ep,
            "duration":    dur,
            "length":      end_ep,   # cumulative == end_ep
            "lr":          ("" if "lr" not in p else float(p["lr"])),
            "clip":        ("" if "clip" not in p else float(p["clip"])),
            "entropy":     ("" if "entropy" not in p else float(p["entropy"])),
            "epochs":      ("" if "epochs" not in p else int(p["epochs"])),
            # uncomment if you want them in the table:
            # "temperature": ("" if "temperature" not in p else float(p["temperature"])),
            # "steps/update":("" if "steps_per_update" not in p else int(p["steps_per_update"])),
        })

    df = pd.DataFrame(rows)
    display(HTML(df.to_html(index=False)))
    return end_ep, last_phase


# ------------------------------
# Board helpers / scripted teacher
# ------------------------------

def legal_moves_from_board(board: np.ndarray) -> List[int]:
    """
    Compute legal columns from either a scalar board (6,7) or planes (2,6,7)/(4,6,7).
    Convention: a column is legal if the TOP cell (row 0) is empty (==0).
    """
    x = np.asarray(board)
    if x.ndim == 3:
        if x.shape[0] == 2:   # [me, opp]
            me, opp = x[0] > 0.5, x[1] > 0.5
            x = me.astype(np.int8) - opp.astype(np.int8)
        elif x.shape[0] == 4: # [me, opp, row, col]
            me, opp = x[0] > 0.5, x[1] > 0.5
            x = me.astype(np.int8) - opp.astype(np.int8)
        else:
            raise ValueError(f"Unexpected encoded board shape {x.shape}")
    if x.shape != (6, 7):
        raise ValueError(f"legal_moves_from_board expects (6,7), got {x.shape}")
    return [c for c in range(7) if x[0, c] == 0]


def select_opponent_action(state_board: np.ndarray, player: int, depth: Optional[int] = None) -> Optional[int]:
    """
    Choose opponent action:
      - depth is None or <=0 -> random legal
      - else -> n-step lookahead via Connect4Lookahead
    Returns None if no legal move exists (caller should guard).
    """
    legal = legal_moves_from_board(state_board)
    if not legal: raise ValueError(f"[select_opponent_action] No legal moves from state (top row = {state_board[0]})")

    if depth is None or (isinstance(depth, (int, np.integer)) and depth <= 0): return random.choice(legal)

    try:
        a = Lookahead.n_step_lookahead(state_board, player=player, depth=int(depth))
        return a if a in legal else random.choice(legal)
    except Exception: return random.choice(legal)


def is_draw(board: np.ndarray) -> bool:
    """
    A draw if the board has no empty cells AND neither player has a 4-in-a-row.
    """
    if (board == 0).any():
        return False
    pats = Lookahead.board_to_patterns(board, [1, -1])
    return (Lookahead.count_windows(pats, 4, 1) == 0 and
            Lookahead.count_windows(pats, 4, -1) == 0)

