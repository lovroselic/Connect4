# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 13:43:45 2025

@author: Lovro
"""

# ppo_utilities.py

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, Optional, List
import pandas as pd
from IPython.display import HTML, display
import torch
import numpy as np
from C4.connect4_lookahead import Connect4Lookahead
from C4.connect4_env import Connect4Env
from PPO.ppo_training_phases_config import TRAINING_PHASES
from PPO.ppo_update import PPOUpdateCfg

Lookahead = Connect4Lookahead()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Connect4Env()


def encode_two_channel_agent_centric(state_np: np.ndarray, perspective: int) -> np.ndarray:
    """
    Accepts any of:
      (6,7) scalar board in {-1,0,+1}
      (2,6,7) [me,opp] planes (0/1)
      (4,6,7) [me,opp,row,col] from env.get_state()
    Returns (2,6,7) float32 planes agent-centric for the given perspective.
    """
    x = np.asarray(state_np)

    # --- reconstruct scalar board in {-1,0,+1} ---
    if x.shape == (6, 7):
        board = x.astype(np.int8)
    elif x.shape == (2, 6, 7):
        me, opp = x[0] > 0.5, x[1] > 0.5
        board = me.astype(np.int8) - opp.astype(np.int8)
    elif x.shape == (4, 6, 7):
        me, opp = x[0] > 0.5, x[1] > 0.5
        board = me.astype(np.int8) - opp.astype(np.int8)
    else:
        raise ValueError(f"Unsupported state shape {x.shape}; expected (6,7), (2,6,7) or (4,6,7).")

    # --- agent-centric re-labeling ---
    board = (board * int(perspective)).astype(np.int8)
    me_plane  = (board == +1).astype(np.float32)
    opp_plane = (board == -1).astype(np.float32)
    return np.stack([me_plane, opp_plane], axis=0)


def _cfg_fallback(cfg: PPOUpdateCfg, names: List[str], default):
    """Return the first present attribute from 'names' on cfg, else default."""
    for n in names:
        if hasattr(cfg, n):
            return getattr(cfg, n)
    return default


def params_for_phase(name: str, cfg: PPOUpdateCfg):
    """
    Expect TRAINING_PHASES[name].get("params", {})

    Returns a dict with:
      lr, clip, entropy, epochs, target_kl, vf_coef, vf_clip, batch_size,
      max_grad_norm, temperature, steps_per_update

    Fallbacks read common alternative cfg attribute names where possible.
    """
    p = TRAINING_PHASES.get(name, {}).get("params", {}) or {}
    return {
        "lr":            float(p.get("lr", _cfg_fallback(cfg, ["lr", "learning_rate"], 3e-4))),
        "clip":          float(p.get("clip", _cfg_fallback(cfg, ["clip", "clip_range", "clip_coef"], 0.20))),
        "entropy":       float(p.get("entropy", _cfg_fallback(cfg, ["entropy", "ent_coef", "entropy_coef"], 0.0))),
        "epochs":        int(p.get("epochs", _cfg_fallback(cfg, ["epochs", "update_epochs"], 4))),
        "target_kl":     float(p.get("target_kl", _cfg_fallback(cfg, ["target_kl"], 0.01))),
        "vf_coef":       float(p.get("vf_coef", _cfg_fallback(cfg, ["vf_coef", "value_coef"], 0.5))),
        "vf_clip":       float(p.get("vf_clip", _cfg_fallback(cfg, ["vf_clip", "vf_clip_range"], 0.2))),
        "batch_size":    int(p.get("batch_size", _cfg_fallback(cfg, ["batch_size", "mb_size", "minibatch_size"], 256))),
        "max_grad_norm": float(p.get("max_grad_norm", _cfg_fallback(cfg, ["max_grad_norm", "grad_clip"], 0.5))),
        "temperature":   float(p.get("temperature", 1.0)),
        "steps_per_update": int(p.get("steps_per_update", _cfg_fallback(cfg, ["steps_per_update", "rollout_steps"], 2048))),
    }


def legal_moves_from_board(board: np.ndarray) -> List[int]:
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
    # x is now (6,7) scalar board
    return [c for c in range(x.shape[1]) if x[0, c] == 0]



def select_opponent_action(
    state_board: np.ndarray,
    player: int,
    depth: Optional[int] = None,
) -> Optional[int]:
    """
    Choose opponent action:
      - If depth None/<=0 -> random legal
      - Else -> n-step lookahead using Connect4Lookahead
    Returns None if no legal move exists.

    Note: signature kept intentionally minimal for backward compatibility.
    """
    
    legal = legal_moves_from_board(state_board)
    if not legal:
        print(f"[Warning] select_opponent_action:: No legal moves found from state! (board top row = {state_board[0]})")
        raise ValueError(f"[Warning] select_opponent_action:: No legal moves found from state! (board top row = {state_board[0]})")

    if depth is None or (isinstance(depth, (int, np.integer)) and depth <= 0):
        return random.choice(legal)
    # Depth given -> scripted lookahead
    try:
        return Lookahead.n_step_lookahead(state_board, player=player, depth=int(depth))
    except Exception:
        # Failsafe to random if heuristic errors out
        return random.choice(legal)


def is_draw(board: np.ndarray) -> bool:
    """
    A draw if the board has no empty cells AND neither player has a 4-in-a-row.
    """
    if (board == 0).any():
        return False
    pats = Lookahead.board_to_patterns(board, [1, -1])
    return (Lookahead.count_windows(pats, 4, 1) == 0 and
            Lookahead.count_windows(pats, 4, -1) == 0)


@dataclass(frozen=True)
class PhaseInfo:
    name: str
    start_ep: int           # inclusive
    end_ep: Optional[int]   # inclusive; None => infinity
    lookahead: Optional[object]   # may be None, int, or "self"


class PhaseTracker:
    """
    Episode-based curriculum tracker using 'duration' per phase.
    Builds a timeline of [start_ep, end_ep] for each phase (Final can be open-ended).
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
            lookahead = cfg.get("lookahead", None)

            if dur is None:
                # open-ended final phase
                cfg["start_ep"] = cursor
                cfg["end_ep"] = None
                cfg["length"] = float("inf")
                timeline.append(PhaseInfo(name, start_ep=cursor, end_ep=None, lookahead=lookahead))
                break
            else:
                dur = int(dur)
                start_ep = cursor
                end_ep = cursor + dur - 1
                cursor = end_ep + 1
                cumulative += dur
                cfg["start_ep"] = start_ep
                cfg["end_ep"] = end_ep
                cfg["length"] = cumulative
                timeline.append(PhaseInfo(name, start_ep=start_ep, end_ep=end_ep, lookahead=lookahead))
        return timeline

    def get_phase(self, episode: int) -> PhaseInfo:
        for p in self.timeline:
            if p.end_ep is None:
                if episode >= p.start_ep:
                    return p
            elif p.start_ep <= episode <= p.end_ep:
                return p
        return self.timeline[-1]

    def start_episode(self, episode: int):
        p = self.get_phase(episode)
        changed = (p.name != self.current_name)
        self.current_name = p.name
        return p, changed


#----------------------------

def _lk_to_int(lk):
    if lk is None:
        return 0      # Random
    if isinstance(lk, str) and lk.lower() == "self":
        return -1     # Self-play
    return int(lk)


def display_phases_table(phases: dict):
    """
    Build timeline in original config order.
    Assumes every phase has an integer 'duration' (or 'length' as alias).
    Returns the final 'end_ep' (cumulative episodes).
    """
    rows = []
    cursor = 1  # episode counter

    for name, TF in phases.items():   # preserve insertion order
        # normalize duration
        dur = TF.get("duration")
        if dur is None:
            dur = TF.get("length")
        dur = int(dur)

        start_ep = cursor
        end_ep   = cursor + dur - 1
        cursor   = end_ep + 1

        p = TF.get("params", {}) or {}
        rows.append({
            "phase":       name,
            "start_ep":    start_ep,
            "end_ep":      end_ep,
            "duration":    dur,
            "lookahead":   _lk_to_int(TF.get("lookahead")),
            "length":      end_ep,   # cumulative == end_ep
            "lr":          ("" if "lr" not in p else float(p["lr"])),
            "clip":        ("" if "clip" not in p else float(p["clip"])),
            "entropy":     ("" if "entropy" not in p else float(p["entropy"])),
            "epochs":      ("" if "epochs" not in p else int(p["epochs"])),
            # Optional (shown if present):
            # "temperature": ("" if "temperature" not in p else float(p["temperature"])),
            # "steps/update":("" if "steps_per_update" not in p else int(p["steps_per_update"])),
        })

    df = pd.DataFrame(rows)
    display(HTML(df.to_html(index=False)))
    return end_ep, name


def _key_from_lookahead(lk) -> str:
    """
    Map lookahead spec to a short opponent key used by the trainer:
      None  -> "R"   (Random)
      "self"-> "SP"  (Self-play)
      int k -> "Lk"  (Lookahead-k)
    """
    if lk is None:
        return "R"
    if isinstance(lk, str) and lk.lower() == "self":
        return "SP"
    return f"L{int(lk)}"
