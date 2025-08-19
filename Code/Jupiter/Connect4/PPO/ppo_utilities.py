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


def params_for_phase(name: str, cfg: PPOUpdateCfg):
    """
    Expect TRAINING_PHASES[name].get("params", {})
    """
    p = TRAINING_PHASES.get(name, {}).get("params", {})
    return {
        "lr":         float(p.get("lr", cfg.clip_range)),   # Fallback: just use cfg default or a hardcoded LR
        "clip":       float(p.get("clip", cfg.clip_range)),
        "entropy":    float(p.get("entropy", cfg.ent_coef)),
        "epochs":     int(p.get("epochs", cfg.epochs)),
        "target_kl":  float(p.get("target_kl", cfg.target_kl)),
        "vf_coef":    float(p.get("vf_coef", cfg.vf_coef)),
        "vf_clip":    float(p.get("vf_clip", cfg.vf_clip_range)),
        "batch_size": int(p.get("batch_size", cfg.batch_size)),
        "max_grad_norm": float(p.get("max_grad_norm", cfg.max_grad_norm)),
    }


def legal_moves_from_board(board: np.ndarray) -> List[int]:
    """Return list of legal columns based on top row emptiness; no env dependency."""
    _, cols = board.shape
    return [c for c in range(cols) if board[0, c] == 0]

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
    """
    legal = legal_moves_from_board(state_board)
    if not legal:
        return None
    if depth is None or depth <= 0:
        return random.choice(legal)
    return Lookahead.n_step_lookahead(state_board, player=player, depth=depth)

def is_draw(board: np.ndarray) -> bool:
    """
    A draw if the board has no empty cells AND neither player has a 4-in-a-row.
    """
    if (board == 0).any():
        return False
    # quick pattern-based check
    pats = Lookahead.board_to_patterns(board, [1, -1])
    return Lookahead.count_windows(pats, 4, 1) == 0 and Lookahead.count_windows(pats, 4, -1) == 0


@dataclass(frozen=True)
class PhaseInfo:
    name: str
    start_ep: int           # inclusive
    end_ep: Optional[int]   # inclusive; None => infinity
    lookahead: Optional[int]

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
            "phase":     name,
            "start_ep":  start_ep,
            "end_ep":    end_ep,
            "duration":  dur,
            "lookahead": _lk_to_int(TF.get("lookahead")),
            "length":    end_ep,   # cumulative == end_ep
            "lr":        ("" if "lr" not in p else float(p["lr"])),
            "clip":      ("" if "clip" not in p else float(p["clip"])),
            "entropy":   ("" if "entropy" not in p else float(p["entropy"])),
            "epochs":    ("" if "epochs" not in p else int(p["epochs"])),
        })

    df = pd.DataFrame(rows)
    display(HTML(df.to_html(index=False)))
