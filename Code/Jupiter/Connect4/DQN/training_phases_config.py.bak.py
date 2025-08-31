# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 14:44:26 2025

@author: lovro
"""

# [R, L1, L2, L3, L4, L5, L6, L7]

from IPython.display import display, HTML
import pandas as pd
import numpy as np

    
TRAINING_PHASES = {
    # ---------------- Warmup (short, just to populate buffer) ----------------
    "Random1": {
        "duration": 200,
        "weights": [1, 0, 0, 0, 0, 0, 0, 0],
        "epsilon": 0.22, "epsilon_min": 0.08,
        "memory_prune_low": 0.00,
        "opponent": [1, 0, 0, 0, 0, 0, 0, 0],
    },
    "Random2": {
        "duration": 200,
        "weights": [0.5, 0.5, 0, 0, 0, 0, 0, 0],
        "epsilon": 0.22, "epsilon_min": 0.08,
        "memory_prune_low": 0.00,
        "opponent": [1, 0, 0, 0, 0, 0, 0, 0],
    },

    # --------------- R + L1 mix (quick lift to tactical play) ----------------
    "Mixed_R1_A": {
        "duration": 350,
        "weights": [0.25, 0.70, 0.05, 0, 0, 0, 0, 0],
        "epsilon": 0.35, "epsilon_min": 0.10,
        "memory_prune_low": 0.03,
        "opponent": [0.6, 0.4, 0, 0, 0, 0, 0, 0],
    },
    "Mixed_R1_B": {
        "duration": 350,
        "weights": [0.40, 0.60, 0.00, 0, 0, 0, 0, 0],
        "epsilon": 0.35, "epsilon_min": 0.10,
        "memory_prune_low": 0.05,
        "opponent": [0.5, 0.5, 0, 0, 0, 0, 0, 0],
    },
    "SelfPlay_Mixed_R1": {
        "duration": 300,
        "weights": None,
        "epsilon": 0.02, "epsilon_min": 0.02,
        "memory_prune_low": 0.05,
        "opponent": None,
    },

    # ----------------- R/L1++ (fewer, stronger exposures) --------------------
    "Mixed_R11_A": {
        "duration": 400,
        "weights": [0.0, 0.55, 0.30, 0.15, 0, 0, 0, 0],
        "epsilon": 0.30, "epsilon_min": 0.06,
        "memory_prune_low": 0.07,
        "opponent": [0.25, 0.75, 0, 0, 0, 0, 0, 0],
    },
    "Mixed_R11_B": {
        "duration": 400,
        "weights": [0.05, 0.60, 0.25, 0.10, 0, 0, 0, 0],
        "epsilon": 0.28, "epsilon_min": 0.05,
        "memory_prune_low": 0.08,
        "opponent": [0.15, 0.85, 0, 0, 0, 0, 0, 0],
    },
    "SelfPlay_Mixed_R11": {
        "duration": 350,
        "weights": None,
        "epsilon": 0.02, "epsilon_min": 0.02,
        "memory_prune_low": 0.08,
        "opponent": None,
    },

    # ---------------------------- Pure L1 block ------------------------------
    "L1_Defense": {
        "duration": 450,
        "weights": [0.10, 0.70, 0.20, 0.00, 0, 0, 0, 0],
        "epsilon": 0.22, "epsilon_min": 0.05,
        "memory_prune_low": 0.08,
        "opponent": [0.35, 0.65, 0, 0, 0, 0, 0, 0],
    },
    "L1_Consolidate": {
        "duration": 450,
        "weights": [0.05, 0.65, 0.30, 0.00, 0, 0, 0, 0],
        "epsilon": 0.18, "epsilon_min": 0.04,
        "memory_prune_low": 0.10,
        "opponent": [0.20, 0.80, 0, 0, 0, 0, 0, 0],
    },
    "SelfPlay_L1": {
        "duration": 500,
        "weights": None,
        "epsilon": 0.02, "epsilon_min": 0.02,
        "memory_prune_low": 0.10,
        "opponent": None,
    },

    # ------------------------- Bridge to L2 (12 mix) -------------------------
    "Mixed12_Attack": {
        "duration": 400,
        "weights": [0.00, 0.10, 0.45, 0.45, 0, 0, 0, 0],
        "epsilon": 0.16, "epsilon_min": 0.04,
        "memory_prune_low": 0.10,
        "opponent": [0.10, 0.55, 0.35, 0, 0, 0, 0, 0],
    },
    "Mixed12_Defense": {
        "duration": 400,
        "weights": [0.00, 0.35, 0.65, 0.00, 0, 0, 0, 0],
        "epsilon": 0.14, "epsilon_min": 0.04,
        "memory_prune_low": 0.12,
        "opponent": [0.10, 0.30, 0.60, 0, 0, 0, 0, 0],
    },
    "Mixed12_Consolidate": {
        "duration": 400,
        "weights": [0.05, 0.50, 0.45, 0.00, 0, 0, 0, 0],
        "epsilon": 0.12, "epsilon_min": 0.03,
        "memory_prune_low": 0.12,
        "opponent": [0.10, 0.40, 0.50, 0, 0, 0, 0, 0],
    },
    "SelfPlay_Mixed12": {
        "duration": 350,
        "weights": None,
        "epsilon": 0.02, "epsilon_min": 0.02,
        "memory_prune_low": 0.12,
        "opponent": None,
    },

    # ----------------------------- L2 bridge ---------------------------------
    "Fixed_L2_Short": {
        "duration": 450,
        "weights": [0, 0, 0.55, 0.45, 0, 0, 0, 0],
        "epsilon": 0.10, "epsilon_min": 0.04,
        "memory_prune_low": 0.12,
        "opponent": [0, 0, 1, 0, 0, 0, 0, 0],
    },
    "Variable_123": {
        "duration": 500,
        "weights": [0.00, 0.25, 0.45, 0.30, 0, 0, 0, 0],
        "epsilon": 0.10, "epsilon_min": 0.03,
        "memory_prune_low": 0.12,
        "opponent": [0.00, 0.30, 0.45, 0.25, 0, 0, 0, 0],
    },

    # ----------------------------- L2 proper ---------------------------------
    "L2_Attack": {
        "duration": 450,
        "weights": [0, 0.40, 0.45, 0.15, 0, 0, 0, 0],
        "epsilon": 0.08, "epsilon_min": 0.03,
        "memory_prune_low": 0.14,
        "opponent": [0.10, 0.45, 0.45, 0, 0, 0, 0, 0],
    },
    "L2_Defense": {
        "duration": 450,
        "weights": [0, 0.20, 0.70, 0.10, 0, 0, 0, 0],
        "epsilon": 0.08, "epsilon_min": 0.03,
        "memory_prune_low": 0.15,
        "opponent": [0.05, 0.25, 0.70, 0, 0, 0, 0, 0],
    },
    "L2_Consolidate": {
        "duration": 700,
        "weights": [0, 0.25, 0.45, 0.30, 0, 0, 0, 0],
        "epsilon": 0.06, "epsilon_min": 0.02,
        "memory_prune_low": 0.15,
        "opponent": [0.10, 0.40, 0.50, 0, 0, 0, 0, 0],
    },

    # ---------------------------- Goal state ---------------------------------
    "SelfPlay_L2": {
        "duration": 900,               # anchor here
        "weights": None,
        "epsilon": 0.02, "epsilon_min": 0.02,
        "memory_prune_low": 0.15,
        "opponent": None,
    },
}





def set_training_phases_length(DICT):
    length = 0
    for name, TF in DICT.items():
        length += int(TF["duration"])
        TF["length"] = length
        TF["sumWeights"] = float(np.sum(TF.get("weights", []))) if TF.get("weights") is not None else 0.0
        opp = TF.get("opponent", [])
        TF["sumOppo"] = (float(np.sum(opp)) if isinstance(opp, (list, tuple, np.ndarray))
                         else (0.0 if opp is None else float(opp)))
    display_dict_as_tab(DICT)

def display_dict_as_tab(DICT):
    DF = pd.DataFrame.from_dict(DICT, orient="index")
    cols = [
        "length", "duration",
        "epsilon", "epsilon_min",
        "memory_prune_low",
        "sumWeights", "sumOppo"
    ]
    DF = DF[cols]
    DF = DF.sort_values(by=['length'], ascending=True)
    display(HTML(DF.to_html()))
