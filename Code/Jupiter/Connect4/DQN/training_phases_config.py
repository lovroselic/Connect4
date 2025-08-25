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
    
    ######################### Random Section ############################################

    "Random1": {
        "duration": 300,
        "weights": [0.50, 0.45, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 1.0, "epsilon_min": 0.10,
        "memory_prune_low": 0.05,
        "opponent": [1, 0, 0, 0, 0, 0, 0, 0],
    },
    "Random2": {
        "duration": 200,
        "weights": [0.30, 0.60, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 0.90, "epsilon_min": 0.10,
        "memory_prune_low": 0.15,
        "opponent": [1, 0, 0, 0, 0, 0, 0, 0],
    },
    
    ########################## Mixed R/L1 #########################################
    
    "Mixed_R1_Attack": {
         "duration": 300,
         "weights": [0.15, 0.80, 0.05, 0, 0, 0.00, 0.00, 0.00], 
         "epsilon": 0.75, "epsilon_min": 0.10,
         "memory_prune_low": 0.15,
         "opponent": [0.70, 0.30, 0, 0, 0, 0, 0, 0],
     },
    
    "Mixed_R1_Defense": {
         "duration": 300,
         "weights": [0.40, 0.55, 0.05, 0, 0, 0.00, 0.00, 0.00], 
         "epsilon": 0.70, "epsilon_min": 0.10,
         "memory_prune_low": 0.15,
         "opponent": [0.50, 0.50, 0, 0, 0, 0, 0, 0],
     },
    
    "Mixed_R1_Equal": {
         "duration": 300,
         "weights": [0.40, 0.60, 0.00, 0, 0, 0.00, 0.00, 0.00], 
         "epsilon": 0.65, "epsilon_min": 0.10,
         "memory_prune_low": 0.15,
         "opponent": [0.50, 0.50, 0, 0, 0, 0, 0, 0],
     },
    
    "Mixed_R1_Conclude": {
         "duration": 300,
         "weights": [0.15, 0.80, 0.05, 0, 0, 0.00, 0.00, 0.00], 
         "epsilon": 0.60, "epsilon_min": 0.10,
         "memory_prune_low": 0.15,
         "opponent": [0.50, 0.50, 0, 0, 0, 0, 0, 0],
     },
    
    "SelfPlay_Mixed_R1": {
         "duration": 300,
         "weights": None,
         "epsilon": 0.02, "epsilon_min": 0.02,
         "memory_prune_low": 0.15,
         "opponent": None,
     },
    

   ########################## Mixed R/L1++  #########################################  
   
   "Mixed_R11_Attack": {
        "duration": 300,
        "weights": [0.0, 0.50, 0.30, 0.20, 0, 0.00, 0.00, 0.00], 
        "epsilon": 0.70, "epsilon_min": 0.10,
        "memory_prune_low": 0.15,
        "opponent": [0.30, 0.70, 0, 0, 0, 0, 0, 0],
    },
   
   "Mixed_R11_Defense": {
        "duration": 300,
        "weights": [0.10, 0.90, 0.00, 0, 0, 0.00, 0.00, 0.00], 
        "epsilon": 0.65, "epsilon_min": 0.10,
        "memory_prune_low": 0.15,
        "opponent": [0.10, 0.90, 0, 0, 0, 0, 0, 0],
    },
   
   "Mixed_R11_Equal": {
        "duration": 300,
        "weights": [0.00, 0.60, 0.30, 0.10, 0, 0.00, 0.00, 0.00], 
        "epsilon": 0.60, "epsilon_min": 0.10,
        "memory_prune_low": 0.15,
        "opponent": [0.10, 0.90, 0, 0, 0, 0, 0, 0],
    },
   
   "Mixed_R11_Conclude": {
        "duration": 300,
        "weights": [0.0, 0.65, 0.15, 0.20, 0, 0.00, 0.00, 0.00], 
        "epsilon": 0.50, "epsilon_min": 0.10,
        "memory_prune_low": 0.15,
        "opponent": [0.10, 0.90, 0, 0, 0, 0, 0, 0],
    },
   
   "SelfPlay_Mixed_R11": {
        "duration": 300,
        "weights": None,
        "epsilon": 0.02, "epsilon_min": 0.02,
        "memory_prune_low": 0.15,
        "opponent": None,
    },

    # === L1 Quartet ===========================================================
    
    "L1_Attack": {
        "duration": 400,
        "weights": [0.0, 0.50, 0.20, 0.30, 0, 0.00, 0.00, 0.00],
        "epsilon": 0.70, "epsilon_min": 0.10,
        "memory_prune_low": 0.12,
        "opponent": [0.60, 0.40, 0, 0, 0, 0, 0, 0],
    },
    
    "L1_Defense": {
        "duration": 400,
        "weights": [0.40, 0.55, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 0.65, "epsilon_min": 0.10,
        "memory_prune_low": 0.12,
        "opponent": [0.30, 0.60, 0.10, 0, 0, 0, 0, 0],
    },
    
    "L1_Consolidate": {
        "duration": 400,
        "weights": [0.10, 0.70, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 0.50, "epsilon_min": 0.08,
        "memory_prune_low": 0.10,
        "opponent": [0.30, 0.70, 0, 0, 0, 0, 0, 0],
    },
    
    "SelfPlay_L1": {
        "duration": 400,
        "weights": None,
        "epsilon": 0.02, "epsilon_min": 0.02,
        "memory_prune_low": 0.10,
        "opponent": None,
    },
    

    ########################## Mixed R/L1/L2 #########################################
    
    "Mixed12_Attack": {
        "duration": 400,
        "weights": [0.0, 0.10, 0.45, 0.45, 0, 0.00, 0.00, 0.00],
        "epsilon": 0.12, "epsilon_min": 0.05,
        "memory_prune_low": 0.18,
        "opponent": [0.20, 0.50, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00],
    },
    
    "L1_Win_Focus_A": {
         "duration": 500,
         "weights": [0.00, 0.08, 0.22, 0.70, 0,0,0,0],
         "epsilon": 0.40,
         "epsilon_min": 0.10,
         "memory_prune_low": 0.15,
         "opponent": [0.05, 0.92, 0.03, 0,0,0,0,0],
     },
    
    "Mixed12_Defense": {
        "duration": 400,
        "weights": [0.00, 0.40, 0.60, 0.00, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 0.10, "epsilon_min": 0.05,
        "memory_prune_low": 0.18,
        "opponent": [0.10, 0.30, 0.60, 0.00, 0.00, 0.00, 0.00, 0.00],
    },
    
    "L1_Win_Focus_B": {
         "duration": 500,
         "weights": [0.00, 0.10, 0.25, 0.65, 0,0,0,0],
         "epsilon": 0.65, "epsilon_min": 0.15,
         "memory_prune_low": 0.15,
         "opponent": [0.05, 0.90, 0.05, 0,0,0,0,0],
     },
    
    "Mixed12_Consolidate": {
        "duration": 400,
        "weights": [0.05, 0.55, 0.40, 0.00, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 0.08, "epsilon_min": 0.05,
        "memory_prune_low": 0.18,
        "opponent": [0.15, 0.45, 0.40, 0.00, 0.00, 0.00, 0.00, 0.00],
    },
    
    "SelfPlay_Mixed12": {
        "duration": 400,
        "weights": None,
        "epsilon": 0.02, "epsilon_min": 0.02,
        "memory_prune_low": 0.15,
        "opponent": None,
    },

    ########################## R/L1/L2 — Breathers ############################
    
    "R12_Breather_A": {
        "duration": 400,
        "weights":  [0.20, 0.60, 0.20, 0,0,0,0,0],
        "epsilon":  0.10, "epsilon_min": 0.05,
        "memory_prune_low": 0.18,
        "opponent": [0.20, 0.60, 0.20, 0,0,0,0,0],
    },
    
    "R12_Breather_B": {
        "duration": 400,
        "weights":  [0.20, 0.55, 0.25, 0,0,0,0,0],
        "epsilon":  0.08, "epsilon_min": 0.05,
        "memory_prune_low": 0.18,
        "opponent": [0.20, 0.55, 0.25, 0,0,0,0,0],
    },
    
   ############################# L2 bridge #################################
   
    "Fixed_L2_Short": {
        "duration": 400,
        "weights": [0.0, 0.0, 0.50, 0.50 ,0,0,0,0],
        "epsilon": 0.20, "epsilon_min": 0.08,
        "memory_prune_low": 0.18,
        "opponent": [0,0,1, 0,0,0,0,0],
    },
    
    "Variable_123": {
        "duration": 500,
        "weights": [0.00, 0.33, 0.33, 0.34,0,0,0,0],
        "epsilon": 0.12, "epsilon_min": 0.05,
        "memory_prune_low": 0.18,
        "opponent": [0.00, 0.33, 0.33, 0.34,0,0,0,0],
    },
    
    ############################ L2 Quartet ###################################
    
    "L2_Attack": {
        "duration": 400,
        "weights":  [0.0, 0.45, 0.40, 0.15 ,0,0,0,0],
        "epsilon":  0.15, "epsilon_min": 0.05,
        "memory_prune_low": 0.18,
        "opponent": [0.15, 0.45, 0.40, 0,0,0,0,0],
    },
    
    "L2_Defense": {
        "duration": 400,
        "weights":  [0.0, 0.25, 0.70, 0.05,0,0,0,0],
        "epsilon":  0.08, "epsilon_min": 0.05,
        "memory_prune_low": 0.18,
        "opponent": [0.05, 0.25, 0.70, 0,0,0,0,0],
    },
    
    "L2_Consolidate": {
        "duration": 600,
        "weights":  [0.0, 0.30, 0.40, 0.30,0,0,0,0],
        "epsilon":  0.06, "epsilon_min": 0.05,
        "memory_prune_low": 0.18,
        "opponent": [0.10, 0.40, 0.50, 0,0,0,0,0],
    },
    
    "SelfPlay_L2": {
        "duration": 600,
        "weights":  None,
        "epsilon":  0.02, "epsilon_min": 0.02,
        "memory_prune_low": 0.18,
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
