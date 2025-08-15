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
    # Random ############################################
    
    "Random1": {
        "duration": 500,
        "weights": [0.50, 0.45, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 1.0, "epsilon_min": 0.20, "memory_prune": 0.00,
        "opponent": [1, 0, 0, 0, 0, 0, 0, 0],  
    },
    "Random2": {
        "duration": 200,
        "weights": [0.45, 0.45, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 1.0, "epsilon_min": 0.20, "memory_prune": 0.20,
        "opponent": [1, 0, 0, 0, 0, 0, 0, 0],  
    },

    # Mixed R/L1 #########################################
    
    "Mixed_RR1": {
        "duration": 500,
        "weights": [0.30, 0.45, 0.20, 0.05, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 0.90, "epsilon_min": 0.20, "memory_prune": 0.20,
        "opponent": [0.75, 0.25, 0, 0, 0, 0, 0, 0],  
    },
    "Mixed_R1": {
        "duration": 500,
        "weights": [0.25, 0.55, 0.125, 0.05, 0.025, 0.00, 0.00, 0.00],
        "epsilon": 0.85, "epsilon_min": 0.15, "memory_prune": 0.20,
        "opponent": [0.50, 0.50, 0, 0, 0, 0, 0, 0], 
    },
    "Mixed_R11": {
        "duration": 500,
        "weights": [0.20, 0.50, 0.25, 0.05, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 0.80, "epsilon_min": 0.12, "memory_prune": 0.25,
        "opponent": [0.40, 0.60, 0, 0, 0, 0, 0, 0],  
    },

    "SelfPlay_Begin": {
        "duration": 500,
        "weights": [0, 1, 0, 0, 0, 0, 0, 0],
        "epsilon": 0.0, "epsilon_min": 0.05, "memory_prune": 0.30,
        "opponent": None,  
    },

    # L1 ##########################################
    
    "Fixed1": {
        "duration": 500,
        "weights": [0.10, 0.55, 0.25, 0.10, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 0.65, "epsilon_min": 0.10, "memory_prune": 0.20,
        "opponent": [0.10, 0.90, 0, 0, 0, 0, 0, 0],  
    },
    "Fixed1-Reprise": {
        "duration": 500,
        "weights": [0.05, 0.60, 0.25, 0.08, 0.02, 0.00, 0.00, 0.00],
        "epsilon": 0.55, "epsilon_min": 0.08, "memory_prune": 0.20,
        "opponent": [0.15, 0.85, 0, 0, 0, 0, 0, 0],  
    },

    # Mixed L1/L2 ramp ############################
    
    "Mixed_R12": {
        "duration": 500,
        "weights": [0.02, 0.25, 0.45, 0.18, 0.10, 0.00, 0.00, 0.00],
        "epsilon": 0.55, "epsilon_min": 0.08, "memory_prune": 0.15,
        "opponent": [0.25, 0.375, 0.375, 0, 0, 0, 0, 0],  
    },
    "Challenge_R12": {
        "duration": 500,
        "weights": [0.10, 0.35, 0.35, 0.15, 0.05, 0.00, 0.00, 0.00],
        "epsilon": 0.20, "epsilon_min": 0.08, "memory_prune": 0.15,
        "opponent": [0, 0.05, 0.45, 0.40, 0.10, 0, 0, 0],  
    },
    "Relax_R12": {
        "duration": 500,
        "weights": [0.00, 0.30, 0.50, 0.20, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 0.25, "epsilon_min": 0.12, "memory_prune": 0.10,
        "opponent": [0.30, 0.50, 0.20, 0, 0, 0, 0, 0],  
    },
    
 # -------------------- L1/L2 ramp (2nd set) --------------------
    "Mixed_R1122": {
        "duration": 500,
        "weights":   [0.02, 0.25, 0.45, 0.18, 0.10, 0.00, 0.00, 0.00],
        "opponent":  [0.20, 0.40, 0.40, 0.00, 0.00, 0.00, 0.00, 0.00],  
        "epsilon": 0.55, "epsilon_min": 0.08, "memory_prune": 0.15,
    },
    "Challenge_R1122": {
        "duration": 500,
        "weights":   [0.10, 0.35, 0.35, 0.15, 0.05, 0.00, 0.00, 0.00],
        "opponent":  [0.00, 0.05, 0.45, 0.40, 0.10, 0.00, 0.00, 0.00],  
        "epsilon": 0.20, "epsilon_min": 0.08, "memory_prune": 0.15,
    },
    "Relax_R1122": {
        "duration": 500,
        "weights":   [0.00, 0.30, 0.50, 0.20, 0.00, 0.00, 0.00, 0.00],
        "opponent":  [0.30, 0.49, 0.21, 0.00, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 0.25, "epsilon_min": 0.12, "memory_prune": 0.10,
    },
        
 # -------------------- L1/L2 endcap --------------------
    "Mixed12": {
        "duration": 500,
        "weights":   [0.00, 0.25, 0.25, 0.20, 0.20, 0.10, 0.00, 0.00],
        "opponent":  [0.00, 0.50, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00],  
        "epsilon": 0.40, "epsilon_min": 0.08, "memory_prune": 0.15,
    },
    "Breather_L1L2Bridge": {
        "duration": 250,
        "weights":   [0.00, 0.25, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00],
        "opponent":  [0.00, 0.25, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00],  
        "epsilon": 0.40, "epsilon_min": 0.08, "memory_prune": 0.10,
    },
    "SelfPlay_L1L2": {
        "duration": 500,
        "weights":   [],
        "opponent": None, 
        "epsilon": 0.0, "epsilon_min": 0.05, "memory_prune": 0.15,
    },

# -------------------- L2 block --------------------
    "Mixed123": {
        "duration": 500,
        "weights":   [0.05, 0.30, 0.35, 0.25, 0.05, 0.00, 0.00, 0.00],
        "opponent":  [0.00, 1/3, 1/3, 1/3, 0.00, 0.00, 0.00, 0.00],     
        "epsilon": 0.40, "epsilon_min": 0.06, "memory_prune": 0.10,
    },
    "Fixed2": {
        "duration": 500,
        "weights":   [0.05, 0.20, 0.45, 0.20, 0.10, 0.00, 0.00, 0.00],
        "opponent":  [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # pure L2
        "epsilon": 0.35, "epsilon_min": 0.05, "memory_prune": 0.10,
    },
    "Fixed2_Reprise": {
        "duration": 500,
        "weights":   [0.00, 0.15, 0.50, 0.20, 0.10, 0.05, 0.00, 0.00],
        "opponent":  [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # pure L2
        "epsilon": 0.40, "epsilon_min": 0.05, "memory_prune": 0.05,
    },
    "Breather_L2Consolidate": {
        "duration": 200,
        "weights":   [0.00, 0.15, 0.65, 0.20, 0.00, 0.00, 0.00, 0.00],
        "opponent":  [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # pure L2
        "epsilon": 0.20, "epsilon_min": 0.05, "memory_prune": 0.10,
    },
    "SelfPlay_L2": {
        "duration": 500,
        "weights":   [],
        "opponent": None,  
        "epsilon": 0.0, "epsilon_min": 0.05, "memory_prune": 0.15,
    },

# -------------------- L3 block --------------------
    "Variable23": {
        "duration": 500,
        "weights":   [0.00, 0.10, 0.40, 0.40, 0.05, 0.05, 0.00, 0.00],
        "opponent":  [0.00, 0.00, 0.50, 0.50, 0.00, 0.00, 0.00, 0.00],  
        "epsilon": 0.40, "epsilon_min": 0.05, "memory_prune": 0.10,
    },
    "Variable3": {
        "duration": 500,
        "weights":   [0.00, 0.05, 0.35, 0.50, 0.05, 0.05, 0.00, 0.00],
        "opponent":  [0.00, 1/3, 1/3, 1/3, 0.00, 0.00, 0.00, 0.00],     
        "epsilon": 0.40, "epsilon_min": 0.05, "memory_prune": 0.10,
    },
    "Breather_L2L3Blend": {
        "duration": 200,
        "weights":   [0.00, 0.05, 0.45, 0.45, 0.05, 0.00, 0.00, 0.00],
        "opponent":  [0.00, 0.00, 0.50, 0.50, 0.00, 0.00, 0.00, 0.00],  
        "epsilon": 0.25, "epsilon_min": 0.05, "memory_prune": 0.10,
    },
    "Fixed3": {
        "duration": 500,
        "weights":   [0.00, 0.00, 0.20, 0.50, 0.20, 0.10, 0.00, 0.00],
        "opponent":  [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],  
        "epsilon": 0.40, "epsilon_min": 0.05, "memory_prune": 0.10,
    },
    "SelfPlay_L3": {
        "duration": 500,
        "weights":   [],
        "opponent": None, 
        "epsilon": 0.0, "epsilon_min": 0.05, "memory_prune": 0.05,
    },

# -------------------- Final --------------------
    "Final": {
        "duration": 500,
        "weights":   [],
        "opponent": None, 
        "epsilon": 0.0, "epsilon_min": 0.0, "memory_prune": 0.05,
    },
}

def set_training_phases_length(DICT):
    length = 0;
    for name, TF in DICT.items():
        length += int(TF["duration"])
        TF["length"] = length
        TF["sumWeights"] = np.sum(TF["weights"])
        TF["sumOppo"] = np.sum(TF["opponent"])
        
    display_dict_as_tab(DICT)
        
def display_dict_as_tab(DICT):
    DF = pd.DataFrame.from_dict(DICT, orient="index")  
    cols = ['length', "duration", "epsilon", "epsilon_min", "memory_prune", "sumWeights"]
    DF = DF[cols]
    DF = DF.sort_values(by=['length'], ascending=True)
    display(HTML(DF.to_html()))
    
