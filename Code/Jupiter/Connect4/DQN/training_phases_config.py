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
    
    ######################### Mixed Section ############################################
    
    "SelfPlay_Seed": {
        "duration": 50,
        "weights": None,
        "epsilon": 0.00, "epsilon_min": 0.00,
        "memory_prune_low": 0.00,
        "opponent": None,
    },
    
    #keep
    "Mixed": {
        "duration": 550, 
        "weights": [0.2, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  
        "epsilon": 0.75,
        "epsilon_min": 0.1,
        "memory_prune_low": 0.00, # no sense pruning yet!
        "opponent": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  
        },
    
    "SelfPlay_M1": {
        "duration": 100,
        "weights": None,
        "epsilon": 0.05, "epsilon_min": 0.01,
        "memory_prune_low": 0.00,
        "opponent": None,
    },
    
    "L1_vs_Mixed2": {
        "duration": 500, #500
        "weights": [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  
        "epsilon": 0.75,
        "epsilon_min": 0.1,
        "memory_prune_low": 0.05,
        "opponent": [0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  
        },
    
    "SelfPlay_M2": {
        "duration": 100,
        "weights": None,
        "epsilon": 0.05, "epsilon_min": 0.01,
        "memory_prune_low": 0.00,
        "opponent": None,
    },
    
    "L1_vs_Mixed3": {
        "duration": 500, #500
        "weights": [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  
        "epsilon": 0.75,
        "epsilon_min": 0.1,
        "memory_prune_low": 0.05,
        "opponent": [0.25, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  
        },
    
    "SelfPlay_M3": {
        "duration": 100,
        "weights": None,
        "epsilon": 0.05, "epsilon_min": 0.01,
        "memory_prune_low": 0.00,
        "opponent": None,
    },
    

    "L1_vs_Mixed4": {
        "duration": 600, 
        "weights": [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        "epsilon": 0.75, 
        "epsilon_min": 0.1,
        "memory_prune_low": 0.05,
        "opponent": [0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    },
    
    "SelfPlay_Random": {
        "duration": 1,                 
        "weights": None,
        "epsilon": 0.0, "epsilon_min": 0.00,
        "memory_prune_low": 0.00, #no sense pruning last replay
        "opponent": None,
    },


    # "L1L2_vs_Mixed2": {
    #     "duration": 500,
    #     "weights": [0.0, 0.5, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0],  
    #     "epsilon": 0.75,
    #     "epsilon_min": 0.1,
    #     "memory_prune_low": 0.05,
    #     "opponent": [0.10, 0.90, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  
    #     },
    
    
    # "SelfPlay_Mixed3": {
    #     "duration": 500,
    #     "weights": None,
    #     "epsilon": 0.00, "epsilon_min": 0.00,
    #     "memory_prune_low": 0.00,
    #     "opponent": None,
    # },
    
    
    ##### FINAL guard #####
    
    "SelfPlay_Final": {
        "duration": 1,                 
        "weights": None,
        "epsilon": 0.0, "epsilon_min": 0.00,
        "memory_prune_low": 0.00, #no sense pruning last replay
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
    return length

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
