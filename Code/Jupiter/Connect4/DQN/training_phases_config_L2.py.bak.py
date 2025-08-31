# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 14:44:26 2025

@author: lovro
"""

# [R, L1, L2, L3, L4, L5, L6, L7]


    
TRAINING_PHASES = {

    # ---- buffer fill for the loaded model ----
    
    "L2_Random": {
        "duration": 400,                 
        "weights": [0.00, 0.00, 0.60, 0.40, 0, 0, 0, 0],   
        "epsilon": 0.55, "epsilon_min": 0.1,
        "memory_prune_low": 0.05,
        "opponent": [0.90, 0.10, 0.0, 0, 0, 0, 0, 0],     
    },
    
    "L1_Drill_Attack": {
        "duration": 400,
        "weights": [0.00, 0.15, 0.55, 0.30, 0, 0, 0, 0],   # agent explores via L2/L3
        "epsilon": 0.55, "epsilon_min": 0.10,              # force curriculum to act
        "memory_prune_low": 0.05,
        "opponent": [0.05, 0.95, 0.00, 0.00, 0, 0, 0, 0],  # play (mostly) L1
    },
    "L1_Drill_Defense": {
        "duration": 400,
        "weights": [0.00, 0.35, 0.60, 0.05, 0, 0, 0, 0],   # bias to L2 replies
        "epsilon": 0.50, "epsilon_min": 0.10,
        "memory_prune_low": 0.05,
        "opponent": [0.00, 0.95, 0.05, 0.00, 0, 0, 0, 0],  # mostly L1 with a little L2 pressure
    },
    
    "SelfPlay_L1_init": {
        "duration": 600,                 
        "weights": None,
        "epsilon": 0.02, "epsilon_min": 0.02,
        "memory_prune_low": 0.00,        
        "opponent": None,
    },
    
    # ---- L2 ----
    
    "L2_Attack": {
        "duration": 500,                 
        "weights": [0.00, 0.10, 0.65, 0.25, 0, 0, 0, 0],   
        "epsilon": 0.60, "epsilon_min": 0.1,
        "memory_prune_low": 0.05,
        "opponent": [0.15, 0.35, 0.50, 0, 0, 0, 0, 0],     
    },
    
    
    
    "L2_Defense": {
        "duration": 500,                 
        "weights": [0.00, 0.30, 0.65, 0.05, 0, 0, 0, 0],   
        "epsilon": 0.50, "epsilon_min": 0.1,
        "memory_prune_low": 0.05,
        "opponent": [0.00, 0.10, 0.90, 0, 0, 0, 0, 0],
    },
    
    "SelfPlay_L2_MID": {
        "duration": 600,                 
        "weights": None,
        "epsilon": 0.02, "epsilon_min": 0.02,
        "memory_prune_low": 0.05,
        "opponent": None,
    },
    
    "XPlay_L1_vs_L23_Defense": {           
        "duration": 500,                   
        "weights":  [0.0, 0.90, 0.10, 0.00, 0,0,0,0],   
        "epsilon":  0.70, "epsilon_min": 0.1,
        "memory_prune_low": 0.05,
        "opponent": [0.00, 0.10, 0.60, 0.30, 0,0,0,0],   
    },
    
    "XPlay_L3_vs_L1_Attack": {             
        "duration": 500,                   
        "weights":  [0.00, 0.0, 0.10, 0.90, 0,0,0,0],   
        "epsilon":  0.70, "epsilon_min": 0.1,
        "memory_prune_low": 0.05,
        "opponent": [0.05, 0.90, 0.05, 0.00, 0,0,0,0],   
    },
    
    "L1_Patch": {
        "duration": 300,
        "weights": [0.00, 0.25, 0.60, 0.15, 0, 0, 0, 0],
        "epsilon": 0.40, "epsilon_min": 0.08,
        "memory_prune_low": 0.05,
        "opponent": [0.05, 0.90, 0.05, 0.00, 0, 0, 0, 0],
    },

    "L2_Consolidate": {
        "duration": 600,
        "weights": [0.00, 0.30, 0.45, 0.25, 0, 0, 0, 0],   
        "epsilon": 0.30, "epsilon_min": 0.06,              
        "memory_prune_low": 0.05,
        "opponent": [0.10, 0.35, 0.55, 0, 0, 0, 0, 0],
    },
    
    "SelfPlay_L2": {
        "duration": 1500,                 
        "weights": None,
        "epsilon": 0.02, "epsilon_min": 0.02,
        "memory_prune_low": 0.05,
        "opponent": None,
    },
}
