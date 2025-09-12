# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 14:44:26 2025

@author: lovro
"""

# [R, L1, L2, L3, L4, L5, L6, L7]


TRAINING_PHASES = {
    
  "SelfPlay_Mixed": {                 
    "duration": 50, "weights": None, #100
    "epsilon": 0.04, "epsilon_min": 0.01,
    "memory_prune_low": 0.000,        
    "opponent": None,
    "TU": 0, "TU_mode": "soft", "tau": 0.0078,
    "guard_prob": 0.05,
    "center_start": 0.40,             
  },
  
  "L1_Reintroduce": {                
    "duration": 500, "weights": [0,0.65,0.35,0,0,0,0,0],
    "epsilon": 0.22, "epsilon_min": 0.06,
    "memory_prune_low": 0.004,       
    "opponent": [0.10,0.80,0.10,0,0,0,0,0],
    "TU": 500, "TU_mode": "hard",
    "guard_prob": 0.25,
    "center_start": 0.35,
  },

#   "L2_Intro": {                       
#     "duration": 600, "weights": [0,0.55,0.35,0.10,0,0,0,0],
#     "epsilon": 0.28, "epsilon_min": 0.06,
#     "memory_prune_low": 0.003,        # slightly lighter here
#     "opponent": [0.10,0.80,0.10,0,0,0,0,0],
#     "TU": 500, "TU_mode": "hard",
#     "guard_prob": 0.20,
#     "center_start": 0.30,
#   },

#   "SelfPlay_Anchor": {                
#     "duration": 300, "weights": None,
#     "epsilon": 0.03, "epsilon_min": 0.01,
#     "memory_prune_low": 0.000,        
#     "opponent": None,
#     "TU": 0, "TU_mode": "soft", "tau": 0.008,
#     "guard_prob": 0.10,
#     "center_start": 0.25,
#   },

#   "L1_Bounce": {                      
#     "duration": 500, "weights": [0,0.65,0.25,0.10,0,0,0,0],
#     "epsilon": 0.16, "epsilon_min": 0.05,
#     "memory_prune_low": 0.003,        # light trim
#     "opponent": [0.10,0.90,0.00,0,0,0,0,0],
#     "TU": 500, "TU_mode": "hard", "tau": 0.008,
#     "guard_prob": 0.10,
#     "center_start": 0.20,
#   },

#   "L2_Ramp": {                       
#     "duration": 500, "weights": [0,0.45,0.40,0.15,0,0,0,0],
#     "epsilon": 0.18, "epsilon_min": 0.05,
#     "memory_prune_low": 0.005,        
#     "opponent": [0.05,0.75,0.20,0,0,0,0,0],
#     "TU": 500, "TU_mode": "hard", "tau": 0.008,
#     "guard_prob": 0.08,
#     "center_start": 0.15,
#   },

#   "L2_Challenge": {                   
#     "duration": 600, "weights": [0,0.25,0.55,0.20,0,0,0,0],
#     "epsilon": 0.22, "epsilon_min": 0.05,
#     "memory_prune_low": 0.004,        
#     "opponent": [0.00,0.70,0.30,0,0,0,0,0],
#     "TU": 750, "TU_mode": "hard", "tau": 0.008,
#     "guard_prob": 0.05,
#     "center_start": 0.10,
#   },

#   "SelfPlay_L2": {                    
#     "duration": 700, "weights": None,
#     "epsilon": 0.03, "epsilon_min": 0.01,
#     "memory_prune_low": 0.00,
#     "opponent": None,
#     "TU": 0, "TU_mode": "soft", "tau": 0.008,
#     "guard_prob": 0.15,
#     "center_start": 0.05,
#   },

}



