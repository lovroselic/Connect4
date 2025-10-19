# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 16:52:53 2025

@author: Uporabnik
"""

TRAINING_PHASES = {
    
    # RANDOM I
    "Random_Intro": {
      "duration": 500, 
      "weights": [0.60, 0.40, 0.0, 0.0, 0,0,0,0],
      "epsilon": 0.50, "epsilon_min": 0.10,
      "memory_prune_low": 0.000, #nothing to prune
      "opponent": [0.95, 0.05, 0.0, 0.0, 0,0,0,0],
      "TU_mode": "soft", 
      "tau": 0.0030,
      "guard_prob": 0.75, 
      "center_start": 0.90,
      "seed_fraction": 0.01,
      "lr": 2e-4, 
      }
    ,
    
    
    # # RANDOM II
    # "Random_Continue": {
    #   "duration": 500, 
    #   "weights": [0.30, 0.70, 0.0, 0.0, 0,0,0,0],
    #   "epsilon": 0.50, "epsilon_min": 0.10,
    #   "memory_prune_low": 0.000, 
    #   "opponent": [0.80, 0.20, 0.0, 0.0, 0,0,0,0],
    #   "TU": 500, "TU_mode": "soft", "tau": 0.010,
    #   "guard_prob": 0.70, "center_start": 0.80,
    # },    
    
    # # RANDOM III
    # "Random_More": {
    #   "duration": 500, 
    #   "weights": [0.15, 0.80, 0.05, 0.0, 0,0,0,0],
    #   "epsilon": 0.45, "epsilon_min": 0.10,
    #   "memory_prune_low": 0.000, 
    #   "opponent": [0.70, 0.30, 0.0, 0.0, 0,0,0,0],
    #   "TU": 500, "TU_mode": "soft", "tau": 0.010,
    #   "guard_prob": 0.65, "center_start": 0.75,
    # },    
    
    # # RANDOM IV
    # "Random_Extreme": {
    #   "duration": 500, 
    #   "weights": [0.05, 0.85, 0.10, 0.0, 0,0,0,0],
    #   "epsilon": 0.45, 
    #   "epsilon_min": 0.10,
    #   "memory_prune_low": 0.000, 
    #   "opponent": [0.50, 0.50, 0.0, 0.0, 0,0,0,0],
    #   "TU": 500, 
    #   "TU_mode": "soft", 
    #   "tau": 0.010,
    #   "guard_prob": 0.60,       #0.6
    #   "center_start": 0.70,     #0.7
    # }, 
    
    # # RANDOM V
    # "Random_Final": {
    #   "duration": 500, 
    #   "weights": [0.01, 0.80, 0.19, 0.0, 0,0,0,0],
    #   "epsilon": 0.45, 
    #   "epsilon_min": 0.10,
    #   "memory_prune_low": 0.000, 
    #   "opponent": [0.40, 0.60, 0.0, 0.0, 0,0,0,0],
    #   "TU": 500, 
    #   "TU_mode": "soft", 
    #   "tau": 0.010,
    #   "guard_prob": 0.60,       #0.6
    #   "center_start": 0.70,     #0.7
    # },  
    
    # # RANDOM VI
    # "Random_Reprise": {
    #   "duration": 500, 
    #   "weights": [0.01, 0.7, 0.2, 0.09, 0, 0, 0, 0],
    #   "epsilon": 0.35, 
    #   "epsilon_min": 0.10,
    #   "memory_prune_low": 0.003, 
    #   "opponent": [0.40, 0.60, 0.0, 0.0, 0,0,0,0],
    #   "TU_mode": "soft", 
    #   "tau": 0.004,
    #   "guard_prob": 0.60,       #0.6
    #   "center_start": 0.70,     #0.7
    # }, 
    
    # #RANDOM VII
    #         "Random_Explore": { 
    #             "duration": 350, 
    #             "weights": [0.01, 0.60, 0.19, 0.20, 0,0,0,0], 
    #             "epsilon": 0.60, "epsilon_min": 0.10, 
    #             "memory_prune_low": 0.000, 
    #             "opponent": [0.45, 0.55, 0.0, 0.0, 0,0,0,0], 
    #             "TU_mode": "soft", 
    #             "tau": 0.004, 
    #             "guard_prob": 0.50, 
    #             "center_start": 0.65, 
    #             "seed_fraction": 0.90,
    #             },
            
    #         "Random_Explore2": {
    #           "duration": 750, 
    #           "weights": [0.01, 0.50, 0.24, 0.25, 0,0,0,0],
    #           "epsilon": 0.45, 
    #           "epsilon_min": 0.10,
    #           "memory_prune_low": 0.003, 
    #           "opponent": [0.45, 0.55, 0.0, 0.0, 0,0,0,0],
    #           "TU_mode": "soft", 
    #           "tau": 0.0035,
    #           "guard_prob": 0.50,       #0.6
    #           "center_start": 0.65,     #0.7
    #           "seed_fraction": 0.85,    #0.90
    #         }, 
    
    
    # #intermeditae were distillations
    
    # #RE3_NO_RND DQN model
    # "Random_Explore3": { 
    #     "duration": 700, 
    #     "weights": [0.01, 0.45, 0.24, 0.30, 0,0,0,0], 
    #     "epsilon": 0.45, 
    #     "epsilon_min": 0.10, 
    #     "memory_prune_low": 0.000, 
    #     "opponent": [0.40, 0.60, 0.0, 0.0, 0,0,0,0], 
    #     "TU_mode": "soft", "tau": 0.0030, 
    #     "guard_prob": 0.48, 
    #     "center_start": 0.60, 
    #     "seed_fraction": 0.00, 
    #     }, 
    
    
    #     #RE5
    #     "Random_Explore3a": { 
    #         "duration": 700, 
    #         "weights": [0.01, 0.45, 0.24, 0.30, 0,0,0,0], 
    #         "epsilon": 0.45, 
    #         "epsilon_min": 0.10, 
    #         "memory_prune_low": 0.000, 
    #         "opponent": [0.40, 0.60, 0.0, 0.0, 0,0,0,0], 
    #         "TU_mode": "soft", "tau": 0.0030, 
    #         "guard_prob": 0.48, 
    #         "center_start": 0.60, 
    #         "seed_fraction": 0.00, 
    #         }, 
        
    #     "Random_Explore4": { 
    #         "duration": 700, 
    #         "weights": [0.01, 0.39, 0.30, 0.30, 0,0,0,0], 
    #         "epsilon": 0.42, "epsilon_min": 0.10, 
    #         "memory_prune_low": 0.003, 
    #         "opponent": [0.30, 0.60, 0.05, 0.05, 0,0,0,0], 
    #         "TU_mode": "soft", "tau": 0.0028, 
    #         "guard_prob": 0.45, 
    #         "center_start": 0.55, 
    #         "seed_fraction": 0.01},
    
    #     "Random_Explore5": {
    #       "duration": 1200, 
    #       "weights": [0.01, 0.29, 0.35, 0.35, 0,0,0,0],
    #       "epsilon": 0.38, 
    #       "epsilon_min": 0.08,
    #       "memory_prune_low": 0.005, 
    #       "opponent": [0.25, 0.55, 0.15, 0.05, 0,0,0,0],
    #       "TU_mode": "soft", 
    #       "tau": 0.0026,
    #       "guard_prob": 0.40,       
    #       "center_start": 0.50,     
    #       "seed_fraction": 0.02,    
    #     }, 
    
    # #intermeditae were distillations
    
    # #L1CONT I
    # "L1_Intro": {
    #   "duration": 560, 
    #   "weights": [0.01, 0.19, 0.40, 0.40, 0,0,0,0],
    #   "epsilon": 0.38, 
    #   "epsilon_min": 0.10,
    #   "memory_prune_low": 0.000, 
    #   "opponent": [0.10, 0.50, 0.30, 0.10, 0,0,0,0],
    #   "TU_mode": "soft", 
    #   "tau": 0.0023, 
    #   "guard_prob": 0.37,       
    #   "center_start": 0.45,     
    #   "seed_fraction": 0.02,    
    #   "lr": 2e-4, 
    # }, 
    
    # "L1_Continue": {
    #   "duration": 400, 
    #   "weights": [0.01, 0.09, 0.50, 0.40, 0,0,0,0],
    #   "epsilon": 0.37, 
    #   "epsilon_min": 0.08,
    #   "memory_prune_low": 0.001, 
    #   "opponent": [0.05, 0.40, 0.40, 0.15, 0,0,0,0],
    #   "TU_mode": "soft", 
    #   "tau": 0.0024,  
    #   "guard_prob": 0.33,       
    #   "center_start": 0.40,     
    #   "seed_fraction": 0.01,    
    #   "lr": 1.6e-4, 
    # }, 
    
    # ## failed
    
    # "L1_Intro_B": {
    #   "duration": 200, 
    #   "weights": [0.01, 0.19, 0.40, 0.40, 0,0,0,0],
    #   "epsilon": 0.40, 
    #   "epsilon_min": 0.10,
    #   "memory_prune_low": 0.000, 
    #   "opponent": [0.10, 0.50, 0.35, 0.05, 0,0,0,0],
    #   "TU_mode": "soft", 
    #   "tau": 0.0024, 
    #   "guard_prob": 0.35,       
    #   "center_start": 0.40,     
    #   "seed_fraction": 0.02,    
    #   "lr": 2e-4, 
    # }, 
    
    # "L1_Continue_B": {
    #   "duration": 500, 
    #   "weights": [0.01, 0.09, 0.50, 0.40, 0,0,0,0],
    #   "epsilon": 0.37, 
    #   "epsilon_min": 0.08,
    #   "memory_prune_low": 0.001, 
    #   "opponent": [0.075, 0.45, 0.40, 0.075, 0,0,0,0],
    #   "TU_mode": "soft", 
    #   "tau": 0.0023,  
    #   "guard_prob": 0.33,       
    #   "center_start": 0.40,     
    #   "seed_fraction": 0.01,    
    #   "lr": 1.6e-4, 
    # }, 
    
    # "L1_Late": {
    #   "duration": 500, 
    #   "weights": [0.01, 0.04, 0.50, 0.45, 0,0,0,0],
    #   "epsilon": 0.36, 
    #   "epsilon_min": 0.08,
    #   "memory_prune_low": 0.001, 
    #   "opponent": [0.05, 0.40, 0.42, 0.13, 0,0,0,0], 
    #   "TU_mode": "soft", 
    #   "tau": 0.0023,  
    #   "guard_prob": 0.30,       
    #   "center_start": 0.35,     
    #   "seed_fraction": 0.01,    
    #   "lr": 1.55e-4, 
    # }, 
    
    # "L1_End": {
    #   "duration": 500, 
    #   "weights": [0.01, 0.01, 0.50, 0.48, 0,0,0,0],
    #   "epsilon": 0.36, 
    #   "epsilon_min": 0.08,
    #   "memory_prune_low": 0.001, 
    #   "opponent": [0.05, 0.35, 0.45, 0.15, 0,0,0,0], 
    #   "TU_mode": "soft", 
    #   "tau": 0.0022,  
    #   "guard_prob": 0.28,       
    #   "center_start": 0.32,     
    #   "seed_fraction": 0.01,    
    #   "lr": 1.53e-4, 
    # }, 
    }