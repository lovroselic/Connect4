# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 16:52:53 2025

@author: Uporabnik
"""

TRAINING_PHASES = {
    
    
    #################################################################
    
    #RANDOM I
    "Random_Intro": {
      "duration": 500, 
      "weights": [0.60, 0.40, 0.0, 0.0, 0,0,0,0],
      "epsilon": 0.50, 
      "epsilon_min": 0.10,
      "memory_prune_low": 0.000, #nothing to prune
      "opponent": [0.95, 0.05, 0.0, 0.0, 0,0,0,0],
      "TU_mode": "soft", 
      "tau": 0.0030,
      "guard_prob": 0.75, 
      "center_start": 0.90,
      "max_seed_fraction": 0.04,          
      "min_seed_fraction": 0.00,              
      "lr": 2e-4, 
      }
    ,
     "Random_Continue": {
       "duration": 500, 
       "weights": [0.30, 0.70, 0.0, 0.0, 0,0,0,0],
       "epsilon": 0.50, 
       "epsilon_min": 0.10,
       "memory_prune_low": 0.001, 
       "opponent": [0.80, 0.20, 0.0, 0.0, 0,0,0,0],
       "TU_mode": "soft", 
       "tau": 0.0030,
       "guard_prob": 0.75, 
       "center_start": 0.90,
       "seed_fraction": 0.05,
       "max_seed_fraction": 0.04,          
       "min_seed_fraction": 0.00, 
       "lr": 20e-5, 
     }, 
    
    
    # "Random_More": {
    #   "duration": 500, 
    #   "weights": [0.01, 0.84, 0.15, 0.0, 0,0,0,0],  
    #   "epsilon": 0.45,                              #0.47
    #   "epsilon_min": 0.08,                          #0.09
    #   "memory_prune_low": 0.00,                     #nothing to prune
    #   "opponent": [0.30, 0.65, 0.05, 0.0, 0,0,0,0], #[0.35, 0.60, 0.05, 0.0, 0,0,0,0]
    #   "TU_mode": "soft", 
    #   "tau": 0.0038,                                
    #   "guard_prob": 0.75,                           #0.80
    #   "center_start": 0.90,                         #0.95
    #   "max_seed_fraction": 0.041,                       
    #   "min_seed_fraction": 0.002,                   
    #   "lr": 20e-5,  
    # },  
    
    
    # # RANDOM VI
    # "L1_Random_Explorer_A": {
    #   "duration": 600, #600 
    #   "weights": [0.02, 0.85, 0.10, 0.03, 0,0,0,0],
    #   "epsilon": 0.45, 
    #   "epsilon_min": 0.09,          #0.10, 0.11 
    #   "memory_prune_low": 0.000, 
    #   "opponent": [0.60, 0.40, 0.0, 0.0, 0,0,0,0], 
    #   "TU_mode": "soft", 
    #   "tau": 0.0040,                    #0.0040     
    #   "guard_prob": 0.75,               #0.75
    #   "center_start": 0.75,             #0.85
    #   "max_seed_fraction": 0.04,        #0.04       
    #   "min_seed_fraction": 0.00,        #0.00       
    #   "lr": 20e-5,  
    # },  
    
    # # RANDOM VI COPY
    # "L1_Random_Explorer_A": {
    #   "duration": 600, #600 
    #   "weights": [0.02, 0.85, 0.10, 0.03, 0,0,0,0],
    #   "epsilon": 0.45, 
    #   "epsilon_min": 0.09,          #0.10, 0.11 
    #   "memory_prune_low": 0.000, 
    #   "opponent": [0.60, 0.40, 0.0, 0.0, 0,0,0,0], 
    #   "TU_mode": "soft", 
    #   "tau": 0.0040,                    #0.0040     
    #   "guard_prob": 0.75,               #0.75
    #   "center_start": 0.75,             #0.85
    #   "max_seed_fraction": 0.04,        #0.04       
    #   "min_seed_fraction": 0.00,        #0.00       
    #   "lr": 20e-5,  
    # },  
    
    
    
    
    
    # RANDOM BASE
    

 # "Random_More": {
 #   "duration": 500, 
 #   "weights": [0.15, 0.80, 0.05, 0.0, 0,0,0,0],
 #   "epsilon": 0.45, 
 #   "epsilon_min": 0.10,
 #   "memory_prune_low": 0.001, 
 #   "opponent": [0.70, 0.30, 0.0, 0.0, 0,0,0,0],
 #   "TU_mode": "soft", 
 #   "tau": 0.0028,
 #   "guard_prob": 0.75, 
 #   "center_start": 0.90,
 #   "seed_fraction": 0.05,
 #   "lr": 20e-5,  
 # },  
    
 # "Random_Extreme_new": {
 #     "duration": 500, 
 #     "weights": [0.05, 0.85, 0.10, 0.0, 0,0,0,0],
 #     "epsilon": 0.45, 
 #     "epsilon_min": 0.10,
 #     "memory_prune_low": 0.001, 
 #     "opponent": [0.60, 0.40, 0.0, 0.0, 0,0,0,0],
 #     "TU_mode": "soft", 
 #     "tau": 0.0027,
 #     "guard_prob": 0.73, 
 #     "center_start": 0.88,
 #     "seed_fraction": 0.05,
 #     "lr": 20e-5,  
 #     }, 
 
 #    "Random_Extreme": {
 #        "duration": 500, 
 #        "weights": [0.05, 0.85, 0.10, 0.0, 0,0,0,0],
 #        "epsilon": 0.45, 
 #        "epsilon_min": 0.10,
 #        "memory_prune_low": 0.001, 
 #        "opponent": [0.50, 0.50, 0.0, 0.0, 0,0,0,0],
 #        "TU_mode": "soft", 
 #        "tau": 0.0027,
 #        "guard_prob": 0.73, 
 #        "center_start": 0.88,
 #        "seed_fraction": 0.05,
 #        "lr": 20e-5,  
 #        }, 
    

 #    ###########################################
 
 #    # RANDOM VI - not run but was stable
 #    "L1_Random_Explorer_A": {
 #      "duration": 600, #600 
 #      "weights": [0.02, 0.85, 0.10, 0.03, 0,0,0,0],
 #      "epsilon": 0.45, 
 #      "epsilon_min": 0.09,          #0.10, 0.11 
 #      "memory_prune_low": 0.000, 
 #      "opponent": [0.60, 0.40, 0.0, 0.0, 0,0,0,0], 
 #      "TU_mode": "soft", 
 #      "tau": 0.0040,                    #0.0040     
 #      "guard_prob": 0.75,               #0.75
 #      "center_start": 0.75,             #0.85
 #      "max_seed_fraction": 0.04,        #0.04       
 #      "min_seed_fraction": 0.00,        #0.00       
 #      "lr": 20e-5,  
 #    },  


    }