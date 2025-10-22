# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 16:52:53 2025

@author: Uporabnik
"""

TRAINING_PHASES = {
    
    
    #################################################################
    # RANDOM BASE
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
 "Random_Continue": {
   "duration": 500, 
   "weights": [0.30, 0.70, 0.0, 0.0, 0,0,0,0],
   "epsilon": 0.50, 
   "epsilon_min": 0.10,
   "memory_prune_low": 0.001, 
   "opponent": [0.80, 0.20, 0.0, 0.0, 0,0,0,0],
   "TU_mode": "soft", 
   "tau": 0.0029,
   "guard_prob": 0.75, 
   "center_start": 0.90,
   "seed_fraction": 0.05,
   "lr": 20e-5, 
 },    
 
 "Random_More": {
   "duration": 500, 
   "weights": [0.15, 0.80, 0.05, 0.0, 0,0,0,0],
   "epsilon": 0.45, 
   "epsilon_min": 0.10,
   "memory_prune_low": 0.001, 
   "opponent": [0.70, 0.30, 0.0, 0.0, 0,0,0,0],
   "TU_mode": "soft", 
   "tau": 0.0028,
   "guard_prob": 0.75, 
   "center_start": 0.90,
   "seed_fraction": 0.05,
   "lr": 20e-5,  
 },  
    
 "Random_Extreme_new": {
     "duration": 500, 
     "weights": [0.05, 0.85, 0.10, 0.0, 0,0,0,0],
     "epsilon": 0.45, 
     "epsilon_min": 0.10,
     "memory_prune_low": 0.001, 
     "opponent": [0.60, 0.40, 0.0, 0.0, 0,0,0,0],
     "TU_mode": "soft", 
     "tau": 0.0027,
     "guard_prob": 0.73, 
     "center_start": 0.88,
     "seed_fraction": 0.05,
     "lr": 20e-5,  
     }, 
 
    "Random_Extreme": {
        "duration": 500, 
        "weights": [0.05, 0.85, 0.10, 0.0, 0,0,0,0],
        "epsilon": 0.45, 
        "epsilon_min": 0.10,
        "memory_prune_low": 0.001, 
        "opponent": [0.50, 0.50, 0.0, 0.0, 0,0,0,0],
        "TU_mode": "soft", 
        "tau": 0.0027,
        "guard_prob": 0.73, 
        "center_start": 0.88,
        "seed_fraction": 0.05,
        "lr": 20e-5,  
        }, 
    

    ###########################################
 



    }