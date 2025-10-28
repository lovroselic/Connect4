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
   "epsilon": 0.65, 
   "epsilon_min": 0.10,
   "memory_prune_low": 0.000,                            #nothing to prune
   "opponent": [0.95, 0.05, 0.0, 0.0, 0,0,0,0],
   "TU_mode": "soft", 
   "tau": 0.0035,                                        #"tau": 0.0030,
   "guard_prob": 0.75, 
   "center_start": 0.90,
   "max_seed_fraction": 0.45,                            #"max_seed_fraction": 0.15
   "min_seed_fraction": 0.15,                            #min_seed_fraction": 0.05,  
   "lr": 2e-4, 
   },
 
 
 
    #RANDOM II R2 (0.225, 0.075)
    "Random_Cont": {
      "duration": 500, 
      "weights": [0.40, 0.60, 0.0, 0.0, 0,0,0,0],
      "epsilon": 0.65, 
      "epsilon_min": 0.10,
      "memory_prune_low": 0.000,                            #nothing to prune
      "opponent": [0.85, 0.15, 0.0, 0.0, 0,0,0,0],
      "TU_mode": "soft", 
      "tau": 0.0035,                                        #"tau": 0.0030,
      "guard_prob": 0.75, 
      "center_start": 0.90,
      "max_seed_fraction": 0.225,                            #"max_seed_fraction": 0.45
      "min_seed_fraction": 0.075,                            #min_seed_fraction": 0.15,  
      "lr": 2e-4, 
      },
    #RANDOM III R3(T 0.0040, Max 0.30, Min 0.10)
    "Random_More": {
      "duration": 500, 
      "weights": [0.20, 0.75, 0.05, 0.0, 0,0,0,0],
      "epsilon": 0.60,                                         #0.65
      "epsilon_min": 0.10,
      "memory_prune_low": 0.000,                               #nothing to prune
      "opponent": [0.65, 0.30, 0.05, 0.0, 0,0,0,0],
      "TU_mode": "soft", 
      "tau": 0.0040,                                           #"tau": 0.0035,
      "guard_prob": 0.74,                                      #"guard_prob": 0.75,
      "center_start": 0.90,                                    #"center_start": 0.89,
      "max_seed_fraction": 0.30,                              #"max_seed_fraction": 0.225
      "min_seed_fraction": 0.10,                              #"min_seed_fraction": 0.075,  
      "lr": 2e-4, 
      },
    
    #RANDOM IV R4(T 0.0045, Max 0.35, Min 0.10)
    "Random_Extreme": {
      "duration": 500, 
      "weights": [0.05, 0.85, 0.10, 0.0, 0,0,0,0],#            "weights": [0.20, 0.75, 0.05, 0.0, 0,0,0,0],
      "epsilon": 0.57,                                         #0.60
      "epsilon_min": 0.10,
      "memory_prune_low": 0.000,                               #nothing to prune
      "opponent": [0.50, 0.45, 0.05, 0.0, 0,0,0,0],            #opponent": [0.65, 0.30, 0.05, 0.0, 0,0,0,0],
      "TU_mode": "soft", 
      "tau": 0.0045,                                           #"tau": 0.0040,
      "guard_prob": 0.73,                                      #"guard_prob": 0.75,
      "center_start": 0.90,                                    #"center_start": 0.90,
      "max_seed_fraction": 0.35,                               #"max_seed_fraction": 0.30
      "min_seed_fraction": 0.10,                               #"min_seed_fraction": 0.10,  
      "lr": 2e-4, 
      },
   
    
    }