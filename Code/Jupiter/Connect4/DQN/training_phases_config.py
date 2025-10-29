# training_phases_config.py
# [R, L1, L2, L3, L4, L5, L6, L7]



TRAINING_PHASES = {
    

    
    # #RANDOM IV R4(T 0.0045, Max 0.35, Min 0.10)
    # "Random_Extreme": {
    #   "duration": 500, 
    #   "weights": [0.05, 0.85, 0.10, 0.0, 0,0,0,0],#            "weights": [0.20, 0.75, 0.05, 0.0, 0,0,0,0],
    #   "epsilon": 0.57,                                         #0.60
    #   "epsilon_min": 0.10,
    #   "memory_prune_low": 0.000,                               #nothing to prune
    #   "opponent": [0.50, 0.45, 0.05, 0.0, 0,0,0,0],            #opponent": [0.65, 0.30, 0.05, 0.0, 0,0,0,0],
    #   "TU_mode": "soft", 
    #   "tau": 0.0045,                                           #"tau": 0.0040,
    #   "guard_prob": 0.73,                                      #"guard_prob": 0.75,
    #   "center_start": 0.90,                                    #"center_start": 0.90,
    #   "max_seed_fraction": 0.35,                               #"max_seed_fraction": 0.30
    #   "min_seed_fraction": 0.10,                               #"min_seed_fraction": 0.10,  
    #   "lr": 2e-4, 
    #   },
   
    
    ##--##
    
   #RANDOM VI
   "Random_Final": {
     "duration": 500, 
     "weights": [0.01, 0.79, 0.20, 0.0, 0,0,0,0],             #"weights": [0.01, 0.74, 0.25, 0.0, 0,0,0,0]
     "epsilon": 0.40,                                         #0.45
     "epsilon_min": 0.10,
     "memory_prune_low": 0.000,                               #nothing to prune
     "opponent": [0.40, 0.55, 0.05, 0.0, 0,0,0,0],            #"opponent": [0.35, 0.60, 0.05, 0.0, 0,0,0,0],
     "TU_mode": "soft", 
     "tau": 0.0040,                                           #"tau": 0.0045,
     "guard_prob": 0.70,                                      #"guard_prob": 0.73,
     "center_start": 0.95,                                    #"center_start": 0.90,
     "max_seed_fraction": 0.30,                              #"max_seed_fraction": 0.35
     "min_seed_fraction": 0.05,                               #"min_seed_fraction": 0.10,  
     "lr": 2e-4, 
     },
   
   ##--##
    
   # #RANDOM VI RS6(T0.0040 maxseed 0.50, E0.50)
   # "Random_Final": {
   #   "duration": 500, 
   #   "weights": [0.01, 0.74, 0.25, 0.0, 0,0,0,0],             #"weights": [0.01, 0.74, 0.25, 0.0, 0,0,0,0],
   #   "epsilon": 0.50,                                         #0.45
   #   "epsilon_min": 0.10,
   #   "memory_prune_low": 0.000,                               #nothing to prune
   #   "opponent": [0.35, 0.60, 0.05, 0.0, 0,0,0,0],            #"opponent": [0.30, 0.60, 0.10, 0.0, 0,0,0,0],
   #   "TU_mode": "soft", 
   #   "tau": 0.0040,                                           #"tau": 0.0045,
   #   "guard_prob": 0.70,                                      #"guard_prob": 0.73,
   #   "center_start": 0.90,                                    #"center_start": 0.90,
   #   "max_seed_fraction": 0.50,                              #"max_seed_fraction": 0.35
   #   "min_seed_fraction": 0.10,                               #"min_seed_fraction": 0.10,  
   #   "lr": 2e-4, 
   #   },
    
    # #RANDOM VI RS6(T0.0045 maxseed 0.40, E0.45)
    # "Random_Final": {
    #   "duration": 500, 
    #   "weights": [0.01, 0.74, 0.25, 0.0, 0,0,0,0],             #"weights": "weights": [0.05, 0.85, 0.10, 0.0, 0,0,0,0],
    #   "epsilon": 0.45,                                         #0.60
    #   "epsilon_min": 0.10,
    #   "memory_prune_low": 0.000,                               #nothing to prune
    #   "opponent": [0.30, 0.60, 0.10, 0.0, 0,0,0,0],            #opponent": [0.65, 0.30, 0.05, 0.0, 0,0,0,0],
    #   "TU_mode": "soft", 
    #   "tau": 0.0045,                                           #"tau": 0.0045,
    #   "guard_prob": 0.70,                                      #"guard_prob": 0.73,
    #   "center_start": 0.90,                                    #"center_start": 0.90,
    #   "max_seed_fraction": 0.40,                              #"max_seed_fraction": 0.35
    #   "min_seed_fraction": 0.10,                               #"min_seed_fraction": 0.10,  
    #   "lr": 2e-4, 
    #   },
}
