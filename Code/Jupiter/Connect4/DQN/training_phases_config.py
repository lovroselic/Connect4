# training_phases_config.py
# [R, L1, L2, L3, L4, L5, L6, L7]


TRAINING_PHASES = {
    
    
    # RANDOM VI
    "L1_Random_Explorer_A": {
      "duration": 600, #600 
      "weights": [0.02, 0.85, 0.10, 0.03, 0,0,0,0],
      "epsilon": 0.45, 
      "epsilon_min": 0.09,          #0.10, 0.11 
      "memory_prune_low": 0.000, 
      "opponent": [0.60, 0.40, 0.0, 0.0, 0,0,0,0], 
      "TU_mode": "soft", 
      "tau": 0.0040,                    #0.0040     
      "guard_prob": 0.75,               #0.75
      "center_start": 0.75,             #0.85
      "max_seed_fraction": 0.04,        #0.04       
      "min_seed_fraction": 0.00,        #0.00       
      "lr": 20e-5,  
    },  
    
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
 

}
