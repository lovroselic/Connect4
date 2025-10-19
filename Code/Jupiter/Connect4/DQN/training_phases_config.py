# training_phases_config.py
# [R, L1, L2, L3, L4, L5, L6, L7]


TRAINING_PHASES = {
    
    "L1_Intro_B": {
      "duration": 200, 
      "weights": [0.01, 0.19, 0.40, 0.40, 0,0,0,0],
      "epsilon": 0.40, 
      "epsilon_min": 0.10,
      "memory_prune_low": 0.000, 
      "opponent": [0.10, 0.50, 0.35, 0.05, 0,0,0,0],
      "TU_mode": "soft", 
      "tau": 0.0024, 
      "guard_prob": 0.35,       
      "center_start": 0.40,     
      "seed_fraction": 0.02,    
      "lr": 2e-4, 
    }, 
    
    "L1_Continue_B": {
      "duration": 500, 
      "weights": [0.01, 0.09, 0.50, 0.40, 0,0,0,0],
      "epsilon": 0.37, 
      "epsilon_min": 0.08,
      "memory_prune_low": 0.001, 
      "opponent": [0.075, 0.45, 0.40, 0.075, 0,0,0,0],
      "TU_mode": "soft", 
      "tau": 0.0023,  
      "guard_prob": 0.33,       
      "center_start": 0.40,     
      "seed_fraction": 0.01,    
      "lr": 1.6e-4, 
    }, 
    
    "L1_Late": {
      "duration": 500, 
      "weights": [0.01, 0.04, 0.50, 0.45, 0,0,0,0],
      "epsilon": 0.36, 
      "epsilon_min": 0.08,
      "memory_prune_low": 0.001, 
      "opponent": [0.05, 0.40, 0.42, 0.13, 0,0,0,0], 
      "TU_mode": "soft", 
      "tau": 0.0023,  
      "guard_prob": 0.30,       
      "center_start": 0.35,     
      "seed_fraction": 0.01,    
      "lr": 1.55e-4, 
    }, 
    
    "L1_End": {
      "duration": 500, 
      "weights": [0.01, 0.01, 0.50, 0.48, 0,0,0,0],
      "epsilon": 0.36, 
      "epsilon_min": 0.08,
      "memory_prune_low": 0.001, 
      "opponent": [0.05, 0.35, 0.45, 0.15, 0,0,0,0], 
      "TU_mode": "soft", 
      "tau": 0.0022,  
      "guard_prob": 0.28,       
      "center_start": 0.32,     
      "seed_fraction": 0.01,    
      "lr": 1.53e-4, 
    }, 
    

}
