# training_phases_config.py
# [R, L1, L2, L3, L4, L5, L6, L7]


TRAINING_PHASES = {
    
    # RANDOM V
    "L1_VS_RANDOM": {
      "duration": 400, #380
      "weights": [0.05, 0.90, 0.05, 0.0, 0,0,0,0],
      "epsilon": 0.45, 
      "epsilon_min": 0.10,
      "memory_prune_low": 0.000, 
      "opponent": [0.65, 0.35, 0.0, 0.0, 0,0,0,0], 
      "TU_mode": "soft", 
      "tau": 0.0027,               
      "guard_prob": 0.75, 
      "center_start": 0.88,
      "seed_fraction": 0.02,
      "lr": 20e-5,  
    },  
    
    # RANDOM XIV
    
    # "L1_Random_Explorer_A": {
    #   "duration": 500, 
    #   "weights": [0.01, 0.90, 0.09, 0.0, 0,0,0,0],
    #   "epsilon": 0.45,              #0.45 
    #   "epsilon_min": 0.10,          #0.10
    #   "memory_prune_low": 0.000, 
    #   "opponent": [0.60, 0.40, 0.0, 0.0, 0,0,0,0],  
    #   "TU_mode": "soft", 
    #   "tau": 0.0022,                #0.0022,
    #   "guard_prob": 0.80,           #0.80
    #   "center_start": 0.90,         #0.90
    #   "seed_fraction": 0.02,        # 0.02
    #   "lr": 19e-5,                  #19e-5,  
    # },  
    
    # "L1_Random_Explorer_B": {
    #   "duration": 500, 
    #   "weights": [0.01, 0.90, 0.09, 0.0, 0,0,0,0],
    #   "epsilon": 0.45,              #0.45 
    #   "epsilon_min": 0.10,          #0.10
    #   "memory_prune_low": 0.002, 
    #   "opponent": [0.50, 0.50, 0.00, 0.00, 0,0,0,0, 0],
    #   "TU_mode": "soft", 
    #   "tau": 0.0022,                #0.0022,
    #   "guard_prob": 0.80,           #0.80
    #   "center_start": 0.90,         #0.90
    #   "seed_fraction": 0.02,        # 0.02
    #   "lr": 19e-5,                  #19e-5,  
    # },  
    
    

    

}
