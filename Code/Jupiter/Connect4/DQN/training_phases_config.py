# training_phases_config.py
# [R, L1, L2, L3, L4, L5, L6, L7]


TRAINING_PHASES = {
    

    
    "Random_Intro": {
      "duration": 500, 
      "weights": [0.60, 0.40, 0.0, 0.0, 0,0,0,0],
      "epsilon": 0.50, 
      "epsilon_min": 0.10,
      "memory_prune_low": 0.000,                            #nothing to prune
      "opponent": [0.95, 0.05, 0.0, 0.0, 0,0,0,0],
      "TU_mode": "soft", 
      "tau": 0.0030,                                        #"tau": 0.0030,
      "guard_prob": 0.75, 
      "center_start": 0.90,
      "max_seed_fraction": 0.05,                            #"max_seed_fraction": 0.04
      "min_seed_fraction": 0.00,                            #min_seed_fraction": 0.00,  
      "lr": 2e-4, 
      }


    # "Random_Intro": {
    #   "duration": 500, 
    #   "weights": [0.60, 0.40, 0.0, 0.0, 0,0,0,0],
    #   "epsilon": 0.50, 
    #   "epsilon_min": 0.10,
    #   "memory_prune_low": 0.000,                            #nothing to prune
    #   "opponent": [0.95, 0.05, 0.0, 0.0, 0,0,0,0],
    #   "TU_mode": "soft", 
    #   "tau": 0.0030,                                        #"tau": 0.0030,
    #   "guard_prob": 0.75, 
    #   "center_start": 0.90,
    #   "max_seed_fraction": 0.04,                            #"max_seed_fraction": 0.04
    #   "min_seed_fraction": 0.00,                            #min_seed_fraction": 0.00,  
    #   "lr": 2e-4, 
    #   }
    
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
    
    

}
