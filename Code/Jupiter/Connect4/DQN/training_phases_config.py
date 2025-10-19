# training_phases_config.py
# [R, L1, L2, L3, L4, L5, L6, L7]


TRAINING_PHASES = {
    
    # RANDOM II
    "Random_Continue": {
      "duration": 500, 
      "weights": [0.30, 0.70, 0.0, 0.0, 0,0,0,0],
      "epsilon": 0.50, "epsilon_min": 0.10,
      "memory_prune_low": 0.000, 
      "opponent": [0.80, 0.20, 0.0, 0.0, 0,0,0,0],
      "TU_mode": "soft", 
      "tau": 0.0029,
      "guard_prob": 0.75, 
      "center_start": 0.90,
      "seed_fraction": 0.0,
      "lr": 20e-5, 
    },    
    
    # RANDOM III
    "Random_More": {
      "duration": 500, 
      "weights": [0.15, 0.80, 0.05, 0.0, 0,0,0,0],
      "epsilon": 0.45, "epsilon_min": 0.10,
      "memory_prune_low": 0.001, 
      "opponent": [0.70, 0.30, 0.0, 0.0, 0,0,0,0],
      "TU_mode": "soft", 
      "tau": 0.0028,
      "guard_prob": 0.75, 
      "center_start": 0.90,
      "seed_fraction": 0.0,
      "lr": 20e-5,  
    },
    

}
