# training_phases_config_L2.py
# [R, L1, L2, L3, L4, L5, L6, L7]


    
TRAINING_PHASES = {

    # ---- buffer fill for the loaded model ----
    "SelfPlay_L1_init": {
        "duration": 500,                 
        "weights": None,
        "epsilon": 0.02, "epsilon_min": 0.02,
        "memory_prune_low": 0.00,        
        "opponent": None,
    },
    
    # ---- L2 ----
    
    "L2_Attack": {
        "duration": 750,                 
        "weights": [0.00, 0.10, 0.65, 0.25, 0, 0, 0, 0],   #65% L2, 25% unfair competition
        "epsilon": 0.60, "epsilon_min": 0.1,
        "memory_prune_low": 0.05,
        "opponent": [0.05, 0.50, 0.45, 0.00, 0, 0, 0, 0],     # 50% L1 45% L2
    },
    
      
    # --- L2 conclusion
    
    "SelfPlay_L2": {
        "duration": 500,                 
        "weights": None,
        "epsilon": 0.02, "epsilon_min": 0.02,
        "memory_prune_low": 0.00, #no sense pruning last replay
        "opponent": None,
    },
    
    #### final - guard against off by one
    "Final": {
        "duration": 1,                 
        "weights": None,
        "epsilon": 0.01, "epsilon_min": 0.01,
        "memory_prune_low": 0.00, #no sense pruning last replay
        "opponent": None,
    },
}
