# training_phases_config.py
# [R, L1, L2, L3, L4, L5, L6, L7]




TRAINING_PHASES = {
    
    "Random_Intro": {
      "duration": 500, 
      "weights": [0.80, 0.20, 0.0, 0.0, 0,0,0,0],
      "epsilon": 0.50, "epsilon_min": 0.10,
      "memory_prune_low": 0.003, #nothing to prune
      "opponent": [0.95, 0.05, 0.0, 0.0, 0,0,0,0],
      "TU": 500, "TU_mode": "soft", "tau": 0.010,
      "guard_prob": 0.75, "center_start": 0.90,
    },    
    
    # "L1_Quick": {
    #   "duration": 300, #300
    #   "weights": [0.01, 0.40, 0.20, 0.39, 0,0,0,0],
    #   "epsilon": 0.10, "epsilon_min": 0.05,
    #   "memory_prune_low": 0.003,
    #   "opponent": [0.15, 0.5, 0.30, 0.05, 0,0,0,0],
    #   "TU": 500, "TU_mode": "soft", "tau": 0.010, #"tau": 0.0008, 0.0012
    #   "guard_prob": 0.40, "center_start": 0.70,
    # },    

    # "L2_Intro": {
    #   "duration": 500, #700
    #   "weights": [0.01, 0.20, 0.40, 0.39, 0,0,0,0],
    #   "epsilon": 0.20, "epsilon_min": 0.05,
    #   "memory_prune_low": 0.003,
    #   "opponent": [0.05, 0.60, 0.30, 0.05, 0,0,0,0],
    #   "TU": 500, "TU_mode": "soft", "tau": 0.0009, #"tau": 0.0008, 0.0012
    #   "guard_prob": 0.38, "center_start": 0.66,
    # },    
    
    # "L2_Continue": {
    #   "duration": 700, #700
    #   "weights": [0.01, 0.14, 0.35, 0.50, 0,0,0,0],
    #   "epsilon": 0.20, "epsilon_min": 0.05,
    #   "memory_prune_low": 0.003,
    #   "opponent": [0.05, 0.40, 0.50, 0.05, 0,0,0,0],
    #   "TU": 500, "TU_mode": "soft", "tau": 0.0008,
    #   "guard_prob": 0.38, "center_start": 0.66,
    # },    
    
    # "L3_Intro": {
    #   "duration": 700, #700
    #   "weights": [0.01, 0.04, 0.25, 0.70, 0,0,0,0],
    #   "epsilon": 0.20, "epsilon_min": 0.05,
    #   "memory_prune_low": 0.003,
    #   "opponent": [0.05, 0.35, 0.40, 0.20, 0,0,0,0],
    #   "TU": 500, "TU_mode": "soft", "tau": 0.0008,
    #   "guard_prob": 0.38, "center_start": 0.66,
    # },    
    
  # "L3_Continue": {
  #   "duration": 700, #700
  #   "weights": [0.01, 0.04, 0.15, 0.80, 0,0,0,0],
  #   "epsilon": 0.20, "epsilon_min": 0.05,
  #   "memory_prune_low": 0.003,
  #   "opponent": [0.05, 0.30, 0.35, 0.30, 0,0,0,0],
  #   "TU": 500, "TU_mode": "soft", "tau": 0.0008,
  #   "guard_prob": 0.38, "center_start": 0.66,
  # },    

  # "L3_Continue2": {
  #   "duration": 700, #700
  #   "weights": [0.01, 0.04, 0.05, 0.90, 0,0,0,0],
  #   "epsilon": 0.20, "epsilon_min": 0.05,
  #   "memory_prune_low": 0.003,
  #   "opponent": [0.05, 0.25, 0.30, 0.40, 0,0,0,0],
  #   "TU": 500, "TU_mode": "soft", "tau": 0.0008,
  #   "guard_prob": 0.38, "center_start": 0.66,
  # },

  # "L4_Intro": {
  #   "duration": 700,    #700
  #   "weights": [0.01, 0.04, 0.05, 0.80, 0.10,0,0,0],
  #   "epsilon": 0.20, "epsilon_min": 0.05,
  #   "memory_prune_low": 0.003,
  #   "opponent": [0.05, 0.25, 0.30, 0.40, 0,0,0,0],
  #   "TU": 500, "TU_mode": "soft", "tau": 0.0007,
  #   "guard_prob": 0.35, "center_start": 0.65,
  # },
  
}
