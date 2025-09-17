# training_phases_config.py
# [R, L1, L2, L3, L4, L5, L6, L7]


TRAINING_PHASES = {
    
    "Random_Clinic": {                
      "duration": 500, #500
      "weights": [0,0.75,0.25, 0,0,0,0,0], 
      "epsilon": 0.22, "epsilon_min": 0.05,
      "memory_prune_low": 0.00,         
      "opponent": [0.50,0.50,0.0,0,0,0,0,0], 
      "TU": 500, "TU_mode": "hard",
      "guard_prob": 0.50 ,   
      "center_start": 0.70, 
    },
    
    "L1_Clinic": {                
      "duration": 500, #500
      "weights": [0,0.65,0.30, 0.05, 0,0,0,0], 
      "epsilon": 0.22, "epsilon_min": 0.05,
      "memory_prune_low": 0.00,         
      "opponent": [0.40,0.60,0.0,0,0,0,0,0], 
      "TU": 500, "TU_mode": "hard","tau":0.006, 
      "guard_prob": 0.40 ,   
      "center_start": 0.60, 
    },
    
    "L1_Intro": {                
      "duration": 500, #500
      "weights": [0,0.60,0.35,0.05,0,0,0,0], 
      "epsilon": 0.21, "epsilon_min": 0.04,
      "memory_prune_low": 0.00,         
      "opponent": [0.20,0.70,0.10,0,0,0,0,0], 
      "TU": 500, "TU_mode": "hard",
      "guard_prob": 0.36 ,   
      "center_start": 0.57, 
    },
      
    "L1_Reintroduce": {                
      "duration": 600, #600
      "weights": [0,0.55,0.35,0.10,0,0,0,0], 
      "epsilon": 0.20, "epsilon_min": 0.04,
      "memory_prune_low": 0.00,         
      "opponent": [0.20,0.70,0.10,0,0,0,0,0], 
      "TU": 500, "TU_mode": "hard",
      "guard_prob": 0.32 ,   
      "center_start": 0.55, 
    },
    
    "L1_RampA": {                
      "duration": 500, #500
      "weights": [0,0.52,0.35,0.13,0,0,0,0], #[0,0.48,0.37,0.15,0,0,0,0], 
      "epsilon": 0.18, "epsilon_min": 0.04,
      "memory_prune_low": 0.006,         
      "opponent": [0.20,0.70,0.10,0,0,0,0,0], 
      "TU": 500, "TU_mode": "hard",
      "guard_prob": 0.31,   
      "center_start": 0.52, 
    },
    
    "L1_RampB": {                
      "duration": 500, #500
      "weights": [0,0.48,0.37,0.15,0,0,0,0], #[0,0.48,0.37,0.15,0,0,0,0], 
      "epsilon": 0.18, "epsilon_min": 0.04,
      "memory_prune_low": 0.006,         
      "opponent": [0.20,0.70,0.10,0,0,0,0,0], 
      "TU": 500, "TU_mode": "hard",
      "guard_prob": 0.305,   
      "center_start": 0.51, 
    },
    
    "L1_MainA": {                
      "duration": 700, #500
      "weights": [0, 0.45, 0.35, 0.20, 0,0,0,0], 
      "epsilon": 0.16, "epsilon_min": 0.04,
      "memory_prune_low": 0.006,       
      "opponent": [0.20, 0.75, 0.05,0,0,0,0,0], #[0.20, 0.75, 0.05,0,0,0,0,0], 
      "TU": 500, "TU_mode": "hard",
      "guard_prob": 0.30, 
      "center_start": 0.50, 
    },
    
    "L1_MainB": {                
      "duration": 700, #500
      "weights": [0, 0.45, 0.35, 0.20, 0,0,0,0], 
      "epsilon": 0.16, "epsilon_min": 0.04,
      "memory_prune_low": 0.006,       
      "opponent": [0.13, 0.82, 0.05,0,0,0,0,0], #[0.20, 0.75, 0.05,0,0,0,0,0], 
      "TU": 500, "TU_mode": "hard",
      "guard_prob": 0.30, 
      "center_start": 0.48, 
    },
      
    "L1_Bounce": {                      
      "duration": 500, 
      "weights": [0,0.40, 0.30, 0.30, 0,0,0,0], 
      "epsilon": 0.14, "epsilon_min": 0.03,
      "memory_prune_low": 0.005,        
      "opponent": [0.05,0.90,0.05,0,0,0,0,0], #[0.05,0.90,0.05,0,0,0,0,0],
      "TU": 500, "TU_mode": "hard", "tau": 0.008,
      "guard_prob": 0.30, 
      "center_start": 0.45, 
    },
    

    "L2_Intro": {                       
      "duration": 500, "weights": [0,0.40,0.30,0.30,0,0,0,0],
      "epsilon": 0.12, "epsilon_min": 0.02,
      "memory_prune_low": 0.005,        
      "opponent": [0.05,0.85,0.10,0,0,0,0,0],
      "TU": 500, "TU_mode": "hard",
      "guard_prob": 0.30,
      "center_start": 0.45,
    },

}



