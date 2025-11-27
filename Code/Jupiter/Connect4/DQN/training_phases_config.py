# training_phases_config.py
# [R, L1, L2, L3, L4, L5, L6, L7]



TRAINING_PHASES = {
    
    
    "Random_Push": {
      "duration": 500, 
      "weights": None,             
      "epsilon": 0.50,                                                          # "epsilon": 0.50,                                      
      "epsilon_min": 0.10,                                                      # "epsilon_min": 0.10,                                            
      "memory_prune_low": 0.000,                                                
      "opponent": [0.25, 0.40, 0.10, 0.15, 0.0, 0.0, 0.0, 0.0 ],                # "opponent": [0.70, 0.20, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0 ],
      "TU_mode": "soft", 
      "guard_prob": 0.75,                                                       # "guard_prob": 0.80, 
      "center_start": 0.90,                                                     # center_start": 0.95,           
      "win_now_prob": 0.95,                                                     # "win_now_prob": 0.95              
      "max_seed_fraction": 0.50,                                                #"max_seed_fraction": 0.75,                                            
      "min_seed_fraction": 0.10,                                                #min_seed_fraction": 0.25,                                      
      "tau": 35e-4,                                                             # "tau": 40e-5,       
      "lr": 10e-5,                                                              # "lr": 10e-5,                                                                                  
      },
    

}
