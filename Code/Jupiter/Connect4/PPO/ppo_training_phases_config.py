
TRAINING_PHASES = {
    
    "BootL1a": {
        "duration": 300, "lookahead": 1,
        "opponent_mix": {"L1": 0.85, "R": 0.10, "L2": 0.05},
        "params": {
            "lr": 1.6e-4, "clip": 0.22, "entropy": 0.012, "epochs": 4,
            "batch_size": 256, "temperature": 0.25, "steps_per_update": 4096
        }
    },
    "Rinse": {
        "duration": 100, "lookahead": None,
        "opponent_mix": {"R": 0.7, "L1": 0.3},
        "params": {
            "lr": 1.6e-4, "clip": 0.22, "entropy": 0.008, "epochs": 3,
            "batch_size": 256, "temperature": 0.8, "steps_per_update": 2048
        }
    },
    "BootL1b": {
        "duration": 300, "lookahead": 1,
        "opponent_mix": {"L1": 0.9, "R": 0.07, "L2": 0.03},
        "params": {
            "lr": 1.2e-4, "clip": 0.25, "entropy": 0.012, "epochs": 4,
            "batch_size": 256, "temperature": 0.2, "steps_per_update": 4096
        }
    },

   "L2a": {
       "duration": 100, "lookahead": 2,
       "opponent_mix": {"R":0.25, "L1":0.55, "L2":0.18, "SP":0.02},
       "params": { "lr":1.6e-4, "clip":0.10, "entropy":0.004, "epochs":4,
                   "batch_size":256, "temperature":0.05, "steps_per_update":4096 }
   },
   
   "L3a": {
       "duration": 100, "lookahead": 2,
       "opponent_mix": {"R":0.10, "L1":0.50, "L2":0.20, "SP":0.02, "L3": 0.18},
       "params": { "lr":1.6e-4, "clip":0.10, "entropy":0.004, "epochs":4,
                   "batch_size":256, "temperature":0.05, "steps_per_update":4096 }
   },
   
    "SPa": {
        "duration": 100, "lookahead": "self",
        "opponent_mix": {"SP":1.0},
        "params": { "lr":1.2e-4, "clip":0.08, "entropy":0.004, "epochs":3,
                    "batch_size":256, "temperature":0.06, "steps_per_update":4096 }
    },
}

    # "R0": {
    #     "duration": 100, "lookahead": None,       
    #     "opponent_mix": {"R":0.90, "L1":0.10},   
    #     "params": { "lr":3e-4, "clip":0.20, "entropy":0.010, "epochs":4,
    #                 "batch_size":256, "temperature":0.8, "steps_per_update":2048 }
    # },

    # "L1a": {
    #     "duration": 100, "lookahead": 1,
    #     "opponent_mix": {"R":0.40, "L1":0.60},
    #     "params": { "lr":2.0e-4, "clip":0.12, "entropy":0.006, "epochs":5,
    #                 "batch_size":256, "temperature":0.08, "steps_per_update":4096 }
    # },

    # "R1": {
    #     "duration": 100, "lookahead": None,
    #     "opponent_mix": {"R":0.70, "L1":0.25, "L2":0.05},
    #     "params": { "lr":2.0e-4, "clip":0.18, "entropy":0.008, "epochs":3,
    #                 "batch_size":256, "temperature":0.8, "steps_per_update":2048 }
    # },

    # "L1b": {
    #     "duration":1200, "lookahead": 1,
    #     "opponent_mix": {"R":0.35, "L1":0.55, "L2":0.10},
    #     "params": { "lr":1.8e-4, "clip":0.10, "entropy":0.005, "epochs":5,
    #                 "batch_size":256, "temperature":0.06, "steps_per_update":4096 }
    # },

    # "R2": {
    #     "duration": 100, "lookahead": None,
    #     "opponent_mix": {"R":0.55, "L1":0.35, "L2":0.10},
    #     "params": { "lr":1.8e-4, "clip":0.16, "entropy":0.008, "epochs":3,
    #                 "batch_size":256, "temperature":0.8, "steps_per_update":2048 }
    # },

    # "L2a": {
    #     "duration": 100, "lookahead": 2,
    #     "opponent_mix": {"R":0.25, "L1":0.55, "L2":0.18, "SP":0.02},
    #     "params": { "lr":1.6e-4, "clip":0.10, "entropy":0.004, "epochs":4,
    #                 "batch_size":256, "temperature":0.05, "steps_per_update":4096 }
    # },




