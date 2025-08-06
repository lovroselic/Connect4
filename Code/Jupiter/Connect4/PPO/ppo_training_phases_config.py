# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 09:17:31 2025

@author: Lovro
"""

# ppo_training_phases_config.py
# memory prune not yes applied

TRAINING_PHASES = {
    "Random": {
        "length": 500,
        "lookahead": None,  # Random opponent
        "memory_prune": 0.0,
    },
    # "Fixed1": {
    #     "length": 1500,
    #     "lookahead": 1,
    #     "memory_prune": 0.0,
    # },
    # "Fixed2": {
    #     "length": 2500,
    #     "lookahead": 2,
    #     "memory_prune": 0.0,
    },
    # "Fixed3": {
    #     "length": 3500,
    #     "lookahead": 3,
    #     "memory_prune": 0.0,
    # },
    # "SelfPlay3": {
    #     "length": 4000,
    #     "lookahead": None,  # Use model as opponent
    #     "memory_prune": 0.0,
    # },


    ## ---- FINAL ---- ##
    "Final": {
        "length": None,
        "lookahead": None,
        "memory_prune": 0.0,
    }
}
