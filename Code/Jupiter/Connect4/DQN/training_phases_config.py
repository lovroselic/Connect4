# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 14:44:26 2025

@author: lovro
"""

TRAINING_PHASES = {
    # Random
    "Random1": {
        "length": 500,
        "weights": [0.5, 0.45, 0.05, 0, 0, 0], 
        "epsilon": 1.0,
        "memory_prune": 0.0,
    },
    "Random2": {
        "length": 650,
        "weights": [0.35, 0.55, 0.1, 0, 0, 0], 
        "epsilon": 1.0,
        "memory_prune": 0.0,
    },

    #Mixed R L1
    "Mixed_RR1": {
        "length": 1000,
        "weights": [0.3, 0.6, 0.1, 0, 0, 0], 
        "epsilon": 0.9, 
        "memory_prune": 0.1,
    },
    "Mixed_R1": {
        "length": 1250,
        "weights": [0.25, 0.6, 0.15, 0, 0, 0], #8 
        "epsilon": 0.65,
        "memory_prune": 0.4, 
    },
    "Mixed_R11": {
        "length": 1500,
        "weights": [0.2, 0.6, 0.2, 0, 0, 0], #8 
        "epsilon": 0.65,
        "memory_prune": 0.4, 
    },
        
    "SelfPlay_Begin": {
        "length": 2000,
        "weights": [],
        "epsilon": 0.0,
        "memory_prune": 0.4,
    },

    #L1
    "Fixed1": {
        "length": 2500,
        "weights": [0.2, 0.55, 0.25, 0, 0, 0], 
        "epsilon": 0.4,
        "memory_prune": 0.3,
    },
     "Fixed1-Reprise": {
        "length": 3000,
        "weights": [0.1, 0.6, 0.3, 0, 0, 0], 
        "epsilon": 0.4,
        "memory_prune": 0.3,
    },
    
    "SelfPlay_F1": {
        "length": 3500,
        "weights": [],
        "epsilon": 0.0,
        "memory_prune": 0.3,
    },

    #Mixed l2
    
    "Mixed_R12": {
        "length": 4000,
        "weights": [0.1, 0.4, 0.5, 0, 0, 0], 
        "epsilon": 0.4,
        "memory_prune": 0.25,
    },
    "Mixed_R1122": {
        "length": 4500,
        "weights": [0.05, 0.4, 0.5, 0.05, 0, 0],
        "epsilon": 0.4,
        "memory_prune": 0.25,
    },

    "Mixed12": {
        "length": 5000,
        "weights": [0.05, 0.4, 0.5, 0.05, 0, 0],
        "epsilon": 0.4,
        "memory_prune": 0.2,
    },

    "SelfPlay_L1L2": {
        "length": 5500,
        "weights": [],
        "epsilon": 0.0,
        "memory_prune": 0.15,
    },

    # L2

    "Mixed123": {
        "length": 6000,
        "weights": [0.05, 0.3, 0.45, 0.2, 0, 0],
        "epsilon": 0.4,
        "memory_prune": 0.2,
    },

    "Fixed2": {
        "length": 6250,
        "weights": [0.05, 0.25, 0.50, 0.2, 0, 0],
        "epsilon": 0.4,
        "memory_prune": 0.15,
    },

    "Fixed2_Reprise": {
        "length": 6500,
        "weights": [0.05, 0.25, 0.45, 0.25, 0, 0],
        "epsilon": 0.4,
        "memory_prune": 0.15,
    },
    
    "SelfPlay_L2": {
        "length": 7000,
        "weights": [],
        "epsilon": 0.0,
        "memory_prune": 0.1,
    },

    #L3

    # "Variable23": {
    #     "length": 7500,
    #     "weights": [0.15, 0.25, 0.6, 0.15, 0, 0],
    #     "epsilon": 0.5,
    #     "memory_prune": 0.1,
    # },
    # "Variable3": {
    #     "length": 8000,
    #     "weights": [0.15, 0.25, 0.6, 0.15, 0, 0],
    #     "epsilon": 0.5,
    #     "memory_prune": 0.1,
    # },
    # "SelfPlay_L3": {
    #     "length": 8500,
    #     "weights": [],
    #     "epsilon": 0.0,
    #     "memory_prune": 0.05,
    # },

    ## ---- FINAL ---- ##
    
    "Final": {
        "length": None,
        "weights": [],
        "epsilon": 0.0,
        "memory_prune": 0.0,
    }
}
