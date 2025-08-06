# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 14:44:26 2025

@author: lovro
"""

TRAINING_PHASES = {
    "Random1": {
        "length": 500,
        "weights": [0.5, 0.45, 0.05, 0, 0, 0], #4
        "epsilon": 1.0,
        "memory_prune": 0.0,
    },
    # "Mixed": {
    #     "length": 1500,
    #     "weights": [0.25, 0.6, 0.15, 0, 0, 0], #8 
    #     "epsilon": 0.65, #best
    #     "memory_prune": 0.5, #best - start learning
    # },
    # "SelfPlay_Begin": {
    #     "length": 2000,
    #     "weights": [],
    #     "epsilon": 0.0,
    #     "memory_prune": 0.5,
    # },
    # "Mixed2": {
    #     "length": 2500,
    #     "weights": [0.2, 0.55, 0.25, 0, 0, 0], ##3 this wa increasing training
    #     "epsilon": 0.4,
    #     "memory_prune": 0.3,
    # },
    # "Fixed1": {
    #     "length": 3000,
    #     "weights": [0.2, 0.55, 0.25, 0, 0, 0], #prev
    #     "epsilon": 0.4,
    #     "memory_prune": 0.3,
    # },
    #  "Fixed1-Reprise": {
    #     "length": 3500,
    #     "weights": [0.1, 0.6, 0.3, 0, 0, 0], #1
    #     "epsilon": 0.4,
    #     "memory_prune": 0.3,
    # },
    
    # "SelfPlay_F1": {
    #     "length": 4000,
    #     "weights": [],
    #     "epsilon": 0.0,
    #     "memory_prune": 0.15,
    # },
    # "Mixed_R12": {
    #     "length": 4500,
    #     "weights": [0.1, 0.4, 0.5, 0, 0, 0], #2
    #     "epsilon": 0.3,
    #     "memory_prune": 0.4,
    # },
    # "Mixed_R1122": {
    #     "length": 5000,
    #     "weights": [0.05, 0.4, 0.5, 0.05, 0, 0], #2
    #     "epsilon": 0.3,
    #     "memory_prune": 0.4,
    },
    # "SelfPlay_L1L2": {
    #     "length": 5000,
    #     "weights": [],
    #     "epsilon": 0.0,
    #     "memory_prune": 0.2,
    # },
    # "Shallow": {
    #     "length": 5500,
    #     "weights": [0.25, 0.5, 0.25, 0.1, 0, 0],
    #     "epsilon": 0.6,
    #     "memory_prune": 0.2,
    # },
    # "SelfPlay_shallow2": {
    #     "length": 6000,
    #     "weights": [],
    #     "epsilon": 0.0,
    #     "memory_prune": 0.2,
    # },
    # "Fixed2": {
    #     "length": 6500,
    #     "weights": [0.2, 0.35, 0.35, 0.1, 0, 0],
    #     "epsilon": 0.5,
    #     "memory_prune": 0.1,
    # },
    # "SelfPlay_L2": {
    #     "length": 7000,
    #     "weights": [],
    #     "epsilon": 0.0,
    #     "memory_prune": 0.1,
    # },
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
