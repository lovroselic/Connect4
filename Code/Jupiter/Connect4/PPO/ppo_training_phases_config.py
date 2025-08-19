# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 09:17:31 2025

@author: Lovro
"""

# ppo_training_phases_config.py

TRAINING_PHASES = {
    "Random1": {
        "duration": 500,
        "lookahead": None,
        "params": {"lr": 3e-4, "clip": 0.20, "entropy": 0.010, "epochs": 4},
    },
    "SelfPlay_Random": {
        "duration": 500,
        "lookahead": "self",
        "params": {"lr": 1.5e-4, "clip": 0.12, "entropy": 0.007, "epochs": 4},
    },
    "L1_Intro": {
        "duration": 750,
        "lookahead": 1,
        "params": {"lr": 3e-4, "clip": 0.20, "entropy": 0.010, "epochs": 4},
    },
    "Random2": {
        "duration": 250,
        "lookahead": None,
        "params": {"lr": 3e-4, "clip": 0.20, "entropy": 0.010, "epochs": 4},
    },
    "L1_Consol": {
        "duration": 1000,
        "lookahead": 1,
        "params": {"lr": 2e-4, "clip": 0.18, "entropy": 0.009, "epochs": 4},
    },
    "Random3": {
        "duration": 250,
        "lookahead": None,
        "params": {"lr": 2e-4, "clip": 0.18, "entropy": 0.009, "epochs": 4},
    },
    "L2_Intro": {
        "duration": 750,
        "lookahead": 2,
        "params": {"lr": 2e-4, "clip": 0.15, "entropy": 0.008, "epochs": 4},
    },
    "Random4": {
        "duration": 250,
        "lookahead": None,
        "params": {"lr": 2e-4, "clip": 0.15, "entropy": 0.008, "epochs": 4},
    },
    "L2_Consol": {
        "duration": 1000,
        "lookahead": 2,
        "params": {"lr": 1.5e-4, "clip": 0.12, "entropy": 0.007, "epochs": 4},
    },
    "SelfPlay_L1L2": {
        "duration": 500,
        "lookahead": "self",
        "params": {"lr": 1.5e-4, "clip": 0.12, "entropy": 0.007, "epochs": 4},
    },
    "L3_Intro": {
        "duration": 500,
        "lookahead": 3,
        "params": {"lr": 1e-4, "clip": 0.10, "entropy": 0.006, "epochs": 4},
    },
    "Random5": {
        "duration": 250,
        "lookahead": None,
        "params": {"lr": 1e-4, "clip": 0.10, "entropy": 0.006, "epochs": 4},
    },
    "L3_Consol": {
        "duration": 1000,
        "lookahead": 3,
        "params": {"lr": 1e-4, "clip": 0.10, "entropy": 0.006, "epochs": 4},
    },
}

