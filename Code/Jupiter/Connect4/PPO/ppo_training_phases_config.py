TRAINING_PHASES = {
    "Random1": {
        "duration": 300, "lookahead": None,
        "params": {"lr": 3e-4, "clip": 0.20, "entropy": 0.012,
                   "epochs": 4, "temperature": 1.0, "steps_per_update": 2048},
    },

    "L1_Intro": {
        "duration": 1200, "lookahead": 1,
        "params": {"lr": 2.5e-4, "clip": 0.15, "entropy": 0.025,
                   "epochs": 5, "temperature": 0.60, "steps_per_update": 4096},
    },

    "Random2": {
        "duration": 150, "lookahead": None,
        "params": {"lr": 2.5e-4, "clip": 0.18, "entropy": 0.010,
                   "epochs": 4, "temperature": 0.8, "steps_per_update": 2048},
    },

    "L1_Consol": {
        "duration": 1200, "lookahead": 1,
        "params": {"lr": 2.0e-4, "clip": 0.13, "entropy": 0.015,
                   "epochs": 5, "temperature": 0.40, "steps_per_update": 4096},
    },

    "Random3": {
        "duration": 150, "lookahead": None,
        "params": {"lr": 2.0e-4, "clip": 0.18, "entropy": 0.010,
                   "epochs": 4, "temperature": 0.8, "steps_per_update": 2048},
    },

    "L2_Intro": {
        "duration": 1200, "lookahead": 2,
        "params": {"lr": 2.0e-4, "clip": 0.13, "entropy": 0.015,
                   "epochs": 5, "temperature": 0.40, "steps_per_update": 4096},
    },

    "L1_Spar_AfterL2Intro": {
        "duration": 300, "lookahead": 1,
        "params": {"lr": 2.0e-4, "clip": 0.13, "entropy": 0.012,
                   "epochs": 4, "temperature": 0.35, "steps_per_update": 4096},
    },

    "Random4": {
        "duration": 150, "lookahead": None,
        "params": {"lr": 2.0e-4, "clip": 0.15, "entropy": 0.010,
                   "epochs": 4, "temperature": 0.8, "steps_per_update": 2048},
    },

    "L2_Consol": {
        "duration": 1500, "lookahead": 2,
        "params": {"lr": 1.5e-4, "clip": 0.12, "entropy": 0.010,
                   "epochs": 5, "temperature": 0.30, "steps_per_update": 12288},
    },

    "L1_Spar_AfterL2Consol": {
        "duration": 300, "lookahead": 1,
        "params": {"lr": 1.5e-4, "clip": 0.12, "entropy": 0.010,
                   "epochs": 4, "temperature": 0.30, "steps_per_update": 8192},
    },

    "SelfPlay_L1L2": {
        "duration": 1500, "lookahead": "self",
        "params": {"lr": 1.5e-4, "clip": 0.12, "entropy": 0.006,
                   "epochs": 4, "temperature": 0.20, "steps_per_update": 8192},
    },
}
