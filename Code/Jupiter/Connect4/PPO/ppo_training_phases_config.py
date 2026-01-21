
# PPO/ppo_training_phases_config.py

# ### exploration
# TRAINING_PHASES = {
#     "EXPLORE": {
#         "duration": 500,
#         "opponent_mix": {
#             "L1": 0.15,
#             "L3": 0.25,
#             "L5": 0.25,
#             "L7": 0.20,
#             "L9": 0.10,
#             "L11": 0.005,
#             "L13": 0.005,
#             "POP": 0.02,
#             "SP": 0.02,
#         },
#         "params": {
#             "lr": 3.0e-4,
#             "clip": 0.30,
#             "entropy": 0.020,
#             "epochs": 2,
#             "batch_size": 256,
#             "steps_per_update": 512,
#             "vf_clip": 0.20,
#             "max_grad_norm": 0.5,
#             "target_kl": 0.03,
#             "temperature": 0.92,
#         },
#     },
# }

TRAINING_PHASES = {

    "Random": {
        "duration": 50,
        "opponent_mix": {
            "POP": 0.05,
            "SP": 0.05,
            "R": 0.80,
            "L1": 0.10,
        },
        "params": {
            "lr": 3.0e-4,
            "clip": 0.30,
            "entropy": 0.020,
            "epochs": 2,
            "batch_size": 256,
            "steps_per_update": 512,
            "vf_clip": 0.20,
            "max_grad_norm": 0.5,
            "target_kl": 0.03,
            "temperature": 0.92,
        },
    },

    "POP_intro": {
        "duration": 50,
        "opponent_mix": {
            "POP": 0.60,
            "SP": 0.30,
            "R": 0.05,
            "L1": 0.05, 
        },
    },

    "L1_Intro": {
        "duration": 100,
        "opponent_mix": {
            "POP": 0.20,
            "SP": 0.20,
            "L1": 0.60,
        },
    },

    "L1L2L3": {
        "duration": 100,
        "opponent_mix": {
            "L1": 0.20,
            "L2": 0.20,
            "L3": 0.40,
            "POP": 0.10,
            "SP": 0.10
        },
    },
    
    "Focused_POP_SP_1": {
        "duration": 100,
        "opponent_mix": {
            "L1": 0.03,
            "L3": 0.05,
            "L5": 0.05,
            "L7": 0.05,
            "L9": 0.05,
            "L11": 0.04,
            "L13": 0.03,
            "POP": 0.35,
            "SP": 0.35,
        },
    },

    "L2L3L4": {
        "duration": 100,
        "opponent_mix": {
            "L2": 0.20,
            "L3": 0.20,
            "L4": 0.40,
            "POP": 0.10,
            "SP": 0.10
        },
    },

    "L3": {
        "duration": 100,
        "opponent_mix": {
            "L1": 0.20,
            "L3": 0.60,
            "POP": 0.10,
            "SP": 0.10
        },
    },

    "L3L4L5": {
        "duration": 200,
        "opponent_mix": {
            "L1": 0.05,
            "L3": 0.20,
            "L4": 0.20,
            "L5": 0.35,
            "POP": 0.10,
            "SP": 0.10
        },
    },
    
    "Focused_POP_SP_2": {
        "duration": 100,
        "opponent_mix": {
            "L1": 0.03,
            "L3": 0.05,
            "L5": 0.05,
            "L7": 0.05,
            "L9": 0.05,
            "L11": 0.04,
            "L13": 0.03,
            "POP": 0.35,
            "SP": 0.35,
        },
    },

    "L5": {
        "duration": 200,
        "opponent_mix": {
            "L1": 0.10,
            "L3": 0.20,
            "L5": 0.50,
            "POP": 0.10,
            "SP": 0.10
        },
    },

    "L4L5L6": {
        "duration": 100,
        "opponent_mix": {
            "L1": 0.025,
            "L2": 0.025,
            "L4": 0.20,
            "L5": 0.20,
            "L6": 0.35,
            "POP": 0.10,
            "SP": 0.10
        },
    },

    "L5L6L7": {
        "duration": 200,
        "opponent_mix": {
            "L1": 0.025,
            "L3": 0.025,
            "L5": 0.20,
            "L6": 0.20,
            "L7": 0.35,
            "POP": 0.10,
            "SP": 0.10
        },
    },

    "L6L7L8": {
        "duration": 100,
        "opponent_mix": {
            "L1": 0.025,
            "L3": 0.025,
            "L6": 0.20,
            "L7": 0.20,
            "L8": 0.35,
            "POP": 0.10,
            "SP": 0.10
        },
    },
    
    "Focused_POP_SP_3": {
        "duration": 100,
        "opponent_mix": {
            "L1": 0.03,
            "L3": 0.05,
            "L5": 0.05,
            "L7": 0.05,
            "L9": 0.05,
            "L11": 0.04,
            "L13": 0.03,
            "POP": 0.35,
            "SP": 0.35,
        },
    },

    "L7": {
        "duration": 200,
        "opponent_mix": {
            "L1": 0.05,
            "L3": 0.05,
            "L5": 0.20,
            "L7": 0.50,
            "POP": 0.10,
            "SP": 0.10
        },
    },
    
    "L7L8L9": {
        "duration": 200,
        "opponent_mix": {
            "L1": 0.01,
            "L3": 0.02,
            "L5": 0.02,
            "L7": 0.20,
            "L8": 0.20,
            "L9": 0.35,
            "POP": 0.10,
            "SP": 0.10
        },
    },

    "L9L10L11": {
        "duration": 200,
        "opponent_mix": {
            "L1": 0.01,
            "L3": 0.02,
            "L5": 0.02,
            "L7": 0.10,
            "L9": 0.20,
            "L10": 0.20,
            "L11": 0.25,
            "POP": 0.10,
            "SP": 0.10
        },
    },
    
    "Focused_POP_SP_4": {
        "duration": 100,
        "opponent_mix": {
            "L1": 0.03,
            "L3": 0.05,
            "L5": 0.05,
            "L7": 0.05,
            "L9": 0.05,
            "L11": 0.04,
            "L13": 0.03,
            "POP": 0.35,
            "SP": 0.35,
        },
    },

    "L11L12L13": {
        "duration": 100,
        "opponent_mix": {
            "L1": 0.01,
            "L3": 0.02,
            "L5": 0.02,
            "L7": 0.10,
            "L9": 0.10,
            "L11": 0.20,
            "L12": 0.20,
            "L13": 0.25,
            "POP": 0.05,
            "SP": 0.05
        },
    },

    "Finale_A": {
        "duration": 100,
        "opponent_mix": {
            "L1": 0.01,
            "L3": 0.05,
            "L5": 0.10,
            "L7": 0.20,
            "L9": 0.20,
            "L11": 0.20,
            "L13": 0.20,
            "POP": 0.01,
            "SP": 0.03,
        },
        "params": {
            "lr": 2.9e-4,
            "temperature": 0.91,
        },
    },

    "Finale_B": {
        "duration": 200,
        "opponent_mix": {
            "L1": 0.01,
            "L3": 0.08,
            "L5": 0.13,
            "L7": 0.18,
            "L9": 0.15,
            "L11": 0.15,
            "L13": 0.15,
            "SP": 0.15,
        },
        "params": {
            "lr": 2.8e-4,
            "entropy": 0.021,
            "temperature": 0.90,
        },
    },
}

###############################################################################################################

# TRAINING_PHASES = {
#     "Focused_POP_SP": {
#         "duration": 1000,
#         "opponent_mix": {
#             "L1": 0.03,
#             "L3": 0.05,
#             "L5": 0.05,
#             "L7": 0.05,
#             "L9": 0.05,
#             "L11": 0.04,
#             "L13": 0.03,
#             "POP": 0.35,
#             "SP": 0.35,
#         },
#         "params": {
#             "lr": 3.0e-4,
#             "clip": 0.30,
#             "entropy": 0.020,
#             "epochs": 2,
#             "batch_size": 256,
#             "steps_per_update": 512,
#             "vf_clip": 0.20,
#             "max_grad_norm": 0.5,
#             "target_kl": 0.03,
#             "temperature": 0.92,
#         },
#     },
# }


###############################################################################################################
