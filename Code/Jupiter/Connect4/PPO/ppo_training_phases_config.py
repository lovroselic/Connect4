# PPO/ppo_training_phases_config.py

TRAINING_PHASES = {

    "POP": {
        "duration": 300,
        "opponent_mix": {
            "POP": 0.65,
            "SP": 0.30,
            "R": 0.05,
        },
        "params": {
            "lr": 1.2e-4,
            "clip": 0.20,
            "entropy": 0.012,
            "epochs": 3,
            "batch_size": 256,
            "steps_per_update": 256, #256
            "vf_clip": 0.20,
            "max_grad_norm": 0.5,
            "target_kl": 0.02,
            "temperature": 1.03,
            
            #deprecated
            "center_start": 0.0, "guard_prob": 0.0, "win_now_prob": 0.0,
            "guard_ply_min": 5, "guard_ply_max": 13,
            "distill_coef": 0.0, "mentor_depth": 1, "mentor_prob": 0.0, "mentor_coef": 0.0
        },
    },
    
    "L1L2L3": {
        "duration": 300,
        "opponent_mix": {
            "L1": 0.20,
            "L2": 0.20,
            "L3": 0.40,
            "POP": 0.10,
            "SP": 0.10
        },
        "params": {
            "lr": 1.2e-4,
            "clip": 0.20,
            "entropy": 0.012,
            "epochs": 3,
            "batch_size": 256,
            "steps_per_update": 256, #256
            "vf_clip": 0.20,
            "max_grad_norm": 0.5,
            "target_kl": 0.02,
            "temperature": 1.03,
            
            #deprecated
            "center_start": 0.0, "guard_prob": 0.0, "win_now_prob": 0.0,
            "guard_ply_min": 5, "guard_ply_max": 13,
            "distill_coef": 0.0, "mentor_depth": 1, "mentor_prob": 0.0, "mentor_coef": 0.0
        },
    },
    
    "L3L4L5": {
        "duration": 300,
        "opponent_mix": {
            "L1": 0.05,
            "L3": 0.20,
            "L4": 0.20,
            "L5": 0.35,
            "POP": 0.10,
            "SP": 0.10
        },
        "params": {
            "lr": 1.2e-4,
            "clip": 0.20,
            "entropy": 0.012,
            "epochs": 3,
            "batch_size": 256,
            "steps_per_update": 256, #256
            "vf_clip": 0.20,
            "max_grad_norm": 0.5,
            "target_kl": 0.02,
            "temperature": 1.03,
            
            #deprecated
            "center_start": 0.0, "guard_prob": 0.0, "win_now_prob": 0.0,
            "guard_ply_min": 5, "guard_ply_max": 13,
            "distill_coef": 0.0, "mentor_depth": 1, "mentor_prob": 0.0, "mentor_coef": 0.0
        },
    },
    
    "L5L6L7": {
        "duration": 400,
        "opponent_mix": {
            "L1": 0.025,
            "L3": 0.025, 
            "L5": 0.20,
            "L6": 0.20,
            "L7": 0.35,
            "POP": 0.10,
            "SP": 0.10
        },
        "params": {
            "lr": 1.2e-4,
            "clip": 0.20,
            "entropy": 0.012,
            "epochs": 3,
            "batch_size": 256,
            "steps_per_update": 256, #256
            "vf_clip": 0.20,
            "max_grad_norm": 0.5,
            "target_kl": 0.02,
            "temperature": 1.03,
            
            #deprecated
            "center_start": 0.0, "guard_prob": 0.0, "win_now_prob": 0.0,
            "guard_ply_min": 5, "guard_ply_max": 13,
            "distill_coef": 0.0, "mentor_depth": 1, "mentor_prob": 0.0, "mentor_coef": 0.0
        },
    },
    
    "L7L8L9": {
        "duration": 400,
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
        "params": {
            "lr": 1.2e-4,
            "clip": 0.20,
            "entropy": 0.012,
            "epochs": 3,
            "batch_size": 256,
            "steps_per_update": 256, #256
            "vf_clip": 0.20,
            "max_grad_norm": 0.5,
            "target_kl": 0.02,
            "temperature": 1.03,
            
            #deprecated
            "center_start": 0.0, "guard_prob": 0.0, "win_now_prob": 0.0,
            "guard_ply_min": 5, "guard_ply_max": 13,
            "distill_coef": 0.0, "mentor_depth": 1, "mentor_prob": 0.0, "mentor_coef": 0.0
        },
    },
    
    "L9L10L11": {
        "duration": 400,
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
        "params": {
            "lr": 1.2e-4,
            "clip": 0.20,
            "entropy": 0.0115,
            "epochs": 3,
            "batch_size": 256,
            "steps_per_update": 256, #256
            "vf_clip": 0.20,
            "max_grad_norm": 0.5,
            "target_kl": 0.02,
            "temperature": 1.03,
            
            #deprecated
            "center_start": 0.0, "guard_prob": 0.0, "win_now_prob": 0.0,
            "guard_ply_min": 5, "guard_ply_max": 13,
            "distill_coef": 0.0, "mentor_depth": 1, "mentor_prob": 0.0, "mentor_coef": 0.0
        },
    },
    
    "L11L12L13": {
        "duration": 400,
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
        "params": {
            "lr": 1.2e-4,
            "clip": 0.20,
            "entropy": 0.0110,
            "epochs": 3,
            "batch_size": 256,
            "steps_per_update": 256, #256
            "vf_clip": 0.20,
            "max_grad_norm": 0.5,
            "target_kl": 0.02,
            "temperature": 1.03,
            
            #deprecated
            "center_start": 0.0, "guard_prob": 0.0, "win_now_prob": 0.0,
            "guard_ply_min": 5, "guard_ply_max": 13,
            "distill_coef": 0.0, "mentor_depth": 1, "mentor_prob": 0.0, "mentor_coef": 0.0
        },
    },
    
    "Finale": {
        "duration": 600,
        "opponent_mix": {
            "L3": 0.05,
            "L5": 0.10,
            "L7": 0.20,
            "L9": 0.20,
            "L11": 0.20,
            "L13": 0.20,
            "POP": 0.01,
            "SP": 0.04,
        },
        "params": {
            "lr": 1.1e-4,
            "clip": 0.20,
            "entropy": 0.0105,
            "epochs": 3,
            "batch_size": 256,
            "steps_per_update": 256, #256
            "vf_clip": 0.20,
            "max_grad_norm": 0.5,
            "target_kl": 0.019,
            "temperature": 1.02,
            
            #deprecated
            "center_start": 0.0, "guard_prob": 0.0, "win_now_prob": 0.0,
            "guard_ply_min": 5, "guard_ply_max": 13,
            "distill_coef": 0.0, "mentor_depth": 1, "mentor_prob": 0.0, "mentor_coef": 0.0
        },
    },
    

}
