# PPO/ppo_training_phases_config.py

TRAINING_PHASES = {

    # * refreshing L1 * "
    "Refresh_L1": {
        "duration": 100,
        "opponent_mix": {
            "L1":  0.90,
            "L3":  0.05,
            "POP": 0.05,
        },
        "params": {

            "lr":              625e-7,
            "clip":            0.20,
            "entropy":         0.0065,
            "epochs":          3,
            "batch_size":      1024,
            "steps_per_update": 128,
            "vf_clip":         0.20,
            "max_grad_norm":   0.5,
            "target_kl":       0.016,
            "temperature":     1.01,
            "center_start":    0.15,
            "guard_prob":      0.09,
            "win_now_prob":    0.19,
            "guard_ply_min":   5,
            "guard_ply_max":   13,
            "distill_coef":    0.10,
            "mentor_depth":    7,
            "mentor_prob":     0.99,
            "mentor_coef":     0.49,
        },
    },

    # * refreshing L1 * "
    "Refresh_L1L3": {
        "duration": 100,
        "opponent_mix": {
            "L1":  0.50,
            "L3":  0.45,
            "POP": 0.05,
        },
        "params": {

            "lr":              625e-7,
            "clip":            0.20,
            "entropy":         0.0065,
            "epochs":          3,
            "batch_size":      1024,
            "steps_per_update": 128,
            "vf_clip":         0.20,
            "max_grad_norm":   0.5,
            "target_kl":       0.015,
            "temperature":     1.01,
            "center_start":    0.16,
            "guard_prob":      0.09,
            "win_now_prob":    0.19,
            "guard_ply_min":   5,
            "guard_ply_max":   13,
            "distill_coef":    0.10,
            "mentor_depth":    7,
            "mentor_prob":     0.99,
            "mentor_coef":     0.49,
        },
    },

    # * refreshing L3 * "
    "Refresh_L3": {
        "duration": 100,
        "opponent_mix": {
            "L3":  0.90,
            "L5":  0.05,
            "POP": 0.05,
        },
        "params": {

            "lr":              625e-7,
            "clip":            0.20,
            "entropy":         0.0060,
            "epochs":          3,
            "batch_size":      1024,
            "steps_per_update": 128,
            "vf_clip":         0.20,
            "max_grad_norm":   0.5,
            "target_kl":       0.016,
            "temperature":     1.02,
            "center_start":    0.15,
            "guard_prob":      0.09,
            "win_now_prob":    0.19,
            "guard_ply_min":   5,
            "guard_ply_max":   13,
            "distill_coef":    0.10,
            "mentor_depth":    7,
            "mentor_prob":     0.99,
            "mentor_coef":     0.49,
        },
    },

    # * refreshing L3L5 * "
    "Refresh_L3L5": {
        "duration": 100,
        "opponent_mix": {
            "L3":  0.50,
            "L5":  0.45,
            "POP": 0.05,
        },
        "params": {

            "lr":              625e-7,
            "clip":            0.20,
            "entropy":         0.0060,
            "epochs":          3,
            "batch_size":      1024,
            "steps_per_update": 128,
            "vf_clip":         0.20,
            "max_grad_norm":   0.5,
            "target_kl":       0.016,
            "temperature":     1.03,
            "center_start":    0.15,
            "guard_prob":      0.09,
            "win_now_prob":    0.19,
            "guard_ply_min":   5,
            "guard_ply_max":   13,
            "distill_coef":    0.10,
            "mentor_depth":    7,
            "mentor_prob":     0.99,
            "mentor_coef":     0.49,
        },
    },


    # * breaking the POP ensemble, extreme mentor guidance, stabilizes the center * "
    "POP_Intro": {
        "duration": 300,
        "opponent_mix": {
            "R": 0.01,
            "L1":  0.04,
            "L3": 0.04,
            "L5": 0.04,
            "L7": 0.02,
            "POP": 0.85,
        },
        "params": {

            "lr":              625e-7,
            "clip":            0.20,
            "entropy":         0.0055,
            "epochs":          3,
            "batch_size":      1024,
            "steps_per_update": 128,
            "vf_clip":         0.20,
            "max_grad_norm":   0.5,
            "target_kl":       0.020,
            "temperature":     1.05,
            "center_start":    0.15,
            "guard_prob":      0.09,
            "win_now_prob":    0.19,
            "guard_ply_min":   5,
            "guard_ply_max":   13,
            "distill_coef":    0.00,
            "mentor_depth":    7,
            "mentor_prob":     0.99,
            "mentor_coef":     0.49,
        },
    },

    # * normal start, mentor guidance, minimal POP regularization, dwarfed by mentor * #
    "START": {
        "duration": 200,
        "opponent_mix": {
            "L1": 0.02,
            "L3": 0.35,
            "L5": 0.40,
            "L7": 0.02,
            "POP": 0.20,
            "SP": 0.01,
        },
        "params": {

            "lr":              625e-7,  # 625e-7
            "clip":            0.20,
            "entropy":         0.0058,  # 0.0050
            "epochs":          3,  # 3
            "batch_size":      1024,  # 1024
            "steps_per_update": 128,  # 128 - no sense yet to go lower, until games become longer!
            "vf_clip":         0.20,  # 0.19
            "max_grad_norm":   0.5,
            "target_kl":       0.020,  # 0.20
            "temperature":     1.03,  # 0.95 #.090
            "center_start":    0.10,
            "guard_prob":      0.09,
            "win_now_prob":    0.19,
            "guard_ply_min":   5,
            "guard_ply_max":   13,
            "distill_coef":    0.10,  # 0  to 0.10
            "mentor_depth":    7,
            "mentor_prob":     0.89,  # 0.50
            "mentor_coef":     0.29,  # 0.25 - 0.30
        },
    },


    # * main part, mentor guidance * #
    "MAIN": {
        "duration": 200,  # 600
        "opponent_mix": {
            "L1": 0.02,
            "L3": 0.40,
            "L5": 0.40,
            "L7": 0.02,
            "POP": 0.15,
            "SP": 0.01,
        },
        "params": {

            "lr":              625e-7,  # 625e-7
            "clip":            0.20,
            "entropy":         0.0055,  # 0.0050
            "epochs":          3,  # 3
            "batch_size":      1024,  # 1024
            "steps_per_update": 128,  # 128 - no sense yet to go lower, until games become longer!
            "vf_clip":         0.20,  # 0.19
            "max_grad_norm":   0.5,
            "target_kl":       0.020,  # 0.20
            "temperature":     1.02,  # 0.95 #.090
            "center_start":    0.16,
            "guard_prob":      0.09,
            "win_now_prob":    0.19,
            "guard_ply_min":   5,
            "guard_ply_max":   13,
            "distill_coef":    0.05,  # 0  to 0.10
            "mentor_depth":    7,
            "mentor_prob":     0.49,  # 0.50
            "mentor_coef":     0.24,  # 0.25 - 0.30
        },
    },

    # *  reduced mentor guidance * #
    "FINALE": {
        "duration": 200,  # 600
        "opponent_mix": {
            "L1": 0.02,
            "L3": 0.40,
            "L5": 0.40,
            "L7": 0.02,
            "POP": 0.15,
            "SP": 0.01,
        },
        "params": {

            "lr":              600e-7,  # 625e-7
            "clip":            0.20,
            "entropy":         0.0050,  # 0.0050
            "epochs":          3,  # 3
            "batch_size":      1024,  # 1024
            "steps_per_update": 128,  # 128 - no sense yet to go lower, until games become longer!
            "vf_clip":         0.20,  # 0.19
            "max_grad_norm":   0.5,
            "target_kl":       0.020,  # 0.20
            "temperature":     1.05,  # 0.95 #.090
            "center_start":    0.28,
            "guard_prob":      0.09,
            "win_now_prob":    0.19,
            "guard_ply_min":   5,
            "guard_ply_max":   13,

            "distill_coef":    0.01,  # 0  to 0.10
            "mentor_depth":    7,
            "mentor_prob":     0.44,  # 0.50
            "mentor_coef":     0.19,  # 0.25 - 0.30
        },
    },

    # *  pure policy training, little guidance * #
    "EXIT": {
        "duration": 300,  # 600
        "opponent_mix": {
            "L1": 0.02,
            "L3": 0.40,
            "L5": 0.50,
            "L6": 0.02,
            "L7": 0.02,
            "SP": 0.01,
        },
        "params": {

            "lr":              600e-7,  # 625e-7
            "clip":            0.20,
            "entropy":         0.0045,  # 0.0050
            "epochs":          3,  # 3
            "batch_size":      1024,  # 1024
            "steps_per_update": 128,  # 128 - no sense yet to go lower, until games become longer!
            "vf_clip":         0.18,  # 0.19
            "max_grad_norm":   0.5,
            "target_kl":       0.020,  # 0.20
            "temperature":     1.05,  # 0.95 #0.90
            "center_start":    0.28,
            "guard_prob":      0.09,
            "win_now_prob":    0.19,
            "guard_ply_min":   5,
            "guard_ply_max":   13,
            "distill_coef":    0.01,  # 0  to 0.10
            "mentor_depth":    7,
            "mentor_prob":     0.34,  # 0.50
            "mentor_coef":     0.15,  # 0.25 - 0.30
        },
    },



    # *  selfplay, some mentor guidance* #
    "SELF": {
        "duration": 400,
        "opponent_mix": {
            "L3": 0.10,
            "SP":  0.90,

        },
        "params": {

            "lr":              600e-7,  # 625e-7
            "clip":            0.20,
            "entropy":         0.0055,  # 0.0050
            "epochs":          3,  # 3
            "batch_size":      1024,  # 1024
            "steps_per_update": 128,  # 128 - no sense yet to go lower, until games become longer!
            "vf_clip":         0.17,  # 0.19
            "max_grad_norm":   0.5,
            "target_kl":       0.020,  # 0.20
            "temperature":     1.00,  # 0.95 #0.90

            "center_start":    0.24,
            "guard_prob":      0.09,
            "win_now_prob":    0.19,
            "guard_ply_min":   5,
            "guard_ply_max":   13,

            "distill_coef":    0.00,  # 0  to 0.10
            "mentor_depth":    7,
            "mentor_prob":     0.20,  # 0.50
            "mentor_coef":     0.10,  # 0.25 - 0.30
        },
    },


}
