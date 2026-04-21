# PPO/ppo_training_phases_config.py

# PPO 

BASE_PARAMS = {
    "lr": 4.25e-4,                      #4.0 -> 4.25
    "clip": 0.22,
    "entropy": 0.0250,                  
    "epochs": 9,                        # 6->8 ->9 (8, 9, 10) same results -> 10
    "batch_size": 256,
    "steps_per_update": 256,            #256->512->256
    "vf_clip": 0.20,
    "max_grad_norm": 1.0,
    "target_kl": 0.020,                
    "temperature": 1.00,               
}


# Slow anneal
MID  = dict(BASE_PARAMS, lr=4.00e-4, entropy=0.0225, temperature=0.99)
LATE = dict(BASE_PARAMS, lr=3.75e-4, entropy=0.0200, temperature=0.97)
SP   = dict(BASE_PARAMS, lr=3.50e-4, entropy=0.0200, temperature=0.95)


TRAINING_PHASES = {
    "BASE": {
        "duration": 700,
        "opponent_mix": {
            "POP":  0.05,
            "SP":   0.06,
            "R":    0.01,
            "C":    0.01,
            "LEFT": 0.01,   

            "L1":   0.07,
            "L2":   0.05,
            "L3":   0.10,
            "L4":   0.05,
            "L5":   0.11,
            "L6":   0.05,
            "L7":   0.07,
            "L8":   0.05,
            "L9":   0.07,
            "L10":  0.05,
            "L11":  0.07,
            "L12":  0.05,
            "L13":  0.07,
        },
        "params": dict(BASE_PARAMS),
    },
    "ODD_MORE": {
        "duration": 600,
        "opponent_mix": {
            "POP":  0.06,
            "SP":   0.08,
            "R":    0.01,
            "C":    0.01,
            "LEFT": 0.01,   

            "L1":   0.07,
            "L2":   0.04,
            "L3":   0.10,
            "L4":   0.04,
            "L5":   0.10,
            "L6":   0.04,
            "L7":   0.08,
            "L8":   0.04,
            "L9":   0.08,
            "L10":  0.04,
            "L11":  0.08,
            "L12":  0.04,
            "L13":  0.08,
        },
        "params": dict(MID),
    },
    "FINAL": {
        "duration": 500,
        "opponent_mix": {
            "POP":  0.08,
            "SP":   0.10,
 
            "L1":   0.07,
            "L2":   0.03,
            "L3":   0.10,
            "L4":   0.03,
            "L5":   0.10,
            "L6":   0.05,
            "L7":   0.08,
            "L8":   0.03,
            "L9":   0.09,
            "L10":  0.04,
            "L11":  0.08,
            "L12":  0.04,
            "L13":  0.08,
        },
        "params": dict(LATE),
    },
    "SP_FINALE": {
        "duration": 200,
        "opponent_mix": {
            "POP":  0.12,
            "SP":   0.13,
 
            "L1":   0.07,
            "L2":   0.03,
            "L3":   0.07,
            "L4":   0.03,
            "L5":   0.07,
            "L6":   0.05,
            "L7":   0.07,
            "L8":   0.03,
            "L9":   0.09,
            "L10":  0.04,
            "L11":  0.08,
            "L12":  0.04,
            "L13":  0.08,
        },
        "params": dict(SP),
    }
}


# -------------------------------------------------------------------------
# DQN additions
# -------------------------------------------------------------------------

# DQN defaults.
DQN_BASE_PARAMS = {
    "epsilon": 0.50,
    "epsilon_min": 0.10,
    "lr": 1.0e-4,

    # Target network update
    # - "soft": Polyak update with tau
    # - "hard": copy every TU updates (loop defines the unit)
    "TU_mode": "soft",
    "tau": 35e-4,
    "TU": 500,
}


def attach_dqn_defaults(training_phases: dict) -> None:
    """
    Adds training_phases[phase]["dqn_params"].
    Each phase gets its own dict. Phase overrides win.
    """
    for _name, ph in training_phases.items():
        overrides = ph.get("dqn_params", {}) or {}
        dqn = dict(DQN_BASE_PARAMS)
        dqn.update(overrides)
        ph["dqn_params"] = dqn


attach_dqn_defaults(TRAINING_PHASES)


def validate_training_phases(training_phases: dict, tol: float = 1e-6) -> None:
    """
    Sanity checks.
    Raises if opponent_mix does not sum to ~1.0, or if required DQN keys missing.
    """
    for name, ph in training_phases.items():

        if "duration" not in ph:
            raise ValueError(f"Phase '{name}' missing required key: duration")
        if "opponent_mix" not in ph:
            raise ValueError(
                f"Phase '{name}' missing required key: opponent_mix")

        mix = ph["opponent_mix"]
        s = float(sum(float(x) for x in mix.values()))
        if abs(s - 1.0) > tol:
            raise ValueError(
                f"Phase '{name}' opponent_mix sums to {s:.6f} (expected 1.0)")

        dqn = ph.get("dqn_params", None)
        if not isinstance(dqn, dict):
            raise ValueError(
                f"Phase '{name}' missing required dict: dqn_params")

        for k in ("epsilon", "epsilon_min", "lr", "TU_mode", "tau", "TU"):
            if k not in dqn:
                raise ValueError(f"Phase '{name}' dqn_params missing key: {k}")
