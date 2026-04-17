# DQN/dqn_training_phases_config.py
# per-game-sampling-friendly curriculum.

# log::
   

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

TRAINING_PHASES = {
    # ---------------------------
    # 1) First stabilization after checkpoint
    # ---------------------------
    "Intro_POP_DQN": {
        "duration": 10,        #100
        "opponent_mix": {
            # POP + a little SP, plus early basics
            "POP": 0.30,
            "SP":  0.08,
            "R":   0.02,

            # early basics
            "L1":  0.12,
            "L2":  0.08,

            # foundation (present, even if light)
            "L3":  0.08,
            "L5":  0.08,
            "L7":  0.08,
            "L9":  0.06,

            # a taste of pain (heavies) so policy doesn't become a pacifist
            "L11": 0.05,
            "L13": 0.05,
        },
        "params": dict(DQN_BASE_PARAMS),
    },

    # ---------------------------
    # 2) "All opponents", but not uniform (3x100 bands)
    #    Still totals 300 like your original Intro_ALL.
    # ---------------------------
    "Intro_ALL_A": {  # L1..L5 emphasis, but all present
        "duration": 10,    #100
        "opponent_mix": {
            "POP": 0.22,
            "SP":  0.06,
            "R":   0.02,

            "L1":  0.12,
            "L2":  0.10,
            "L3":  0.10,
            "L4":  0.08,
            "L5":  0.06,

            "L6":  0.03,
            "L7":  0.03,
            "L8":  0.03,
            "L9":  0.03,
            "L10": 0.03,
            "L11": 0.03,
            "L12": 0.03,
            "L13": 0.03,
        },
        "params": dict(DQN_BASE_PARAMS),
    },

    "Intro_ALL_B": {  # L5..L9 emphasis, all present
        "duration": 10,    #100
        "opponent_mix": {
            "POP": 0.18,
            "SP":  0.08,
            "R":   0.01,

            "L5":  0.12,
            "L6":  0.10,
            "L7":  0.10,
            "L8":  0.08,
            "L9":  0.10,

            "L1":  0.05,
            "L2":  0.02,
            "L3":  0.02,
            "L4":  0.02,
            "L10": 0.04,
            "L11": 0.03,
            "L12": 0.025,
            "L13": 0.025,
        },
        "params": dict(DQN_BASE_PARAMS),
    },

    "Intro_ALL_C": {  # L9..L13 emphasis, all present
        "duration": 10,    #100
        "opponent_mix": {
            "POP": 0.14,
            "SP":  0.10,
            "R":   0.01,

            "L9":  0.10,
            "L10": 0.10,
            "L11": 0.13,
            "L12": 0.10,
            "L13": 0.12,

            "L1":  0.05,
            "L2":  0.02,
            "L3":  0.04,
            "L4":  0.01,
            "L5":  0.03,
            "L6":  0.01,
            "L7":  0.03,
            "L8":  0.01,
        },
        "params": dict(DQN_BASE_PARAMS),
    },

    # ---------------------------
    # 3) Stages: focus + refresh tail (320 + 80 = 400 each)
    #    Focus: x + (x+1 even) get extra weight.
    #    Refresh: re-balance anchors to reduce forgetting.
    # ---------------------------
    "L3_focus": {
        "duration": 10,    #320
        "opponent_mix": {
            "SP":  0.10,
            "POP": 0.18,

            "L1":  0.08,
            "L3":  0.14,
            "L4":  0.10,

            "L5":  0.06,
            "L7":  0.06,
            "L9":  0.06,

            "L11": 0.10,
            "L13": 0.12,
        },
        "params": dict(DQN_BASE_PARAMS),
    },
    "L3_refresh": {
        "duration": 10,  #80
        "opponent_mix": {
            "SP":  0.08,
            "POP": 0.22,

            "L1":  0.12,
            "L3":  0.10,
            "L5":  0.10,
            "L7":  0.08,
            "L9":  0.08,
            "L11": 0.11,
            "L13": 0.11,
        },
        "params": dict(DQN_BASE_PARAMS),
    },

    "L5_focus": {
        "duration": 10,  #320
        "opponent_mix": {
            "SP":  0.16,
            "POP": 0.14,

            "L1":  0.08,
            "L5":  0.14,
            "L6":  0.10,

            "L3":  0.06,
            "L7":  0.06,
            "L9":  0.06,

            "L11": 0.10,
            "L13": 0.10,
        },
        "params": dict(DQN_BASE_PARAMS),
    },
    "L5_refresh": {
        "duration": 10,     #80
        "opponent_mix": {
            "SP":  0.12,
            "POP": 0.18,

            "L1":  0.12,
            "L3":  0.09,
            "L5":  0.11,
            "L7":  0.08,
            "L9":  0.08,
            "L11": 0.11,
            "L13": 0.11,
        },
        "params": dict(DQN_BASE_PARAMS),
    },

    "L7_focus": {
        "duration": 10,    #320
        "opponent_mix": {
            "SP":  0.22,
            "POP": 0.12,

            "L1":  0.08,
            "L7":  0.14,
            "L8":  0.09,

            "L3":  0.06,
            "L5":  0.06,
            "L9":  0.05,

            "L11": 0.09,
            "L13": 0.09,
        },
        "params": dict(DQN_BASE_PARAMS),
    },
    "L7_refresh": {
        "duration": 10,   #80
        "opponent_mix": {
            "SP":  0.16,
            "POP": 0.14,

            "L1":  0.12,
            "L3":  0.09,
            "L5":  0.09,
            "L7":  0.11,
            "L9":  0.07,
            "L11": 0.11,
            "L13": 0.11,
        },
        "params": dict(DQN_BASE_PARAMS),
    },

    "L9_focus": {
        "duration": 10,    #320
        "opponent_mix": {
            "SP":  0.28,
            "POP": 0.10,

            "L1":  0.08,
            "L9":  0.14,
            "L10": 0.09,

            "L3":  0.05,
            "L5":  0.05,
            "L7":  0.06,

            "L11": 0.08,
            "L13": 0.07,
        },
        "params": dict(DQN_BASE_PARAMS),
    },
    "L9_refresh": {
        "duration": 10,     #80
        "opponent_mix": {
            "SP":  0.22,
            "POP": 0.10,

            "L1":  0.12,
            "L3":  0.08,
            "L5":  0.08,
            "L7":  0.08,
            "L9":  0.10,
            "L11": 0.11,
            "L13": 0.11,
        },
        "params": dict(DQN_BASE_PARAMS),
    },

    "L11_focus": {
        "duration": 10,    #320
        "opponent_mix": {
            "SP":  0.34,
            "POP": 0.08,

            "L1":  0.08,
            "L11": 0.16,
            "L12": 0.08,

            "L3":  0.05,
            "L5":  0.05,
            "L7":  0.05,
            "L9":  0.05,

            "L13": 0.06,
        },
        "params": dict(DQN_BASE_PARAMS),
    },
    "L11_refresh": {
        "duration": 80, #80
        "opponent_mix": {
            "SP":  0.28,
            "POP": 0.08,

            "L1":  0.12,
            "L3":  0.07,
            "L5":  0.07,
            "L7":  0.07,
            "L9":  0.07,
            "L11": 0.12,
            "L13": 0.12,
        },
        "params": dict(DQN_BASE_PARAMS),
    },

    "L13_focus": {
        "duration": 10,    #320
        "opponent_mix": {
            "SP":  0.40,
            "POP": 0.06,

            "L1":  0.08,
            "L13": 0.16,
            "L12": 0.08,  # x+1 doesn't exist, so we pair with even (12)

            "L3":  0.04,
            "L5":  0.04,
            "L7":  0.04,
            "L9":  0.04,

            "L11": 0.06,
        },
        "params": dict(DQN_BASE_PARAMS),
    },
    "L13_refresh": {
        "duration": 10,     #80
        "opponent_mix": {
            "SP":  0.34,
            "POP": 0.06,

            "L1":  0.12,
            "L3":  0.06,
            "L5":  0.06,
            "L7":  0.06,
            "L9":  0.06,
            "L11": 0.12,
            "L13": 0.12,
        },
        "params": dict(DQN_BASE_PARAMS),
    },

    # ---------------------------
    # 4) Final self-play heavy, still anchored
    # ---------------------------
    "SP_final_DQN": {
        "duration": 10,    #200
        "opponent_mix": {
            "SP":  0.50,
            "POP": 0.06,

            "L1":  0.06,
            "L3":  0.07,
            "L5":  0.07,
            "L7":  0.07,
            "L9":  0.07,
            "L11": 0.05,
            "L13": 0.05,
        },
        "params": dict(DQN_BASE_PARAMS),
    },
}

# -------------------------------------------------------------------------
# DQN additions 
# -------------------------------------------------------------------------



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
            raise ValueError(f"Phase '{name}' missing required key: opponent_mix")

        mix = ph["opponent_mix"]
        s = float(sum(float(x) for x in mix.values()))
        if abs(s - 1.0) > tol:
            raise ValueError(f"Phase '{name}' opponent_mix sums to {s:.6f} (expected 1.0)")

        dqn = ph.get("dqn_params", None)
        if not isinstance(dqn, dict):
            raise ValueError(f"Phase '{name}' missing required dict: dqn_params")

        for k in ("epsilon", "epsilon_min", "lr", "TU_mode", "tau", "TU"):
            if k not in dqn:
                raise ValueError(f"Phase '{name}' dqn_params missing key: {k}")
