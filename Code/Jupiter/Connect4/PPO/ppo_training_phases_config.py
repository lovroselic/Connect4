TRAINING_PHASES = {

    "Random1":  {"duration":300,  "lookahead":None,
      "params":{"lr":3e-4,"clip":0.20,"entropy":0.012,"epochs":3,"temperature":1.0,"steps_per_update":2048}},
    "L1_Intro": {"duration":1200, "lookahead":1,
      "params":{"lr":2.0e-4,"clip":0.12,"entropy":0.008,"epochs":4,"temperature":0.15,"steps_per_update":8192}},
    "Random2":  {"duration":150,  "lookahead":None,
      "params":{"lr":2.0e-4,"clip":0.18,"entropy":0.010,"epochs":3,"temperature":0.9,"steps_per_update":2048}},
    
    # "L1_Consol":{"duration":1200, "lookahead":1,
    #   "params":{"lr":1.8e-4,"clip":0.12,"entropy":0.006,"epochs":4,"temperature":0.12,"steps_per_update":8192}},
    # "Random3":  {"duration":150,  "lookahead":None,
    #   "params":{"lr":1.8e-4,"clip":0.18,"entropy":0.010,"epochs":3,"temperature":0.9,"steps_per_update":2048}},

    # "L2_Intro": {"duration":1200, "lookahead":2,
    #   "params":{"lr":1.8e-4,"clip":0.10,"entropy":0.006,"epochs":4,"temperature":0.10,"steps_per_update":8192}},
    # "L1_Spar_AfterL2Intro":{"duration":300,"lookahead":1,
    #   "params":{"lr":1.8e-4,"clip":0.12,"entropy":0.006,"epochs":3,"temperature":0.12,"steps_per_update":8192}},
    # "Random4":  {"duration":150, "lookahead":None,
    #   "params":{"lr":1.8e-4,"clip":0.15,"entropy":0.010,"epochs":3,"temperature":0.9,"steps_per_update":2048}},
    # "L2_Consol":{"duration":1500,"lookahead":2,
    #   "params":{"lr":1.2e-4,"clip":0.10,"entropy":0.005,"epochs":4,"temperature":0.08,"steps_per_update":12288}},
    # "L1_Spar_AfterL2Consol":{"duration":300,"lookahead":1,
    #   "params":{"lr":1.2e-4,"clip":0.10,"entropy":0.005,"epochs":3,"temperature":0.10,"steps_per_update":8192}},
    # "SelfPlay_L1L2":{"duration":1200,"lookahead":"self",
    #   "params":{"lr":1.0e-4,"clip":0.08,"entropy":0.004,"epochs":3,"temperature":0.10,"steps_per_update":8192}},


    # "Random5":  {"duration":150,  "lookahead":None,
    #   "params":{"lr":9e-5,"clip":0.15,"entropy":0.009,"epochs":3,"temperature":0.9,"steps_per_update":2048}},

    # "L3_Intro": {"duration":700,  "lookahead":3,   # shorter burst
    #   "params":{"lr":9e-5,"clip":0.08,

    #     "entropy_start":0.010, "entropy_end":0.007,
    #     "temp_start":0.15,     "temp_end":0.10,
    #     "epochs":4,"steps_per_update":16384}},

    # "L2_Spar_AfterL3Intro":{"duration":500,"lookahead":2,
    #   "params":{"lr":8e-5,"clip":0.09,"entropy":0.006,"epochs":3,"temperature":0.10,"steps_per_update":12288}},

    # "Random6": {"duration":150,"lookahead":None,
    #   "params":{"lr":8e-5,"clip":0.15,"entropy":0.009,"epochs":3,"temperature":0.9,"steps_per_update":2048}},

    # "L3_Consol":{"duration":1200,"lookahead":3,   # also shorter, then repeat cycle
    #   "params":{"lr":6e-5,"clip":0.07,
    #     "entropy_start":0.008,"entropy_end":0.005,
    #     "temp_start":0.12,    "temp_end":0.08,
    #     "epochs":4,"steps_per_update":16384}},

    # "L2_Spar_AfterL3Consol":{"duration":600,"lookahead":2,
    #   "params":{"lr":6e-5,"clip":0.08,"entropy":0.005,"epochs":3,"temperature":0.10,"steps_per_update":12288}},
    # "L1_Spar_AfterL3Consol":{"duration":300,"lookahead":1,
    #   "params":{"lr":6e-5,"clip":0.09,"entropy":0.005,"epochs":3,"temperature":0.12,"steps_per_update":8192}},

    # "SelfPlay_L1L2L3":{"duration":1500,"lookahead":"self",
    #   "params":{"lr":5e-5,"clip":0.06,
    #     "entropy_start":0.007,"entropy_end":0.004,
    #     "temp_start":0.12,    "temp_end":0.08,
    #     "epochs":3,"steps_per_update":12288}},

}
