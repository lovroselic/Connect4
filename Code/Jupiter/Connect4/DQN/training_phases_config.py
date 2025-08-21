# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 14:44:26 2025

@author: lovro
"""

# [R, L1, L2, L3, L4, L5, L6, L7]

from IPython.display import display, HTML
import pandas as pd
import numpy as np

TRAINING_PHASES = {
    
    ######################### Random Section ############################################

    "Random1": {
        "duration": 300,
        "weights": [0.50, 0.45, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 1.0, "epsilon_min": 0.10,         
        "memory_prune_recent": 0.0,      
        "memory_prune_low":    0.0,      
        "opponent": [1, 0, 0, 0, 0, 0, 0, 0],
    },
    "Random2": {
        "duration": 300,
        "weights": [0.30, 0.60, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00],
        "epsilon": 0.90, "epsilon_min": 0.10,
        "memory_prune_recent": 0.02,
        "memory_prune_low":    0.20,
        "opponent": [1, 0, 0, 0, 0, 0, 0, 0],
    },
    
    ########################## Mixed R/L1 #########################################
    
    "Mixed_R1_Attack": {
         "duration": 300,
         "weights": [0.15, 0.80, 0.05, 0, 0, 0.00, 0.00, 0.00], 
         "epsilon": 0.75, "epsilon_min": 0.10,
         "memory_prune_recent": 0.02,
         "memory_prune_low":    0.20,
         "opponent": [0.70, 0.30, 0, 0, 0, 0, 0, 0],
     },
    
    "Mixed_R1_Defense": {
         "duration": 300,
         "weights": [0.40, 0.55, 0.05, 0, 0, 0.00, 0.00, 0.00], 
         "epsilon": 0.70, "epsilon_min": 0.10,
         "memory_prune_recent": 0.02,
         "memory_prune_low":    0.20,
         "opponent": [0.50, 0.50, 0, 0, 0, 0, 0, 0],
     },
    
    "Mixed_R1_Equal": {
         "duration": 300,
         "weights": [0.40, 0.60, 0.00, 0, 0, 0.00, 0.00, 0.00], 
         "epsilon": 0.65, "epsilon_min": 0.10,
         "memory_prune_recent": 0.02,
         "memory_prune_low":    0.20,
         "opponent": [0.50, 0.50, 0, 0, 0, 0, 0, 0],
     },
    
    "Mixed_R1_Conclude": {
         "duration": 300,
         "weights": [0.15, 0.80, 0.05, 0, 0, 0.00, 0.00, 0.00], 
         "epsilon": 0.60, "epsilon_min": 0.10,
         "memory_prune_recent": 0.02,
         "memory_prune_low":    0.20,
         "opponent": [0.50, 0.50, 0, 0, 0, 0, 0, 0],
     },
    
    "SelfPlay_Mixed_R1": {
         "duration": 300,
         "weights": None,
         "epsilon": 0.05, "epsilon_min": 0.05,
         "memory_prune_recent": 0.02,
         "memory_prune_low":    0.20,
         "opponent": None,
     },
    

   ########################## Mixed R/L1++  #########################################  
   
   "Mixed_R11_Attack": {
        "duration": 300,
        "weights": [0.0, 0.60, 0.30, 0.10, 0, 0.00, 0.00, 0.00], 
        "epsilon": 0.70, "epsilon_min": 0.10,
        "memory_prune_recent": 0.0,
        "memory_prune_low":    0.2,
        "opponent": [0.30, 0.70, 0, 0, 0, 0, 0, 0],
    },
   
   "Mixed_R11_Defense": {
        "duration": 300,
        "weights": [0.10, 0.90, 0.00, 0, 0, 0.00, 0.00, 0.00], 
        "epsilon": 0.65, "epsilon_min": 0.10,
        "memory_prune_recent": 0.0,
        "memory_prune_low":    0.20,
        "opponent": [0.10, 0.90, 0, 0, 0, 0, 0, 0],
    },
   
   "Mixed_R11_Equal": {
        "duration": 300,
        "weights": [0.00, 0.60, 0.30, 0.10, 0, 0.00, 0.00, 0.00], 
        "epsilon": 0.60, "epsilon_min": 0.10,
        "memory_prune_recent": 0.0,
        "memory_prune_low":    0.20,
        "opponent": [0.10, 0.90, 0, 0, 0, 0, 0, 0],
    },
   
   "Mixed_R11_Conclude": {
        "duration": 300,
        "weights": [0.0, 0.65, 0.15, 0.20, 0, 0.00, 0.00, 0.00], 
        "epsilon": 0.50, "epsilon_min": 0.10,
        "memory_prune_recent": 0.02,
        "memory_prune_low":    0.20,
        "opponent": [0.10, 0.90, 0, 0, 0, 0, 0, 0],
    },
   
   "SelfPlay_Mixed_R11": {
        "duration": 300,
        "weights": None,
        "epsilon": 0.05, "epsilon_min": 0.05,
        "memory_prune_recent": 0.02,
        "memory_prune_low":    0.20,
        "opponent": None,
    },


   #  "Mixed_R1": {
   #      "duration": 500,
   #      "weights": [0.15, 0.80, 0.05, 0, 0, 0.00, 0.00, 0.00], #learn to attack
   #      "epsilon": 0.75, "epsilon_min": 0.10,
   #      "memory_prune_recent": 0.20,
   #      "memory_prune_low":    0.10,
   #      "opponent": [0.50, 0.50, 0, 0, 0, 0, 0, 0],
   #  },
    
   #  "Mixed_R11": {
   #      "duration": 650,                  
   #      "weights": [0.08, 0.77, 0.15, 0,0,0,0,0],
   #      "epsilon": 0.65, "epsilon_min": 0.10,  # a touch more explore than before
   #      "memory_prune_recent": 0.15,      
   #      "memory_prune_low":    0.15,      
   #      "opponent": [0.25, 0.75, 0, 0,0,0,0,0],  # more L1 pressure
   #  },

   #  "SelfPlay_Mixed_R1": {
   #      "duration": 250,
   #      "weights": None,
   #      "epsilon": 0.05, "epsilon_min": 0.05,
   #      "memory_prune_recent": 0.15,
   #      "memory_prune_low":    0.15,
   #      "opponent": None,
   #  },
    
   #  ######################### intro L1 ##########################################
    
   #  "L1_Defense_Intro": {
   #      "duration": 350,                   
   #      "weights": [0.00, 0.78, 0.20, 0.02, 0,0,0,0],
   #      "epsilon": 0.60, "epsilon_min": 0.08,  
   #      "memory_prune_recent": 0.15,       
   #      "memory_prune_low":    0.20,       
   #      "opponent": [0.05, 0.90, 0.05, 0,0,0,0,0],
   #  },
    
   # "L1_Defense_Core": {
   #      "duration": 500,
   #      "weights": [0.00, 0.70, 0.25, 0.05, 0,0,0,0],
   #      "epsilon": 0.28, "epsilon_min": 0.06,
   #      "memory_prune_recent": 0.15,       
   #      "memory_prune_low":    0.18,       
   #      "opponent": [0.00, 0.95, 0.05, 0,0,0,0,0],
   #  },
   
   # "L1_Win_Focus_A": {
   #      "duration": 500,
   #      "weights": [0.00, 0.08, 0.22, 0.70, 0,0,0,0],   # push depth during explore
   #      "epsilon": 0.40, 
   #      "epsilon_min": 0.10,          
   #      "memory_prune_recent": 0.10,                     # keep most recent sequences
   #      "memory_prune_low":    0.12,
   #      "opponent": [0.05, 0.92, 0.03, 0,0,0,0,0],      # almost all L1
   #  },
   
   # "SelfPlay_L1_anchor": {
   #      "duration": 250,
   #      "weights": None,
   #      "epsilon": 0.02, "epsilon_min": 0.02,   # pure greedy in training step selection
   #      "memory_prune_recent": 0.15,
   #      "memory_prune_low":    0.15,
   #      "opponent": None
   #  },
   
   # "L1_Win_Focus_B": {
   #      "duration": 500,
   #      "weights": [0.00, 0.10, 0.25, 0.65, 0,0,0,0],
   #      "epsilon": 0.65, "epsilon_min": 0.15,
   #      "memory_prune_recent": 0.10,       
   #      "memory_prune_low":    0.10,       
   #      "opponent": [0.05, 0.90, 0.05, 0,0,0,0,0],
   #  },
   
   # ############################################################
    
   #  "Fixed1_Attack":  { 
   #      "duration": 350, 
   #      "weights": [0.00,0.55,0.15,0.30,0,0,0,0],
   #      "epsilon": 0.40, "epsilon_min": 0.08,
   #      "memory_prune_recent": 0.10, 
   #      "memory_prune_low": 0.10,
   #      "opponent": [0.05,0.90,0.05,0,0,0,0,0] 
   #      },
    
   #  "Fixed1_Defense": { 
   #      "duration": 350, 
   #      "weights": [0.05,0.65,0.25,0.05,0,0,0,0],
   #      "epsilon": 0.40, "epsilon_min": 0.08,
   #      "memory_prune_recent": 0.10, 
   #      "memory_prune_low": 0.10,
   #      "opponent": [0.05,0.85,0.10,0,0,0,0,0] 
   #      },

   #  "SelfPlay_L1": {
   #      "duration": 350,
   #      "weights": None,
   #      "epsilon": 0.02, "epsilon_min": 0.02,
   #      "memory_prune_recent": 0.10,
   #      "memory_prune_low":    0.10,
   #      "opponent": None
   #  }

    
    
    


    # ########################### L1 ##########################################

    # "Fixed1": {
    #     "duration": 500,
    #     "weights": [0.10, 0.55, 0.25, 0.10, 0.00, 0.00, 0.00, 0.00],
    #     "epsilon": 0.65, "epsilon_min": 0.10,
    #     "memory_prune_recent": 0.15,   # keep a slice of Random/Mixed
    #     "memory_prune_low":    0.10,
    #     "opponent": [0.10, 0.90, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    # },
    
    # "Fixed1_Defense": {  
    #     "duration": 500,
    #     "weights": [0.05, 0.65, 0.25, 0.05, 0.00, 0.00, 0.00, 0.00],
    #     "epsilon": 0.50, "epsilon_min": 0.08,   # a bit quicker consolidation
    #     "memory_prune_recent": 0.20,            # slightly stronger cleanup
    #     "memory_prune_low":    0.15,
    #     "opponent": [0.05, 0.80, 0.15, 0.00, 0.00, 0.00, 0.00, 0.00],  # sprinkle L2
    # },
    
    # "Fixed1_Attack": {  
    #     "duration": 500,
    #     "weights": [0.00, 0.50, 0.15, 0.35, 0.00, 0.00, 0.00, 0.00],
    #     "epsilon": 0.50, "epsilon_min": 0.08,   
    #     "memory_prune_recent": 0.20,            
    #     "memory_prune_low":    0.15,
    #     "opponent": [0.05, 0.90, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00],  
    # },
    
    # "L1_Focus": {
    #     "duration": 600,
    #     "weights": [0.00, 0.80, 0.15, 0.05, 0, 0, 0, 0],
    #     "epsilon": 0.30, "epsilon_min": 0.06,
    #     "memory_prune_recent": 0.10,
    #     "memory_prune_low":    0.25,   # drop easy stuff, keep hard mistakes
    #     "opponent": [0.00, 1.00, 0.00, 0, 0, 0, 0, 0],
    # },
    
    # "SelfPlay_L1": {
    #     "duration": 500,
    #     "weights": [0, 1, 0, 0, 0, 0, 0, 0],
    #     "epsilon": 0.0, "epsilon_min": 0.05,
    #     "memory_prune_recent": 0.30,   # snapshot-focused buffer
    #     "memory_prune_low":    0.15,
    #     "opponent": None,
    # },

    # ############################# L1–L2 ramp  ############################

    # "Mixed_R12": {
    #     "duration": 320,
    #     "weights": [0.02, 0.22, 0.46, 0.20, 0.10, 0.00, 0.00, 0.00],  # slight L2 bias
    #     "epsilon": 0.50, "epsilon_min": 0.08,
    #     "memory_prune_recent": 0.12,     # mild freshness
    #     "memory_prune_low":    0.10,     # keep high-|δ|
    #     "opponent": [0.25, 0.375, 0.375, 0, 0, 0, 0, 0],
    # },
    
    # "Challenge_R12": {
    #     "duration": 320,
    #     "weights": [0.08, 0.30, 0.40, 0.17, 0.05, 0.00, 0.00, 0.00],  # emphasize L2 threats
    #     "epsilon": 0.18, "epsilon_min": 0.06,
    #     "memory_prune_recent": 0.15,     # stronger cleanup
    #     "memory_prune_low":    0.18,     # lean harder into high-|δ|
    #     "opponent": [0.00, 0.05, 0.50, 0.40, 0.05, 0, 0, 0],          # L2 exposure ↑
    # },
    
    # "Relax_R12": {
    #     "duration": 300,
    #     "weights": [0.00, 0.30, 0.50, 0.20, 0.00, 0.00, 0.00, 0.00],  # consolidate vs L2
    #     "epsilon": 0.28, "epsilon_min": 0.10,
    #     "memory_prune_recent": 0.08,     # gentler to retain mixture
    #     "memory_prune_low":    0.08,
    #     "opponent": [0.30, 0.50, 0.20, 0, 0, 0, 0, 0],
    # },
    
    # # --- quick self-play anchor to stabilize learned policies ---
    # "SelfPlay_L1L2_quick": {
    #     "duration": 250,
    #     "weights": [],
    #     "epsilon": 0.0, "epsilon_min": 0.05,
    #     "memory_prune_recent": 0.20,     # snapshot-focus
    #     "memory_prune_low":    0.12,
    #     "opponent": None,
    # },
    
    # # -------------------- L1/L2 ramp (2nd set) --------------------
    # "Mixed_R1122": {
    #     "duration": 320,
    #     "weights": [0.02, 0.22, 0.46, 0.20, 0.10, 0.00, 0.00, 0.00],
    #     "epsilon": 0.48, "epsilon_min": 0.08,
    #     "memory_prune_recent": 0.12,
    #     "memory_prune_low":    0.10,
    #     "opponent": [0.18, 0.42, 0.40, 0.00, 0.00, 0.00, 0.00, 0.00],
    # },
    
    # "Challenge_R1122": {
    #     "duration": 320,
    #     "weights": [0.10, 0.30, 0.40, 0.15, 0.05, 0.00, 0.00, 0.00],
    #     "epsilon": 0.18, "epsilon_min": 0.06,
    #     "memory_prune_recent": 0.16,
    #     "memory_prune_low":    0.20,     # strongest PER bias here
    #     "opponent": [0.00, 0.05, 0.55, 0.35, 0.05, 0.00, 0.00, 0.00],  # L2 = 0.55
    # },
    
    # "Relax_R1122": {
    #     "duration": 300,
    #     "weights": [0.00, 0.28, 0.50, 0.22, 0.00, 0.00, 0.00, 0.00],
    #     "epsilon": 0.26, "epsilon_min": 0.10,
    #     "memory_prune_recent": 0.08,
    #     "memory_prune_low":    0.08,
    #     "opponent": [0.30, 0.48, 0.22, 0.00, 0.00, 0.00, 0.00, 0.00],
    # },
    
    # # --- endcap: push both directions evenly, then short anchor ---
    # "Mixed12": {
    #     "duration": 320,
    #     "weights": [0.00, 0.24, 0.26, 0.22, 0.18, 0.10, 0.00, 0.00],  # introduce a touch of L4
    #     "epsilon": 0.36, "epsilon_min": 0.08,
    #     "memory_prune_recent": 0.10,
    #     "memory_prune_low":    0.10,
    #     "opponent": [0.00, 0.50, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00],
    # },
    
    # "Challenge_L1L2_HeadToHead": {
    #     "duration": 320,
    #     "weights": [0.00, 0.25, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00],
    #     "epsilon": 0.34, "epsilon_min": 0.08,
    #     "memory_prune_recent": 0.12,
    #     "memory_prune_low":    0.12,
    #     "opponent": [0.00, 0.25, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00],
    # },
    
    # "SelfPlay_Final_L1L2": {
    #     "duration": 250,
    #     "weights": [],
    #     "epsilon": 0.0, "epsilon_min": 0.05,
    #     "memory_prune_recent": 0.20,
    #     "memory_prune_low":    0.12,
    #     "opponent": None,
    # },
    
    # ############# L1 recover ####################
    
    # "L1_Focus_Recover": {           
    #     "duration": 350,
    #     "weights": [0.00, 0.85, 0.15, 0.00, 0, 0, 0, 0],
    #     "epsilon": 0.40, "epsilon_min": 0.06,
    #     "memory_prune_recent": 0.15,
    #     "memory_prune_low":    0.22,
    #     "opponent": [0.00, 0.78, 0.22, 0, 0, 0, 0, 0],
    # },

    # "L1_Focus_Hard": {             
    #     "duration": 350,
    #     "weights": [0.00, 0.90, 0.10, 0.00, 0, 0, 0, 0],
    #     "epsilon": 0.30, "epsilon_min": 0.06,
    #     "memory_prune_recent": 0.10,
    #     "memory_prune_low":    0.30,  # lean into hard mistakes
    #     "opponent": [0.00, 0.85, 0.15, 0, 0, 0, 0, 0],
    # },
    
    # "SelfPlay_L1_recover": {      
    #     "duration": 200,
    #     "weights": [0, 1, 0, 0, 0, 0, 0, 0],
    #     "epsilon": 0.0, "epsilon_min": 0.05,
    #     "memory_prune_recent": 0.30,
    #     "memory_prune_low":    0.15,
    #     "opponent": None,
    # },

    # # ---------- L2 On‑Ramp (scaffolded) ----------
    # "L2_Warmup": {
    #     "duration": 300,
    #     "weights": [0.00, 0.55, 0.35, 0.10, 0.00, 0.00, 0.00, 0.00],  # keep L1 dominant
    #     "epsilon": 0.28, "epsilon_min": 0.08,
    #     "memory_prune_recent": 0.12,
    #     "memory_prune_low":    0.18,   # bias toward high-|δ| mistakes
    #     "opponent": [0.25, 0.55, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00], # small L2 drip
    # },
    
    # "L2_Patterns": {
    #     "duration": 320,
    #     "weights": [0.00, 0.40, 0.45, 0.15, 0.00, 0.00, 0.00, 0.00],  # L2 > L1
    #     "epsilon": 0.22, "epsilon_min": 0.06,
    #     "memory_prune_recent": 0.16,
    #     "memory_prune_low":    0.22,
    #     "opponent": [0.00, 0.25, 0.60, 0.15, 0.00, 0.00, 0.00, 0.00],  # emphasize L2 threats
    # },
    
    # "L2_Consolidate": {
    #     "duration": 300,
    #     "weights": [0.00, 0.30, 0.55, 0.15, 0.00, 0.00, 0.00, 0.00],
    #     "epsilon": 0.20, "epsilon_min": 0.06,
    #     "memory_prune_recent": 0.10,
    #     "memory_prune_low":    0.18,
    #     "opponent": [0.10, 0.30, 0.55, 0.05, 0.00, 0.00, 0.00, 0.00],
    # },
    
    # "SelfPlay_L2_anchor": {
    #     "duration": 220,
    #     "weights": [],
    #     "epsilon": 0.0, "epsilon_min": 0.05,
    #     "memory_prune_recent": 0.20,
    #     "memory_prune_low":    0.12,
    #     "opponent": None,
    # },
    
    # "Fixed2_probe": {  # brief probe, not a long grind
    #     "duration": 200,
    #     "weights": [0.00, 0.20, 0.65, 0.15, 0.00, 0.00, 0.00, 0.00],
    #     "epsilon": 0.18, "epsilon_min": 0.06,
    #     "memory_prune_recent": 0.12,
    #     "memory_prune_low":    0.22,
    #     "opponent": [0.00, 0.10, 0.90, 0.00, 0.00, 0.00, 0.00, 0.00],  # pure L2 opponent
    # },


    # # -------------------- L2 block --------------------
    # "Mixed123": {
    #     "duration": 500,
    #     "weights":   [0.00, 0.30, 0.30, 0.30, 0.10, 0.00, 0.00, 0.00],
    #     "epsilon": 0.40, "epsilon_min": 0.06,
    #     # "memory_prune": 0.10,
    #     "memory_prune_recent": 0.55,
    #     "memory_prune_low":    0.10,
    #     "opponent":  [0.00, 1/3, 1/3, 1/3, 0.00, 0.00, 0.00, 0.00],
    # },
    # "Fixed2": {
    #     "duration": 500,
    #     "weights":   [0.05, 0.20, 0.45, 0.20, 0.10, 0.00, 0.00, 0.00],
    #     "epsilon": 0.35, "epsilon_min": 0.05,
    #     # "memory_prune": 0.10,
    #     "memory_prune_recent": 0.55,
    #     "memory_prune_low":    0.10,
    #     "opponent":  [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    # },
    # "Fixed2_Reprise": {
    #     "duration": 500,
    #     "weights":   [0.00, 0.10, 0.50, 0.25, 0.10, 0.05, 0.00, 0.00],
    #     "epsilon": 0.40, "epsilon_min": 0.05,
    #     # "memory_prune": 0.05,
    #     "memory_prune_recent": 0.50,
    #     "memory_prune_low":    0.10,
    #     "opponent":  [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    # },
    # "Breather_L2_Consolidate": {
    #     "duration": 500,
    #     "weights":   [0.00, 0.10, 0.60, 0.20, 0.10, 0.00, 0.00, 0.00],
    #     "epsilon": 0.20, "epsilon_min": 0.05,
    #     # "memory_prune": 0.10,
    #     "memory_prune_recent": 0.50,
    #     "memory_prune_low":    0.10,
    #     "opponent":  [0.30, 0.30, 0.40, 0.00, 0.00, 0.00, 0.00, 0.00],
    # },
    # "SelfPlay_L2": {
    #     "duration": 500,
    #     "weights":   [],
    #     "epsilon": 0.0, "epsilon_min": 0.05,
    #     # "memory_prune": 0.15,
    #     "memory_prune_recent": 0.70,
    #     "memory_prune_low":    0.15,
    #     "opponent": None,
    # },

    # # -------------------- L3 block --------------------
    # "Variable23": {
    #     "duration": 500,
    #     "weights":   [0.00, 0.10, 0.40, 0.40, 0.05, 0.05, 0.00, 0.00],
    #     "epsilon": 0.40, "epsilon_min": 0.05,
    #     # "memory_prune": 0.10,
    #     "memory_prune_recent": 0.55,
    #     "memory_prune_low":    0.15,
    #     "opponent":  [0.00, 0.00, 0.50, 0.50, 0.00, 0.00, 0.00, 0.00],
    # },
    # "Variable3": {
    #     "duration": 500,
    #     "weights":   [0.00, 0.05, 0.35, 0.50, 0.05, 0.05, 0.00, 0.00],
    #     "epsilon": 0.40, "epsilon_min": 0.05,
    #     # "memory_prune": 0.10,
    #     "memory_prune_recent": 0.55,
    #     "memory_prune_low":    0.15,
    #     "opponent":  [0.00, 1/3, 1/3, 1/3, 0.00, 0.00, 0.00, 0.00],
    # },
    # "Breather_L2L3_Blend": {
    #     "duration": 200,
    #     "weights":   [0.00, 0.05, 0.40, 0.40, 0.15, 0.00, 0.00, 0.00],
    #     "epsilon": 0.25, "epsilon_min": 0.05,
    #     # "memory_prune": 0.10,
    #     "memory_prune_recent": 0.50,
    #     "memory_prune_low":    0.10,
    #     "opponent":  [0.00, 0.00, 0.50, 0.50, 0.00, 0.00, 0.00, 0.00],
    # },
    # "Fixed3": {
    #     "duration": 500,
    #     "weights":   [0.00, 0.00, 0.20, 0.50, 0.20, 0.10, 0.00, 0.00],
    #     "epsilon": 0.40, "epsilon_min": 0.05,
    #     # "memory_prune": 0.10,
    #     "memory_prune_recent": 0.55,
    #     "memory_prune_low":    0.15,
    #     "opponent":  [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
    # },
    # "SelfPlay_L3": {
    #     "duration": 500,
    #     "weights":   [],
    #     "epsilon": 0.0, "epsilon_min": 0.05,
    #     # "memory_prune": 0.05,
    #     "memory_prune_recent": 0.65,
    #     "memory_prune_low":    0.15,
    #     "opponent": None,
    # },

    # # -------------------- Final --------------------
    # "Final": {
    #     "duration": 500,
    #     "weights":   [],
    #     "epsilon": 0.0, "epsilon_min": 0.0,
    #     # "memory_prune": 0.05,
    #     "memory_prune_recent": 0.50,
    #     "memory_prune_low":    0.10,
    #     "opponent": None,
    # },
}

def set_training_phases_length(DICT):
    length = 0
    for name, TF in DICT.items():
        length += int(TF["duration"])
        TF["length"] = length
        TF["sumWeights"] = float(np.sum(TF.get("weights", []))) if TF.get("weights") is not None else 0.0
        opp = TF.get("opponent", [])
        TF["sumOppo"] = (float(np.sum(opp)) if isinstance(opp, (list, tuple, np.ndarray))
                         else (0.0 if opp is None else float(opp)))
    display_dict_as_tab(DICT)

def display_dict_as_tab(DICT):
    DF = pd.DataFrame.from_dict(DICT, orient="index")
    cols = [
        "length", "duration",
        "epsilon", "epsilon_min",
        # "memory_prune",               # old single knob, hidden from table but kept in dict comments
        "memory_prune_recent", "memory_prune_low",
        "sumWeights", "sumOppo"
    ]
    DF = DF[cols]
    DF = DF.sort_values(by=['length'], ascending=True)
    display(HTML(DF.to_html()))
