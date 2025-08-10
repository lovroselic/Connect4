# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 18:50:36 2025

@author: LOVRO
"""

# get_phase.py

from DQN.training_phases_config import TRAINING_PHASES

def get_phase(episode):
    for phase_name, phase_data in TRAINING_PHASES.items():
        end_episode = phase_data["length"]
        if end_episode is None or episode < end_episode:
            return (
                phase_name,
                phase_data["weights"],
                phase_data["epsilon"],
                phase_data["memory_prune"]
            )
