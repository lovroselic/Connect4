# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 18:46:31 2025

@author: Uporabnik
"""

# ppo_memory.py

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.valid_actions_list = []  # ⬅️ new field

    def store(self, state, action, log_prob, reward, done, value, valid_actions):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.valid_actions_list.append(valid_actions)  # ⬅️ store valid actions

    def clear(self):
        self.__init__()

