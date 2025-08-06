# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 18:17:00 2025

@author: Uporabnik
"""

import random
from collections import deque

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

    def prune(self, fraction=0.8):
        if fraction > 1.0:
            return
        prune_size = int(len(self.memory) * fraction)
        if prune_size == 0:
            return
        self.memory = deque(list(self.memory)[prune_size:], maxlen=self.memory.maxlen)
