# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 18:17:00 2025

@author: Lovro
"""

import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'player', 'action', 'reward', 'next_state', 'next_player', 'done'))

class ReplayMemory:
    def __init__(self, capacity=100000):
        self.memory = deque(maxlen=capacity)

    def push(self, s, p, a, r, s2, p2, done):
        self.memory.append(Transition(s, p, a, r, s2, p2, done))

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
