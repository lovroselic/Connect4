# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 18:34:01 2025

@author: Lovro
"""
# actor_critic.py

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from C4.connect4_lookahead import Connect4Lookahead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Lookahead = Connect4Lookahead()  # instantiate once for reuse

class ActorCritic(nn.Module):
    def __init__(self, input_shape=(6, 7), output_size=7):
        super(ActorCritic, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            conv_out_size = self.conv_layers(dummy).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(128, output_size)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        policy_logits = self.policy_head(x)          # ⬅️ raw logits
        value = self.value_head(x)
        return policy_logits, value                   # ⬅️ no softmax here

    def act(self, state, valid_actions=None):
        state = state.to(DEVICE)
        logits, value = self.forward(state)

        if valid_actions is not None:
            mask = torch.full_like(logits, -float('inf'))
            mask[0, valid_actions] = 0
            logits = logits + mask                    # ⬅️ mask applied to logits

        probs = F.softmax(logits, dim=-1)             # ⬅️ softmax only here
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
