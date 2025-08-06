# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 18:26:15 2025

@author: Uporabnik
"""

import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape=(6, 7), output_size=7):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size

        h, w = input_shape

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        # Compute conv output size dynamically
        dummy_input = torch.zeros(1, 1, *input_shape)
        with torch.no_grad():
            conv_out_size = self.conv(dummy_input).shape[1]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv(x)
        return self.fc(x)
