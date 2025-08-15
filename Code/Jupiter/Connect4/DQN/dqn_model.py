# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 18:26:15 2025

@author: LOvro
"""

# dqn_model.py

import torch.nn as nn

class DQN(nn.Module):
    """
    Compact Dueling DQN with Global Average Pooling (GAP) + explicit weight init.
    Input:  (B, 6, 7) or (B, 1, 6, 7)
    Output: (B, 7) Q-values, one per column.
    """
    def __init__(self, input_shape=(6, 7), output_size=7):
        super().__init__()
        c = 2  # dual channel board, per player )1,-1)

        # ---- Feature extractor ----
        self.conv = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Global Average Pool over HxW -> (B, 128)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ---- Dueling heads ----
        self.val_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_size)
        )

        # Apply weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 2, 6, 7) with values in {0,1}
        feat = self.conv(x)
        feat = self.gap(feat).flatten(1)
        V = self.val_head(feat)
        A = self.adv_head(feat)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q
