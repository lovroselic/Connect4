# az_net.py
# Compact AlphaZero-style CNN (policy+value) for 6x7 Connect-4.

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

# Robust import (works whether files are in a package folder A0/ or flat):
try:
    from A0.a0_utilities import mask_logits
except Exception:
    from a0_utilities import mask_logits

class ResidBlock(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)

class AZNet(nn.Module):
    def __init__(self, in_ch: int = 4, board_shape=(6, 7), channels: int = 64, blocks: int = 4):
        super().__init__()
        H, W = board_shape
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidBlock(channels) for _ in range(blocks)])

        # Policy head
        self.p_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.p_bn   = nn.BatchNorm2d(2)
        self.p_fc   = nn.Linear(2 * H * W, 7)

        # Value head
        self.v_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.v_bn   = nn.BatchNorm2d(1)
        self.v_fc1  = nn.Linear(H * W, 64)
        self.v_fc2  = nn.Linear(64, 1)

    def forward(self, x, legal_mask=None):
        # Feature tower
        h = self.tower(self.stem(x))

        # Policy
        p = F.relu(self.p_bn(self.p_conv(h))).flatten(1)   # [B, 2*H*W]
        p = self.p_fc(p)                                   # [B, 7]
        if legal_mask is not None:
            p = mask_logits(p, legal_mask)                 # −inf on illegal

        # Value
        v = F.relu(self.v_bn(self.v_conv(h))).flatten(1)   # [B, H*W]
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)          # [B]

        return p, v
