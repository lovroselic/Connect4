# DQN/dueling_qnet192.py
from __future__ import annotations

#from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNet192(nn.Module):
    """
    Dueling Q network matching CNet192 trunk:
      - conv1 (k=4) -> optional conv_mid (k=3) -> conv2 (k=2) -> fc(->192)
      - advantage head: 192 -> 192 -> 7
      - value head:     192 -> 192 -> 1
      - Q = V + (A - mean(A))
    """

    def __init__(self, in_channels: int = 1, use_mid_3x3: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.use_mid_3x3 = bool(use_mid_3x3)

        self.conv1 = nn.Conv2d(self.in_channels, 192, kernel_size=4, padding=0)  # 6x7 -> 3x4
        self.conv_mid = nn.Conv2d(192, 192, kernel_size=3, padding=1) if self.use_mid_3x3 else None
        self.conv2 = nn.Conv2d(192, 192, kernel_size=2, padding=0)  # 3x4 -> 2x3

        # infer flatten size robustly
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, 6, 7)
            z = self._forward_conv(dummy)
            self.flat = int(z.numel() // z.shape[0])

        self.fc = nn.Linear(self.flat, 192)

        self.adv_fc = nn.Linear(192, 192)
        self.adv_out = nn.Linear(192, 7)

        self.val_fc = nn.Linear(192, 192)
        self.val_out = nn.Linear(192, 1)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        if self.conv_mid is not None:
            x = F.relu(self.conv_mid(x))
        x = F.relu(self.conv2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, 6, 7)
        returns Q: (B, 7)
        """
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        adv = F.relu(self.adv_fc(x))
        adv = self.adv_out(adv)  # (B,7)

        val = F.relu(self.val_fc(x))
        val = self.val_out(val)  # (B,1)

        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q

    def freeze_conv_block(self, freeze: bool = True) -> None:
        """
        Freeze only conv layers (transfer-style). FC+heads stay trainable.
        """
        req = not bool(freeze)
        for p in self.conv1.parameters():
            p.requires_grad = req
        if self.conv_mid is not None:
            for p in self.conv_mid.parameters():
                p.requires_grad = req
        for p in self.conv2.parameters():
            p.requires_grad = req

    @torch.no_grad()
    def init_from_cnet192(self, cnet) -> None:
        """
        Initialize weights from a CNet192 instance:
          - convs + fc
          - policy head -> advantage head
          - value head  -> value head
        """
        self.conv1.load_state_dict(cnet.conv1.state_dict(), strict=True)
        if self.conv_mid is not None and getattr(cnet, "conv_mid", None) is not None:
            self.conv_mid.load_state_dict(cnet.conv_mid.state_dict(), strict=True)
        self.conv2.load_state_dict(cnet.conv2.state_dict(), strict=True)

        self.fc.load_state_dict(cnet.fc.state_dict(), strict=True)

        # policy logits are a perfect "advantage-like" initializer
        self.adv_fc.load_state_dict(cnet.policy_fc.state_dict(), strict=True)
        self.adv_out.load_state_dict(cnet.policy_out.state_dict(), strict=True)

        self.val_fc.load_state_dict(cnet.value_fc.state_dict(), strict=True)
        self.val_out.load_state_dict(cnet.value_out.state_dict(), strict=True)
