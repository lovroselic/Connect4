#dqn_model.py
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape=(6, 7), output_size=7):
        super().__init__()
        self.H, self.W = input_shape
        c = 4  # 2 game channels + 2 positional encodings

        # ---- Feature extractor ----
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=2, padding=0),  # small patterns
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # mid patterns
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, padding=1), # larger blobs
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                 # (B, 64, 3, 3)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))          # (B, 64, 1, 1)

        # ---- Dueling heads ----
        self.val_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_size)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        with torch.no_grad():
            self.val_head[-1].weight.mul_(0.5)
            self.adv_head[-1].weight.mul_(0.5)
            self.val_head[-1].bias.fill_(0.3)
            if self.adv_head[-1].bias is not None:
                self.adv_head[-1].bias.zero_()

    def _add_positional_planes(self, x: torch.Tensor) -> torch.Tensor:
        """Append normalized row and column encodings to input tensor."""
        B, _, H, W = x.shape
        device = x.device

        row_plane = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        col_plane = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)

        # Normalize to [0,1]
        row_plane = row_plane.float() / (H - 1)
        col_plane = col_plane.float() / (W - 1)

        return torch.cat([x, row_plane, col_plane], dim=1)

    def forward(self, x):
        # x: (B, 4, 6, 7)
        x = self.conv(x)                     # (B, 64, 3, 3)
        x = self.gap(x).flatten(1)           # (B, 64)
        V = self.val_head(x)                 # (B, 1)
        A = self.adv_head(x)                 # (B, 7)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q
