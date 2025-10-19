# dqn_model.py
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape=(6, 7), output_size=7):
        super().__init__()
        self.H, self.W = input_shape

        # --- precompute positional buffers (1,1,H,W) ---
        H, W = self.H, self.W
        r = torch.arange(H).view(1, 1, H, 1).float()
        c = torch.arange(W).view(1, 1, 1, W).float()
        denom = (H - 1) + (W - 1)
        diag_diff = (r - c + (W - 1)) / denom   # normalized r-c   in [0,1]
        diag_sum  = (r + c) / denom             # normalized r+c   in [0,1]
        self.register_buffer("diag_diff", diag_diff)  # shape (1,1,H,W)
        self.register_buffer("diag_sum",  diag_sum)   # shape (1,1,H,W)

        in_ch = 6  # 2 game + 2 (row,col) + 2 (diag_diff,diag_sum)

        # ---- Feature extractor ----
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # -> (B,64,3,3)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # -> (B,64,1,1)

        # ---- Dueling heads ----
        self.val_head = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Linear(128, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Linear(128, output_size)
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

    def _append_diagonals(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W). If C==4 (me, opp, row, col), append 2 diagonal planes.
        If C==6 already, return as-is.
        """
        if x.shape[1] == 6:
            return x
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4 or 6 channels, got {x.shape[1]}")
        B = x.shape[0]
        # reuse precomputed (1,1,H,W) buffers and expand to batch
        d1 = self.diag_diff.expand(B, -1, -1, -1)
        d2 = self.diag_sum.expand(B, -1, -1, -1)
        return torch.cat([x, d1, d2], dim=1)  # -> (B,6,H,W)

    def forward(self, x):
        # x: (B, 4, 6, 7)  [me, opp, row, col]
        x = self._append_diagonals(x)         # -> (B, 6, 6, 7)
        x = self.conv(x)                      # -> (B, 64, 3, 3)
        x = self.gap(x).flatten(1)            # -> (B, 64)
        V = self.val_head(x)                  # -> (B, 1)
        A = self.adv_head(x)                  # -> (B, 7)
        return V + (A - A.mean(dim=1, keepdim=True))
