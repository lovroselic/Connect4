# dqn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Simple 3x3 -> ReLU -> 3x3 residual block (same channels).
    Keeps HxW; padding=1 to preserve spatial size.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x), inplace=True)
        h = self.conv2(h)
        return F.relu(h + x, inplace=True)


class C4DirectionalBackbone(nn.Module):
    """
    Direction-aware feature extractor for Connect-4.
    Parallel (1x4), (4x1), (2x2) paths catch horizontal, vertical, and local clamps.
    On 6x7 input (with 6 channels) -> (B, 64, 2, 3) before GAP.
    Adds one residual mixing block after the 3x3 'mix' conv.
    """
    def __init__(self, in_ch: int, use_diag4: bool = False):
        super().__init__()
        # Three primary directional paths
        self.h4 = nn.Conv2d(in_ch, 16, kernel_size=(1, 4), padding=(0, 1))  # 6x7 -> 6x6
        self.v4 = nn.Conv2d(in_ch, 16, kernel_size=(4, 1), padding=(1, 0))  # 6x7 -> 5x7
        self.k2 = nn.Conv2d(in_ch, 16, kernel_size=(2, 2), padding=0)       # 6x7 -> 5x6

        # Optional extra diagonal-ish context (4x4). Off by default.
        self.use_diag4 = bool(use_diag4)
        if self.use_diag4:
            self.d4 = nn.Conv2d(in_ch, 16, kernel_size=(4, 4), padding=0)   # 6x7 -> 3x4

        # Mix into 64 channels (align all to 5x6), then residual block and pool
        in_mix = 48 + (16 if self.use_diag4 else 0)
        self.mix = nn.Conv2d(in_mix, 64, kernel_size=3, padding=1)          # -> 5x6
        self.res = ResidualBlock(64)                                        # -> 5x6
        self.pool = nn.MaxPool2d(kernel_size=2)                             # 5x6 -> 2x3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Directional paths
        h = F.relu(self.h4(x), inplace=True)             # (B,16,6,6)
        v = F.relu(self.v4(x), inplace=True)             # (B,16,5,7)
        k = F.relu(self.k2(x), inplace=True)             # (B,16,5,6)

        # Align to common 5x6 canvas
        h = h[:, :, :5, :6]                              # 6x6 -> 5x6
        v = v[:, :, :5, :6]                              # 5x7 -> 5x6
        parts = [h, v, k]

        if self.use_diag4:
            d = F.relu(self.d4(x), inplace=True)         # (B,16,3,4)
            # centrally upsample/align to 5x6 with nearest (lightweight & fine here)
            d = F.interpolate(d, size=(5, 6), mode="nearest")
            parts.append(d)

        x = torch.cat(parts, dim=1)                      # (B, 48 or 64, 5, 6)
        x = F.relu(self.mix(x), inplace=True)            # (B,64,5,6)
        x = self.res(x)                                  # (B,64,5,6)
        x = self.pool(x)                                 # (B,64,2,3)
        return x


class DQN(nn.Module):
    """
    Dueling DQN with a directional backbone (+ residual mixing block).
    Expects mover-first planes: x: (B, 4, 6, 7) = [me, opp, row, col].
    Internally appends 2 diagonal coord planes -> 6 channels total.
    Returns Q(s,·): (B, 7).
    """
    def __init__(self, input_shape=(6, 7), output_size=7, use_diag4: bool = True):
        super().__init__()
        self.H, self.W = input_shape

        # --- precompute positional buffers (1,1,H,W) ---
        H, W = self.H, self.W
        r = torch.arange(H).view(1, 1, H, 1).float()
        c = torch.arange(W).view(1, 1, 1, W).float()
        denom = (H - 1) + (W - 1)
        diag_diff = (r - c + (W - 1)) / denom   # normalized r-c in [0,1]
        diag_sum  = (r + c) / denom             # normalized r+c in [0,1]
        self.register_buffer("diag_diff", diag_diff)  # (1,1,H,W)
        self.register_buffer("diag_sum",  diag_sum)   # (1,1,H,W)

        self.backbone = C4DirectionalBackbone(in_ch=6, use_diag4=use_diag4)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # -> (B,64,1,1)

        # ---- Dueling heads ----
        self.val_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.adv_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Linear(128, output_size))
        self._init_weights()

    # ---------- utils ----------
    def _append_diagonals(self, x: torch.Tensor) -> torch.Tensor:
        """
        If x has 4 channels (me, opp, row, col), append 2 diagonal planes.
        """
        if x.shape[1] == 6:
            return x
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4 or 6 channels, got {x.shape[1]}")
        B = x.shape[0]
        d1 = self.diag_diff.expand(B, -1, -1, -1)
        d2 = self.diag_sum.expand(B, -1, -1, -1)
        return torch.cat([x, d1, d2], dim=1)  # (B,6,H,W)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Mild scaling for the last layers + slight value bias to break symmetry
        with torch.no_grad():
            self.val_head[-1].weight.mul_(0.5)
            self.adv_head[-1].weight.mul_(0.5)
            self.val_head[-1].bias.fill_(0.3)
            if self.adv_head[-1].bias is not None:
                self.adv_head[-1].bias.zero_()

    # ---------- forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, 6, 7)  [me, opp, row, col]
        x = self._append_diagonals(x)        # (B, 6, 6, 7)
        x = self.backbone(x)                 # (B, 64, 2, 3)
        x = self.gap(x).flatten(1)           # (B, 64)
        V = self.val_head(x)                 # (B, 1)
        A = self.adv_head(x)                 # (B, 7)
        return V + (A - A.mean(dim=1, keepdim=True))
