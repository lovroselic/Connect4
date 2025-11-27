# DQN/dqn_model_qr.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# =======================
#  Building blocks
# =======================

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
    """
    def __init__(self, in_ch: int, use_diag4: bool = False):
        super().__init__()
        # Three primary directional paths
        self.h4 = nn.Conv2d(in_ch, 16, kernel_size=(1, 4), padding=(0, 1))  # 6x7 -> 6x6
        self.v4 = nn.Conv2d(in_ch, 16, kernel_size=(4, 1), padding=(1, 0))  # 6x7 -> 5x7
        self.k2 = nn.Conv2d(in_ch, 16, kernel_size=(2, 2), padding=0)       # 6x7 -> 5x6

        # IMPORTANT: keep off (no upsampling path); three paths only
        self.use_diag4 = False

        # Symmetric shrinkers (equivariant under left–right / top–bottom)
        self.shrink_h = nn.AvgPool2d(kernel_size=(2, 1), stride=(1, 1))  # 6x6 -> 5x6
        self.shrink_w = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 1))  # 5x7 -> 5x6

        self.mix = nn.Conv2d(48, 64, kernel_size=3, padding=1)  # 3 paths → 48 ch
        self.res = ResidualBlock(64)
        self.pool = nn.MaxPool2d(kernel_size=2)                 # 5x6 -> 2x3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.h4(x), inplace=True)     # (B,16,6,6)
        v = F.relu(self.v4(x), inplace=True)     # (B,16,5,7)
        k = F.relu(self.k2(x), inplace=True)     # (B,16,5,6)

        h = self.shrink_h(h)                     # -> 5x6
        v = self.shrink_w(v)                     # -> 5x6

        x = torch.cat([h, v, k], dim=1)          # (B,48,5,6)
        x = F.relu(self.mix(x), inplace=True)
        x = self.res(x)
        x = self.pool(x)                         # (B,64,2,3)
        return x


# =======================
#  QR-DQN model
# =======================

class QRDQN(nn.Module):
    """
    Quantile Regression DQN (QR-DQN) with the same directional backbone you use in DQN.
    Input: (B, 4, 6, 7) = [me, opp, row, col] with row/col already in [-1,1].
    We append 2 diagonal planes built from the CURRENT row/col: [row-col, row+col].
    Output: (B, 7, N) quantile values per action.
    """
    def __init__(
        self,
        input_shape=(6, 7),
        output_size: int = 7,
        n_quantiles: int = 51,
        use_diag4: bool = True,  # kept for API compat; backbone pins this off internally
        # --- init knobs for action prior ---
        init_center_bias_scale: float = 2e-3,
        init_center_sigma: float = 1.25,
        # control whether to enforce LR symmetry on conv filters at init
        enforce_lr_symmetry: bool = True,
        # control whether to zero the head weights at init (makes ZERO depend only on bias)
        zero_head_weights: bool = True,
    ):
        super().__init__()
        self.H, self.W = input_shape
        self.output_size = int(output_size)
        self.n_quantiles = int(n_quantiles)

        # Feature extractor
        self.backbone = C4DirectionalBackbone(in_ch=6, use_diag4=False)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # -> (B,64,1,1)

        # Small MLP before quantile head
        self.fc = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(inplace=True)
        )
        # Quantile head: 7 actions × N quantiles
        self.quantile_head = nn.Linear(128, self.output_size * self.n_quantiles)

        # store init knobs for reproducibility
        self._init_center_bias_scale = float(init_center_bias_scale)
        self._init_center_sigma = float(init_center_sigma)
        self._enforce_lr_symmetry_flag = bool(enforce_lr_symmetry)
        self._zero_head_weights_flag = bool(zero_head_weights)

        self._init_weights()

    # ---------- utils ----------
    def _append_diagonals(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build diag planes from the CURRENT row/col channels so that a horizontal flip
        (which reverses the col plane) transforms diags consistently.
        x: (B, 4, 6, 7) with channels [me, opp, row, col] in [-1, 1]
        """
        if x.shape[1] != 4:
            x = x[:, :4, :, :]
        row = x[:, 2:3, :, :]   # [-1, 1]
        col = x[:, 3:4, :, :]   # [-1, 1]
        diag_diff = row - col   # flips → row - (-col) = row + col
        diag_sum  = row + col   # flips → row + (-col) = row - col
        return torch.cat([x, diag_diff, diag_sum], dim=1)  # (B,6,6,7)

    @staticmethod
    def _enforce_lr_symmetry_(module: nn.Module):
        """
        Make every Conv2d LR-equivariant by averaging kernels with their left-right flips.
        """
        with torch.no_grad():
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.copy_(0.5 * (m.weight + torch.flip(m.weight, dims=[-1])))
                    # bias is fine as-is

    def _init_weights(self):
        # Kaiming for convs/linears
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Make the quantile head gentle at start:
        # 1) optional: zero head weights so init Q depends solely on bias prior
        with torch.no_grad():
            if self._zero_head_weights_flag:
                self.quantile_head.weight.zero_()
            else:
                self.quantile_head.weight.mul_(0.10)

            if self.quantile_head.bias is None:
                self.quantile_head.bias = nn.Parameter(torch.zeros(self.output_size * self.n_quantiles))

            # ----- action prior: center peak, symmetric decay -----
            a = torch.arange(self.output_size, dtype=torch.float32)
            mu = 3.0
            sigma = max(1e-6, self._init_center_sigma)
            gauss = torch.exp(-0.5 * ((a - mu) / sigma) ** 2)           # unnormalized
            gauss = gauss / (gauss.max() + 1e-12)                        # peak = 1
            bias_per_action = self._init_center_bias_scale * gauss       # [7], peak at column 3
            bias_per_action = bias_per_action - bias_per_action.mean()   # zero-mean

            # copy same bias to all quantiles for each action
            bias = self.quantile_head.bias.view(self.output_size, self.n_quantiles)  # (7, N)
            bias.copy_(bias_per_action.view(-1, 1).expand_as(bias))
            self.quantile_head.bias.copy_(bias.view(-1))

        # 2) optionally enforce LR symmetry on all convs so RAW mirror holds at init
        if self._enforce_lr_symmetry_flag:
            self._enforce_lr_symmetry_(self)

    # ---------- forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns quantiles: shape [B, 7, N]
        x: (B, 4, 6, 7)  [me, opp, row, col], row/col already in [-1,1].
        """
        x = x.clone()
        x = self._append_diagonals(x)         # (B, 6, 6, 7)
        x = self.backbone(x)                  # (B, 64, 2, 3)
        x = self.gap(x).flatten(1)            # (B, 64)
        x = self.fc(x)                        # (B, 128)
        q = self.quantile_head(x)             # (B, 7*N)
        q = q.view(q.size(0), self.output_size, self.n_quantiles)  # (B, 7, N)
        return q

    @torch.no_grad()
    def q_mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience: expected Q = mean over quantiles. Shape [B, 7]
        """
        return self.forward(x).mean(dim=-1)
