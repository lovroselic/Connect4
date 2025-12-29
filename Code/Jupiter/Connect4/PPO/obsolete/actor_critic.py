# PPO/actor_critic.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


# Device is resolved once here; model can be moved later with .to(DEVICE)
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEG_INF = -1e9  # large negative for masked logits
CENTER_COL = 3

def _lowest_empty_row(board: np.ndarray, col: int) -> int | None:
    """
    Return the bottom-most empty row index for a column in a 6x7 board.
    Board uses values {-1, 0, +1}. None if column is full.
    """
    rows, _ = board.shape
    for r in range(rows - 1, -1, -1):  # bottom → top
        if board[r, col] == 0:
            return r
    return None


def _has_four_from(board: np.ndarray, row: int, col: int, player: int) -> bool:
    """
    Check if 'player' has a 4-in-a-row passing through (row, col).
    """
    rows, cols = board.shape
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
        count = 1

        rr, cc = row + dr, col + dc
        while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == player:
            count += 1
            rr += dr
            cc += dc

        rr, cc = row - dr, col - dc
        while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == player:
            count += 1
            rr -= dr
            cc -= dc

        if count >= 4:
            return True

    return False


def _is_winning_drop(board: np.ndarray, col: int, player: int) -> bool:
    """
    If 'player' drops in 'col', does it immediately win?
    """
    r = _lowest_empty_row(board, col)
    if r is None:
        return False

    old = board[r, col]
    board[r, col] = player
    win = _has_four_from(board, r, col, player)
    board[r, col] = old
    return win


def _find_immediate_wins(board: np.ndarray, player: int, legal_actions: list[int]) -> list[int]:
    """Return list of columns that are immediate wins for 'player'."""
    wins = []
    for c in legal_actions:
        if _is_winning_drop(board, c, player):
            wins.append(int(c))
    return wins


def _tactical_guard(
    board: np.ndarray,
    legal_actions: list[int],
    me: int = +1,
    opp: int = -1,
) -> tuple[int | None, list[int]]:
    """
    Basic tactical guard (like in DQN):

      • If opponent has exactly one immediate win: must block that column.
      • Otherwise, return a 'safe_actions' subset where, after we play,
        opponent has no 1-ply win.

    Returns:
      (must_play, safe_actions)
    """
    # Opponent immediate wins now
    opp_wins_now = _find_immediate_wins(board, opp, legal_actions)
    if len(opp_wins_now) == 1 and opp_wins_now[0] in legal_actions:
        return opp_wins_now[0], legal_actions

    # Safe actions: after we move, opponent has no immediate win
    safe_actions: list[int] = []
    for c in legal_actions:
        r = _lowest_empty_row(board, c)
        if r is None:
            continue
        old = board[r, c]
        board[r, c] = me
        opp_wins_after = _find_immediate_wins(board, opp, legal_actions)
        board[r, c] = old

        if len(opp_wins_after) == 0:
            safe_actions.append(int(c))

    return None, safe_actions

def _epsilon_soften_probs(probs: torch.Tensor, mask: torch.Tensor, eps_explore: float) -> torch.Tensor:
    """
    Mix a bit of uniform over LEGAL actions into probs:
      pi_eps = (1 - eps) * pi + eps * U(legal)
    probs: [B,A], mask: [B,A] bool (True = legal)
    """
    if eps_explore <= 0.0:
        return probs

    # Uniform over legal actions
    legal_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
    uniform = mask.float() / legal_counts  # 0 on illegal, 1/|legal| on legal

    mixed = (1.0 - eps_explore) * probs + eps_explore * uniform
    mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return mixed



# =======================
#  Building blocks (from DQN)
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
    def __init__(self, in_ch: int, use_diag4: bool = True):
        super().__init__()
        # Three primary directional paths
        self.h4 = nn.Conv2d(in_ch, 16, kernel_size=(1, 4), padding=(0, 1))  # 6x7 -> 6x6
        self.v4 = nn.Conv2d(in_ch, 16, kernel_size=(4, 1), padding=(1, 0))  # 6x7 -> 5x7
        self.k2 = nn.Conv2d(in_ch, 16, kernel_size=(2, 2), padding=0)       # 6x7 -> 5x6

        # No upsampling path here; three paths only
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
#  Actor-Critic with directional backbone
# =======================

class ActorCritic(nn.Module):
    """
    Connect-4 PPO Actor-Critic with the DQN-style directional backbone.

    Accepted inputs (single or batched):
      Scalar board (values in {-1,0,+1}):
        (6,7), (1,6,7), (B,6,7), (B,1,6,7)
      Plane boards:
        (2,6,7), (B,2,6,7)              -> [me, opp]
        (4,6,7), (B,4,6,7)              -> [me, opp, row, col]
        (6,6,7), (B,6,6,7)              -> [me, opp, row, col, row-col, row+col]

    Notes:
    - Keep boards agent-centric upstream (agent pieces == +1). If you only have
      an environment-centric board and the current player can be -1, flip with:
        s_agent = ActorCritic.ensure_agent_centric(s_env, current_player)

    - If you perform horizontal-flip augmentation outside the model, prefer to
      feed **4-channel** inputs [me, opp, row, col] so row/col flip along with
      the pieces. If you pass just 2 channels, we synthesize row/col inside,
      which will not reflect your external flip.
    """

    def __init__(
        self,
        action_dim: int = 7,
        init_center_bias_scale: float = 2e-3,
        init_center_sigma: float = 1.25,
        enforce_lr_symmetry: bool = True,
    ):
        super().__init__()
        self.action_dim = int(action_dim)

        # Direction-aware backbone expects 6 channels after appending diagonals
        self.backbone = C4DirectionalBackbone(in_ch=6, use_diag4=False)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # -> (B,64,1,1)

        # Small MLP trunk before heads
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

        # Heads
        self.policy_head = nn.Linear(128, self.action_dim)
        self.value_head  = nn.Linear(128, 1)

        # Coord buffers (fixed for 6x7)
        # row in [-1,1] top->bottom, col in [-1,1] left->right (consistent with DQN)
        row = torch.linspace(-1.0, 1.0, steps=6).view(1, 1, 6, 1).expand(1, 1, 6, 7)
        col = torch.linspace(-1.0, 1.0, steps=7).view(1, 1, 1, 7).expand(1, 1, 6, 7)
        self.register_buffer("coord_row", row)  # (1,1,6,7)
        self.register_buffer("coord_col", col)  # (1,1,6,7)

        # Store init knobs
        self._init_center_bias_scale = float(init_center_bias_scale)
        self._init_center_sigma = float(init_center_sigma)
        self._enforce_lr_symmetry_flag = bool(enforce_lr_symmetry)

        self._init_weights()
        
        # --- Heuristic knobs (phase-dependent) ---
        self.center_start: float = 0.0     # P(center push) on early plies
        self.center_ply_max: int = 1       # only on ply 0/1, like DQN
        self.guard_prob: float = 0.0       # P(activating tactical guard)
        self.guard_ply_min: int = 2        # guard only in this ply window
        self.guard_ply_max: int = 10
        self.win_now_prob: float = 0.0     # P(take 1-ply win when found)
        self.center_forced_used = False
 
        # --- Action-mix stats for plotting ---
        self._act_stats_total: int = 0
        self._act_stats = {
            "win_now": 0,
            "center": 0,
            "guard": 0,
            "policy": 0,
        }


    # ---------- shape/encoding helpers ----------

    @staticmethod
    def ensure_agent_centric(s: np.ndarray | torch.Tensor, current_player: int):
        """
        Returns a scalar-board where agent is always +1.
        If current_player == -1, it flips signs.
        """
        if isinstance(s, np.ndarray):
            return s if current_player == 1 else -s
        t = s
        return t if current_player == 1 else -t

    @staticmethod
    def _planes_from_scalar_board(s: torch.Tensor) -> torch.Tensor:
        """s: [..., 6, 7] with values in {-1,0,+1} -> returns two planes (agent, opp)."""
        agent = (s > 0.5).float()
        opp   = (s < -0.5).float()
        return agent, opp

    def _append_coords(self, x_2ch: torch.Tensor) -> torch.Tensor:
        """
        x_2ch: (B,2,6,7) -> (B,4,6,7) by appending [row, col] from buffers.
        """
        B = x_2ch.size(0)
        row = self.coord_row.expand(B, -1, -1, -1)
        col = self.coord_col.expand(B, -1, -1, -1)
        return torch.cat([x_2ch, row, col], dim=1)

    @staticmethod
    def _append_diagonals(x_4ch: torch.Tensor) -> torch.Tensor:
        """
        Build diag planes from the CURRENT row/col channels so that a horizontal flip
        (which reverses the col plane) transforms diags consistently.
        x_4ch: (B, 4, 6, 7) with channels [me, opp, row, col] in [-1, 1]
        Returns: (B, 6, 6, 7) channels = [me, opp, row, col, row-col, row+col]
        """
        if x_4ch.shape[1] != 4:
            x_4ch = x_4ch[:, :4, :, :]
        row = x_4ch[:, 2:3, :, :]
        col = x_4ch[:, 3:4, :, :]
        diag_diff = row - col
        diag_sum  = row + col
        return torch.cat([x_4ch, diag_diff, diag_sum], dim=1)

    def _to_six_channel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize arbitrary input to (B, 6, 6, 7) on DEVICE.
        Accepts scalar boards, 2-ch, 4-ch, or already-6-ch inputs.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        #x = x.to(dtype=torch.float32, device=DEVICE)
        dev = self.coord_row.device   # buffers move with the model
        x = x.to(dtype=torch.float32, device=dev)

        # Scalar boards
        if x.dim() == 2:  # (6,7)
            a, o = self._planes_from_scalar_board(x)
            x2 = torch.stack([a, o], dim=0).unsqueeze(0)  # (1,2,6,7)
            x4 = self._append_coords(x2)                  # (1,4,6,7)
            return self._append_diagonals(x4)             # (1,6,6,7)

        if x.dim() == 3:
            # Either (C,6,7) single sample OR (B,6,7) batch of scalar boards
            if x.size(1) == 6 and x.size(2) == 7 and x.size(0) in (1, 2, 4, 6):
                # treat as channel-first single example: (C,6,7)
                C = x.size(0)
                if C == 2:
                    x4 = self._append_coords(x.unsqueeze(0))   # -> (1,4,6,7)
                    return self._append_diagonals(x4)          # -> (1,6,6,7)
                if C == 4:
                    return self._append_diagonals(x.unsqueeze(0))
                if C == 6:
                    return x.unsqueeze(0)
                # C == 1 -> scalar
                s = x[0]
                a, o = self._planes_from_scalar_board(s)
                x2 = torch.stack([a, o], dim=0).unsqueeze(0)
                x4 = self._append_coords(x2)
                return self._append_diagonals(x4)
            else:
                # treat as batch of scalar boards: (B,6,7)
                s = x
                a = (s > 0.5).float()
                o = (s < -0.5).float()
                x2 = torch.stack([a, o], dim=1)                # (B,2,6,7)
                x4 = self._append_coords(x2)                   # (B,4,6,7)
                return self._append_diagonals(x4)              # (B,6,6,7)

        if x.dim() == 4:
            # (B,C,6,7)
            _, C = x.size(0), x.size(1)
            if C == 2:
                x4 = self._append_coords(x)
                return self._append_diagonals(x4)
            if C == 4:
                return self._append_diagonals(x)
            if C == 6:
                return x
            if C == 1:
                s = x[:, 0, :, :]
                a, o = self._planes_from_scalar_board(s)
                x2 = torch.stack([a, o], dim=1)                # (B,2,6,7)
                x4 = self._append_coords(x2)
                return self._append_diagonals(x4)
            raise ValueError(f"Unexpected channel count {C}; expected 1/2/4/6.")

        raise ValueError(f"Unexpected input shape {tuple(x.shape)}")

    # ---------- core forward ----------

    def _forward_backbone(self, x6: torch.Tensor) -> torch.Tensor:
        """
        x6: (B, 6, 6, 7) -> (B, 64)
        """
        h = self.backbone(x6)            # (B,64,2,3)
        h = self.gap(h).flatten(1)       # (B,64)
        return h

    def forward(self, x: torch.Tensor):
        """
        Returns:
          logits: [B, action_dim]
          value : [B]
        """
        x6 = self._to_six_channel(x)     # (B,6,6,7)
        h  = self._forward_backbone(x6)  # (B,64)
        h  = self.fc(h)                  # (B,128)
        logits = self.policy_head(h)     # (B, A)
        value  = self.value_head(h).squeeze(-1)  # (B)
        return logits, value

    # ---------- action helpers ----------

    @staticmethod
    def make_legal_action_mask(batch_legal_actions, action_dim: int = 7) -> torch.Tensor:
        """
        batch_legal_actions: iterable of iterables/sets of ints
        returns: [B, A] bool mask (on DEVICE)
        """
        B = len(batch_legal_actions)
        mask = torch.zeros(B, action_dim, dtype=torch.bool, device=DEVICE)
        for i, acts in enumerate(batch_legal_actions):
            if acts:  # guard against empty list
                idx = torch.as_tensor(list(acts), dtype=torch.long, device=DEVICE)
                mask[i, idx] = True
        return mask

    def _apply_center_mix_single(
        self,
        probs: torch.Tensor,          # [1, A]
        mask: torch.Tensor,           # [1, A] bool, legal mask
        legal_actions: list[int],
        ply_idx: int | None,
    ) -> tuple[torch.Tensor, bool]:
        """
        Option B: on-policy-ish opening mix.
        Blend network policy with a one-hot 'center' book for early plies.

        Returns:
            mixed_probs [1,A], center_mixed_flag (whether we applied the mix).
        """
        if (
            self.center_start <= 0.0        # lambda = 0 -> no mix
            or ply_idx is None
            or ply_idx > self.center_ply_max
            or CENTER_COL not in legal_actions
            or self.center_forced_used      # you already used center once this episode
        ):
            return probs, False

        lam = float(self.center_start)      # λ in [0,1]
        if lam <= 0.0: return probs, False

        # opening-book distribution: all mass on center (within legal set)
        book = torch.zeros_like(probs)
        book[0, CENTER_COL] = 1.0

        mixed = (1.0 - lam) * probs + lam * book

        # ensure we only keep probability on legal actions
        mixed = mixed * mask.float()
        mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        return mixed, True


    @torch.inference_mode()
    def act(
        self,
        state_np: np.ndarray,
        legal_actions,
        temperature: float = 1.0,
        ply_idx: int | None = None,
    ) -> tuple[int, torch.Tensor, torch.Tensor, dict]:
    
        legal_actions = [int(a) for a in legal_actions]
        assert legal_actions, "act called with empty legal_actions"
    
        s_np = np.asarray(state_np, dtype=np.float32)
        if s_np.ndim != 3 or s_np.shape[1:] != (6, 7) or s_np.shape[0] < 2:
            raise ValueError(f"Expected state (C,6,7) with C>=2, got {s_np.shape}")
    
        # board for heuristics: use first two planes (me/opp)
        board = (s_np[0] - s_np[1]).astype(np.int8)
    
        dev = next(self.parameters()).device
        x = torch.from_numpy(s_np).unsqueeze(0).to(dev)
    
        logits, value = self.forward(x)  # logits [1,A], value [1]
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[0, legal_actions] = True
    
        T = max(float(temperature), 1e-6)
        masked = logits.masked_fill(~mask, NEG_INF) / T
        probs = torch.softmax(masked, dim=-1)  # [1,A]
    
        # Track sets for mode assignment after sampling
        center_applied = False
        win_moves: list[int] = []
        must_play: int | None = None
        safe_actions: list[int] = []
    
        # ---------- 1) Center mix (deterministic) ----------
        probs, center_applied = self._apply_center_mix_single(probs, mask, legal_actions, ply_idx)
    
        # ---------- 2) Win mix (deterministic) ----------
        win_moves = _find_immediate_wins(board, +1, legal_actions)
        if win_moves and self.win_now_prob > 0.0:
            win_mask = torch.zeros_like(probs, dtype=torch.bool)
            win_mask[0, win_moves] = True
    
            win_probs = probs * win_mask.float()
            if win_probs.sum() <= 1e-8:
                win_probs = torch.zeros_like(probs)
                win_probs[0, win_moves] = 1.0 / max(1, len(win_moves))
            else:
                win_probs = self._renorm(win_probs)
    
            probs = self._mix(probs, win_probs, self.win_now_prob)
    
        # ---------- 3) Guard mix (deterministic, only in ply window) ----------
        if (
            self.guard_prob > 0.0
            and ply_idx is not None
            and self.guard_ply_min <= ply_idx <= self.guard_ply_max
        ):
            must_play, safe_actions = _tactical_guard(board, legal_actions, me=+1, opp=-1)
    
            if must_play is not None:
                block = torch.zeros_like(probs)
                block[0, int(must_play)] = 1.0
                probs = self._mix(probs, block, self.guard_prob)
    
            elif safe_actions and len(safe_actions) < len(legal_actions):
                safe_mask = torch.zeros_like(probs, dtype=torch.bool)
                safe_mask[0, [int(a) for a in safe_actions]] = True
    
                safe_probs = probs * safe_mask.float()
                if safe_probs.sum() <= 1e-8:
                    safe_probs = torch.zeros_like(probs)
                    safe_probs[0, [int(a) for a in safe_actions]] = 1.0 / max(1, len(safe_actions))
                else:
                    safe_probs = self._renorm(safe_probs)
    
                probs = self._mix(probs, safe_probs, self.guard_prob)
    
        # ---------- 4) Sample ----------
        dist = Categorical(probs=probs[0])
        a_t = dist.sample()
        lp = dist.log_prob(a_t).detach()
        a = int(a_t.item())
    
        # ---------- Mode + stats ----------
        # Priority reporting by membership (not by whether mix was applied).
        if win_moves and (a in win_moves):
            mode = "win_now"
            self._bump_stat("win_now")
        elif must_play is not None and a == int(must_play):
            mode = "guard_block"
            self._bump_stat("guard")
        elif safe_actions and (a in safe_actions) and (len(safe_actions) < len(legal_actions)):
            mode = "guard_safe"
            self._bump_stat("guard")
        elif center_applied and a == CENTER_COL:
            mode = "center"
            self._bump_stat("center")
            self.center_forced_used = True
        else:
            mode = "policy"
            self._bump_stat("policy")
    
        return a, lp, value.squeeze(0).detach(), {"mode": mode}



    def evaluate_actions(
    self,
    states: torch.Tensor,
    actions: torch.Tensor,
    legal_action_masks: torch.Tensor | None = None,
    ply_indices: torch.Tensor | None = None,
    temperature: float | torch.Tensor = 1.0,
):
        dev = self.coord_row.device
    
        # --- encode once (avoid double work) ---
        x6 = self._to_six_channel(states)  # now uses model device
        h = self._forward_backbone(x6)
        h = self.fc(h)
        logits = self.policy_head(h)
        values = self.value_head(h).squeeze(-1)
    
        # --- temperature (match act) ---
        if isinstance(temperature, torch.Tensor):
            T = temperature.to(device=dev, dtype=logits.dtype).clamp_min(1e-6)
            logits = logits / (T.view(-1, 1) if T.dim() > 0 else T)
        else:
            logits = logits / max(float(temperature), 1e-6)
    
        # --- legal mask ---
        mask = None
        if legal_action_masks is not None:
            mask = legal_action_masks.to(device=dev, dtype=torch.bool)
            no_legal = (mask.sum(dim=-1) == 0)
            if no_legal.any():
                mask = mask.clone()
                mask[no_legal] = True
            logits = logits.masked_fill(~mask, NEG_INF)
        else:
            mask = torch.ones_like(logits, dtype=torch.bool, device=dev)
    
        probs = torch.softmax(logits, dim=-1)
    
        # --- center mix (vectorized) ---
        if ply_indices is not None and self.center_start > 0.0 and self.center_ply_max >= 0:
            lam = float(self.center_start)
            ply = ply_indices.to(device=dev).long()
            apply_row = (ply <= int(self.center_ply_max)) & mask[:, CENTER_COL]
            if apply_row.any():
                book = torch.zeros_like(probs)
                book[:, CENTER_COL] = 1.0
                mixed = (1.0 - lam) * probs + lam * book
                mixed = mixed * mask.float()
                mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                probs = torch.where(apply_row.unsqueeze(-1), mixed, probs)
    
        # --- win_now / guard mix (loop, uses numpy board checks) ---
        # board from x6 first two channels: me/opp planes -> {-1,0,1}
        board = (x6[:, 0] - x6[:, 1]).detach().cpu().numpy().astype(np.int8)
        mask_cpu = mask.detach().cpu().numpy()
    
        B, A = probs.shape
        probs_out = probs.clone()
    
        for i in range(B):
            legal = [int(a) for a in np.where(mask_cpu[i])[0].tolist()]
            if not legal:
                continue
    
            pi = probs_out[i:i+1, :]
    
            # win mix
            if self.win_now_prob > 0.0:
                wins = _find_immediate_wins(board[i], +1, legal)
                if wins:
                    win_mask = torch.zeros_like(pi, dtype=torch.bool)
                    win_mask[0, wins] = True
                    win_pi = pi * win_mask.float()
                    if (win_pi.sum() <= 1e-8):
                        win_pi = torch.zeros_like(pi)
                        win_pi[0, wins] = 1.0 / max(1, len(wins))
                    else:
                        win_pi = self._renorm(win_pi)
                    pi = self._mix(pi, win_pi, self.win_now_prob)
    
            # guard mix (ply window)
            if self.guard_prob > 0.0 and ply_indices is not None:
                ply_i = int(ply_indices[i].item())
                if self.guard_ply_min <= ply_i <= self.guard_ply_max:
                    must_play, safe_actions = _tactical_guard(board[i], legal, me=+1, opp=-1)
                    if must_play is not None:
                        block = torch.zeros_like(pi)
                        block[0, int(must_play)] = 1.0
                        pi = self._mix(pi, block, self.guard_prob)
                    elif safe_actions and len(safe_actions) < len(legal):
                        safe_mask = torch.zeros_like(pi, dtype=torch.bool)
                        safe_mask[0, [int(a) for a in safe_actions]] = True
                        safe_pi = pi * safe_mask.float()
                        if (safe_pi.sum() <= 1e-8):
                            safe_pi = torch.zeros_like(pi)
                            safe_pi[0, [int(a) for a in safe_actions]] = 1.0 / max(1, len(safe_actions))
                        else:
                            safe_pi = self._renorm(safe_pi)
                        pi = self._mix(pi, safe_pi, self.guard_prob)
    
            probs_out[i:i+1, :] = pi
    
        dist = Categorical(probs=probs_out)
        actions = actions.long()
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprobs, entropy, values

    # ---------- init ----------

    @staticmethod
    def _enforce_lr_symmetry_(module: nn.Module):
        """
        Make every Conv2d LR-equivariant by averaging kernels with their left-right flips.
        """
        with torch.no_grad():
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.copy_(0.5 * (m.weight + torch.flip(m.weight, dims=[-1])))
                    # bias untouched

    def _init_weights(self):
        # Kaiming for convs/linears
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Heads: orthogonal helps PPO stability
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

        # Center-column action prior on policy bias (gentle, symmetric)
        with torch.no_grad():
            a = torch.arange(self.action_dim, dtype=torch.float32, device=self.policy_head.bias.device)
            mu = 3.0  # center column
            sigma = max(1e-6, self._init_center_sigma)
            gauss = torch.exp(-0.5 * ((a - mu) / sigma) ** 2)
            gauss = gauss / (gauss.max() + 1e-12)
            bias_per_action = self._init_center_bias_scale * gauss
            bias_per_action = bias_per_action - bias_per_action.mean()
            self.policy_head.bias.add_(bias_per_action)

        # Optionally enforce LR symmetry on all convs so mirror holds at init
        if self._enforce_lr_symmetry_flag:  self._enforce_lr_symmetry_(self)
            
    # ----- phase hooks -----
    def set_phase_heuristics(
        self,
        *,
        center_start: float | None = None,
        guard_prob: float | None = None,
        win_now_prob: float | None = None,
        guard_ply_min: int | None = None,
        guard_ply_max: int | None = None,
    ) -> None:
        """Set heuristic knobs from phase config."""
        if center_start is not None:            self.center_start = float(center_start)
        if guard_prob is not None:              self.guard_prob = float(guard_prob)
        if win_now_prob is not None:            self.win_now_prob = float(win_now_prob)
        if guard_ply_min is not None:           self.guard_ply_min = int(guard_ply_min)
        if guard_ply_max is not None:           self.guard_ply_max = int(guard_ply_max)

    # ----- stats helpers -----
    def _bump_stat(self, key: str) -> None:
        if key not in self._act_stats:
            self._act_stats[key] = 0
        self._act_stats[key] += 1
        self._act_stats_total += 1

    def act_stats_summary(self, normalize: bool = True) -> dict[str, float]:
        """
        Return action-mix stats, e.g. fractions of:
          • win_now
          • center
          • guard
          • policy   (pure policy sample)
        """
        if not self._act_stats_total:
            if normalize:
                return {k: 0.0 for k in self._act_stats}
            return dict(self._act_stats)

        if normalize:
            return {k: v / self._act_stats_total for k, v in self._act_stats.items()}
        return dict(self._act_stats)

    def _renorm(self, p: torch.Tensor) -> torch.Tensor:
        return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    
    def _mix(self, p: torch.Tensor, q: torch.Tensor, w: float) -> torch.Tensor:
        w = float(w)
        if w <= 0.0: return p
        if w >= 1.0: return q
        return self._renorm((1.0 - w) * p + w * q)
