# PPO/actor_critic.py
# CNet192-based Actor-Critic for PPO, using 1-channel POV boards.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from C4.CNet192 import CNet192, load_cnet192

NEG_INF = -1e9
CENTER_COL = 3


# -------------------------
# Tiny board tactics (optional)
# -------------------------

def _lowest_empty_row(board: np.ndarray, col: int) -> Optional[int]:
    """Bottom-most empty row for a column on a 6x7 board. None if full."""
    for r in range(5, -1, -1):
        if board[r, col] == 0:
            return r
    return None


def _has_four_from(board: np.ndarray, row: int, col: int, player: int) -> bool:
    """Check if player has 4-in-a-row passing through (row, col)."""
    rows, cols = board.shape
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
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
    """If player drops into col, does it immediately win?"""
    r = _lowest_empty_row(board, col)
    if r is None:
        return False

    old = board[r, col]
    board[r, col] = player
    win = _has_four_from(board, r, col, player)
    board[r, col] = old
    return win


def _find_immediate_wins(board: np.ndarray, player: int, legal_actions: List[int]) -> List[int]:
    wins: List[int] = []
    for c in legal_actions:
        if _is_winning_drop(board, c, player):
            wins.append(int(c))
    return wins


def _tactical_guard(
    board: np.ndarray,
    legal_actions: List[int],
    me: int = +1,
    opp: int = -1,
) -> Tuple[Optional[int], List[int]]:
    """
    Basic tactical guard:
      - If opponent has exactly one immediate win now: must block that column.
      - Otherwise return safe_actions where opponent has no immediate 1-ply win after our move.
    """
    opp_wins_now = _find_immediate_wins(board, opp, legal_actions)
    if len(opp_wins_now) == 1 and opp_wins_now[0] in legal_actions:
        return opp_wins_now[0], legal_actions

    safe_actions: List[int] = []
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


# -------------------------
# Actor-Critic wrapper
# -------------------------

@dataclass
class TransferCfg:
    """Optional transfer-learning knobs."""
    freeze_conv: bool = False          # Marco-style: freeze pre-trained conv block
    strict_load: bool = True           # strict checkpoint load


class ActorCritic(nn.Module):
    """
    PPO Actor-Critic using your CNet192 architecture.

    Input convention (IMPORTANT):
      - 1-channel POV board, shape (B,1,6,7) or (1,6,7) or (6,7)
      - values in {-1, 0, +1}
      - POV means: current-to-play pieces are +1, opponent pieces are -1

    Compatibility:
      - Also accepts 2-channel (me, opp) as (2,6,7) or (B,2,6,7) and converts to POV scalar.
    """

    def __init__(self, use_mid_3x3: bool = True):
        super().__init__()
        self.net = CNet192(in_channels=1, use_mid_3x3=use_mid_3x3)

        # --- Heuristic knobs (phase-dependent, default OFF) ---
        self.center_start: float = 0.0
        self.center_ply_max: int = 1
        self.guard_prob: float = 0.0
        self.guard_ply_min: int = 2
        self.guard_ply_max: int = 10
        self.win_now_prob: float = 0.0

        # Per-episode flag (center-book should be a gentle nudge, not a religion)
        self.center_forced_used: bool = False

        # Action-mix stats (handy for your live plots)
        self._act_stats_total: int = 0
        self._act_stats: Dict[str, int] = {
            "win_now": 0,
            "center": 0,
            "guard": 0,
            "policy": 0,
        }

    # -------------------------
    # Transfer helpers
    # -------------------------

    @classmethod
    def from_cnet192_checkpoint(
        cls,
        path: str,
        device: torch.device,
        *,
        override_cfg: Optional[Dict[str, Any]] = None,
        transfer: Optional[TransferCfg] = None,
    ) -> "ActorCritic":
        """
        Build ActorCritic and load weights from a CNet192 checkpoint created by save_cnet192().
        """
        transfer = transfer or TransferCfg()
        model_cnet, ckpt = load_cnet192(
            path=path,
            device=device,
            strict=transfer.strict_load,
            override_cfg=override_cfg,
        )
        use_mid = bool(ckpt.get("cfg", {}).get("use_mid_3x3", True)) if override_cfg is None else bool(override_cfg.get("use_mid_3x3", True))
        ac = cls(use_mid_3x3=use_mid).to(device)
        ac.net.load_state_dict(model_cnet.state_dict(), strict=True)

        if transfer.freeze_conv:
            ac.freeze_conv_block(True)

        return ac

    def freeze_conv_block(self, freeze: bool = True) -> None:
        """Freeze/unfreeze the convolutional feature extractor (Marco-style transfer)."""
        for p in self.net.conv1.parameters():
            p.requires_grad = not freeze
        if self.net.conv_mid is not None:
            for p in self.net.conv_mid.parameters():
                p.requires_grad = not freeze
        for p in self.net.conv2.parameters():
            p.requires_grad = not freeze

    # -------------------------
    # Episode / stats helpers
    # -------------------------

    def begin_episode(self) -> None:
        """Call at the start of each game/episode."""
        self.center_forced_used = False

    def _bump_stat(self, key: str) -> None:
        if key not in self._act_stats:
            self._act_stats[key] = 0
        self._act_stats[key] += 1
        self._act_stats_total += 1

    def act_stats_summary(self, normalize: bool = True) -> Dict[str, float]:
        if self._act_stats_total == 0:
            return {k: 0.0 for k in self._act_stats} if normalize else dict(self._act_stats)
        if normalize:
            return {k: v / self._act_stats_total for k, v in self._act_stats.items()}
        return dict(self._act_stats)

    # -------------------------
    # Phase hooks (same names you already use)
    # -------------------------

    def set_phase_heuristics(
        self,
        *,
        center_start: Optional[float] = None,
        guard_prob: Optional[float] = None,
        win_now_prob: Optional[float] = None,
        guard_ply_min: Optional[int] = None,
        guard_ply_max: Optional[int] = None,
        center_ply_max: Optional[int] = None,
    ) -> None:
        if center_start is not None:
            self.center_start = float(center_start)
        if guard_prob is not None:
            self.guard_prob = float(guard_prob)
        if win_now_prob is not None:
            self.win_now_prob = float(win_now_prob)
        if guard_ply_min is not None:
            self.guard_ply_min = int(guard_ply_min)
        if guard_ply_max is not None:
            self.guard_ply_max = int(guard_ply_max)
        if center_ply_max is not None:
            self.center_ply_max = int(center_ply_max)

    # -------------------------
    # Input normalization (1-channel POV)
    # -------------------------

    def _dev(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def pov_from_env_board(board: np.ndarray, player_to_move: int) -> np.ndarray:
        """
        Convert env-centric scalar board to POV scalar board:
          POV = board * player_to_move
        So the current player always becomes +1.
        """
        b = np.asarray(board, dtype=np.int8)
        p = 1 if int(player_to_move) == 1 else -1
        return (b * p).astype(np.int8)

    def _to_1ch_pov(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Normalize to (B,1,6,7) float32 on model device.

        Accepts:
          - (6,7)
          - (1,6,7)
          - (B,6,7)
          - (B,1,6,7)
          - (2,6,7) or (B,2,6,7) as (me,opp) planes -> POV scalar = me - opp
        """
        dev = self._dev()

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        x = x.to(device=dev, dtype=torch.float32)

        if x.dim() == 2:  # (6,7)
            return x.unsqueeze(0).unsqueeze(0)

        if x.dim() == 3:
            # (1,6,7) or (2,6,7) or (B,6,7)
            if x.shape == (1, 6, 7):
                return x.unsqueeze(0)
            if x.shape == (2, 6, 7):
                s = x[0] - x[1]
                return s.unsqueeze(0).unsqueeze(0)
            # assume (B,6,7)
            if x.shape[1:] == (6, 7):
                return x.unsqueeze(1)
            raise ValueError(f"Unexpected 3D input shape: {tuple(x.shape)}")

        if x.dim() == 4:
            # (B,1,6,7) or (B,2,6,7)
            if x.shape[1:] == (1, 6, 7):
                return x
            if x.shape[1:] == (2, 6, 7):
                s = x[:, 0] - x[:, 1]
                return s.unsqueeze(1)
            raise ValueError(f"Unexpected 4D input shape: {tuple(x.shape)}")

        raise ValueError(f"Unexpected input rank: {x.dim()}")

    # -------------------------
    # Core forward
    # -------------------------

    def forward(self, x: torch.Tensor | np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits: (B,7)
          value : (B,)
        """
        x1 = self._to_1ch_pov(x)
        logits, value = self.net(x1)
        return logits, value

    # -------------------------
    # Legal masking utils
    # -------------------------

    @staticmethod
    def make_legal_action_mask(
        batch_legal_actions: Iterable[Iterable[int]],
        action_dim: int = 7,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        acts_list = list(batch_legal_actions)
        B = len(acts_list)
        dev = device if device is not None else torch.device("cpu")
        mask = torch.zeros(B, action_dim, dtype=torch.bool, device=dev)
        for i, acts in enumerate(acts_list):
            a = list(acts)
            if not a:
                continue
            idx = torch.as_tensor(a, dtype=torch.long, device=dev)
            mask[i, idx] = True
        return mask

    @staticmethod
    def _renorm(p: torch.Tensor) -> torch.Tensor:
        return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _mix(p: torch.Tensor, q: torch.Tensor, w: float) -> torch.Tensor:
        w = float(w)
        if w <= 0.0:
            return p
        if w >= 1.0:
            return q
        return ActorCritic._renorm((1.0 - w) * p + w * q)

    def _apply_center_mix_single(
        self,
        probs: torch.Tensor,      # [1,7]
        mask: torch.Tensor,       # [1,7] bool
        legal_actions: List[int],
        ply_idx: Optional[int],
    ) -> Tuple[torch.Tensor, bool]:
        if (
            self.center_start <= 0.0
            or ply_idx is None
            or ply_idx > self.center_ply_max
            or CENTER_COL not in legal_actions
            or self.center_forced_used
        ):
            return probs, False

        lam = float(self.center_start)
        book = torch.zeros_like(probs)
        book[0, CENTER_COL] = 1.0

        mixed = (1.0 - lam) * probs + lam * book
        mixed = mixed * mask.float()
        mixed = self._renorm(mixed)
        return mixed, True

    # -------------------------
    # Action selection
    # -------------------------

    @torch.inference_mode()
    def act(
        self,
        state_np: np.ndarray,
        legal_actions: Iterable[int],
        *,
        temperature: float = 1.0,
        ply_idx: Optional[int] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Sample an action and return:
          (action:int, logprob:Tensor[], value:Tensor[], info:dict)

        state_np can be:
          - (6,7) POV scalar
          - (1,6,7) POV 1ch
          - (2,6,7) (me,opp) planes
        """
        legal = [int(a) for a in legal_actions]
        if not legal:
            raise ValueError("act() called with empty legal_actions")

        dev = self._dev()

        x = torch.from_numpy(np.asarray(state_np, dtype=np.float32)).to(dev)
        logits, value = self.forward(x)  # logits [1,7] if single

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[0, legal] = True

        T = max(float(temperature), 1e-6)
        masked_logits = logits.masked_fill(~mask, NEG_INF) / T
        probs = torch.softmax(masked_logits, dim=-1)

        # board for tactic checks (POV scalar)
        x1 = self._to_1ch_pov(x)
        board = x1[0, 0].detach().cpu().numpy().astype(np.int8)

        center_applied = False
        win_moves: List[int] = []
        must_play: Optional[int] = None
        safe_actions: List[int] = []

        # 1) Center mix
        probs, center_applied = self._apply_center_mix_single(probs, mask, legal, ply_idx)

        # 2) Win-now mix
        win_moves = _find_immediate_wins(board, +1, legal)
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

        # 3) Guard mix (ply window)
        if (
            self.guard_prob > 0.0
            and ply_idx is not None
            and self.guard_ply_min <= ply_idx <= self.guard_ply_max
        ):
            must_play, safe_actions = _tactical_guard(board, legal, me=+1, opp=-1)

            if must_play is not None:
                block = torch.zeros_like(probs)
                block[0, must_play] = 1.0
                probs = self._mix(probs, block, self.guard_prob)

            elif safe_actions and len(safe_actions) < len(legal):
                safe_mask = torch.zeros_like(probs, dtype=torch.bool)
                safe_mask[0, safe_actions] = True
                safe_probs = probs * safe_mask.float()
                if safe_probs.sum() <= 1e-8:
                    safe_probs = torch.zeros_like(probs)
                    safe_probs[0, safe_actions] = 1.0 / max(1, len(safe_actions))
                else:
                    safe_probs = self._renorm(safe_probs)
                probs = self._mix(probs, safe_probs, self.guard_prob)

        # 4) Sample
        dist = Categorical(probs=probs[0])
        a_t = dist.sample()
        lp = dist.log_prob(a_t).detach()
        a = int(a_t.item())

        # Mode + stats
        if win_moves and (a in win_moves):
            mode = "win_now"
            self._bump_stat("win_now")
        elif must_play is not None and a == int(must_play):
            mode = "guard_block"
            self._bump_stat("guard")
        elif safe_actions and (a in safe_actions) and (len(safe_actions) < len(legal)):
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

    # -------------------------
    # PPO evaluation helper
    # -------------------------

    def evaluate_actions(
        self,
        states: torch.Tensor | np.ndarray,
        actions: torch.Tensor,
        *,
        legal_action_masks: Optional[torch.Tensor] = None,
        ply_indices: Optional[torch.Tensor] = None,
        temperature: float | torch.Tensor = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch evaluation for PPO:
          returns (logprobs, entropy, values)

        - states: (B,1,6,7) or compatible shapes (see _to_1ch_pov)
        - actions: (B,)
        - legal_action_masks: (B,7) bool, optional
        - ply_indices: (B,), optional (for center/guard window behavior)
        """
        dev = self._dev()

        logits, values = self.forward(states)  # logits (B,7), values (B,)

        # temperature
        if isinstance(temperature, torch.Tensor):
            T = temperature.to(device=dev, dtype=logits.dtype).clamp_min(1e-6)
            logits = logits / (T.view(-1, 1) if T.dim() > 0 else T)
        else:
            logits = logits / max(float(temperature), 1e-6)

        # legal mask
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

        # center mix (vectorized)
        if ply_indices is not None and self.center_start > 0.0 and self.center_ply_max >= 0:
            lam = float(self.center_start)
            ply = ply_indices.to(device=dev).long()
            apply_row = (ply <= int(self.center_ply_max)) & mask[:, CENTER_COL]
            if apply_row.any():
                book = torch.zeros_like(probs)
                book[:, CENTER_COL] = 1.0
                mixed = (1.0 - lam) * probs + lam * book
                mixed = mixed * mask.float()
                mixed = self._renorm(mixed)
                probs = torch.where(apply_row.unsqueeze(-1), mixed, probs)

        # win_now / guard mix (loop, matches act())
        # NOTE: If you keep these probs > 0 during training, PPO is "learning a policy + a rulebook".
        # That can be fine for curriculum bootstrapping, but you'll likely want them -> 0 later.
        if (self.win_now_prob > 0.0) or (self.guard_prob > 0.0):
            x1 = self._to_1ch_pov(states)
            boards = x1[:, 0].detach().cpu().numpy().astype(np.int8)
            mask_cpu = mask.detach().cpu().numpy()

            probs_out = probs.clone()
            B = probs_out.shape[0]

            for i in range(B):
                legal = [int(a) for a in np.where(mask_cpu[i])[0].tolist()]
                if not legal:
                    continue

                pi = probs_out[i:i+1, :]

                # win mix
                if self.win_now_prob > 0.0:
                    wins = _find_immediate_wins(boards[i], +1, legal)
                    if wins:
                        win_mask = torch.zeros_like(pi, dtype=torch.bool)
                        win_mask[0, wins] = True
                        win_pi = pi * win_mask.float()
                        if win_pi.sum() <= 1e-8:
                            win_pi = torch.zeros_like(pi)
                            win_pi[0, wins] = 1.0 / max(1, len(wins))
                        else:
                            win_pi = self._renorm(win_pi)
                        pi = self._mix(pi, win_pi, self.win_now_prob)

                # guard mix
                if self.guard_prob > 0.0 and ply_indices is not None:
                    ply_i = int(ply_indices[i].item())
                    if self.guard_ply_min <= ply_i <= self.guard_ply_max:
                        must_play, safe_actions = _tactical_guard(boards[i], legal, me=+1, opp=-1)
                        if must_play is not None:
                            block = torch.zeros_like(pi)
                            block[0, int(must_play)] = 1.0
                            pi = self._mix(pi, block, self.guard_prob)
                        elif safe_actions and len(safe_actions) < len(legal):
                            safe_mask = torch.zeros_like(pi, dtype=torch.bool)
                            safe_mask[0, safe_actions] = True
                            safe_pi = pi * safe_mask.float()
                            if safe_pi.sum() <= 1e-8:
                                safe_pi = torch.zeros_like(pi)
                                safe_pi[0, safe_actions] = 1.0 / max(1, len(safe_actions))
                            else:
                                safe_pi = self._renorm(safe_pi)
                            pi = self._mix(pi, safe_pi, self.guard_prob)

                probs_out[i:i+1, :] = pi

            probs = probs_out

        dist = Categorical(probs=probs)
        actions = actions.long()
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()

        return logprobs, entropy, values
