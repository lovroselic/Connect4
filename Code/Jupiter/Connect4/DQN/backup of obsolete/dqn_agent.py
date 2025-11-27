# dqn_agent.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from collections import defaultdict
from collections import Counter

from DQN.DQN_replay_memory_per import PrioritizedReplayMemory
from DQN.dqn_model import DQN
from C4.fast_connect4_lookahead import Connect4Lookahead


class DQNAgent:
    def __init__(
        self,
        lr=2e-4, 
        gamma=0.98,                     #0.98
        epsilon=0.08,
        epsilon_min=0.02,
        epsilon_decay=0.995,
        device=None,
        memory_capacity=500_000,
        
        #per
        per_alpha=0.40,                 #0.60
        per_eps=1e-2,                   #1e-2 
        per_beta_start=0.45,            #0.45
        per_beta_end=0.90,              #0.98
        per_beta_steps=250_000,         #200_000
        per_mix_1step=0.50,             #0.70
        
        #defaults, overwritten by phase change
        target_update_interval=500,     #very obsolete
        target_update_mode="soft", 
        tau=0.005,
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.update_target_model()
        self.memory = PrioritizedReplayMemory(capacity=memory_capacity, alpha=per_alpha, eps=per_eps)

        # PER β schedule
        self.per_beta = float(per_beta_start)
        self.per_beta_end = float(per_beta_end)
        steps = max(1, int(per_beta_steps))
        self.per_beta_step = (self.per_beta_end - self.per_beta) / steps
        self.per_mix_1step = float(per_mix_1step)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss(beta=0.5, reduction="none")
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lookahead = Connect4Lookahead()
        self.reward_scale = 0.01
        self.guard_prob = 0.0                       # controlled by phase
        self.guard_boost_start = 5
        self.guard_boost_end   = 17
        self.center_start = 0.3                     # controlled by phase

        self.td_hist = deque(maxlen=50_000)
        self.loss_hist = deque(maxlen=10_000)
        self.per_w_hist = deque(maxlen=50_000)
        self.sample_mix_hist = deque(maxlen=20_000)
        
        self.target_update_interval = int(target_update_interval)
        self.target_update_mode = str(target_update_mode)   # "hard" or "soft"
        self.tau = float(tau)
        self._last_target_update_step = 0
        
        self.q_abs_hist      = deque(maxlen=100_000)
        self.q_max_hist      = deque(maxlen=100_000)
        self.target_abs_hist = deque(maxlen=100_000)
        
        self.center_forced_used = False
        
        # --- policy symmetry + tie-breaking knobs ---
        self.use_symmetry_policy = True             # avg original + horizontally flipped Qs
        self.tie_tol = 1e-9                         # moves within tol of the max are considered ties
        self.prefer_center = True                   # bias ties toward the center column
        self.rng = np.random.default_rng(0)
        
        self.max_seed_frac = 0.90                   # phase ccontrolled
        self.min_seed_frac = 0.10                   # phase ccontrolled
        
        self.target_lag_hist = deque(maxlen=50_000)  # ||θ-θ̄||/||θ||
        self.lag_log_interval = 250                  # log every N env steps (tune)
        self._last_lag_log_step = 0                  # last env step we logged at
        self.target_lag_hist.append(0.0)
        
        self.act_stats = defaultdict(int)   # counts
        self._act_stats_total = 0 
        self.opp_act_stats = defaultdict(int)
        self._opp_act_total = 0
        
    def _bump_stat(self, key: str):
        self.act_stats[key] += 1
        
    def act_stats_summary(self, normalize=True):
        total = max(1, self._act_stats_total)
        if not normalize: return dict(self.act_stats)
        return {k: v / total for k, v in self.act_stats.items()}
    
    def _bump_opp(self, key: str):
        self.opp_act_stats[key] += 1
 
    def opp_act_stats_summary(self, normalize: bool = True):
        total = max(1, self._opp_act_total)
        if not normalize:
            return dict(self.opp_act_stats)
        return {k: v / total for k, v in self.opp_act_stats.items()}


    # ------------------------- helper encodings -------------------------

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # ------------------------- tactical guard -------------------------

    @staticmethod
    def _drop_piece(board: np.ndarray, col: int, player: int):
        if board[0, col] != 0: return None, None
        
        b = board.copy()
        for r in range(b.shape[0] - 1, -1, -1):
            if b[r, col] == 0:
                b[r, col] = player
                return b, r
        return None, None

    @staticmethod
    def _is_win(board: np.ndarray) -> int:
        R, C = board.shape
        for r in range(R):
            for c in range(C - 3):
                s = np.sum(board[r, c:c + 4])
                if abs(s) == 4:
                    return int(np.sign(s))
        for r in range(R - 3):
            for c in range(C):
                s = np.sum(board[r:r + 4, c])
                if abs(s) == 4:
                    return int(np.sign(s))
        for r in range(3, R):
            for c in range(C - 3):
                s = board[r, c] + board[r - 1, c + 1] + board[r - 2, c + 2] + board[r - 3, c + 3]
                if abs(s) == 4:
                    return int(np.sign(s))
        for r in range(R - 3):
            for c in range(C - 3):
                s = board[r, c] + board[r + 1, c + 1] + board[r + 2, c + 2] + board[r + 3, c + 3]
                if abs(s) == 4:
                    return int(np.sign(s))
        return 0

    def _tactical_guard(self, oriented_board: np.ndarray, valid_actions):
     """
         Board is already oriented so that agent == +1, opponent == -1.
         Returns: (must_play: int|None, safe_actions: list[int])
     """
     must_play = None
     safe_actions = []
 
     for a in valid_actions:
         b1, _ = self._drop_piece(oriented_board, a, +1)   # us = +1
         if b1 is None:
             continue
 
         # Immediate win for us?
         if self.lookahead.check_win(b1, +1):
             must_play = a
             safe_actions.append(a)
             continue
 
         # Would opponent have an immediate reply win?
         opp_threat = False
         for oa in valid_actions:
             b2, _ = self._drop_piece(b1, oa, -1)          # opp = -1
             if b2 is None: continue
             if self.lookahead.check_win(b2, -1):
                 opp_threat = True
                 break
 
         if not opp_threat: safe_actions.append(a)
 
     return must_play, safe_actions

    
    @staticmethod
    def _center_col_is_empty(state) -> bool:
        """
        state: (4, 6, 7) [agent, opp, row, col]
        Return True iff column 3 has no stones from either player.
        """
        #assert state.ndim == 3 and state.shape[0] == 4, f"Expected (4,6,7), got {state.shape}"
        col_occ = state[0, :, 3] + state[1, :, 3]
        return (col_occ == 0).all()

    def board_to_state(self, board: np.ndarray, player: int) -> np.ndarray:
        agent_plane = (board == player).astype(np.float32)
        opp_plane = (board == -player).astype(np.float32)
        row_plane = np.tile(np.linspace(-1, 1, board.shape[0])[:, None], (1, board.shape[1]))
        col_plane = np.tile(np.linspace(-1, 1, board.shape[1])[None, :], (board.shape[0], 1))
        return np.stack([agent_plane, opp_plane, row_plane, col_plane])

    def act(self, state: np.ndarray, valid_actions, depth: int, strategy_weights=None, ply = None):
        self._act_stats_total += 1
        assert len(valid_actions) > 0, f"ACT called with empty valid actions {valid_actions}"
        board = state[0] - state[1]                                     # mover (+1) minus opp (–1)
                            
        # === immediate win  ===
        win_now, win_a = self.had_one_ply_win(board, valid_actions)
        if win_now:
            self._bump_stat("win_now")
            return int(win_a)
        
        # === CENTER START ===
        if not self.center_forced_used and ply is not None and ply <= 1:
            center_start_on = (random.random() < self.center_start)
            if center_start_on and 3 in valid_actions: 
                if self._center_col_is_empty(state):
                    self.center_forced_used = True
                    self._bump_stat("center")
                    return 3 # start bottom center
                
        # === TACTICAL GUARD WINDOW ===
        guard_on = False
        if ply is not None and (self.guard_boost_start <= ply <= self.guard_boost_end):
           guard_on = (random.random() < float(self.guard_prob))
        
        safe_actions = None
        if guard_on:
            orig_n = len(valid_actions)
            must_play, safe_actions = self._tactical_guard(board, valid_actions)
            if must_play is not None:
                self._bump_stat("guard")
                return must_play
            if safe_actions: 
                if len(safe_actions) < orig_n:
                    self._bump_stat("guard")
                valid_actions = safe_actions
    
        # === ε-greedy Exploration ===
        if random.random() < self.epsilon:
            self._bump_stat("epsilon")
            if strategy_weights is None: strategy_weights = [1, 0, 0, 0, 0, 0, 0, 0]
            assert len(strategy_weights) == 8, f"Expected 8 strategy weights, got {len(strategy_weights)}: {strategy_weights}"
            choice = random.choices(range(8), weights=strategy_weights)[0]
    
            if choice == 0:
                action = random.choice(valid_actions)
                self._bump_stat("rand")
                return action
            else:
                a = self.lookahead.n_step_lookahead(board, player=+1, depth=choice)                 # this is agents perspective
                self._bump_stat(f"L{choice}")
                if a not in valid_actions:
                    a = random.choice(valid_actions)
                return a
    
    
        # === Exploitation ===
        
        q_values = (self._symmetry_avg_q(state) if self.use_symmetry_policy else self._q_values_np(state))
        action = self._argmax_legal_with_tiebreak(q_values, valid_actions)
        self._bump_stat("exploit")
        if action not in valid_actions: raise ValueError(f"[act] DQN chose illegal action: {action} — valid: {valid_actions}")
        return action

    
    def remember(self, s, p, a, r, s2, p2, done, add_mirror=True, add_colorswap=True, add_mirror_colorswap=True):
        # base
        self.memory.push_1step(s, a, r, s2, done, p)
    
        # mirror (geometry only)
        if add_mirror:
            fs  = np.flip(s,  axis=-1).copy()
            fs2 = np.flip(s2, axis=-1).copy()
            fa  = 6 - a
            self.memory.push_1step(fs, fa, r, fs2, done, p)   # player unchanged
    
        # color-swap (swap channels 0/1, row/col unchanged), invert reward, flip player
        if add_colorswap:
            cs  = s.copy();  cs  = cs[[1,0,2,3], ...]
            cs2 = s2.copy(); cs2 = cs2[[1,0,2,3], ...]
            self.memory.push_1step(cs, a, -r, cs2, done, -p)
    
        # mirror + color-swap
        if add_mirror_colorswap:
            mcs  = np.flip(cs,  axis=-1).copy()
            mcs2 = np.flip(cs2, axis=-1).copy()
            cfa  = 6 - a
            self.memory.push_1step(mcs, cfa, -r, mcs2, done, -p)


    def replay(self, batch_size, mix_1step=0.7):
        def pack_many(states):
            # states are already (4,6,7) in mover POV
            arr = np.asarray(states, dtype=np.float32)
            return torch.tensor(arr, dtype=torch.float32, device=self.device)
        
        if (len(self.memory.bank_1) + len(self.memory.bank_n)) < batch_size: return
        
        (batch_1, batch_n), (idx_1, idx_n), (w_1, w_n) = self.memory.sample_mixed_seedaware(batch_size, mix=mix_1step, beta=self.per_beta, 
                                                                                            max_seed_frac=self.max_seed_frac, min_seed_frac=self.min_seed_frac)
    
        batch_all = list(batch_1) + list(batch_n)
        n1 = len(batch_1)
        is_w_all = (np.concatenate([w_1, w_n]) if (w_1.size or w_n.size) else np.ones((len(batch_all),), dtype=np.float32))
        states, next_states = [], []
        actions, rewards, dones, gammas_n = [], [], [], []

        for t in batch_all:
            if hasattr(t, "reward_n"):
                s, a, rN, nsN, dN, p, n_used = t
                states.append(s)
                next_states.append(nsN)
                actions.append(a)
                rewards.append(rN * self.reward_scale)
                dones.append(bool(dN))
                gammas_n.append(self.gamma ** n_used)
            else:
                s, a, r, ns, d, p = t
                states.append(s)
                next_states.append(ns)
                actions.append(a)
                rewards.append(r * self.reward_scale)
                dones.append(bool(d))
                gammas_n.append(self.gamma)

        X = pack_many(states)
        X_next = pack_many(next_states)

        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).view(-1, 1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)
        gpow_t = torch.tensor(gammas_n, dtype=torch.float32, device=self.device)

        q_all = self.model(X)
        q_sa = q_all.gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            legal_masks_t = ((X_next[:, 0, 0, :] + X_next[:, 1, 0, :]) == 0)  # [B, 7], bool
        
            q_next_online = self.model(X_next)                           # [B, 7]
            q_next_online = q_next_online.masked_fill(~legal_masks_t, -1e9)
            next_a = q_next_online.argmax(dim=1, keepdim=True)           # [B, 1]
        
            # Target net evaluates that action (also masked for safety)
            q_next_tgt = self.target_model(X_next)                       # [B, 7]
            q_next_tgt = q_next_tgt.masked_fill(~legal_masks_t, -1e9)
            next_q = q_next_tgt.gather(1, next_a).squeeze(1)             # [B]
        
            target = rewards_t + (~dones_t).float() * gpow_t * next_q
        
            # telemetry
            self.q_abs_hist.append(float(q_all.abs().mean().item()))
            self.q_max_hist.append(float(q_all.abs().max().item()))
            self.target_abs_hist.append(float(target.abs().mean().item()))


        per_sample_loss = self.loss_fn(q_sa, target)
        if self.per_beta > 0:
            isw_t = torch.as_tensor(is_w_all, device=self.device, dtype=per_sample_loss.dtype)
            isw_t = isw_t / (isw_t.mean() + 1e-8)   # mean≈1
            isw_t = torch.clamp(isw_t, 0.5, 3.0)    # gentle symmetric cap
            loss = (isw_t * per_sample_loss).mean()
        else: loss = per_sample_loss.mean()

        
        q_reg_coeff = 1e-6   # 1e-7 
        q_reg = q_reg_coeff * (q_all.pow(2).mean())
        loss = loss + q_reg

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0) #1.0
        self.optimizer.step()

        td_all = (q_sa - target).detach().cpu().numpy()
        td_all = np.nan_to_num(td_all, nan=0.0, posinf=1e6, neginf=-1e6)
        td_err_1 = td_all[:n1]
        td_err_n = td_all[n1:]
        self.memory.update_priorities(idx_1, td_err_1, indices_n=idx_n, td_errors_n=td_err_n)

        # --- telemetry tracking ---
        self.loss_hist.append(float(loss.detach().cpu().item()))
        self.td_hist.extend(td_all.tolist())             # TD history
        self.per_w_hist.extend(is_w_all.tolist())        # IS weights

        # --- β schedule update ---
        if self.per_beta < self.per_beta_end:
            self.per_beta = min(self.per_beta_end, self.per_beta + self.per_beta_step)

    @staticmethod
    def flip_transition(state, action, next_state):
        """
        Mirror a transition left-right in mover-POV.
        Flips planes: me(0), opp(1), col(3). Row (2) remains unchanged.
        """
        fs = state.copy()
        fs[0] = fs[0][:, ::-1]
        fs[1] = fs[1][:, ::-1]
        fs[3] = fs[3][:, ::-1]
    
        fns = next_state.copy()
        fns[0] = fns[0][:, ::-1]
        fns[1] = fns[1][:, ::-1]
        fns[3] = fns[3][:, ::-1]
    
        fa = 6 - action
        return fs, fa, fns


    def gave_one_ply_win(self, oriented_board, action):
        b1, _ = self._drop_piece(oriented_board, action, +1)
        if b1 is None:
            return False
        for oa in range(7):
            if b1[0, oa] != 0:
                continue
            b2, _ = self._drop_piece(b1, oa, -1)
            if b2 is not None and self._is_win(b2) == -1:
                return True
        return False
    
    def had_one_ply_win(self, oriented_board, valid_actions):
        for a in valid_actions:
            b1, _ = self._drop_piece(oriented_board, a, +1)
            if b1 is not None and self._is_win(b1) == +1:
                return True, a
        return False, None

    # -----------------
    
    def set_target_update_interval(self, steps: int):
        self.target_update_interval = int(steps)
    
    def reset_target_update_timer(self, current_step: int = 0):
        self._last_target_update_step = int(current_step)
    
    def soft_update_target(self, tau: float = None):
        tau = self.tau if tau is None else float(tau)
        with torch.no_grad():
            for tgt, src in zip(self.target_model.parameters(), self.model.parameters()):
                tgt.data.mul_(1.0 - tau).add_(tau * src.data)
    
    def maybe_update_target(self, global_env_step: int):
        if self.target_update_mode == "soft":
            self.soft_update_target()  # apply Polyak every step
            self._maybe_log_target_lag_(global_env_step)
        else:  # hard update
            if global_env_step - self._last_target_update_step >= self.target_update_interval:
                self.update_target_model()
                self._last_target_update_step = global_env_step
                self._maybe_log_target_lag_(global_env_step, force=True)

    @torch.no_grad()
    def _flatten_params_(self, module: torch.nn.Module) -> torch.Tensor:
        """Concatenate all trainable parameters into a single 1-D cpu tensor."""
        parts = []
        for p in module.parameters():
            if p.requires_grad:
                parts.append(p.detach().reshape(-1).cpu())
        return torch.cat(parts) if parts else torch.tensor([], dtype=torch.float32)

    @torch.no_grad()
    def _compute_target_lag_(self, eps: float = 1e-12) -> float | None:
        """Return ||θ-θ̄||/||θ|| for online vs target nets, or None if not available."""
        w  = self._flatten_params_(self.model)
        wt = self._flatten_params_(self.target_model)
        if w.numel() == 0 or wt.numel() == 0 or w.numel() != wt.numel():
            return None
        num = torch.linalg.norm(w - wt, ord=2)
        den = torch.linalg.norm(w,     ord=2).clamp(min=eps)
        return float((num / den).item())

    def _maybe_log_target_lag_(self, global_env_step: int, force: bool = False):
        if not force and (global_env_step - self._last_lag_log_step) < self.lag_log_interval: return
        val = self._compute_target_lag_()
        if val is not None:
            self.target_lag_hist.append(val)
            self._last_lag_log_step = global_env_step

    # ------------------------- symmetry helpers -------------------------

    @torch.no_grad()
    def _q_values_np(self, state) -> np.ndarray:
        """
        Q(s,·) from the online net as numpy.
        Accepts:
          - np.ndarray with shape (4,6,7)
          - torch.Tensor with shape (4,6,7) or (1,4,6,7)
        Returns: np.ndarray with shape (7,)
        """
        if isinstance(state, torch.Tensor):
            x = state.to(self.device, dtype=torch.float32)
            if x.ndim == 3:                # (4,6,7) -> (1,4,6,7)
                x = x.unsqueeze(0)
            elif x.ndim != 4:
                raise ValueError(f"_q_values_np: bad tensor shape {tuple(x.shape)}")
        else:
            x = torch.tensor(state, dtype=torch.float32, device=self.device)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            elif x.ndim != 4:
                raise ValueError(f"_q_values_np: bad ndarray shape {np.shape(state)}")

        q = self.model(x).squeeze(0).detach().cpu().numpy()  # [7]
        return q


    @torch.no_grad()
    def _symmetry_avg_q(self, state) -> np.ndarray:
        """
        Symmetry-averaged Q(s,·) under horizontal flip.
        Input `state` may be np.ndarray (4,6,7) or torch.Tensor (4,6,7)/(1,4,6,7).
        We flip: me(0), opp(1), col(3); keep row(2) unchanged.
        """
        # normalize to (1,4,6,7) float32 on device
        if isinstance(state, torch.Tensor):
            x = state.to(self.device, dtype=torch.float32)
            if x.ndim == 3: x = x.unsqueeze(0)
            elif x.ndim != 4: raise ValueError(f"_symmetry_avg_q: bad tensor shape {tuple(x.shape)}")
        else:
            x = torch.tensor(state, dtype=torch.float32, device=self.device)
            if x.ndim == 3: x = x.unsqueeze(0)
            elif x.ndim != 4: raise ValueError(f"_symmetry_avg_q: bad ndarray shape {np.shape(state)}")
    
        # forward on original
        q = self.model(x).squeeze(0)  # (7,)
    
        # build mirrored state: flip last dim (W) for me, opp, col; row untouched
        x_m = x.clone()
        x_m[:, 0] = torch.flip(x_m[:, 0], dims=(-1,))  # me
        x_m[:, 1] = torch.flip(x_m[:, 1], dims=(-1,))  # opp
        x_m[:, 3] = torch.flip(x_m[:, 3], dims=(-1,))  # col
    
        # forward on mirror, then unflip action indices
        q_m = self.model(x_m).squeeze(0)               # (7,) in mirrored frame
        q_m_unflip = torch.flip(q_m, dims=(0,))        # map back: a -> 6-a
    
        return (0.5 * (q + q_m_unflip)).detach().cpu().numpy()


    def _argmax_legal_with_tiebreak(self, q, valid_actions) -> int:
    
        va = np.asarray(list(valid_actions), dtype=int)
        if va.size == 0: raise ValueError("[_argmax_legal_with_tiebreak] no valid actions")
    
        q = np.asarray(q, dtype=np.float64)
        if q.shape[0] != 7: raise ValueError(f"expected q shape (7,), got {q.shape}")
    
        # mask illegal to -inf
        masked = np.full_like(q, -np.inf, dtype=float)
        masked[va] = q[va]
    
        tol = float(getattr(self, "tie_tol", 1e-6))
        qmax = float(np.max(masked))
        tied = va[masked[va] >= (qmax - tol)]
    
        rng = getattr(self, "rng", None)
        if not isinstance(rng, np.random.Generator):  rng = np.random.default_rng()
    
        # single winner or nothing weird
        if tied.size <= 1: return int(tied[0]) if tied.size == 1 else int(rng.choice(va))
    
        center_in_ties = (3 in tied)
    
        # CENTER-FIRST policy:
        # - If center (3) is among the tied winners → weight by center_tie_w on the tied set.
        # - Otherwise → uniform among the tied moves (no way to prefer center if it isn't tied).
        if center_in_ties:
            cw = np.array([1.0, 1.0, 1.2, 1.6, 1.2, 1.0, 1.0], dtype=np.float64)
            probs = cw[tied].astype(np.float64, copy=False)
            s = probs.sum()
            if s > 0 and np.isfinite(s): return int(rng.choice(tied, p=probs / s))
            # degenerate weights → fall back to uniform
        return int(rng.choice(tied))


    def _mirror_state_lr(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,4,6,7) torch tensor on any device."""
        x_m = x.clone()
        x_m[:, 0] = torch.flip(x_m[:, 0], dims=(-1,))  # me
        x_m[:, 1] = torch.flip(x_m[:, 1], dims=(-1,))  # opp
        x_m[:, 3] = torch.flip(x_m[:, 3], dims=(-1,))  # col
        return x_m

# ---------------------- debug / QA utilities ----------------------

    @staticmethod
    def _mirror_state_planes(state: np.ndarray) -> np.ndarray:
        """
        Mirror mover-POV planes horizontally.
        Flips: me(0), opp(1), col(3); keeps row(2) unchanged.
        state: (4,6,7) numpy array.
        """
        sm = state.copy()
        sm[0] = np.flip(sm[0], axis=-1)
        sm[1] = np.flip(sm[1], axis=-1)
        sm[3] = np.flip(sm[3], axis=-1)
        return sm
    
    @torch.no_grad()
    def symmetry_check_once(self, env, use_symmetry: bool = True, atol: float = 1e-5, verbose: bool = True):
        """
        Probe mirror consistency at the current env state (mover POV).
        Returns a dict with q, qm, ok flag, argmaxes, etc.
        """
        s  = env.get_state(perspective=env.current_player)   # (4,6,7)
        sm = self._mirror_state_planes(s)
    
        q  = (self._symmetry_avg_q(s)  if use_symmetry else self._q_values_np(s))
        qm = (self._symmetry_avg_q(sm) if use_symmetry else self._q_values_np(sm))
    
        qm_unflip = qm[::-1]                      # map mirrored actions back
        arg_q, arg_qm = int(q.argmax()), int(qm.argmax())
        ok = (arg_qm == 6 - arg_q) and np.allclose(q, qm_unflip, atol=atol)
        maxdiff = float(np.max(np.abs(q - qm_unflip)))
    
        if verbose:
            print("q: ",  np.round(q, 4))
            print("qm:",  np.round(qm, 4), "  (mirror argmax:", arg_qm, "→ should equal:", 6 - arg_q, ")")
            print("max diff after unflip:", maxdiff)
            print("ARGMAX:", arg_q, "| MIRROR ARGMAX:", arg_qm, "| OK:", ok)
            print("-" * 60)
    
        return {"q": q, "qm": qm, "qm_unflip": qm_unflip, "arg": arg_q, "arg_mirror": arg_qm, "ok": ok, "maxdiff": maxdiff}
    
    @torch.no_grad()
    def exploit_histogram_probe(
        self,
        env,
        trials: int = 100,
        max_random_prefix: int = 6,
        use_symmetry: bool = True,
        seed: int | None = 0,
        prefer_center: bool = False,
        tie_tol: float = 0.0,
        verbose: bool = True,
        plot: bool = False,
    ):
       
        # backup knobs
        bak_eps, bak_epsmin = self.epsilon, self.epsilon_min
        bak_sym = self.use_symmetry_policy
        bak_center, bak_tol = self.prefer_center, self.tie_tol
    
        # force pure exploit
        self.epsilon = 0.0
        self.epsilon_min = 0.0
        self.use_symmetry_policy = bool(use_symmetry)
        self.prefer_center = bool(prefer_center)
        self.tie_tol = float(tie_tol)
    
        rng = np.random.default_rng(seed)
        counts = Counter()
    
        for _ in range(trials):
            env.reset()
            for __ in range(int(rng.integers(0, max_random_prefix + 1))):
                if env.done: break
                a = int(rng.choice(env.available_actions()))
                env.step(a)
            if env.done: 
                continue
    
            s = env.get_state(perspective=env.current_player)
            q = (self._symmetry_avg_q(s) if self.use_symmetry_policy else self._q_values_np(s))
            a = self._argmax_legal_with_tiebreak(q, env.available_actions())
            counts[int(a)] += 1
    
        # restore knobs
        self.epsilon, self.epsilon_min = bak_eps, bak_epsmin
        self.use_symmetry_policy = bak_sym
        self.prefer_center, self.tie_tol = bak_center, bak_tol
    
        # ensure keys 0..6 exist
        out = {i: counts.get(i, 0) for i in range(7)}
    
        if verbose:
            print(f"Pure-exploit action histogram over {trials} randomized states:")
            print(out)
    
        if plot:
            import matplotlib.pyplot as plt
            xs = list(out.keys()); ys = list(out.values())
            plt.figure()
            plt.bar(xs, ys)
            plt.xticks(xs)
            plt.xlabel("Action (column index)")
            plt.ylabel(f"Count (out of {trials})")
            plt.title("Pure-exploit action histogram")
            # annotate bars
            for x, y in zip(xs, ys):
                plt.text(x, y, str(y), ha="center", va="bottom")
            plt.show()
    
        return out

    
    @torch.no_grad()
    def debug_policy_smoke_test(
        self,
        env,
        symmetry_atol: float = 1e-5,
        symmetry_states: list[int] = (0, 1, 2, 3, 4, 5, 8),
        hist_trials: int = 100,
        hist_max_prefix: int = 6,
        seed: int | None = 0,
    ):
        """
        Convenience: run a few symmetry checks (including empty board) + histogram probe.
        """
        # empty board
        env.reset()
        print("— Symmetry check on empty board —")
        self.symmetry_check_once(env, use_symmetry=True, atol=symmetry_atol, verbose=True)
    
        # a few partially-filled boards
        rng = np.random.default_rng(seed)
        for moves in symmetry_states:
            env.reset()
            for _ in range(moves):
                if env.done: break
                a = int(rng.choice(env.available_actions()))
                env.step(a)
            print(f"— Symmetry check after {moves} random moves —")
            self.symmetry_check_once(env, use_symmetry=True, atol=symmetry_atol, verbose=True)
    
        # histogram
        self.exploit_histogram_probe(
            env,
            trials=hist_trials,
            max_random_prefix=hist_max_prefix,
            use_symmetry=True,
            seed=seed,
            prefer_center=False,
            tie_tol=0.0,
            verbose=True,
        )
    