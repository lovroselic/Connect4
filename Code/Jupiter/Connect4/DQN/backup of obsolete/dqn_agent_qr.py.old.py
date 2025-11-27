# DQN/dqn_agent_qr.py

import random
import numpy as np
import torch

import torch.optim as optim
from collections import deque, defaultdict, Counter

from DQN.DQN_replay_memory_per import PrioritizedReplayMemory
from DQN.dqn_model_qr import QRDQN
from C4.fast_connect4_lookahead import Connect4Lookahead


class DQNAgent:
    def __init__(
        self,
        lr=2e-4,
        gamma=0.98,
        epsilon=0.08,
        epsilon_min=0.02,
        epsilon_decay=0.995,
        device=None,
        memory_capacity=500_000,

        # PER
        per_alpha=0.40,
        per_eps=1e-2,
        per_beta_start=0.45,
        per_beta_end=0.90,
        per_beta_steps=250_000,
        per_mix_1step=0.50,

        # target updates
        target_update_interval=500,   # kept for "hard"
        target_update_mode="soft",
        tau=0.005,

        # QR-DQN
        n_quantiles=51,
        huber_kappa=1.0,
    ):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- online / target models ----
        self.model = QRDQN(n_quantiles=n_quantiles).to(self.device)
        self.target_model = QRDQN(n_quantiles=n_quantiles).to(self.device)
        self.update_target_model()

        # ---- replay ----
        self.memory = PrioritizedReplayMemory(capacity=memory_capacity, alpha=per_alpha, eps=per_eps)

        # PER β schedule
        self.per_beta = float(per_beta_start)
        self.per_beta_end = float(per_beta_end)
        steps = max(1, int(per_beta_steps))
        self.per_beta_step = (self.per_beta_end - self.per_beta) / steps
        self.per_mix_1step = float(per_mix_1step)

        # opt / loss knobs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.lookahead = Connect4Lookahead()
        self.reward_scale = 0.01

        # guard / center-start (controlled by phase)
        self.guard_prob = 0.0
        self.guard_boost_start = 5
        self.guard_boost_end   = 17
        self.center_start = 0.3
        self.win_now_prob = 1.0

        # telemetry buffers
        self.td_hist = deque(maxlen=50_000)
        self.loss_hist = deque(maxlen=10_000)
        self.per_w_hist = deque(maxlen=50_000)
        self.sample_mix_hist = deque(maxlen=20_000)

        self.target_update_interval = int(target_update_interval)
        self.target_update_mode = str(target_update_mode)
        self.tau = float(tau)
        self._last_target_update_step = 0

        self.q_abs_hist      = deque(maxlen=100_000)
        self.q_max_hist      = deque(maxlen=100_000)
        self.target_abs_hist = deque(maxlen=100_000)

        self.center_forced_used = False

        # --- policy symmetry + tie-breaking knobs ---
        self.use_symmetry_policy = True
        self.tie_tol = 1e-9
        self.prefer_center = True
        self.rng = np.random.default_rng(0)

        self.max_seed_frac = 0.90
        self.min_seed_frac = 0.10

        self.target_lag_hist = deque(maxlen=50_000)
        self.lag_log_interval = 250
        self._last_lag_log_step = 0
        self.target_lag_hist.append(0.0)

        self.act_stats = defaultdict(int)
        self._act_stats_total = 0
        self.opp_act_stats = defaultdict(int)
        self._opp_act_total = 0

        # ===== QR-DQN bits =====
        self.n_quantiles = int(n_quantiles)
        self.huber_kappa = float(huber_kappa)
        taus = (torch.arange(self.n_quantiles, dtype=torch.float32) + 0.5) / self.n_quantiles  # [N]
        self.register_buffer_("taus", taus.view(1, 1, -1))   # broadcastable [1,1,N]

    # small helper to register buffers on self without subclassing nn.Module
    def register_buffer_(self, name, tensor):
        setattr(self, name, tensor.to(self.device))

    # ------------------------- helper encodings -------------------------

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # ------------------------- tactical guard -------------------------

    @staticmethod
    def _drop_piece(board: np.ndarray, col: int, player: int):
        if board[0, col] != 0:
            return None, None
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
                if b2 is None:
                    continue
                if self.lookahead.check_win(b2, -1):
                    opp_threat = True
                    break

            if not opp_threat:
                safe_actions.append(a)

        return must_play, safe_actions

    @staticmethod
    def _center_col_is_empty(state) -> bool:
        col_occ = state[0, :, 3] + state[1, :, 3]
        return (col_occ == 0).all()


    def board_to_state(self, board: np.ndarray, player: int) -> np.ndarray:
        agent_plane = (board == player).astype(np.float32)
        opp_plane   = (board == -player).astype(np.float32)
        # >>> change to [0,1] <<<
        H, W = board.shape
        row_plane = np.tile(np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None], (1, W))
        col_plane = np.tile(np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :], (H, 1))
        return np.stack([agent_plane, opp_plane, row_plane, col_plane])


    # ------------------------- policy -------------------------

    def act(self, state: np.ndarray, valid_actions, depth: int, strategy_weights=None, ply=None):
        self._act_stats_total += 1
        assert len(valid_actions) > 0, f"ACT called with empty valid actions {valid_actions}"
        board = state[0] - state[1]  # mover (+1) minus opp (–1)

        # 1-ply immediate win
        win_now, win_a = self.had_one_ply_win(board, valid_actions)
        if win_now and (random.random() < float(self.win_now_prob)):
            self._bump_stat("win_now")
            return int(win_a)

        # Center start
        if not self.center_forced_used and ply is not None and ply <= 1:
            center_start_on = (random.random() < self.center_start)
            if center_start_on and 3 in valid_actions:
                if self._center_col_is_empty(state):
                    self.center_forced_used = True
                    self._bump_stat("center")
                    return 3

        # Tactical guard window
        guard_on = False
        if ply is not None and (self.guard_boost_start <= ply <= self.guard_boost_end):
            guard_on = (random.random() < float(self.guard_prob))

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

        # ε-greedy
        if random.random() < self.epsilon:
            self._bump_stat("epsilon")
            if strategy_weights is None:
                strategy_weights = [1, 0, 0, 0, 0, 0, 0, 0]
            assert len(strategy_weights) == 8, f"Expected 8 strategy weights, got {len(strategy_weights)}"
            choice = random.choices(range(8), weights=strategy_weights)[0]
            if choice == 0:
                action = random.choice(valid_actions)
                self._bump_stat("rand")
                return action
            else:
                a = self.lookahead.n_step_lookahead(board, player=+1, depth=choice)
                self._bump_stat(f"L{choice}")
                if a not in valid_actions:
                    a = random.choice(valid_actions)
                return a

        # Exploitation (use mean over quantiles as Q)
        q_values = (self._symmetry_avg_q(state) if self.use_symmetry_policy else self._q_values_np(state))
        action = self._argmax_legal_with_tiebreak(q_values, valid_actions)
        self._bump_stat("exploit")
        if action not in valid_actions:
            raise ValueError(f"[act] chose illegal action: {action} — valid: {valid_actions}")
        return action

    # ------------------------- memory -------------------------

    def remember(self, s, p, a, r, s2, p2, done, add_mirror=True, add_colorswap=True, add_mirror_colorswap=True):
        self.memory.push_1step(s, a, r, s2, done, p)
    
        if add_mirror:
            fs  = np.flip(s,  axis=-1).copy()
            fs2 = np.flip(s2, axis=-1).copy()
            fa  = 6 - a
            self.memory.push_1step(fs, fa, r, fs2, done, p)
    
        # ensure cs/cs2 exist if mirror_colorswap is requested but colorswap is off
        cs = cs2 = None
        if add_colorswap or add_mirror_colorswap:
            cs  = s.copy();  cs  = cs[[1,0,2,3], ...]
            cs2 = s2.copy(); cs2 = cs2[[1,0,2,3], ...]
    
        if add_colorswap:
            self.memory.push_1step(cs, a, -r, cs2, done, -p)
    
        if add_mirror_colorswap:
            mcs  = np.flip(cs,  axis=-1).copy()
            mcs2 = np.flip(cs2, axis=-1).copy()
            cfa  = 6 - a
            self.memory.push_1step(mcs, cfa, -r, mcs2, done, -p)


    # ------------------------- QR replay -------------------------

    def replay(self, batch_size, mix_1step=0.7):
        def pack_many(states):
            arr = np.asarray(states, dtype=np.float32)
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        if (len(self.memory.bank_1) + len(self.memory.bank_n)) < batch_size:
            return

        (batch_1, batch_n), (idx_1, idx_n), (w_1, w_n) = self.memory.sample_mixed_seedaware(
            batch_size, mix=mix_1step, beta=self.per_beta,
            max_seed_frac=self.max_seed_frac, min_seed_frac=self.min_seed_frac
        )

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

        X = pack_many(states)          # [B,4,6,7]
        X_next = pack_many(next_states)

        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).view(-1, 1, 1)  # for gather over action
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)  # [B,1]
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device).view(-1, 1)         # [B,1]
        gpow_t = torch.tensor(gammas_n, dtype=torch.float32, device=self.device).view(-1, 1)    # [B,1]

        # ---- predict quantiles for current state/action ----
        q_all = self.model(X)  # [B,7,N]
        q_sa = q_all.gather(1, actions_t.expand(-1, 1, self.n_quantiles)).squeeze(1)  # [B,N]

        with torch.no_grad():
            # Legal mask from top row availability
            legal_masks_t = ((X_next[:, 0, 0, :] + X_next[:, 1, 0, :]) == 0)  # [B,7]

            # Double DQN: online chooses action by mean over quantiles
            q_next_online = self.model(X_next).mean(dim=-1)                    # [B,7]
            q_next_online = q_next_online.masked_fill(~legal_masks_t, -1e9)
            next_a = q_next_online.argmax(dim=1, keepdim=True)                 # [B,1]

            # Target net provides quantiles at that action
            q_next_tgt = self.target_model(X_next)                             # [B,7,N]
            q_next_tgt = q_next_tgt.masked_fill(~legal_masks_t.unsqueeze(-1), -1e9)
            q_next_tgt_sa = q_next_tgt.gather(1, next_a.view(-1, 1, 1).expand(-1, 1, self.n_quantiles)).squeeze(1)  # [B,N]

            target_quantiles = rewards_t + (~dones_t).float() * gpow_t * q_next_tgt_sa  # [B,N]

            # telemetry (means)
            q_mean_abs = q_all.mean(dim=-1).abs().mean()
            self.q_abs_hist.append(float(q_mean_abs.item()))
            self.q_max_hist.append(float(q_all.abs().amax().item()))
            self.target_abs_hist.append(float(target_quantiles.abs().mean().item()))

        # ---- Quantile Huber loss (pairwise) ----
        # u_{i,j} = Z'_j - θ_i
        diff = target_quantiles.unsqueeze(1) - q_sa.unsqueeze(2)          # [B,N,N]
        huber = self._huber(diff, kappa=self.huber_kappa)                 # [B,N,N]
        inv = (diff.detach() < 0.0).float()                               # [B,N,N]  1_{u<0}
        weight = torch.abs(self.taus - inv)                               # [B,N,N], |τ_i - 1_{u<0}|
        per_sample_qr_loss = (weight * huber).mean(dim=(1, 2))            # [B]

        # PER importance weights
        if self.per_beta > 0:
            isw_t = torch.as_tensor(is_w_all, device=self.device, dtype=per_sample_qr_loss.dtype)
            isw_t = isw_t / (isw_t.mean() + 1e-8)
            isw_t = torch.clamp(isw_t, 0.5, 3.0)
            loss = (isw_t * per_sample_qr_loss).mean()
        else:
            loss = per_sample_qr_loss.mean()

        # small L2 on quantiles
        q_reg = 1e-6 * (q_all.pow(2).mean())
        loss = loss + q_reg

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        # ---- priorities: use per-sample QR loss (mean over quantiles) ----
        priorities = per_sample_qr_loss.detach().cpu().numpy()  # [B]
        priorities = np.nan_to_num(priorities, nan=0.0, posinf=1e6, neginf=-1e6)
        self.memory.update_priorities(
            idx_1, priorities[:n1],
            indices_n=idx_n, td_errors_n=priorities[n1:]
        )

        # telemetry
        self.loss_hist.append(float(loss.detach().cpu().item()))
        self.td_hist.extend(priorities.tolist())       # log per-sample QR loss
        self.per_w_hist.extend(is_w_all.tolist())

        # β schedule
        if self.per_beta < self.per_beta_end:
            self.per_beta = min(self.per_beta_end, self.per_beta + self.per_beta_step)

    # ---- loss pieces ----
    @staticmethod
    def _huber(x: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        if kappa <= 0:
            return torch.abs(x)
        absx = torch.abs(x)
        cap = torch.tensor(kappa, device=x.device, dtype=x.dtype)
        quad = torch.minimum(absx, cap)
        return 0.5 * quad ** 2 + kappa * (absx - quad)

    # ----------------- misc helpers -----------------

    @staticmethod
    def flip_transition(state, action, next_state):
        fs = state.copy()
        fs[0] = fs[0][:, ::-1]; fs[1] = fs[1][:, ::-1]; fs[3] = fs[3][:, ::-1]
        fns = next_state.copy()
        fns[0] = fns[0][:, ::-1]; fns[1] = fns[1][:, ::-1]; fns[3] = fns[3][:, ::-1]
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

    # ----------------- target updates -----------------

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
            self.soft_update_target()
            self._maybe_log_target_lag_(global_env_step)
        else:
            if global_env_step - self._last_target_update_step >= self.target_update_interval:
                self.update_target_model()
                self._last_target_update_step = global_env_step
                self._maybe_log_target_lag_(global_env_step, force=True)

    @torch.no_grad()
    def _flatten_params_(self, module: torch.nn.Module) -> torch.Tensor:
        parts = []
        for p in module.parameters():
            if p.requires_grad:
                parts.append(p.detach().reshape(-1).cpu())
        return torch.cat(parts) if parts else torch.tensor([], dtype=torch.float32)

    @torch.no_grad()
    def _compute_target_lag_(self, eps: float = 1e-12) -> float | None:
        w  = self._flatten_params_(self.model)
        wt = self._flatten_params_(self.target_model)
        if w.numel() == 0 or wt.numel() == 0 or w.numel() != wt.numel():
            return None
        num = torch.linalg.norm(w - wt, ord=2)
        den = torch.linalg.norm(w,     ord=2).clamp(min=eps)
        return float((num / den).item())

    def _maybe_log_target_lag_(self, global_env_step: int, force: bool = False):
        if not force and (global_env_step - self._last_lag_log_step) < self.lag_log_interval:
            return
        val = self._compute_target_lag_()
        if val is not None:
            self.target_lag_hist.append(val)
            self._last_lag_log_step = global_env_step

    # ------------------------- symmetry helpers -------------------------

    @torch.no_grad()
    def _q_values_np(self, state) -> np.ndarray:
        """
        Return mean Q(s,·) (mean over quantiles) as numpy [7].
        Accepts np.ndarray (4,6,7) or torch.Tensor (4,6,7)/(1,4,6,7).
        """
        if isinstance(state, torch.Tensor):
            x = state.to(self.device, dtype=torch.float32)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            elif x.ndim != 4:
                raise ValueError(f"_q_values_np: bad tensor shape {tuple(x.shape)}")
        else:
            x = torch.tensor(state, dtype=torch.float32, device=self.device)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            elif x.ndim != 4:
                raise ValueError(f"_q_values_np: bad ndarray shape {np.shape(state)}")

        q = self.model.q_mean(x).squeeze(0).detach().cpu().numpy()  # [7]
        return q

    @torch.no_grad()
    def _symmetry_avg_q(self, state) -> np.ndarray:
        """
        Symmetry-averaged mean-Q(s,·) under horizontal flip.
        """
        if isinstance(state, torch.Tensor):
            x = state.to(self.device, dtype=torch.float32)
            if x.ndim == 3: x = x.unsqueeze(0)
            elif x.ndim != 4: raise ValueError(f"_symmetry_avg_q: bad tensor shape {tuple(x.shape)}")
        else:
            x = torch.tensor(state, dtype=torch.float32, device=self.device)
            if x.ndim == 3: x = x.unsqueeze(0)
            elif x.ndim != 4: raise ValueError(f"_symmetry_avg_q: bad ndarray shape {np.shape(state)}")

        q = self.model.q_mean(x).squeeze(0)

        x_m = x.clone()
        x_m[:, 0] = torch.flip(x_m[:, 0], dims=(-1,))
        x_m[:, 1] = torch.flip(x_m[:, 1], dims=(-1,))
        x_m[:, 3] = torch.flip(x_m[:, 3], dims=(-1,))

        q_m = self.model.q_mean(x_m).squeeze(0)
        q_m_unflip = torch.flip(q_m, dims=(0,))
        return (0.5 * (q + q_m_unflip)).detach().cpu().numpy()

    def _argmax_legal_with_tiebreak(self, q, valid_actions) -> int:
        va = np.asarray(list(valid_actions), dtype=int)
        if va.size == 0:
            raise ValueError("[_argmax_legal_with_tiebreak] no valid actions")

        q = np.asarray(q, dtype=np.float64)
        if q.shape[0] != 7:
            raise ValueError(f"expected q shape (7,), got {q.shape}")

        masked = np.full_like(q, -np.inf, dtype=float)
        masked[va] = q[va]

        tol = float(getattr(self, "tie_tol", 1e-6))
        qmax = float(np.max(masked))
        tied = va[masked[va] >= (qmax - tol)]

        rng = getattr(self, "rng", None)
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng()

        if tied.size <= 1:
            return int(tied[0]) if tied.size == 1 else int(rng.choice(va))

        # prefer center only if it's in the tied set
        if 3 in tied:
            cw = np.array([1.0, 1.0, 1.2, 1.6, 1.2, 1.0, 1.0], dtype=np.float64)
            probs = cw[tied]
            s = probs.sum()
            if s > 0 and np.isfinite(s):
                return int(rng.choice(tied, p=probs / s))
        return int(rng.choice(tied))

    def _mirror_state_lr(self, x: torch.Tensor) -> torch.Tensor:
        x_m = x.clone()
        x_m[:, 0] = torch.flip(x_m[:, 0], dims=(-1,))
        x_m[:, 1] = torch.flip(x_m[:, 1], dims=(-1,))
        x_m[:, 3] = torch.flip(x_m[:, 3], dims=(-1,))
        return x_m

    # ---------------------- debug / QA utilities ----------------------

    def _bump_stat(self, key: str):
        self.act_stats[key] += 1

    def act_stats_summary(self, normalize=True):
        total = max(1, self._act_stats_total)
        if not normalize:
            return dict(self.act_stats)
        return {k: v / total for k, v in self.act_stats.items()}

    def _bump_opp(self, key: str):
        self.opp_act_stats[key] += 1

    def opp_act_stats_summary(self, normalize: bool = True):
        total = max(1, self._opp_act_total)
        if not normalize:
            return dict(self.opp_act_stats)
        return {k: v / total for k, v in self.opp_act_stats.items()}

    @staticmethod
    def _mirror_state_planes(state: np.ndarray) -> np.ndarray:
        sm = state.copy()
        sm[0] = np.flip(sm[0], axis=-1)
        sm[1] = np.flip(sm[1], axis=-1)
        sm[3] = np.flip(sm[3], axis=-1)
        return sm

    @torch.no_grad()
    def symmetry_check_once(self, env, use_symmetry: bool = True, atol: float = 1e-5, verbose: bool = True):
        s  = env.get_state(perspective=env.current_player)
        sm = self._mirror_state_planes(s)

        q  = (self._symmetry_avg_q(s)  if use_symmetry else self._q_values_np(s))
        qm = (self._symmetry_avg_q(sm) if use_symmetry else self._q_values_np(sm))

        qm_unflip = qm[::-1]
        arg_q, arg_qm = int(q.argmax()), int(qm.argmax())
        ok = (arg_qm == 6 - arg_q) and np.allclose(q, qm_unflip, atol=atol)
        maxdiff = float(np.max(np.abs(q - qm_unflip)))

        if verbose:
            print("q: ",  np.round(q, 4))
            print("qm:",  np.round(qm, 4), "  (mirror argmax:", arg_qm, "→ should equal:", 6 - arg_q, ")")
            print("max diff after unflip:", maxdiff)
            print("ARGMAX:", arg_q, "| MIRROR ARGMAX:", arg_qm, "| OK:", ok)
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

        # pure exploit
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

        out = {i: counts.get(i, 0) for i in range(7)}
        if verbose:
            print(f"Pure-exploit action histogram over {trials} randomized states:")
            print(out)

        if plot:
            import matplotlib.pyplot as plt
            xs = list(out.keys()); ys = list(out.values())
            plt.figure()
            plt.bar(xs, ys)
            plt.xticks(xs); plt.xlabel("Action (column index)")
            plt.ylabel(f"Count (out of {trials})")
            plt.title("Pure-exploit action histogram")
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
        env.reset()
        print("— Symmetry check on empty board —")
        self.symmetry_check_once(env, use_symmetry=True, atol=symmetry_atol, verbose=True)

        rng = np.random.default_rng(seed)
        for moves in symmetry_states:
            env.reset()
            for _ in range(moves):
                if env.done: break
                a = int(rng.choice(env.available_actions()))
                env.step(a)
            print(f"— Symmetry check after {moves} random moves —")
            self.symmetry_check_once(env, use_symmetry=True, atol=symmetry_atol, verbose=True)

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
