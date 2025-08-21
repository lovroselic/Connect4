# dqn_agent.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from DQN.DQN_replay_memory_per import PrioritizedReplayMemory
from DQN.dqn_model import DQN
from C4.connect4_lookahead import Connect4Lookahead


class DQNAgent:
    def __init__(
        self,
        lr=0.001,
        gamma=0.97,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        device=None,
        memory_capacity=200_000,
        per_alpha=0.55,          # was 0.6  -> slightly less skewed priorities
        per_eps=1e-2,           # was 1e-3 -> prevents tiny probs from collapsing IS weights
        per_beta_start=0.5,     # was 0.4  -> start correcting earlier
        per_beta_end=0.9,
        per_beta_steps=150_000, # was 200k -> reach 1.0 sooner
        per_mix_1step=0.6,      # default blend (tweak per-phase via replay(...))

        ):
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.update_target_model()

        self.memory = PrioritizedReplayMemory(
            capacity=memory_capacity, alpha=per_alpha, eps=per_eps
        )

        # PER β schedule
        self.per_beta = float(per_beta_start)
        self.per_beta_end = float(per_beta_end)
        steps = max(1, int(per_beta_steps))
        self.per_beta_step = (self.per_beta_end - self.per_beta) / steps

        self.per_mix_1step = float(per_mix_1step)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        # we weight per-sample, so keep 'none'
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.lookahead = Connect4Lookahead()

        # keep small scale if env rewards are ~±100
        self.reward_scale = 0.01
        self.guard_prob = 0.1 # tactical guard
        
        self.td_hist = deque(maxlen=50_000)             # recent per-sample TD errors
        self.loss_hist = deque(maxlen=10_000)           # per-update loss
        self.per_w_hist = deque(maxlen=50_000)          # recent importance weights (IS)
        self.sample_mix_hist = deque(maxlen=20_000)     # tuples: (is_nstep, player_anchor)

    # ------------------------- helper encodings -------------------------

    @staticmethod
    def _agent_planes(board: np.ndarray, agent_side: int):
        """Return (2,6,7) planes in {0,1}: plane0=agent stones, plane1=opponent stones."""
        agent_plane = (board == agent_side).astype(np.float32)
        opp_plane = (board == -agent_side).astype(np.float32)
        return np.stack([agent_plane, opp_plane], axis=0)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # ------------------------- tactical guard -------------------------

    @staticmethod
    def _drop_piece(board: np.ndarray, col: int, player: int):
        """Simulate dropping a piece; return (new_board, placed_row) or (None, None) if full."""
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
        """Return +1/-1 winner if 4-in-a-row exists, else 0."""
        R, C = board.shape
        # horiz
        for r in range(R):
            for c in range(C - 3):
                s = np.sum(board[r, c : c + 4])
                if abs(s) == 4:
                    return int(np.sign(s))
        # vert
        for r in range(R - 3):
            for c in range(C):
                s = np.sum(board[r : r + 4, c])
                if abs(s) == 4:
                    return int(np.sign(s))
        # diag /
        for r in range(3, R):
            for c in range(C - 3):
                s = (
                    board[r, c]
                    + board[r - 1, c + 1]
                    + board[r - 2, c + 2]
                    + board[r - 3, c + 3]
                )
                if abs(s) == 4:
                    return int(np.sign(s))
        # diag \
        for r in range(R - 3):
            for c in range(C - 3):
                s = (
                    board[r, c]
                    + board[r + 1, c + 1]
                    + board[r + 2, c + 2]
                    + board[r + 3, c + 3]
                )
                if abs(s) == 4:
                    return int(np.sign(s))
        return 0

    def _tactical_guard(self, board: np.ndarray, valid_actions, player: int):
        """
        1) If we can win now, return that move (must_play).
        2) Else, filter actions that allow opponent to win next ply.
        Returns (must_play, safe_actions). If safe_actions is empty, returns all valid_actions.
        """
        # 1) our immediate wins
        for a in valid_actions:
            b1, _ = self._drop_piece(board, a, player)
            if b1 is None:
                continue
            if self._is_win(b1) == player:
                return a, valid_actions  # must play the win

        # 2) avoid handing opponent a 1-move win
        opp = -player
        safe = []
        for a in valid_actions:
            b1, _ = self._drop_piece(board, a, player)
            if b1 is None:
                continue
            opp_valid = [c for c in range(7) if b1[0, c] == 0]
            opp_wins = False
            for oa in opp_valid:
                b2, _ = self._drop_piece(b1, oa, opp)
                if b2 is None:
                    continue
                if self._is_win(b2) == opp:
                    opp_wins = True
                    break
            if not opp_wins:
                safe.append(a)

        return None, (safe if safe else valid_actions)

    # ------------------------- action selection -------------------------

    def act(self, state, valid_actions, player, depth, strategy_weights=None):
        board = np.array(state)

        guard_on = (random.random() < self.guard_prob)
        
        must_play, safe_actions = self._tactical_guard(board, valid_actions, player)
        
        # --- one-ply tactical guard applies to both explore & exploit ---
        if guard_on:
            #must_play, safe_actions = self._tactical_guard(board, valid_actions, player)
            if must_play is not None:
                return must_play

            valid_actions = safe_actions


        # Explore
        if random.random() < self.epsilon:
            if strategy_weights is None:
                strategy_weights = [1, 0, 0, 0, 0, 0, 0, 0]
            choice = random.choices(range(8), weights=strategy_weights)[0]
            if choice == 0:
                return random.choice(valid_actions)
            else:
                a = self.lookahead.n_step_lookahead(board, player=player, depth=choice)
                return a if a in valid_actions else random.choice(valid_actions)

        # Exploit
        #exploit guard -  always
        if must_play is not None:
            return must_play
    
        x = torch.tensor(
            self._agent_planes(board, agent_side=player),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model(x).cpu().numpy()[0]
        masked = np.full_like(q_values, -np.inf)
        
        for a in valid_actions:
            masked[a] = q_values[a]
        return int(np.argmax(masked))

    def act_with_curriculum(self, *args, **kwargs):
        return self.act(*args, **kwargs)

    # (kept for compatibility; your training uses NStepBuffer -> push_*_aug)
    def remember(
        self,
        s,
        p,
        a,
        r,
        s2,
        p2,
        done,
        add_mirror=True,
        add_colorswap=True,
        add_mirror_colorswap=True,
    ):
        # original (RAW reward)
        self.memory.push_1step(s, a, r, s2, done, p)

        if add_mirror:
            fs = np.flip(s, axis=-1).copy()
            fs2 = np.flip(s2, axis=-1).copy()
            fa = 6 - a
            self.memory.push_1step(fs, fa, r, fs2, done, p)

        if add_colorswap:
            cs, cs2 = -s, -s2
            self.memory.push_1step(cs, a, -r, cs2, done, -p)

        if add_mirror_colorswap:
            mcs = np.flip(-s, axis=-1).copy()
            mcs2 = np.flip(-s2, axis=-1).copy()
            cfa = 6 - a
            self.memory.push_1step(mcs, cfa, -r, mcs2, done, -p)

    # ------------------------- training step -------------------------

    def replay(self, batch_size, mix_1step=0.7):
        """
        Train on a blend of 1-step and n-step transitions.
        mix_1step: fraction of the minibatch drawn from the 1-step bank.
        """
        if (len(self.memory.bank_1) + len(self.memory.bank_n)) < batch_size:
            return

        # 1) sample
        (batch_1, batch_n), (idx_1, idx_n), (w_1, w_n) = self.memory.sample_mixed(
            batch_size, mix=mix_1step, beta=self.per_beta
        )
        batch_all = list(batch_1) + list(batch_n)
        n1 = len(batch_1)

        is_w_all = (
            np.concatenate([w_1, w_n]) if (w_1.size or w_n.size) else np.ones(
                (len(batch_all),), dtype=np.float32
            )
        )
        
        # normalize then cap
        if is_w_all.size > 0:
            is_w_all = is_w_all / (is_w_all.max() + 1e-8)
            np.clip(is_w_all, 0.05, 1.0, out=is_w_all)


        # 2) unpack -> tensors
        states, next_states = [], []
        s_players, ns_players = [], []
        actions, rewards, dones, gammas_n = [], [], [], []

        for t in batch_all:
            if hasattr(t, "reward_n"):  # NStepTransition
                s, a, rN, nsN, dN, p, n_used = (
                    t.state,
                    t.action,
                    t.reward_n,
                    t.next_state_n,
                    t.done_n,
                    t.player,
                    int(t.n_steps),
                )
                p_next = p if (n_used % 2 == 0) else -p
                states.append(s)
                next_states.append(nsN)
                s_players.append(p)
                ns_players.append(p_next)
                actions.append(a)
                rewards.append(rN * self.reward_scale)  # scale here
                dones.append(bool(dN))
                gammas_n.append(self.gamma ** n_used)
            else:  # 1-step Transition
                s, a, r, ns, d, p = (
                    t.state,
                    t.action,
                    t.reward,
                    t.next_state,
                    t.done,
                    t.player,
                )
                p_next = -p
                states.append(s)
                next_states.append(ns)
                s_players.append(p)
                ns_players.append(p_next)
                actions.append(a)
                rewards.append(r * self.reward_scale)  # scale here as well
                dones.append(bool(d))
                gammas_n.append(self.gamma)  # ^1

        def pack_many(boards, players):
            planes = [self._agent_planes(b, p) for b, p in zip(boards, players)]
            return torch.tensor(
                np.asarray(planes), dtype=torch.float32, device=self.device
            )

        X = pack_many(states, s_players)
        X_next = pack_many(next_states, ns_players)

        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).view(
            -1, 1
        )
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).view(
            -1
        )
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device).view(-1)
        gpow_t = torch.tensor(gammas_n, dtype=torch.float32, device=self.device).view(
            -1
        )
        isw_t = torch.tensor(is_w_all, dtype=torch.float32, device=self.device).view(-1)
        isw_t = isw_t / (isw_t.mean() + 1e-8)

        # 3) Q(s,a) and Double-DQN target
        q_all = self.model(X)  # (B,7)
        q_sa = q_all.gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_a = self.model(X_next).argmax(dim=1, keepdim=True)  # (B,1)
            next_q = self.target_model(X_next).gather(1, next_a).squeeze(1)
            target = rewards_t + (~dones_t).float() * gpow_t * next_q

        # 4) loss
        per_sample_loss = F.smooth_l1_loss(q_sa, target, reduction="none")
        loss = (isw_t * per_sample_loss).mean() if self.per_beta > 0 else per_sample_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # 5) update PER priorities
        td_all = (q_sa - target).detach().cpu().numpy()
        td_all = np.nan_to_num(td_all, nan=0.0, posinf=1e6, neginf=-1e6)
        td_err_1 = td_all[:n1]
        td_err_n = td_all[n1:]
        self.memory.update_priorities(idx_1, td_err_1, indices_n=idx_n, td_errors_n=td_err_n)
        
        # --- telemetry ---
        # loss
        self.loss_hist.append(float(loss.detach().cpu().item()))
        # per-sample td + IS weights
        self.td_hist.extend(td_all.tolist())
        self.per_w_hist.extend(is_w_all.tolist())
        # sample composition: mark which were n-step and which anchor player they came from
        for _ in range(n1):    self.sample_mix_hist.append((0, int(s_players[len(self.sample_mix_hist) % len(s_players)])))
        for _ in range(len(batch_n)): self.sample_mix_hist.append((1, int(s_players[(len(self.sample_mix_hist)+1) % len(s_players)])))


        # 6) anneal β
        if self.per_beta < self.per_beta_end:
            self.per_beta = min(self.per_beta_end, self.per_beta + self.per_beta_step)
            

    # ---------------------------------------------------------------------

    @staticmethod
    def flip_transition(state, action, next_state):
        flipped_state = np.flip(state, axis=-1).copy()
        flipped_next_state = np.flip(next_state, axis=-1).copy()
        flipped_action = 6 - action
        return flipped_state, flipped_action, flipped_next_state

    # ---------------------------------------------------------------------
    
    def gave_one_ply_win(self, board, action, player):
        """Return True if taking `action` lets the opponent win next ply."""
        b1, _ = self._drop_piece(board, action, player)
        if b1 is None: 
            return False
        opp = -player
        for oa in range(7):
            if b1[0, oa] != 0: 
                continue
            b2, _ = self._drop_piece(b1, oa, opp)
            if b2 is not None and self._is_win(b2) == opp:
                return True
        return False
    
    def had_one_ply_win(self, board, valid_actions, player):
        """Return (has_win, winning_action) for our immediate wins."""
        for a in valid_actions:
            b1, _ = self._drop_piece(board, a, player)
            if b1 is not None and self._is_win(b1) == player:
                return True, a
        return False, None
