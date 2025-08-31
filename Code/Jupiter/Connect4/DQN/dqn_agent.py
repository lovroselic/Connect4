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
        epsilon=0.08,
        epsilon_min=0.02,
        epsilon_decay=0.995,
        device=None,
        memory_capacity=500_000,
        per_alpha=0.55,
        per_eps=1e-2,
        per_beta_start=0.5,
        per_beta_end=0.9,
        per_beta_steps=150_000,
        per_mix_1step=0.6,
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
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.lookahead = Connect4Lookahead()

        self.reward_scale = 0.01
        self.guard_prob = 0.5

        self.td_hist = deque(maxlen=50_000)
        self.loss_hist = deque(maxlen=10_000)
        self.per_w_hist = deque(maxlen=50_000)
        self.sample_mix_hist = deque(maxlen=20_000)

    # ------------------------- helper encodings -------------------------

    def _agent_planes(self, board: np.ndarray, agent_side: int) -> np.ndarray:
        agent_plane = (board[0] if agent_side == 1 else board[1])
        opp_plane   = (board[1] if agent_side == 1 else board[0])
        row_plane   = board[2]
        col_plane   = board[3]
        return np.stack([agent_plane, opp_plane, row_plane, col_plane])



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

    def _tactical_guard(self, board: np.ndarray, valid_actions, player: int):
        """
        Simulates one-move lookahead to block instant win/loss.
        Returns: (must_play: int or None, safe_actions: list)
        """
    
        must_play = None
        safe_actions = []
    
        for a in valid_actions:
            b1, _ = self._drop_piece(board, a, player)
            if b1 is None:
                continue
            
            # Check if we would win by playing this
            if self.lookahead.check_win(b1, player):
                must_play = a
                safe_actions.append(a)
            else:
                # Also check opponent's chance to win right after
                opp_win = False
                for opp_a in valid_actions:
                    b2, _ = self._drop_piece(b1, opp_a, -player)
                    if b2 is None:
                        continue
                    if self.lookahead.check_win(b2, -player):
                        opp_win = True
                        break
                if not opp_win:
                    safe_actions.append(a)
    
        return must_play, safe_actions


    # ------------------------- action selection -------------------------
    
    def state_to_board(self, state: np.ndarray) -> np.ndarray:
        return self.decode_board_from_state(state)

    def decode_board_from_state(self, state, player):
        """
        Reconstruct board from 4-channel state, using known player sign.
        """
        if player == 1:
            return state[0] - state[1]
        elif player == -1:
            return state[1] - state[0]
        else:
            raise ValueError(f"Invalid player sign: {player}")

    

    def board_to_state(self, board: np.ndarray, player: int) -> np.ndarray:
        agent_plane = (board == player).astype(np.float32)
        opp_plane = (board == -player).astype(np.float32)
        row_plane = np.tile(np.linspace(-1, 1, board.shape[0])[:, None], (1, board.shape[1]))
        col_plane = np.tile(np.linspace(-1, 1, board.shape[1])[None, :], (board.shape[0], 1))
        return np.stack([agent_plane, opp_plane, row_plane, col_plane])

    def act(self, state: np.ndarray, valid_actions, player: int, depth: int, strategy_weights=None):
        board = self.decode_board_from_state(state, player)
    
        # === Tactical Guard ===
        guard_on = (random.random() < self.guard_prob)
        must_play, safe_actions = self._tactical_guard(board, valid_actions, player)
    
        if guard_on:
            if must_play is not None:
                return must_play
            if safe_actions: valid_actions = safe_actions
    
        # === ε-greedy Exploration ===
        if random.random() < self.epsilon:
            if strategy_weights is None:
                strategy_weights = [1, 0, 0, 0, 0, 0, 0, 0]
    
            assert len(strategy_weights) == 8, f"Expected 8 strategy weights, got {len(strategy_weights)}: {strategy_weights}"
            choice = random.choices(range(8), weights=strategy_weights)[0]
    
            if choice == 0:
                action = random.choice(valid_actions)
                return action
            else:
                a = self.lookahead.n_step_lookahead(board, player=player, depth=choice)
                if a not in valid_actions:
                    # action was not safe, it would result in a blunder
                    a = random.choice(valid_actions)
                return a
    
        # === If guard forced move (again, in exploitation phase) ===
        if must_play is not None:
            return must_play
    
        # === DQN Inference ===
        assert state.shape == (4, 6, 7), f"[act] Expected (4,6,7), got {state.shape}"
        x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
    
        with torch.no_grad():
            q_values = self.model(x).cpu().numpy()[0]
    
        # Mask illegal actions
        masked = np.full_like(q_values, -np.inf)
        for a in valid_actions:
            masked[a] = q_values[a]
    
        action = int(np.argmax(masked))
    
        if action not in valid_actions:
            print(f"❌ [act] DQN chose ILLEGAL action: {action} — valid: {valid_actions}")
            raise ValueError(f"[act] DQN chose illegal action: {action} — valid: {valid_actions}")
    
        return action


    def act_with_curriculum(self, *args, **kwargs):
        return self.act(*args, **kwargs)
    

    def remember(self, s, p, a, r, s2, p2, done,
                 add_mirror=True, add_colorswap=True, add_mirror_colorswap=True):
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

    def replay(self, batch_size, mix_1step=0.7):
        if (len(self.memory.bank_1) + len(self.memory.bank_n)) < batch_size:
            return

        (batch_1, batch_n), (idx_1, idx_n), (w_1, w_n) = self.memory.sample_mixed(
            batch_size, mix=mix_1step, beta=self.per_beta
        )
        batch_all = list(batch_1) + list(batch_n)
        n1 = len(batch_1)

        is_w_all = (
            np.concatenate([w_1, w_n]) if (w_1.size or w_n.size) else np.ones((len(batch_all),), dtype=np.float32)
        )
        is_w_all = is_w_all / (is_w_all.max() + 1e-8)
        np.clip(is_w_all, 0.05, 1.0, out=is_w_all)

        states, next_states = [], []
        s_players, ns_players = [], []
        actions, rewards, dones, gammas_n = [], [], [], []

        for t in batch_all:
            if hasattr(t, "reward_n"):
                s, a, rN, nsN, dN, p, n_used = t
                p_next = p if (n_used % 2 == 0) else -p
                states.append(s)
                next_states.append(nsN)
                s_players.append(p)
                ns_players.append(p_next)
                actions.append(a)
                rewards.append(rN * self.reward_scale)
                dones.append(bool(dN))
                gammas_n.append(self.gamma ** n_used)
            else:
                s, a, r, ns, d, p = t
                p_next = -p
                states.append(s)
                next_states.append(ns)
                s_players.append(p)
                ns_players.append(p_next)
                actions.append(a)
                rewards.append(r * self.reward_scale)
                dones.append(bool(d))
                gammas_n.append(self.gamma)

        def pack_many(boards, players):
            planes = [self._agent_planes(b, p) for b, p in zip(boards, players)]
            return torch.tensor(np.asarray(planes), dtype=torch.float32, device=self.device)

        X = pack_many(states, s_players)
        X_next = pack_many(next_states, ns_players)

        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).view(-1, 1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)
        gpow_t = torch.tensor(gammas_n, dtype=torch.float32, device=self.device)
        isw_t = torch.tensor(is_w_all, dtype=torch.float32, device=self.device)
        isw_t = isw_t / (isw_t.mean() + 1e-8)

        q_all = self.model(X)
        q_sa = q_all.gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_a = self.model(X_next).argmax(dim=1, keepdim=True)
            next_q = self.target_model(X_next).gather(1, next_a).squeeze(1)
            target = rewards_t + (~dones_t).float() * gpow_t * next_q

        per_sample_loss = F.smooth_l1_loss(q_sa, target, reduction="none")
        loss = (isw_t * per_sample_loss).mean() if self.per_beta > 0 else per_sample_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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

        for _ in range(n1):
            self.sample_mix_hist.append((0, int(s_players[_])))
        for _ in range(len(batch_n)):
            self.sample_mix_hist.append((1, int(s_players[n1 + _])))

        # --- β schedule update ---
        if self.per_beta < self.per_beta_end:
            self.per_beta = min(self.per_beta_end, self.per_beta + self.per_beta_step)

    @staticmethod
    def flip_transition(state, action, next_state):
        flipped_state = np.flip(state, axis=-1).copy()
        flipped_next_state = np.flip(next_state, axis=-1).copy()
        flipped_action = 6 - action
        return flipped_state, flipped_action, flipped_next_state

    def gave_one_ply_win(self, board, action, player):
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
        for a in valid_actions:
            b1, _ = self._drop_piece(board, a, player)
            if b1 is not None and self._is_win(b1) == player:
                return True, a
        return False, None
