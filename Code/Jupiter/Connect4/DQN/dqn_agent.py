# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 18:20:49 2025

@author: Lovro
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from DQN.DQN_replay_memory import ReplayMemory
from DQN.dqn_model import DQN
from C4.connect4_lookahead import Connect4Lookahead  

class DQNAgent:
    def __init__(self, lr=0.001, gamma=0.97, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.update_target_model()
        self.memory = ReplayMemory()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lookahead = Connect4Lookahead()
        self.reward_scale = 0.01
        
   
    @staticmethod
    def _agent_planes(board: np.ndarray, agent_side: int):
        """
        board: (6,7) in {-1,0,+1} env encoding
        agent_side: +1 if agent uses +1 stones on board, else -1 (if you ever flip roles)
        Returns (2,6,7) float32 planes in {0,1}.
        """
        agent_plane = (board == agent_side).astype(np.float32)
        opp_plane   = (board == -agent_side).astype(np.float32)
        return np.stack([agent_plane, opp_plane], axis=0)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, valid_actions, player, depth, strategy_weights=None):
        """
        state: (6,7) env board in {-1,0,+1}
        player: +1 or -1 (the side to move NOW in env). For exploit we use it
                only to build agent-centric planes; for explore, lookahead needs it.
        """
        board = np.array(state)

        # Explore (ε): can use curriculum/lookahead
        if random.random() < self.epsilon:
            if strategy_weights is None:
                strategy_weights = [1, 0, 0, 0, 0, 0, 0, 0]
                
            choice = random.choices(range(8), weights=strategy_weights)[0]
            if choice == 0:
                return random.choice(valid_actions)
            else:
                return self.lookahead.n_step_lookahead(board, player=player, depth=choice)

        # Exploit (1-ε): agent-centric planes (always "agent = plane 0")
        x = torch.tensor(self._agent_planes(board, agent_side=player),
                         dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model(x).cpu().numpy()[0]
            
        masked = np.full_like(q_values, -np.inf)
        for a in valid_actions:
            masked[a] = q_values[a]
        return int(np.argmax(masked))

    def act_with_curriculum(self, state, valid_actions, player, depth, strategy_weights=None):
        return self.act(state, valid_actions, player, depth, strategy_weights)

    def remember(self, s, p, a, r, s2, p2, done):
       scaled = r * self.reward_scale
       self.memory.push(s,  p,  a, scaled, s2,  p2,  done)

       # horizontal flip augmentation
       fs  = np.flip(s,  axis=-1).copy()
       fs2 = np.flip(s2, axis=-1).copy()
       fa  = 6 - a
       self.memory.push(fs, p, fa, scaled, fs2, p2, done)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        # unpack
        states, s_players, actions, rewards, next_states, ns_players, dones = zip(*batch)

        # pack to 2-plane tensors (agent-centric for each transition)
        def pack_many(boards, players):
            planes = [self._agent_planes(b, p) for b, p in zip(boards, players)]
            return torch.tensor(np.array(planes), dtype=torch.float32, device=self.device)

        X      = pack_many(states,      s_players)   # (B,2,6,7)
        X_next = pack_many(next_states, ns_players)

        actions = torch.tensor(actions, dtype=torch.long,    device=self.device).view(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).view(-1)
        dones   = torch.tensor(dones,   dtype=torch.bool,    device=self.device).view(-1)

        q_all = self.model(X)                           # (B,7)
        q_sa  = q_all.gather(1, actions).squeeze(1)     # (B,)

        with torch.no_grad():
            next_a = self.model(X_next).argmax(dim=1, keepdim=True)  # (B,1)
            next_q = self.target_model(X_next).gather(1, next_a).squeeze(1)
            target = rewards + (~dones).float() * self.gamma * next_q

        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
     
    @staticmethod
    def flip_transition(state, action, next_state):
        flipped_state = np.flip(state, axis=-1).copy()
        flipped_next_state = np.flip(next_state, axis=-1).copy()
        flipped_action = 6 - action
        return flipped_state, flipped_action, flipped_next_state 
    



