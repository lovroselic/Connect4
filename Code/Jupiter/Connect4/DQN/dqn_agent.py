# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 18:20:49 2025

@author: Uporabnik
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from DQN.DQN_replay_memory import ReplayMemory
from DQN.dqn_model import DQN
from C4.connect4_lookahead import Connect4Lookahead  # you must define/import this

class DQNAgent:
    def __init__(self, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.update_target_model()

        self.memory = ReplayMemory()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.lookahead = Connect4Lookahead()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, valid_actions, player, depth, strategy_weights=None):
        board = np.array(state)

        if random.random() < self.epsilon:
            if strategy_weights is None:
                strategy_weights = [1, 0, 0, 0, 0, 0]

            choice = random.choices([0, 1, 2, 3, 4, 5], weights=strategy_weights)[0]

            if choice == 0:
                return random.choice(valid_actions)
            elif choice == 1:
                return self.lookahead.n_step_lookahead(board, player=player, depth=1)
            elif choice == 2:
                return self.lookahead.n_step_lookahead(board, player=player, depth=3)
            elif choice == 3:
                return self.lookahead.n_step_lookahead(board, player=player, depth=5)
            elif choice == 4:
                return self.lookahead.n_step_lookahead(board, player=player, depth=max(1, depth - 1))
            elif choice == 5:
                return self.lookahead.n_step_lookahead(board, player=player, depth=depth)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]

        masked_q = np.full_like(q_values, -np.inf)
        for a in valid_actions:
            masked_q[a] = q_values[a]

        return int(np.argmax(masked_q))

    def act_with_curriculum(self, state, valid_actions, player, depth, strategy_weights=None):
        return self.act(state, valid_actions, player, depth, strategy_weights)

    def remember(self, s, a, r, s2, done):
        self.memory.push(s, a, r, s2, done)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        q_values = self.model(states)

        next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
        next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze()

        q_targets = q_values.clone()
        for i in range(len(batch)):
            q_targets[i][actions[i]] = rewards[i] + (0 if dones[i] else self.gamma * next_q_values[i])

        loss = self.loss_fn(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
