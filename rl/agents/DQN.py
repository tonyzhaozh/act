import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from collections import deque


class DQNAgent:
    def __init__(self, env, hidden_dim=256, depth=2, learning_rate=0.01, gamma=0.999, epsilon=1.0, lambda_=0.8,
                 epsilon_decay=0.999, epsilon_min=0.01, replay_buffer_size=10000, update_freq = 10,
                 batch_size=32):
        self.env = env
        self.num_states = env.obs_space
        self.num_actions = env.action_space
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.lambda_ = lambda_

        self.device = torch.device(" if torch.cuda.is_available() else "cpu")

        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.update_freq = update_freq

        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.update_target_network()

    def _build_network(self):
        # MLP with depth hidden layers
        layers = [nn.Linear(self.num_states, self.hidden_dim), nn.ReLU()]
        for _ in range(self.depth):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.num_actions))
        model = nn.Sequential(*layers)
        return model

    def update_target_network(self):
        # soft update with lambda
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.lambda_ * param.data + (1. - self.lambda_) * target_param.data)

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        return action

    def get_best_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        return action

    def remember(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        states_tensor = torch.tensor(states, dtype=torch.float).unsqueeze(1).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float).unsqueeze(1).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float).unsqueeze(1).to(self.device)

        q_values = self.q_network(states_tensor).gather(1, actions_tensor)
        next_q_values = self.target_network(next_states_tensor).max(1)[0].unsqueeze(1)
        target = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

        loss = self.loss_fn(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def train(self, num_episodes, max_steps_per_episode):
        print("Start Training")
        for episode in tqdm(range(num_episodes)):
            state = self.env.reset()
            total_reward = 0

            for step in range(max_steps_per_episode):
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.remember(state, action, next_state, reward, done)
                self.replay()
                self.decay_epsilon()

                state = next_state
                total_reward += reward

                if done:
                    break

            if episode % self.update_freq == 0:
                self.update_target_network()

            if episode % 10 == 0:
                total_reward_formatted = "{:.2f}".format(total_reward)
                print(f"Episode {episode + 1}: Total reward = {total_reward_formatted}")

        print("Training finished.")
