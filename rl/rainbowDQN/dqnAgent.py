import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
from typing import Deque, Dict, Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

from .replayBuffer import PrioritizedReplayBuffer, ReplayBuffer
from .network import Network


class DQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self, 
        env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        seed: int,
        lr: float = 1e-4,
        #lr = 1e-5,
        gamma: float = 0.97,
        tau: float = 0.5,

        # exploration
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        min_epsilon = 0.6,
        hard_exploration_steps = 0,
        #exploration_steps = 1000,
        exploration_steps = 0,

        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,

        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 120.0,
        atom_size: int = 121,

        # N-step Learning
        n_step: int = 3,
        n_step_alpha: float = 1.0,

        # Logs
        log_dir = "logs",
        file_path = None,
        name = "",
        ckpt_save_freq = None

    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        obs_dim = env.obs_space
        action_dim = env.action_space
        
        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.seed = seed
        self.gamma = gamma
        self.tau = tau
        # NoisyNet: All attributes related to epsilon are removed
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # exploration
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.exploration_steps = exploration_steps
        self.hard_exploration_steps = hard_exploration_steps
        self.min_epsilon = min_epsilon
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma
        )
        
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        self.n_step_alpha = n_step_alpha
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )
            
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

        # logging
        self.name = name
        self.writer = SummaryWriter(log_dir + os.sep + name + datetime.now().strftime("%Y%m%d-%H%M%S") )
        self.ckpt_save_freq = ckpt_save_freq

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""

        if np.random.uniform() < self.epsilon and not self.is_test:
            selected_action = 0
        else:
            # NoisyNet: no epsilon greedy action selection
            q_values = self.dqn(
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            )

            if self.is_test:
                print(q_values)

            selected_action = q_values.argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        #next_state, reward, terminated, truncated, _ = self.env.step(action)
        #done = terminated or truncated
        
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += self.n_step_alpha * elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def decay_epsilon(self, idx):
        if idx <= self.hard_exploration_steps:
            self.epsilon = 1.0
            return
        if idx >= self.exploration_steps - 1:
            self.epsilon = 0.0
            return
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

        
    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        update_cnt = 0
        losses = 0
        scores = []
        score = 0
        done_cnt = 0
        prev_done = 0
        success = False
        success_cnt = 0

        for frame_idx in tqdm(range(1, num_frames + 1)):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
            
            # NoisyNet: removed decrease of epsilon
            
            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            self.decay_epsilon(frame_idx)

            # if episode ends
            if done:
                done_cnt += 1
                episode_length = frame_idx - prev_done

                self.writer.add_scalar("Score", score, done_cnt)
                self.writer.add_scalar("Length", episode_length, done_cnt)

                #state, _ = self.env.reset(seed=self.seed)
                state = self.env.reset()
                scores.append(score)
                prev_done = frame_idx

                success = score >= 10.0
                self.writer.add_scalar("Success", 1 if success else 0, done_cnt)
                if success:
                    self.writer.add_scalar("Success Length", episode_length, success_cnt)
                    self.writer.add_scalar("Success Score", score, success_cnt)
                    success_cnt += 1

            # if training is ready
            if len(self.memory) >= self.batch_size and done:
                print("updating model")
                update_num = min(100, len(self.memory) // self.batch_size // 2)
                for _ in tqdm(range(update_num)):
                    loss = self.update_model()
                    losses += loss
                    update_cnt += 1
                
                    # if hard update is needed
                    if update_cnt % self.target_update == 0:
                        #self._target_hard_update()
                        self._target_soft_update()
                avg_loss = losses / episode_length
                print("Episode", done_cnt, " frame_idx:", frame_idx, "length:", episode_length, "loss:", avg_loss, "score:", score, " success:", success)
                self.writer.add_scalar("Loss", avg_loss, done_cnt)
            
            if done:
                losses = 0
                score = 0
                success = False

            # plotting
            if frame_idx % plotting_interval == 0:
                # self._plot(frame_idx, scores, losses)
                pass


            if self.ckpt_save_freq and frame_idx % self.ckpt_save_freq == 0:
                print("Saving Checkpoint")
                self.save("dynamic_act_speed_rainbow/" + "ckpt_" + repr(frame_idx) + "_" + self.name)
   
                
        self.env.close()
                
    def test(self, num_tests, test_max_length = 500, video_folder: str = None) -> None:
        """Test the agent."""
        self.is_test = True
        
        # for recording a video
        naive_env = self.env
        #self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        
        done_cnt = 0

        while done_cnt < num_tests:
            #state, _ = self.env.reset(seed=self.seed)
            state = self.env.reset()
            done = False
            score = 0
            step = 0
            
            while not done and step < test_max_length:
                #self.env.render_image()

                action = self.select_action(state)
                print(action)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward
                step += 1

                #self.env.render_image()
            
            print("score: ", score, " length:", step)
            self.env.close()
            
            # reset
            self.env = naive_env

            done_cnt += 1

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _target_soft_update(self):
        """Soft update: target <- local."""
        for target_param, local_param in zip(
            self.dqn_target.parameters(), self.dqn.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
                
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float],
    ):
        """Plot the training progresses."""
        #clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.show()

    def save(self, filename, directory: str = "/scr2/tonyzhao/train_logs"):
        """Save trained model."""
        torch.save(self.dqn.state_dict(), "%s/%s_dqn.pth" % (directory, filename))

    def load(self, filename: str, directory: str = "/scr2/tonyzhao/train_logs"):
        """Load trained model."""
        self.dqn.load_state_dict(torch.load("%s/%s_dqn.pth" % (directory, filename)))
        self._target_hard_update()
