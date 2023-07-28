import gymnasium as gym

import os
import time
from itertools import count
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
from tqdm import trange


class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.emb = nn.Embedding(500, 6)
        self.l1 = nn.Linear(6, 50)
        self.l2 = nn.Linear(50, 50)
        self.l3 = nn.Linear(50, 50)
        self.l4 = nn.Linear(50, outputs)

    def forward(self, x):
        x = F.relu(self.l1(self.emb(x)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.rng = np.random.default_rng()

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idx = self.rng.choice(np.arange(len(self.memory)), batch_size, replace=False)
        res = []
        for i in idx:
            res.append(self.memory[i])
        return res

    def __len__(self):
        return len(self.memory)


class QAgent():
    def __init__(self, pt_path=None):
        self.env = gym.make("Taxi-v3").env
        self.model_dir = Path('./model_backup')
        self.model_class = DQN
        # self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = None
        self.rng = np.random.default_rng(42)
        self.episode_durations = []
        self.reward_in_episode = []
        self.epsilon_vec = []
        self.last_step = 0
        self.last_episode = 0

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.is_resume = True if pt_path else False
        self.pt_path = pt_path if self.is_resume else f"{self.model_dir}/pytorch_{int(time.time())}.pt"
        self.checkpoint = torch.load(self.pt_path) if self.is_resume else None

        # Config pytorch
        # training
        self.batch_size = 128
        self.learning_rate = 0.001
        self.loss = self.checkpoint['loss'] if self.is_resume else 'huber'
        self.num_episodes = 10000
        self.train_steps = 1000000
        self.warmup_episode = 0 if self.is_resume else 100
        self.save_freq = 1000
        # optimizer
        self.lr_min = 0.0001
        self.lr_decay = 5000
        # rl
        self.gamma = 0.99
        self.max_steps_per_episode = 100
        self.target_model_update_episodes = 20
        self.max_queue_length = 50000
        # epsilon
        self.max_epsilon = 1
        self.min_epsilon = 0.001
        self.decay_epsilon = 400

    def compile(self):
        n_actions = self.env.action_space.n

        self.model = self.model_class(n_actions).to(self.device)
        if self.is_resume:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.target_model = self.model_class(n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.is_resume:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

    def _get_epsilon(self, episode):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-episode / self.decay_epsilon)

    def _get_action_for_state(self, state):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            f_state = state if type(state) == int else state[0]
            predicted = self.model(torch.tensor([f_state], device=self.device))
            action = predicted.max(1)[1]
        return action.item()

    def _choose_action(self, state, epsilon):
        if self.rng.uniform() < epsilon:
            # Explore
            action = self.env.action_space.sample()
        else:
            # Exploit
            action = self._get_action_for_state(state)
        return action

    def _adjust_learning_rate(self, episode):
        delta = self.learning_rate - self.lr_min
        base = self.lr_min
        rate = self.lr_decay
        lr = base + delta * np.exp(-episode / rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _train_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        # Compute predicted Q values
        predicted_q_value = self.model(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute the expected Q values
        next_state_values= self.target_model(next_state_batch).max(1)[0]
        expected_q_values = (~done_batch * next_state_values * self.gamma) + reward_batch

        # Compute loss
        loss = self.loss(predicted_q_value, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def _update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _remember(self, state, action, next_state, reward, done):
        f_state = state if type(state) == int else state[0]
        self.memory.push(torch.tensor([f_state], device=self.device),
                         torch.tensor([action], device=self.device, dtype=torch.long),
                         torch.tensor([next_state], device=self.device),
                         torch.tensor([reward], device=self.device),
                         torch.tensor([done], device=self.device, dtype=torch.bool))

    def _get_loss(self):
        return F.smooth_l1_loss

    def fit(self):
        try:
            self.loss = self._get_loss()
            self.memory = ReplayMemory(50000)

            self.episode_durations = []
            self.reward_in_episode = []
            self.epsilon_vec = []
            reward_in_episode = 0
            epsilon = 1

            progress_bar = trange(0,
                                  self.num_episodes,
                                  initial=self.last_episode,
                                  total=self.num_episodes)

            for i_episode in progress_bar:
                # Initialize the environment and state
                state = self.env.reset()
                if i_episode >= self.warmup_episode:
                    epsilon = self._get_epsilon(i_episode - self.warmup_episode)

                for step in count():
                    # Select and perform an action
                    action = self._choose_action(state, epsilon)
                    next_state, reward, done, truncated, _ = self.env.step(action)

                    # Store the transition in memory
                    self._remember(state, action, next_state, reward, done)

                    # Perform one step of the optimization (on the target network)
                    if i_episode >= self.warmup_episode:
                        self._train_model()
                        self._adjust_learning_rate(i_episode - self.warmup_episode + 1)
                        done = (step == self.max_steps_per_episode - 1) or done
                    else:
                        done = (step == 5 * self.max_steps_per_episode - 1) or done

                    # Move to the next state
                    state = next_state
                    reward_in_episode += reward

                    if done:
                        self.episode_durations.append(step + 1)
                        self.reward_in_episode.append(reward_in_episode)
                        self.epsilon_vec.append(epsilon)
                        reward_in_episode = 0
                        N = min(10, len(self.episode_durations))
                        progress_bar.set_postfix({
                            "reward": np.mean(self.reward_in_episode[-N:]),
                            "steps": np.mean(self.episode_durations[-N:]),
                            "epsilon": epsilon
                            })
                        self.plot_durations()
                        break

                # Update the target network, copying all weights and biases in DQN
                if i_episode % self.target_model_update_episodes == 0:
                    self._update_target()

                if i_episode % self.save_freq == 0:
                    self.save()

                self.last_episode = i_episode

        except KeyboardInterrupt:
            self.plot_durations()
            print("Training has been interrupted")

    @staticmethod
    def _moving_average(x, periods=5):
        if len(x) < periods:
            return x
        cumsum = np.cumsum(np.insert(x, 0, 0))
        res = (cumsum[periods:] - cumsum[:-periods]) / periods
        return np.hstack([x[:periods-1], res])

    def plot_durations(self):
        lines = []
        plt.ion()
        fig = plt.figure(1, figsize=(15, 7))
        plt.clf()
        ax1 = fig.add_subplot(111)

        plt.title('Training...')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Duration & Rewards')
        ax1.set_ylim(-2 * self.max_steps_per_episode, self.max_steps_per_episode + 10)
        ax1.plot(self.episode_durations, color="C1", alpha=0.2)
        ax1.plot(self.reward_in_episode, color="C2", alpha=0.2)
        mean_steps = self._moving_average(self.episode_durations, periods=5)
        mean_reward = self._moving_average(self.reward_in_episode, periods=5)
        lines.append(ax1.plot(mean_steps, label="steps", color="C1")[0])
        lines.append(ax1.plot(mean_reward, label="rewards", color="C2")[0])

        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon')
        lines.append(ax2.plot(self.epsilon_vec, label="epsilon", color="C3")[0])
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc=3)

        plt.draw()
        plt.pause(0.001)

    def save(self):
        # save format ready for loading
        torch.save({
            # testing ready model save format
            "emb.weight": self.model.emb.weight,
            "l1.weight": self.model.l1.weight,
            "l1.bias": self.model.l1.bias,
            "l2.weight": self.model.l2.weight,
            "l2.bias": self.model.l2.bias,
            "l3.weight": self.model.l3.weight,
            "l3.bias": self.model.l3.bias,
            "l4.weight": self.model.l4.weight,
            "l4.bias": self.model.l4.bias,
            # resume training model save format
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
        }, self.pt_path)

    def play(self, verbose:bool=False, sleep:float=0.2, max_steps:int=100):
        # Play an episode
        try:
            actions_str = ["South", "North", "East", "West", "Pickup", "Dropoff"]

            iteration = 0
            state = self.env.reset()  # reset environment to a new, random state
            self.env.render()
            if verbose:
                print(f"Iter: {iteration} - Action: *** - Reward ***")
            # time.sleep(sleep)
            done = False

            while not done:
                action = self._get_action_for_state(state)
                iteration += 1
                state, reward, done, info = self.env.step(action)
                # display.clear_output(wait=True)
                self.env.render()
                if verbose:
                    print(f"Iter: {iteration} - Action: {action}({actions_str[action]}) - Reward {reward}")
                # time.sleep(sleep)
                if iteration == max_steps:
                    print("cannot converge :(")
                    break
        except KeyboardInterrupt:
            pass

    def evaluate(self, max_steps:int=100):
        try:
            total_steps, total_penalties = 0, 0
            episodes = 100

            for episode in trange(episodes):
                state = self.env.reset()  # reset environment to a new, random state
                nb_steps, penalties, reward = 0, 0, 0

                done = False

                while not done:
                    action = self._get_action_for_state(state)
                    state, reward, done, info = self.env.step(action)

                    if reward == -10:
                        penalties += 1

                    nb_steps += 1
                    if nb_steps == max_steps:
                        done = True

                total_penalties += penalties
                total_steps += nb_steps

            print(f"Results after {episodes} episodes:")
            print(f"Average timesteps per episode: {total_steps / episodes}")
            print(f"Average penalties per episode: {total_penalties / episodes}")
        except KeyboardInterrupt:
            pass
