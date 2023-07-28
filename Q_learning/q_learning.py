from IPython.display import clear_output
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from time import sleep
import random
from os import system


class Taxi:
    def __init__(self, render_mode="ansi"):
        # Generate gymnasium environment variables
        self.env = gym.make("Taxi-v3", render_mode=render_mode).env
        # self.env = TaxiEnv(render_mode=render_mode)
        self.render_mode = render_mode
        self.state, self.info = self.env.reset()
        self.stop_anim = False

    def _init_train_variables(self, train_episodes):
        # clear the system console before training
        system('clear')
        # train environment specific variables
        self.alpha = 0.5
        self.gamma = 0.5
        self.epsilon = 0.0
        self.penalties = 0
        self.reward = 0
        self.done = False
        self.train_episodes = train_episodes # Number of batch to train algorithm
        # total results variables
        self.all_epochs = []
        self.all_penalties = []
        # Generate Q-learning training algorithm variables
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        # Generate reward array for CVaR calculation
        self.reward_array = []

    def _init_test_variables(self, test_episodes, timestamp, fast_testing, final_frame_pause):
        # test environment specific variables
        self.penalties = 0
        self.reward = 0
        self.test_episodes = test_episodes # Number of episode to run in test environment
        self.window = None
        self.done = False
        # total results variables
        self.total_epochs = 0
        self.total_penalties = 0
        self.total_rewards = 0
        # arguments related variables
        self.timestamp = timestamp
        self.fast_testing = fast_testing
        self.final_frame_pause = final_frame_pause

    def _cvar_array(self):
        # prepare numpy array for CVaR metrics
        max_len = 0
        for elem in self.reward_array:
            max_len = len(elem) if len(elem) > max_len else max_len
        for elem in self.reward_array:
            while len(elem) < max_len:
                elem.append(0)
        return np.array(self.reward_array)

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
        ax1.set_ylim(-2 * 100, 100 + 10)
        ax1.plot(self.episode_durations, color="C1", alpha=0.2)
        ax1.plot(self.total_reward, color="C2", alpha=0.2)
        mean_steps = self._moving_average(self.episode_durations, periods=5)
        mean_reward = self._moving_average(self.total_reward, periods=5)
        lines.append(ax1.plot(mean_steps, label="steps", color="C1")[0])
        lines.append(ax1.plot(mean_reward, label="rewards", color="C2")[0])

        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon')
        lines.append(ax2.plot(self.epsilon_vec, label="epsilon", color="C3")[0])
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc=3)

        plt.draw()
        plt.pause(0.001)

    def train(self, train_episodes=25000, training_graph=True):
        self._init_train_variables(train_episodes)
        self.episode_durations = []
        self.epsilon_vec = []
        self.total_reward = []
        for i in range(self.train_episodes):
            ep_reward = 0
            step = 0
            state = self.env.reset()[0]
            self.done = False
            self.penalties, self.reward, = 0, 0
            episode_reward = []
            # self.epsilon = 0.0 + (1.0 - 0.0) * np.exp(-i / 400)
            while not self.done:
                rdm = random.uniform(0, 1)
                if rdm < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])
                next_state, self.reward, self.done, truncated, info = self.env.step(action)
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                new_value = (1 - self.alpha) * old_value + self.alpha * (self.reward + self.gamma * next_max)
                self.q_table[state, action] = new_value
                ep_reward += self.reward
                episode_reward.append(self.reward)
                state = next_state
                step += 1
            self.episode_durations.append(step + 1)
            self.epsilon_vec.append(rdm if rdm < self.epsilon else self.epsilon)
            # self.epsilon_vec.append(self.epsilon)
            self.total_reward.append(ep_reward)
            self.reward_array.append(episode_reward)
            self.plot_durations() if training_graph else None
            if i % 100 == 0:
                system('clear')
                print(f"Train episode: {i} / {self.train_episodes}")
        return self._cvar_array()

    def test(self, test_episodes=1, timestamp=0.2, fast_testing=False, final_frame_pause=0):
        self._init_test_variables(test_episodes, timestamp, fast_testing, final_frame_pause)
        if self.render_mode == 'rgb_array' and not fast_testing:
            fig, self.window = plt.subplots()

        for i in range(self.test_episodes):
            state = self.env.reset()[0]
            episode_reward, epochs, self.penalties, self.reward = 0, 0, 0, 0
            self.done = False
            while not self.done:
                action = np.argmax(self.q_table[state])
                state, self.reward, self.done, truncated, info = self.env.step(action)
                episode_reward += self.reward
                if self.reward == -10:
                    self.penalties += 1
                epochs += 1
                self.display_test(state, action, episode_reward, i)
                if epochs > 50:
                    break
            self.total_penalties += self.penalties
            self.total_rewards += episode_reward
            self.total_epochs += epochs

        if self.render_mode == 'rgb_array' and not self.fast_testing:
            while plt.show():
                if plt.waitforbuttonpress():
                    break
            plt.close()
        print(f"Results after {self.test_episodes} episodes:")
        print(f"Average timesteps per episode: {self.total_epochs / self.test_episodes}")
        print(f"Average penalties per episode: {self.total_penalties / self.test_episodes}")
        print(f"Average rewards per episode: {self.total_rewards / self.test_episodes}")

    def display_test(self, state, action, episode_reward, i):
        system('clear')
        if not self.fast_testing:
            frame = {
                'frame': self.env.render(),
                'state': state,
                'action': action,
                'reward': self.reward,
                'episode_reward': episode_reward,
                'episode': i+1,
            }
            if self.final_frame_pause and self.window and self.done:
                self.__print_frames(frame, 2, self.window)
            elif self.window:
                self.__print_frames(frame, self.timestamp, self.window)
            else:
                self.__print_frames(frame, self.timestamp)
        else:
            print(f"Test episode: {i+1} / {self.test_episodes}")

    def __print_frames(self, frame, timestamp, window=None):
        if self.render_mode == 'rgb_array':
            window.imshow(frame['frame'])
            plt.title("State: %d\nAction: %d\nReward: %d\nEpisode reward: %d\nEpisode: %d / %d" % (
                frame['state'],
                frame['action'],
                frame['reward'],
                frame['episode_reward'],
                frame['episode'],
                self.test_episodes))
            plt.figtext(0.5, 0.05, "Press 'q' to quit", ha="center", fontsize=10)
            window.axis('off')
            plt.pause(timestamp)
            window.cla()
        else:
            clear_output(wait=True)
            print(frame['frame'])
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            print(f"Action: {frame['action']}")
            print(f"Eposide: {frame['episode']}")
            sleep(timestamp)
