from os import system
from time import sleep

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import gymnasium as gym
import torch


class TestModel:
    def __init__(self, model_class, model_file, render_mode="ansi"):
        # Generate gymnasium environment variables
        self.env = gym.make("Taxi-v3", render_mode=render_mode).env
        self.render_mode = render_mode
        self.state, self.info = self.env.reset()
        self.stop_anim = False

        # define model related variables
        self.model_class = model_class
        self.path = model_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = self.env.action_space.n

        self.model = model_class(self.n_actions).to(self.device)
        checkpoint = torch.load(self.path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

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

    def _get_action_for_state(self, state):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            f_state = state if type(state) == int else state[0]
            predicted = self.model(torch.tensor([f_state], device=self.device))
            action = predicted.max(1)[1]
        return action.item()

    def test(self, test_episodes=1, timestamp=0.2, fast_testing=False, final_frame_pause=0):
        self._init_test_variables(test_episodes, timestamp, fast_testing, final_frame_pause)
        if self.render_mode == 'rgb_array' and not fast_testing:
            fig, self.window = plt.subplots()

        for i in range(self.test_episodes):
            state = self.env.reset()[0]
            episode_reward, epochs, self.penalties, self.reward = 0, 0, 0, 0
            self.done = False
            while not self.done:
                action = self._get_action_for_state(state)
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