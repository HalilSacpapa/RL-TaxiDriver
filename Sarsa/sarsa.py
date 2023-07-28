#from IPython.display import clear_output, display
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from time import sleep
import random
from os import system

# import local python library from ../Gymnasium/gymnasium/
# from sys import path
# path.insert(0, '../../Gymnasium/gymnasium/envs/toy_text')
# from taxi import TaxiEnv

class Taxi:
    def __init__(self, render_mode="ansi"):
         # Generate gymnasium environment variables
        self.env = gym.make("Taxi-v3", render_mode=render_mode).env
        # self.env = TaxiEnv(render_mode=render_mode)
        self.render_mode = render_mode
        self.state, self.info = self.env.reset()
        self.stop_anim = False
    
    def _init_train_variables(self, n_episodes):
        # clear the system console before training
        system('clear')
        # train environment specific variables
        self.alpha = 0.1
        self.gamma = 0.95
        self.start_epsilon = 1.0
        self.min_epsilon = 0.0
        self.decay_rate = 0.00001
        self.done = False
        self.n_max_steps = 100
        self.n_episodes = n_episodes # Number of batch to train algorithm
        # total results variables
        self.all_epochs = []
        self.all_penalties = []
        # Initialize Q-table
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
        self.n_max_steps = 100
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
    
    # This is our acting policy (epsilon-greedy), which selects an action for exploration/exploitation during training
    def epsilon_greedy(self, Qtable, state, epsilon, env):
        # Generate a random number and compare to epsilon, if lower then explore, otherwise exploit
        randnum = np.random.uniform(0, 1)
        if randnum < epsilon:
            action = env.action_space.sample()    # explore
        else:
            action = np.argmax(Qtable[state, :])  # exploit
        return action
    
    # This function is to update the Qtable.
    # It is also based on epsilon-greedy approach because the next_action is decided by epsilon-greedy policy
    def update_Q(self, Qtable, state, action, reward, next_state, next_action, alpha, gamma):
        # 𝑄(𝑆𝑡,𝐴𝑡)=𝑄(𝑆𝑡,𝐴𝑡)+𝛼[𝑅𝑡+1+𝛾𝑄(𝑆𝑡+1,𝐴𝑡+1)−𝑄(𝑆𝑡,𝐴𝑡)]
        Qtable[state][action] = Qtable[state][action] + alpha * (reward + gamma * (Qtable[next_state][next_action]) - Qtable[state][action])
        return Qtable   

    # This function (greedy) will return the action from Qtable when we do evaluation
    def eval_greedy(self, Qtable, state):
        action = np.argmax(Qtable[state, :])
        return action

    def train(self, train_episodes=25000):
        self._init_train_variables(train_episodes)
        for episode in range(train_episodes):
        
            # Reset the environment at the start of each episode
            state, info = self.env.reset()
            t = 0
            self.done = False
            episode_reward = []

            # Calculate epsilon value based on decay rate
            epsilon = max(self.min_epsilon, (self.start_epsilon - self.min_epsilon)*np.exp(-self.decay_rate*episode))
            
            # Choose an action using previously defined epsilon-greedy policy
            action = self.epsilon_greedy(self.q_table, state, epsilon, self.env)
            
            for t in range(self.n_max_steps):
                
                
                # Perform the action in the environment, get reward and next state
                next_state, self.reward, self.done, _, info = self.env.step(action)
                
                # Choose next action
                next_action=self.epsilon_greedy(self.q_table, next_state, epsilon, self.env)
                
                # Update Q-table
                self.q_table = self.update_Q(self.q_table, state, action, self.reward, next_state, next_action, self.alpha, self.gamma)
                
                # Update current state 
                
                state = next_state
                action = next_action
                self.reward_array.append(episode_reward)
            
                # Finish the episode when done=True, i.e., reached the goal or fallen into a hole
                if self.done:
                    break
            episode_reward.append(self.reward)
        return self._cvar_array()
    
    def save_q_table(self, Qtable):
        np.save("qtable",Qtable)
    
    def test(self, test_episodes=1, timestamp=0.2, fast_testing=False, final_frame_pause=0, path: str = "qtable.npy"):
        self._init_test_variables(test_episodes, timestamp, fast_testing, final_frame_pause)
    
        ep_rewards=[]
        # test environment variables
        total_epochs = 0
        q_table = np.load(path)
    
        # Evaluate for each episode
        for episode in range(self.test_episodes):
        
            # Reset the environment at the start of each episode
            state, info = self.env.reset()
             # Initialize an empty list to store rewards for each episode
            episode_reward=[]
            t = 0
            done = False
            tot_episode_reward = 0
            epochs = 0
        
            for t in range(self.n_max_steps):
            
                # Use greedy policy to evaluate
                action = self.eval_greedy(q_table, state)

                # Pass action into step function
                next_state, reward, done, _, info = self.env.step(action)

                # Sum episode rewards
                tot_episode_reward += reward
                episode_reward.append(reward)

                # Update current state 
                state = next_state

                if reward == -10:
                    self.penalties += 1
                
                epochs += 1

                self.display_test(state, action, episode_reward, episode)

                #if epochs > self.n_max_steps:
                #    break

                # Finish the episode when done=True, i.e., reached the goal or fallen into a hole
                if done:
                    break

            self.reward_array.append(episode_reward)    
            total_epochs += epochs
                
            ep_rewards.append(tot_episode_reward)
        
        mean_reward = np.mean(ep_rewards)
        std_reward = np.std(ep_rewards)
        print(f"Average timesteps per episode: {total_epochs / self.test_episodes}")
        print(f"Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Min Reward = {min(ep_rewards):.1f} and Max Reward {max(ep_rewards):.1f}")


        return self._cvar_array()
    
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

    def __print_distribution_plot(self, episode_rewards):
        # Show the distribution of rewards obtained from evaluation
        plt.figure(figsize=(9,6), dpi=200)
        plt.title(label='Rewards distribution from evaluation', loc='center')
        plt.hist(episode_rewards, bins=25, color='#00000f')
        plt.show()

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
    