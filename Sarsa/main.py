
import sarsa as Taxi
import numpy as np
import time

from os import system


def cvar_metrics(reward_array):
    # Percentile compute about trust level relative to CVaR metrics
    #! CVaR\power-index{ℓ|α}(F) = \frac{max|w∈∆\index{n},w≼\frac{1|αn}}\sum{n|i=1}w\index{i}ℓ(F(x\index{i}),y\index{i})
    sorted_reward_array = np.sort(reward_array)
    confidence_level = 0.95
    percentile_index = int((1 - confidence_level) * len(reward_array))
    percentile_value = sorted_reward_array[percentile_index]
    cvar = reward_array[reward_array <= percentile_value].mean()
    print(f"CVaR à {confidence_level} de niveau de confiance : {cvar}")

# setup object and variables
start = time.time()
taxi = Taxi.Taxi('rgb_array')

reward_array = taxi.train(train_episodes=10000)
train_time = round(time.time() - start, 2)
taxi.test(test_episodes=1000, # number of test episode to execute
          timestamp=0.1, # time between each frame
          fast_testing=True, # display graphical interface or print only test informations
          final_frame_pause=1) # time to wait after the last frame of each episode

# print execution time and metrics
print(f"Train execution time: {train_time}s\n\n")