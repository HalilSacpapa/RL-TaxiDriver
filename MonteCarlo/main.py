import gym
import treeSearch as treeSearchFile
import node as nodeFile

numberSimulation = 10000

env = gym.make('Taxi-v3', render_mode='ansi')
actionNum = env.action_space.n
env.reset()
env.render()

root = nodeFile.Node(env, actionNum)
isDone = False
totalReward, penalty, epochs = 0, 0, 0

while not isDone:
    mcts = treeSearchFile.MonteCarloTreeSearch(root)
    bestChild = mcts.BestAction(numberSimulation)
    newState, reward, isDone, truncated, info = env.step(bestChild.action)
    totalReward += reward
    if reward == -10:
        penalty += 1
    epochs += 1
    root = bestChild

print('Timesteps :', epochs)
print('Penalty:', penalty)
print('Total Reward:', totalReward)
