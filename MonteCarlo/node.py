import numpy
from copy import copy
import numpy as np

class Node:

    def __init__(self, env, actionNum, parent = None):
        self.actionNum = actionNum
        self.state = env
        self.parent = parent
        self.children = []
        self.untriedActions = [action for action in range(actionNum)]
        self.visitingTimes = 0
        self.q = 0
        self.isDone = False
        self.observation = None
        self.reward = 0
        self.action = None

    def IsFullyExpended(self):
        return len(self.untriedActions) == 0

    def IsTerminalNode(self):
        return self.isDone

    def ComputeMeanValue(self):
        if self.visitingTimes == 0:
            return 0
        return self.q / self.visitingTimes

    def ComputeScore(self, scale = 10, max_score = 10e100):
        if self.visitingTimes == 0:
            return max_score
        parentVisitingTimes = self.parent.visitingTimes
        ucb = 2 * np.sqrt(np.log(parentVisitingTimes) / self.visitingTimes)
        result = self.ComputeMeanValue() + scale * ucb
        return result

    def BestChild(self):
        scores = [child.ComputeScore() for child in self.children]
        childIndex = np.argmax(scores)
        return self.children[childIndex]

    def Expand(self):
        action = self.untriedActions.pop()
        nextState = copy(self.state)
        self.observation, self.reward, self.truncated, self.isDone,_ = nextState.step(action)
        childNode = Node(nextState, self.actionNum, parent = self)
        childNode.action = action
        self.children.append(childNode)
        return childNode
  
    def RolloutPolicy(self, state):
        return state.action_space.sample()
  
    def Rollout(self, t_max = 10**8):
        state = copy(self.state)
        rolloutReturn = 0
        gamma = 0.6
        done = False
        while not done:
            action = self.RolloutPolicy(state)
            obs, reward, truncated, done, _ = state.step(action)
            rolloutReturn += gamma * reward
            if done:
                break
        return rolloutReturn

    def BackPropagate(self, childValue):
        nodeValue = self.reward + childValue
        self.q += nodeValue
        self.visitingTimes += 1
        if self.parent:
            return self.parent.BackPropagate(nodeValue)
