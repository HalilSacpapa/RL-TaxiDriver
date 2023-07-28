class MonteCarloTreeSearch(object):
    def __init__(self, node):
        self.root = node

    def BestAction(self, simulationsNumber):
        for _ in range(0, simulationsNumber):
            v = self.TreePolicy()
            reward = v.Rollout()
            v.BackPropagate(reward)
        return self.root.BestChild()

    def TreePolicy(self):
        currentNode = self.root
        while not currentNode.IsTerminalNode():
            if not currentNode.IsFullyExpended():
                return currentNode.Expand()
            else:
                currentNode = currentNode.BestChild()
        return currentNode