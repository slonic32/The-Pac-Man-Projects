# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import backend
import gridworld


import random,util,math
import numpy as np
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter()

        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
      return self.qValues[(state, action)]


    def computeValueFromQValues(self, state):
      legalActions = self.getLegalActions(state)
      if not legalActions:
          return 0.0
      return max(self.getQValue(state, action) for action in legalActions)


    def computeActionFromQValues(self, state):
      legalActions = self.getLegalActions(state)
      if not legalActions:
        return None
      maxQ = float('-inf')
      bestActions = []
      for action in legalActions:
        q = self.getQValue(state, action)
        if q > maxQ:
            maxQ = q
            bestActions = [action]
        elif q == maxQ:
            bestActions.append(action)
      return random.choice(bestActions)


    def getAction(self, state):
      legalActions = self.getLegalActions(state)
      if not legalActions:
        return None
      if util.flipCoin(self.epsilon):
        return random.choice(legalActions)
      return self.computeActionFromQValues(state)


    def update(self, state, action, nextState, reward: float):
      sample = reward + self.discount * self.computeValueFromQValues(nextState)
      self.qValues[(state, action)] = (1 - self.alpha) * self.qValues[(state, action)] + self.alpha * sample


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
      features = self.featExtractor.getFeatures(state, action)
      q_value = 0.0
      for feature, value in features.items():
        q_value += self.weights[feature] * value
      return q_value


    def update(self, state, action, nextState, reward: float):
      correction = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
      features = self.featExtractor.getFeatures(state, action)
      for feature, value in features.items():
        self.weights[feature] += self.alpha * correction * value


    def final(self, state):
      PacmanQAgent.final(self, state)
      if self.episodesSoFar == self.numTraining:
        print("Final learned weights:")
        for feature, weight in self.weights.items():
            print(f"{feature}: {weight:.4f}")

