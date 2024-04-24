# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        allStates = self.mdp.getStates()
        
        for _ in range(self.iterations):
            
           tempValues = self.values.copy()
           #print(tempValues)
           for state in allStates:
            
            if self.mdp.isTerminal(state):
                continue
            
            allPossActions = self.mdp.getPossibleActions(state)
            maxVal = float('-inf')
            #print(allPossActions)
            #print(allStates)
            for action in allPossActions:
              transitions = self.mdp.getTransitionStatesAndProbs(state, action)
              values = 0
              #print(transitions)
              for nxtSt, prob in transitions:
                  
                  reward = self.mdp.getReward(state, action, nxtSt)
                  #print(reward)
                  
                  values = prob * (reward + self.discount * self.values[nxtSt]) + values
              
              maxVal = max(maxVal, values)
            
            tempValues[state] = maxVal

           self.values = tempValues
           #print(self.values)
           #print("--")

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Q_Val = 0

        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        for nxtSt, prob in transitions:
            reward = self.mdp.getReward(state, action, nxtSt)
            
            Q_Val = prob * (reward + self.discount * self.values[nxtSt]) + Q_Val
        #print(Q_Val)
        return Q_Val
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        allPossibleActions = self.mdp.getPossibleActions(state)
        #print(allPossibleActions)
        
        if len(allPossibleActions) == 0:
            return None
        
        bestAction = None
        bestVal = float('-inf')
        
        for eachAction in allPossibleActions:
            Qvalue = self.computeQValueFromValues(state, eachAction)
            #print(Qvalue)
            if Qvalue > bestVal:
                bestVal = Qvalue
                bestAction = eachAction
                #print(bestAction, bestVal)
        return bestAction
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        
        allStates = self.mdp.getStates()
        total_states = len(allStates)

        for i in range(self.iterations):
            state = allStates[i % total_states]  # Used ChatGPT to understand and write this line of code 

            if self.mdp.isTerminal(state):
                continue
        
            allPossibleActions = self.mdp.getPossibleActions(state)
            maxVal = float('-inf')
        
            for action in allPossibleActions:
                allTransitions = self.mdp.getTransitionStatesAndProbs(state, action)
                expectedVal = 0

                for nxtSt, prob in allTransitions:
                    reward = self.mdp.getReward(state, action, nxtSt)
                    expectedVal += prob * (reward +self.values[nxtSt]*self.discount)
            
                maxVal = max(maxVal, expectedVal)
        
        # To asynchronously update the value of the state immediately
            self.values[state] = maxVal
                  #print(self.values)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

