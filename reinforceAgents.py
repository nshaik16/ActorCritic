from game import Directions, Agent, Actions
from pacman import GameState
import random,util,time,math
import sys
import numpy as np

def bfsDistance(state, returnCondition, getLegalActions) -> int:
    queue = util.Queue()
    queue.push((state, 0))

    closed = set()

    while not queue.isEmpty():
        currentState, currentDistance = queue.pop()

        if returnCondition(currentState.getPacmanState().getPosition()):
            return currentDistance

        if currentState.getPacmanPosition() not in closed:
            closed.add(currentState.getPacmanPosition())

            for action in getLegalActions(currentState):
                queue.push((currentState.generatePacmanSuccessor(action), currentDistance + 1))

    return sys.maxsize
    raise Exception("Could not find returnCondtion.\n" + str(state))

def softmaxPolicy(action, state, thetaVector, getLegalActions):
    # Implementation Help: https://towardsdatascience.com/policy-based-reinforcement-learning-the-easy-way-8de9a3356083
    numeratorFeatureVector = getFeatureVector(action, state, getLegalActions)

    if len(numeratorFeatureVector) != len(thetaVector):
        print(f"Theta (Legnth {len(thetaVector)}) and Feature Vector (Length {len(numeratorFeatureVector)}) are different lengths.")
        exit()

    hValue = 0
    for i in range(len(thetaVector)):
        hValue += thetaVector[i] * numeratorFeatureVector[i]

    try:
        numerator = math.exp(hValue)
    except:
        if hValue < 0:
            numerator = 0
        else:
            numerator = sys.maxsize

    denominator = 0
    for legalAction in getLegalActions(state):
        featureVector = getFeatureVector(legalAction, state, getLegalActions)
        hValue = 0
        for i in range(len(thetaVector)):
            hValue += thetaVector[i] * featureVector[i]

        try:
            denominator += math.exp(hValue)
        except:
            if hValue < 0:
                denominator += 0
            else:
                denominator += sys.maxsize

    # if numerator == 0:
    #     print("Numerator is zero: \n\tFeature Vec:", numeratorFeatureVector, "\n\tDenominator: ", denominator,"\n\t Theta: ", thetaVector)

    if denominator == 0:
        for legalAction in getLegalActions(state):
            featureVector = getFeatureVector(legalAction, state, getLegalActions)
            print("Denominator is Zero!\n\tAction:", legalAction, "\n\tFeature Vector:", featureVector)
    return (numerator / denominator) if denominator != 0 else 0


def getFeatureVector(action, state, getLegalActions):
    # Features
    # Features inpsired by https://cs229.stanford.edu/proj2017/final-reports/5241109.pdf

    # if type(state) != GameState:
    #     util.raiseNotDefined()

    # Current State Calculations
    pacmanState = state.getPacmanState()

    # Distance to closest food
    minFoodDistance = 0
    if state.getNumFood() > 0:
        try:
            minFoodDistance = bfsDistance(state,
                lambda position: state.hasFood(position[0], position[1]) or position in state.getCapsules(), getLegalActions)
        except Exception as e:
            print("Exception in minFoodDistance: " + str(e));
            minFoodDistance = sys.maxsize

    # Minimum Distance to Active Ghost
    ghostStates = state.getGhostStates()
    activeGhostPositions = []
    scaredGhostPositions = []

    for ghostState in ghostStates:
        if ghostState.scaredTimer > 0:
            scaredGhostPositions.append(ghostState.getPosition())
        else:
            activeGhostPositions.append(ghostState.getPosition())

    nearbyActiveGhostsTwoStep = 0
    nearbyActiveGhostsOneStep = 0
    for ghost in activeGhostPositions:
        distance = bfsDistance(state, lambda position: position == ghost, getLegalActions)
        if distance <= 2:
            nearbyActiveGhostsTwoStep += 1
        if distance <= 1:
            nearbyActiveGhostsOneStep += 1

    # New State Calculations
    newState = state.generatePacmanSuccessor(action)
    newGhostStates = newState.getGhostStates()
    newActiveGhostPositions = []
    newScaredGhostPositions = []

    for ghostState in newGhostStates:
        if ghostState.scaredTimer > 0:
            newScaredGhostPositions.append(ghostState.getPosition())
        else:
            newActiveGhostPositions.append(ghostState.getPosition())

    newNearbyActiveGhostsTwoStep = 0
    newNearbyActiveGhostsOneStep = 0
    for ghost in activeGhostPositions:
        distance = bfsDistance(newState, lambda position: position == ghost, getLegalActions)
        if distance <= 2:
            newNearbyActiveGhostsTwoStep += 1
        if distance <= 1:
            newNearbyActiveGhostsOneStep += 1

    #Next State is terminal
    if newState.isWin():
        return [0,
            newNearbyActiveGhostsTwoStep - nearbyActiveGhostsTwoStep,
            newNearbyActiveGhostsOneStep - nearbyActiveGhostsOneStep,
            abs(newState.getNumFood() - state.getNumFood()),
            0]
    elif newState.isLose():
        return [sys.maxsize,
            newNearbyActiveGhostsTwoStep - nearbyActiveGhostsTwoStep,
            newNearbyActiveGhostsOneStep - nearbyActiveGhostsOneStep,
            abs(newState.getNumFood() - state.getNumFood()),
            sys.maxsize]

    # Distance to closest food
    newMinFoodDistance = 0
    if newState.getNumFood() > 0:
        try:
            newMinFoodDistance = bfsDistance(newState,
                lambda position: newState.hasFood(position[0], position[1]) or position in newState.getCapsules(), getLegalActions)
        except Exception as e:
            print("Exception in newMinFoodDistance: " + str(e));
            newMinFoodDistance = sys.maxsize

    return [newMinFoodDistance - minFoodDistance,
        newNearbyActiveGhostsTwoStep - nearbyActiveGhostsTwoStep,
        newNearbyActiveGhostsOneStep - nearbyActiveGhostsOneStep,
        abs(newState.getNumFood() - state.getNumFood()),
        len(newScaredGhostPositions) - len(scaredGhostPositions)]

class ReinforceAgent(Agent):
    def __init__(self, actionFn = None, gamma=1, alpha=0.2, policy = softmaxPolicy, numTraining=100):
        """
        actionFn: Function which takes a state and returns the list of legal actions

        numTraining - number of training episodes, i.e. no learning after these many episodes
        """

        print(gamma, alpha)
        if actionFn == None:
            actionFn = lambda state: state.getLegalActions()
        self.actionFn = actionFn

        self.numTraining = int(numTraining)
        self.episodesSoFar = 0

        self.policy = policy

        self.theta = [2, 1, 100, 1, 1]
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def update(self):
        # Softmax Derivative: https://math.stackexchange.com/questions/2013050/log-of-softmax-function-derivative
        for t in range(len(self.episodeTriplets) - 1):
            gValue = self.episodeTriplets[t + 1][2]

            # Calculates gradient vector
            gradientVector = [0] * len(self.theta)

            featureVector = getFeatureVector(self.episodeTriplets[t][1], self.episodeTriplets[t][0], self.getLegalActions)
            actionProbabilties = []
            actionFeatureVectors = []

            for action in self.getLegalActions(self.episodeTriplets[t][0]):
                actionProbabilties.append(self.policy(action, self.episodeTriplets[t][0], self.theta, self.getLegalActions))
                actionFeatureVectors.append(getFeatureVector(action, self.episodeTriplets[t][0], self.getLegalActions))

            # x(s,a) - Sum pi(s,*)x(s,*)
            for i in range(len(gradientVector)):
                actionSum = 0
                for j in range(len(actionProbabilties)):
                    actionSum += actionProbabilties[j] * actionFeatureVectors[j][i]

                gradientVector[i] = featureVector[i] - actionSum

            for j in range(len(self.theta)):
                self.theta[j] += self.alpha * pow(self.gamma, t) * gValue * gradientVector[j]
        print("New Theta Values: ", self.theta)

    def getAction(self, state):
        actionProbabilities = []
        probabilitySum = 0
        for action in self.getLegalActions(state):
            probability = self.policy(action, state, self.theta, self.getLegalActions)
            actionProbabilities.append((probability, action))
            probabilitySum += probability

        randomNum = random.random()
        for action in actionProbabilities:
            randomNum -= action[0]
            if randomNum <= 0:
                self.doAction(state, action[1])
                return action[1]

        print("Did not find a legal action!")
        print("Action Probabilities: ", actionProbabilities)

        #Choose action uniformly at random
        return random.choice(self.getLegalActions(state))


    def getPolicy(self, state):
        util.raiseNotDefined()

    def getLegalActions(self,state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return self.actionFn(state)

    def observeTransition(self, state, action, nextState, deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        reward = deltaReward
        self.episodeTriplets.append((state, action, reward))

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.episodeTriplets = []
        self.lastState = None
        self.lastAction = None

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        self.update()

        print(f"Episode {self.episodesSoFar} finished")

        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    ################################
    # Controls needed for Crawler  #
    ################################
    def setEpsilon(self, epsilon):
        self.epsilon = 0

    def setLearningRate(self, alpha):
        self.alpha = 0

    def setDiscount(self, discount):
        self.discount = 0

    def doAction(self,state,action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action

    ###################
    # Pacman Specific #
    ###################
    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

    def registerInitialState(self, state):
        self.startEpisode()
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))

    def final(self, state):
        """
          Called by Pacman game at the terminal state
        """
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()


        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print('Reinforcement Learning Status:')
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print('\tCompleted %d out of %d training episodes' % (
                       self.episodesSoFar,self.numTraining))
                print('\tAverage Rewards over all training: %.2f' % (
                        trainAvg))
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
                print('\tAverage Rewards over testing: %.2f' % testAvg)
            print('\tAverage Rewards for last %d episodes: %.2f'  % (
                    NUM_EPS_UPDATE,windowAvg))
            print('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime))
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))




