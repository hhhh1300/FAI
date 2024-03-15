# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = -20 * successorGameState.getNumFood()
        closedFood = [manhattanDistance(newPos, food) for food in newFood.asList()]
        # closedGhost = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        if(len(closedFood)):
            if min(closedFood) != 1:
                score += 10/min(closedFood)
            else:
                score = -20 * (successorGameState.getNumFood() - 1)
        for ghost in newGhostStates:
            if manhattanDistance(newPos, ghost.getPosition()) == 0:
                score = -1e9
                break
            elif ghost.scaredTimer > 0:
                score += ghost.scaredTimer
            else:
                score -= 40/manhattanDistance(newPos, ghost.getPosition())
        # print(newScaredTimes)
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def value(self, gameState: GameState, agent_type, depth, agent_index):
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return (self.evaluationFunction(gameState), -1)
        if agent_type == 'MAX':
            return self.max_value(gameState, depth, agent_index)
        elif agent_type == 'MINI':
            return self.mini_value(gameState, depth, agent_index)
        
    def max_value(self, gameState: GameState, depth, agent_index):
        v = (-1e9, -1)
        actions = gameState.getLegalActions(agent_index)
        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            new_value = self.value(successor, 'MINI', depth, agent_index+1)
            if new_value[0] > v[0]:
                v = (new_value[0], action)
        return v

    def mini_value(self, gameState: GameState, depth, agent_index):
        v = (1e9, -1)
        num_agent = gameState.getNumAgents()
        actions = gameState.getLegalActions(agent_index)    
        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            new_value = 0
            if agent_index == num_agent-1:
                new_value = self.value(successor, 'MAX', depth+1, 0)
            else:
                new_value = self.value(successor, 'MINI', depth, agent_index+1)
            if new_value[0] < v[0]:
                v = (new_value[0], action)
        return v


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 'MAX', 1, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def value(self, gameState: GameState, agent_type, depth, agent_index, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return (self.evaluationFunction(gameState), -1)
        if agent_type == 'MAX':
            return self.max_value(gameState, depth, agent_index, alpha, beta)
        elif agent_type == 'MINI':
            return self.mini_value(gameState, depth, agent_index, alpha, beta)
        
    def max_value(self, gameState: GameState, depth, agent_index, alpha, beta):
        v = (-1e9, -1)
        actions = gameState.getLegalActions(agent_index)
        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            new_value = self.value(successor, 'MINI', depth, agent_index+1, alpha, beta)
            if new_value[0] > v[0]:
                v = (new_value[0], action)
            if v[0] > beta:
                return v
            alpha = max(v[0], alpha)
        return v

    def mini_value(self, gameState: GameState, depth, agent_index, alpha, beta):
        v = (1e9, -1)
        num_agent = gameState.getNumAgents()
        actions = gameState.getLegalActions(agent_index)    
        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            new_value = 0
            if agent_index == num_agent-1:
                new_value = self.value(successor, 'MAX', depth+1, 0, alpha, beta)
            else:
                new_value = self.value(successor, 'MINI', depth, agent_index+1, alpha, beta)
            if new_value[0] < v[0]:
                v = (new_value[0], action)
            if v[0] < alpha:
                return v
            beta = min(beta, v[0])
        return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 'MAX', 1, 0, -1e9, 1e9)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def value(self, gameState: GameState, agent_type, depth, agent_index):
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return (self.evaluationFunction(gameState), -1)
        if agent_type == 'MAX':
            return self.max_value(gameState, depth, agent_index)
        elif agent_type == 'EXP':
            return self.mini_value(gameState, depth, agent_index)
        
    def max_value(self, gameState: GameState, depth, agent_index):
        v = (-1e9, -1)
        actions = gameState.getLegalActions(agent_index)
        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            new_value = self.value(successor, 'EXP', depth, agent_index+1)
            if new_value[0] > v[0]:
                v = (new_value[0], action)
        return v

    def mini_value(self, gameState: GameState, depth, agent_index):
        v = (0, -1)
        num_agent = gameState.getNumAgents()
        actions = gameState.getLegalActions(agent_index)    
        p = 1/len(actions)
        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            new_value = 0
            if agent_index == num_agent-1:
                new_value = self.value(successor, 'MAX', depth+1, 0)
            else:
                new_value = self.value(successor, 'EXP', depth, agent_index+1)
            v = (v[0]+p * new_value[0], action)
        return v
    
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 'MAX', 1, 0)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    if currentGameState.isLose():
        return -1e9
    elif currentGameState.isWin():
        return 1e9
    # score = -20 * currentGameState.getNumFood()
    score = 0
    closestFood = [manhattanDistance(newPos, food) for food in newFood.asList()]
    # sorted(closestFood)
    # closedGhost = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
    if(len(closestFood)):
        # for i in range(3):
        #     if len(closestFood) > i:
        #         if closestFood[i] != 0:
        #             score += 10/closestFood[i]
        #         else:
        #             score += 10
        if min(closestFood) != 0:
            score += 10 / min(closestFood)
        else:
            score += 10
            # score = -20 * (currentGameState.getNumFood() - 1)
    else:
        return 1e9
    for ghost in newGhostStates:
        distence = manhattanDistance(newPos, ghost.getPosition())
        if distence >= 5:
            continue
        elif distence > 0:
            if ghost.scaredTimer > 0:
                score += 100/distence
            else:
                score -= 10/distence
        elif distence == 0:
            if ghost.scaredTimer > 0:
                score += 1000
            else:
                score = -1e9
                break
    score += currentGameState.getScore()
    # print(score)
    return score

# Abbreviation
better = betterEvaluationFunction
