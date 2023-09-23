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
import sys
class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.
        getAction chooses among the best options according to the evaluation function.
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        # print (successorGameState)
        # print(newPos)
        # print(newFood)
        # print(newGhostStates[0])
        #print(newScaredTimes)


        #Adjast the score based on the distance between pacman and foods
        def penalize_wrt_food_distance():
          penalize = 0
          newFood = successorGameState.getFood().asList()
          for food in newFood:
            distance = util.manhattanDistance(food, newPos)
            if distance <=2:
              penalize+=1
            elif distance >2 and distance <4:
              penalize+=0.8
            elif distance >=4 and distance <10:
              penalize +=0.6
            else:
              penalize+=0.2
          return (penalize)

        #Adjast the score based on the distance between pacman and ghosts
        def penalize_wrt_ghost_distance():
          penalize = 0
          ghost_positions = currentGameState.getGhostPositions()
          for new_ghost_pos in ghost_positions:
            distance = util.manhattanDistance(new_ghost_pos, newPos)
            if distance < 2:
              penalize -=10
            elif distance > 2 and distance <=5:
              penalize -=7
            elif distance > 5 and distance <=10:
              penalize -=5
            else:
              penalize +=1
          return (penalize)

        #Adjast the score based on the ghosts directions
        def penalize_wrt_ghost_direction():
          penalize = 0 
          reverse = {
                    "North": "South",
                    "South": "North",
                    "East": "West",
                    "West": "East",
                    "Stop": "Stop"
                    }
          pacman_dir = currentGameState.getPacmanState().getDirection()
          ghost_state = currentGameState.getGhostStates()

          if len(newGhostStates)==1:
           
            ghost_dir = ghost_state[0].getDirection()
            
            ghost_pos = ghost_state[0].getPosition()
            if pacman_dir==reverse[ghost_dir]:
              if pacman_dir == "North" and newPos[1] < ghost_pos[1]:
                penalize -=10
              elif pacman_dir =="South" and newPos[1] > ghost_pos[1]:
                penalize -=10
              elif pacman_dir == "East" and newPos[0] < ghost_pos[0]:
                penalize -=10
              elif pacman_dir == "West" and newPos[0] > ghost_pos[0]:
                penalize -=10

          if len(newGhostStates)==2:
            ghost_1_dir = ghost_state[0].getDirection()
            ghost_2_dir = ghost_state[1].getDirection()
            ghost_1_pos = ghost_state[0].getPosition()
            ghost_2_pos = ghost_state[1].getPosition()
            if pacman_dir==reverse[ghost_1_dir]:
              if pacman_dir == "North" and newPos[1] < ghost_1_pos[1]:
                penalize -=10
              elif pacman_dir =="South" and newPos[1] > ghost_1_pos[1]:
                penalize -=10
              elif pacman_dir == "East" and newPos[0] < ghost_1_pos[0]:
                penalize -=10
              elif pacman_dir == "West" and newPos[0] > ghost_1_pos[0]:
                penalize -=10

            if pacman_dir==reverse[ghost_2_dir]:
              if pacman_dir == "North" and newPos[1] < ghost_2_pos[1]:
                penalize -=10
              elif pacman_dir =="South" and newPos[1] > ghost_2_pos[1]:
                penalize -=10
              elif pacman_dir == "East" and newPos[0] < ghost_2_pos[0]:
                penalize -=10
              elif pacman_dir == "West" and newPos[0] > ghost_2_pos[0]:
                penalize -=10
         # if pacman_dir==ghost_2_dir and (newPos[0]==ghost_2_pos[0] or newPos[1]==ghost_2_pos[1]):
          return (penalize)
        
        penalty = penalize_wrt_ghost_direction() + penalize_wrt_ghost_distance() + penalize_wrt_food_distance()
        return successorGameState.getScore() + penalty

def scoreEvaluationFunction(currentGameState):
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
    
    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
      

        def minimax(gameState, depth, agent, number_of_ghosts):
          legal_moves=gameState.getLegalActions(agent)
          #if we reach a leaf
          if depth==self.depth or legal_moves==[]:
            score=self.evaluationFunction(gameState)
            return (score, "")

          #last ghost
          if agent==(number_of_ghosts):
            #next one is pacman
            recursive_agent = 0
            depth+=1
          #we get the next ghost
          else:
            recursive_agent = agent + 1

          info_list=[]
          for move in legal_moves:
            
            if len(info_list)==0:
              successor_score, _ = minimax(gameState.generateSuccessor(agent,move), depth, recursive_agent, number_of_ghosts)
              info_list.append(successor_score)
              info_list.append(move)
            
            else:
              score=info_list[0]
              successor_score, _ = minimax(gameState.generateSuccessor(agent,move), depth, recursive_agent, number_of_ghosts)
              
              if (agent==0 and successor_score > score) or (agent>0 and successor_score<score):
                  info_list[0]=successor_score
                  info_list[1]=move
         
          return info_list

        number_of_ghosts=gameState.getNumAgents() - 1
        _ , action = minimax(gameState, 0, 0, number_of_ghosts)
        return(action)
      
import math
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alpha_beta_prunning(gameState, depth, agent, number_of_ghosts, alpha, beta):
          legal_moves = gameState.getLegalActions(agent)
          #if we reach a leaf
          if depth == self.depth or legal_moves == []:
            score = self.evaluationFunction(gameState)
            return (score, "")

          #last ghost
          if agent == (number_of_ghosts):
            #next one is pacman
            recursive_agent = 0
            depth += 1
          #we get the next ghost
          else:
            recursive_agent = agent + 1

          info_list = []
          for move in legal_moves:
            
            if len(info_list)==0:
              successor_score, _ = alpha_beta_prunning(gameState.generateSuccessor(agent,move), depth, recursive_agent, number_of_ghosts, alpha, beta)
              info_list.append(successor_score)
              info_list.append(move)
              #update alpha, beta
              if agent == 0: alpha = max(info_list[0], alpha) 
              else: beta = min(info_list[0], beta) 

            else:
              #Prunning operation
              if (successor_score < alpha and agent > 0) or (successor_score > beta and agent == 0):
                return info_list

              score = info_list[0]
              successor_score, _ = alpha_beta_prunning(gameState.generateSuccessor(agent,move), depth, recursive_agent, number_of_ghosts, alpha, beta)
              
              if (agent == 0 and successor_score > score) or (agent > 0 and successor_score < score):
                  info_list[0] = successor_score
                  info_list[1] = move
                  #update alpha, beta
                  if agent == 0: alpha = max(info_list[0], alpha) 
                  else: beta = min(info_list[0], beta) 
         
          return info_list

        number_of_ghosts=gameState.getNumAgents() - 1
        _ , action = alpha_beta_prunning(gameState, 0, 0, number_of_ghosts, -math.inf, math.inf)
        return(action)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
          The expectimax function returns a tuple of (actions,
        """
        "*** YOUR CODE HERE ***"
        def ExpectiMax(gameState, depth, agent, number_of_ghosts):
            legal_moves = gameState.getLegalActions(agent)
            #if we reach a leaf
            if depth == self.depth or legal_moves == []:
              score = self.evaluationFunction(gameState)
              return (score, "")


            #last ghost
            if agent == (number_of_ghosts):
              #next one is pacman
              recursive_agent = 0
              depth += 1
            #we get the next ghost
            else:
              recursive_agent = agent + 1

            info_list = []
            for move in legal_moves:
                if len(info_list)==0: # First move
                    successor_score, _ = ExpectiMax(gameState.generateSuccessor(agent,move), depth, recursive_agent, number_of_ghosts)
                    if agent==0: info_list.append(successor_score); info_list.append(move) 
                    else: info_list.append(float(1/len(legal_moves))*successor_score); info_list.append(move)
                else:
                    score=info_list[0]
                    successor_score, _ = ExpectiMax(gameState.generateSuccessor(agent,move), depth, recursive_agent, number_of_ghosts)
                    
                    # Check if miniMax value is better than the previous one #
                    if agent == 0:
                        if successor_score > score:
                          info_list[0]=successor_score
                          info_list[1]=move

                    else:
                        info_list[0] = info_list[0] + float((1/len(gameState.getLegalActions(agent)))) * successor_score
                        info_list[1] = move
            return info_list

        number_of_ghosts=gameState.getNumAgents() - 1
        _ , action = ExpectiMax(gameState, 0, 0, number_of_ghosts)

        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
      Evaluate state by  :
            * closest food
            * food left (?)
            * capsules 
            * directions of ghosts and pacman
            * distance to ghost
    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()

     #Adjast the score based on the distance between pacman and foods
    def penalize_wrt_capsules():
      penalize = 0 
      capsule = currentGameState.getCapsules()
      for cap in capsule:
        distance = util.manhattanDistance(cap, Pos)
        if distance <=2:
          penalize +=5 
        elif distance >2 and distance <4: 
          penalize +=3
        elif distance >=4 and distance <10:
              penalize +=1.5
        else: 
          penalize+=0.5
      penalize -= len(capsule)*20
      return (penalize)

    def penalize_wrt_food_left():
      food_left = len(Food.asList())
      penalize = - food_left * 20
      return(penalize)

    def penalize_wrt_food_distance():
          penalize = 0
          Food = currentGameState.getFood().asList()
          for food in Food:
            distance = util.manhattanDistance(food, Pos)
            if distance <=2:
              penalize+=1
            elif distance >2 and distance <4:
              penalize+=0.8
            elif distance >=4 and distance <10:
              penalize +=0.6
            else:
              penalize+=0.2
          if len(Food)<20:penalize*10
          return (penalize)

        #Adjast the score based on the distance between pacman and ghosts
    def penalize_wrt_ghost_distance():
          penalize = 0
          ghost_positions = currentGameState.getGhostPositions()
          for new_ghost_pos in ghost_positions:
            distance = util.manhattanDistance(new_ghost_pos, Pos)
            if distance < 2:
              penalize -=10
            elif distance > 2 and distance <=5:
              penalize -=7
            elif distance > 5 and distance <=10:
              penalize -=5
            else:
              penalize +=1
          return (penalize)

        #Adjast the score based on the ghosts directions
    def penalize_wrt_ghost_direction():
          penalize = 0 
          reverse = {
                    "North": "South",
                    "South": "North",
                    "East": "West",
                    "West": "East",
                    "Stop": "Stop"
                    }
          pacman_dir = currentGameState.getPacmanState().getDirection()
          ghost_state = currentGameState.getGhostStates()

          if len(GhostStates)==1:
           
            ghost_dir = ghost_state[0].getDirection()
            
            ghost_pos = ghost_state[0].getPosition()
            if pacman_dir==reverse[ghost_dir]:
              if pacman_dir == "North" and Pos[1] < ghost_pos[1]:
                penalize -=20
              elif pacman_dir =="South" and Pos[1] > ghost_pos[1]:
                penalize -=20
              elif pacman_dir == "East" and Pos[0] < ghost_pos[0]:
                penalize -=20
              elif pacman_dir == "West" and Pos[0] > ghost_pos[0]:
                penalize -=20

          if len(GhostStates)==2: 
            ghost_1_dir = ghost_state[0].getDirection()
            ghost_2_dir = ghost_state[1].getDirection()
            ghost_1_pos = ghost_state[0].getPosition()
            ghost_2_pos = ghost_state[1].getPosition()
            if pacman_dir==reverse[ghost_1_dir]:
              if pacman_dir == "North" and Pos[1] < ghost_1_pos[1]:
                penalize -=10
              elif pacman_dir =="South" and Pos[1] > ghost_1_pos[1]:
                penalize -=10
              elif pacman_dir == "East" and Pos[0] < ghost_1_pos[0]:
                penalize -=10
              elif pacman_dir == "West" and Pos[0] > ghost_1_pos[0]:
                penalize -=10

            if pacman_dir==reverse[ghost_2_dir]:
              if pacman_dir == "North" and Pos[1] < ghost_2_pos[1]:
                penalize -=10
              elif pacman_dir =="South" and Pos[1] > ghost_2_pos[1]:
                penalize -=10
              elif pacman_dir == "East" and Pos[0] < ghost_2_pos[0]:
                penalize -=10
              elif pacman_dir == "West" and Pos[0] > ghost_2_pos[0]:
                penalize -=10
          return (penalize)
        
    penalty = penalize_wrt_ghost_direction() + penalize_wrt_ghost_distance()*1/2 + penalize_wrt_food_distance() + penalize_wrt_food_left()*2 + penalize_wrt_capsules()
    return(penalty)
# Abbreviation
better = betterEvaluationFunction
