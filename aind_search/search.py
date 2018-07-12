# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util
from copy import deepcopy

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]


def graphSearch(problem, frontier):
    node = problem.getStartState()

    if problem.isGoalState(node):
        return []

    explored = set()

    frontier.push((node, []))

    while not frontier.isEmpty():

        current_state, path = frontier.pop()

        if problem.isGoalState(current_state):
          # should return list of actions
          return path

        explored.add(current_state)

        for successor in problem.getSuccessors(current_state):
            new_node = successor[0]
            new_path = successor[1]

            if new_node not in explored:
                path2 = deepcopy(path)
                path2.append(new_path)

                frontier.push((new_node, path2))

    return []


def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first
  [2nd Edition: p 75, 3rd Edition: p 87]
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm 
  [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  return graphSearch(problem, frontier = util.Stack())

def breadthFirstSearch(problem):
  """
  Search the shallowest nodes in the search tree first.
  [2nd Edition: p 73, 3rd Edition: p 82]
  """
  return graphSearch(problem, frontier = util.Queue())
      
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  current_node = problem.getStartState()
  frontier = util.PriorityQueue()
  explored = set()

  cost =  problem.getCostOfActions([])
  frontier.push((current_node, []), cost)

  while frontier.isEmpty() is not True:

    state, path = frontier.pop()


    if problem.isGoalState(state):
        return path

    explored.add(state)

    for successor in problem.getSuccessors(state):

        #if successor[0] not in explored or successor[0] not in frontier.heap:
        if successor[0] not in explored:
            f_node = successor[0]
            f_path = deepcopy(path)
            f_path.append(successor[1])
            f_cost = problem.getCostOfActions(path)
            frontier.push((f_node, f_path), f_cost)
        elif state in frontier.heap:
            frontier.push((f_node, f_path), f_cost)
  return []


def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."

  current_node = problem.getStartState()
  frontier = util.PriorityQueue()
  explored = set()

  cost = heuristic(current_node, problem)
  frontier.push((current_node, []), cost)

  while frontier.isEmpty() is not True:

      state, path = frontier.pop()

      if problem.isGoalState(state):
          return path

      explored.add(state)

      for successor in problem.getSuccessors(state):

          # if successor[0] not in explored or successor[0] not in frontier.heap:
          if successor[0] not in explored:
              f_node = successor[0]
              f_path = deepcopy(path)
              f_path.append(successor[1])
              f_cost = heuristic(f_node, problem)
              frontier.push((f_node, f_path), f_cost)
          elif state in frontier.heap:
              frontier.push((f_node, f_path), f_cost)
  return []
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
