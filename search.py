# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from operator import itemgetter
import math


# manhattanHeuristic here, because searchAgents.py is depended on search.py, can't directly import to use manhattanHeuristic
def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
    
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    A sample depth first search implementation is provided for you to help you understand how to interact with the problem.
    """
    
    mystack = util.Stack()
    startState = (problem.getStartState(), '', 0, [])
    mystack.push(startState)
    visited = set()
    while mystack :
        state = mystack.pop()
        node, action, cost, path = state
        if node not in visited :
            visited.add(node)
            if problem.isGoalState(node) :
                path = path + [(node, action)]
                break;
            succStates = problem.getSuccessors(node)
            for succState in succStates :
                succNode, succAction, succCost = succState
                newstate = (succNode, succAction, cost + succCost, path + [(node, action)])
                mystack.push(newstate)
    actions = [action[1] for action in path]
    del actions[0]
    return actions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# def aStarSearch(problem, startState, goalState, heuristic=manhattanHeuristic):
#     startNode = (startState, '', 0, [])#state, aciotn, g_n, path
#     # goalState = getGoalState
#     # startNode = (startState, "", 0, []) #state, aciotn, g_n, path
#     openPQ = util.PriorityQueue()
#     openPQ.push(startNode, startNode[2] + util.manhattanDistance(startState, goalState))
#     closeSet = set()
#     best_gDic = {}
#     while not openPQ.isEmpty():
#         node = openPQ.pop()
#         state, action, g_n, path = node
#         if state in best_gDic.keys():
#             best_g = best_gDic[state]
#         else:
#             best_gDic[state] = g_n
#             best_g = g_n

#         if state not in closeSet or g_n < best_g:
#             closeSet.add(state)
#             best_gDic[state] = g_n
#             if state == goalState:
#                 #extract solution
#                 actions = [action[1] for action in path]
#                 pathStates = [state[0] for state in path]
#                 actions.append(action)
#                 pathStates.append(state)
#                 del actions[0]
#                 return (pathStates, actions, best_gDic)
#             stateSuccessors = problem.getSuccessors(state)
#             for successor in stateSuccessors:
#                 succState, succAction, succCost = successor
#                 newNode = (succState, succAction, g_n + succCost, path + [(state, action)])
#                 if util.manhattanDistance(succState, goalState) < math.inf:
#                     openPQ.push(newNode, newNode[2] + util.manhattanDistance(newNode[0], goalState))

def aStarSearch(problem, startState, goalState, realGoalState, fakeGoalState, pid3 = False, heuristic=manhattanHeuristic):
    startNode = (startState, '', 0, 0, [])#state, aciotn, g_n, f_n, path
    # goalState = getGoalState
    # startNode = (startState, "", 0, []) #state, aciotn, g_n, path
    openPQ = util.PriorityQueue()
    openPQ.push(startNode, startNode[2] + util.manhattanDistance(startState, goalState))
    closeSet = set()
    best_gDic = {}
    cnt = 0
    while not openPQ.isEmpty():
        node = openPQ.pop()
        state, action, g_n, f_n, path = node
        if state in best_gDic.keys():
            best_g = best_gDic[state]
        else:
            best_gDic[state] = g_n
            best_g = g_n

        if state not in closeSet or g_n < best_g:
            closeSet.add(state)
            best_gDic[state] = g_n
            if state == goalState:
                #extract solution
                actions = [action[1] for action in path]
                pathStates = [state[0] for state in path]
                actions.append(action)
                pathStates.append(state)
                del actions[0]
                return (pathStates, actions, best_gDic)
            stateSuccessors = problem.getSuccessors(state)
            for successor in stateSuccessors:
                succState, succAction, succCost = successor
                #pid3
                if pid3 and util.manhattanDistance(succState, realGoalState) < util.manhattanDistance(succState, fakeGoalState):
                    alpha = 2
                else:
                    alpha = 1

                newNode = (succState, succAction, g_n + succCost, g_n + succCost + alpha * util.manhattanDistance(succState, goalState), path + [(state, action)])
                if util.manhattanDistance(succState, goalState) < math.inf:
                    openPQ.push(newNode, priority = newNode[3])

# def aStarSearch(problem, heuristic=manhattanHeuristic):
#     """Search the node that has the lowest combined cost and heuristic first."""
#     "*** YOUR CODE HERE ***"
#     util.raiseNotDefined()

def enforcedHillClimbing(problem, heuristic=manhattanHeuristic):
    """COMP90054 your solution to part 1 here """
# ###### function #######
## TODO: revised path
## TODO: revised Heuristics

    # improve function
    def improve(node_0):
        myqueue = util.Queue()
        myqueue.push(node_0)
        state_0, action_0, cost_0, path_0 = node_0
        closed = set() #visited states set
        
        while not myqueue.isEmpty() :
            node = myqueue.pop()
            state, action, cost, path = node
            if state not in closed :
                closed.add(state)
                if manhattanHeuristic(state, problem) < manhattanHeuristic(state_0, problem):
                    return (state, action, cost, path)
                curStateSuccessors = problem.getSuccessors(state)
                #get (nextState, action, cost) in current state
                for curStateSuccessor in curStateSuccessors:
                    succState, succAction, succCost = curStateSuccessor
                    newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                    myqueue.push(newNode)
    
    # initialize
    curNode = (problem.getStartState(), '', 0, []) #(state, action, cost, path)
    curState, curAction, curCost, curPath = curNode
    # run imrpove
    while not problem.isGoalState(curState):
        curNode = improve(curNode)
        curState, curAction, curCost, curPath = curNode

    #extract solution
    actions = [action[1] for action in curPath]
    actions.append(curAction)
    del actions[0]
    return actions

    
def idaStarSearch(problem, heuristic=manhattanHeuristic):
    """COMP90054 your solution to part 2 here """
    #ref1: https://en.wikipedia.org/wiki/Iterative_deepening_A*
    #ref2: https://en.wikipedia.org/wiki/Iterative_deepening_A*#cite_note-re1985_7-2


    def search(mypath, g_n, threshold) : #recursive function
        node = mypath[-1]
        state, action, cost, path = node
        f_n = g_n + manhattanHeuristic(state, problem)
        # print("state", state, "MH", manhattanHeuristic(state, problem))
        if f_n > threshold :
            return f_n
        if problem.isGoalState(state) :
            return "FOUND"
        min = float("inf")

        curStateSuccessors = problem.getSuccessors(state)
        # sorted(curStateSuccessors, key=manhattanHeuristic(itemgetter(0), problem), reverse=True) #sort by manhattanHeuristic desending
        # print(curStateSuccessors)
        for curStateSuccessor in curStateSuccessors :
            succState, succAction, succCost = curStateSuccessor
            if curStateSuccessor not in mypath :
                newNode = (succState, succAction, succCost, path + [(state, action)])
                mypath.append(newNode)
                #something wrong with my g_n?
                t = search(mypath, g_n + succCost, threshold)
                if t == "FOUND":
                    return "FOUND"
                if t < min:
                    min = t
                mypath.pop()
        return min

    # initialize
    curNode = (problem.getStartState(), '', 0, []) #(state, action, cost, path)
    curState, curAction, curCost, curPath = curNode
    
    # initial threshold set as manhattanHeuristic
    threshold = 0
    # Fianl Path use stack
    mypath = []
    mypath.append(curNode)

    while True :
        t = search(mypath, 0, threshold)
        if t == "FOUND" :
            print("FOUND")
            break
            # return mypath
        if math.isinf(t) :
            print("NOT_FOUND")
            # return "NOT_FOUND"
        # print("threshold",threshold)
        threshold = t #???
    # print(mypath[-1])
    finalpath = mypath[-1]
    # for action in finalpath[3]:
        # print("action:",action)
    # extract solutions
    actions = [action[1] for action in finalpath[3]]
    print(actions)
    actions.append(finalpath[1])
    del actions[0]
    return actions

#### temp
                    # successorList.append(newNode)
                    # >>> data = [('abc', 121),('abc', 231),('abc', 148), ('abc',221)]
                    # >>> sorted(data,key=itemgetter(1))
                    # current iteration
                    # successorsList.append(curStateSuccessor)
                # sort current iteration in descending by f_n (how to get f_n)
                # sorted(successorsList, key=itemgetter(2), reverse=True)
                # for i in range(len(successorList)):
                    # successorNode = successorsList[i]

                # for i in range(len(successorList)):
                #     mystack.push(successorList[i])


                
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ehc = enforcedHillClimbing
ida = idaStarSearch