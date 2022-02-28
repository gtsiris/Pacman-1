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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from game import Directions  # Needed to give instructions
    n = Directions.NORTH  # Go north
    s = Directions.SOUTH  # Go south
    e = Directions.EAST  # Go east
    w = Directions.WEST  # Go west
    from util import Stack  # Used in order to achieve DFS traversal and build the path
    nodesToExplore = Stack()  # Nodes that will be explored
    startState = problem.getStartState()  # Initial state of pacman
    startNode = (startState, 'Stop', 0)  # Initial node
    nodesToExplore.push(startNode)  # Add it for exploration
    exploredNodes = Stack()  # Keep track of previously encountered nodes
    exploredEdges = []  # Keep track of movements between states
    path = []  # The final path that pacman is going to follow
    while not nodesToExplore.isEmpty():  # There are more nodes to be explored
        currentNode = nodesToExplore.pop()  # Get the most recently encountered node
        currentState = currentNode[0]  # State of current node
        exploredNodes.push(currentNode)  # Add it to the explored states
        if problem.isGoalState(currentState):  # If the current state is a goal
            targetNode = currentNode  # This is target node
            targetState = targetNode[0]  # State of target node
            while (targetState != startState) and (not exploredNodes.isEmpty()):  # Build the path backwards
                # start from the detected goal state and reach the start
                tempNode = exploredNodes.pop()  # Get an already explored node
                tempState = tempNode[0]  # State of temp node
                if tempState == targetState:  # If its state is the target
                    targetNode = tempNode  # Then this is the target node
                    targetDirection = targetNode[1]  # Check the direction that leads here
                    targetX = targetState[0]  # First coordinate of target
                    targetY = targetState[1]  # Second coordinate of target
                    if targetDirection == 'North':  # If moving north is needed to get to that target
                        path.insert(0, n)  # Add north in the beginning of path
                        targetY -= 1  # The previous node is one block down
                    elif targetDirection == 'South':  # Else if moving south is needed to get to that target
                        path.insert(0, s)  # Add south in the beginning of path
                        targetY += 1  # The previous node is one block up
                    elif targetDirection == 'East':  # Else if moving east is needed to get to that target
                        path.insert(0, e)  # Add east in the beginning of path
                        targetX -= 1  # The previous node is one block left
                    elif targetDirection == 'West':  # Else if moving west is needed to get to that target
                        path.insert(0, w)  # Add west in the beginning of path
                        targetX += 1  # The previous node is one block right
                    targetState = (targetX, targetY)  # Construct the state of the previous node
            return path  # Final path is found
        # If current state is not a goal, look for successors
        successorNodes = problem.getSuccessors(currentState)  # List of current state's successor nodes
        for successorNode in successorNodes:  # For each successor node in this list
            successorState = successorNode[0]  # State of successor
            edge = {currentState, successorState}  # Movement between those two states
            if edge not in exploredEdges:  # If there has not been a movement between the two states
                nodesToExplore.push(successorNode)  # Add it for future exploration
                exploredEdges.append(edge)  # Add the edge to the explored ones


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from game import Directions  # Needed to give instructions
    n = Directions.NORTH  # Go north
    s = Directions.SOUTH  # Go south
    e = Directions.EAST  # Go east
    w = Directions.WEST  # Go west
    path = []  # The final path that pacman is going to follow
    startState = problem.getStartState()  # Initial state
    if problem.isGoalState(startState):  # If the initial state is a goal
        return path  # No moves required
    from util import Queue  # Used in order to achieve BFS traversal
    nodesToExplore = Queue()  # Nodes that will be explored
    startNode = (startState, 'Stop', 0)  # Initial node
    nodesToExplore.push(startNode)  # Add it for exploration
    from util import Stack  # Used in order to build the path
    exploredNodes = Stack()  # Keep track of previously encountered nodes
    exploredEdges = []  # Keep track of movements between states
    while not nodesToExplore.isEmpty():  # There are more nodes to be explored
        currentNode = nodesToExplore.pop()  # Get the least recently encountered node
        currentState = currentNode[0]  # State of current node
        exploredNodes.push(currentNode)  # Add it to the explored nodes
        successorNodes = problem.getSuccessors(currentState)  # List of current state's successor nodes
        for successorNode in successorNodes:  # For each successor node in this list
            successorState = successorNode[0]  # State of successor
            edge = {currentState, successorState}  # Movement between those two states
            if edge not in exploredEdges:  # If there has not been a movement between the two states
                if problem.isGoalState(successorState):  # If the successor's state is a goal
                    exploredNodes.push(successorNode)  # Add it to the explored nodes
                    targetNode = successorNode  # This is target node
                    targetState = targetNode[0]  # State of target node
                    while (targetState != startState) and (not exploredNodes.isEmpty()):  # Build the path backwards
                        # start from the detected goal state and reach the start
                        tempNode = exploredNodes.pop()  # Get an already explored node
                        tempState = tempNode[0]  # State of temp node
                        if tempState == targetState:  # If its state is the target
                            targetNode = tempNode  # Then this is the target node
                            targetDirection = targetNode[1]  # Check the direction that leads here
                            targetX = targetState[0]  # First coordinate of target
                            targetY = targetState[1]  # Second coordinate of target
                            if len(targetState) == 3:  # Added to solve corners problem
                                cornerStatus = list(targetState[2])  # Number of visits per corner
                                for i in range(len(cornerStatus)):  # For each corner
                                    if (targetX, targetY) == cornerStatus[i][0]:  # If target is a corner
                                        cornerStatus[i] = (cornerStatus[i][0], cornerStatus[i][1] - 1)
                                        # The previous node in the path is going to have one less visit in this corner
                            if targetDirection == 'North':  # If moving north is needed to get to that target
                                path.insert(0, n)  # Add north in the beginning of path
                                targetY -= 1  # The previous node is one block down
                            elif targetDirection == 'South':  # Else if moving south is needed to get to that target
                                path.insert(0, s)  # Add south in the beginning of path
                                targetY += 1  # The previous node is one block up
                            elif targetDirection == 'East':  # Else if moving east is needed to get to that target
                                path.insert(0, e)  # Add east in the beginning of path
                                targetX -= 1  # The previous node is one block left
                            elif targetDirection == 'West':  # Else if moving west is needed to get to that target
                                path.insert(0, w)  # Add west in the beginning of path
                                targetX += 1  # The previous node is one block right
                            # Construct the state of the previous node
                            if len(targetState) == 3:  # Added to solve corners problem
                                targetState = (targetX, targetY, tuple(cornerStatus))
                            else:
                                targetState = (targetX, targetY)
                    return path  # Final path is found
                # If its state is not goal
                nodesToExplore.push(successorNode)  # Add it for future exploration
                exploredEdges.append(edge)  # Add the edge to the explored ones


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from game import Directions  # Needed to give instructions
    n = Directions.NORTH  # Go north
    s = Directions.SOUTH  # Go south
    e = Directions.EAST  # Go east
    w = Directions.WEST  # Go west
    from util import Stack  # Used in order to build the path
    from util import PriorityQueue  # Used in order to achieve UCS
    nodesToExplore = PriorityQueue()  # Nodes that will be explored
    startState = problem.getStartState()  # Initial state
    startNode = (startState, 'Stop', 0)  # Initial node
    startCost = startNode[2]  # Cost of initial node
    nodesToExplore.push(startNode, startCost)  # Add it for exploration
    exploredNodes = PriorityQueue()  # Keep track of previously encountered nodes
    exploredEdges = []  # Keep track of movements between states
    path = []  # The final path that pacman is going to follow
    while not nodesToExplore.isEmpty():  # There are more nodes to be explored
        currentNode = nodesToExplore.pop()  # Get the node with the highest priority
        currentState = currentNode[0]  # State of current node
        currentCost = currentNode[2]  # Cost of current node
        exploredNodes.update(currentNode, currentCost)  # Update the explored nodes
        if problem.isGoalState(currentState):  # If the current state is a goal
            targetNode = currentNode  # This is target node
            targetState = targetNode[0]  # State of target node
            exploredNodesStack = Stack()  # Stack helps to get the explored nodes in the right order to build the path
            while not exploredNodes.isEmpty():  # Until there are no more nodes in the priority queue
                exploredNodesStack.push(exploredNodes.pop())  # Transfer them in a stack
            while (targetState != startState) and (not exploredNodesStack.isEmpty()):  # Build the path backwards
                # start from the detected goal state and reach the start
                tempNode = exploredNodesStack.pop()  # Get an already explored node
                tempState = tempNode[0]  # State of temp node
                if tempState == targetState:  # If its state is the target
                    targetNode = tempNode  # Then this is the target node
                    targetDirection = targetNode[1]  # Check the direction that leads here
                    targetX = targetState[0]  # First coordinate of target
                    targetY = targetState[1]  # Second coordinate of target
                    if targetDirection == 'North':  # If moving north is needed to get to that target
                        path.insert(0, n)  # Add north in the beginning of path
                        targetY -= 1  # The previous node is one block down
                    elif targetDirection == 'South':  # Else if moving south is needed to get to that target
                        path.insert(0, s)  # Add south in the beginning of path
                        targetY += 1  # The previous node is one block up
                    elif targetDirection == 'East':  # Else if moving east is needed to get to that target
                        path.insert(0, e)  # Add east in the beginning of path
                        targetX -= 1  # The previous node is one block left
                    elif targetDirection == 'West':  # Else if moving west is needed to get to that target
                        path.insert(0, w)  # Add west in the beginning of path
                        targetX += 1  # The previous node is one block right
                    targetState = (targetX, targetY)  # Construct the state of the previous node
            return path  # Final path is found
        # If current state is not a goal, look for successors
        successorNodes = problem.getSuccessors(currentState)  # List of current state's successor nodes
        for successorNode in successorNodes:  # For each successor node in this list
            successorCost = successorNode[2] + currentCost  # Add the current node's cost to its successor's cost
            successorNode = (successorNode[0], successorNode[1], successorCost)  # Update the successor's cost
            successorState = successorNode[0]  # State of successor node
            edge = {currentState, successorState}  # Movement between those two states
            if edge not in exploredEdges:  # If there has not been a movement between the two states
                nodesToExplore.update(successorNode, successorCost)
                # Add it or update its cost if it is already encountered
                exploredEdges.append(edge)  # Add the edge to the explored ones


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    def priorityFunction(node):  # Inner function that computes the priority of a node
        state = node[0]  # State of given node
        cost = node[2]  # Cost of given state
        heur = heuristic(state, problem)  # Heuristic evaluation of given state
        return cost + heur  # Priority depends both on cost and heuristic

    from game import Directions  # Needed to give instructions
    n = Directions.NORTH  # Go north
    s = Directions.SOUTH  # Go south
    e = Directions.EAST  # Go east
    w = Directions.WEST  # Go west
    from util import Stack  # Used in order to build the path
    from util import PriorityQueue  # Used in order to achieve UCS
    nodesToExplore = PriorityQueue()  # Nodes that will be explored
    startState = problem.getStartState()  # Initial state
    startNode = (startState, 'Stop', 0)  # Initial node
    nodesToExplore.push(startNode, priorityFunction(startNode))  # Add it for exploration
    exploredNodes = PriorityQueue()  # Keep track of previously encountered nodes
    exploredEdges = []  # Keep track of movements between states
    path = []  # The final path that pacman is going to follow
    while not nodesToExplore.isEmpty():  # There are more nodes to be explored
        currentNode = nodesToExplore.pop()  # Get the node with the highest priority
        currentState = currentNode[0]  # State of current node
        currentCost = currentNode[2]  # Cost of current node
        exploredNodes.update(currentNode, priorityFunction(currentNode))
        # Add it or update its cost if it is already explored
        if problem.isGoalState(currentState):  # If the current state is a goal
            targetNode = currentNode  # This is target node
            targetState = targetNode[0]  # State of target node
            exploredNodesStack = Stack()  # Stack helps to get the explored nodes in the right order to build the path
            while not exploredNodes.isEmpty():  # Until there are no more nodes in the priority queue
                exploredNodesStack.push(exploredNodes.pop())  # Transfer them in a stack
            while (targetState != startState) and (not exploredNodesStack.isEmpty()):  # Build the path backwards
                # start from the detected goal state and reach the start
                tempNode = exploredNodesStack.pop()  # Get an already explored node
                tempState = tempNode[0]  # State of temp node
                if tempState == targetState:  # If its state is the target
                    targetNode = tempNode  # Then this is the target node
                    targetDirection = targetNode[1]  # Check the direction that leads here
                    targetX = targetState[0]  # First coordinate of target
                    targetY = targetState[1]  # Second coordinate of target
                    if targetDirection == 'North':  # If moving north is needed to get to that target
                        path.insert(0, n)  # Add north in the beginning of path
                        targetY -= 1  # The previous node is one block down
                    elif targetDirection == 'South':  # Else if moving south is needed to get to that target
                        path.insert(0, s)  # Add south in the beginning of path
                        targetY += 1  # The previous node is one block up
                    elif targetDirection == 'East':  # Else if moving east is needed to get to that target
                        path.insert(0, e)  # Add east in the beginning of path
                        targetX -= 1  # The previous node is one block left
                    elif targetDirection == 'West':  # Else if moving west is needed to get to that target
                        path.insert(0, w)  # Add west in the beginning of path
                        targetX += 1  # The previous node is one block right
                    targetState = (targetX, targetY)  # Construct the state of the previous node
            return path  # Final path is found
        # If current state is not a goal, look for successors
        successorNodes = problem.getSuccessors(currentState)  # List of current state's successor nodes
        for successorNode in successorNodes:  # For each successor node in this list
            successorCost = successorNode[2] + currentCost  # Add the current node's cost to its successor's cost
            successorNode = (successorNode[0], successorNode[1], successorCost)  # Update the successor's cost
            successorState = successorNode[0]  # State of successor node
            edge = {currentState, successorState}  # Movement between those two states
            if edge not in exploredEdges:  # If there has not been a movement between the two states
                nodesToExplore.update(successorNode, priorityFunction(successorNode))
                # Add it or update its cost if it is already encountered
                exploredEdges.append(edge)  # Add the edge to the explored ones


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
