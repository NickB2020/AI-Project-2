#Nick Bischoff
#AI
#Project 2 MDP part 1
#This program utilizes the policy evalutation MDP alogrithm, given an MDP.
#This takes in through command line argument, a number for reward
#to all other states and a policy in form of a .csv file.
#It then calculates the expected utility of the given policy  from iteration
#number 19 and outputs it to console.

#libraries
import pandas as pd
import numpy as np
import sys

#input command line arguments
fileName = sys.argv[2]
policies = pd.read_csv(fileName, header=None)
policiesArr = policies.to_numpy()

#define reward
reward = float(sys.argv[1])

#initialize arrays
policiesArr[1][1] = 0
V = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]
Rewards = [[0,0,0,0],[0, 0, 0, 0], [0,0,0,0]]
for k in range(len(policiesArr)):
    for l in range(len(policiesArr[k])):
        if (k == 0 and l == 3):
            Rewards[k][l] = 1
        elif (k == 1 and l == 3):
            Rewards[k][l] = -1
        else:
            Rewards[k][l] = reward


#pre: need way of evaluating the policy_return
#post: this function allows for policy evaluation
def policyEvaluation(pi, U, maxIterations=20, gamma = 0.95):

    for iteration in range(maxIterations):
        for i in range(len(policiesArr)):
            for j in range(len(policiesArr[i])):
                U[i][j] = sum([p * (Rewards[nextState[0]][nextState[1]] + gamma * U[nextState[0]][nextState[1]]) for (p, (nextState)) in transitionProbability((i, j), pi[i][j])])
    return U

#pre:  need way to find next state,
#current state as a (x,y) tuple, current action from policy array
#post: this function allows for finding the next state in the policy
#list of tuples where (probability, next state)
def transitionProbability(state, action):
    i = state[0]
    j = state[1]
    actionList = []
    if ((i == 0 and j == 3)):
        return [(0.0, (0,3))]
    elif (i == 1 and j == 3):
        return [(0.0, (1, 3))]

    else:
        if action == 1:

            if (i-1 < 0 or (i-1 == 1 and j == 1)):
                intended = state
            else:
                intended = i-1, j
            if (j-1 < 0 or (i==1 and j-1==1)):
                stateLeft = state
            else:
                stateLeft = (i, j-1)
            if (j+1 > 3 or (i==1 and j+1==1)):
                stateRight = state
            else:
                stateRight = (i, j+1)

            actionList = [(0.8, (intended)),
                    (0.1, (stateLeft)),
                    (0.1, (stateRight))]
        elif action == -1:
            if (i+1 > 2 or (i+1 == 1 and j == 1)):
                intended = state
            else:
                intended = i+1,j
            if (j - 1 < 0 or (i == 1 and j - 1 == 1)):
                stateLeft = state
            else:
                stateLeft = (i,j-1)
            if (j + 1 > 3 or (i == 1 and j + 1 == 1)):
                stateRight = state
            else:
                stateRight = (i, j+1)

            actionList = [(0.8, (intended)),
                    (0.1, (stateLeft)),
                    (0.1, (stateRight))]
        elif action == 2:
            if (j+1 > 3 or (i == 1 and j+1 == 1)):
                intended = state
            else:
                intended = i, j+1
            if (i - 1 < 0 or (i - 1 == 1 and j == 1)):
                stateUp = state
            else:
                stateUp = (i-1,j)

            if (i + 1 > 2 or (i+1 == 1 and j == 1)):
                stateDown = state
            else:
                stateDown = (i+1,j)

            actionList = [(0.8, (intended)),
                    (0.1, (stateUp)),
                    (0.1, (stateDown))]
        elif action == -2:
            if (j-1 < 0 or (i == 1 and j-1 == 1)):
                intended = state
            else:
                intended = i, j-1
            if (i - 1 < 0 or (i - 1 == 1 and j == 1)):
                stateUp = state
            else:
                stateUp = (i-1,j)
            if (i + 1 > 2 or (i + 1 == 1 and j == 1)):
                stateDown = state
            else:
                stateDown = (i+1,j)

            actionList = [(0.8, (intended)),
                    (0.1, (stateUp)),
                    (0.1, (stateDown))]
    return actionList

answer = policyEvaluation(policiesArr, V)
print(answer[2][0])
