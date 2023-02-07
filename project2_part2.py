
#Nick Bischoff
#AI
#Project 2 MDP part 2
#This program utilizes the value iteration alogrithm, given an MDP.
#This takes in through command line argument, a number for transition probability
#to the intended direction and a number for rewards to states other than the two
#terminal states.
#It then calculates the optimal policy, and outputs it to console and also
# updates it in a "expectimax.csv" file.

#libraries
import csv
import sys
import numpy as np
import random
import numbers

#define discount and maxium error
DISCOUNT = 0.95
MAX_ERROR = 10**(-3)

# Set up the initial environment
NUM_ACTIONS = 4
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)] # Up, Right, Down, Left
NUM_ROW = 3
NUM_COL = 4
U = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]

#for command line arguemnts
input1 = sys.argv[1]
input2 = sys.argv[2]

#set the first command line arguemnt to transition probabilty and
#the second to the reward
transitionProbability = input1
defaultReward = input2
# define the policy and the return
policy = [[random.randint(0,3) for j in range(NUM_COL)] for i in range(NUM_ROW)]
policy_return = [[-1 for j in range(NUM_COL)] for i in range(NUM_ROW)]

#visualization
#pre: need a way to print to console the optimal policy_return
#post: Now we have a function to print the optimal policy for the user to see
def printEnvironment(arr, policy=False):
    res = ""
    for r in range(NUM_ROW):
        res += "["
        for c in range(NUM_COL):
            if r == c == 1:
                val = "0"
                policy_return[r][c] = 0
            elif r <= 1 and c == 3:
                val = "0" if r == 0 else "0"
                policy_return[r][c] = 0
            else:
                val = ["1", "2", "-1", "-2"][arr[r][c]]
                policy_return[r][c] = ["1", "2", "-1", "-2"][arr[r][c]]
            res += " " + val[:5].ljust(5) # format
            if(c == 3):
                res += "]"
            else:
                res += ","
        res += "\n"
    print(res)

#pre: need way to define a multi demensional arrays
#post: This function allows use of a multi demensional array
def getU(U, r, c, action):
    dr, dc = ACTIONS[action]
    newR, newC = r+dr, c+dc
    if newR < 0 or newC < 0 or newR >= NUM_ROW or newC >= NUM_COL or (newR == newC == 1):
        return U[r][c]
    else:
        return U[newR][newC]

#pre: need way to persom value iteration
#post: This function will calculate the value iteration
def calculateU(U, r, c, action):
    u = float(defaultReward)
    u += 0.1 * DISCOUNT * getU(U, r, c, (action-1)%4)
    u += 0.8 * DISCOUNT * getU(U, r, c, action)
    u += 0.1 * DISCOUNT * getU(U, r, c, (action+1)%4)
    return u

#pre: need way of evalutating which policy is the optimal policy
#post: This function allows for evaluating the best (optimal) policy
def policyEvaluation(policy, U):
    while True:
        nextU = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]
        error = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                if (r <= 1 and c == 3) or (r == c == 1):
                    continue
                nextU[r][c] = calculateU(U, r, c, policy[r][c])
                error = max(error, abs(nextU[r][c]-U[r][c]))
        U = nextU
        if error < MAX_ERROR * (1 - DISCOUNT) / DISCOUNT:
            break
    return U

#pre: need way of iterating through all of the policies
#post: this function allows for iterating through all of the calculated policies
def policyIteration(policy, U):
    while True:
        U = policyEvaluation(policy, U)
        unchanged = True
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                if(r <= 1 and c == 3) or (r == c == 1):
                    continue
                maxAction, maxU = None, -float("inf")
                for action in range(NUM_ACTIONS):
                    u = calculateU(U, r, c, action)
                    if u > maxU:
                        maxAction, maxU = action, u
                    if maxU > calculateU(U, r, c, policy[r][c]):
                        policy[r][c] = maxAction
                        unchanged = False
                if maxU > calculateU(U, r, c, policy[r][c]):
                    policy[r][c] = maxAction
                    unchanged = false
        if unchanged:
            break
    return policy

policy = policyIteration(policy, U)

# call to the print environment function to output the optimal policy to console
print("The optimal policy is :\n")
printEnvironment(policy)

# Writes optiml policy to expectimax.csv
with open("expectimax.csv", "w+") as expectimax_csv:
    csvWriter = csv.writer(expectimax_csv,delimiter=',')
    csvWriter.writerows(policy_return)
