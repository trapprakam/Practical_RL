# Code is reimplemented from Andrej Karpathy Github
# https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d

"""
A bare bones examples of optimizing a black-box function (f) using
Natural Evolution Strategies (NES), where the parameter distribution is a
gaussian of fixed standard deviation.
"""

import numpy as np
np.random.seed(0)


# the function we want to optimize
def f(w):
    """
    here we would normally:
    ... 1) create a neural network with weights w
    ... 2) run the neural network on the environment for some time
    ... 3) sum up and return the total reward
    but for the purposes of an example, lets try to minimize
    the L2 distance to a specific solution vector. So the highest reward
    we can achieve is 0, when the vector w is exactly equal to solution
    :param w: (array) An array containing thee changed parameters
    :return: The loss/reward of the parameters with respect to the solution
    """
    return -np.sum(np.square(solution - w))


# hyper parameters
n_pop = 50  # population size
sigma = 0.1  # noise standard deviation
alpha = 0.001  # learning rate

# start the optimization
solution = np.array([0.5, 0.1, -0.3])
w = np.random.randn(3)  # our initial guess is random

for i in range(300):
    # print current fitness of the most likely parameter setting
    if i % 20 == 0:
        print(f"iter {i}. w: {w}, solution: {solution}, reward: {f(w)}")

        # initialize memory for a population of w's, and their rewards
        N = np.random.randn(n_pop, 3)  # samples from a normal distribution N(0,1)
        R = np.zeros(n_pop)
        for j in range(n_pop):
            w_try = w + sigma * N[j]  # jitter w using gaussian of sigma 0.1
            R[j] = f(w_try)  # evaluate the jittered version
            # standardize the rewards to have a gaussian distribution
            A = (R - np.mean(R)) / np.std(R)
            # perform the parameter update. The matrix multiply below
            # is just an efficient way to sum up all the rows of the noise matrix N,
            # where each row N[j] is weighted by A[j]
            w = w + alpha/(n_pop * sigma) * np.dot(N.T, A)


