import random
import numpy as np
from kaggle_environments import make

N = None
Q = None
epsilon = 0.15

def random_agent(observation, configuration):
    return random.randrange(configuration.banditCount)

def simple_bandit_agent(observation, configuration):
    global N, Q, A, R_total, epsilon

    banditCount = configuration.banditCount

    if observation.step == 0:
        # Initialize, for a=1 to k:
        N = np.zeros(banditCount)
        Q = np.zeros(banditCount)
    else: # observation.step > 0:
        # R = bandit(A)
        R = observation.reward - R_total
        R_total = observation.reward
        # ...
        N[A] += 1
        Q[A] += (R - Q[A]) / N[A]

    if np.random.binomial(1, epsilon, 1):
        A = np.random.randint(0, banditCount)
    else:
        A = np.argmax(Q)
    return int(A)

if __name__ == '__main__':
    from kaggle_environments import make
    env = make("mab", debug=True)
    env.reset()
    env.run([simple_bandit_agent, random_agent])

