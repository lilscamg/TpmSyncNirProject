import numpy as np
from enum import Enum


class UpdateRules(Enum):
    Hebbian = 1,
    AntiHebbian = 2,
    RandomWalk = 3


def theta(t1, t2):
    return 1 if t1 == t2 else 0


def hebbian(W, X, sigma, tau1, tau2, L):
    for (i, j), _ in np.ndenumerate(W):
        W[i, j] += X[i, j] * tau1 * theta(sigma[i], tau1) * theta(tau1, tau2)
        W[i, j] = np.clip(W[i, j], -L, L)


def anti_hebbian(W, X, sigma, tau1, tau2, L):
    for (i, j), _ in np.ndenumerate(W):
        W[i, j] -= X[i, j] * tau1 * theta(sigma[i], tau1) * theta(tau1, tau2)
        W[i, j] = np.clip(W[i, j], -L, L)


def random_walk(W, X, sigma, tau1, tau2, L):
    for (i, j), _ in np.ndenumerate(W):
        W[i, j] += X[i, j] * theta(sigma[i], tau1) * theta(tau1, tau2)
        W[i, j] = np.clip(W[i, j], -L, L)
