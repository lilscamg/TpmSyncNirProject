import numpy as np
from UpdateRules.update_rules import hebbian, anti_hebbian, random_walk, UpdateRules


# Древовидная машина четности. Generates a binary digit(tau) for a given random vector(X).
# Параметры:
# K - число скрытых нейронов
# N - число входных нейронов у каждого скрытого нейрона
# L - порог весовых коэффициентов
# W - The weight matrix between input and hidden layers. Dimensions : [K, N]
from Utils.utils import sgn


class TreeParityMachine:

    # Конструктор
    # Параметры:
    # K - число скрытых нейронов
    # N - число входных нейронов у каждого скрытого нейрона
    # L - порог весовых коэффициентов
    def __init__(self, K, N, L):
        self.K = K
        self.N = N
        self.L = L
        self.W = np.random.randint(-L, L + 1, size=(K, N))
        self.sigma = None
        self.tau = None
        self.X = None

    # Расчет выхода ДМЧ - tau
    # Параметры:
    # X - Входной вектор
    def calc_tau(self, X):
        # self.sigma = np.sign(np.sum(X * self.W, axis=1))
        self.sigma = sgn(np.sum(X * self.W, axis=1))
        self.tau = np.prod(self.sigma)
        self.X = X
        return self.sigma, self.tau

    # Обновление весов по заданному правилу
    # Параметры
    # tau2 - выходной бит из другой ДМЧ;
    # update_rule - правило
    def update(self, tau2, update_rule):

        X = self.X
        tau1 = self.tau
        sigma = self.sigma
        W = self.W
        L = self.L

        if tau1 == tau2:
            if update_rule == UpdateRules.Hebbian:
                hebbian(W, X, sigma, tau1, tau2, L)
            elif update_rule == UpdateRules.AntiHebbian:
                anti_hebbian(W, X, sigma, tau1, tau2, L)
            elif update_rule == UpdateRules.RandomWalk:
                random_walk(W, X, sigma, tau1, tau2, L)
