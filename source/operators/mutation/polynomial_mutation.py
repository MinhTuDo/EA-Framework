from model.operation import Operation
import numpy as np
from numpy.random import random
from numpy import minimum, maximum
class PolynomialMutation(Operation):
    def __init__(self, eta=20, prob=None):
        super().__init__()
        self.eta = eta
        self.prob = prob

    def _do(self, ga):
        (n_inds, n_params) = ga.offs.shape
        (xl, xu) = ga.problem.domain

        pop = ga.offs.copy()

        if self.prob is None:
            self.prob = 1/n_params

        indices = np.arange(n_inds)

        for idx in indices:
            R = random((n_params,))
            mutation_points = np.where(R <= self.prob)
            x = pop[idx][mutation_points]

            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random((x.shape[0],))
            mut_pow = 1 / (self.eta + 1)
            delta_q = self.__calc_delta_q(rand, mut_pow, delta_1, delta_2)

            x = x + delta_q * (xu - xl)
            x = minimum(maximum(x, xl), xu)
            
            pop[idx][mutation_points] = x
        return pop

    def __calc_delta_q(self, rand, mut_pow, delta_1, delta_2):
        delta_q = np.empty(rand.shape)

        mask = np.where(rand < 0.5)
        xy = 1 - delta_1[mask]
        val = 2 * rand[mask] + (1 - 2 * rand[mask]) * xy ** (self.eta + 1)
        delta_q[mask] = val ** mut_pow - 1

        mask_not = np.where(rand >= 0.5)
        xy = 1 - delta_2[mask_not]
        val = 2 * (1 - rand[mask_not]) + 2 * (rand[mask_not] - 0.5) * xy ** (self.eta + 1)
        delta_q[mask_not] = 1 - val ** mut_pow

        return delta_q