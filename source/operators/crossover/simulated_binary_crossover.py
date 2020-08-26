from model.operation import Operation
import numpy as np
from numpy.random import random
from numpy import minimum, maximum, logical_and
#import random

class SimulatedBinaryCrossover(Operation):
    def __init__(self, eta=15, prob=0.5):
        super().__init__()
        self.eta = eta
        self.prob = prob

    def _do(self, ga):
        (n_inds, n_params) = ga.pop.shape
        (xl, xu) = ga.problem.domain
        
        indices = np.arange(n_inds)

        offs = []
        np.random.shuffle(indices)

        for i in range(0, n_inds, 2):
            idx1, idx2 = indices[i], indices[i+1]
            offs1, offs2 = ga.pop[idx1].copy(), ga.pop[idx2].copy()

            R = random((n_params,))
            diffs = np.abs(offs1 - offs2)
            crossover_points = np.where(logical_and(R <= self.prob, diffs > 1e-14))
            x1 = minimum(offs1[crossover_points], offs2[crossover_points])
            x2 = maximum(offs1[crossover_points], offs2[crossover_points])
            rand = random((crossover_points[0].shape[0],))

            beta = 1 + (2 * (x1-xl)/x2-xl)
            beta_q = self.__calc_beta_q(rand, beta)
            c1 = 0.5 * (x1+x2 - beta_q*(x2-x1))

            beta = 1 + (2 * (xu-x2)/x2-x1)
            beta_q = self.__calc_beta_q(rand, beta)
            c2 = 0.5 * (x1+x2 + beta_q*(x2-x1))

            c1 = minimum(maximum(c1, xl), xu)
            c2 = minimum(maximum(c2, xl), xu)

            r = random(rand.shape)
            c1[r <= 0.5], c2[r <= 0.5] = c2[r <= 0.5], c1[r <= 0.5]
            offs1[crossover_points] = c1
            offs2[crossover_points] = c2

            offs.append(offs1)
            offs.append(offs2)  
        
        return np.reshape(offs, ga.pop.shape)

    def __calc_beta_q(self, rand, beta):
        alpha = 2 - beta**-(self.eta+1)
        beta_q = np.empty(beta.shape)
        mask, mask_not = np.where(rand <= 1/alpha), np.where(rand > 1/alpha)
        
        beta_q[mask] = (rand[mask] * alpha[mask]) ** (1 / (self.eta + 1))
        beta_q[mask_not] = (1 / (2 - rand[mask_not] * alpha[mask_not])) ** (1 / (self.eta + 1))
        return beta_q
