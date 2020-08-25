from model.operation import Operation
import numpy as np
from numpy.random import random
from numpy import minimum, maximum, logical_and
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
        np.random.shuffle(indices)

        for idx in indices:
            for i in range(n_params):
                if random() <= self.prob:
                    x = pop[idx][i]
                    delta_1 = (x - xl) / (xu - xl)
                    delta_2 = (xu - x) / (xu - xl)
                    rand = random()
                    mut_pow = 1 / (self.eta + 1)

                    if rand < 0.5:
                        xy = 1 - delta_1
                        val = 2 * rand + (1 - 2 * rand) * xy ** (self.eta + 1)
                        delta_q = val ** mut_pow - 1
                    else:
                        xy = 1 - delta_2
                        val = 2 * (1 - rand) + 2 * (rand - 0.5) * xy ** (self.eta + 1)
                        delta_q = 1 - val ** mut_pow

                    x = x + delta_q * (xu - xl)
                    x = min(max(x, xl), xu)
                    pop[idx][i] = x
        return pop