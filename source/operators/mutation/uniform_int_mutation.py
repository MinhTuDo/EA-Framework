import numpy as np
from model.operation import Operation

class UniformIntMutation(Operation):
    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob

    def _do(self, ga):
        (xl, xu) = ga.problem.domain
        self.prob = 1/ga.problem.n_params if self.prob is None else self.prob
        R = np.random.random(ga.offs.shape)
        pop = ga.offs.copy()
        n_mut = len(R[R < self.prob])
        mutation_vector = np.random.randint(low=xl, high=xu, size=(n_mut,))
        pop[R < self.prob] = mutation_vector
        return pop

        