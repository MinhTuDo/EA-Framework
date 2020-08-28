from numpy.random import uniform
from model.operation import Operation
import numpy as np

class RandomInitialization(Operation):
    def __init__(self):
        super().__init__()
        pass
    def _do(self, ga):
        n = ga.pop_size
        m = ga.problem.n_params
        shape = (n, m)
        domain = ga.problem.domain
        pop = uniform(low=domain[0], high=domain[1], size=shape)
        if ga.problem.param_type == np.int:
            pop = pop.round().astype(np.int)
        return pop