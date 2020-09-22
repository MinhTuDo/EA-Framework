from numpy.random import random
from model.operation import Operation
import numpy as np
from numpy import maximum, minimum
from utils import denormalize

class RandomInitialization(Operation):
    def __init__(self):
        super().__init__()
        pass
    def _do(self, ga):
        n = ga.pop_size
        m = ga.problem.n_params
        shape = (n, m)
        (XL, XU) = ga.problem.domain
        pop = random(shape)
        pop = denormalize(pop, XL, XU)
        if ga.problem.param_type == np.int:
            pop = pop.round().astype(np.int)
        
        return pop