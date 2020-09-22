from model.operation import Operation
import numpy as np

class BitFlipMutation(Operation):
    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob

    def _do(self, ga):
        self.prob = 1/ga.problem.n_params if self.prob is None else self.prob
        R = np.random.random(ga.offs.shape)
        pop = ga.offs.copy()
        mutation_points = pop[R < self.prob].astype(np.bool)
        pop[R < self.prob] = (mutation_points == False).astype(ga.problem.param_type)
        return pop

    