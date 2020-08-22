from model.operation import Operation
import numpy as np

class BitFlipMutation(Operation):
    def __init__(self):
        super().__init__()

    def _do(self, ga):
        R = np.random.random(ga.pop.shape)
        pop = ga.pop.copy()
        pop[R < ga.mutation_prob] = (not pop[R < ga.mutation_prob].astype(np.bool))\
                                                                  .astype(ga.problem.param_type)
        return pop

    