import numpy as np
from model.operation import Operation

class DECrossover(Operation):
    def __init__(self):
        pass

    def _do(self, ga):
        (n_inds, n_params) = ga.pop.shape
        offs = []
        for i in range(n_inds):
            x_i, v_i = ga.pop[i].copy(), ga.mutant_vectors[i].copy()
            points = np.random.randn(n_params,)
            x_i[points <= ga.cr] = v_i[points <= ga.cr]

            offs.append(x_i)
        return np.reshape(offs, ga.pop.shape)