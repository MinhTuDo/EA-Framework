import numpy as np
from model.operation import Operation
class UX(Operation):
    def __init__(self):
        pass

    def _do(self, ga):
        (n_inds, n_params) = ga.pop.shape
        indices = np.arange(n_inds)

        offs = []
        np.random.shuffle(indices)

        for i in range(0, n_inds, 2):
            idx1, idx2 = indices[i], indices[i+1]
            offs1, offs2 = ga.pop[idx1].copy(), ga.pop[idx2].copy()

            points = np.random.randn(n_params,)
            offs1[points < 0.5], offs2[points < 0.5] = offs2[points < 0.5], offs1[points < 0.5]

            offs.append(offs1)
            offs.append(offs2)
        
        return np.reshape(offs, ga.pop.shape)