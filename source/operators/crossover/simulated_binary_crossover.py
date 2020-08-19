from model.operation import Operation
import numpy as np

class SBX(Operation):
    def __init__(self, eta=15):
        super().__init__()
        self.eta = eta

    def _do(self, ga):
        (n_inds, n_params) = ga.pop.shape
        (xl, xu) = ga.problem.domain
        indices = np.arange(n_inds)

        offs = []
        np.random.shuffle(indices)

        for i in range(0, n_inds, 2):
            idx1, idx2 = indices[i], indices[i+1]
            offs1, offs2 = ga.pop[idx1].copy(), ga.pop[idx2].copy()

            points = np.random.random((n_params,))
            beta = np.ones((points.shape))
            beta[points <= 0.5] *= 2 * points[points <= 0.5]
            beta[points > 0.5] *= 1 / (2 * (1 - points[points > 0.5]))

            beta **= 1 / (self.eta+1)

            offs1 = 0.5 * (((1+beta) * offs1) + ((1-beta) * offs2))
            offs2 = 0.5 * (((1-beta) * offs1) + ((1+beta) * offs2))

            offs1[offs1 < xl] = xl
            offs1[offs1 > xu] = xu
            offs2[offs2 < xl] = xl
            offs2[offs2 > xu] = xu
            # offs1[offs1 < xl] = np.random.rand()
            # offs1[offs1 > xu] = np.random.rand()
            # offs2[offs2 < xl] = np.random.rand()
            # offs2[offs2 > xu] = np.random.rand()
            offs.append(offs1)
            offs.append(offs2)  
        
        return np.reshape(offs, ga.pop.shape)
