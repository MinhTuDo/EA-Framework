from model.operation import Operation
import numpy as np

class MB1X(Operation):
    def __init__(self):
        super().__init__()

    def _do(self, ga):
        if not hasattr(ga, 'model'):
            raise Exception('Model not found!')
        (n_inds, n_params) = ga.pop.shape
        indices = np.arange(n_inds)
        n_groups = len(ga.model)


        offs = []
        np.random.shuffle(indices)

        for i in range(0, n_inds, 2):
            idx1, idx2 = indices[i], indices[i+1]
            offs1, offs2 = ga.pop[idx1].copy(), ga.pop[idx2].copy()

            point = np.random.randint(low=0, high=n_groups-1)
            for idx, group in enumerate(ga.model):
                if idx < point:
                    offs1[group], offs2[group] = offs2[group].copy(), offs1[group]

            offs.append(offs1)
            offs.append(offs2)
        
        return np.reshape(offs, ga.pop.shape)
        pass