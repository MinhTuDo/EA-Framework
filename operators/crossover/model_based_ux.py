from model.operation import Operation
import numpy as np

class MBUniformCrossover(Operation):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

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

            points = np.random.uniform(low=0, high=1, size=(n_params,))
            for idx, group in enumerate(ga.model):
                if points[idx] < self.prob:
                    offs1[group], offs2[group] = offs2[group].copy(), offs1[group]

            # for group in ga.model:
            #     if np.random.rand() < 0.5:
            #         offs1[group], offs2[group] = offs2[group].copy(), offs1[group]
                
            # swap_groups = ga.model[points < 0.5]
            # offs1[swap_groups], offs2[swap_groups] = offs2[swap_groups], offs1[swap_groups]

            offs.append(offs1)
            offs.append(offs2)
        
        return np.reshape(offs, ga.pop.shape)
        pass