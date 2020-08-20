from model.operation import Operation
import numpy as np
import random

class SBX(Operation):
    def __init__(self, eta=15, prob=0.9):
        super().__init__()
        self.eta = eta
        self.prob = prob

    def _do(self, ga):
        (n_inds, n_params) = ga.pop.shape
        (xl, xu) = ga.problem.domain
        XL = np.ones((n_params,)) * xl
        XU = np.ones((n_params,)) * xu
        indices = np.arange(n_inds)

        offs = []
        np.random.shuffle(indices)

        for i in range(0, n_inds, 2):
            idx1, idx2 = indices[i], indices[i+1]
            offs1, offs2 = ga.pop[idx1].copy(), ga.pop[idx2].copy()


            for j, xl, xu in zip(range(n_params), XL, XU):
                if random.random() <= 0.5:
                    if abs(offs1[j] - offs2[j]) > 1e-14:
                        x1 = min(offs1[j], offs2[j])
                        x2 = max(offs1[j], offs2[j])
                        rand = random.random()

                        beta = 1 + (2 * (x1 - xl) / (x2 - x1))
                        alpha = 2 - beta ** -(self.eta + 1)
                        if rand <= 1/alpha:
                            beta_q = (rand * alpha) ** (1 / (self.eta + 1))
                        else:
                            beta_q = (1 / (2 - rand * alpha)) ** (1 / (self.eta + 1))

                        c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                        beta = 1 + (2 * (xu - x2) / (x2 - x1))
                        alpha = 2 - beta ** -(self.eta + 1)
                        if rand <= 1/alpha:
                            beta_q = (rand * alpha) ** (1 / (self.eta + 1))
                        else:
                            beta_q = (1 / (2 - rand * alpha)) ** (1 / (self.eta + 1))
                        
                        c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                        c1 = min(max(c1, xl), xu)
                        c2 = min(max(c2, xl), xu)

                        if random.random() <= 0.5:
                            offs1[j] = c2
                            offs2[j] = c1
                        else:
                            offs1[j] = c1
                            offs2[j] = c2

            offs.append(offs1)
            offs.append(offs2)  
        
        return np.reshape(offs, ga.pop.shape)
