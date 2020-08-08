from model.operation import Operation
import numpy as np
from numpy import uniform

class MutantVector(Operation):
    def __init__(self):
        super().__init__()
        self.r_v = []
        self.domain = None
    
    def create_mutant_vector(self, ind):
        for _ in range(3):
            self.r_v += uniform(low=self.domain[0],
                                high=self.domain[1],
                                size=ind.shape)
        scaled_diff = np.sqrt(((self.r_v[0] - self.r_v[1])**2).sum())
        return self.r_v[2] + scaled_diff

    def _do(self, ga):
        self.domain = ga.problem.domain
        return np.array(list(map(self.create_mutant_vector, ga.pop)))
                            
        