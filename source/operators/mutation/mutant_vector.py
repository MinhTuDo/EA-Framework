from model.operation import Operation
import numpy as np
from numpy.random import uniform

class MutantVector(Operation):
    def __init__(self):
        super().__init__()
        self.domain = None
        self.f = None
        self.pop_idx = None
        self.pop = None
    
    def create_mutant_vector(self, idx):
        x_r = np.random.choice(self.pop_idx[self.pop_idx != idx], 3)
        return self.pop[x_r[0]] + self.f * (self.pop[x_r[1]] - self.pop[x_r[2]])


    def _do(self, ga):
        self.f = ga.f
        self.pop = ga.pop
        self.pop_idx = np.arange(len(ga.pop))
        mutant_vectors = np.array(list(map(self.create_mutant_vector, self.pop_idx)))
        return mutant_vectors
                            
        