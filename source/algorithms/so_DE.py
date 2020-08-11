from model.GA import GA
from terminations.max_time import MaxTimeTermination
from operators.mutation.mutant_vector import MutantVector
from operators.crossover.de_crossover import DECrossover
from operators.initialization.random_initialization import RandomInitialization
from operators.selection.historical_best_selection import HistoricalBestSelection
import numpy as np
from numpy.random import uniform
from numpy.random import rand

class DE(GA):
    def __init__(self,
                 pop_size=50,
                 initialization=RandomInitialization(),
                 cr=0.9,
                 f=0.8,
                 **kwargs):

        super().__init__(pop_size, initialization, **kwargs)
        self.selection = HistoricalBestSelection()
        self.mutation = MutantVector()
        self.crossover = DECrossover()
        self.f = f
        self.cr = cr
        self.default_termination = MaxTimeTermination(5)
        self.mutant_vectors = None
        self.pop_prev = None
        self.f_pop_prev = None


    def _next(self):
        self.mutant_vectors = self.mutation._do(self)

        self.pop_prev = self.crossover._do(self)

        self.f_pop_prev = self.evaluate(self.pop_prev)
        self.f_pop = self.evaluate(self.pop)
        prev_best_idx = self.selection._do(self)
        self.pop[prev_best_idx] = self.pop_prev[prev_best_idx]

        
