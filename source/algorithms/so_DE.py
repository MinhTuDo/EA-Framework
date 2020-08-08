from model.GA import GA
from terminations.max_time import MaxTimeTermination
from operators.mutation.mutant_vector import MutantVector
from operators.initialization.random_initialization import RandomInitialization
from operators.selection.historical_best_selection import HistoricalBestSelection
import numpy as np
from numpy.random import uniform
from numpy.random import rand

class DE(GA):
    def __init__(self,
                 pop_size=50,
                 initialization=RandomInitialization(),
                 selection=HistoricalBestSelection(),
                 mutation=MutantVector(),
                 cr=0.5,
                 
                 **kwargs):
        super().__init__(pop_size, initialization, selection, **kwargs)
        self.default_termination = MaxTimeTermination(5)
        self.mutant_vectors = None

    def _initialize(self):
        self.f_pop = self.evaluate(self.pop)

    def _next(self):
        self.mutant_vectors = self.mutation._do(self)
        pass
        
