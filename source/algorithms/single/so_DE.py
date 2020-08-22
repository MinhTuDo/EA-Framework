from algorithms.GA import GA
from terminations import MaxTimeTermination
from operators.mutation import MutantVector
from operators.crossover import DECrossover
from operators.initialization import RandomInitialization
from operators.selection import HistoricalBestSelection
import numpy as np
from numpy.random import uniform, rand


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
        self.offs = None
        self.fitness_offs = None


    def _next(self):
        self.mutant_vectors = self.mutation._do(self)

        self.offs = self.crossover._do(self)

        self.fitness_offs = self.evaluate(self.offs)
        self.fitness_pop = self.evaluate(self.pop)
        selected_indices = self.selection._do(self)
        self.pop[selected_indices] = self.offs[selected_indices]

    def _save_history(self):
        res = {'P': self.pop.copy(), 
               'F': self.fitness_pop.copy()}
        self.history.append(res)

        
