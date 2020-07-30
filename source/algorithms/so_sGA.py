from model.GA import GA
from operators.initialization.random_initialization import RandomInitialization
from operators.crossover.uniform_crossover import UniformCrossover
from operators.selection.tournament_selection import TournamentSelection
import numpy as np
from utils.termination.convergence import Convergence


class SGA(GA):
    def __init__(self,
                 pop_size,
                 n_offs=None,
                 initialization=RandomInitialization(),
                 selection=None,
                 crossover=UniformCrossover(),
                 elitist_archive=2,
                 mutation=None,
                 **kwargs):
        super().__init__(pop_size, initialization, selection,
                         crossover, mutation, n_offs, **kwargs)
        self.default_termination = Convergence()
        self.elitist_archive = elitist_archive
        if selection is None:
            self.selection = TournamentSelection(elitist_archive)
        if n_offs is None:
            self.n_offs = self.pop_size

    def _next(self):
        offs = self.crossover._do(self)
        f_offs = self.evaluate(offs)

        self.pop = np.vstack((self.pop, offs))
        self.f_pop = np.hstack((self.f_pop, f_offs))

        selected_indices = self.selection._do(self)

        self.pop = self.pop[selected_indices]
        self.f_pop = self.f_pop[selected_indices]