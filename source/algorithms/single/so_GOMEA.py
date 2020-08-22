from algorithms.GA import GA
from operators.initialization import RandomInitialization
from operators.selection import TournamentSelection
import numpy as np
from terminations import Convergence
from operators.crossover import GOM
from operators.model_builder import LinkageTreeModel

class GOMEA(GA):
    def __init__(self,
                 pop_size=50,
                 n_offs=None,
                 initialization=RandomInitialization(),
                 selection=None,
                 crossover=GOM(),
                 elitist_archive=2,
                 **kwargs):
        super().__init__(pop_size, initialization, selection,
                         crossover, n_offs, **kwargs)
        self.model_builder = LinkageTreeModel()
        self.default_termination = Convergence()
        self.elitist_archive = elitist_archive
        if selection is None:
            self.selection = TournamentSelection(elitist_archive)
        if n_offs is None:
            self.n_offs = self.pop_size

    def _initialize(self):
        self.fitness_pop = self.evaluate(self.pop)
        self.sub_tasks_each_gen()

    def _next(self):
        self.model = self.model_builder.build(self)
        
        offs = self.crossover._do(self)
        fitness_offs = self.evaluate(offs)

        self.pop = np.vstack((self.pop, offs))
        self.fitness_pop = np.vstack((self.fitness_pop, fitness_offs))

        selected_indices = self.selection._do(self)

        self.pop = self.pop[selected_indices]
        self.fitness_pop = self.fitness_pop[selected_indices]

    def _save_result_ga(self):
        self.result.model = self.model

    def _save_history(self):
        res = {'P': self.pop.copy(), 
               'F': self.fitness_pop.copy()}
        self.history.append(res)