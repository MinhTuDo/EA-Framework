from algorithms.GA import GA
from operators.initialization.random_initialization import RandomInitialization
from operators.selection.tournament_selection import TournamentSelection
import numpy as np
from terminations.convergence import Convergence
from operators.crossover.gene_pool_optimal_mixing import GOM
from operators.model_builder.linkage_tree_model import LinkageTreeModel

class GOMEA(GA):
    def __init__(self,
                 pop_size=50,
                 n_offs=None,
                 initialization=RandomInitialization(),
                 selection=None,
                 elitist_archive=2,
                 mutation=None,
                 **kwargs):
        super().__init__(pop_size=pop_size, 
                         initialization=initialization, 
                         selection=selection,
                         n_offs=n_offs, 
                         **kwargs)
        self.crossover = GOM()
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
        
        self.offs = self.crossover._do(self)
        self.offs = self.mutation._do(self) if self.mutation is not None else self.offs
        self.fitness_offs = self.evaluate(self.offs)

        self.pop = np.vstack((self.pop, self.offs))
        self.fitness_pop = np.vstack((self.fitness_pop, self.fitness_offs))

        selected_indices = self.selection._do(self)

        self.pop = self.pop[selected_indices]
        self.fitness_pop = self.fitness_pop[selected_indices]

    def _save_result_ga(self):
        self.result.model = self.model

    def _save_history(self):
        res = {'P': self.pop.copy(), 
               'F': self.fitness_pop.copy()}
        self.history.append(res)