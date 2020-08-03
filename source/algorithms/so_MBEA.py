from model.GA import GA
from operators.initialization.random_initialization import RandomInitialization
from operators.selection.tournament_selection import TournamentSelection
import numpy as np
from terminations.convergence import Convergence
from operators.crossover.model_based_ux import ModelBasedUniformCrossover
from operators.model_builder.marginal_product_model import MarginalProductModel
class MBEA(GA):
    def __init__(self,
                 pop_size,
                 n_offs=None,
                 initialization=RandomInitialization(),
                 model_builder=MarginalProductModel(),
                 selection=None,
                 crossover=ModelBasedUniformCrossover(),
                 elitist_archive=2,
                 mutation=None,
                 **kwargs):
        super().__init__(pop_size, initialization, selection,
                         crossover, mutation, n_offs, **kwargs)
        self.default_termination = Convergence()
        self.model_builder = model_builder
        self.elitist_archive = elitist_archive
        if selection is None:
            self.selection = TournamentSelection(elitist_archive)
        if n_offs is None:
            self.n_offs = self.pop_size


    def _initialize(self):
        self.model = [[group] for group in np.arange(self.problem.n_params)]
        self.f_pop = self.evaluate(self.pop)
        self.sub_tasks_each_gen()
    
    def _next(self):
        self.model = self.model_builder.build(self)
        
        offs = self.crossover._do(self)
        f_offs = self.evaluate(offs)

        self.pop = np.vstack((self.pop, offs))
        self.f_pop = np.hstack((self.f_pop, f_offs))

        selected_indices = self.selection._do(self)

        self.pop = self.pop[selected_indices]
        self.f_pop = self.f_pop[selected_indices]

    def _sub_tasks_each_gen(self):
        pass

    def _save_result_ga(self):
        self.result.model = self.model