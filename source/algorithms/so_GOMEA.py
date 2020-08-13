from model.GA import GA
from model.GA import GA
from operators.initialization.random_initialization import RandomInitialization
from operators.selection.tournament_selection import TournamentSelection
import numpy as np
from terminations.convergence import Convergence
from operators.crossover.model_based_ux import ModelBasedUniformCrossover
from operators.model_builder.linkage_tree_model import LinkageTreeModel

class GOMEA(GA):
    def __init__(self,
                 pop_size=50,
                 n_offs=None,
                 initialization=RandomInitialization(),
                 selection=None,
                 crossover=ModelBasedUniformCrossover(),
                 elitist_archive=2,
                 **kwargs):
        super().__init__(pop_size, initialization, selection,
                         crossover, mutation, n_offs, **kwargs)
        self.model_builder = LinkageTreeModel()
        self.default_termination = Convergence()
        self.elitist_archive = elitist_archive
        if selection is None:
            self.selection = TournamentSelection(elitist_archive)
        if n_offs is None:
            self.n_offs = self.pop_size