import algorithms.single as so
import algorithms.multi_objective as mo

from terminations import *

from model import LogSaver, Display

from optimize import optimize

import operators.crossover as cx
import operators.initialization as init
import operators.mutation as mut
import operators.selection as sel
import operators.model_builder as mb

import problems.single as sp
import problems.multi as mp

class MySaver(LogSaver):
    def _do(self, algorithm):
        self.add_attributes('n_gens', algorithm.n_gens) 
        self.add_attributes('n_evals', algorithm.n_evals) 
        self.add_attributes('F', algorithm.fitness_pop[algorithm.elite_idx])

class MyDisplay(Display):
    def _do(self, algorithm):
        self.display_top = -1
        self.add_attributes('n_gens', algorithm.n_gens)
        self.add_attributes('n_evals', algorithm.n_evals)
        self.add_attributes('min', algorithm.fitness_pop.min(), width=5)
        self.add_attributes('max', algorithm.fitness_pop.max(), width=5)
        self.add_attributes('mean', algorithm.fitness_pop.mean(), width=5)
        self.add_attributes('F', algorithm.fitness_pop[algorithm.elite_idx])

display = MyDisplay()
log_saver = MySaver()

problem = sp.TrapMax(n_params=20, trap_size=5)
termination = Convergence()
crossover = cx.GOM()
algorithm = so.MBEA(pop_size=50, elitist_archive=2)

result = optimize(problem,
                  algorithm,
                  termination=termination,
                  verbose=True,
                  log=False,
                  save_history=True,
                  seed=2,
                  display=display,
                  log_saver=log_saver)

print(result.model)
print(result.exec_time)


