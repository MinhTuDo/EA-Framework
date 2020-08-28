import algorithms.multi_objective as mo

from terminations import *

from model import LogSaver, Display

from optimize import optimize

from utils import MOGifMaker

import operators.crossover as cx
import operators.initialization as init
import operators.mutation as mut
import operators.selection as sel
import operators.model_builder as mb

import problems.multi as mp


class MySaver(LogSaver):
    def _do(self, ga):
        self.add_attributes('n_gens', ga.n_gens) 
        self.add_attributes('n_evals', ga.n_evals) 
        self.add_attributes('F', ga.F_pop[self.elite_idx])

class MyDisplay(Display):
    def _do(self, ga):
        self.display_top = -1
        self.add_attributes('n_gens', ga.n_gens)
        self.add_attributes('n_evals', ga.n_evals)
        self.add_attributes('F', ga.F_pop[ga.elite_idx])
        

display = MyDisplay()
log_saver = MySaver()
problem = mp.ZDT4(n_params=10)
problem._plot()

termination = MaxEvalTermination(20000)

algorithm = mo.NSGAII(pop_size=100, elitist_archive=4)

result = optimize(problem, 
                  algorithm, 
                  termination=termination, 
                  verbose=True,
                  log=False, 
                  save_history=True, 
                  seed=1, 
                  display=display,
                  log_saver=log_saver)
print(result.model)
print(result.exec_time)

problem._plot(result)

gif_saver = MOGifMaker(problem, 
                       directory='gif', 
                       filename='ZDT4-NSGA-II')
gif_saver.make(result)
