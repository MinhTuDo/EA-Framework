import algorithms.single as so

from terminations import *

from model import LogSaver, Display

from optimize import optimize

from utils import SOGifMaker

import operators.crossover as cx
import operators.initialization as init
import operators.mutation as mut
import operators.selection as sel
import operators.model_builder as mb

import problems.single as sp

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

problem = sp.GoldsteinPrice(n_params=2)
problem._plot(plot_3D=True)


termination = MaxGenTermination(50)
# crossover = cx.UniformCrossover()
# mutation = mut.BitFlipMutation()
algorithm = so.PSO(pop_size=200, 
                  elitist_archive=2,
                  mutation=None)

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

problem._plot(result, plot_3D=False)

gif_saver = SOGifMaker(problem, 
                       directory='assets', 
                       filename='GoldsteinPrice-PSO', 
                       contour_density=20)
gif_saver.make(result, plot_3D=False)