
import ea.algorithms.single as so

from ea.terminations import *

from ea.model import LogSaver, Display

from ea.optimize import optimize

from ea.utils import SOGifMaker

import ea.operators.crossover as cx
import ea.operators.initialization as init
import ea.operators.mutation as mut
import ea.operators.selection as sel
import ea.operators.model_builder as mb

import ea.problems.single as sp

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

problem = sp.Shubert(n_params=2, trap_size=5)
problem._plot(plot_3D=True)


termination = MaxEvalTermination(20000)
# crossover = cx.UniformCrossover()
# mutation = mut.BitFlipMutation()
algorithm = so.PSO(pop_size=50, 
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

# gif_saver = SOGifMaker(problem, 
#                        directory='gif', 
#                        filename='Modified-Rastrigin-PSO', 
#                        contour_density=20)
# gif_saver.make(result, plot_3D=False)