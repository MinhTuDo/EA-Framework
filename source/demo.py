from model.display import Display
from factory import GAFactory
from optimize import optimize
from utils.gif_saver import GifSaver
from model.log_saver import LogSaver

import numpy as np

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
factory = GAFactory()
problem = factory.get_problem('TrapMax')(n_params=20, trap_size=5)
# problem.plot(plot_3D=True, contour_density=20, colorbar=True)

termination = factory.get_termination('Convergence')()

crossover = factory.get_crossover('GOM')()

algorithm = factory.get_algorithm('MBEA')(pop_size=50, elitist_archive=2)

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

# gif_saver = GifSaver(problem, 'gif', 'Rastrigin-DE', contour_density=20)
# gif_saver.make(result, display_optimum=True, loop=False)
