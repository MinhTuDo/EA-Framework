from model.display import Display
from factory import GAFactory
from optimize import optimize
from utils.gif_saver import GifSaver
from model.log_saver import LogSaver

import numpy as np

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
        # self.add_attributes('min', ga.F_pop.min(), width=5)
        # self.add_attributes('max', ga.F_pop.max(), width=5)
        # self.add_attributes('mean', ga.F_pop.mean(), width=5)
        self.add_attributes('F', ga.F_pop[ga.elite_idx])
        self.add_attributes('rank', ga.rank[ga.elite_idx])
        

display = MyDisplay()
log_saver = MySaver()
factory = GAFactory()
problem = factory.get_problem('SchafferN1')(A=100)
# problem.plot(plot_3D=True, contour_density=20, colorbar=True)

termination = factory.get_termination('MaxGenTermination')(50)

crossover = factory.get_crossover('GOM')()

algorithm = factory.get_algorithm('NSGAII')(pop_size=20, elitist_archive=2)

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
