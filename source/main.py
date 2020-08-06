from model.display import Display
from factory import GAFactory
from optimize import optimize
from utils.gif_saver import GifSaver

import numpy as np

class MyDisplay(Display):
    def _do(self, algorithm):
        self.display_top = 10
        self.add_attributes('n_gens', algorithm.n_gens)
        self.add_attributes('n_evals', algorithm.n_evals)
        self.add_attributes('min', algorithm.f_pop.std(), width=5)
        self.add_attributes('mean', algorithm.f_pop.mean(), width=5)
        self.add_attributes('F', algorithm.f_opt)
        

display = MyDisplay()
factory = GAFactory()
problem = factory.get_problem('Rastrigin')()
# problem.plot(plot_3D=False, contour_density=20, colorbar=True)

termination = factory.get_termination('MaxGenTermination')(50)

crossover = factory.get_crossover('ModelBasedUniformCrossover')()

algorithm = factory.get_algorithm('PSO')(pop_size=500, 
                                         termination=termination, 
                                         crossover=crossover,
                                         topology='ring')

result = optimize(problem, 
                  algorithm, 
                  termination=termination, 
                  verbose=True, 
                  save_history=True, 
                  seed=18521578, 
                  display=display)

print(result.success)

# gif_saver = GifSaver(problem, 'gif', 'Rastrigin-PSO-Star', contour_density=20)
# gif_saver.make(result, display_optimum=True, loop=False)
